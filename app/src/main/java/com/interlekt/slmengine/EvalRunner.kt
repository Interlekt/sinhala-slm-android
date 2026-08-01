package com.interlekt.slmengine

import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

private const val TAG = "SLMEval"

/**
 * Batch evaluation runner for the Module 3 quantisation study.
 *
 * Reads a question set, runs one experimental condition end to end, and writes
 * a JSONL record per question containing the raw output plus every systems
 * metric. Quality metrics (Token F1, ROUGE-L, EM, HCM) are computed afterwards
 * on the desktop, where the NLI model for HCM lives — the device only needs to
 * produce the generations and the timings.
 *
 * Input   /sdcard/Download/eval/questions.jsonl
 *         {"id": "q001", "question": "...", "reference": "...", "in_syllabus": true}
 *
 * Output  /sdcard/Download/eval/results_<runTag>.jsonl
 *
 * Pull with:
 *   adb pull /sdcard/Download/eval/results_<runTag>.jsonl
 */
class EvalRunner(
    private val wrapper: LlamaWrapper,
    private val rag: RagEngine,
) {

    companion object {
        /**
         * UPDATE THIS whenever the vendored llama.cpp submodule moves.
         *
         *   cd app/src/main/cpp/llama.cpp && git log -1 --format=%h
         *
         * Cells collected on different engine versions are not comparable: a
         * difference between them could be the engine rather than the
         * quantisation level, and nothing in the data would distinguish the
         * two. Recording it here makes a mid-study version change visible
         * instead of silent.
         */
        const val LLAMA_CPP_COMMIT = "053e01d"
        const val GGML_VERSION = "0.12.0"

        /**
         * Bumped whenever the prompt contract changes in a way that makes
         * older results files incomparable.
         *
         * v1  Llama-3 chat template, app-invented abstention string, no
         *     repetition penalty, no first-line truncation, no grounding.
         * v2  qa-evaluation-updated0729.ipynb format: plain instruction text,
         *     the checkpoint's own NO_ANSWER, repetition_penalty=1.05,
         *     first-line answer extraction, post-generation evidence
         *     grounding at 0.50.
         *
         * A v1 file and a v2 file cannot be placed in the same results table.
         */
        const val PROMPT_FORMAT_VERSION = "v2-notebook-0729"

        /**
         * Identifies which boundary detector produced boundary_* / sig_* in
         * this results file. There is currently exactly one — the logistic
         * regression in knowledge_boundary.py / RagConfig's kb* fields — so
         * this is a fixed provenance string, not a switch. Bump it if a
         * differently-fit model ever replaces this one, so old and new result
         * files aren't silently compared as if the gate were the same.
         */
        const val BOUNDARY_METHOD = "logistic_regression"
    }

    data class Condition(
        /** Identifies the quantisation of the loaded SLM, e.g. "Q4_K_M". */
        val quant: String,
        val ragEnabled: Boolean,
        val boundaryGateEnabled: Boolean,
        val compressionEnabled: Boolean,
        val tokenBudget: Int = 384,
        /**
         * MAX_NEW_TOKENS from the notebook, annotated there as audited to
         * cover 100% of gold answers under this tokenizer.
         *
         * VERIFY THIS FOR YOUR SPLIT before a measurement run — the audit was
         * performed on the notebook's own test set. Cell 8 does it:
         * tokenize every gold answer, check max <= MAX_NEW_TOKENS. A gold
         * answer that does not fit is silently truncated and scores as a
         * partial match, which looks like model error.
         */
        val maxTokens: Int = PromptBuilder.MAX_NEW_TOKENS,
        /**
         * Log the device's query embedding. Needed only for the port parity
         * check (simulate_device.py --use-device-qvec), which has already
         * passed at 50/50, and it is ~8 KB per question as JSON — several
         * times everything else in the record combined. Leave off for
         * measurement runs.
         */
        val logQueryVec: Boolean = false,
    ) {
        /** Stable filename/label for this cell of the design. */
        val tag: String get() = buildString {
            append(quant)
            append(if (ragEnabled) "_rag" else "_norag")
            if (ragEnabled) {
                append(if (boundaryGateEnabled) "_gate" else "_nogate")
                if (compressionEnabled) append("_compress")
            }
        }
    }

    data class Progress(val done: Int, val total: Int, val currentId: String)

    /**
     * wrapper.generate() is itself a suspend function that returns only when
     * nativeGenerate's loop finishes, so it needs no suspendCoroutine wrapper
     * and no onDone() callback — returning IS completion.
     *
     * Peak RSS is sampled per token rather than only before/after, since the
     * maximum is what a memory ceiling actually cares about. Reads through
     * HardwareMonitor.pssMb() — a single Debug.getMemoryInfo() call, not the
     * full snapshot() — because this runs once per token and a full snapshot
     * (thermal zones, cpufreq nodes, /proc/self/stat) would add measurable
     * overhead to the thing being measured.
     */
    private suspend fun generateBlocking(
        prompt: String,
        maxTokens: Int,
        hw: HardwareMonitor.Session? = null,
    ): GenResult {
        val t0 = System.currentTimeMillis()
        var firstAt = 0L
        var count = 0
        var promptTokens = -1
        var peakRss = HardwareMonitor.pssMb()
        val sb = StringBuilder()

        wrapper.generate(prompt, maxTokens, object : LlamaWrapper.GenerationCallback {
            override fun onToken(piece: String) {
                val now = System.currentTimeMillis()
                if (count == 0) firstAt = now
                count++
                sb.append(piece)
                val rss = HardwareMonitor.pssMb()
                if (rss > peakRss) peakRss = rss
                // Every 8th token. At 272 ms/token that is ~2 s apart —
                // frequent enough to catch a throttling step, rare enough not
                // to perturb what it measures.
                if (hw != null && count % 8 == 0) hw.sample()
            }

            override fun onMetrics(msPerToken: Float, ramMB: Long, cpuPct: Float) {}

            override fun onPrompt(nTokens: Int) { promptTokens = nTokens }
        })

        val end = System.currentTimeMillis()
        return GenResult(
            text = sb.toString(),
            promptTokens = promptTokens,
            tokensGenerated = count,
            ttftMs = if (count > 0) firstAt - t0 else -1,
            // Measured from the first token, so prefill is excluded. A single
            // blended ms/token would be dominated by prefill on RAG prompts.
            decodeMsPerToken = if (count > 1)
                (end - firstAt).toDouble() / (count - 1) else -1.0,
            totalMs = end - t0,
            peakRssMb = peakRss,
        )
    }

    data class GenResult(
        val text: String,
        val promptTokens: Int,
        val tokensGenerated: Int,
        val ttftMs: Long,
        val decodeMsPerToken: Double,
        val totalMs: Long,
        val peakRssMb: Long,
    )

    /**
     * @param cooldownMs pause between questions to let the SoC settle. 20 s was
     *        measured sufficient on the target device: decode drifted only
     *        +3.8% (269.6 -> 279.8 ms/token) from first third to last third of
     *        a 50-question run — verify per device via the decode-drift check
     *        in analyze_results.py before trusting a shorter cooldown.
     */
    suspend fun run(
        condition: Condition,
        evalDir: String = "/sdcard/Download/eval",
        cooldownMs: Long = 20_000,
        onProgress: (Progress) -> Unit = {},
    ): String? = withContext(Dispatchers.IO) {

        val dir = File(evalDir)
        val qFile = File(dir, "questions.jsonl")
        if (!qFile.exists()) {
            Log.e(TAG, "questions.jsonl not found at $evalDir")
            return@withContext null
        }

        // Fail at launch, not silently across 45 minutes, if the prompt
        // layout has drifted from the trained format. Ports the assertions in
        // cell 9 of the evaluation notebook, which exist because the format
        // drifted once before.
        try {
            PromptBuilder.assertFormat()
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "PROMPT FORMAT CHECK FAILED — refusing to run", e)
            return@withContext null
        }

        val questions = qFile.readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
        val out = File(dir, "results_${condition.tag}.jsonl")
        out.writeText("")

        // Probe BEFORE writing the header, which consumes hw0. Logging the
        // capabilities here means an unreadable sensor shows up immediately
        // rather than as a column of -1 after a 45-minute run.
        HardwareMonitor.logCapabilities()
        val hw0 = HardwareMonitor.snapshot()

        // One header line so a results file is self-describing even if the
        // filename is renamed or the run parameters are forgotten.
        out.appendText(JSONObject(mapOf(
            "record_type" to "run_header",
            "run_tag" to condition.tag,
            "quant" to condition.quant,
            "rag_enabled" to condition.ragEnabled,
            "boundary_gate" to condition.boundaryGateEnabled,
            "compression" to condition.compressionEnabled,
            "boundary_method" to BOUNDARY_METHOD,
            "boundary_features" to if (rag.loaded) rag.config.kbFeatures else null,
            "boundary_probability_threshold" to if (rag.loaded) rag.config.kbProbabilityThreshold else null,
            "token_budget" to condition.tokenBudget,
            "max_tokens" to condition.maxTokens,
            "n_questions" to questions.size,
            "device" to "${Build.MANUFACTURER} ${Build.MODEL}",
            "android" to Build.VERSION.RELEASE,
            "n_chunks" to if (rag.loaded) rag.config.nChunks else -1,
            "log_query_vec" to condition.logQueryVec,

            // ── Generation contract ────────────────────────────────────────
            // Recorded because these changed, and a results file must state
            // which contract produced it. A v1 file (chat template, no
            // penalty, no truncation) is not comparable with a v2 file.
            "prompt_format" to PROMPT_FORMAT_VERSION,
            "sampler" to "greedy+repetition_penalty",
            "repetition_penalty" to PromptBuilder.REPETITION_PENALTY.toDouble(),
            "grounding_threshold" to PromptBuilder.GROUNDING_THRESHOLD,
            // The exact abstention string, so an analysis script can match
            // against it without hardcoding a copy that may drift.
            "no_answer_string" to PromptBuilder.NO_ANSWER,

            // Engine provenance. system_info carries the CPU feature flags, so
            // the results file itself evidences whether the DOTPROD/FP16
            // kernels were compiled in — no need to trust a build log.
            "llama_cpp_commit" to LLAMA_CPP_COMMIT,
            "ggml_version" to GGML_VERSION,
            "system_info" to wrapper.systemInfo().trim(),
            "started_at" to System.currentTimeMillis(),

            // Battery temperature is NOT recorded: /sys/class/power_supply/
            // battery/temp is unreadable on the target device (confirmed by
            // logCapabilities()). thermal_c_* below, from cpu/soc thermal
            // zones, is the substitute, cross-checked against the decode-drift
            // trend at analysis time.
            "thermal_c_start" to hw0.thermalC,
            "cpu_freq_mhz_start" to hw0.cpuFreqMhz,
            "avail_mem_mb_start" to hw0.availMemMb,
        )).toString() + "\n")

        Log.i(TAG, "run ${condition.tag}: ${questions.size} questions -> ${out.name}")

        for ((i, q) in questions.withIndex()) {
            val id = q.optString("id", "q$i")
            val question = q.optString("question")
            onProgress(Progress(i, questions.size, id))

            val rec = HashMap<String, Any?>()
            rec["record_type"] = "result"
            rec["id"] = id
            rec["question"] = question
            // Accept both field-name conventions seen in question sets so far:
            // {"reference": ..., "in_syllabus": ...} and {"answer": ...,
            // "answerable": ...}. optString/.has() default silently on a
            // missing key rather than throwing, so without this fallback a
            // file using the other convention would score every row against
            // an empty reference and a null gold label with no error at all.
            rec["reference"] = when {
                q.has("reference") -> q.optString("reference", "")
                q.has("answer") -> q.optString("answer", "")
                else -> ""
            }
            rec["in_syllabus_gold"] = when {
                q.has("in_syllabus") -> q.optBoolean("in_syllabus")
                q.has("answerable") -> q.optBoolean("answerable")
                else -> null
            }
            rec["run_tag"] = condition.tag

            // Spans the WHOLE question — embedding, retrieval, boundary and
            // generation — so cpu_ms covers an abstained query too. That is the
            // point: an abstention's cost is exactly what makes the gate worth
            // having, and it cannot be shown without measuring it. Also
            // supplies pss_mb_start/end.
            val hw = HardwareMonitor.begin()

            try {
                if (!condition.ragEnabled) {
                    val g = generateBlocking(question, condition.maxTokens, hw)
                    rec["abstained"] = false
                    rec["output_raw"] = g.text
                    // First-line extraction still applies with no RAG — it is
                    // the reference's stopping mechanism, not a RAG feature.
                    // Grounding does NOT apply: there is no context to ground
                    // against, which is precisely what this cell measures.
                    rec["output"] = PromptBuilder.extractAnswer(g.text)
                    rec["evidence_support"] = -1.0
                    rec["grounding_overridden"] = false
                    putGen(rec, g)
                } else {
                    val r = rag.retrieve(question)
                    if (r == null) {
                        rec["error"] = "retrieval_failed"
                    } else {
                        rec["embed_ms"] = r.embedMs
                        rec["retrieve_ms"] = r.retrievalMs
                        rec["retrieved_ids"] = r.results.map { it.chunk.id }
                        rec["retrieved_scores"] = r.results.map { it.score }
                        rec["retrieved_pages"] = r.results.map { it.chunk.page }
                        rec["retrieved_cosines"] = r.results.map { it.cosine }

                        // The retrieved evidence AS TEXT. Without this the file
                        // carries no premise, and an NLI-based hallucination
                        // metric has nothing to entail against — ids cannot
                        // supply it. Also what a human annotator reads when
                        // judging groundedness.
                        rec["retrieved_texts"] = r.results.map { it.chunk.text }

                        if (condition.logQueryVec) {
                            rec["query_vec"] = r.queryVec?.map { it.toDouble() }
                        }

                        rec["boundary_label"] = r.boundary.label
                        rec["boundary_reason"] = r.boundary.reason
                        rec["out_of_syllabus_pred"] = r.boundary.isOutOfSyllabus
                        // P(within syllabus) from the logistic model,
                        // round(...,4) as in Python.
                        rec["boundary_probability"] = r.boundary.probability

                        // ── The five KB_MODEL signals, from the two SEPARATE
                        // dense-only / bm25-only top-k passes — not the fused
                        // RRF list used for the LLM context. Names match
                        // KB_MODEL["features"] 1:1 so nothing needs
                        // translating at analysis time.
                        rec["sig_dense_top"] = r.signals.denseTop
                        rec["sig_dense_avg"] = r.signals.denseAvg
                        // Token overlap vs the DENSE-ONLY context, not the
                        // fused hybrid context used for generation.
                        rec["sig_overlap"] = r.signals.overlap
                        rec["sig_bm25_top"] = r.signals.bm25Top
                        rec["sig_bm25_avg"] = r.signals.bm25Avg
                        rec["sig_probability"] = r.signals.probability

                        val abstain = condition.boundaryGateEnabled && r.boundary.isOutOfSyllabus
                        rec["abstained"] = abstain

                        if (abstain) {
                            // The SLM is never invoked. This is the mitigation,
                            // and it is also why abstained queries are ~50x
                            // cheaper than answered ones.
                            rec["output"] = PromptBuilder.abstentionAnswer()
                            rec["output_raw"] = null
                            // run_qa() assigns support 1.0 to a refusal —
                            // abstaining is trivially grounded. Recorded here
                            // so the column is non-null on every row and the
                            // mean is not silently taken over answered rows
                            // only.
                            rec["evidence_support"] = 1.0
                            rec["grounding_overridden"] = false
                            rec["ttft_ms"] = r.embedMs + r.retrievalMs
                            rec["prompt_tokens"] = 0
                            rec["tokens_generated"] = 0
                            rec["decode_ms_per_token"] = 0.0
                            rec["total_ms"] = r.embedMs + r.retrievalMs
                        } else {
                            val ctx = if (condition.compressionEnabled)
                                PromptBuilder.compress(question, r.results, condition.tokenBudget)
                            else r.results.joinToString(" ") { it.chunk.text }

                            val prompt = PromptBuilder.build(question, ctx)

                            // THE PREMISE. The exact string the model was
                            // given, after selective-context compression. Not
                            // reconstructable offline from retrieved_ids,
                            // because compress() performs sentence selection
                            // under a token budget. HCM needs this string, not
                            // the raw chunks.
                            rec["context"] = ctx
                            rec["prompt_chars"] = prompt.length

                            val g = generateBlocking(prompt, condition.maxTokens, hw)

                            // Post-generation grounding, per run_qa(). Three
                            // fields rather than one, so the override rate is
                            // measurable instead of hidden inside the final
                            // answer:
                            //   output_raw        what the model emitted, whole
                            //   output_first_line after truncation, before grounding
                            //   output            what is actually reported
                            val gr = PromptBuilder.applyGrounding(g.text, ctx)
                            rec["output_raw"] = g.text
                            rec["output_first_line"] = gr.rawAnswer
                            rec["output"] = gr.answer
                            rec["evidence_support"] = gr.support
                            rec["grounding_overridden"] = gr.overridden
                            putGen(rec, g)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "question $id failed", e)
                rec["error"] = e.toString()
            }

            rec.putAll(hw.finish())
            out.appendText(JSONObject(rec as Map<*, *>).toString() + "\n")

            if (i < questions.size - 1 && cooldownMs > 0) delay(cooldownMs)
        }

        onProgress(Progress(questions.size, questions.size, "done"))
        val hwEnd = HardwareMonitor.snapshot()
        out.appendText(JSONObject(mapOf(
            "record_type" to "run_footer",
            "run_tag" to condition.tag,
            "finished_at" to System.currentTimeMillis(),
            "thermal_c_end" to hwEnd.thermalC,
            "cpu_freq_mhz_end" to hwEnd.cpuFreqMhz,
            "avail_mem_mb_end" to hwEnd.availMemMb,
        )).toString() + "\n")

        Log.i(TAG, "run ${condition.tag} complete -> ${out.absolutePath}")
        out.absolutePath
    }

    private fun putGen(rec: HashMap<String, Any?>, g: GenResult) {
        rec["prompt_tokens"] = g.promptTokens
        rec["tokens_generated"] = g.tokensGenerated
        rec["ttft_ms"] = g.ttftMs
        rec["decode_ms_per_token"] = g.decodeMsPerToken
        rec["total_ms"] = g.totalMs
        rec["peak_rss_mb"] = g.peakRssMb
        // hit_token_cap is derivable from tokens_generated vs max_tokens, but
        // recorded explicitly because it is the single most important
        // diagnostic for whether a generation was complete or guillotined —
        // and because the reference's first-line truncation makes the raw
        // count easy to misread as "the answer was this long".
        rec["prefill_tok_per_s"] =
            if (g.promptTokens > 0 && g.ttftMs > 0)
                g.promptTokens * 1000.0 / g.ttftMs else -1.0
    }
}
