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
 * metric. Quality metrics (Token F1, ROUGE-L, EM, abstention accuracy, HCM) are
 * computed afterwards on the desktop, where the NLI model for HCM lives — the
 * device only needs to produce the generations and the timings.
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

    data class Condition(
        /** Identifies the quantisation of the loaded SLM, e.g. "Q4_K_M". */
        val quant: String,
        val ragEnabled: Boolean,
        val boundaryGateEnabled: Boolean,
        val compressionEnabled: Boolean,
        val boundaryMode: BoundaryMode = BoundaryMode.DESKTOP_PARITY,
        val tokenBudget: Int = 384,
        val maxTokens: Int = 128,
    ) {
        /** Stable filename/label for this cell of the design. */
        val tag: String get() = buildString {
            append(quant)
            append(if (ragEnabled) "_rag" else "_norag")
            if (ragEnabled) {
                append(if (boundaryGateEnabled) "_gate" else "_nogate")
                if (compressionEnabled) append("_compress")
                if (boundaryGateEnabled) append("_${boundaryMode.name.lowercase()}")
            }
        }
    }

    data class Progress(val done: Int, val total: Int, val currentId: String)

    /**
     * Battery temperature in celsius, or null if unreadable.
     *
     * Thermal state is the largest confound in on-device latency work. The
     * Helio G85 throttles hard, so a slow configuration heats the device and
     * then looks even slower — which would be misattributed to quantisation.
     * Recording this per question makes the confound visible instead of silent.
     */
    private fun batteryTempC(): Double? = try {
        val raw = File("/sys/class/power_supply/battery/temp").readText().trim().toInt()
        raw / 10.0        // decidegrees
    } catch (_: Exception) {
        null
    }

    /**
     * wrapper.generate() is itself a suspend function that returns only when
     * nativeGenerate's loop finishes, so it needs no suspendCoroutine wrapper
     * and no onDone() callback — returning IS completion.
     *
     * Peak RSS is sampled per token rather than only before/after, since the
     * maximum is what a memory ceiling actually cares about.
     */
    private suspend fun generateBlocking(
        prompt: String,
        maxTokens: Int,
    ): GenResult {
        val t0 = System.currentTimeMillis()
        var firstAt = 0L
        var count = 0
        var promptTokens = -1
        var peakRss = MetricsCollector.ramUsedMB()
        val sb = StringBuilder()

        wrapper.generate(prompt, maxTokens, object : LlamaWrapper.GenerationCallback {
            override fun onToken(piece: String) {
                val now = System.currentTimeMillis()
                if (count == 0) firstAt = now
                count++
                sb.append(piece)
                val rss = MetricsCollector.ramUsedMB()
                if (rss > peakRss) peakRss = rss
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
     * @param cooldownMs pause between questions to let the SoC settle. 20-30 s
     *        is a reasonable starting point on the G85; verify by checking that
     *        batteryTempC does not drift upward across the run.
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

        val questions = qFile.readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
        val out = File(dir, "results_${condition.tag}.jsonl")
        out.writeText("")

        rag.boundaryMode = condition.boundaryMode

        // One header line so a results file is self-describing even if the
        // filename is renamed or the run parameters are forgotten.
        out.appendText(JSONObject(mapOf(
            "record_type" to "run_header",
            "run_tag" to condition.tag,
            "quant" to condition.quant,
            "rag_enabled" to condition.ragEnabled,
            "boundary_gate" to condition.boundaryGateEnabled,
            "compression" to condition.compressionEnabled,
            "boundary_mode" to condition.boundaryMode.name,
            "token_budget" to condition.tokenBudget,
            "max_tokens" to condition.maxTokens,
            "n_questions" to questions.size,
            "device" to "${Build.MANUFACTURER} ${Build.MODEL}",
            "android" to Build.VERSION.RELEASE,
            "n_chunks" to if (rag.loaded) rag.config.nChunks else -1,
            "sampler" to "greedy",
            "started_at" to System.currentTimeMillis(),
            "battery_temp_c_start" to (batteryTempC() ?: -1.0),
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
            rec["reference"] = q.optString("reference", "")
            rec["in_syllabus_gold"] = if (q.has("in_syllabus")) q.optBoolean("in_syllabus") else null
            rec["run_tag"] = condition.tag
            rec["battery_temp_c"] = batteryTempC() ?: -1.0
            rec["rss_mb_before"] = MetricsCollector.ramUsedMB()

            try {
                if (!condition.ragEnabled) {
                    val g = generateBlocking(question, condition.maxTokens)
                    rec["abstained"] = false
                    rec["output"] = g.text
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
                        rec["top_cosine"] = r.denseCosine.firstOrNull()?.second ?: -1.0
                        // The device's own query vector. Lets simulate_device.py
                        // run on the exact input the phone used, so ARM-vs-x86
                        // embedding drift stops confounding the parity check.
                        // ~8 KB per question as text; negligible.
                        rec["query_vec"] = r.queryVec?.map { it.toDouble() }
                        rec["retrieved_cosines"] = r.results.map { it.cosine }
                        rec["boundary_label"] = r.boundary.label
                        rec["boundary_top_score"] = r.boundary.topScore
                        rec["boundary_avg_score"] = r.boundary.avgScore
                        rec["boundary_overlap"] = r.boundary.overlap
                        rec["boundary_reason"] = r.boundary.reason
                        rec["boundary_fired"] = r.boundary.firedConditions
                        // Which of the three OR'd conditions actually passed.
                        // Aggregated across the run this quantifies the claim
                        // that top_score is unreachable and the decision rests
                        // on lexical overlap alone.
                        rec["boundary_fired"] = r.boundary.firedConditions
                        rec["out_of_syllabus_pred"] = r.boundary.isOutOfSyllabus

                        val abstain = condition.boundaryGateEnabled && r.boundary.isOutOfSyllabus
                        rec["abstained"] = abstain

                        if (abstain) {
                            // The SLM is never invoked. This is the mitigation,
                            // and it is also why abstained queries are ~100x
                            // cheaper than answered ones.
                            rec["output"] = PromptBuilder.abstentionAnswer()
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
                            rec["context_chars"] = ctx.length
                            rec["prompt_chars"] = prompt.length

                            val g = generateBlocking(prompt, condition.maxTokens)
                            rec["output"] = g.text
                            putGen(rec, g)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "question $id failed", e)
                rec["error"] = e.toString()
            }

            rec["rss_mb_after"] = MetricsCollector.ramUsedMB()
            out.appendText(JSONObject(rec as Map<*, *>).toString() + "\n")

            if (i < questions.size - 1 && cooldownMs > 0) delay(cooldownMs)
        }

        onProgress(Progress(questions.size, questions.size, "done"))
        out.appendText(JSONObject(mapOf(
            "record_type" to "run_footer",
            "run_tag" to condition.tag,
            "finished_at" to System.currentTimeMillis(),
            "battery_temp_c_end" to (batteryTempC() ?: -1.0),
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
        // Derived here rather than offline so a bad prompt-token count is
        // obvious in the raw file.
        rec["prefill_tok_per_s"] =
            if (g.promptTokens > 0 && g.ttftMs > 0)
                g.promptTokens * 1000.0 / g.ttftMs else -1.0
    }
}
