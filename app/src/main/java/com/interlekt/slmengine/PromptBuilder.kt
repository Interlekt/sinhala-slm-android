package com.interlekt.slmengine

import java.text.Normalizer

/**
 * Prompt construction, text handling, evidence grounding and answer
 * normalisation.
 *
 * Ported cell-for-cell from qa-evaluation-updated0729.ipynb (cells 5, 6, 7),
 * which is itself copied from qa-finetuning_v6.ipynb — the notebook that
 * produced the deployed checkpoint. Every constant and every string-handling
 * step below matches the reference; where the Android pipeline necessarily
 * differs (it has chunk structure the notebook does not), that is called out
 * at the point of difference rather than silently reconciled.
 */
object PromptBuilder {

    // ── Constants: must match the notebook's cell 3 ──────────────────────────

    /**
     * The abstention string the checkpoint was FINE-TUNED to emit. Not a
     * string chosen for this app.
     *
     * Independently corroborated on device: an earlier batch's raw output for
     * q002 began with exactly this sentence before degenerating into
     * repetition. The model already produces this phrase natively; the app
     * must use the same one so that gate-triggered abstention and
     * model-native abstention are the same string in the results file, not
     * two different strings denoting the same event.
     */
    const val NO_ANSWER =
        "මෙම ප්‍රශ්නයට පිළිතුරු දීමට ප්‍රමාණවත් තොරතුරු නොමැත."

    /** MAX_NEW_TOKENS. The notebook records this as audited to cover 100% of
     *  gold answers under this tokenizer. */
    const val MAX_NEW_TOKENS = 48

    /** REPETITION_PENALTY. Greedy decoding with no penalty degenerates into
     *  loops on this checkpoint. */
    const val REPETITION_PENALTY = 1.05f

    /** GROUNDING_THRESHOLD: minimum fraction of an answer's content words
     *  that must be traceable to the context for the answer to stand. */
    const val GROUNDING_THRESHOLD = 0.50

    /** MAX_LENGTH: total prompt+completion budget the notebook assumes. */
    const val MAX_LENGTH = 2048

    // ── Instruction block: verbatim from cell 5 ──────────────────────────────
    //
    // PLAIN INSTRUCTION TEXT. There is no chat-template markup of any kind in
    // the trained format — no role headers, no special-token delimiters. The
    // previous version of this file wrapped everything in Llama-3 chat
    // headers, which put the model in a format it was never trained on.

    private val INSTRUCTION = """උපදෙස්: පහත සන්දර්භය පමණක් භාවිතා කර ප්‍රශ්නයට පිළිතුරු දෙන්න.
- පිළිතුර සන්දර්භයේ තිබේ නම්, එයින් කෙටිම නිශ්චිත වචන පෙළ පමණක් දෙන්න.
- අමතර පැහැදිලි කිරීම්, පිටත දැනුම හෝ අනුමාන එකතු නොකරන්න.
- සන්දර්භය ප්‍රශ්නයට අදාළ නොවේ නම්, ප්‍රශ්නයට පිළිතුරු දීමට සුදුසු නොවේ නම්, හෝ පිළිතුර සන්දර්භයේ පැහැදිලිව නොමැති නම්, හරියටම මෙය පමණක් දෙන්න: $NO_ANSWER"""

    fun abstentionAnswer(): String = NO_ANSWER

    // ── Text handling: cell 5 ────────────────────────────────────────────────

    /** Port of clean_text(): NFC-normalise, unify line endings, trim. */
    fun cleanText(value: String?): String {
        if (value == null) return ""
        return Normalizer.normalize(value, Normalizer.Form.NFC)
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .trim()
    }

    /** Port of SINHALA_WORD_RE = re.compile(r"[\w඀-෿]+", re.UNICODE).
     *  Written as an explicit class because Android's ICU regex treats bare
     *  \w as ASCII-only; \p{L}\p{N}_ reproduces Python's Unicode \w. */
    private val sinhalaWordRe = Regex("""[\p{L}\p{N}_\u0D80-\u0DFF]+""")

    /**
     * Port of STOPWORDS.
     *
     * Scope matters: these are used ONLY by lexical_tokens(), which feeds
     * evidence_support() and token-overlap scoring. They are NOT applied to
     * the prompt — the context and question go into build_prompt() as full
     * text with nothing removed. Filtering them from a similarity computation
     * is correct (they match almost any context and would inflate the score);
     * filtering them from the prompt would be corruption.
     */
    private val STOPWORDS = setOf(
        "හා", "සහ", "හෝ", "දී", "ද", "ය", "යි", "වේ", "විය", "වූ", "ලෙස",
        "විසින්", "සඳහා", "සිට", "දක්වා", "එම", "මෙම", "ඒ", "ඔහු", "ඇය",
        "කුමක්ද", "කවුද", "කවදාද", "කෙසේද", "කොපමණද", "මොනවාද",
    )

    /** Port of lexical_tokens(): tokenise, casefold, drop <2 chars and
     *  stopwords. */
    fun lexicalTokens(value: String?): List<String> =
        sinhalaWordRe.findAll(cleanText(value))
            .map { it.value.lowercase() }
            .filter { it.length >= 2 && it !in STOPWORDS }
            .toList()

    // ── Prompt: cell 5 build_prompt() ────────────────────────────────────────

    /**
     * Port of build_prompt(context, question).
     *
     * The exact whitespace layout is part of the trained format: blank line
     * after the instruction, "සන්දර්භය:\n" then context, blank line,
     * "ප්‍රශ්නය:\n" then question, blank line, "පිළිතුර:\n" with a trailing
     * newline and nothing after it. The notebook asserts each of these
     * (cell 9) precisely because the format drifted once before. Note the
     * cue ends with a newline, NOT with an opening bracket — an earlier
     * version of the training format ended "පිළිතුර: [" and the notebook
     * explicitly guards against reintroducing it.
     */
    fun build(query: String, context: String): String =
        "$INSTRUCTION\n\n" +
                "සන්දර්භය:\n${cleanText(context)}\n\n" +
                "ප්‍රශ්නය:\n${cleanText(query)}\n\n" +
                "පිළිතුර:\n"

    /** Assertions from cell 9, callable at startup so a format drift fails
     *  loudly at launch rather than silently across a 45-minute batch. */
    fun assertFormat() {
        val p = build("ප්‍රශ්නය", "සන්දර්භය")
        require(p.startsWith("උපදෙස්: පහත සන්දර්භය පමණක්")) { "instruction text drifted" }
        require(p.contains("\nසන්දර්භය:\n")) { "context label drifted" }
        require(p.contains("\nප්‍රශ්නය:\n")) { "question label drifted" }
        require(p.endsWith("පිළිතුර:\n")) { "answer cue drifted (must NOT end with '[')" }
    }

    // ── Answer post-processing: cell 6 generate_candidate() ──────────────────

    /**
     * Port of the two-line answer cleanup in generate_candidate():
     *
     *     answer = decode(...).strip()
     *     answer = answer.splitlines()[0].strip(" []{}()<>\"'`") if answer else ""
     *
     * FIRST LINE ONLY. This is the reference's stopping mechanism — there is
     * no stop-sequence machinery in generation, so the model runs to
     * MAX_NEW_TOKENS and everything after the first newline is discarded in
     * post-processing.
     *
     * This is what an earlier device run needed and did not have: raw outputs
     * hit the 128-token cap 83% of the time and were scored in full, including
     * self-generated follow-up questions and repetition, which drove
     * reference-based metrics to near zero regardless of quantisation. Applied
     * here, the same generations reduce to their first line.
     */
    fun extractAnswer(raw: String?): String {
        val trimmed = cleanText(raw)
        if (trimmed.isEmpty()) return ""
        return trimmed.lineSequence().first()
            .trim(' ', '[', ']', '{', '}', '(', ')', '<', '>', '"', '\'', '`')
    }

    // ── Metrics helpers: cell 7 ──────────────────────────────────────────────

    /** Port of normalize_answer(). The strip set includes the Devanagari
     *  danda U+0964 and the Sinhala kunddaliya U+0DF4, both present in this
     *  corpus. */
    fun normalizeAnswer(value: String?): String =
        cleanText(value).lowercase()
            .replace(Regex("""\s+"""), " ")
            .trim(' ', '\t', '\r', '\n', '[', ']', '{', '}', '(', ')',
                '<', '>', '"', '\'', '`', '.', ',', '!', '?', ';', ':',
                '\u0964', '\u0DF4')

    /** Port of is_no_answer(). The substring branch catches truncated or
     *  near-miss refusals that exact normalised equality would reject. */
    fun isNoAnswer(value: String?): Boolean {
        val n = normalizeAnswer(value)
        return n == normalizeAnswer(NO_ANSWER) ||
                n.contains("ප්‍රමාණවත් තොරතුරු නොමැත")
    }

    // ── Evidence grounding: cell 6 ───────────────────────────────────────────

    /**
     * Port of token_supported(): exact substring match, or a one-character
     * stem match for tokens of length >= 4. Sinhala case endings commonly add
     * a single character, so the stem check keeps genuinely grounded
     * inflected variants that exact matching would miss.
     */
    private fun tokenSupported(token: String, normalizedContext: String): Boolean =
        normalizedContext.contains(token) ||
                (token.length >= 4 && normalizedContext.contains(token.dropLast(1)))

    /**
     * Port of evidence_support(): fraction of the ANSWER's lexical tokens
     * present in the context. Returns 0.0 for an empty token list — an answer
     * with no checkable content words cannot be treated as grounded.
     */
    fun evidenceSupport(answer: String, context: String): Double {
        val answerTokens = lexicalTokens(answer)
        if (answerTokens.isEmpty()) return 0.0
        val normalizedContext = lexicalTokens(context).joinToString(" ")
        return answerTokens.count { tokenSupported(it, normalizedContext) }.toDouble() /
                answerTokens.size
    }

    /**
     * Port of run_qa()'s grounding decision.
     *
     * This is a SECOND hallucination-mitigation layer, independent of the
     * retrieval boundary gate and running at a different point in the
     * pipeline. The boundary gate decides BEFORE generation whether the
     * corpus can answer the question at all. This runs AFTER generation and
     * asks a narrower question: are the words in the answer the model
     * actually produced traceable to the context it was actually given? A
     * query can pass the gate legitimately — real evidence exists — and the
     * model can still assert specifics absent from that evidence. That is the
     * confidently-wrong failure mode, and this catches it for the cost of
     * some string comparisons, with no second model call.
     *
     * The notebook generates over TOP_K_WINDOWS=5 candidate windows and picks
     * the best-supported. Here there is one candidate, because RagEngine's
     * compress() has already fitted the context under the token budget before
     * generation — which is precisely the branch where rank_context_windows()
     * returns a single window immediately (`if len(context_ids) <=
     * token_budget: return [(context, 1.0)]`). Same code path, reached
     * earlier.
     */
    fun applyGrounding(
        rawAnswer: String,
        context: String,
        threshold: Double = GROUNDING_THRESHOLD,
    ): GroundingResult {
        val answer = extractAnswer(rawAnswer)

        if (answer.isEmpty() || isNoAnswer(answer)) {
            // Already a refusal, or nothing usable. run_qa() assigns support
            // 1.0 to a refusal: abstaining is trivially grounded.
            return GroundingResult(NO_ANSWER, answer, support = 1.0, overridden = false)
        }

        val support = evidenceSupport(answer, context)
        return if (support >= threshold) {
            GroundingResult(answer, answer, support, overridden = false)
        } else {
            GroundingResult(NO_ANSWER, answer, support, overridden = true)
        }
    }

    data class GroundingResult(
        /** What to report: the model's answer, or NO_ANSWER if ungrounded. */
        val answer: String,
        /** The model's own first-line output, before the grounding decision.
         *  Recorded separately so the override rate is measurable. */
        val rawAnswer: String,
        val support: Double,
        val overridden: Boolean,
    )

    // ── Selective context (this project's own, retrieval-side) ───────────────

    private const val CHARS_PER_TOKEN = 2.5

    /**
     * Serves the same purpose as the notebook's rank_context_windows() — fit
     * the evidence under a token budget while keeping the query-relevant part
     * — but by a different mechanism, because the inputs differ.
     *
     * rank_context_windows slides a fixed-width token window over one long
     * raw context and scores each window by query-token coverage, because
     * that is all the notebook has. Here, RagEngine has already retrieved and
     * RANKED discrete chunks, so this scores and selects at sentence level
     * using that existing rank as a prior. Both end with a context that fits
     * the budget; the Android version starts from more structure.
     */
    fun compress(query: String, results: List<Retrieved>, tokenBudget: Int = 384): String {
        val qTerms = lexicalTokens(query).toSet()
        val budgetChars = (tokenBudget * CHARS_PER_TOKEN).toInt()

        data class Scored(val chunkRank: Int, val order: Int, val text: String, val score: Double)

        val sentences = results.flatMap { r ->
            r.chunk.text.split(Regex("""(?<=[.\u0964?!])\s+""")).filter { it.isNotBlank() }
                .mapIndexed { i, s ->
                    val terms = lexicalTokens(s).toSet()
                    val hits = qTerms.count { it in terms }.toDouble()
                    Scored(r.rank, i, s.trim(), hits / (1.0 + r.rank))
                }
        }

        val kept = sentences.sortedByDescending { it.score }
            .fold(Pair(mutableListOf<Scored>(), 0)) { (acc, used), s ->
                if (used + s.text.length <= budgetChars) {
                    acc.add(s); Pair(acc, used + s.text.length)
                } else Pair(acc, used)
            }.first

        // Restore document order so the model sees coherent prose.
        return kept.sortedWith(compareBy({ it.chunkRank }, { it.order }))
            .joinToString(" ") { it.text }
    }
}
