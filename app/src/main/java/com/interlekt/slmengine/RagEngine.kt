package com.interlekt.slmengine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer

private const val TAG = "SLMRag"

// ─── Data ────────────────────────────────────────────────────────────────────

data class Chunk(
    val id: String,
    val text: String,
    val grade: String,
    val chapter: String,
    val section: String,
    val page: String,
    val source: String,
)

data class Retrieved(
    val rank: Int,
    val score: Double,      // fused RRF score
    val cosine: Double,     // dense cosine, or 0 if not in the dense pool
    val chunk: Chunk,
)

data class Boundary(
    val label: String,
    val isOutOfSyllabus: Boolean,
    val confidence: Double,
    val topScore: Double,
    val avgScore: Double,
    val overlap: Double,
    val reason: String,
    val firedConditions: String,   // which gate(s) actually passed
) {
    val inBoundary: Boolean get() = !isOutOfSyllabus
}

/**
 * DESKTOP_PARITY  reproduces knowledge_boundary_detection() exactly, including
 *                 the top_score gate that can never fire (threshold 0.035 vs a
 *                 maximum achievable fused score of 2/61 = 0.032787).
 * CALIBRATED      gates on dense cosine + IDF-weighted overlap, combined with
 *                 AND rather than OR.
 * Report both.
 */
enum class BoundaryMode { DESKTOP_PARITY, CALIBRATED }

data class RagResult(
    val results: List<Retrieved>,
    val contextStr: String,
    val boundary: Boundary,
    val retrievalMs: Long,
    val embedMs: Long,
    /** (docIndex, cosine) for the dense pool, best first. Needed by the
     *  calibrated boundary detector and recorded by EvalRunner. */
    val denseCosine: List<Pair<Int, Double>>,
    /** The query vector the device actually computed. Logged so a desktop
     *  reference can be driven by it, removing ARM-vs-x86 embedding drift as a
     *  confound in parity checks. */
    val queryVec: FloatArray? = null,
)

// ─── Config, read from manifest.json so device and desktop cannot drift ──────

data class RagConfig(
    val embeddingDim: Int,
    val nChunks: Int,
    val topK: Int,
    val rrfK: Int,
    val wDense: Double,
    val wBm25: Double,
    val densePool: Int,
    val bm25Pool: Int,
    val k1: Double,
    val b: Double,
    // desktop parity
    val topScoreThreshold: Double,
    val avgScoreThreshold: Double,
    val overlapThreshold: Double,
    // calibrated
    val cosineTopThreshold: Double,
    val cosineMeanThreshold: Double,
    val idfOverlapThreshold: Double,
) {
    companion object {
        fun fromManifest(json: String): RagConfig {
            val m = JSONObject(json)
            val r = m.getJSONObject("retrieval")
            val bm = m.getJSONObject("bm25")
            val dp = m.getJSONObject("boundary_desktop_parity")
            val cal = m.getJSONObject("boundary_calibrated")
            return RagConfig(
                embeddingDim = m.getInt("embedding_dim"),
                nChunks = m.getInt("n_chunks"),
                topK = r.getInt("top_k"),
                rrfK = r.getInt("rrf_k"),
                wDense = r.getDouble("w_dense"),
                wBm25 = r.getDouble("w_bm25"),
                densePool = r.getInt("dense_pool"),
                bm25Pool = r.getInt("bm25_pool"),
                k1 = bm.getDouble("k1"),
                b = bm.getDouble("b"),
                topScoreThreshold = dp.getDouble("top_score_threshold"),
                avgScoreThreshold = dp.getDouble("avg_score_threshold"),
                overlapThreshold = dp.getDouble("overlap_threshold"),
                cosineTopThreshold = cal.getDouble("cosine_top_threshold"),
                cosineMeanThreshold = cal.getDouble("cosine_mean_threshold"),
                idfOverlapThreshold = cal.getDouble("idf_overlap_threshold"),
            )
        }
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

class RagEngine(private val embedder: EmbeddingEngine) {

    lateinit var config: RagConfig
        private set

    private lateinit var chunks: List<Chunk>
    private lateinit var vectors: FloatArray            // nChunks * dim, fp32
    private lateinit var postings: Map<String, IntArray> // term -> [doc, tf, ...]
    private lateinit var idf: Map<String, Double>
    private lateinit var docLens: IntArray
    private var avgdl: Double = 0.0
    private var maxIdf: Double = 1.0

    var boundaryMode: BoundaryMode = BoundaryMode.DESKTOP_PARITY

    var loaded: Boolean = false
        private set

    // ── Loading ──────────────────────────────────────────────────────────────

    suspend fun load(bundleDir: String): Boolean = withContext(Dispatchers.IO) {
        val dir = File(bundleDir)
        if (!dir.isDirectory) {
            Log.e(TAG, "bundle dir not found or unreadable: $bundleDir")
            return@withContext false
        }
        try {
            val t0 = System.currentTimeMillis()

            config = RagConfig.fromManifest(File(dir, "manifest.json").readText())

            chunks = File(dir, "chunks.jsonl").readLines()
                .filter { it.isNotBlank() }
                .map { line ->
                    val o = JSONObject(line)
                    val md = o.optJSONObject("metadata") ?: JSONObject()
                    Chunk(
                        id = o.optString("id"),
                        text = o.optString("text"),
                        grade = md.optString("grade", ""),
                        chapter = md.optString("toc_chapter_title", ""),
                        section = md.optString("toc_section_title", ""),
                        page = md.optString("page", ""),
                        source = md.optString("source_file", ""),
                    )
                }

            // fp32 little-endian, row-major, written by build_bundle.py
            val raw = File(dir, "vectors.f32").readBytes()
            val expected = config.nChunks * config.embeddingDim * 4
            require(raw.size == expected) {
                "vectors.f32 is ${raw.size} bytes, manifest implies $expected"
            }
            vectors = FloatArray(config.nChunks * config.embeddingDim)
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(vectors)

            loadBm25(File(dir, "bm25.json"))

            require(chunks.size == config.nChunks) {
                "chunks.jsonl has ${chunks.size} rows, manifest says ${config.nChunks}"
            }

            loaded = true
            Log.i(TAG, "bundle loaded in ${System.currentTimeMillis() - t0}ms: " +
                    "${config.nChunks} chunks, dim ${config.embeddingDim}, " +
                    "${postings.size} terms, avgdl ${"%.1f".format(avgdl)}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "bundle load failed", e)
            false
        }
    }

    private fun loadBm25(file: File) {
        val root = JSONObject(file.readText())
        avgdl = root.getDouble("avgdl")

        val dl = root.getJSONArray("doc_lens")
        docLens = IntArray(dl.length()) { dl.getInt(it) }

        val idfObj = root.getJSONObject("idf")
        val idfMap = HashMap<String, Double>(idfObj.length() * 2)
        idfObj.keys().forEach { k -> idfMap[k] = idfObj.getDouble(k) }
        idf = idfMap
        maxIdf = idfMap.values.maxOrNull() ?: 1.0

        val pObj = root.getJSONObject("postings")
        val pMap = HashMap<String, IntArray>(pObj.length() * 2)
        pObj.keys().forEach { term ->
            val arr = pObj.getJSONArray(term)
            val flat = IntArray(arr.length() * 2)
            for (i in 0 until arr.length()) {
                val pair = arr.getJSONArray(i)
                flat[i * 2] = pair.getInt(0)
                flat[i * 2 + 1] = pair.getInt(1)
            }
            pMap[term] = flat
        }
        postings = pMap
    }

    // ── Tokenizers ───────────────────────────────────────────────────────────
    //
    // The desktop uses TWO different tokenizers. They are not interchangeable:
    //   tokenize()       <- _tokenize_si    in rag/src/retrieval.py       (BM25)
    //   boundaryTokens() <- _boundary_tokens in knowledge_boundary.py  (overlap)

    /**
     * Verbatim port of _tokenize_si.
     *
     * The ZWJ/ZWNJ stripping is load-bearing. Sinhala writes conjunct
     * consonants as consonant + virama + U+200D + consonant, so a \w-based
     * regex — which does not match U+200D (category Cf) — splits those words
     * in half. Removing the joiner first, as Python does, keeps them whole.
     *
     * Java's Character.getType() maps onto Python's unicodedata.category():
     * types 20-24, 29, 30 are the P* categories; 25-28 are the S* categories.
     */
    fun tokenize(text: String?): List<String> {
        if (text == null) return emptyList()

        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
            .replace("\u200d", "")
            .replace("\u200c", "")

        val sb = StringBuilder(normalized.length)
        var i = 0
        while (i < normalized.length) {
            val cp: Int = normalized.codePointAt(i)
            val n: Int = Character.charCount(cp)
            val type: Int = Character.getType(cp)
            // Python's str.split() separates on ALL Unicode whitespace, so
            // anything Python would treat as a separator must become one here.
            // isWhitespace covers ASCII controls and most Zs; isSpaceChar adds
            // the non-breaking spaces isWhitespace omits (U+00A0, U+202F,
            // U+2007), which PDF-extracted text is full of. Splitting on ASCII
            // whitespace alone silently merges tokens and shifts BM25 ranks.
            if (isPunctOrSymbol(type) ||
                Character.isWhitespace(cp) ||
                Character.isSpaceChar(cp)) {
                sb.append(' ')
            } else {
                sb.append(normalized, i, i + n)
            }
            i += n
        }

        return sb.toString().lowercase()
            .split(' ')
            .filter { s -> s.isNotEmpty() }
    }

    private fun isPunctOrSymbol(type: Int): Boolean = when (type) {
        Character.DASH_PUNCTUATION.toInt(),          // Pd
        Character.START_PUNCTUATION.toInt(),         // Ps
        Character.END_PUNCTUATION.toInt(),           // Pe
        Character.CONNECTOR_PUNCTUATION.toInt(),     // Pc
        Character.OTHER_PUNCTUATION.toInt(),         // Po
        Character.INITIAL_QUOTE_PUNCTUATION.toInt(), // Pi
        Character.FINAL_QUOTE_PUNCTUATION.toInt(),   // Pf
        Character.MATH_SYMBOL.toInt(),               // Sm
        Character.CURRENCY_SYMBOL.toInt(),           // Sc
        Character.MODIFIER_SYMBOL.toInt(),           // Sk
        Character.OTHER_SYMBOL.toInt(),              // So
            -> true
        else -> false
    }

    /**
     * Port of _boundary_tokens: re.findall(r"[\w\u0D80-\u0DFF]+", text.lower())
     *
     * Written without \w because Android's regex engine is ICU, which rejects
     * the JDK-only (?U) flag, and bare \w on Android is ASCII-only.
     *
     * \p{L}\p{N}_ reproduces Python's \w: CPython's SRE_UNI_IS_WORD is
     * isalnum() || '_', i.e. categories L* and N* plus underscore. Note that
     * does NOT include combining marks — which is exactly why the original
     * pattern adds the Sinhala block explicitly, since the vowel signs at
     * U+0DCF-U+0DDF are Mc/Mn and would otherwise be dropped.
     */
    private val boundaryRe = Regex("""[\p{L}\p{N}_\u0D80-\u0DFF]+""")

    private fun boundaryTokens(text: String): Set<String> {
        // Python's \w matches neither U+200D nor U+200C, so _boundary_tokens
        // SPLITS at the joiners: ප්‍රධාන -> {ප්, රධාන}. Converting them to
        // spaces reproduces that exactly, without depending on how ICU parses
        // the \u0D80-\u0DFF range inside a Kotlin raw string.
        //
        // Note this is the OPPOSITE of tokenize(), which STRIPS the joiners
        // because _tokenize_si does. Two tokenizers, two treatments of the same
        // character; the desktop does this and it must be mirrored, not unified.
        val prepared = text.lowercase()
            .replace('\u200d', ' ')
            .replace('\u200c', ' ')
        return boundaryRe.findAll(prepared).map { it.value }.toSet()
    }

    // ── Dense retrieval ──────────────────────────────────────────────────────
    // Flat fp32 scan. 1767 x 1024 is ~1.8M multiply-adds, single-digit ms.
    // Both sides are unit-norm, so the dot product IS cosine similarity —
    // no conversion, and the calibrated boundary detector can gate on it.

    private fun denseSearch(query: FloatArray, pool: Int): List<Pair<Int, Double>> {
        val dim = config.embeddingDim
        val scored = ArrayList<Pair<Int, Double>>(config.nChunks)
        var off = 0
        for (doc in 0 until config.nChunks) {
            var acc = 0.0
            for (i in 0 until dim) acc += query[i] * vectors[off + i]
            off += dim
            scored.add(doc to acc)
        }
        return scored.sortedByDescending { it.second }.take(pool)
    }

    // ── BM25 ─────────────────────────────────────────────────────────────────

    /**
     * Matches rank_bm25 BM25Okapi.get_scores + KeyWordRetriever.search.
     *
     * Three parity details:
     *  - Query tokens are iterated WITH duplicates, so a repeated term counts
     *    twice, as rank_bm25 does.
     *  - Every document is scored and exactly `pool` returned regardless of
     *    score. KeyWordRetriever.search applies no floor, so zero-scoring
     *    documents still enter the fusion and receive RRF weight. Returning
     *    only matched documents would shift every subsequent BM25 rank.
     *  - Kotlin's sortedByDescending is stable, matching Python's
     *    sorted(..., reverse=True), which breaks ties by ascending index.
     */
    private fun bm25Search(query: String, pool: Int): List<Pair<Int, Double>> {
        val terms = tokenize(query)
        if (terms.isEmpty()) return emptyList()

        val scores = DoubleArray(config.nChunks)
        val k1 = config.k1
        val b = config.b

        for (term in terms) {
            val plist = postings[term] ?: continue
            val termIdf = idf[term] ?: continue
            var i = 0
            while (i < plist.size) {
                val doc = plist[i]
                val tf = plist[i + 1].toDouble()
                val norm = 1.0 - b + b * (docLens[doc] / avgdl)
                scores[doc] += termIdf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
                i += 2
            }
        }

        return (0 until config.nChunks)
            .sortedByDescending { scores[it] }
            .take(minOf(pool, config.nChunks))
            .map { it to scores[it] }
    }

    // ── RRF fusion. Must be arithmetically identical to HybridRetriever. ─────

    private fun rrfFuse(
        dense: List<Pair<Int, Double>>,
        bm25: List<Pair<Int, Double>>,
    ): List<Pair<Int, Double>> {
        // LinkedHashMap, dense inserted FIRST, because HybridRetriever builds
        // its result list by iterating a Python dict — which preserves
        // insertion order.
        //
        // Not a micro-detail: exact ties are the norm here. A document at dense
        // rank r and a different document at bm25 rank r both score precisely
        // w/(rrf_k + r). On any query where no document appears in both lists —
        // i.e. every out-of-domain query — the whole top-5 is decided by
        // tie-breaking. HashMap breaks those ties in hash-bucket order and
        // silently reshuffles the result.
        //
        // LinkedHashMap.put on an existing key leaves its position unchanged,
        // matching Python dict semantics, and sortedByDescending is stable, so
        // ties resolve to insertion order.
        val fused = LinkedHashMap<Int, Double>()
        val k = config.rrfK.toDouble()
        dense.forEachIndexed { idx, (doc, _) ->
            fused[doc] = (fused[doc] ?: 0.0) + config.wDense / (k + (idx + 1))
        }
        bm25.forEachIndexed { idx, (doc, _) ->
            fused[doc] = (fused[doc] ?: 0.0) + config.wBm25 / (k + (idx + 1))
        }
        return fused.entries.sortedByDescending { it.value }.map { it.key to it.value }
    }

    // ── Boundary detection ───────────────────────────────────────────────────

    /** Faithful port of knowledge_boundary_detection(). OR across three gates. */
    private fun boundaryDesktopParity(
        query: String,
        results: List<Retrieved>,
        contextStr: String,
        minChunks: Int = 1,
    ): Boundary {
        val queryTokens = boundaryTokens(query)
        val contextTokens = boundaryTokens(contextStr)

        val topScore = results.maxOfOrNull { it.score } ?: 0.0
        val avgScore = if (results.isEmpty()) 0.0 else results.map { it.score }.average()
        val overlap = if (queryTokens.isEmpty()) 0.0
        else queryTokens.count { it in contextTokens }.toDouble() / queryTokens.size
        val evidenceChars = contextStr.trim().length

        val topOk = topScore >= config.topScoreThreshold
        val avgOk = avgScore >= config.avgScoreThreshold
        val ovlOk = overlap >= config.overlapThreshold

        val supported = results.isNotEmpty() && results.size >= minChunks &&
                evidenceChars > 0 && (topOk || avgOk || ovlOk)

        // Which gate actually carried the decision. Log this: on your config
        // topOk can never be true, so if only ovlOk ever fires, the abstention
        // mechanism is lexical overlap alone and the dense retriever plays no
        // part in it.
        val fired = listOfNotNull(
            if (topOk) "top" else null,
            if (avgOk) "avg" else null,
            if (ovlOk) "overlap" else null,
        ).joinToString("+").ifEmpty { "none" }

        // Mirrors the Python confidence field. Because it maxes a ~0.03-scale
        // RRF score against a 0-1 ratio it is always `overlap`. Kept for parity;
        // do not present it as a confidence.
        val confidence = minOf(0.99, maxOf(0.01, maxOf(topScore, avgScore, overlap)))

        val reasons = if (supported) {
            listOf("top_score=%.3f".format(topScore),
                "avg_score=%.3f".format(avgScore),
                "overlap=%.3f".format(overlap))
        } else buildList {
            if (results.isEmpty()) add("no retrieved chunks")
            if (!topOk) add("top score below threshold (%.3f < %.3f)"
                .format(topScore, config.topScoreThreshold))
            if (!avgOk) add("average score below threshold (%.3f < %.3f)"
                .format(avgScore, config.avgScoreThreshold))
            if (!ovlOk) add("query/context overlap below threshold (%.3f < %.3f)"
                .format(overlap, config.overlapThreshold))
            if (evidenceChars == 0) add("empty retrieved context")
        }

        return Boundary(
            label = if (supported) "within syllabus" else "out of syllabus",
            isOutOfSyllabus = !supported,
            confidence = confidence,
            topScore = topScore,
            avgScore = avgScore,
            overlap = overlap,
            reason = if (reasons.isEmpty()) "retrieved context supports the query"
            else reasons.joinToString("; "),
            firedConditions = fired,
        )
    }

    /**
     * Proposed variant:
     *   1. gates on dense COSINE, not fused RRF score (calibrated, interpretable,
     *      and on a scale where thresholds mean something)
     *   2. IDF-weighted overlap — plain token overlap on inflected Sinhala
     *      rewards common particles and penalises specific queries
     *   3. AND across two independent signals instead of OR across three
     */
    private fun boundaryCalibrated(
        query: String,
        results: List<Retrieved>,
        contextStr: String,
        denseCosine: List<Pair<Int, Double>>,
    ): Boundary {
        val topCos = denseCosine.maxOfOrNull { it.second } ?: 0.0
        val meanCos = if (denseCosine.isEmpty()) 0.0
        else denseCosine.take(config.topK).map { it.second }.average()

        val queryTokens = boundaryTokens(query)
        val contextTokens = boundaryTokens(contextStr)

        // A query term absent from the whole corpus is maximally informative,
        // and its absence maximally damning — so it gets maxIdf, not zero.
        var num = 0.0
        var den = 0.0
        for (t in queryTokens) {
            val w = idf[t] ?: maxIdf
            den += w
            if (t in contextTokens) num += w
        }
        val idfOverlap = if (den > 0.0) num / den else 0.0

        val semanticOk = topCos >= config.cosineTopThreshold ||
                meanCos >= config.cosineMeanThreshold
        val lexicalOk = idfOverlap >= config.idfOverlapThreshold
        val supported = results.isNotEmpty() && contextStr.trim().isNotEmpty() &&
                semanticOk && lexicalOk

        val reasons = buildList {
            if (results.isEmpty()) add("no retrieved chunks")
            if (!semanticOk) add("no semantic support (top_cos=%.3f mean_cos=%.3f)"
                .format(topCos, meanCos))
            if (!lexicalOk) add("no lexical anchor (idf_overlap=%.3f < %.3f)"
                .format(idfOverlap, config.idfOverlapThreshold))
        }

        return Boundary(
            label = if (supported) "within syllabus" else "out of syllabus",
            isOutOfSyllabus = !supported,
            confidence = topCos,          // an actual similarity, one scale
            topScore = topCos,
            avgScore = meanCos,
            overlap = idfOverlap,
            reason = if (reasons.isEmpty())
                "top_cos=%.3f mean_cos=%.3f idf_overlap=%.3f".format(topCos, meanCos, idfOverlap)
            else reasons.joinToString("; "),
            firedConditions = listOfNotNull(
                if (semanticOk) "semantic" else null,
                if (lexicalOk) "lexical" else null,
            ).joinToString("+").ifEmpty { "none" },
        )
    }

    // ── Entry point ──────────────────────────────────────────────────────────

    suspend fun retrieve(query: String): RagResult? {
        if (!loaded || !embedder.loaded) {
            Log.e(TAG, "retrieve() before load()")
            return null
        }

        val tEmbed = System.currentTimeMillis()
        val qVec = embedder.embed(query) ?: return null
        val embedMs = System.currentTimeMillis() - tEmbed

        val tRet = System.currentTimeMillis()
        val (results, denseCosine) = withContext(Dispatchers.Default) {
            val dense = denseSearch(qVec, config.densePool)
            val bm25 = bm25Search(query, config.bm25Pool)
            val cosByDoc = dense.toMap()
            val fused = rrfFuse(dense, bm25)
                .take(config.topK)
                .mapIndexed { i, (doc, score) ->
                    Retrieved(i + 1, score, cosByDoc[doc] ?: 0.0, chunks[doc])
                }
            fused to dense
        }
        val retrievalMs = System.currentTimeMillis() - tRet

        // Same concatenation as the desktop pipeline: "[text][text]..."
        val contextStr = results.joinToString("") { "[${it.chunk.text}]" }

        val boundary = when (boundaryMode) {
            BoundaryMode.DESKTOP_PARITY -> boundaryDesktopParity(query, results, contextStr)
            BoundaryMode.CALIBRATED -> boundaryCalibrated(query, results, contextStr, denseCosine)
        }

        Log.i(TAG, "q=\"${query.take(40)}\" embed=${embedMs}ms retrieve=${retrievalMs}ms " +
                "ids=${results.map { it.chunk.id }} " +
                "fused=${results.map { "%.6f".format(it.score) }} " +
                "cos=${results.map { "%.4f".format(it.cosine) }} " +
                "boundary=${boundary.label} fired=${boundary.firedConditions}")

        return RagResult(results, contextStr, boundary, retrievalMs, embedMs,
            denseCosine, qVec)
    }
}

// ─── Prompt construction and abstention ──────────────────────────────────────

object PromptBuilder {

    // Review this wording yourself — it becomes part of your abstention-accuracy
    // metric, so its exact phrasing matters.
    private const val ABSTAIN =
        "මට ලබා දී ඇති තොරතුරු අනුව එයට පිළිතුරු දිය නොහැක."

    private const val SYSTEM = """ඔබ ලබා දී ඇති සන්දර්භය මත පමණක් පිළිතුරු දෙන සහායකයෙකි.
සන්දර්භයේ නොමැති කිසිදු තොරතුරක් එකතු නොකරන්න.
සන්දර්භයේ පිළිතුර නොමැති නම්, "$ABSTAIN" යනුවෙන් පමණක් පිළිතුරු දෙන්න."""

    /** Calibrate against SinLlama's actual BPE rather than trusting this. */
    private const val CHARS_PER_TOKEN = 2.5

    fun abstentionAnswer(): String = ABSTAIN

    /**
     * Lexical sentence filter under a hard token budget.
     *
     * At 42 ms/token prefill, 384 tokens ≈ 16 s to first token. Raising the
     * budget buys evidence and costs latency roughly linearly.
     *
     * This can INCREASE hallucination by dropping needed evidence. Measure HCM
     * with it on and off before trusting it.
     */
    fun compress(query: String, results: List<Retrieved>, tokenBudget: Int = 384): String {
        val qTerms = query.lowercase().split(Regex("""\W+""")).filter { it.length > 1 }.toSet()
        val budgetChars = (tokenBudget * CHARS_PER_TOKEN).toInt()

        data class Scored(val chunkRank: Int, val order: Int, val text: String, val score: Double)

        val sentences = results.flatMap { r ->
            r.chunk.text.split(Regex("""(?<=[.।?!])\s+""")).filter { it.isNotBlank() }
                .mapIndexed { i, s ->
                    val terms = s.lowercase().split(Regex("""\W+""")).toSet()
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

        // Restore document order so the SLM sees coherent prose.
        return kept.sortedWith(compareBy({ it.chunkRank }, { it.order }))
            .joinToString(" ") { it.text }
    }

    /** Llama-3 template. nativeGenerate must NOT wrap this again — see step 7. */
    fun build(query: String, context: String): String = buildString {
        // No <|begin_of_text|>: llama_jni.cpp tokenizes with add_special=true,
        // which already prepends BOS. Including it here would double it.
        append("<|start_header_id|>system<|end_header_id|>\n\n")
        append(SYSTEM)
        append("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")
        append("සන්දර්භය:\n")
        append(context)
        append("\n\nප්‍රශ්නය: ")
        append(query)
        append("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    }
}
