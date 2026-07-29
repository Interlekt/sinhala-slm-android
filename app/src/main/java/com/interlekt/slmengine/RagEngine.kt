package com.interlekt.slmengine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import kotlin.math.exp
import kotlin.math.round

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
    /** P(within syllabus) from the logistic model. -1.0 for the two legacy
     *  modes, which have no single probability. */
    val probability: Double = -1.0,
) {
    val inBoundary: Boolean get() = !isOutOfSyllabus
}

/**
 * DESKTOP_PARITY        legacy fixed-threshold rule: top_score OR avg_score OR
 *                       overlap, on FUSED RRF scores. top_score never fires in
 *                       practice — calibration selected a threshold above the
 *                       achievable RRF maximum, deliberately switching that
 *                       branch off. Kept for historical comparison only; this
 *                       is NOT what the desktop runs any more (see LOGISTIC).
 * CALIBRATED            this project's proposed variant: (topCos OR meanCos)
 *                       AND idfOverlap.
 * LOGISTIC_REGRESSION   the CURRENT desktop implementation, per
 *                       knowledge_boundary.py: a logistic regression over
 *                       five raw signals from two SEPARATE retrieval passes
 *                       (dense-only top-5, BM25-only top-5) — not the fused
 *                       hybrid list used to build the LLM-facing context.
 *                       Report all three.
 */
enum class BoundaryMode { DESKTOP_PARITY, CALIBRATED, LOGISTIC_REGRESSION }

/**
 * Every signal any of the three detectors uses, computed on every query
 * regardless of which mode is active.
 *
 * Recording all of these means one results file supports evaluating any of the
 * three boundaries offline and sweeping thresholds without re-running 45
 * minutes of inference. Retrieval is unchanged by the choice of gate, so the
 * signals are identical across modes — only which of them is consulted
 * differs.
 */
data class BoundarySignals(
    // ── legacy fixed-threshold inputs (fused RRF list) ──────────────────────
    val topScore: Double,        // max fused RRF score  — rank-derived
    val avgScore: Double,        // mean fused RRF score — rank-derived
    val overlap: Double,         // token overlap, query vs FUSED hybrid context
    val topCos: Double,          // max dense cosine     — similarity
    val meanCos: Double,         // mean dense cosine over top_k
    val idfOverlap: Double,      // IDF-weighted token overlap (fused context)

    // ── logistic-regression inputs (SEPARATE dense-only / bm25-only passes) ─
    /** dense_top / dense_avg in knowledge_boundary.py. Numerically identical
     *  to topCos/meanCos above — both come from the same dense-only ranking —
     *  duplicated under the Python names so sig_* fields map 1:1 to KB_MODEL's
     *  feature list without a name-translation step at analysis time. */
    val denseTop: Double,
    val denseAvg: Double,
    /** Token overlap against the DENSE-ONLY context, NOT the fused hybrid
     *  context. This is the one signal that genuinely differs from `overlap`
     *  above — the two contexts can contain different chunks entirely. */
    val overlapDense: Double,
    /** Raw, unbounded BM25 Okapi score. NOT normalised on the Kotlin side —
     *  ported to match knowledge_boundary.py's inference-time behaviour
     *  exactly, which multiplies KB_MODEL coefficients against these raw
     *  values. See the caveat on KbModel below before trusting this mode. */
    val bm25Top: Double,
    val bm25Avg: Double,
    /** True iff the dense-only top-5 pool is non-empty. Mirrors Python's
     *  `bool(dense_results)` gate — the logistic score alone cannot admit a
     *  query with zero retrieved evidence. */
    val hasDenseResults: Boolean,
    val logisticProbability: Double,
)

data class RagResult(
    val results: List<Retrieved>,
    val contextStr: String,
    val boundary: Boundary,
    val retrievalMs: Long,
    val embedMs: Long,
    /** (docIndex, cosine) for the dense pool, best first. Needed by the
     *  calibrated and logistic boundary detectors and recorded by EvalRunner. */
    val denseCosine: List<Pair<Int, Double>>,
    /** All signals across all three boundary modes, so any of them can be
     *  evaluated offline regardless of which one produced this run. */
    val signals: BoundarySignals,
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
    // desktop parity (legacy)
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

/**
 * Port of KB_MODEL from knowledge_boundary.py — a logistic regression fit
 * offline on 746 labelled queries (373 in / 373 out), 5-fold CV + held-out
 * test. precision=0.881 recall=0.787 f1=0.831 auc=0.900 on the reported
 * holdout.
 *
 * RE-FIT PROCEDURE: whenever the Python side re-runs threshold.ipynb and
 * updates KB_MODEL in knowledge_boundary.py, COEF and INTERCEPT below must be
 * copied over in the same commit. Nothing checks that these two copies agree;
 * a stale copy here fails silently as a wrong-but-plausible probability.
 * Consider moving this into manifest.json to remove the duplication.
 *
 * CAVEAT — read before trusting LOGISTIC_REGRESSION mode. The Python
 * docstring states bm25_top/bm25_avg were "StandardScaler-normalized at fit
 * time", but _kb_probability multiplies COEF straight against raw, unbounded
 * BM25 scores with no scaling step. If the coefficients were extracted from a
 * fitted Pipeline(StandardScaler, LogisticRegression) without algebraically
 * folding the scaler's mean/std into the coefficients (coef' = coef / std,
 * intercept' = intercept - sum(coef * mean / std)), every prediction on real,
 * unbounded BM25 scores is wrong. This port is byte-for-byte faithful to the
 * Python inference code as given — it reproduces the same behaviour,
 * correct or not. Verify the fitting notebook actually performs that folding
 * before reporting LOGISTIC_REGRESSION results.
 */
object KbModel {
    // order: dense_top, dense_avg, overlap, bm25_top, bm25_avg
    val COEF = doubleArrayOf(
        16.635537971221343,
        -3.600315847669902,
        6.80005994581478,
        0.17017883789280802,
        -0.2632195057850517,
    )
    const val INTERCEPT = -10.486943570616237
    const val PROBABILITY_THRESHOLD = 0.5

    fun probability(denseTop: Double, denseAvg: Double, overlap: Double,
                    bm25Top: Double, bm25Avg: Double): Double {
        val z = INTERCEPT +
                COEF[0] * denseTop +
                COEF[1] * denseAvg +
                COEF[2] * overlap +
                COEF[3] * bm25Top +
                COEF[4] * bm25Avg
        return 1.0 / (1.0 + exp(-z))
    }
}

private fun round3(x: Double): Double = round(x * 1000.0) / 1000.0

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

    var boundaryMode: BoundaryMode = BoundaryMode.LOGISTIC_REGRESSION

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
    //
    // _boundary_tokens is unchanged by the logistic-regression rewrite:
    // re.findall(r"[\w඀-෿]+", text.lower()) is identical to the version this
    // was ported against before. \w already excludes ZWJ/ZWNJ (category Cf),
    // so findall naturally splits on them without any explicit stripping —
    // boundaryTokens() below reproduces that by converting the joiners to
    // spaces before matching, which is equivalent.

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

    /** Port of _boundary_tokens: re.findall(r"[\w඀-෿]+", text.lower()). */
    private val boundaryRe = Regex("""[\p{L}\p{N}_\u0D80-\u0DFF]+""")

    private fun boundaryTokens(text: String): Set<String> {
        val prepared = text.lowercase()
            .replace('\u200d', ' ')
            .replace('\u200c', ' ')
        return boundaryRe.findAll(prepared).map { it.value }.toSet()
    }

    // ── Dense retrieval ──────────────────────────────────────────────────────
    // Flat fp32 scan. 1767 x 1024 is ~1.8M multiply-adds, single-digit ms.
    // Both sides are unit-norm, so the dot product IS cosine similarity —
    // no conversion, and both the calibrated and logistic detectors gate on it.

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
     * Every document is scored and exactly `pool` returned regardless of
     * score, matching the reference's no-floor behaviour.
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
        // LinkedHashMap, dense inserted FIRST: HybridRetriever builds its
        // result list by iterating a Python dict, which preserves insertion
        // order. Equal-weight RRF produces exact ties whenever a document
        // appears in only one list, so tie-breaking order is load-bearing,
        // not cosmetic — see VALIDATION_RECORD.md D1.
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

    private fun contextOf(items: List<Pair<Int, Double>>): String =
        items.joinToString("") { (doc, _) -> "[${chunks[doc].text}]" }

    // ── Boundary detection ───────────────────────────────────────────────────

    /**
     * All signals used by any of the three detectors, computed once so every
     * mode sees identical inputs — a difference between modes is then the
     * decision rule alone, not a difference in what was measured.
     *
     * @param denseTop5 the dense-ONLY pool, already sorted descending
     *        (denseCosine from the caller), truncated to config.topK.
     * @param bm25Top5  the BM25-ONLY pool, same truncation.
     * @param denseOnlyContext context built from denseTop5 alone — NOT the
     *        fused hybrid context. This is what Python's mode="dense" call
     *        produces and what the logistic overlap feature is measured
     *        against.
     */
    private fun computeSignals(
        query: String,
        fusedResults: List<Retrieved>,
        fusedContextStr: String,
        denseTop5: List<Pair<Int, Double>>,
        bm25Top5: List<Pair<Int, Double>>,
        denseOnlyContext: String,
    ): BoundarySignals {
        val queryTokens = boundaryTokens(query)

        // legacy: overlap against the FUSED hybrid context
        val fusedContextTokens = boundaryTokens(fusedContextStr)
        val overlapFused = if (queryTokens.isEmpty()) 0.0
        else queryTokens.count { it in fusedContextTokens }.toDouble() / queryTokens.size

        // logistic: overlap against the DENSE-ONLY context — genuinely
        // different chunks from the fused list in general
        val denseContextTokens = boundaryTokens(denseOnlyContext)
        val overlapDense = if (queryTokens.isEmpty()) 0.0
        else queryTokens.count { it in denseContextTokens }.toDouble() / queryTokens.size

        // IDF-weighted overlap (this project's CALIBRATED mode), against the
        // fused context, unchanged by the logistic rewrite.
        var num = 0.0
        var den = 0.0
        for (t in queryTokens) {
            val w = idf[t] ?: maxIdf
            den += w
            if (t in fusedContextTokens) num += w
        }
        val idfOverlap = if (den > 0.0) num / den else 0.0

        val denseScores = denseTop5.map { it.second }
        val bm25Scores = bm25Top5.map { it.second }
        val denseTop = denseScores.maxOrNull() ?: 0.0
        val denseAvg = if (denseScores.isEmpty()) 0.0 else denseScores.average()
        val bm25Top = bm25Scores.maxOrNull() ?: 0.0
        val bm25Avg = if (bm25Scores.isEmpty()) 0.0 else bm25Scores.average()

        val probability = KbModel.probability(denseTop, denseAvg, overlapDense, bm25Top, bm25Avg)

        return BoundarySignals(
            topScore = fusedResults.maxOfOrNull { it.score } ?: 0.0,
            avgScore = if (fusedResults.isEmpty()) 0.0 else fusedResults.map { it.score }.average(),
            overlap = overlapFused,
            topCos = denseTop,     // identical value, Python-named alias below
            meanCos = denseAvg,
            idfOverlap = idfOverlap,
            denseTop = denseTop,
            denseAvg = denseAvg,
            overlapDense = overlapDense,
            bm25Top = bm25Top,
            bm25Avg = bm25Avg,
            hasDenseResults = denseTop5.isNotEmpty(),
            logisticProbability = probability,
        )
    }

    /** Faithful port of the legacy knowledge_boundary_detection() threshold
     *  rule. Superseded on the desktop by LOGISTIC_REGRESSION; kept for
     *  historical comparison. OR across three gates. */
    private fun boundaryDesktopParity(
        sig: BoundarySignals,
        results: List<Retrieved>,
        contextStr: String,
        minChunks: Int = 1,
    ): Boundary {
        val topScore = sig.topScore
        val avgScore = sig.avgScore
        val overlap = sig.overlap
        val evidenceChars = contextStr.trim().length

        val topOk = topScore >= config.topScoreThreshold
        val avgOk = avgScore >= config.avgScoreThreshold
        val ovlOk = overlap >= config.overlapThreshold

        val supported = results.isNotEmpty() && results.size >= minChunks &&
                evidenceChars > 0 && (topOk || avgOk || ovlOk)

        val fired = listOfNotNull(
            if (topOk) "top" else null,
            if (avgOk) "avg" else null,
            if (ovlOk) "overlap" else null,
        ).joinToString("+").ifEmpty { "none" }

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

    /** This project's proposed variant: cosine gate + IDF-weighted overlap,
     *  AND-combined. */
    private fun boundaryCalibrated(
        sig: BoundarySignals,
        results: List<Retrieved>,
        contextStr: String,
    ): Boundary {
        val topCos = sig.topCos
        val meanCos = sig.meanCos
        val idfOverlap = sig.idfOverlap

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
            confidence = topCos,
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

    /**
     * Port of the CURRENT knowledge_boundary_detection(): logistic regression
     * over (dense_top, dense_avg, overlap-vs-dense-context, bm25_top,
     * bm25_avg). See KbModel's doc comment for the StandardScaler caveat
     * before trusting this in a report.
     */
    private fun boundaryLogistic(sig: BoundarySignals): Boundary {
        val p = sig.logisticProbability
        val supported = sig.hasDenseResults && p >= KbModel.PROBABILITY_THRESHOLD
        val confidence = round3(if (supported) p else 1.0 - p)

        // Matches Python's f-string exactly: P(...)={p:.3f}, dense_top/avg and
        // overlap at 3 decimals, bm25_top/avg at 2 — porting the precision
        // difference too, since it is what the reference actually prints.
        val reason = "P(within syllabus)=%.3f (dense_top=%.3f, dense_avg=%.3f, overlap=%.3f, bm25_top=%.2f, bm25_avg=%.2f)"
            .format(p, sig.denseTop, sig.denseAvg, sig.overlapDense, sig.bm25Top, sig.bm25Avg)

        return Boundary(
            label = if (supported) "within syllabus" else "out of syllabus",
            isOutOfSyllabus = !supported,
            confidence = confidence,
            topScore = sig.denseTop,
            avgScore = sig.denseAvg,
            overlap = sig.overlapDense,
            reason = reason,
            firedConditions = if (supported) "logistic_accept" else "logistic_reject",
            probability = round(p * 10000.0) / 10000.0,   // round(...,4) in Python
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
        val (results, denseCosine, bm25Pool) = withContext(Dispatchers.Default) {
            val dense = denseSearch(qVec, config.densePool)
            val bm25 = bm25Search(query, config.bm25Pool)
            val cosByDoc = dense.toMap()
            val fused = rrfFuse(dense, bm25)
                .take(config.topK)
                .mapIndexed { i, (doc, score) ->
                    Retrieved(i + 1, score, cosByDoc[doc] ?: 0.0, chunks[doc])
                }
            Triple(fused, dense, bm25)
        }
        val retrievalMs = System.currentTimeMillis() - tRet

        // Fused hybrid context — what the LLM prompt is built from.
        val contextStr = results.joinToString("") { "[${it.chunk.text}]" }

        // Dense-only and BM25-only top-k pools, truncated from the pools
        // already computed above — no extra retrieval pass needed, since both
        // pools are already sorted descending by their own score.
        val denseTop5 = denseCosine.take(config.topK)
        val bm25Top5 = bm25Pool.take(config.topK)
        val denseOnlyContext = contextOf(denseTop5)

        val signals = computeSignals(
            query, results, contextStr, denseTop5, bm25Top5, denseOnlyContext)

        val boundary = when (boundaryMode) {
            BoundaryMode.DESKTOP_PARITY -> boundaryDesktopParity(signals, results, contextStr)
            BoundaryMode.CALIBRATED -> boundaryCalibrated(signals, results, contextStr)
            BoundaryMode.LOGISTIC_REGRESSION -> boundaryLogistic(signals)
        }

        Log.i(TAG, "q=\"${query.take(40)}\" embed=${embedMs}ms retrieve=${retrievalMs}ms " +
                "ids=${results.map { it.chunk.id }} " +
                "boundary=${boundary.label} fired=${boundary.firedConditions} " +
                "prob=${"%.4f".format(signals.logisticProbability)} " +
                "sig[dTop=${"%.4f".format(signals.denseTop)} dAvg=${"%.4f".format(signals.denseAvg)} " +
                "ovlD=${"%.3f".format(signals.overlapDense)} " +
                "bTop=${"%.4f".format(signals.bm25Top)} bAvg=${"%.4f".format(signals.bm25Avg)}]")

        return RagResult(results, contextStr, boundary, retrievalMs, embedMs,
            denseCosine, signals, qVec)
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

    private const val CHARS_PER_TOKEN = 2.5

    fun abstentionAnswer(): String = ABSTAIN

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

        return kept.sortedWith(compareBy({ it.chunkRank }, { it.order }))
            .joinToString(" ") { it.text }
    }

    fun build(query: String, context: String): String = buildString {
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
