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

/**
 * Port of the CURRENT knowledge_boundary_detection() in knowledge_boundary.py:
 * a logistic regression over five signals from two SEPARATE retrieval passes
 * (dense-only top-k, BM25-only top-k) — not the fused hybrid list used to
 * build the LLM-facing context.
 */
data class Boundary(
    val label: String,
    val isOutOfSyllabus: Boolean,
    val confidence: Double,
    /** P(within syllabus) from the logistic model, round(...,4) as in Python. */
    val probability: Double,
    val denseTop: Double,
    val denseAvg: Double,
    val overlap: Double,
    val bm25Top: Double,
    val bm25Avg: Double,
    val reason: String,
) {
    val inBoundary: Boolean get() = !isOutOfSyllabus
}

/**
 * The five raw signals KB_MODEL scores, plus the resulting probability.
 * dense_top/dense_avg/bm25_top/bm25_avg come from two SEPARATE top-k passes
 * (mode="dense" and mode="bm25" in Python), not the fused/RRF list. overlap
 * is token overlap between the query and the DENSE-ONLY context — a
 * different context string than the fused hybrid one used for the LLM
 * prompt.
 */
data class BoundarySignals(
    val denseTop: Double,
    val denseAvg: Double,
    val overlap: Double,
    val bm25Top: Double,
    val bm25Avg: Double,
    /** True iff the dense-only pool is non-empty. Mirrors Python's
     *  `bool(dense_results)` gate — the logistic score alone cannot admit a
     *  query with zero retrieved evidence. */
    val hasDenseResults: Boolean,
    val probability: Double,
)

data class RagResult(
    val results: List<Retrieved>,
    val contextStr: String,
    val boundary: Boundary,
    val retrievalMs: Long,
    val embedMs: Long,
    /** (docIndex, cosine) for the dense pool, best first. */
    val denseCosine: List<Pair<Int, Double>>,
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
    // ── knowledge-boundary logistic regression, read from manifest so a
    // re-fit (threshold.ipynb -> KB_MODEL -> build_bundle.py) only requires
    // rebuilding the bundle, never an APK rebuild. ──────────────────────────
    /** top_k used for the two boundary-only retrieval passes. NOT the same
     *  knob as `topK` above conceptually, even if numerically equal — `topK`
     *  truncates the fused/RRF list used for the LLM context. */
    val boundaryTopK: Int,
    /** Feature order KB_MODEL was fit on. `kbCoef[i]` multiplies whichever
     *  signal is named `kbFeatures[i]` — resolved by name at call time so a
     *  reordering in the manifest can't silently pair the wrong coefficient
     *  with the wrong signal. */
    val kbFeatures: List<String>,
    val kbCoef: DoubleArray,
    val kbIntercept: Double,
    val kbProbabilityThreshold: Double,
) {
    companion object {
        fun fromManifest(json: String): RagConfig {
            val m = JSONObject(json)
            val r = m.getJSONObject("retrieval")
            val bm = m.getJSONObject("bm25")
            val bnd = m.getJSONObject("boundary")

            val featuresArr = bnd.getJSONArray("features")
            val features = List(featuresArr.length()) { featuresArr.getString(it) }
            val coefArr = bnd.getJSONArray("coef")
            val coef = DoubleArray(coefArr.length()) { coefArr.getDouble(it) }
            require(features.size == coef.size) {
                "manifest boundary.features (${features.size}) and boundary.coef " +
                        "(${coef.size}) length mismatch"
            }

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
                boundaryTopK = bnd.getInt("top_k"),
                kbFeatures = features,
                kbCoef = coef,
                kbIntercept = bnd.getDouble("intercept"),
                kbProbabilityThreshold = bnd.getDouble("probability_threshold"),
            )
        }
    }
}

private fun round3(x: Double): Double = round(x * 1000.0) / 1000.0

/**
 * z = intercept + sum(coef[i] * values[features[i]]), sigmoid(z).
 *
 * CAVEAT — read before trusting this. knowledge_boundary.py's docstring
 * states bm25_top/bm25_avg were "StandardScaler-normalized at fit time", but
 * _kb_probability multiplies the coefficients straight against raw,
 * unbounded BM25 scores with no scaling step. This is a byte-for-byte port of
 * that inference code, correct or not — it reproduces the same behaviour as
 * the desktop. If the fitting notebook extracted coefficients from a fitted
 * Pipeline(StandardScaler, LogisticRegression) without algebraically folding
 * the scaler's mean/std into the coefficients (coef' = coef / std,
 * intercept' = intercept - sum(coef * mean / std)), every prediction on real,
 * unbounded BM25 scores is wrong regardless of which side runs it. Verify
 * that folding happened before reporting results from this.
 */
private fun logisticProbability(config: RagConfig, values: Map<String, Double>): Double {
    var z = config.kbIntercept
    for (i in config.kbFeatures.indices) {
        val name = config.kbFeatures[i]
        val v = values[name] ?: error("manifest boundary.features names \"$name\" but no such signal was computed")
        z += config.kbCoef[i] * v
    }
    return 1.0 / (1.0 + exp(-z))
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
                    "${postings.size} terms, avgdl ${"%.1f".format(avgdl)}, " +
                    "boundary features ${config.kbFeatures}")
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
        Character.OTHER_SYMBOL.toInt(),               // So
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
    // Both sides are unit-norm, so the dot product IS cosine similarity.

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
    // Still used to build the fused context for the LLM prompt — unrelated to
    // boundary detection, which runs on the separate dense-only/bm25-only
    // passes below.

    private fun rrfFuse(
        dense: List<Pair<Int, Double>>,
        bm25: List<Pair<Int, Double>>,
    ): List<Pair<Int, Double>> {
        // LinkedHashMap, dense inserted FIRST: HybridRetriever builds its
        // result list by iterating a Python dict, which preserves insertion
        // order. Equal-weight RRF produces exact ties whenever a document
        // appears in only one list, so tie-breaking order is load-bearing,
        // not cosmetic.
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

    // ── Boundary detection ────────────────────────────────────────────────────
    // Port of the CURRENT knowledge_boundary_detection(): logistic regression
    // over (dense_top, dense_avg, overlap-vs-dense-only-context, bm25_top,
    // bm25_avg), each drawn from a SEPARATE top-k pass, not the fused list.

    private fun computeSignals(
        config: RagConfig,
        query: String,
        denseTopK: List<Pair<Int, Double>>,
        bm25TopK: List<Pair<Int, Double>>,
        denseOnlyContext: String,
    ): BoundarySignals {
        val queryTokens = boundaryTokens(query)
        val contextTokens = boundaryTokens(denseOnlyContext)
        val overlap = if (queryTokens.isEmpty()) 0.0
        else queryTokens.count { it in contextTokens }.toDouble() / queryTokens.size

        val denseScores = denseTopK.map { it.second }
        val bm25Scores = bm25TopK.map { it.second }
        val denseTop = denseScores.maxOrNull() ?: 0.0
        val denseAvg = if (denseScores.isEmpty()) 0.0 else denseScores.average()
        val bm25Top = bm25Scores.maxOrNull() ?: 0.0
        val bm25Avg = if (bm25Scores.isEmpty()) 0.0 else bm25Scores.average()

        val probability = logisticProbability(
            config,
            mapOf(
                "dense_top" to denseTop,
                "dense_avg" to denseAvg,
                "overlap" to overlap,
                "bm25_top" to bm25Top,
                "bm25_avg" to bm25Avg,
            ),
        )

        return BoundarySignals(
            denseTop = denseTop,
            denseAvg = denseAvg,
            overlap = overlap,
            bm25Top = bm25Top,
            bm25Avg = bm25Avg,
            hasDenseResults = denseTopK.isNotEmpty(),
            probability = probability,
        )
    }

    private fun computeBoundary(config: RagConfig, sig: BoundarySignals): Boundary {
        val supported = sig.hasDenseResults && sig.probability >= config.kbProbabilityThreshold
        val confidence = round3(if (supported) sig.probability else 1.0 - sig.probability)

        // Matches Python's f-string exactly: P(...)={p:.3f}, dense_top/avg and
        // overlap at 3 decimals, bm25_top/avg at 2 — porting the precision
        // difference too, since it is what the reference actually prints.
        val reason = "P(within syllabus)=%.3f (dense_top=%.3f, dense_avg=%.3f, overlap=%.3f, bm25_top=%.2f, bm25_avg=%.2f)"
            .format(sig.probability, sig.denseTop, sig.denseAvg, sig.overlap, sig.bm25Top, sig.bm25Avg)

        return Boundary(
            label = if (supported) "within syllabus" else "out of syllabus",
            isOutOfSyllabus = !supported,
            confidence = confidence,
            probability = round(sig.probability * 10000.0) / 10000.0,   // round(...,4) in Python
            denseTop = sig.denseTop,
            denseAvg = sig.denseAvg,
            overlap = sig.overlap,
            bm25Top = sig.bm25Top,
            bm25Avg = sig.bm25Avg,
            reason = reason,
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

        // Dense-only and BM25-only top-k pools for the boundary detector,
        // truncated from the pools already computed above — no extra
        // retrieval pass needed, since both pools are already sorted
        // descending by their own score. Uses boundaryTopK, not topK: these
        // are conceptually separate knobs even when numerically equal.
        val denseTopK = denseCosine.take(config.boundaryTopK)
        val bm25TopK = bm25Pool.take(config.boundaryTopK)
        val denseOnlyContext = contextOf(denseTopK)

        val signals = computeSignals(config, query, denseTopK, bm25TopK, denseOnlyContext)
        val boundary = computeBoundary(config, signals)

        Log.i(TAG, "q=\"${query.take(40)}\" embed=${embedMs}ms retrieve=${retrievalMs}ms " +
                "ids=${results.map { it.chunk.id }} " +
                "boundary=${boundary.label} prob=${"%.4f".format(signals.probability)} " +
                "sig[dTop=${"%.4f".format(signals.denseTop)} dAvg=${"%.4f".format(signals.denseAvg)} " +
                "ovl=${"%.3f".format(signals.overlap)} " +
                "bTop=${"%.4f".format(signals.bm25Top)} bAvg=${"%.4f".format(signals.bm25Avg)}]")

        return RagResult(results, contextStr, boundary, retrievalMs, embedMs,
            denseCosine, signals, qVec)
    }
}
