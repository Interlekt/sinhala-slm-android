package com.interlekt.slmengine

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class InferenceViewModel : ViewModel() {

    // ── UI state ─────────────────────────────────────────────────────────────

    data class UiState(
        val output: String = "",
        val ttftMs: Long = 0L,
        val msPerToken: Float = 0f,
        val promptTokens: Int = 0,
        val ramMB: Long = 0L,
        val isLoading: Boolean = false,
        val modelLoaded: Boolean = false,
        val statusMsg: String = "No model loaded",
    )

    data class ModelInfo(
        val path: String,
        val name: String,
        val quant: String,
        val sizeMb: Long,
    )

    data class RagState(
        val ragReady: Boolean = false,
        val ragStatus: String = "RAG not loaded",
        val abstained: Boolean = false,
        val boundary: Boundary? = null,
        val sources: List<Retrieved> = emptyList(),
        val embedMs: Long = 0L,
        val retrieveMs: Long = 0L,
        val promptChars: Int = 0,
    )

    /**
     * One cell of the experimental design. Every field here is an axis you can
     * vary between runs without rebuilding.
     *
     * There is no boundary-mode field here at all: RagEngine now implements
     * exactly one gate, the logistic regression read from manifest.json's
     * "boundary" block, matching the current desktop implementation in
     * knowledge_boundary.py. The legacy DESKTOP_PARITY/CALIBRATED rules and
     * their sig_* fields no longer exist anywhere in the pipeline, so there is
     * nothing left to sweep offline for them either.
     */
    data class Experiment(
        val ragEnabled: Boolean = true,
        val gateEnabled: Boolean = true,
        val compressEnabled: Boolean = true,
        val tokenBudget: Int = 384,
        val maxTokens: Int = 128,
        val cooldownMs: Long = 20_000,
        val retrieveOnly: Boolean = false,
    )

    data class EvalState(
        val running: Boolean = false,
        val done: Int = 0,
        val total: Int = 0,
        val currentId: String = "",
        val lastOutput: String? = null,
        val message: String = "",
    )

    private val _state = MutableStateFlow(UiState())
    val state = _state.asStateFlow()

    private val _rag = MutableStateFlow(RagState())
    val ragState = _rag.asStateFlow()

    private val _exp = MutableStateFlow(Experiment())
    val experiment = _exp.asStateFlow()

    private val _eval = MutableStateFlow(EvalState())
    val evalState = _eval.asStateFlow()

    private val _models = MutableStateFlow<List<ModelInfo>>(emptyList())
    val models = _models.asStateFlow()

    private val _currentModel = MutableStateFlow<ModelInfo?>(null)
    val currentModel = _currentModel.asStateFlow()

    private val wrapper = LlamaWrapper()
    private val embedder = EmbeddingEngine()
    private val rag = RagEngine(embedder)
    private val evalRunner by lazy { EvalRunner(wrapper, rag) }

    private var batchJob: Job? = null
    private var modelLoadMs: Long = 0L

    // ── Model discovery ──────────────────────────────────────────────────────

    companion object {
        private const val EMBEDDER_NAME = "bge-m3"

        private val SCAN_DIRS = listOf(
            "/sdcard/Download",
            "/storage/emulated/0/Download",
            "/data/local/tmp"
        )

        private val PROBE_PATHS = listOf(
            "/data/local/tmp/sinllama.gguf",
            "/data/local/tmp/sinllama-Q3_K_M.gguf",
            "/data/local/tmp/sinllama-1b-qa-v6-merged-Q4_K_M.gguf",
            "/data/local/tmp/sinllama-1b-qa-v6-merged-Q4_K_M-imat.gguf",
            "/data/local/tmp/sinllama-Q8_0.gguf",
            "/data/local/tmp/sinllama-f16.gguf",
            "/data/local/tmp/sinllama-3b-qa-v6-merged-Q4_K_M.gguf"
        )

        private val QUANT_RE =
            Regex("""(Q\d+_[A-Z0-9_]+|Q\d+_\d+|F16|BF16|F32)""", RegexOption.IGNORE_CASE)

        fun quantOf(path: String): String {
            val name = File(path).name
            return QUANT_RE.find(name)?.value?.uppercase()
                ?: name.removeSuffix(".gguf")
        }
    }

    fun discoverModels() = viewModelScope.launch(Dispatchers.IO) {
        val found = LinkedHashMap<String, ModelInfo>()

        fun consider(f: File) {
            if (!f.isFile) return
            if (!f.name.endsWith(".gguf", ignoreCase = true)) return
            if (f.name.contains(EMBEDDER_NAME, ignoreCase = true)) return
            found[f.absolutePath] = ModelInfo(
                path = f.absolutePath,
                name = f.name,
                quant = quantOf(f.absolutePath),
                sizeMb = f.length() / (1024 * 1024),
            )
        }

        SCAN_DIRS.forEach { d -> File(d).listFiles()?.forEach { consider(it) } }
        PROBE_PATHS.forEach { p -> consider(File(p)) }

        _models.value = found.values.sortedBy { it.name }
        _eval.update { it.copy(message = "found ${found.size} generator models") }
    }

    // ── Loading ──────────────────────────────────────────────────────────────

    fun loadSlm(info: ModelInfo) = viewModelScope.launch {
        if (_eval.value.running) return@launch
        _state.update { it.copy(statusMsg = "Loading ${info.name} ...", modelLoaded = false) }

        val ms = withContext(Dispatchers.IO) {
            wrapper.freeModel()
            val t0 = System.currentTimeMillis()
            val ok = wrapper.loadModel(info.path)
            val dt = System.currentTimeMillis() - t0
            if (ok) dt else -1L
        }

        if (ms < 0) {
            _state.update { it.copy(modelLoaded = false, statusMsg = "FAILED: ${info.name}") }
            return@launch
        }
        modelLoadMs = ms
        _currentModel.value = info
        _state.update {
            it.copy(modelLoaded = true,
                statusMsg = "${info.quant} ready (${info.sizeMb} MB, loaded in ${ms}ms)")
        }
    }

    fun loadRag(
        embedderPath: String = "/data/local/tmp/bge-m3.gguf",
        bundleDir: String = "/sdcard/Download/rag_bundle",
    ) = viewModelScope.launch {
        _rag.update { it.copy(ragStatus = "Loading embedder ...") }

        if (!File(embedderPath).exists()) {
            _rag.update { it.copy(ragStatus = "Embedder missing: $embedderPath") }
            return@launch
        }
        if (!withContext(Dispatchers.IO) { embedder.load(embedderPath, nThreads = 2) }) {
            _rag.update { it.copy(ragStatus = "Embedder failed to load") }
            return@launch
        }

        _rag.update { it.copy(ragStatus = "Loading corpus ...") }
        if (!rag.load(bundleDir)) {
            _rag.update { it.copy(ragStatus = "Bundle failed: $bundleDir") }
            return@launch
        }

        // No mode to set here any more: RagEngine.retrieve() always runs the
        // logistic-regression gate it read out of manifest.json's "boundary"
        // block at load() time.
        _rag.update {
            it.copy(ragReady = true,
                ragStatus = "RAG ready (${rag.config.nChunks} chunks, dim ${rag.config.embeddingDim})")
        }
    }

    // ── Experiment controls ──────────────────────────────────────────────────

    fun setRag(on: Boolean) = _exp.update { it.copy(ragEnabled = on) }
    fun setGate(on: Boolean) = _exp.update { it.copy(gateEnabled = on) }
    fun setCompress(on: Boolean) = _exp.update { it.copy(compressEnabled = on) }
    fun setRetrieveOnly(on: Boolean) = _exp.update { it.copy(retrieveOnly = on) }
    fun setTokenBudget(n: Int) = _exp.update { it.copy(tokenBudget = n) }
    fun setCooldown(ms: Long) = _exp.update { it.copy(cooldownMs = ms) }

    private fun condition(): EvalRunner.Condition {
        val e = _exp.value
        return EvalRunner.Condition(
            quant = _currentModel.value?.quant ?: "unknown",
            ragEnabled = e.ragEnabled,
            boundaryGateEnabled = e.gateEnabled,
            compressionEnabled = e.compressEnabled,
            tokenBudget = e.tokenBudget,
            maxTokens = e.maxTokens,
        )
    }

    /** The filename this configuration will write to. Shown in the UI so you
     *  can confirm the cell before committing 45 minutes of device time. */
    fun runTag(): String = condition().tag

    // ── Batch ────────────────────────────────────────────────────────────────

    fun runBatch() {
        if (_eval.value.running) return
        if (!_state.value.modelLoaded) {
            _eval.update { it.copy(message = "load a generator first") }
            return
        }
        if (_exp.value.ragEnabled && !_rag.value.ragReady) {
            _eval.update { it.copy(message = "RAG not ready") }
            return
        }

        val cond = condition()
        batchJob = viewModelScope.launch {
            _eval.value = EvalState(running = true, message = "running ${cond.tag}")
            val path = evalRunner.run(
                condition = cond,
                cooldownMs = _exp.value.cooldownMs,
            ) { p ->
                _eval.update {
                    it.copy(done = p.done, total = p.total, currentId = p.currentId)
                }
            }
            _eval.update {
                it.copy(running = false, lastOutput = path,
                    message = path?.let { p -> "done -> ${File(p).name}" } ?: "failed")
            }
        }
    }

    fun cancelBatch() {
        batchJob?.cancel()
        batchJob = null
        _eval.update { it.copy(running = false, message = "cancelled") }
    }

    // ── Single query ─────────────────────────────────────────────────────────

    fun ask(query: String) {
        if (query.isBlank() || _eval.value.running) return
        val e = _exp.value

        if (!e.ragEnabled || !_rag.value.ragReady) {
            generate(query)
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(output = "", isLoading = true) }
            _rag.update { it.copy(abstained = false, sources = emptyList(), boundary = null) }

            val r = rag.retrieve(query)
            if (r == null) {
                _state.update { it.copy(isLoading = false, output = "Retrieval failed.") }
                return@launch
            }

            _rag.update {
                it.copy(sources = r.results, boundary = r.boundary,
                    embedMs = r.embedMs, retrieveMs = r.retrievalMs)
            }

            // The mitigation: when the query is outside the syllabus boundary
            // we answer directly and the SLM is never invoked, so it has no
            // chance to confabulate. Also why abstentions are ~100x cheaper.
            if (e.gateEnabled && r.boundary.isOutOfSyllabus) {
                _rag.update { it.copy(abstained = true) }
                _state.update {
                    it.copy(isLoading = false, output = PromptBuilder.abstentionAnswer(),
                        ttftMs = r.embedMs + r.retrievalMs, msPerToken = 0f, promptTokens = 0)
                }
                return@launch
            }

            if (e.retrieveOnly) {
                _state.update { it.copy(isLoading = false, output = report(r)) }
                return@launch
            }

            val ctx = if (e.compressEnabled)
                PromptBuilder.compress(query, r.results, e.tokenBudget)
            else r.results.joinToString(" ") { it.chunk.text }

            val prompt = PromptBuilder.build(query, ctx)
            _rag.update { it.copy(promptChars = prompt.length) }
            generate(prompt, clearFirst = false)
        }
    }

    private fun report(r: RagResult): String = buildString {
        appendLine("BOUNDARY: ${r.boundary.label}")
        appendLine("  ${r.boundary.reason}")
        appendLine("  P(within syllabus)=${"%.4f".format(r.boundary.probability)}")
        appendLine()
        appendLine("embed ${r.embedMs}ms   retrieve ${r.retrievalMs}ms")
        appendLine()
        r.results.forEach { res ->
            appendLine("#${res.rank}  fused=${"%.6f".format(res.score)}  " +
                    "cos=${"%.4f".format(res.cosine)}")
            appendLine("  id=${res.chunk.id.take(12)}")
            appendLine("  ${res.chunk.grade} / ${res.chunk.chapter} / p.${res.chunk.page}")
            appendLine("  ${res.chunk.text.take(120).replace("\n", " ")}...")
            appendLine()
        }
    }

    fun generate(prompt: String, clearFirst: Boolean = true) {
        if (!_state.value.modelLoaded) return
        viewModelScope.launch {
            val t0 = System.currentTimeMillis()
            var firstAt = 0L
            var count = 0

            _state.update {
                if (clearFirst) it.copy(output = "", isLoading = true) else it.copy(isLoading = true)
            }

            wrapper.generate(prompt, maxTokens = _exp.value.maxTokens,
                callback = object : LlamaWrapper.GenerationCallback {
                    override fun onToken(piece: String) {
                        val now = System.currentTimeMillis()
                        if (count == 0) firstAt = now
                        count++
                        val decode = if (count > 1)
                            (now - firstAt).toFloat() / (count - 1) else 0f
                        _state.update { s ->
                            s.copy(output = s.output + piece,
                                ttftMs = firstAt - t0,
                                msPerToken = decode,
                                ramMB = HardwareMonitor.pssMb())
                        }
                    }
                    override fun onMetrics(msPerToken: Float, ramMB: Long, cpuPct: Float) {}
                    override fun onPrompt(nTokens: Int) {
                        _state.update { it.copy(promptTokens = nTokens) }
                    }
                })

            _state.update { it.copy(isLoading = false) }
        }
    }

    override fun onCleared() {
        super.onCleared()
        batchJob?.cancel()
        wrapper.freeModel()
        if (embedder.loaded) embedder.free()
    }
}
