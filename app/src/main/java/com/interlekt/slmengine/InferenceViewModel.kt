package com.interlekt.slmengine

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class InferenceViewModel : ViewModel() {

    data class UiState(
        val output: String = "",
        // TTFT and decode speed are now separate. The old combined metric was
        // (prefill + decode) / tokens, which RAG's long prompts would dominate.
        val ttftMs: Long = 0L,
        val msPerToken: Float = 0f,
        val ramMB: Long = 0L,
        val cpuPct: Float = 0f,
        val isLoading: Boolean = false,
        val modelLoaded: Boolean = false,
        val statusMsg: String = "Model not loaded",
    )

    data class RagState(
        val ragReady: Boolean = false,
        val ragStatus: String = "RAG not loaded",
        val useRag: Boolean = true,
        val retrieveOnly: Boolean = false,
        val mode: BoundaryMode = BoundaryMode.DESKTOP_PARITY,
        val abstained: Boolean = false,
        val boundary: Boundary? = null,
        val sources: List<Retrieved> = emptyList(),
        val embedMs: Long = 0L,
        val retrieveMs: Long = 0L,
        val promptChars: Int = 0,
    )

    private val _state = MutableStateFlow(UiState())
    val state = _state.asStateFlow()

    private val _ragState = MutableStateFlow(RagState())
    val ragState = _ragState.asStateFlow()

    private val wrapper = LlamaWrapper()
    private val embedder = EmbeddingEngine()
    private val rag = RagEngine(embedder)

    // ── Loading ──────────────────────────────────────────────────────────────

    fun loadModel(path: String) {
        viewModelScope.launch {
            val possiblePaths = listOf(
                "/data/local/tmp/sinllama.gguf",
                "/sdcard/Download/sinllama.gguf",
                "/data/local/tmp/qwen.gguf",
                "/sdcard/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                path,
            )

            _state.update { it.copy(statusMsg = "Loading model...") }

            for (p in possiblePaths) {
                if (!File(p).exists()) continue
                val ok = withContext(Dispatchers.IO) { wrapper.loadModel(p) }
                if (ok) {
                    _state.update {
                        it.copy(modelLoaded = true, statusMsg = "Model ready ✓ (${File(p).name})")
                    }
                    return@launch
                }
            }
            _state.update { it.copy(modelLoaded = false, statusMsg = "Failed — no model found") }
        }
    }

    /**
     * Load the embedder and the corpus bundle. Call after loadModel().
     *
     * Peak RSS with both resident is ~610 MB (embedder) + ~900 MB (generator)
     * + KV + 8 MB corpus. Fine on 8 GB. On a 4 GB device, call
     * embedder.free() after each retrieval and reload before the next; that
     * costs 1-3 s from page cache and is worth reporting as a separate row.
     */
    fun loadRag(
        embedderPath: String = "/data/local/tmp/bge-m3.gguf",
        bundleDir: String = "/sdcard/Download/rag_bundle",
    ) = viewModelScope.launch {
        _ragState.update { it.copy(ragStatus = "Loading embedder...") }

        if (!File(embedderPath).exists()) {
            _ragState.update { it.copy(ragStatus = "Embedder not found: $embedderPath") }
            return@launch
        }

        val embOk = withContext(Dispatchers.IO) { embedder.load(embedderPath, nThreads = 4) }
        if (!embOk) {
            _ragState.update { it.copy(ragStatus = "Embedder failed to load") }
            return@launch
        }

        _ragState.update { it.copy(ragStatus = "Loading corpus...") }
        val bundleOk = rag.load(bundleDir)
        if (!bundleOk) {
            _ragState.update { it.copy(ragStatus = "Bundle failed: $bundleDir") }
            return@launch
        }

        rag.boundaryMode = _ragState.value.mode
        _ragState.update {
            it.copy(
                ragReady = true,
                ragStatus = "RAG ready ✓ (${rag.config.nChunks} chunks, dim ${rag.config.embeddingDim})",
            )
        }
    }

    /** For the asset-bundled thesis build. See STEP_BY_STEP.md step 4. */
    fun loadRagFromAssets(context: Context, embedderPath: String) = viewModelScope.launch {
        val dir = withContext(Dispatchers.IO) { copyAssetBundle(context) }
        loadRag(embedderPath, dir).join()
    }

    private fun copyAssetBundle(context: Context): String {
        val dest = File(context.filesDir, "rag_bundle")
        if (!File(dest, "manifest.json").exists()) {
            dest.mkdirs()
            context.assets.list("rag_bundle")?.forEach { name ->
                context.assets.open("rag_bundle/$name").use { input ->
                    File(dest, name).outputStream().use { input.copyTo(it) }
                }
            }
        }
        return dest.absolutePath
    }

    // ── Toggles ──────────────────────────────────────────────────────────────

    fun setUseRag(on: Boolean) = _ragState.update { it.copy(useRag = on) }
    fun setRetrieveOnly(on: Boolean) = _ragState.update { it.copy(retrieveOnly = on) }

    fun setBoundaryMode(mode: BoundaryMode) {
        rag.boundaryMode = mode
        _ragState.update { it.copy(mode = mode) }
    }

    // ── Ask ──────────────────────────────────────────────────────────────────

    fun ask(query: String) {
        if (query.isBlank()) return
        val rs = _ragState.value
        if (!rs.useRag || !rs.ragReady) {
            generate(query)
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(output = "", isLoading = true) }
            _ragState.update {
                it.copy(abstained = false, sources = emptyList(), boundary = null)
            }

            val r = rag.retrieve(query)
            if (r == null) {
                _state.update { it.copy(isLoading = false, output = "Retrieval failed.") }
                return@launch
            }

            _ragState.update {
                it.copy(
                    sources = r.results,
                    boundary = r.boundary,
                    embedMs = r.embedMs,
                    retrieveMs = r.retrievalMs,
                )
            }

            // ── The mitigation ───────────────────────────────────────────────
            // When the query falls outside the syllabus boundary we return the
            // refusal directly. The SLM is never invoked, so it has no
            // opportunity to confabulate. This is a guarantee rather than a
            // tendency, and it is why this is stronger than instructing the
            // model to refuse inside the prompt.
            if (r.boundary.isOutOfSyllabus) {
                _ragState.update { it.copy(abstained = true) }
                _state.update {
                    it.copy(
                        isLoading = false,
                        output = PromptBuilder.abstentionAnswer(),
                        ttftMs = r.embedMs + r.retrievalMs,
                        msPerToken = 0f,
                    )
                }
                return@launch
            }

            if (_ragState.value.retrieveOnly) {
                _state.update {
                    it.copy(isLoading = false, output = buildRetrievalReport(r))
                }
                return@launch
            }

            val context = PromptBuilder.compress(query, r.results, tokenBudget = 512)
            val prompt = PromptBuilder.build(query, context)
            _ragState.update { it.copy(promptChars = prompt.length) }

            // llama_jni.cpp applies no chat template and tokenizes with
            // add_special=true, so PromptBuilder's output (which omits
            // <|begin_of_text|>) goes straight through.
            generate(prompt, clearFirst = false)
        }
    }

    private fun buildRetrievalReport(r: RagResult): String = buildString {
        appendLine("BOUNDARY: ${r.boundary.label}")
        appendLine("  ${r.boundary.reason}")
        appendLine("  top=${"%.6f".format(r.boundary.topScore)}  " +
                "avg=${"%.6f".format(r.boundary.avgScore)}  " +
                "overlap=${"%.3f".format(r.boundary.overlap)}")
        appendLine()
        appendLine("embed ${r.embedMs}ms   retrieve ${r.retrievalMs}ms")
        appendLine()
        r.results.forEach { res ->
            appendLine("#${res.rank}  score=${"%.6f".format(res.score)}  id=${res.chunk.id}")
            appendLine("  ${res.chunk.grade} / ${res.chunk.chapter} / p.${res.chunk.page}")
            appendLine("  ${res.chunk.text.take(140).replace("\n", " ")}...")
            appendLine()
        }
    }

    // ── Generation ───────────────────────────────────────────────────────────

    fun generate(prompt: String, clearFirst: Boolean = true) {
        if (!_state.value.modelLoaded) return
        viewModelScope.launch {
            val t0 = System.currentTimeMillis()
            var firstTokenAt = 0L
            var tokenCount = 0

            if (clearFirst) _state.update { it.copy(output = "", isLoading = true) }
            else _state.update { it.copy(isLoading = true) }

            // 128, not 512: at 194 ms/token a 512-token answer would take 99 s.
            wrapper.generate(prompt, maxTokens = 128,
                callback = object : LlamaWrapper.GenerationCallback {
                    override fun onToken(piece: String) {
                        val now = System.currentTimeMillis()
                        if (tokenCount == 0) firstTokenAt = now
                        tokenCount++

                        // Decode speed measured from the first token onward, so
                        // prefill no longer contaminates it.
                        val decodeMs =
                            if (tokenCount > 1) (now - firstTokenAt).toFloat() / (tokenCount - 1)
                            else 0f

                        _state.update { s ->
                            s.copy(
                                output = s.output + piece,
                                ttftMs = firstTokenAt - t0,
                                msPerToken = decodeMs,
                                ramMB = MetricsCollector.ramUsedMB(),
                                cpuPct = MetricsCollector.cpuPercent(),
                            )
                        }
                    }

                    override fun onMetrics(msPerToken: Float, ramMB: Long, cpuPct: Float) {}
                    override fun onPrompt(nTokens: Int) {}
                })

            _state.update { it.copy(isLoading = false) }
        }
    }

    override fun onCleared() {
        super.onCleared()
        wrapper.freeModel()
        if (embedder.loaded) embedder.free()
    }

    private val evalRunner by lazy { EvalRunner(wrapper, rag) }
    private val _evalStatus = MutableStateFlow("")
    val evalStatus = _evalStatus.asStateFlow()

    fun runBatch(quant: String) = viewModelScope.launch {
        val cond = EvalRunner.Condition(
            quant = quant,
            ragEnabled = true,
            boundaryGateEnabled = true,
            compressionEnabled = true,
        )
        val path = evalRunner.run(cond) { p ->
            _evalStatus.value = "${p.done}/${p.total}  ${p.currentId}"
        }
        _evalStatus.value = path?.let { "done -> $it" } ?: "failed"
    }
}
