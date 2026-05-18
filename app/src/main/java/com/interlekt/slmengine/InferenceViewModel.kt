package com.interlekt.slmengine

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class InferenceViewModel : ViewModel() {

    data class UiState(
        val output: String   = "",
        val msPerToken: Float = 0f,
        val ramMB: Long      = 0L,
        val cpuPct: Float    = 0f,
        val isLoading: Boolean = false,
        val modelLoaded: Boolean = false,
        val statusMsg: String = "Model not loaded"
    )

    private val _state = MutableStateFlow(UiState())
    val state = _state.asStateFlow()

    private val wrapper = LlamaWrapper()

    fun loadModel(path: String) {
        viewModelScope.launch {
            val possiblePaths = listOf(
                "/data/local/tmp/sinllama.gguf",                                       // NEW: SinLlama
                "/data/local/tmp/qwen.gguf",                                           // fallback: old Qwen baseline
                "/sdcard/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                "/storage/emulated/0/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                path
            )

            _state.update { it.copy(statusMsg = "Loading model...") }

            var loaded = false
            for (p in possiblePaths) {
                val file = java.io.File(p)
                if (file.exists()) {
                    val ok = kotlinx.coroutines.withContext(
                        kotlinx.coroutines.Dispatchers.IO) {
                        wrapper.loadModel(p)
                    }
                    if (ok) {
                        loaded = true
                        _state.update { it.copy(
                            modelLoaded = true,
                            statusMsg = "Model ready ✓ ($p)"
                        )}
                        break
                    }
                }
            }

            if (!loaded) {
                _state.update { it.copy(
                    modelLoaded = false,
                    statusMsg = "Failed — no model found"
                )}
            }
        }
    }

    fun generate(prompt: String) {
        if (!_state.value.modelLoaded) return
        viewModelScope.launch {
            val t0 = System.currentTimeMillis()
            var tokenCount = 0
            _state.update { it.copy(output = "", isLoading = true) }

            wrapper.generate(prompt, maxTokens = 512,
                callback = object : LlamaWrapper.GenerationCallback {
                    override fun onToken(piece: String) {
                        tokenCount++
                        val elapsed = System.currentTimeMillis() - t0
                        _state.update { s -> s.copy(
                            output     = s.output + piece,
                            msPerToken = MetricsCollector.msPerToken(elapsed, tokenCount),
                            ramMB      = MetricsCollector.ramUsedMB(),
                            cpuPct     = MetricsCollector.cpuPercent()
                        )}
                    }
                    override fun onMetrics(msPerToken: Float, ramMB: Long, cpuPct: Float) {}
                }
            )
            _state.update { it.copy(isLoading = false) }
        }
    }

    override fun onCleared() {
        super.onCleared()
        wrapper.freeModel()
    }
}
