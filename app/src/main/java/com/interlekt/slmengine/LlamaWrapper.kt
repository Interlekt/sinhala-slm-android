package com.interlekt.slmengine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class LlamaWrapper {

    companion object {
        init { System.loadLibrary("llama_jni") }
    }

    interface GenerationCallback {
        fun onToken(piece: String)
        fun onMetrics(msPerToken: Float, ramMB: Long, cpuPct: Float)
        fun onPrompt(nTokens: Int)
    }

    fun loadModel(path: String): Boolean = nativeLoadModel(path)
    fun freeModel() = nativeFreeModel()

    suspend fun generate(
        prompt: String,
        maxTokens: Int = 512,
        callback: GenerationCallback
    ) = withContext(Dispatchers.IO) {
        nativeGenerate(prompt, maxTokens, callback)
    }

    private external fun nativeLoadModel(path: String): Boolean
    private external fun nativeGenerate(
        prompt: String, maxTokens: Int, callback: GenerationCallback)
    private external fun nativeFreeModel()

    fun systemInfo(): String = nativeSystemInfo()
    private external fun nativeSystemInfo(): String
}
