package com.interlekt.slmengine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Kotlin side of embed_jni.cpp — a second llama.cpp context configured as an
 * encoder for BGE-M3 dense retrieval.
 *
 * Shares libllama.so with LlamaWrapper; the two contexts are independent.
 * System.loadLibrary is idempotent, so calling it here as well is harmless.
 *
 * Memory: the Q8_0 BGE-M3 weights are ~610 MB resident. If the device is tight
 * on RAM, call free() after each retrieval and load() again before the next —
 * reloading from page cache costs 1–3 s.
 */
class EmbeddingEngine {

    companion object {
        init { System.loadLibrary("llama_jni") }
    }

    var loaded: Boolean = false
        private set

    /**
     * @param nThreads 2 on the Helio G85 — big cores only, matching LlamaWrapper.
     * @param nCtx 512 is ample: only user queries pass through this encoder.
     *             Corpus vectors are precomputed on the desktop and shipped as
     *             fp32, never re-embedded on device.
     */
    suspend fun load(
        path: String,
        nThreads: Int = 2,
        nCtx: Int = 512,
    ): Boolean = withContext(Dispatchers.IO) {
        loaded = nativeLoad(path, nThreads, nCtx)
        loaded
    }

    fun dim(): Int = nativeEmbedDim()

    /** Returns an L2-normalised vector, or null if encoding failed. */
    suspend fun embed(text: String): FloatArray? = withContext(Dispatchers.Default) {
        nativeEmbed(text)
    }

    fun free() {
        nativeFree()
        loaded = false
    }

    private external fun nativeLoad(path: String, nThreads: Int, nCtx: Int): Boolean
    private external fun nativeEmbedDim(): Int
    private external fun nativeEmbed(text: String): FloatArray?
    private external fun nativeFree()
}
