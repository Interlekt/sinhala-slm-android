// embed_jni.cpp
//
// Second llama.cpp context, configured as an encoder for BGE-M3 dense retrieval.
// Add this file to your CMakeLists.txt target sources alongside llama_jni.cpp;
// it links against the same libllama.
//
// API NAMES CHURN. This targets the mid/late-2025 API surface (the same
// generation that gave you llama_memory_*). If the build fails on symbol names,
// check llama.h in your vendored tree:
//   llama_model_load_from_file   (was llama_load_model_from_file)
//   llama_init_from_model        (was llama_new_context_with_model)
//   llama_model_get_vocab        (was implicit in llama_tokenize)
//   llama_model_n_embd           (was llama_n_embd)
//   llama_memory_clear           (was llama_kv_cache_clear)

#include <jni.h>
#include <android/log.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "llama.cpp/include/llama.h"

#define TAG "SLMEmbed"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

    llama_model   * g_model = nullptr;
    llama_context * g_ctx   = nullptr;
    int             g_n_embd = 0;

// Your llama_jni.cpp almost certainly calls llama_backend_init() already.
// Guard so a double call is harmless.
    bool g_backend_ready = false;

    void ensure_backend() {
        if (!g_backend_ready) {
            llama_backend_init();
            g_backend_ready = true;
        }
    }

    std::string jstr(JNIEnv * env, jstring js) {
        if (js == nullptr) return {};
        const char * c = env->GetStringUTFChars(js, nullptr);
        std::string s(c ? c : "");
        env->ReleaseStringUTFChars(js, c);
        return s;
    }

} // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_interlekt_slmengine_EmbeddingEngine_nativeLoad(
        JNIEnv * env, jobject /*thiz*/, jstring jpath, jint n_threads, jint n_ctx) {

    ensure_backend();

    if (g_ctx != nullptr) {
        LOGI("embedder already loaded");
        return JNI_TRUE;
    }

    const std::string path = jstr(env, jpath);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;

    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (g_model == nullptr) {
        LOGE("failed to load embedder: %s", path.c_str());
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings      = true;
    // UNSPECIFIED makes llama.cpp honour the pooling type baked into the GGUF
    // by convert_hf_to_gguf.py. For BGE-M3 that should resolve to CLS. Verify
    // in logcat below -- if it resolves to MEAN, your embeddings will NOT match
    // the desktop pipeline and you must force LLAMA_POOLING_TYPE_CLS here.
    cparams.pooling_type    = LLAMA_POOLING_TYPE_UNSPECIFIED;
    cparams.n_ctx           = (uint32_t) n_ctx;   // 512 is plenty for queries
    cparams.n_batch         = (uint32_t) n_ctx;
    cparams.n_ubatch        = (uint32_t) n_ctx;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (g_ctx == nullptr) {
        LOGE("failed to create embedder context");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    g_n_embd = llama_model_n_embd(g_model);
    LOGI("embedder ready: n_embd=%d pooling=%d n_ctx=%d threads=%d",
         g_n_embd, (int) llama_pooling_type(g_ctx), n_ctx, n_threads);

    if (llama_pooling_type(g_ctx) != LLAMA_POOLING_TYPE_CLS) {
        LOGE("WARNING pooling type is not CLS (%d). BGE-M3 dense uses CLS. "
             "On-device embeddings will diverge from desktop.",
             (int) llama_pooling_type(g_ctx));
    }

    return JNI_TRUE;
}

JNIEXPORT jint JNICALL
Java_com_interlekt_slmengine_EmbeddingEngine_nativeEmbedDim(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return g_n_embd;
}

JNIEXPORT jfloatArray JNICALL
Java_com_interlekt_slmengine_EmbeddingEngine_nativeEmbed(
        JNIEnv * env, jobject /*thiz*/, jstring jtext) {

    if (g_ctx == nullptr) {
        LOGE("nativeEmbed called before nativeLoad");
        return nullptr;
    }

    const std::string text = jstr(env, jtext);
    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    const int n_ctx = (int) llama_n_ctx(g_ctx);
    std::vector<llama_token> tokens(n_ctx);

    // add_special = true  -> prepends <s> (the CLS position BGE-M3 pools from)
    // parse_special = false
    int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                               tokens.data(), (int32_t) tokens.size(),
            /*add_special*/ true, /*parse_special*/ false);
    if (n_tok < 0) {
        // negative means "needed this many"; truncate to context
        n_tok = n_ctx;
    }
    if (n_tok <= 0) {
        LOGE("tokenised to zero tokens");
        return nullptr;
    }
    if (n_tok > n_ctx) n_tok = n_ctx;

    // Fresh state per call: pooled embeddings are per-sequence and stale KV
    // from a previous query would corrupt the result.
    llama_memory_clear(llama_get_memory(g_ctx), true);

    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    batch.n_tokens = n_tok;
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = 1;   // pooled output requires outputs enabled
    }

    // Encoder-only models use llama_encode. Older trees only have llama_decode
    // and route BERT through it, so fall back rather than fail.
    int rc = llama_encode(g_ctx, batch);
    if (rc != 0) {
        LOGI("llama_encode rc=%d, retrying with llama_decode", rc);
        rc = llama_decode(g_ctx, batch);
    }
    if (rc != 0) {
        LOGE("encode failed rc=%d", rc);
        llama_batch_free(batch);
        return nullptr;
    }

    const float * emb = llama_get_embeddings_seq(g_ctx, 0);
    if (emb == nullptr) {
        LOGE("llama_get_embeddings_seq returned null (pooling disabled?)");
        llama_batch_free(batch);
        return nullptr;
    }

    // L2 normalise. llama.cpp does NOT do this in-library; the CLI example
    // normalises in userland, and BGE-M3 dense vectors are unit-length.
    std::vector<float> out(g_n_embd);
    double ss = 0.0;
    for (int i = 0; i < g_n_embd; ++i) ss += (double) emb[i] * (double) emb[i];
    const float inv = (ss > 0.0) ? (float) (1.0 / std::sqrt(ss)) : 0.0f;
    for (int i = 0; i < g_n_embd; ++i) out[i] = emb[i] * inv;

    llama_batch_free(batch);

    jfloatArray jout = env->NewFloatArray(g_n_embd);
    env->SetFloatArrayRegion(jout, 0, g_n_embd, out.data());
    return jout;
}

JNIEXPORT void JNICALL
Java_com_interlekt_slmengine_EmbeddingEngine_nativeFree(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_n_embd = 0;
    LOGI("embedder freed");
}

} // extern "C"
