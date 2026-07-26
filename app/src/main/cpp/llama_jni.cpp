#include <jni.h>
#include "llama.cpp/include/llama.h"
#include <android/log.h>
#include <string>
#include <vector>
#include <chrono>

#define LOG_TAG "SLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*   g_model = nullptr;
static llama_context* g_ctx   = nullptr;

static void llama_log_android(ggml_log_level level, const char * text, void *) {
    int prio;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default:                   prio = ANDROID_LOG_INFO;  break;
    }
    __android_log_print(prio, "llamacpp", "%s", text);
}



extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_interlekt_slmengine_LlamaWrapper_nativeLoadModel(
        JNIEnv* env, jobject, jstring jpath) {

    llama_log_set(llama_log_android, nullptr);

    const char* path = env->GetStringUTFChars(jpath, nullptr);
    LOGI("Loading model from: %s", path);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_model) { LOGE("Failed to load model"); return JNI_FALSE; }
    LOGI("Model loaded successfully");

    // Create the context ONCE here, reuse across generations
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 2048;   // smaller context = less KV cache memory and faster attention
    cparams.n_threads = 2;      // big cores only on Helio G85 (2 big + 6 little)
    cparams.n_batch   = 512;
    cparams.no_perf = false; // enable llama_perf_context_print timings

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }
    LOGI("Context created successfully");

    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_interlekt_slmengine_LlamaWrapper_nativeGenerate(
        JNIEnv* env, jobject,
        jstring jprompt, jint maxTokens,
        jobject callback) {

    if (!g_model || !g_ctx) { LOGE("No model/context loaded"); return; }

    // Clear KV cache so prompts don't accumulate from previous generations
    llama_memory_clear(llama_get_memory(g_ctx), true);

    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    // Tokenize using vocab
    const int n_ctx = (int) llama_n_ctx(g_ctx);
    std::vector<llama_token> tokens(n_ctx);
    int n = llama_tokenize(vocab, prompt, strlen(prompt),
                           tokens.data(), tokens.size(), true, true);
    env->ReleaseStringUTFChars(jprompt, prompt);

    // llama_tokenize returns NEGATIVE when the buffer is too small. resize()
    // takes size_t, so passing a negative value allocates ~18 exabytes.
    if (n < 0) {
    LOGE("prompt too long: needs %d tokens, n_ctx is %d", -n, n_ctx);
    return;
    }
    if (n == 0) { LOGE("prompt tokenised to nothing"); return; }
    tokens.resize(n);
    LOGI("prompt: %d tokens", n);

    // Prefill in n_batch-sized chunks. llama_decode REJECTS any batch larger
    // than n_batch (512), so a single call with a ~600-token RAG prompt fails
    // silently and the model generates from an empty context.
    const int n_batch = (int) llama_n_batch(g_ctx);
    for (int i = 0; i < n; i += n_batch) {
    const int len = (n - i < n_batch) ? (n - i) : n_batch;
    llama_batch chunk = llama_batch_get_one(tokens.data() + i, len);
    const int rc = llama_decode(g_ctx, chunk);
    if (rc != 0) {
    LOGE("prefill failed at token %d (rc=%d)", i, rc);
    return;
    }
}


    // Callback references
    jclass    cbClass   = env->GetObjectClass(callback);
    jmethodID onToken   = env->GetMethodID(cbClass, "onToken",   "(Ljava/lang/String;)V");
    jmethodID onMetrics = env->GetMethodID(cbClass, "onMetrics", "(FJF)V");
    jmethodID onPrompt  = env->GetMethodID(cbClass, "onPrompt",  "(I)V");

    env->CallVoidMethod(callback, onPrompt, (jint) n);
    // Sampler
    llama_sampler* smpl = llama_sampler_chain_init(
      llama_sampler_chain_default_params());
      llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    auto t_start    = std::chrono::high_resolution_clock::now();
    int  token_count = 0;

    for (int i = 0; i < maxTokens; i++) {
        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int  len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (len < 0) break;
        buf[len] = '\0';
        token_count++;

        jstring piece = env->NewStringUTF(buf);
        env->CallVoidMethod(callback, onToken, piece);
        env->DeleteLocalRef(piece);

        if (token_count % 5 == 0) {
            auto  now     = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float, std::milli>(
                    now - t_start).count();
            float mpt = elapsed / token_count;
            env->CallVoidMethod(callback, onMetrics,
                                (jfloat)mpt, (jlong)0, (jfloat)0.0f);
        }

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(g_ctx, next) != 0) {
            LOGE("decode failed at token %d", token_count);
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_perf_context_print(g_ctx);
    // Note: context is NOT freed here — it lives until nativeFreeModel
}

JNIEXPORT void JNICALL
Java_com_interlekt_slmengine_LlamaWrapper_nativeFreeModel(JNIEnv*, jobject) {
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    LOGI("Model and context freed");
}

} // extern "C"