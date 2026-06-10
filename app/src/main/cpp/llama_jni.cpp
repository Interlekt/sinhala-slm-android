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

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_interlekt_slmengine_LlamaWrapper_nativeLoadModel(
        JNIEnv* env, jobject, jstring jpath) {

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
    cparams.n_ctx     = 1024;   // smaller context = less KV cache memory and faster attention
    cparams.n_threads = 2;      // big cores only on Helio G85 (2 big + 6 little)
    cparams.n_batch   = 512;    // prompt-eval batch size

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
    std::vector<llama_token> tokens(2048);
    int n = llama_tokenize(vocab, prompt, strlen(prompt),
                           tokens.data(), tokens.size(), true, true);
    tokens.resize(n);
    env->ReleaseStringUTFChars(jprompt, prompt);

    // Initial decode (prompt eval)
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(g_ctx, batch);

    // Callback references
    jclass    cbClass   = env->GetObjectClass(callback);
    jmethodID onToken   = env->GetMethodID(cbClass, "onToken",   "(Ljava/lang/String;)V");
    jmethodID onMetrics = env->GetMethodID(cbClass, "onMetrics", "(FJF)V");

    // Sampler
    llama_sampler* smpl = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

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
        llama_decode(g_ctx, next);
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