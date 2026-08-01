#include <jni.h>
#include "llama.cpp/include/llama.h"
#include <android/log.h>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>

#define LOG_TAG "SLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Sampler configuration, matching qa-evaluation-updated0729.ipynb ──────────
//
// The notebook generates with:
//     do_sample=False              -> greedy
//     repetition_penalty=1.05
//     max_new_tokens=48            (passed in from Kotlin)
//
// The penalty is not optional. Greedy decoding with no penalty degenerates on
// this checkpoint: earlier device runs produced "පොත්තන්පොත්තන්පොත්තන්…" and
// "user Brennan Dissertation" repeated to the token cap. 1.05 is the value the
// reference uses.
static const float  PENALTY_REPEAT  = 1.05f;
// Window of previously GENERATED tokens the penalty considers. 64 covers a
// full 48-token answer, so in practice this is "everything generated so far".
//
// This is where llama.cpp and HuggingFace differ, and it is worth knowing:
// HF's RepetitionPenaltyLogitsProcessor scores against the entire input_ids,
// which INCLUDES the prompt, so it also penalises re-using words from the
// context. llama.cpp's ring buffer is filled by llama_sampler_accept, which is
// only called for sampled tokens, so it penalises generated tokens only.
// For extractive QA the llama.cpp behaviour is arguably the better one --
// copying the exact span out of the context is the desired answer -- but it is
// a deviation from the reference and should be stated as such.
static const int32_t PENALTY_LAST_N = 64;
static const float  PENALTY_FREQ    = 0.0f;   // disabled, as in the reference
static const float  PENALTY_PRESENT = 0.0f;   // disabled, as in the reference

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

    // Create the context ONCE here, reuse across generations.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 2048;   // MAX_LENGTH in the notebook
    cparams.n_threads = 2;      // big cores only on Helio G85 (2 big + 6 little)
    cparams.n_batch   = 512;
    cparams.no_perf   = false;  // enable llama_perf_context_print timings

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

    // Clear KV cache so prompts don't accumulate from previous generations.
    llama_memory_clear(llama_get_memory(g_ctx), true);

    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    // ── Tokenize ────────────────────────────────────────────────────────────
    //
    // add_special = true matches the notebook's
    //     tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    // so BOS is prepended here rather than written into the prompt string.
    // PromptBuilder.build() correspondingly emits no BOS of its own.
    //
    // parse_special = true is now inert: the prompt is plain instruction text
    // with no "<|...|>" substrings to interpret. It was load-bearing under the
    // old chat-template format and is left on only to avoid changing two
    // things at once.
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
    if (n == 0) {
        LOGE("prompt tokenised to nothing");
        return;
    }
    tokens.resize(n);
    LOGI("prompt: %d tokens", n);

    // ── Prefill ─────────────────────────────────────────────────────────────
    //
    // Chunked at n_batch. llama_decode REJECTS any batch larger than n_batch
    // (512), so a single call with a ~600-token RAG prompt fails and, if the
    // return value is unchecked, the model generates from an empty context --
    // fluent output with no grounding, indistinguishable from hallucination.
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

    // ── Callback references ─────────────────────────────────────────────────
    jclass    cbClass   = env->GetObjectClass(callback);
    jmethodID onToken   = env->GetMethodID(cbClass, "onToken",   "(Ljava/lang/String;)V");
    jmethodID onMetrics = env->GetMethodID(cbClass, "onMetrics", "(FJF)V");
    jmethodID onPrompt  = env->GetMethodID(cbClass, "onPrompt",  "(I)V");

    env->CallVoidMethod(callback, onPrompt, (jint) n);

    // ── Sampler chain ───────────────────────────────────────────────────────
    //
    // ORDER MATTERS. The chain runs in insertion order and each link
    // transforms the candidate set before the next sees it. The penalty must
    // therefore be added BEFORE the greedy selector -- greedy takes the argmax
    // of whatever logits reach it, so a penalty added afterwards would never
    // affect the choice.
    //
    // Signature note: llama_sampler_init_penalties has changed across
    // llama.cpp versions. This is the current 4-argument form
    //   (penalty_last_n, penalty_repeat, penalty_freq, penalty_present)
    // which matches a tree new enough to have llama_memory_clear and to have
    // deprecated llama_new_context_with_model -- i.e. this one. Older trees
    // took nine arguments beginning with n_vocab, special_eos_id and
    // linefeed_id; if this fails to compile, check llama.h rather than
    // guessing.
    llama_sampler* smpl = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            PENALTY_LAST_N, PENALTY_REPEAT, PENALTY_FREQ, PENALTY_PRESENT));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    auto t_start     = std::chrono::high_resolution_clock::now();
    int  token_count = 0;

    for (int i = 0; i < maxTokens; i++) {
        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        // Feed the sampled token back into the chain so the penalty's ring
        // buffer sees it. Without this the penalties sampler has no history
        // and silently does nothing -- the chain would run but the repetition
        // loop it exists to prevent would still occur.
        llama_sampler_accept(smpl, id);

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

        // NOTE ON EARLY STOPPING. The reference generates all max_new_tokens
        // and then keeps only the first line
        // (answer.splitlines()[0] in generate_candidate). Stopping here at the
        // first newline would produce identical text and save real time -- at
        // 272 ms/token, roughly 8 s per question when the answer is short.
        // It is deliberately NOT done, because tokens_generated would then
        // stop being comparable to the reference and to the other cells.
        // A production build could enable it; a measurement build should not.

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(g_ctx, next) != 0) {
            LOGE("decode failed at token %d", token_count);
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_perf_context_print(g_ctx);
    // Note: context is NOT freed here -- it lives until nativeFreeModel.
}

JNIEXPORT void JNICALL
Java_com_interlekt_slmengine_LlamaWrapper_nativeFreeModel(JNIEnv*, jobject) {
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    LOGI("Model and context freed");
}

JNIEXPORT jstring JNICALL
Java_com_interlekt_slmengine_LlamaWrapper_nativeSystemInfo(JNIEnv* env, jobject) {
    // CPU feature flags the build actually enabled: NEON, ARM_FMA, DOTPROD,
    // MATMUL_INT8. Written into every results file's header, so a run is
    // self-evidencing about which kernels it used rather than relying on a
    // build log kept elsewhere.
    return env->NewStringUTF(llama_print_system_info());
}

} // extern "C"
