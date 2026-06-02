# Sinhala SLM Android

On-device Sinhala question-answering for Android, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp) and quantised small language models. Runs fully offline — no network, no cloud, no data leaves the device.

Part of the **Interlekt** research project at the University of Moratuwa, Faculty of Information Technology: *Hallucination Mitigation in Small Language Models for Domain-Specific Sinhala Question Answering*.

## Overview

This Android app loads a Llama-3 derivative fine-tuned on Sinhala (extended BPE vocabulary, 139,336 tokens) and runs inference entirely on-device using `llama.cpp` through a thin JNI bridge. The UI is Jetpack Compose, the inference loop is C++, and the model is GGUF-quantised for mobile RAM budgets.

The project is built to support a research workflow that compares quantisation levels (Q3_K_M, Q4_K_M, Q5_K_M, Q8_0) on factual-accuracy, hallucination rate, latency, and memory for Sinhala historical QA.

## Features

- Fully on-device inference — no network access required at runtime
- Streaming token output to the UI as it is generated
- Live metrics: ms/token, RAM usage, CPU%
- Pluggable model path — drop a different `.gguf` into `/data/local/tmp/` and restart
- Supports multiple quantisation levels for comparison studies
- Llama-3 chat template applied automatically for fine-tuned chat models

## Architecture

```
┌─────────────────────────────────────────────┐
│ Jetpack Compose UI                          │
│   MainActivity.kt                           │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│ InferenceViewModel.kt                       │
│   - coroutine scoping                       │
│   - prompt templating                       │
│   - metrics collection                      │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│ LlamaWrapper.kt   (Kotlin ↔ JNI)            │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│ llama_jni.cpp     (JNI bridge)              │
│   - model + context lifecycle               │
│   - sampler chain                           │
│   - tokenise / decode / detokenise loop     │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│ llama.cpp         (vendored)                │
│   built as libllama.so for arm64-v8a        │
└─────────────────────────────────────────────┘
```

## Requirements

- Android Studio (Hedgehog or newer)
- Android NDK 26.1.10909125 or later
- CMake 3.22+ (bundled with Android Studio)
- An arm64-v8a Android device, Android 11+
- ~2 GB free storage on device for the model

## Setup

### 1. Clone the repo with submodules

```bash
git clone --recursive https://github.com/<your-org>/sinhala-slm-android.git
cd sinhala-slm-android
```

If you forgot `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Place a GGUF model on the device

The app looks for models in this order:

1. `/data/local/tmp/sinllama.gguf` — fine-tuned Sinhala model
2. `/data/local/tmp/qwen.gguf` — fallback baseline
3. `/sdcard/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf`
4. `/storage/emulated/0/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf`

Push a model with:

```bash
adb push path/to/your-model.gguf /data/local/tmp/sinllama.gguf
```

### 3. Build and install

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or open the project in Android Studio and hit Run.

## Usage

1. Launch the app. It will scan the known paths and load the first GGUF it finds.
2. Once status shows **Model ready ✓**, type a Sinhala question (or English, depending on the model) and tap Generate.
3. Tokens stream into the output panel. Latency and resource metrics update live.

### Watching the native logs

```bash
adb logcat -c
adb logcat -s SLMEngine
```

You'll see thread count, prompt eval timing, generation timing, and llama.cpp's own performance counters at the end of every generation.

## Model Conversion

To convert a Hugging Face model to GGUF for this app, see the companion notebook `SinLlama_GGUF_Conversion.ipynb`. It handles:

- Downloading the merged HF model
- Patching `convert_hf_to_gguf.py` for extended-vocab pre-tokenizer hashes
- Converting to F16 GGUF
- Quantising to Q3_K_M, Q4_K_M, Q5_K_M, and Q8_0

## Project Structure

```
app/
├── src/main/
│   ├── cpp/
│   │   ├── CMakeLists.txt          # NDK build config
│   │   ├── llama_jni.cpp           # JNI bridge
│   │   └── llama.cpp/              # vendored upstream
│   └── java/com/interlekt/slmengine/
│       ├── MainActivity.kt          # Compose UI
│       ├── InferenceViewModel.kt    # state + coroutines
│       ├── LlamaWrapper.kt          # JNI Kotlin wrapper
│       └── MetricsCollector.kt      # RAM / CPU sampling
└── build.gradle.kts
```

## Configuration

Key inference parameters are set in `llama_jni.cpp` at context creation:

| Parameter | Default | Notes |
|---|---|---|
| `n_ctx` | 1024 | KV cache size in tokens. Raise for long prompts. |
| `n_batch` | 512 | Prompt eval batch size. Raise for faster first-token. |
| `n_ubatch` | 512 | Physical batch size. |
| `n_threads` | auto, capped at 8 | Detected from `std::thread::hardware_concurrency()`. |
| `use_mmap` | true | Loads weights lazily, reduces RAM pressure. |

Sampling defaults to greedy (`temp=0.0`) for deterministic factual QA. Edit `llama_jni.cpp` to use top-p / top-k for creative tasks.

## Performance Notes

Indicative numbers on a Xiaomi mid-range device (Snapdragon 6-series, 8 cores, 8 GB RAM):

| Model | Quant | File size | Prompt eval | Generation |
|---|---|---|---|---|
| Qwen 2.5 1.5B Instruct | Q4_K_M | 0.94 GB | ~200 ms/token | ~6,200 ms/token |
| SinLlama 1B (extended vocab) | Q4_K_M | 0.92 GB | varies | varies |

Mileage varies wildly with thermal throttling, background apps, and core scheduling.

## Research Context

The app is one component of a research project on hallucination mitigation in Sinhala SLMs. Other modules (not in this repo) cover:

- Domain adaptation of the base Llama-3 model on Sinhala corpora
- A retrieval-augmented generation pipeline over Sinhala historical sources
- An evaluation harness for factual accuracy, hallucination rate, ROUGE-L, and on-device latency/memory

If you use this work, please reference the project page (link TBD).

## Troubleshooting

**`Failed — no model found`**
The app could not locate a GGUF at any of the search paths. Push one to `/data/local/tmp/sinllama.gguf`.

**Build fails with `undeclared identifier 'llama_kv_*'` or similar**
The vendored `llama.cpp` is on a newer API. The current `llama_jni.cpp` uses the `llama_memory_*` family. If you've checked out an older `llama.cpp`, you may need to swap names — see the inline comments.

**Generation produces empty output / immediate EOG**
The chat template applied by `InferenceViewModel` may not match what your model was fine-tuned on. Inspect the model's training config and adjust the template strings in `generate()`.

**~25,000+ ms/token on a phone that ran Qwen at ~6,000 ms/token**
Likely thermal throttling, or background load. Let the phone cool, plug it in, and retry. Also confirm `Using N threads (hardware_concurrency=X)` in logcat shows `N ≥ 4`.

## License

This project's source code is licensed under the MIT License. See [LICENSE](LICENSE).

`llama.cpp` is vendored under its own MIT license. Model weights are subject to the licenses of their respective publishers (Meta Llama 3 Community License for Llama-3 derivatives).

## Acknowledgements

- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the inference engine
- Meta AI for the Llama-3 base model
- The fine-tuning team for the Sinhala-extended SinLlama checkpoint
- University of Moratuwa, Faculty of IT
