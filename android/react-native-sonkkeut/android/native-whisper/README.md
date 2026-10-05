# Korean Whisper Android prototype

This is a real CTranslate2 CPU adapter for the user's `whisper-elder-v3-ct2` model. It does not replace the model with mock transcripts or other weights. All production image/audio inference stays on the Android device; only the explicit first model installation downloads an archive.

SDK 0.1.4 uses beam size 5. The execution measurements below describe the earlier beam-size-1 fixture; the updated evaluation and menu retrieval correction are documented in [speech-menu-rag.md](../../../../docs/speech-menu-rag.md). The matching wrapper is also available in [android/native-whisper](../../../native-whisper/).

## Verified execution

- CTranslate2 4.8.2 was cross-compiled with Android NDK 26.1.10909125 / Clang 17 / CMake 3.22.1 for `arm64-v8a`, API 24.
- Ruy is enabled; MKL, oneDNN, OpenBLAS, CUDA, cuDNN, HIP, OpenMP, CLI, and tests are disabled. Custom CT2 threading is used. The optional Linux `pthread_setaffinity_np` path is excluded on Android; inference retains the normal default `cpu_core_offset=-1`.
- `libctranslate2.so` and `libsonkkeut_whisper.so` are real ELF64 AArch64 shared libraries. Their LOAD alignment is 16 KB. NDK 26.1's bundled `libc++_shared.so` has 4 KB LOAD alignment, so this prototype does not establish full 16 KB page-size compatibility.
- On the API 35 x86_64 AVD with ARM64 native translation, a synthetic 5.51-second Korean order was transcribed using the real model and JNI. Menu context produced: `따뜻한 아메리카노 두 잔 하고 카페라떼 한 잔 포장해 주세요.` Inference took 20.376 seconds in this translation environment; this is not a physical ARM64 phone performance measurement.
- Without menu context the same model produced `카페라 때`; PC CTranslate2 reproduced that error. The menu prefix improves the actual model generation rather than editing recognized text afterwards.
- Native Windows and Android log-mel features both match the verified Python reference across 240,000 floats: maximum absolute error `1.9789e-5`, mean absolute error about `2.42e-8`.

## App integration

Copy the four Kotlin files under `kotlin/kr/sonkkeut/android` into the Android library, the two CT2/JNI shared libraries into `src/main/jniLibs/arm64-v8a`, and preserve `consumer-rules.pro` when enabling minification. The app already uses `libc++_shared.so` through React Native/native dependencies; resolve one compatible runtime when packaging rather than adding conflicting copies. No model files are embedded in these libraries.

`KoreanWhisper(context)` exposes:

- `status(): Map<String, Any?>`: provider=`ct2-whisper-v3`, model_version=`v3`, installed, ready, downloading, busy, requires_download.
- `setMenuContext(names: List<String>)`: canonical menu names, at most 24 names / 500 characters. Hints are encoded with the frozen model vocabulary's byte-BPE ranks and limited to 223 tokens, followed by the Korean/transcribe/no-timestamps prompt. The native prompt uses vocabulary token strings rather than hardcoded token IDs.
- `install(downloadUrl = WhisperModelInstaller.DOWNLOAD_URL, onProgress(bytes, total, stage), onReady, onError)`: explicit user action. Stages are downloading, verifying_archive, installing, ready. A negative download total means unknown content length.
- `prepare(onReady, onError)`: load an already installed model on the IO worker.
- `listen(onResult, onError)`: AudioRecord mono PCM16, 16 kHz, maximum 30 seconds. Basic RMS silence endpointing ends a phrase; it is not a trained VAD model.
- `transcribePcm(pcm, onResult, onError)`: actual inference fixture/helper using the same production path.
- `cancel()` and `close()`: recording stops immediately; callback generation guards suppress stale results. An already running CT2 generation finishes before the worker handles another job. The main UI thread never waits on model loading, feature extraction, or generation.

The result contains `text`, `sequenceScore`, `noSpeechProbability`, `inferenceMs`, `audioSeconds`, and `provider`. Sequence score is CT2's length-normalized sequence score; it is not a calibrated per-word confidence. Android SpeechRecognizer and this custom Whisper model must have distinct provider/status labels in the app.

## Installation guarantees

The installer streams the fixed public HTTPS archive, checks the pinned archive SHA-256, rejects unsafe/unknown/duplicate ZIP entries and oversized output, then verifies all five original files by exact length and SHA-256. It prepares an immutable version directory before publishing `current.json` with Android AtomicFile. Cancellation does not publish an incomplete version. Temporary artifacts remain in this app's private `filesDir`; no external credentials are required.

Archive: `https://github.com/fingertip-vision/sonkkeut-ai/releases/download/asr-whisper-elder-v3/whisper-elder-v3-ct2.zip` (484,672,869 bytes). Original model files are preserved. FLOAT16 stored weights are loaded with CPU INT8_FLOAT32 runtime computation; FLOAT16 CPU computation is unsupported by CT2.

## Rebuild

Use the prepared source bundle, or run `fetch-source.ps1` in a fresh directory. It pins CTranslate2 and required CPU submodules, then applies `patches/android-pthread-affinity.patch`. CUDA-only submodules are unnecessary. Run `build-native.ps1 -AndroidSdk PATH_TO_ANDROID_SDK`; this builds and strips the ARM64 libraries using the SDK's NDK/CMake. Other hosts can pass the same flags directly to CMake with their NDK toolchain.

`SOURCE_VERSIONS.json` records the source revisions, toolchain, mel filter digest and model archive digest. The source bundle includes original third-party licenses and copyright headers. `THIRD_PARTY_NOTICES.md` identifies the dependencies. The bundled exact 80-bin filterbank is from OpenAI Whisper's MIT-licensed `mel_filters.npz`; `generate_mel_header.py` reproduces the header with the standard library only.

The `kr.sonkkeut.asrsmoke` APK is an isolated fixture application. It reads only explicitly staged synthetic audio/model files, writes private `result.json` and `native-logmel.f32`, and does not touch `com.sonkkeut` data. Its local debug signing key, APKs, model archives, downloaded weights, caches and compiler build outputs are excluded from the source bundle.

Official source/API references: https://github.com/OpenNMT/CTranslate2/tree/v4.8.2 and https://github.com/openai/whisper.
