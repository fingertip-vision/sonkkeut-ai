# Android Whisper native decoder · SDK 0.1.4

The original trained v3 weights are unchanged. `native/whisper_jni.cpp` now uses beam size **5** instead of 1. The mel frontend, JNI signature, language/prompt, INT8 CT2/ARM64 RUY runtime and 16 KiB page-alignment build flags are unchanged.

These source files match the prebuilt `react-native-sonkkeut/android/src/main/jniLibs/arm64-v8a/libsonkkeut_whisper.so`. The complete pinned CTranslate2 source, affinity patch, build helper and upstream notices are provided in the [v0.1.8 release source bundle](https://github.com/fingertip-vision/sonkkeut-frontend/releases/download/v0.1.8-speech-rag-20261005/sonkkeut-whisper-android-source-v2-beam5.tar.gz).

Extract that bundle and run `build-native.ps1` with Android NDK 26.1.10909125/CMake 3.22.1. This directory can also build the JNI adapter with its `CMakeLists.txt` by pointing `CT2_ROOT` to the extracted source and `CT2_LIBRARY` to its compiled ARM64 CTranslate2 library. No model weights are included in the source bundle. See [evaluation and integration](../../docs/speech-menu-rag.md).

Retain the upstream notices from the complete bundle and the SDK's `android/src/main/assets/sonkkeut/licenses/`. The mel filterbank/tokenizer contract is based on OpenAI Whisper under MIT; CTranslate2 is MIT and its RUY dependency is Apache-2.0.
