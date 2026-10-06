# Android Whisper native decoder · SDK 0.1.4

The original trained v3 weights are unchanged. `native/whisper_jni.cpp` now uses beam size **5** instead of 1. The mel frontend, JNI signature, language/prompt, INT8 CT2/ARM64 RUY runtime and 16 KiB page-alignment build flags are unchanged.

These source files match the prebuilt `react-native-sonkkeut/android/src/main/jniLibs/arm64-v8a/libsonkkeut_whisper.so`. The matching JNI adapter and the pinned dependency fetcher, affinity patch, build helper and upstream notices are also tracked in [the SDK rebuild directory](../react-native-sonkkeut/android/native-whisper/).

In the SDK rebuild directory, run `fetch-source.ps1`, then `build-native.ps1 -AndroidSdk PATH_TO_ANDROID_SDK` with Android NDK 26.1.10909125/CMake 3.22.1. This directory can also build the JNI adapter with its `CMakeLists.txt` by pointing `CT2_ROOT` to the fetched source and `CT2_LIBRARY` to its compiled ARM64 CTranslate2 library. Copy the stripped libraries from the rebuild directory's `jniLibs/arm64-v8a` into the SDK's `android/src/main/jniLibs/arm64-v8a` before rebuilding the application. No model weights are included in this source directory. See [evaluation and integration](../../docs/speech-menu-rag.md).

Retain the fetched dependencies' upstream notices and the SDK's `android/src/main/assets/sonkkeut/licenses/`. The mel filterbank/tokenizer contract is based on OpenAI Whisper under MIT; CTranslate2 is MIT and its RUY dependency is Apache-2.0.
