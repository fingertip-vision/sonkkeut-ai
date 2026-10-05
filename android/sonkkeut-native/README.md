# Sonkkeut native Android AI

Pure Android library for Kotlin applications. The module compiles the existing `kr.sonkkeut.core` and `kr.sonkkeut.android` sources directly, plus the Kotlin menu speech index. It excludes the `kr.sonkkeut.rn` bridge and has no React Native, VisionCamera, JavaScript, Node or Metro dependency.

The original ONNX OCR/vision weights and CT2 Whisper v3 are unchanged. CameraX supplies `android.media.Image` frames directly to `SonkkeutEngine.processYuv(image, rotationDegrees)` on an analysis worker. The hand model is fetched with a pinned SHA-256 on the first build. The original speech model uses its verified, explicit first-use download and stays in the application's private files.

Add this sibling module in the Android project's settings:

```kotlin
include(":sonkkeut-native")
project(":sonkkeut-native").projectDir = file("../sonkkeut-ai/android/sonkkeut-native")
```

Then add `implementation(project(":sonkkeut-native"))` to the app. Initialize `SonkkeutEngine` on a background worker, close every CameraX `ImageProxy` in a `finally` block, pause processing when leaving the camera screen, and release the engine when its owning ViewModel is cleared. `KoreanWhisper`, `MenuRagDatabase`, and `MenuSpeechIndex` are ordinary Kotlin APIs and need no JS bridge.

The shipped ABI is ARM64. Model inference and the menu database run on the device. The standalone module contains OCR, hand guidance, Whisper and menu retrieval; it does not include an LLM. A future on-device LLM can be integrated here with the same scoped menu candidate and explicit-confirmation contract.
