# Sonkkeut native Android AI

앱 0.2.3 로컬 변경: `MenuDocument`에 선택 필드 `relatedTerms: List<String>`와 `description: String`을 추가했습니다. `MenuMatcher.recommend`는 등록된 관련 표현, 분류, 설명을 후보 검색에 사용하며 근거를 `store_knowledge`로 반환합니다. 이 정보는 정확한 별칭으로 승격하지 않고 사용자 확인을 거칩니다. 품절 메뉴는 제외합니다. 앱의 매장별 SQLite 메모를 합쳐 사용할 수 있으며 LLM이나 음식 성분 지식으로 간주하지 않습니다. 정식 이름은 다른 메뉴의 별칭보다 우선하며, 품절된 정식 메뉴를 별칭 상품으로 자동 대체하지 않습니다.

앱 0.2.1에서 `MenuMatcher`와 `DetailScanner`를 추가했습니다. `MenuMatcher`는 확실한 이름·별칭 검색, 편집 거리, 작은 음식 개념 벡터의 코사인 추천을 제공합니다. 신경망 임베딩·LLM·레시피 추론은 포함하지 않으며 추천은 사용자 확인이 필요합니다. `DetailScanner`는 카메라 원본의 겹치는 2×3 타일에서 ML Kit 줄 위치를 구한 뒤 기존 M3 모델로 한국어를 읽습니다. 작업 스레드에서 실행하고 세션 종료 조건을 전달하세요. 취소/시간 초과와 겹치는 줄 중복을 처리합니다. 부분 영상 좌표를 전체 키오스크 목표로 사용하면 안 됩니다.

`FrameResult`의 `tip_conf`와 `tip_pointing`은 스크롤 안내의 검지·신뢰도 검사에 사용합니다. 영상·음성 가중치와 Whisper JNI는 유지했습니다. 새 Android 검사에는 실제 타일 OCR과 화면별 대화 상태 재현이 포함됩니다.

Pure Android library for Kotlin applications. The module compiles the existing `kr.sonkkeut.core` and `kr.sonkkeut.android` sources directly, plus the Kotlin menu speech index. It excludes the `kr.sonkkeut.rn` bridge and has no React Native, VisionCamera, JavaScript, Node or Metro dependency.

The original ONNX OCR/vision weights and CT2 Whisper v3 are unchanged. CameraX supplies `android.media.Image` frames directly to `SonkkeutEngine.processYuv(image, rotationDegrees)` on an analysis worker. The hand model is fetched with a pinned SHA-256 on the first build. The original speech model uses its verified, explicit first-use download and stays in the application's private files.

Add this sibling module in the Android project's settings:

```kotlin
include(":sonkkeut-native")
project(":sonkkeut-native").projectDir = file("../sonkkeut-ai/android/sonkkeut-native")
```

Then add `implementation(project(":sonkkeut-native"))` to the app. Initialize `SonkkeutEngine` on a background worker, close every CameraX `ImageProxy` in a `finally` block, pause processing when leaving the camera screen, and release the engine when its owning ViewModel is cleared. `KoreanWhisper`, `MenuRagDatabase`, and `MenuSpeechIndex` are ordinary Kotlin APIs and need no JS bridge.

The shipped ABI is ARM64. Model inference and the menu database run on the device. The standalone module contains OCR, hand guidance, Whisper and menu retrieval; it does not include an LLM. A future on-device LLM can be integrated here with the same scoped menu candidate and explicit-confirmation contract.
