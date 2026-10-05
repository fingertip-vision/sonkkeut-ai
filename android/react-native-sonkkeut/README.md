# react-native-sonkkeut

손끝길 영상·글자·음성 AI를 React Native 앱에 붙이는 모듈입니다. 화면·요소 검출, 자체 OCR, 자체 Whisper 음성 인식, 손끝 추적과 누름 확인은 휴대폰 안에서 실행하고 JS는 화면 구조와 주문 문장을 받습니다. 음성 모델은 처음 한 번 인터넷으로 내려받습니다. 매장 메뉴와 동의한 익명 통계는 서비스 백엔드에 연결합니다.

```
카메라(VisionCamera) ──프레임──▶ [Kotlin] 화면 꼭짓점 → 화면 요소 → 손끝 → 유도 → 누름 확인
                                        │
JS(useSonkkeut) ◀──결과 이벤트──────────┘  { event: {speak, vibe_hz, ...}, structure, verdict, hint }
```

지원: **Android ARM64** (iOS는 아직 없음). 이번 연결은 React Native 0.76.9, VisionCamera 4.6.4, Worklets Core 1.5.0에서 검증했습니다.

SDK 0.1.3은 프론트의 접근성 설정을 실제 네이티브 음성·진동에 연결합니다. `Sonkkeut.configureFeedback(voice, vibration, rate)`로 자동 음성·진동과 안내 속도를 설정하며 속도는 0.75~1.25로 제한합니다. 설정은 네이티브 초기화 전에도 저장할 수 있습니다. 자동 메시지는 `Sonkkeut.announce(text)`, 사용자가 요청한 재안내·글자 읽기는 `Sonkkeut.say(text)`를 사용합니다. 자동 음성을 꺼도 명시적인 재안내는 유지되며 TalkBack의 touch exploration 사용 중 자동 TTS는 억제합니다. 글자·자막과 주문 인식은 이 설정과 독립적으로 유지됩니다. 프론트 v0.1.7 및 Android 피드백 정책·저장 검사 4개에서 연결을 확인했습니다. 실제 한국어 음성 재생·진동 감각의 실물 검증과 별개입니다.

SDK 0.1.2는 카메라 첫 프레임의 `Value is undefined, expected an Object` 종료 오류를 수정합니다. VisionCamera 4.6.4의 `plugin.call`은 두 번째 인자가 있으면 객체로 변환하므로, 회전값을 지정하지 않을 때는 인자를 생략합니다. 수정된 앱 v0.1.3의 실제 release APK에서 카메라 두 번 시작과 일시 정지·재개, 지속 프레임 처리 및 Jest 회귀 검사를 통과했습니다.

## 1. 설치 (sonkkeut-frontend 앱에서)

```bash
# 카메라와 프레임 처리기
npm i react-native-vision-camera react-native-worklets-core
# 이 모듈 (sonkkeut-ai 레포를 앱 레포 옆에 클론해 두었다고 가정)
npm i ../sonkkeut-ai/android/react-native-sonkkeut --install-links
```

`babel.config.js`에 worklets 플러그인을 추가합니다 (VisionCamera 프레임 처리기에 필요).

```js
module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [['react-native-worklets-core/plugin']],
}
```

`android/app/src/main/AndroidManifest.xml`에 카메라·마이크·진동 권한을 선언하고 카메라·마이크는 실행 중 요청합니다. Android 11 이상 TTS와 음성 인식 서비스 검색을 위해 `android.intent.action.TTS_SERVICE`, `android.speech.RecognitionService`의 queries도 선언합니다.
앱의 `defaultConfig.ndk`에는 `abiFilters "arm64-v8a"`를 지정해 모든 네이티브 라이브러리를 같은 ABI로 패키징합니다.
빌드할 때 MediaPipe 손 관절 모델(약 7.8MB)을 한 번 내려받으므로 첫 빌드는 인터넷이 필요합니다.

```bash
npx react-native run-android
```

## 2. 쓰는 법

가장 짧은 예:

```tsx
import { Camera, useCameraDevice, useCameraFormat } from 'react-native-vision-camera'
import { useSonkkeut } from 'react-native-sonkkeut'

function Guide() {
  const device = useCameraDevice('back')
  const format = useCameraFormat(device, [{ videoAspectRatio: 4 / 3 }, { videoResolution: { width: 1280, height: 960 } }])
  const { ready, result, screen, frameProcessor, setTarget } = useSonkkeut({
    onEvent: (e) => console.log(e.type, e.speak),     // 안내 이벤트
    onVerdict: (v) => console.log('누름 결과', v.result),
    onScreen: (s) => console.log('버튼', s.elements.length, '개'),
  })
  return <Camera style={{ flex: 1 }} device={device!} format={format} isActive={ready}
                 pixelFormat="yuv" frameProcessor={frameProcessor} />
}
```

바로 쓸 수 있는 화면 예시는 [`example/KioskGuideScreen.tsx`](example/KioskGuideScreen.tsx)에 있습니다
(카메라 + 화면 탭으로 목표 지정 + 큰 글씨 안내).

### 음성·진동

기본값(`nativeFeedback: true`)이면 **네이티브가 직접** 한국어 TTS로 읽고 진동합니다. 별도 코드 없이 시연이 됩니다.
앱에서 직접 출력하려면 `useSonkkeut({ nativeFeedback: false, onEvent })`로 끄고 아래 규칙대로 내면 됩니다.

| 필드 | 앱이 할 일 |
| --- | --- |
| `event.speak` | 있으면 읽는다. 이미 0.8초에 한 번 이하로 걸러져 있다 (같은 문장 반복은 2.5초) |
| `event.type === 'press'` | "지금 누르세요" — 읽던 문장을 끊고 바로 읽는다 |
| `event.vibe_hz` | 이 주기로 짧게(25ms) 진동. 멀리 2Hz → 가까이 4~8Hz → 버튼 위 10Hz, 0이면 멈춤 |
| `verdict.speak` | 누름 결과 ("눌렸습니다" / "버튼이 눌리지 않았습니다…") |
| `hint` | 화면을 못 찾았을 때 휴대폰 위치 안내 ("휴대폰을 조금 왼쪽으로") |

### 목표 버튼 정하기

| 함수 | 언제 |
| --- | --- |
| `setTarget(id, expect?)` | 언어 쪽(F-07 버튼 순서 계획)이 `screen.elements`에서 고른 id로 |
| `setTargetAtPreview(x, y, viewW, viewH, result.frame_size)` | 미리보기를 탭해서 (시연·저시력 모드) |
| `clearTarget()` | 안내 중지 |

`expect`는 누른 뒤 기대 결과입니다. 예: `{ screen_type: 'option', success_speak: '옵션 화면이 열렸습니다' }`.
형식은 [`../../docs/interface.md`](../../docs/interface.md)를 보세요.

## 3. 결과 형식

`result`(`SonkkeutResult`)의 필드와 뜻은 [`src/index.ts`](src/index.ts) 타입 주석과 `docs/interface.md`에 정리되어 있습니다.
안내·판정·화면 구조가 있을 때는 즉시, 그 밖에는 0.2초마다 이벤트가 옵니다.

## 4. 언어 쪽(노현석) 연결

기본 `structureProvider`는 M2 요소 검출과 ML Kit의 글자 줄 위치를 사용하고, 팀의 `kiosk_rec_v2`를 ONNX로 변환한 `KioskCtcRecognizer`가 각 줄을 읽습니다. 출력에는 `conf_ocr`, `uncertain`, `ocr_source`, `qty`를 보존합니다. 자체 모델 인식 실패 시 `mlkit_fallback`과 불확실 표시를 남기며, 불확실한 글자로 목표를 고르지 않습니다. 선택 상태는 `선택됨` 문구를 근거로 하며 증거가 부족하면 진행하지 않습니다.

JS에서 `Sonkkeut.setMenuAliases({아아: '아메리카노', 카페라떼: '카페라떼'})`로 메뉴 사전을 연결합니다. OCR 별칭과 실제 Whisper의 메뉴 문맥에 함께 사용합니다. 음성 문장은 앱에서 메뉴·수량·온도·매장/포장으로 해석한 후 사용자 확인을 받아 안내합니다. 모델과 원본 라이선스·재현 정보는 [`../../docs/unified-ai.md`](../../docs/unified-ai.md)에 있습니다.

```ts
const state = await Sonkkeut.getSpeechModelStatus();
const unsubscribe = Sonkkeut.addModelDownloadListener(({bytes, total_bytes, stage}) => {
  // 앱에서 다운로드 진행률과 설치 상태를 표시한다.
});
// 사용자가 약 485MB 다운로드를 선택한 후 호출한다.
if (!state.installed) await Sonkkeut.downloadSpeechModel();
else if (!state.ready) await Sonkkeut.prepareSpeechModel();
const orderText = await Sonkkeut.listen(); // 기본: 자체 CT2 Whisper elder v3
Sonkkeut.cancelListening();              // 녹음 중지 및 오래된 결과 차단
unsubscribe();
```

`cancelSpeechModelDownload()`로 다운로드를 중지할 수 있습니다. ZIP SHA-256 및 각 파일의 크기·해시를 확인한 후 앱 전용 폴더에 설치합니다. 16kHz 모노 PCM을 자체 C++ 전처리와 CTranslate2 ARM64 CPU에서 처리하며, 음성은 서버로 보내거나 파일로 보관하지 않습니다. `addSpeechListener()`는 provider·실제 추론 시간·sequence score·무음 확률을 제공합니다. sequence score를 단어 정확도로 해석하지 마세요.

`listen('system')`은 별도 대체 기능입니다. Android 12 이상에서 기기 오프라인 한국어 인식 서비스가 있을 때만 사용할 수 있으며 자체 Whisper 결과로 표시하지 않습니다. 문자 입력도 사용할 수 있습니다. 녹음 종료는 단순 에너지 기준이며 학습된 VAD 모델은 아닙니다.

글자 인식(F-04)과 화면 종류 판단(F-05)은 Kotlin에서 `SonkkeutEngine.structureProvider`에 꽂습니다.
키오스크 화면이 바뀔 때마다 요소 목록과 요소별 잘라낸 이미지(OpenCV `Mat`, RGB)를 받아 `ScreenStructure`를 돌려주면,
그 결과(`screen_type`, `text`, `price`, `cart_count`, `selected`)가 JS의 `screen`과 누름 결과 판정에 그대로 쓰입니다.

```kotlin
SonkkeutEngine.structureProvider = StructureProvider { elements, crops, flat, keyframeId ->
    // crops[e.id] (MatFrame) → OCR → e.text = ..., e.price = ...
    ScreenStructure(screenType = "menu", keyframeId = keyframeId, elements = elements)
}
```

## 5. 폴더 구조

```
android/src/main/java/kr/sonkkeut/
  core/      순수 Kotlin (안드로이드·OpenCV 의존 없음). 파이썬 sonkkeut_vision과 1:1 대응
             Plane(F-02) · Detect(F-03 해석) · Fingertip(F-08) · Guide(F-09) · Verify(F-10) · Keyframe · Pipeline
  android/   OpenCV·ONNX Runtime·MediaPipe 구현, SonkkeutEngine(앱 전체에서 하나), 기본 음성·진동
  rn/        React Native 모듈(제어·이벤트) + VisionCamera 프레임 처리 플러그인 "sonkkeut"
android/src/main/assets/sonkkeut/   M1·M2 ONNX INT8, M1-R, M3 OCR와 사전·명세·라이선스
android/src/main/jniLibs/arm64-v8a/ CTranslate2와 Whisper JNI (음성 가중치는 최초 실행 때 다운로드)
android/native-whisper/            고정 버전 원본 다운로드·패치·재빌드 스크립트와 C++ 소스
src/index.ts                         JS API (useSonkkeut, Sonkkeut, 타입)
example/KioskGuideScreen.tsx         카메라 화면 예시
```

## 6. 검증 상태

- `core/`는 파이썬 기준 구현과 같은 입력에 같은 출력을 내는지 PC에서 확인했습니다 (`bash android/parity/run_tests.sh`).
  안내 이벤트 3,600건, 키오스크 화면 판단 480건, 모델 출력 해석(박스·꼭짓점) 등 전부 일치했습니다.
  같은 Kotlin 코드로 돌린 누르기 시뮬레이션은 40회 중 40회 성공, 버튼 밖에서 '누르세요' 0회였습니다.
- 자체 OCR의 합성 한국어 문구 3개는 실제 ONNX 출력과 Kotlin CTC 디코딩이 일치했습니다. 자체 Whisper는 API 35 에뮬레이터의 ARM64 번역 실행 환경에서 실제 PCM→멜→CT2→한국어 주문을 확인했습니다. 메뉴 문맥을 사용한 주문은 아메리카노 2잔, 카페라떼 1잔, 포장을 정확히 읽었습니다. 약 20.4초는 해당 에뮬레이터 수치이며 휴대폰 지연 목표의 측정값이 아닙니다.
- Windows CPU 통합 런타임의 OCR→음성→주문 해석→확인된 목표 계획도 실제 HTTP 요청으로 검증했습니다. 브라우저 시뮬레이션의 모의 AI와 구분합니다.
- 휴대폰에서의 처리 시간은 아직 재지 않았습니다. 결과의 `timings.total_ms`로 바로 확인할 수 있습니다.
- `screen_type_not`가 맞더라도 요구한 `cart_delta`를 읽지 못하면 성공으로 판정하지 않습니다. Python과 Kotlin 모두 `uncertain`을 반환합니다.
- 실제 휴대폰 카메라로 전체 주문을 수행하는 현장 검증은 남아 있습니다. 현재 APK는 개발 서명 시연용입니다. CTranslate2/JNI는 16KB ELF 정렬로 빌드했지만 전체 APK의 16KB 페이지 지원은 검증하지 않았으며 현재 NDK 26.1 C++ 런타임은 4KB 정렬입니다.
