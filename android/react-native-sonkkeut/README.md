# react-native-sonkkeut

손끝길 영상 AI를 React Native 앱에 붙이는 모듈입니다. AI 계산(화면 인식, 손끝 추적, 유도, 누름 확인)은
전부 휴대폰 안에서 Kotlin + ONNX Runtime + MediaPipe로 돌고, JS는 결과만 받습니다. 서버가 필요 없습니다.

```
카메라(VisionCamera) ──프레임──▶ [Kotlin] 화면 꼭짓점 → 화면 요소 → 손끝 → 유도 → 누름 확인
                                        │
JS(useSonkkeut) ◀──결과 이벤트──────────┘  { event: {speak, vibe_hz, ...}, structure, verdict, hint }
```

지원: **안드로이드만** (iOS는 아직 없음), React Native 0.73 이상, VisionCamera 4.5 이상.

## 1. 설치 (sonkkeut-frontend 앱에서)

```bash
# 카메라와 프레임 처리기
npm i react-native-vision-camera react-native-worklets-core
# 이 모듈 (sonkkeut-ai 레포를 앱 레포 옆에 클론해 두었다고 가정)
npm i ../sonkkeut-ai/android/react-native-sonkkeut
```

`babel.config.js`에 worklets 플러그인을 추가합니다 (VisionCamera 프레임 처리기에 필요).

```js
module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [['react-native-worklets-core/plugin']],
}
```

`android/app/src/main/AndroidManifest.xml`에 카메라 권한이 있어야 합니다 (`<uses-permission android:name="android.permission.CAMERA" />`).
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
android/src/main/assets/sonkkeut/   모델 (M1·M2 ONNX INT8, M1-R 보정망, 손 관절은 빌드 때 내려받음)
src/index.ts                         JS API (useSonkkeut, Sonkkeut, 타입)
example/KioskGuideScreen.tsx         카메라 화면 예시
```

## 6. 검증 상태

- `core/`는 파이썬 기준 구현과 같은 입력에 같은 출력을 내는지 PC에서 확인했습니다 (`bash android/parity/run_tests.sh`).
  안내 이벤트 3,600건, 키오스크 화면 판단 480건, 모델 출력 해석(박스·꼭짓점) 등 전부 일치했습니다.
  같은 Kotlin 코드로 돌린 누르기 시뮬레이션은 40회 중 40회 성공, 버튼 밖에서 '누르세요' 0회였습니다.
- `android/`, `rn/`은 실제 안드로이드 SDK로 아직 빌드하지 못했습니다. 각 라이브러리의 API 모양을 흉내 낸 대역으로
  Kotlin 문법·타입 검사까지만 했습니다. 첫 빌드에서 라이브러리 버전 차이로 작은 수정이 필요할 수 있습니다.
- 휴대폰에서의 처리 시간은 아직 재지 않았습니다. 결과의 `timings.total_ms`로 바로 확인할 수 있습니다.
