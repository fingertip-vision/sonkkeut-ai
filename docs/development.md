# AI 모듈 개발과 프론트 전달 안내

## 기준 구현

현재 Kotlin 앱이 사용하는 브랜치는 `codex/unified-ai-20261003`이며 앱 0.2.3 기준 커밋은 `d9938b2`입니다. `main`은 구현 전체가 병합된 상태가 아닙니다.

| 구성 | 역할 |
|---|---|
| M1·M1-R | 키오스크 화면 꼭짓점·보정 |
| M2 | 화면 요소 탐지 |
| OCR | 화면의 메뉴·버튼·문구 읽기 |
| 손끝 분석 | 목표와 손끝 위치·검지 상태 확인 |
| Whisper | 한국어 음성을 텍스트로 변환 |
| 메뉴 검색 | 매장 메뉴 이름·별칭·관련 표현과 인식 결과 비교 |

영상·OCR 모델은 앱에 포함하고 Whisper 가중치는 별도 설치합니다. 모델을 업데이트할 때 URL뿐 아니라 SHA-256·파일 크기·입출력 계약도 함께 관리합니다.

## Android에 전달하는 방법

프론트에 Python 파일을 복사해 실행시키는 방식이 아닙니다. Kotlin 라이브러리 `android/sonkkeut-native`를 Gradle 모듈로 연결하고 모델·JNI 자원을 함께 빌드합니다. 일부 소스가 기존 `android/react-native-sonkkeut` 경로에 있어도 순수 Kotlin 모듈은 React Native 브리지를 제외합니다.

```kotlin
// 프론트 settings.gradle.kts
include(":sonkkeut-native")
project(":sonkkeut-native").projectDir = file("../sonkkeut-ai/android/sonkkeut-native")
// 앱 dependencies
implementation(project(":sonkkeut-native"))
```

`SonkkeutEngine`에 CameraX의 YUV 이미지와 회전 값을 전달합니다. 작업 스레드에서 초기화·추론하고 프레임 반환과 엔진 해제를 보장합니다. 결과는 앱의 화면·음성·진동에 연결하며 AI 라이브러리가 직접 외부 키오스크 버튼을 누르지 않습니다.

## 메뉴 검색의 의미

음성 텍스트 → 정식 이름·별칭 비교 → 철자/음식 개념/등록된 관련 표현 후보 검색 → 사용자 확인 순서입니다. 예를 들어 “크림 파스타”를 “까르보나라”의 관련 표현으로 등록하면 후보로 제안할 수 있지만 동일 음식이라고 확정하지 않습니다. 매장에 실제 존재하며 품절이 아닌 메뉴만 안내 대상으로 사용합니다.

현재 검색에는 제한된 개념 벡터와 로컬 메뉴 DB가 사용됩니다. 범용 LLM·신경망 임베딩 기반 음식 지식·알레르기 판정이 구현된 것으로 설명하지 않습니다. 관련 표현은 정확한 별칭으로 자동 승격하지 않습니다.

## 실행·평가 자료

기준 브랜치에서 다음 기존 문서를 읽으세요.

- `android/sonkkeut-native/README.md`: Kotlin 연동과 엔진 수명 관리.
- `docs/unified-ai.md`: 모델 실행·응답 계약·PC 추론 범위.
- `docs/interface.md`: 영상 결과와 언어/앱 연결 규약.
- `docs/speech-menu-rag.md`: 음성 평가 조건과 메뉴 검색 보정.
- `requirements.txt`, `requirements-unified.txt`: 전체 개발/선택적 CPU 추론 의존성.

Python 평가 환경은 Android 앱 런타임과 별개입니다. 전체 환경에서 `python -m pytest -q`를 실행하고, 모델이 필요한 테스트는 실제 가중치와 fixture 존재 여부를 확인합니다. 누락으로 건너뛴 테스트를 통과로 보고하지 않습니다. 합성 음성·에뮬레이터 성능을 실제 한국어 발화 성능으로 제시하지 않습니다.

## 모델 변경 시 전달 항목

1. AI 커밋과 호환 앱 커밋, 모델 버전, 다운로드 출처·해시·크기.
2. 입력 영상 형식·회전·좌표 기준, 출력 필드·신뢰도 기준.
3. 최소 Android/ABI·메모리·설치 공간과 처리 시간 측정 환경.
4. 기존/새 모델을 같은 실음성·OCR 데이터로 비교한 결과와 실패 사례.
5. 취소·백그라운드·화면 변경 시 늦은 결과를 폐기하는 조건.

현재 앱은 온디바이스 추론 경로입니다. PC의 선택적 HTTP 추론 서비스를 운영 백엔드나 필수 AI 서버로 혼동하지 마세요.
