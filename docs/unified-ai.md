# 손끝길 통합 AI 계약 (2026-10-03)

M1 화면 평면·M2 요소·M4 손끝 가이드는 기존 모델을 유지한다. M3는 팀의
`ocr-kiosk-rec-v2` 가중치를 ONNX로 변환하고, M5는 팀의
`asr-whisper-elder-v3` CTranslate2 가중치를 사용한다. 모델을 하나의 신경망으로
재학습한 것이 아니라 서로 다른 모델을 같은 런타임·JSON 계약으로 묶은 것이다.

## 실행 위치

| 경로 | 실제 모델/기능 | 프론트 전달 |
|---|---|---|
| Android `react-native-sonkkeut` | ONNX M1/M2/M1-R + MediaPipe 손 + 자체 M3 CTC OCR | `SonkkeutResult.structure`, `conf_ocr`, `uncertain`, `ocr_source`, `qty` |
| Android 음성 모듈 | CTranslate2 v4.8.2 ARM64 RUY CPU + 자체 Whisper v3, 앱의 모델 설치 절차 | `listen()`의 한국어 발화 문자열, 앱의 명시적 주문 확인 |
| Python `sonkkeut_ai` | 같은 M3 ONNX·M5 Whisper CPU + 팀의 `core` 주문 이해/플래너 | `sonkkeut.ai.v1` HTTP/함수 응답 |
| 공개 웹 시뮬레이션 | 가상 키오스크·모의 화면/주문·결제 | 실제 카메라 모델 실행과 구분해서 표시 |

Cloudflare Worker는 Paddle·CTranslate2 Python 프로세스를 실행하지 않는다. Python API는
선택 가능한 PC 런타임이며 기본 주소는 `127.0.0.1`이다. Android는 이 API가 없어도
기기 안에서 인식한다. Python API를 외부 호스트에 열 때는 `SONKKEUT_AI_KEY`와 명시적
`SONKKEUT_AI_ORIGINS`를 설정한다.

## 모델 재현

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-unified.txt
.venv/Scripts/python.exe scripts/fetch_unified_models.py --output ../../work/ai-model-downloads
```

모델 다운로드는 공개 GitHub 릴리스만 사용하고 ZIP SHA256을 검사한 뒤 추출한다.
AI Hub 학습 데이터·사용자 음성은 저장소에 포함하지 않는다. Whisper CPU 런타임은 동일한
Whisper small의 `tokenizer.json`만 별도로 받고 해시를 고정한다. 튜닝된 `model.bin`을
일반 Whisper 가중치로 바꾸지 않는다. `av==16.0.1`은 `faster-whisper==1.2.1`과
검증한 조합이다.

| 자산 | SHA256 |
|---|---|
| OCR 원본 ZIP | `e14ac5d92849b4e1d542c523882c3f010879420ce29248d8c92cf3faf9561326` |
| OCR ONNX | `80d6462d5cc0d0a189ecd810940df484902594a11087a4592e8a8dec47d87145` |
| ASR 원본 ZIP | `d6d5b3c3efbc7be6b9d6585fca8155e4418fa00f443eeb9aff464075da37c3ea` |
| ASR `model.bin` | `b5468a47157fad449fe6ff093786848badbd7c0c1b28aa15c3dd346ab8a365da` |
| 동일 Whisper tokenizer | `27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566` |

### OCR ONNX 변환

```powershell
.venv/Scripts/python.exe -m pip install paddlepaddle==3.1.1 paddle2onnx==2.1.0 onnx==1.17.0 PyYAML==6.0.2 setuptools==75.8.0
.venv/Scripts/python.exe scripts/export_unified_ocr.py --source ../../work/ai-model-downloads/kiosk_rec_v2 --output ../../work/ocr-android-export
```

Windows에서 Paddle2ONNX 2.1.0의 PIR DLL 심볼은 Paddle 3.2/3.3과 맞지 않았다.
Paddle 3.1.1로 실제 변환을 검증했다. 출력은 13,413,875바이트 ONNX opset 17이다.

- 입력 `x`: float32 `[1,3,48,W]`, BGR, 종횡비 유지, W 최소 320/최대 3200.
- 정규화 `(pixel / 255 - .5) / .5` 이후 오른쪽을 0으로 패딩한다.
- 출력 `fetch_name_0`: float32 `[1,T,11947]` **Softmax 확률**. 추가 Softmax를 적용하지 않는다.
- CTC blank=0. 연속 동일 ID만 합치며 blank 사이 동일 ID는 두 번 읽는다.
- 사전은 원본 11,945자 + 마지막 공백. TXT는 blank를 제외한 11,946줄이며 마지막 공백 줄을 지우면 안 된다.
- 출력 문자열을 NFC로 정규화하고, 실제 방출한 nonblank 토큰 확률의 평균을 신뢰도로 사용한다.
- M2의 0..1 박스는 화면 평면 기준이다. Python `OCRStructure`는 `flatten()`의 0.04 여백을 제거하고 읽는다.
- Android ML Kit는 줄 위치·전체 화면 구조를 보조한다. 자체 모델 인식이 낮아 ML Kit 문구를 보여줄 때는
  `ocr_source="mlkit_fallback"`, `uncertain=true`로 구분하고 누르기 지시는 허용하지 않는다.

## Python 런타임

```python
from sonkkeut_ai.ocr import KioskRecognizer
from sonkkeut_ai.runtime import UnifiedRuntime, create_vision_pipeline
from sonkkeut_ai.speech import WhisperSpeech

ocr = KioskRecognizer("m3_kiosk_rec_v2.onnx", "m3_kiosk_rec_v2_dictionary.txt")
speech = WhisperSpeech("whisper-elder-v3")
ai = UnifiedRuntime(ocr, speech)
reply = ai.transcribe("order.wav", ["아메리카노", "카페라떼"])
# M1/M2 .pt PC 추론은 기존 requirements.txt의 Ultralytics/PyTorch 환경도 필요하다.
vision = create_vision_pipeline("models/m1_screen_corners.pt", "models/m2_screen_elements.pt", ocr)
```

기존 Python `VisionPipeline`의 `structure_fn`에 실제 M3를 연결하며, OCR `uncertain`이면
요소 `conf=0`을 반영한다. M2의 높은 검출 신뢰도만 남아 잘못된 글자의 버튼을 누르라고
안내하는 일을 막는다. 프론트 계약은 `total`과 `total_price`, `cart_count`, `selected` 및
OCR 증거를 함께 보존한다. 명확한 `매장`/`포장` 버튼 쌍을 신뢰도 .80 이상으로 읽었을 때만
`screen_type="method"`로 분류한다.

### HTTP

```powershell
$env:SONKKEUT_ASR_MODEL = 'C:/absolute/path/whisper-elder-v3'
.venv/Scripts/python.exe -m uvicorn sonkkeut_ai.api:app_factory --factory --host 127.0.0.1 --port 18081
```

OCR 경로는 기본으로 Android assets를 사용한다. 필요한 경우 `SONKKEUT_OCR_MODEL`,
`SONKKEUT_OCR_DICTIONARY`를 명시한다. OpenAPI는 `/openapi.json`, 설명 화면은 `/docs`이다.
요청과 응답은 Pydantic 모델로 검증한다. 준비되지 않은 모델은 `capabilities.ready=false`와
`provider="unavailable"`, 추론 요청은 503으로 표시한다. 다른 모델이나 모의 결과로 대체하지 않는다.

| 경로 | 입력 | 응답 |
|---|---|---|
| `GET /v1/ai/capabilities` | 없음 | 실제 OCR/ASR/NLU provider와 준비 여부 |
| `POST /v1/ai/order` | `{text, screen_menus, store_menus}` | 발화, 주문 의도, 확인 문장, `requires_confirmation=true` |
| `POST /v1/ai/speech` | multipart `audio`, `menus`/`store_menus` JSON 배열 | 자체 Whisper 발화와 같은 주문 의도 |
| `POST /v1/ai/screen` | `{image_base64, elements, keyframe_id, previous?}` | 요소 문자·가격·수량·신뢰도·화면 구조 |
| `POST /v1/ai/plan` | `{screen, intent, progress?, confirmed}` | 다음 목표 ID/안내 문장/새 progress |

`/screen` 이미지는 이미 펼쳐진 화면이며 여백을 제거한 이미지다. 카메라 원본의 투영변환은
M1 경로에서 먼저 수행한다. 전체 요청은 chunked 포함 8MiB, 이미지 6MiB/2백만 픽셀,
음성은 0.05~30초로 제한한다. 오디오 디코드는 30초에서 멈춰 무제한 PCM을 만들지 않는다.
모델은 한 번 불러오고 추론을 직렬 처리하며 요청의 이미지·음성·주문을 영구 저장하지 않는다.

```json
{
  "schema_version": "sonkkeut.ai.v1",
  "transcript": "따뜻한 아메리카노 두 잔 하고 카페라떼 한 잔 포장해 주세요.",
  "intent": {
    "items": [
      {"menu": "아메리카노", "qty": 2, "options": {"temp": "hot"}, "status": "pending"},
      {"menu": "카페라떼", "qty": 1, "options": {}, "status": "pending"}
    ],
    "dine": "포장",
    "source": "rule",
    "say": "따뜻한 아메리카노 두 잔, 카페라떼 한 잔, 포장 맞나요?"
  },
  "requires_confirmation": true,
  "plan": null,
  "providers": {"asr": "asr-whisper-elder-v3-ct2", "nlu": "korean-order-rules"}
}
```

발화 인식만으로 목표를 설정하지 않는다. 주문 확인 전 `confirmed=false`의 플래너는
`plan=null`이다. 불확실한 OCR 요소와 `unknown`/`other` 화면은 목표를 추측하지 않고 다시 읽기를
요청한다. 결제 화면에서는 자동 결제를
계획하지 않는다. `core/planner.py`는 Python 프로토타입으로 옵션 진행 상태를 클라이언트가
유지한다. `progress` 응답은 제안 상태이며 실제 F10 성공을 확인한 뒤 반영한다. 실패/불확실 판정이면
이전 상태를 유지하고 화면을 다시 읽는다. Android 프론트는 확인된 주문과 F10 성공/실패 이벤트에
맞춰 자체 진행 상태를 갱신한다.

## 검증 기록과 한계

- 자체 OCR ONNX 실제 CPU 추론: 합성 글자 `아메리카노`, `4,500원`, `따뜻하게` 3개 정확 일치,
  신뢰도 .998~.9997. 동일 출력의 순수 Kotlin CTC decode도 3개 일치했다.
- 자체 Whisper v3 CPU int8: Heami 합성 주문을 정확히 받아쓰고 아메리카노 2잔/hot,
  카페라떼 1잔/포장 주문과 확인 문장으로 변환했다. 한 PC 측정 추론 약 .9~1.1초.
- 실제 HTTP 모델 요청: OCR 화면 → 음성 주문 의도 → 확인된 목표 `e1` 연계 검증.
- 2026-10-03 Python 순수 가이드·통합 계약 테스트 38개 통과. 기존 PyTorch 기반 시뮬레이션
  `test_end_to_end` 3개는 별도 CPU 통합 환경에서 선택 제외했다.
- 테스트 데이터는 공개 시연용 합성 문자·합성 음성이다. 실제 휴대폰 카메라/마이크,
  매장 소음/조명/손 가림, 키오스크 주문 전 과정의 성공률을 입증하는 평가가 아니다.
- Android ARM64 native 빌드·최종 모델 추론 검증 결과는 앱 배포 검증 기록을 기준으로 확인한다.

학습/평가 코드 원본은 `hyunseok/m3-ocr-core-asr`의 `ocr/`, `asr/`, `core/`를 통합했다.
단일 메뉴/빈 메뉴의 추천 후보 처리 오류와 패키지 import 경로를 고쳤다. 기존 로컬 영상 가이드
변경은 보존했다. 원본 AI Hub 데이터는 재배포하지 않는다. 의존성의 라이선스와 데이터 출처는
`THIRD_PARTY_NOTICES.md`를 참조한다.
