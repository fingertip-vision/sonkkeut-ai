# 손끝길 통합 AI 의존성과 모델 출처

이 문서는 프로젝트 전체의 새 라이선스를 지정하지 않는다. 기존 팀 소스와 모델의 저작권 및
데이터 출처를 보존하고, 새 통합에 사용한 외부 구성 요소의 라이선스를 표시한다.

| 구성 요소 | 출처/라이선스 | 사용 |
|---|---|---|
| PaddleOCR | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), Apache-2.0 | 팀 OCR v2의 SVTR_LCNet/PPLCNetV3 인식 구조 |
| Paddle2ONNX | [PaddlePaddle/Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX), Apache-2.0 | Paddle PIR → ONNX 변환 도구 |
| Whisper / Whisper small tokenizer | [openai/whisper](https://github.com/openai/whisper), MIT | 팀 Whisper v3 기반 모델과 동일 토큰 사전 |
| faster-whisper | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper), MIT | Python CPU 추론 |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2), MIT | 실제 Whisper 가중치 추론, Android ARM64 런타임 |
| RUY | [google/ruy](https://github.com/google/ruy), Apache-2.0 | ARM64 CPU 연산 |
| Ultralytics YOLOv8 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 | M1(`yolov8n-pose.pt`)·M2(`yolov8n.pt`) 학습 시작 가중치와 학습·내보내기 도구 (`training/train.py`) |
| MediaPipe Hand Landmarker | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), Apache-2.0 | 손 관절 21점 추정 라이브러리(`tasks-vision`)와 모델 `hand_landmarker.task`(float16 버전 1, 빌드 때 내려받음) |

M1·M2 ONNX 메타데이터에는 Ultralytics의 AGPL-3.0이 표기되어 있다. 이 모델을 담은 앱을 배포할 때
AGPL-3.0의 조건(소스 공개 등)을 어떻게 지킬지는 팀에서 확인한다.

각 upstream LICENSE 원문은 `licenses/`에 보존한다. 팀의 OCR/Whisper 릴리스에는 별도
추가 모델 라이선스가 제공되지 않았으므로 이 통합이 그 모델에 새로운 라이선스를 부여하지 않는다.

팀 학습 모델은 과학기술정보통신부·한국지능정보사회진흥원(NIA)의 AI Hub 데이터를 사용했다.
OCR: 관광 음식메뉴판/야외 실제 촬영 한글 이미지. ASR: 명령어 음성(노인남여)/소음 환경 음성인식.
AI Hub 원본 학습·검증 데이터는 이 저장소나 앱 모델 설치 파일에 넣지 않는다.

공개 모델 릴리스: [OCR v2](https://github.com/fingertip-vision/sonkkeut-ai/releases/tag/ocr-kiosk-rec-v2),
[Whisper v3](https://github.com/fingertip-vision/sonkkeut-ai/releases/tag/asr-whisper-elder-v3).
