# sonkkeut-ai
손끝길 AI — 화면 꼭짓점(M1)·화면 요소(M2) 학습, 합성 데이터, 온디바이스 추론, 그리고 문자 인식·음성 주문 처리

| 폴더 | 기능 (명세) | 담당 | 상태 |
|---|---|---|---|
| (예정) | M1 화면 평면 · M2 화면 요소 · M4 손끝 (F-02·03·08) | 김우주 | – |
| [`ocr/`](ocr/) | **M3 문자 인식** (F-04) — PaddleOCR PP-OCRv5 모바일 + 키오스크 파인튜닝 | 노현석 | 실사진 메뉴판 95.6% |
| [`core/`](core/) | **F-05 화면 구조화 · F-06 음성 주문 이해 · F-07 버튼 순서 계획** + 가짜 키오스크 시뮬레이터 | 노현석 | 시뮬레이션 주문 성공 91~93% |
| [`asr/`](asr/) | **M5 음성인식** — Whisper small 어르신·실제 매장소음 파인튜닝 | 노현석 | 실제 소음 속 대화 CER 13.6→8.1% |
| `tools/` | AIHub 데이터 받기 스크립트 (`AIHUB_APIKEY` 환경변수) | | |

## 모듈 사이 데이터 (명세 6장)
F-03 요소 목록(`id·kind·box`) → **F-04** 가 `text·price·qty` 를 채움 → **F-05** 화면 구조 JSON → **F-07** 다음 버튼
**F-06** 발화 → 주문 의도 JSON → F-07.  요소 종류는 명세 5종에 `cart_item`(장바구니 줄)·`text`(수량 숫자 등)·`title` 추가 제안.

## 모델 내려받기
- M5 음성인식 Whisper v3: [Release `asr-whisper-elder-v3`](https://github.com/fingertip-vision/sonkkeut-ai/releases/tag/asr-whisper-elder-v3) (`whisper-elder-v3-ct2.zip`, faster-whisper 용)

## 데이터 출처
학습·평가에 과학기술정보통신부·한국지능정보사회진흥원(NIA)의 AI Hub(aihub.or.kr) 데이터를 활용했다:
관광 음식메뉴판 데이터, 야외 실제 촬영 한글 이미지, 명령어 음성(노인남여), 소음 환경 음성인식 데이터, 소상공인 고객 주문 질의-응답 데이터.

## 올리지 않은 것
- AIHub 데이터(관광 음식메뉴판·야외 한글·노인 명령어·소음 환경·주문 질의응답) — 약관상 재배포 금지. `tools/aihub_download.sh` 로 각자 받기
- OCR 학습 가중치(`kiosk_rec_v2`) — 필요하면 Release 로
- 스크립트의 절대 경로(`D:/sonkkeutgil/...`)는 개발 PC 기준이다. 옮길 때 각 파일 상단 경로 상수를 바꾼다

각 폴더 README 에 실행 방법·결과·한계를 적었다. 수치는 합성·AIHub 데이터 기준이며, **실제 키오스크 촬영·실제 주문 녹음 평가가 아직 없다.**
