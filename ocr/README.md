# 손끝길 M3 문자 인식 (F-04) — 담당 노현석

기능 명세 F-04: 요소별 영역 → 텍스트·가격·신뢰도. 수용 기준 **메뉴명 완전 일치 95% 이상**.

## 구성
| 파일 | 역할 |
|---|---|
| `ocr_m3.py` | `KioskOCR` — 펼친 화면 + F-03 요소 목록 → 요소별 `text / price / conf_ocr / uncertain`. 가격 정규화(`₩`→W 오인식 포함), 줄 읽기 순서, 매장 메뉴 사전 보정(F-14 연결) |
| `make_synth.py` | 합성 키오스크 화면(세로 720x1280) + 카메라 열화. `--rec` 이면 인식 모델 학습용 줄 이미지 |
| `make_layouts.py` | **학습에 안 쓴 인터페이스** 평가셋: 가로 목록형(list)·장바구니 패널형(cart), 고대비 테마, `--fonts holdout`, `--vocab unseen` |
| `eval_m3.py` | 정답 요소 박스를 넣고 문자만 평가 (종류별·열화 단계별 정확도, 지연) |
| `clean_labels.py` | 반사광으로 글자가 사라진 줄 이미지를 학습 목록에서 제외 |
| `configs/kiosk_rec_v1.yml` | korean_PP-OCRv5_mobile_rec 파인튜닝 설정 |
| `after_train.sh` | export → 두 평가셋에서 비교 |

모델: 검출 `PP-OCRv5_mobile_det`(공식) + 인식 `korean_PP-OCRv5_mobile_rec`(공식 → 키오스크 파인튜닝).
둘 다 모바일용이라 안드로이드 이식(Paddle Lite / ONNX) 대상 그대로.

## 데이터
- `data/synth_val` — 카페 어휘 화면 300장 (학습과 같은 어휘, 다른 렌더)
- `data/synth_unseen` — **학습에 없는** 한식·분식 어휘 화면 300장 → 외운 건지 일반화인지 확인
- `data/rec_train` — 3500화면에서 자른 줄 + 임의 한글 줄 25,000 (총 93.5k, 정제 후)
- `data/rec_val` — 학습 중 검증용 줄 7.4k

열화 단계 lv0 깨끗 / lv1 보통 / lv2 나쁨(저해상도·반사광·JPEG).
**합성 데이터 점수일 뿐**이다. 실제 촬영 500장(F-02~04 공통 평가셋)이 모이면 같은 json 형식으로 `data/real_v1/images/*.jpg|json` 에 넣고 `eval_m3.py --data data/real_v1` 로 다시 잰다.

## 실행
```bash
.venv/Scripts/python eval_m3.py --data data/synth_val                    # 공식 가중치
cd PaddleOCR && ../.venv/Scripts/python tools/train.py -c ../configs/kiosk_rec_v1.yml
bash after_train.sh kiosk_rec_v1                                         # export + 평가
```

## 환경 메모
- venv: Python 3.11, paddlepaddle-gpu 3.2.0 (cu126), paddleocr 3.7, numpy 1.26.4, opencv-contrib 4.10 고정.
  numpy 2.x 가 들어오면 albumentations/paddlex 가 깨진다.
- 배치 128은 Titan RTX 24GB를 넘겨 공유메모리로 새면서 5배 느려진다 → 64.
- 콘솔 한글 깨짐: `PYTHONIOENCODING=utf-8`.

## v2 학습 때 고칠 것
- 학습 데이터 영문이 대문자뿐이라 튜닝 모델이 소문자 x 를 X 로 읽는다 → generic_line 에 소문자·'x2' 형태 추가
  (지금은 ocr_m3 후처리가 x/X/× + 숫자를 `qty` 로 뽑아 흡수)
- 흐린 화면(lv2)의 남은 오류는 대부분 검출 누락 → det 쪽(CLAHE 전처리 또는 det 파인튜닝)
