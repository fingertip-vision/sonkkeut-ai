> 이 폴더(training/) 안에서 실행합니다. 학습이 끝나면 best.pt를 ../models/ 에 m1_screen_corners.pt, m2_screen_elements.pt 로 복사하세요.

# 손끝길 모델 학습 패키지

키오스크 화면 꼭짓점(M1)과 화면 요소(M2)를 학습하는 코드입니다.
두 모델 모두 **합성 데이터**로 학습하므로 외부 데이터셋 승인이나 사람 손 라벨링이 필요 없습니다.

| 파일 | 역할 |
| --- | --- |
| `kiosk_synth.py` | 키오스크 화면(메뉴·옵션·장바구니·결제)을 무작위로 그리고 YOLO 라벨 자동 생성 (M2용) |
| `corner_synth.py` | 생성한 화면을 키오스크 본체에 넣고 비스듬히 찍은 것처럼 합성, 꼭짓점 라벨 자동 생성 (M1용) |
| `train.py` | YOLOv8n(-pose) 학습, 선택적으로 TFLite INT8 변환 |
| `demo_pipeline.py` | 사진 한 장으로 M1 → 화면 펴기 → M2 전체 흐름 확인 |

## 1. 설치

```bash
pip install ultralytics opencv-python pillow numpy
# GPU용 PyTorch는 https://pytorch.org 에서 CUDA 버전에 맞춰 설치
```

Windows는 `C:/Windows/Fonts/malgun.ttf`(맑은 고딕)를 자동으로 찾습니다.
다른 한글 폰트를 섞고 싶으면 `.ttf` 파일을 폴더에 모아 `--font-dir 폴더` 로 넘기세요. 폰트가 다양할수록 실제 키오스크에 강해집니다.

## 2. 데이터 생성 (CPU, 이미지당 약 0.25초)

```bash
python kiosk_synth.py  --out data/m2 --n-train 4000 --n-val 500
python corner_synth.py --out data/m1 --n-train 3000 --n-val 400 --bg-dir 매장사진폴더
```

`--bg-dir` 은 선택입니다. 카페·식당 실내 사진을 수십 장만 넣어도 배경이 현실적으로 바뀌어 M1 성능이 크게 오릅니다.
시간을 줄이려면 `--seed` 를 다르게 주고 `--out` 을 나눠 여러 터미널에서 동시에 돌린 뒤 합치면 됩니다.

## 3. 학습 (RTX GPU 기준)

```bash
python train.py --task m2 --epochs 80 --batch 32
python train.py --task m1 --epochs 80 --batch 32
```

결과는 `runs/detect/m2/`, `runs/pose/m1/` 에 저장됩니다.

> M1의 pose mAP는 꼭짓점 정밀도를 반영하지 못합니다(4점 키포인트의 기본 OKS sigma 0.25가 너무 느슨함). 정밀도는 `../scripts/bench.py --m1-data data/m1` 의 '꼭짓점 오차 %'로 비교하세요. `results.png`(손실 곡선), `val_batch*_pred.jpg`(예측 예시)를 먼저 확인하세요.
모바일 탑재용으로 바꾸려면 `--export` 를 붙입니다(TFLite INT8).

## 4. 확인

```bash
python ../scripts/demo_image.py --m1 runs/pose/m1/weights/best.pt --m2 runs/detect/m2/weights/best.pt --img 키오스크사진.jpg
```

## 5. 실제 성능 평가 (기획서 실측치)

합성 데이터의 검증 점수는 **참고용**입니다. 합성 데이터끼리 평가하면 생성 방식의 버릇까지 맞히게 되어 점수가 부풀려집니다.
기획서에 넣을 수치는 학교 주변 실제 키오스크 사진으로 재야 합니다.

1. 실제 키오스크를 여러 각도에서 촬영(목표 500장, 처음엔 50~100장으로 시작)
2. [Label Studio](https://labelstud.io) 등으로 버튼·메뉴·가격·탭·뒤로가기 박스 라벨링
3. `data/real/images/val`, `data/real/labels/val` 에 넣고 `yolo val model=runs/detect/m2/weights/best.pt data=data/real/data.yaml`
4. 점수가 낮은 유형(예: 사진 메뉴판, 어두운 테마)을 `kiosk_synth.py` 에 추가하고 다시 학습

## 클래스

M2: `0 tab`, `1 menu`, `2 price`, `3 button`, `4 back`
M1: `0 screen` + 꼭짓점 4개(왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래)

`data/m2/meta.csv` 에는 화면 종류(menu/option/cart/payment)가 함께 저장되어, 이후 화면 분류(F-05) 학습에 쓸 수 있습니다.
