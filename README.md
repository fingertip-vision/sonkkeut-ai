# sonkkeut-ai · 손끝길 영상 AI

스마트폰 카메라로 키오스크 화면과 손끝을 함께 보고, 목표 버튼까지 손가락을 음성·진동으로 유도하는 온디바이스 AI입니다.
이 레포에는 영상 AI·모델 학습 코드와 Android 앱에서 사용하는 한국어 OCR·화면 구조화·오프라인 음성 제어 모듈이 들어 있습니다.

| 기능 | 내용 | 코드 |
| --- | --- | --- |
| F-02 화면 평면 추정 | M1으로 화면 네 꼭짓점 검출 → M1-R로 꼭짓점 정밀 보정 → 호모그래피 → 광류로 매 프레임 추적 | `plane.py`, `corner_net.py` |
| F-03 화면 요소 인식 | 펼친 화면에서 M2로 탭·메뉴·가격·버튼·뒤로가기 탐지, 겹침 정리, 읽는 순서 정렬 | `sonkkeut_vision/elements.py` |
| F-04·F-05 OCR·화면 구조 | Android에 포함한 한국어 ML Kit로 글자·메뉴·선택·장바구니·총액 해석 | `android/react-native-sonkkeut/android/src/main/java/kr/sonkkeut/android/KoreanStructure.kt` |
| F-08 손끝 추적 | MediaPipe 손 관절 21점 → 검지 끝 → 화면 좌표, 5프레임 평균, 검지를 접으면 안내 중지 | `sonkkeut_vision/fingertip.py` |
| F-09 손끝 유도(오차 계산) | 8방향·3거리 구간, 음성 0.8초 간격, 0.3초 머무르면 "지금 누르세요" | `sonkkeut_vision/guidance.py` |
| F-10 누름 결과 확인 | 화면 변화 감지 → 새 화면 구조를 기대 결과와 비교 | `sonkkeut_vision/verify.py` |
| (공통) 키프레임 판단 | 화면이 바뀌고 멈췄을 때만 무거운 화면 읽기 실행, 손 영역은 비교에서 제외 | `sonkkeut_vision/keyframe.py` |

언어 쪽(노현석: F-04 문자 인식 ~ F-07 버튼 순서 계획), 앱(임현승: 음성·진동 출력)과의 연결 규약은
[`docs/interface.md`](docs/interface.md)에 있습니다.

이 레포는 두 부분으로 되어 있습니다.

| 부분 | 언어 | 용도 |
| --- | --- | --- |
| `sonkkeut_vision/`, `training/`, `scripts/` | 파이썬 | 모델 학습, 기준 구현, PC 데모·평가 |
| `android/react-native-sonkkeut/` | Kotlin + TS | **앱에 들어가는 모듈.** 프론트는 이것만 설치하면 됨 → [사용법](android/react-native-sonkkeut/README.md) |

두 구현이 같은 결과를 내는지는 `bash android/parity/run_tests.sh`로 확인합니다(파이썬이 만든 정답과 Kotlin 출력 비교).

## 빠른 시작 (Windows)

### 앱 연결 검증 (2026-10-03)

`sonkkeut-ai.tar`에 포함된 M1·M2·M1-R ONNX 모델 3개를 Android 앱에 연결했습니다. 모델 SHA-256과 입출력 형식은 `android/react-native-sonkkeut/android/src/main/assets/sonkkeut/model-manifest.json`에 있습니다.

Python AI 로직 테스트 22개, ONNX 모델 3개의 CPU 추론, Android API 35의 ARM64 실행 환경에서 모델 3개 추론·한국어 OCR·전체 엔진 초기화(MediaPipe 포함)를 확인했습니다. 실제 휴대폰 카메라의 주문 전체 흐름과 기능 명세서의 현장 정확도·지연 목표는 아직 측정하지 않았습니다.

현재 앱 연결은 로컬에서 검증했고 프론트 변경은 프론트 Git 저장소에 업로드하지 않습니다.

1. `setup.bat` 더블클릭 — 패키지 설치, 손 관절 모델 내려받기, 테스트까지 한 번에 합니다.
   예전에 학습에 쓴 `Documents\sonkkeut\.venv`가 있으면 그대로 씁니다(PyTorch 재설치 없음).
2. `simulate.bat` — 카메라 없이 가짜 키오스크·가짜 손으로 전체 흐름을 돌리고 `sim_demo.mp4`를 만듭니다.
3. `run_camera.bat` — 웹캠으로 실시간 확인. 화면 요소를 **마우스로 클릭하면 목표 버튼**이 되고, 안내 문장을 스피커로 읽습니다.
4. `bench.bat` — 테스트, 단계별 처리 시간·꼭짓점 정확도, 누름 시나리오 40회 성공률을 수용 기준과 비교합니다.

명령줄로 직접 실행할 때:

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/simulate.py --out sim_demo.mp4
python scripts/run_camera.py --source 0 --tts
python scripts/bench.py --m1-data ../path/to/data/m1
python scripts/eval_scenarios.py --n 40
```

## 코드에서 쓰는 법

```python
from sonkkeut_vision import VisionPipeline

pipe = VisionPipeline("models/m1_screen_corners.pt", "models/m2_screen_elements.pt",
                      structure_fn=my_language_side)   # 노현석 모듈 (없으면 요소만 담긴 기본 구조)
while True:
    res = pipe.process(frame)                 # 매 프레임
    if res.keyframe:                          # 화면이 바뀌었을 때만
        pipe.set_target("e7", expect={"screen_type": "option"})   # F-07이 정한 목표
    if res.event and res.event.speak:         # 앱: 음성 출력
        tts(res.event.speak)
    vibrate(res.event.vibe_hz if res.event else 0)
    if res.verdict:                           # 누름 결과 → F-07이 다음 목표를 정함
        ...
```

## 폴더 구조

```
sonkkeut_vision/   영상 AI 패키지 (위 표)
  pipeline.py      매 프레임 흐름을 묶는 VisionPipeline
  corner_net.py    M1-R 꼭짓점 정밀 보정망 추론 (기본 사용)
  refine.py        화면 테두리 직선으로 꼭짓점 보정 (실험 기능, 기본 꺼짐)
  sim.py           가짜 카메라·가짜 손 (통합 테스트, 시연 영상용)
  viz.py           화면 표시
training/          합성 데이터 생성과 학습 (kiosk_synth, corner_synth, train, corner_refiner)
models/            학습된 가중치 (M1 꼭짓점, M1-R 보정, M2 화면 요소)
scripts/           run_camera, simulate, bench, eval_scenarios, eval_conditions, eval_real_hand,
                   export_mobile, demo_image
tests/             단위 테스트 + 실제 가중치로 하는 통합 테스트
docs/interface.md  언어 AI·앱과의 연결 규약
android/
  react-native-sonkkeut/  앱용 모듈 (Kotlin core = 파이썬 sonkkeut_vision 이식, ONNX INT8 모델 포함)
  parity/                 파이썬 ↔ Kotlin 동등성 테스트, Kotlin 파이프라인 시뮬레이션
```

## 설계에서 고민한 점

**왜 매 프레임 M1을 돌리지 않나** — 화면은 평평해서, 직전 프레임과의 움직임이 호모그래피 하나로 설명됩니다.
화면 안 특징점 수십 개를 광류로 따라가 RANSAC으로 그 호모그래피를 구하면 수 ms면 끝나고,
손가락이 화면을 가려도 가려진 점은 이상치로 빠집니다. M1은 30프레임마다, 또는 추적이 흔들릴 때만 다시 돌립니다.

**왜 펼친 화면에서 요소를 찾나** — 기울어진 영상에서는 같은 버튼도 각도마다 모양이 달라 데이터가 훨씬 많이 필요합니다.
펼친 화면에서는 모든 키오스크가 정면 사진처럼 보여 합성 데이터만으로도 잘 맞고,
결과 좌표(0~1)가 휴대폰이 움직여도 그대로 유지됩니다.

**왜 음성과 진동을 나눴나** — 시각 서보(visual servoing)를 사람 손에 적용한 것이라, 제어 신호를 두 채널로 나눴습니다.
음성은 정보가 많지만 느려서 0.8초에 한 번까지(방향), 진동은 즉각적이어서 매 프레임(거리)입니다.
오차는 '화면 높이 = 1' 단위로 계산합니다. 0~1 좌표를 그대로 쓰면 가로로 긴 화면에서 방향이 틀어집니다.

**잘못된 "지금 누르세요"를 막는 장치** — 버튼 가장자리 15%는 도달로 치지 않고, 0.3초 머물러야 하며,
버튼·손끝 신뢰도가 모두 기준 이상이어야 하고, 한 번 말하면 손끝이 버튼을 벗어났다 들어오기 전까지 반복하지 않습니다.

## 현재 성능과 한계

아래는 모두 **합성 키오스크 화면** 기준입니다. 실제 키오스크 실측치가 아니므로 기획서에는 '합성 검증'으로 표기해야 합니다.
(측정: 클라우드 CPU 2코어. `bench.py`, `eval_scenarios.py --n 40`, `eval_conditions.py --n 8`, `eval_real_hand.py`)

| 항목 | 결과 | 기능 명세서 기준 |
| --- | --- | --- |
| F-02 꼭짓점 오차 (네 꼭짓점 중 최대, 화면 폭 대비, 새 검증 이미지 144장) | **중앙값 0.74%**, 90% 3.0%, 2% 이내 83% (M1만: 4.5%) | 2% 이내 |
| F-02 매 프레임 갱신(추적) | 4.4 ms (95%: 5.0 ms) | 8 ms 이내 |
| F-03 화면 읽기(펼치기+M2, CPU) | 54 ms | (F-04 포함) 300 ms 이내 |
| F-03 화면 요소 재현율 (IoU 0.5) | 1.00 | — |
| 누름 시나리오 (가짜 손, 사용자 방향 ±15°·거리 ±30% 오차) | 40/40 성공, 잘못된 "누르세요" 0회, 평균 1.9초 | 평균 10초, 0회 |
| F-08 손끝 오차 (실제 손 사진 + 실제 MediaPipe) | 버튼 폭의 0.02배(90%: 0.07배) | 1/4 이내 |
| F-08 손끝 처리 (MediaPipe 포함, CPU) | 21 ms (95%: 28 ms) | 25 ms 이내 |
| 누름 시나리오 (실제 손 사진 + 실제 MediaPipe) | 13/16 성공, 잘못된 "누르세요" 0회, 평균 2.3초 | — |
| 모바일 모델 (ONNX INT8) | M2 mAP50-95 0.964→0.956, 모델 크기 1/2 | — |

촬영 조건별 (가짜 손, 8회씩)

| 조건 | 성공 | 잘못 누름 | 재현율 |
| --- | --- | --- | --- |
| 기본 / 흐림 / 반사광 / 어두움 / 손떨림 4배 / 원거리 | 각 8/8 | 0회 | 0.99~1.00 |
| 비스듬히 촬영(원근 2배) | 7/8 | 0회 | 0.88 |

비스듬한 촬영의 실패 1건은 화면 모서리가 프레임 밖으로 나간 경우로, 시스템이 "휴대폰을 조금 오른쪽 위로"라고
올바르게 안내했지만 가짜 사용자가 휴대폰을 움직이지 않아 끝까지 가지 못한 것입니다.

### 개선 과정에서 확인한 것

**꼭짓점 정확도 (4.8% → 0.74%)** — 학습 때 본 pose mAP50-95 0.995는 실제 정밀도를 반영하지 못했습니다.
Ultralytics는 4점 키포인트에 OKS sigma 0.25를 쓰는데 너무 느슨해서, 꼭짓점이 화면 폭의 5% 어긋나도 만점에 가깝게 나옵니다.
sigma를 조여 다시 학습하면(0.025, 0.08) 오히려 정확도가 떨어졌고, 테두리 직선 검출(refine.py)도 중앙값을 거의 바꾸지 못했습니다.
그래서 거친→정밀 2단계로 바꿨습니다. M1은 '대략 어디'만 맡고, 꼭짓점마다 주변을 확대한 64×64 패치를 작은 CNN(M1-R, 1.6MB)이
보고 정확한 위치까지의 오프셋을 2번 반복해 맞힙니다. 네 꼭짓점을 뒤집어 모두 '왼쪽 위' 모양으로 맞춰 한 모델로 처리합니다.
재검출 프레임에만 돌고(약 16 ms), 매 프레임 추적 비용은 그대로입니다. (`training/corner_refiner.py`)

**좌표계 흔들림** — M1을 다시 돌릴 때마다 기준 좌표가 몇 %씩 튀면 이미 읽어 둔 버튼 좌표가 실제 버튼과 어긋납니다(작은 +/− 버튼에서
잘못된 '누르세요' 발생). 재검출값이 추적값과 변 길이의 8% 이상 다를 때만 좌표계를 바꾸고, 바꿀 때는 버튼 좌표를 옮긴 뒤 화면을 다시 읽습니다.

**실제 손 사진 검증** — 클라우드에서 MediaPipe 공식 모델 서버에 접근할 수 없어, MediaPipe 0.10.14 패키지에 들어 있는
손바닥 검출·손 관절 모델로 같은 형식의 `.task` 파일을 만들어 썼습니다. 손 사진은 Ultralytics hand-keypoints 데이터셋에서
검지를 편 손만 골라 합성 화면 위에 붙였습니다.
처음에는 잘못된 '누르세요'가 2회 나왔는데, 원인을 따라가 보니 데이터셋의 사무실 배경 사진 상당수가 실제로는 엄지척인데
'검지를 편 손'으로 라벨이 붙어 있었습니다(정답 자체가 틀림). 사진을 직접 보고 단색 벽 앞 사진만 쓰도록 거른 뒤에는 0회입니다.
이와 별개로, 검지를 접은 손에서는 MediaPipe의 '검지 끝'이 실제 손끝이 아닌 접힌 마디에 찍힌다는 것도 확인해,
검지 길이 / 손바닥 길이 비율로 '가리키는 자세'를 판단하고 아니면 "검지 하나만 펴서 가리켜 주세요"라고 안내하도록 했습니다.

**남은 한계**
- 실제 손 시나리오의 실패 3건은 모두 손을 끝까지 찾지 못한 경우입니다. 손만 오려 붙인 합성 이미지라 팔이 없어
  손바닥 검출기가 놓치는 것으로 보이며(손 검출률 36/60), 실제 촬영에서 다시 확인해야 합니다.
- 실제 키오스크 사진과 실제 카메라로는 아직 검증하지 못했습니다. 클라우드에서 외부 사진을 내려받을 수 없었습니다.
- 휴대폰에서의 처리 시간은 아직 재지 않았습니다(위 시간은 PC CPU).

## 다음 할 일

- [ ] 실제 키오스크 사진 50~100장으로 M1·M2 실측 (기획서 수치는 합성 데이터 점수가 아니라 실측으로)
- [ ] PC에서 `run_camera.bat`으로 실제 손·실제 화면 확인 (모니터에 키오스크 화면을 띄워도 됨)
- [ ] 프론트 앱에 `react-native-sonkkeut` 설치 후 첫 빌드, 휴대폰에서 처리 시간 측정 (임현승·전채영과 함께)
- [ ] 노현석 모듈(F-04·F-05)과 연결 — 파이썬은 `structure_fn`, 앱은 Kotlin `SonkkeutEngine.structureProvider`
