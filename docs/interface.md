# 영상 AI ↔ 언어 AI ↔ 앱 연결 규약

손끝길 AI는 둘이 나눠 만든다. 서로의 내부를 몰라도 되도록, 아래 세 지점에서만 데이터를 주고받는다.
좌표는 모두 **펼친 화면 기준 0~1 비율**이다(x: 왼쪽 0 → 오른쪽 1, y: 위 0 → 아래 1).
카메라 픽셀 좌표는 영상 AI 밖으로 나가지 않는다.

| 담당 | 기능 | 코드 |
| --- | --- | --- |
| 김우주 (영상) | F-02 화면 평면, F-03 화면 요소, F-08 손끝, F-09 오차 계산, F-10 누름 결과 확인 | `sonkkeut_vision/` |
| 노현석 (언어) | F-04 문자 인식, F-05 화면 구조화, F-06 음성 주문 이해, F-07 버튼 순서 계획 | (노현석 모듈) |
| 임현승 (앱) | F-09 음성·진동 출력 | 안드로이드 앱 |

## ① 영상 → 언어: 키프레임마다 화면 요소 전달

화면이 바뀌어 멈춘 순간(키프레임)에만 한 번 호출된다. 매 프레임 호출되지 않는다.

```python
def structure_fn(elements, crops, flat, keyframe_id) -> dict:
    """
    elements    : list[Element]  F-03 결과 (id, kind, box, conf, parent)
    crops       : dict[str, ndarray]  요소 id → 잘라낸 BGR 이미지 (OCR 입력)
    flat        : ndarray  정면으로 펼친 화면 전체 (BGR, 긴 변 960px)
    keyframe_id : int
    return      : 화면 구조 dict (기능 명세서 6장)
    """
```

`elements` 예시 (`Element.to_dict()`):

```json
[
  {"id": "e1", "kind": "tab",  "box": [0.01, 0.09, 0.25, 0.15], "conf": 0.97},
  {"id": "e7", "kind": "menu", "box": [0.05, 0.25, 0.30, 0.45], "conf": 0.95},
  {"id": "e8", "kind": "price","box": [0.10, 0.40, 0.25, 0.44], "conf": 0.90, "parent": "e7"}
]
```

- `id`는 읽는 순서(위→아래, 왼쪽→오른쪽)로 매긴다. **키프레임마다 새로 매겨진다.**
- `parent`: 가격 박스가 메뉴 카드 안에 있으면 그 메뉴 id. 메뉴명–가격 짝짓기에 쓰면 된다.
- `kind`: `tab`, `menu`, `price`, `button`, `back`
- 꼭짓점이 조금 안쪽으로 잡혀도 가장자리 버튼이 잘리지 않게 화면 바깥 4%까지 함께 보므로,
  가장자리 요소의 `box`는 0보다 약간 작거나 1보다 약간 클 수 있다.

돌려줄 화면 구조(6장)에는 F-04·F-05 결과를 채운다. 영상 쪽이 쓰는 키는 아래뿐이다.

```json
{
  "screen_type": "menu",          // menu | option | cart | payment | start | unknown
  "keyframe_id": 12,
  "elements": [{"id": "e7", "kind": "menu", "text": "아메리카노", "price": 4500, "box": [...], "conf": 0.95}],
  "cart_count": 2,                // 선택. 장바구니 개수를 알면 넣기 (누름 결과 확인에 사용)
  "selected": ["e12"]             // 선택. 선택 상태로 보이는 옵션 id 목록
}
```

F-05가 준비되기 전에는 `elements_to_structure()`가 `screen_type: "unknown"`, 텍스트 없이 대신 돌려준다.

## ② 언어 → 영상: 목표 버튼 지정

F-07이 다음에 누를 버튼을 정하면 호출한다.

```python
pipe.set_target("e7", expect={"screen_type": "option", "success_speak": "옵션 화면이 열렸습니다"})
```

`expect`(누른 뒤 기대 결과) 형식은 `sonkkeut_vision/verify.py` 머리말에 있다.
`screen_type`, `screen_type_not`, `cart_delta`, `selected`, `changed`를 조합할 수 있다.

영상 쪽은 결과를 `FrameResult`로 돌려준다.

| 필드 | 언제 | 언어 쪽이 할 일 |
| --- | --- | --- |
| `verdict.result == "success"` | 누름 성공 | 다음 목표 버튼을 `set_target` |
| `verdict.result == "fail"` | 안 눌림 / 다른 화면 | 같은 버튼 재시도(영상 쪽이 자동 재안내) 또는 뒤로가기 계획 |
| `verdict.result == "uncertain"` | 화면 종류를 확인할 수 없음 | 화면 구조를 보고 직접 판단 |
| `verdict.result == "restarted"` | 키오스크가 첫 화면으로 돌아감 | 진행 상태 유지한 채 처음부터 계획 |
| `target_missing == True` | 새 화면에 목표 버튼이 없음 | 다시 계획 |

## ③ 영상 → 앱: 안내 이벤트 (매 프레임)

`FrameResult.event.to_dict()` — 기능 명세서 6장 형식 그대로다.

```json
{"type": "direction", "target_id": "e7", "dir": "up_right", "distance": "near",
 "speak": "오른쪽 위로 조금", "vibe_hz": 5.6}
```

| type | 의미 | 앱 동작 |
| --- | --- | --- |
| `direction` | 방향 안내 | `speak`가 있으면 읽고, `vibe_hz`로 진동 주기 갱신 |
| `press` | 버튼 위 0.3초 유지 | "지금 누르세요" 읽기, 강한 진동 |
| `hold` | 목표 버튼 인식이 불확실 | "잠시 멈춰 주세요" |
| `reset` | 3초 동안 오차가 커짐 | "화면 가운데에서 다시 시작해 주세요" |
| `no_hand` | 손이 1초 넘게 안 보임 | "검지를 화면 앞으로 가져와 주세요" |
| `point` | 손은 보이지만 검지를 접고 있음 (검지 끝 위치를 믿을 수 없음) | "검지 하나만 펴서 가리켜 주세요" |

- `speak`는 이미 **0.8초에 한 번 이하**로 걸러져 있다. 앱은 `speak`가 있을 때만 읽으면 된다.
  (같은 문장 반복은 2.5초 간격)
- `vibe_hz`는 매 프레임 온다. 멀리 2Hz → 가까이 4~8Hz → 버튼 위 10Hz.
- `FrameResult.hint`(예: "휴대폰을 조금 왼쪽으로")는 화면을 못 찾았을 때 휴대폰 위치 안내다.

## ④ 모바일 모델 파일 (앱 탑재용)

앱 탑재용 ONNX 모델은 `android/react-native-sonkkeut/android/src/main/assets/sonkkeut/` 에 있다(`scripts/export_mobile.py`가 만든다).
앱은 이 모델을 직접 다룰 필요 없이 `react-native-sonkkeut` 모듈을 쓰면 된다(→ `android/react-native-sonkkeut/README.md`). 아래는 참고용이다.

| 파일 | 입력 | 출력 | 후처리 |
| --- | --- | --- | --- |
| `m1_screen_corners_int8.onnx` | `images` 1×3×640×640, RGB, 0~1, 레터박스(회색 114) | 1×17×8400 | 박스 4 + 점수 1 + 꼭짓점 4×(x,y,conf) → 점수 최대 1개 |
| `m2_screen_elements_int8.onnx` | 펼친 화면을 같은 방식으로 | 1×9×8400 | 박스 4 + 클래스 5 → NMS(IoU 0.5) |
| `m1r_corner_refiner.onnx` | `patches` N×1×64×64, 흑백 0~1 | N×2 오프셋 | `training/corner_refiner.py`의 crop_patch / offset_to_image 참고 |

손 관절은 MediaPipe Hand Landmarker 안드로이드 SDK(`com.google.mediapipe:tasks-vision`)를 그대로 쓴다.
