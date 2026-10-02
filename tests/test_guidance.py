from sonkkeut_vision.guidance import Guide, josa_ro, phrase
from sonkkeut_vision.schema import Element, Fingertip

BTN = Element("e7", "menu", (0.40, 0.40, 0.60, 0.50), 0.9)


def tip(x, y, conf=0.9):
    return Fingertip(pos=(x, y), conf=conf)


def make(aspect=1.0):
    g = Guide()
    g.set_target(BTN, aspect)
    return g


def test_josa():
    assert josa_ro("오른쪽") == "으로" and josa_ro("위") == "로" and josa_ro("아래") == "로"
    assert phrase("up_right", "near") == "오른쪽 위로 조금"
    assert phrase("left", "far") == "왼쪽으로"


def test_eight_directions():
    cases = {(0.1, 0.45): "right", (0.9, 0.45): "left", (0.5, 0.95): "up", (0.5, 0.05): "down",
             (0.1, 0.9): "up_right", (0.9, 0.9): "up_left", (0.1, 0.05): "down_right", (0.9, 0.05): "down_left"}
    for (x, y), want in cases.items():
        ev = make().update(tip(x, y), t=0.0)
        assert ev.type == "direction" and ev.dir == want, ((x, y), ev)


def test_aspect_changes_direction():
    # 가로로 긴 화면(16:9)에서는 0~1 좌표로 같은 거리라도 가로 실제 거리가 1.78배
    # dx=0.1, dy=0.3: 정사각형 화면이면 72°(위), 16:9면 dx가 0.178이 되어 59°(오른쪽 위)
    g = make(aspect=1.0)
    assert g.update(tip(0.4, 0.75), 0.0).dir == "up"
    g = make(aspect=16 / 9)
    assert g.update(tip(0.4, 0.75), 0.0).dir == "up_right"


def test_distance_zones_and_vibration():
    g = make()
    far = g.update(tip(0.5, 0.95), 0.0)
    g = make()
    near = g.update(tip(0.5, 0.53), 0.0)
    g = make()
    reach = g.update(tip(0.5, 0.45), 0.0)
    assert (far.distance, near.distance, reach.distance) == ("far", "near", "reach")
    assert far.vibe_hz < near.vibe_hz < reach.vibe_hz


def test_speech_throttle():
    g = make()
    spoken = []
    t = 0.0
    for i in range(30):  # 2초 동안 15FPS로 방향이 번갈아 바뀌어도
        x = 0.1 if i % 2 else 0.9
        ev = g.update(tip(x, 0.45), t)
        if ev.speak:
            spoken.append(t)
        t += 1 / 15
    gaps = [b - a for a, b in zip(spoken, spoken[1:])]
    assert all(gp >= 0.8 - 1e-9 for gp in gaps), gaps


def test_same_phrase_repeats_slowly():
    g = make()
    spoken = [t / 15 for t in range(75) if g.update(tip(0.5, 0.95), t / 15).speak]
    assert len(spoken) == 2  # 0초, 2.5초 이후(5초 안에서는 2번만)


def test_press_requires_dwell_and_core():
    g = make()
    # 가장자리(박스 안이지만 안쪽 85% 밖)에서는 아무리 머물러도 누르라고 하지 않는다
    for i in range(20):
        ev = g.update(tip(0.405, 0.45), i / 15)
        assert ev.type != "press"
    g = make()
    types = [g.update(tip(0.5, 0.45), i / 15).type for i in range(10)]
    first = types.index("press")
    assert first / 15 >= 0.3 - 1e-9  # 0.3초 머문 뒤에야
    assert types.count("press") == 1  # 한 번만


def test_press_rearms_after_leaving():
    g = make()
    seq = [(0.5, 0.45)] * 8 + [(0.5, 0.8)] * 3 + [(0.5, 0.45)] * 8
    types = [g.update(tip(x, y), i / 15).type for i, (x, y) in enumerate(seq)]
    assert types.count("press") == 2


def test_low_conf_blocks_press():
    g = make()
    types = [g.update(tip(0.5, 0.45, conf=0.3), i / 15).type for i in range(15)]
    assert "press" not in types
    g = make()
    evs = [g.update(tip(0.5, 0.45), i / 15, target_conf=0.2) for i in range(15)]
    assert all(e.type == "hold" for e in evs) and evs[0].speak == "잠시 멈춰 주세요"


def test_no_hand_after_one_second():
    g = make()
    assert g.update(Fingertip(pos=None, lost_for=0.5), 0.0) is None
    ev = g.update(Fingertip(pos=None, lost_for=1.0), 0.1)
    assert ev.type == "no_hand" and "검지" in ev.speak


def test_diverging_triggers_reset():
    g = make()
    ev = None
    for i in range(50):  # 3초 넘게 버튼에서 점점 멀어짐
        y = 0.55 + 0.008 * i
        ev = g.update(tip(0.5, min(y, 1.0)), i / 15)
        if ev.type == "reset":
            break
    assert ev.type == "reset" and "가운데" in ev.speak


def test_event_json_shape():
    d = make().update(tip(0.3, 0.6), 0.0).to_dict()
    assert set(d) >= {"type", "target_id", "dir", "distance", "speak", "vibe_hz"}
    assert d["target_id"] == "e7"


def test_folded_finger_blocks_guidance():
    g = make()
    evs = [g.update(Fingertip(pos=(0.5, 0.45), conf=0.2, pointing=0.2), i / 15) for i in range(10)]
    assert all(e.type == "point" for e in evs) and "검지" in evs[0].speak


def test_pointing_score():
    import numpy as np

    from sonkkeut_vision.fingertip import pointing_score

    pts = np.zeros((21, 2), np.float32)
    pts[0], pts[9], pts[5] = (0, 100), (0, 0), (-20, 5)
    pts[8] = (-20, -95)  # 검지를 손바닥 길이만큼 폄
    assert pointing_score(pts) == 1.0
    pts[8] = (-20, -40)  # 반쯤 접음
    assert pointing_score(pts) == 0.0
