import numpy as np

from sonkkeut_vision.keyframe import KeyframeDetector
from sonkkeut_vision.verify import PressVerifier


def S(t, **kw):
    return {"screen_type": t, "elements": [], **kw}


def test_timeout():
    v = PressVerifier()
    v.arm(0.0, S("menu"))
    assert v.poll(1.0, False) == "waiting"
    assert v.poll(1.6, False) == "timeout"
    assert v.timeout_verdict().result == "fail"


def test_expectations():
    v = PressVerifier()
    v.arm(0, S("menu"), {"screen_type": "option"})
    assert v.judge(S("option")).result == "success"
    v.arm(0, S("menu"), {"screen_type": "option"})
    assert v.judge(S("cart")).result == "fail"
    v.arm(0, S("option", cart_count=1), {"cart_delta": 1, "success_speak": "담겼습니다"})
    r = v.judge(S("menu", cart_count=2))
    assert r.result == "success" and r.speak == "담겼습니다"
    v.arm(0, S("menu"), {"screen_type": "option"})
    assert v.judge(S("unknown")).result == "uncertain"  # F-05가 화면 종류를 못 정함
    v.arm(0, S("cart"), {"screen_type": "payment"})
    assert v.judge(S("start")).result == "restarted"


def test_missing_required_evidence_is_never_success():
    v = PressVerifier()
    v.arm(0, S("option"), {"screen_type_not": "option", "cart_delta": 1})
    assert v.judge(S("menu")).result == "uncertain"
    v.arm(0, S("option", cart_count=0), {"screen_type_not": "option", "cart_delta": 1})
    assert v.judge(S("menu", cart_count=0)).result == "fail"
    v.arm(0, S("option", cart_count=0), {"screen_type_not": "option", "cart_delta": 1})
    assert v.judge(S("menu", cart_count=1)).result == "success"


def _screen(seed, h=200, w=120):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 230, np.uint8)
    for _ in range(8):
        y, x = rng.integers(0, h - 30), rng.integers(0, w - 30)
        img[y:y + 25, x:x + 25] = rng.integers(0, 200, 3)
    return img


def test_keyframe_on_change_after_stable():
    kf = KeyframeDetector()
    a, b = _screen(1), _screen(2)
    assert kf.update(a)[0]  # 첫 프레임
    assert not any(kf.update(a)[0] for _ in range(5))
    flags = [kf.update(b)[0] for _ in range(6)]
    assert flags.count(True) == 1 and flags.index(True) >= 2  # 바뀐 뒤 멈춰 있어야 키프레임


def test_hand_occlusion_is_ignored():
    kf = KeyframeDetector()
    a = _screen(1)
    kf.update(a)
    for i in range(10):
        b = a.copy()
        y = 60 + 5 * i
        b[y:y + 80, 40:80] = (150, 180, 225)  # 손가락이 화면 위를 지나감
        box = (40 / 120, y / 200, 80 / 120, (y + 80) / 200)
        assert not kf.update(b, [box])[0]
    assert not kf.changed_since_key(b, [box])
