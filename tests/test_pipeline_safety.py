"""
잘못된 "지금 누르세요"·거짓 성공을 막는 파이프라인 연결부 테스트 (코드 검토 A1~A5)

부품 하나씩은 test_guidance·test_verify_keyframe이 보지만, 아래 문제는 부품 사이에서 생긴다.
  A1 다시 읽은 화면에서 같은 자리·같은 종류의 '다른 글자' 버튼으로 목표가 바뀜
  A2 강제로 다시 읽은 키프레임을 누름 성공으로 판정
  A3 다시 읽기 전에 고른 요소 id로 목표 지정
  A4 추적 없이 새로 검출한 좌표계로 이전 프레임 좌표를 옮김 / 화면을 놓쳤다 찾은 뒤 좌표를 그대로 씀
  A5 손끝 신뢰도가 낮은 프레임까지 머무름 시간에 넣음
모델 없이 돌도록 평면·요소 검출·손끝만 가짜로 두고, 키프레임 판단·유도·판정은 실제 코드를 쓴다.
"""
import numpy as np
import pytest

from sonkkeut_vision.guidance import Guide
from sonkkeut_vision.keyframe import KeyframeDetector
from sonkkeut_vision.pipeline import VisionPipeline
from sonkkeut_vision.plane import ScreenPlaneEstimator, homography_from_corners
from sonkkeut_vision.schema import Element, Fingertip, PlaneState, elements_to_structure
from sonkkeut_vision.verify import PressVerifier

W, H = 320, 480
CORNERS = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
AMERICANO = Element("e1", "menu", (0.10, 0.10, 0.90, 0.40), 0.9, text="아메리카노")
LATTE_SAME_SPOT = Element("e1", "menu", (0.12, 0.10, 0.88, 0.42), 0.9, text="카페라떼")
INSIDE = (0.5, 0.25)  # AMERICANO 안쪽(가장자리 15% 제외 영역)


def screen(seed):
    """화면 내용. 키프레임 판단기는 화면을 작게 줄여 비교하므로 큰 블록 무늬로 만든다.
    씨앗이 다르면 '다른 화면'으로 본다."""
    blocks = np.random.default_rng(seed).integers(0, 255, (12, 8, 3), dtype=np.uint8)
    return np.repeat(np.repeat(blocks, H // 12, axis=0), W // 8, axis=1)


def plane(**kw):
    Hm, aspect = homography_from_corners(CORNERS)
    return PlaneState(CORNERS.copy(), Hm, 0.95, aspect, "track", **kw)


class FakePlane:
    def __init__(self):
        self.next = None
        self.state = None
        self.last_timing = {}

    def update(self, frame, t):
        self.state, self.next = self.next or plane(), None
        return self.state, None


class FakeDetector:
    margin = 0.0
    last_timing = {}

    def __init__(self, elements):
        self.elements = elements

    def detect(self, frame, plane):
        els = [Element(e.id, e.kind, e.box, e.conf, text=e.text) for e in self.elements]
        return els, np.zeros((96, 64, 3), np.uint8)


class FakeTracker:
    last_timing = {}

    def __init__(self):
        self.tip = None

    def update(self, frame, plane, t):
        return self.tip

    def hand_box_screen(self, plane):
        return None


def make_pipe(elements):
    # 생성자는 실제 모델(M1·M2·MediaPipe)을 불러오므로 거치지 않고 부품을 직접 넣는다.
    p = VisionPipeline.__new__(VisionPipeline)
    p.plane_est, p.elem_det, p.tracker = FakePlane(), FakeDetector(elements), FakeTracker()
    p.kf, p.guide, p.verifier = KeyframeDetector(), Guide(), PressVerifier()
    p.structure_fn = lambda els, crops, flat, kid: elements_to_structure(els, kid)
    p.elements, p.structure, p.flat = [], None, None
    p.elements_keyframe_id = 0
    p.expect = None
    p._force_kf = False
    p._last_forced = -1e9
    return p


def run(p, frame, t0, seconds, fps=15):
    """프레임을 돌리며 결과를 모은다"""
    out = []
    for i in range(int(seconds * fps)):
        out.append(p.process(frame, t0 + i / fps))
    return out


def presses(results):
    return [r for r in results if r.event is not None and r.event.type == "press"]


def test_A1_same_spot_different_text_is_not_the_target():
    p = make_pipe([AMERICANO])
    p.process(screen(1), 0.0)  # 첫 키프레임: 아메리카노 화면
    p.set_target("e1")

    # 탭이 넘어가 같은 자리에 카페라떼가 나타남. 손끝은 아직 멀리 있음.
    p.elem_det.elements = [LATTE_SAME_SPOT]
    p.tracker.tip = Fingertip(pos=(0.5, 0.9), conf=0.9)
    res = run(p, screen(2), 1.0, 1.0)
    assert any(r.keyframe for r in res)
    assert any(r.target_missing for r in res)
    assert p.guide.target.text == "아메리카노" and p.guide.target.conf == 0.0

    # 카페라떼 위에 손끝을 올려도 "지금 누르세요"가 나오면 안 된다
    p.tracker.tip = Fingertip(pos=INSIDE, conf=0.9)
    assert presses(run(p, screen(2), 2.0, 2.0)) == []


def test_A1_same_text_keeps_guiding():
    p = make_pipe([AMERICANO])
    p.process(screen(1), 0.0)
    p.set_target("e1")
    # 같은 버튼을 다시 읽음(좌표만 조금 다름) → 목표 유지
    p.elem_det.elements = [Element("e3", "menu", (0.11, 0.10, 0.89, 0.41), 0.9, text="아메 리카노")]
    res = run(p, screen(2), 1.0, 1.0)
    assert not any(r.target_missing for r in res)
    assert p.guide.target.id == "e3" and p.guide.target.conf == 0.9


def test_A2_forced_keyframe_while_waiting_is_not_success():
    p = make_pipe([AMERICANO])
    p.process(screen(1), 0.0)
    p.set_target("e1")
    p.tracker.tip = Fingertip(pos=INSIDE, conf=0.9)
    pressed = presses(run(p, screen(1), 0.1, 1.0))
    assert len(pressed) == 1 and p.verifier.armed

    # 화면은 그대로인데 다시 읽기를 강제(좌표계 교체·앱 요청과 같은 상황)
    p.request_keyframe()
    res = run(p, screen(1), 1.2, 2.0)
    verdicts = [r.verdict for r in res if r.verdict is not None]
    assert res[0].keyframe and res[0].verdict is None  # 강제 키프레임은 판정하지 않음
    # 1.5초 뒤 '눌리지 않음' → 손끝이 그대로면 다시 누르라고 안내하므로 판정이 더 나올 수 있다. 성공만 없으면 된다.
    assert verdicts and verdicts[0].reason == "no_change" and all(v.result != "success" for v in verdicts)


def test_A2_real_screen_change_is_still_judged():
    p = make_pipe([AMERICANO])
    p.process(screen(1), 0.0)
    p.set_target("e1", expect={"changed": True})
    p.tracker.tip = Fingertip(pos=INSIDE, conf=0.9)
    assert len(presses(run(p, screen(1), 0.1, 1.0))) == 1
    res = run(p, screen(3), 1.2, 1.0)  # 실제로 화면이 바뀜
    assert [r.verdict.result for r in res if r.verdict is not None] == ["success"]


def test_A3_set_target_rejects_id_from_an_older_keyframe():
    p = make_pipe([AMERICANO])
    first = p.process(screen(1), 0.0).keyframe_id
    p.elem_det.elements = [LATTE_SAME_SPOT]
    run(p, screen(2), 1.0, 1.0)  # 앱이 확인을 기다리는 사이 화면을 다시 읽음
    assert p.elements_keyframe_id != first
    with pytest.raises(KeyError):
        p.set_target("e1", keyframe_id=first)
    p.set_target("e1", keyframe_id=p.elements_keyframe_id)
    assert p.guide.target.text == "카페라떼"


def test_A4_rebased_plane_rereads_instead_of_moving_coordinates():
    p = make_pipe([AMERICANO])
    p.process(screen(1), 0.0)
    p.set_target("e1")

    # 추적이 끊겨 새로 검출. 실제 버튼은 화면 아래쪽으로 0.3 내려가 보이게 된 상황
    p.elem_det.elements = [Element("e1", "menu", (0.10, 0.40, 0.90, 0.70), 0.9, text="아메리카노")]
    p.plane_est.next = plane(rebased=True)
    res = p.process(screen(1), 1.0)
    assert res.keyframe  # 좌표를 옮기지 않고 그 자리에서 다시 읽음
    assert res.target_missing and p.guide.target.conf == 0.0  # 옛 좌표로 누르라고 하지 않음

    p.tracker.tip = Fingertip(pos=INSIDE, conf=0.9)
    assert presses(run(p, screen(1), 1.1, 2.0)) == []


def test_A5_low_confidence_frames_do_not_count_toward_dwell():
    g = Guide()
    g.set_target(AMERICANO, 1.0)
    t = 0.0
    for _ in range(8):  # 0.5초 동안 신뢰도 0.3
        assert g.update(Fingertip(pos=INSIDE, conf=0.3), t).type != "press"
        t += 1 / 15
    first_confident = t
    ev = g.update(Fingertip(pos=INSIDE, conf=0.9), t)  # 확실한 프레임 한 번
    assert ev.type != "press"
    press_at = None
    for _ in range(8):  # 확실한 프레임으로 0.3초 넘게 머물면 그때 누르라고 한다
        t += 1 / 15
        if g.update(Fingertip(pos=INSIDE, conf=0.9), t).type == "press" and press_at is None:
            press_at = t
    assert press_at is not None and press_at - first_confident >= 0.3


class ScriptedM1:
    """ScreenPlaneEstimator.detect 대신 정해 둔 꼭짓점을 돌려준다"""

    def __init__(self):
        self.next = None

    def __call__(self, frame):
        return self.next


# 꼭짓점이 영상 가장자리에 붙으면 추정기가 '화면이 잘림'으로 보고 검출을 버리므로 안쪽에 둔다
QUAD = np.float32([[30, 40], [W - 30, 40], [W - 30, H - 40], [30, H - 40]])


def make_estimator():
    est = ScreenPlaneEstimator(object(), refiner=None)
    est.detect = ScriptedM1()
    return est


def test_A4_tracking_failure_redetect_is_rebased_not_reframed():
    est = make_estimator()
    est.detect.next = (QUAD.copy(), 0.9, np.ones(4, np.float32))
    first, _ = est.update(screen(1), 0.0)
    assert first is not None

    # 특징점이 없는 프레임 → 추적 실패, 같은 프레임에서 꼭짓점만 3% 옮겨 검출됨
    est.detect.next = (QUAD + np.float32([W * 0.03, 0]), 0.9, np.ones(4, np.float32))
    state, _ = est.update(np.zeros((H, W, 3), np.uint8), 0.1)
    assert est.last_timing["plane_mode"] == "detect"
    assert state.rebased and not state.reframed and state.ref_prev is None


def test_A4_found_again_after_loss_is_rebased():
    est = make_estimator()
    est.detect.next = (QUAD.copy(), 0.9, np.ones(4, np.float32))
    est.update(screen(1), 0.0)
    est.detect.next = None
    lost, _ = est.update(np.zeros((H, W, 3), np.uint8), 0.1)
    assert lost is None
    est.detect.next = (QUAD + np.float32([W * 0.03, 0]), 0.9, np.ones(4, np.float32))
    state, _ = est.update(screen(1), 0.2)
    assert state.rebased
