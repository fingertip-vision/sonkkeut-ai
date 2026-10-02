"""
실제 M1·M2 가중치 + 가짜 카메라·가짜 손으로 전체 흐름을 끝까지 돌린다.

시나리오: 메뉴 화면 → 목표 지정 → 가짜 사용자가 음성 안내대로(조금씩 부정확하게) 손을 움직임
         → "지금 누르세요" → 키오스크 화면이 옵션 화면으로 바뀜 → 누름 결과 '성공'
"""
import os

import numpy as np
import pytest

from sonkkeut_vision import VisionPipeline
from sonkkeut_vision.sim import FakeHandSource, FakeKioskCamera, render, run_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1 = os.path.join(ROOT, "models", "m1_screen_corners.pt")
M2 = os.path.join(ROOT, "models", "m2_screen_elements.pt")
pytestmark = pytest.mark.skipif(not (os.path.exists(M1) and os.path.exists(M2)), reason="가중치 없음")


@pytest.mark.parametrize("seed", [3, 11, 21])
def test_end_to_end(seed):
    screen, labels = render("menu", seed=seed)
    nxt, _ = render("option", seed=seed + 100)
    cam, hand = FakeKioskCamera(screen, seed=seed), FakeHandSource()
    pipe = VisionPipeline(M1, M2, hand_source=hand)
    log = run_scenario(pipe, cam, hand, nxt, labels, seed=seed)
    assert log["pressed_at"] is not None, log["events"][-10:]
    assert log["press_inside_gt"], "버튼 밖에서 누르라고 함"
    assert log["verdict"] and log["verdict"][1] == "success", log["verdict"]
    assert log["elem_recall"] >= 0.8
    speaks = [e[0] for e in log["events"] if e[4]]
    assert all(b - a >= 0.8 - 1e-6 for a, b in zip(speaks, speaks[1:]))
    print(f"seed {seed}: target {log['target']}, press {log['pressed_at']:.1f}s, "
          f"recall {log['elem_recall']:.2f}, plane err {np.median(log['plane_err']) * 100:.2f}%")
