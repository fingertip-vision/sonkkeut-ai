"""
촬영 조건별 견고성 평가: 흐림·반사광·어두움·손떨림·원거리·비스듬한 각도에서
누르기 시나리오 성공률과 잘못된 '누르세요' 횟수를 잰다.

  python scripts/eval_conditions.py --n 12
주의: 합성 화면·가짜 손 기준. 실제 조건에서의 성능은 실촬영으로 다시 확인해야 한다.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO  # noqa: E402

from sonkkeut_vision import VisionPipeline  # noqa: E402
from sonkkeut_vision.sim import CONDITIONS, FakeHandSource, FakeKioskCamera, render, run_scenario  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conditions", default=",".join(str(k) for k in CONDITIONS))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    m1 = YOLO(os.path.join(ROOT, "models", "m1_screen_corners.pt"))
    m2 = YOLO(os.path.join(ROOT, "models", "m2_screen_elements.pt"))
    print(f"{'조건':<34}{'성공':>6}{'잘못 누름':>10}{'평균 도달':>10}{'재현율':>8}")
    for cond in a.conditions.split(","):
        cond = None if cond in ("None", "") else cond
        ok = wrong = 0
        reach, rec = [], []
        for seed in range(a.n):
            kind = ["menu", "menu", "option", "cart"][seed % 4]
            screen, labels = render(kind, seed=seed)
            nxt, _ = render("payment", seed=seed + 1000)
            cam, hand = FakeKioskCamera(screen, seed=seed, condition=cond), FakeHandSource()
            pipe = VisionPipeline(m1, m2, hand_source=hand, device=a.device)
            log = run_scenario(pipe, cam, hand, nxt, labels, seed=seed)
            ok += bool(log["pressed_at"] is not None and log["verdict"] and log["verdict"][1] == "success")
            wrong += log["press_inside_gt"] is False
            if log["pressed_at"] is not None:
                reach.append(log["pressed_at"])
            rec.append(log["elem_recall"] or 0.0)
        r = f"{np.mean(reach):.1f}s" if reach else "-"
        print(f"{CONDITIONS[cond]:<34}{ok:>4}/{a.n}{wrong:>8}회{r:>10}{np.mean(rec):>8.2f}", flush=True)


if __name__ == "__main__":
    main()
