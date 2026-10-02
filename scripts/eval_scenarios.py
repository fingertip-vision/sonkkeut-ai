"""
가짜 키오스크·가짜 사용자로 '목표 버튼 누르기'를 여러 번 반복해 성공률을 잰다.

  python scripts/eval_scenarios.py --n 40

화면 종류(메뉴·옵션·장바구니)와 목표 버튼(메뉴 카드, 작은 +/− 버튼 등)을 바꿔 가며,
가짜 사용자는 안내 방향을 ±15도, 이동 거리를 ±30% 틀리게 따라간다.
지표
  - 누름 성공률          : "지금 누르세요" → 화면 변화 → F-10 판정 'success'까지 간 비율
  - 잘못된 '누르세요'    : 그 순간 실제 손끝이 정답 버튼 밖이었던 횟수 (수용 기준 0회)
  - 버튼 도달 시간        : 목표 지정 → "지금 누르세요" (수용 기준: 평균 10초 이내)
  - F-03 재현율          : 정답 요소 중 IoU 0.5 이상으로 찾은 비율
주의: 합성 화면·가짜 손이므로 실제 성능이 아니라 '로직이 맞게 동작하는지'를 보는 지표다.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO  # noqa: E402

from sonkkeut_vision import VisionPipeline  # noqa: E402
from sonkkeut_vision.sim import FakeHandSource, FakeKioskCamera, render, run_scenario  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    m1 = YOLO(os.path.join(ROOT, "models", "m1_screen_corners.pt"))
    m2 = YOLO(os.path.join(ROOT, "models", "m2_screen_elements.pt"))
    rows = []
    for seed in range(a.start, a.start + a.n):
        kind = ["menu", "menu", "option", "cart"][seed % 4]
        screen, labels = render(kind, seed=seed)
        nxt, _ = render("payment", seed=seed + 1000)
        cam, hand = FakeKioskCamera(screen, seed=seed), FakeHandSource()
        pipe = VisionPipeline(m1, m2, hand_source=hand, device=a.device)
        log = run_scenario(pipe, cam, hand, nxt, labels, seed=seed)
        ok = bool(log["pressed_at"] is not None and log["verdict"] and log["verdict"][1] == "success")
        rows.append(dict(seed=seed, kind=kind, ok=ok, wrong=log["press_inside_gt"] is False,
                         reach=log["pressed_at"], recall=log["elem_recall"] or 0.0,
                         plane=float(np.median(log["plane_err"])) if log["plane_err"] else np.nan))
        r = rows[-1]
        print(f"seed {seed:3d} {kind:7s} {'성공' if ok else '실패'}  목표 {log['target']}  "
              f"도달 {r['reach'] if r['reach'] is None else round(r['reach'], 1)}s  재현율 {r['recall']:.2f}")
    n = len(rows)
    reach = [r["reach"] for r in rows if r["reach"] is not None]
    print("\n===== 요약 =====")
    print(f"누름 성공률         {sum(r['ok'] for r in rows)}/{n}")
    print(f"잘못된 '누르세요'    {sum(r['wrong'] for r in rows)}회 (기준 0회)")
    print(f"버튼 도달 시간       평균 {np.mean(reach):.1f}s, 최대 {np.max(reach):.1f}s (기준 평균 10초 이내)")
    print(f"F-03 재현율          평균 {np.mean([r['recall'] for r in rows]):.2f}")
    print(f"꼭짓점 오차(참고)    중앙값 {np.nanmedian([r['plane'] for r in rows]) * 100:.1f}% (화면 폭 대비)")


if __name__ == "__main__":
    main()
