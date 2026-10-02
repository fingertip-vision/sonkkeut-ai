"""
카메라·키오스크 없이 가짜 화면과 가짜 손으로 전체 흐름을 돌리고 결과 영상을 만든다.
발표 자료용 시연 영상이나 기능 확인에 쓴다.

  python scripts/simulate.py --out sim_demo.mp4 --seed 3 --screen menu

가짜 사용자는 방향 안내를 (조금 부정확하게) 따라 손끝을 움직이고, "지금 누르세요"가 나오면 누른다.
누르면 키오스크 화면이 바뀌고, 누름 결과 확인(F-10)이 '성공'을 내야 한다.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonkkeut_vision import VisionPipeline  # noqa: E402
from sonkkeut_vision.sim import FakeHandSource, FakeKioskCamera, render, run_scenario  # noqa: E402
from sonkkeut_vision.viz import draw_frame  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_demo.mp4")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--screen", default="menu", choices=["menu", "option", "cart"])
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    screen, labels = render(a.screen, seed=a.seed)
    nxt, _ = render("option" if a.screen == "menu" else "payment", seed=a.seed + 100)
    cam, hand = FakeKioskCamera(screen, seed=a.seed), FakeHandSource()
    pipe = VisionPipeline(os.path.join(ROOT, "models", "m1_screen_corners.pt"),
                          os.path.join(ROOT, "models", "m2_screen_elements.pt"), hand_source=hand, device=a.device)
    st = {"writer": None, "speak": None, "frames": 0}

    def on_frame(img, r, p):
        for msg in [r.hint, r.event.speak if r.event else None, r.verdict.speak if r.verdict else None]:
            if msg:
                st["speak"] = msg
                print(f"{st['frames'] / a.fps:5.2f}s 음성: {msg}")
        tid = p.guide.target.id if p.guide.target else None
        vis = draw_frame(img, r, p.elements, tid, st["speak"])
        vis = cv2.resize(vis, (vis.shape[1] * 2 // 3, vis.shape[0] * 2 // 3))
        if st["writer"] is None:
            st["writer"] = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (vis.shape[1], vis.shape[0]))
        st["writer"].write(vis)
        st["last"] = vis
        st["frames"] += 1

    log = run_scenario(pipe, cam, hand, nxt, labels, seed=a.seed, fps=a.fps, on_frame=on_frame)
    for _ in range(a.fps * 2):  # 결과 화면을 2초 더 보여 준다
        st["writer"].write(st["last"])
    st["writer"].release()
    with open(os.path.splitext(a.out)[0] + "_events.jsonl", "w", encoding="utf-8") as f:
        for e in log["events"]:
            f.write(json.dumps(dict(zip(["t", "type", "dir", "distance", "speak"], e)), ensure_ascii=False) + "\n")
    print(f"목표 {log['target']}, 누름 {log['pressed_at']}, 결과 {log['verdict']}")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
