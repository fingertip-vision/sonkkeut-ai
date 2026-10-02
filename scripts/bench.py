"""
단계별 처리 시간과 화면 꼭짓점 정확도를 기능 명세서 수용 기준과 비교한다.

  python scripts/bench.py                # 기본: GPU 있으면 GPU
  python scripts/bench.py --device cpu

수용 기준 (기능 명세서)
  F-02 꼭짓점 오차 화면 폭의 2% 이내, 매 프레임 갱신 8 ms 이내
  F-03 (+F-04) 300 ms 이내
  F-08 손끝 처리 25 ms 이내
주의: PC에서 잰 값이다. 휴대폰(TFLite INT8)에서는 따로 재야 한다.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonkkeut_vision import VisionPipeline  # noqa: E402
from sonkkeut_vision.refine import refine_corners  # noqa: E402
from sonkkeut_vision.sim import FakeHandSource, FakeKioskCamera, render  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corner_accuracy(m1, data_dir, n=200, device=None):
    """corner_synth로 만든 검증 세트에서 꼭짓점 오차(화면 폭 대비, 네 꼭짓점 중 최대)"""
    from ultralytics import YOLO
    model = YOLO(m1)
    from sonkkeut_vision.corner_net import DEFAULT_REFINER, CornerRefiner

    net = CornerRefiner(DEFAULT_REFINER) if os.path.exists(DEFAULT_REFINER) else None
    raw, ref, rnet = [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "images", "val", "*.jpg")))[:n]:
        v = np.array(open(f.replace("images", "labels").replace(".jpg", ".txt")).read().split()[5:], float)
        v = v.reshape(4, 3)
        if (v[:, 2] == 0).any():
            continue
        img = cv2.imread(f)
        h, w = img.shape[:2]
        gt = v[:, :2] * [w, h]
        r = model.predict(img, verbose=False, device=device)[0]
        if len(r.boxes) == 0:
            raw.append(np.inf)
            ref.append(np.inf)
            rnet.append(np.inf)
            continue
        kp = r.keypoints.xy[0].cpu().numpy()
        sw = (np.linalg.norm(gt[1] - gt[0]) + np.linalg.norm(gt[2] - gt[3])) / 2
        raw.append(np.linalg.norm(kp - gt, axis=1).max() / sw)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rk, _ = refine_corners(gray, kp)
        ref.append(np.linalg.norm(rk - gt, axis=1).max() / sw)
        if net is not None:
            rnet.append(np.linalg.norm(net(gray, kp) - gt, axis=1).max() / sw)
    return np.array(raw), np.array(ref), np.array(rnet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--m1", default=os.path.join(ROOT, "models", "m1_screen_corners.pt"))
    ap.add_argument("--m2", default=os.path.join(ROOT, "models", "m2_screen_elements.pt"))
    ap.add_argument("--m1-data", default=None, help="corner_synth 출력 폴더(data/m1). 주면 꼭짓점 정확도도 잰다")
    a = ap.parse_args()

    screen, _ = render("menu", seed=5)
    cam, hand = FakeKioskCamera(screen, seed=5), FakeHandSource()
    pipe = VisionPipeline(a.m1, a.m2, hand_source=hand, device=a.device)
    rows = []
    for i in range(a.frames):
        t = i / 15
        img, info = cam.frame(t, (0.3 + 0.4 * (i % 30) / 30, 0.6))
        hand.current = info["hand"]
        if i % 50 == 25:
            pipe.request_keyframe()
        rows.append(pipe.process(img, t).timings)
    rows = rows[5:]  # 모델 준비(warm-up) 프레임 제외

    def stat(key, cond=lambda r: True):
        v = [r[key] for r in rows if key in r and cond(r)]
        return (np.median(v), np.percentile(v, 95), len(v)) if v else (np.nan, np.nan, 0)

    print("\n단계                         중앙값    95%     기준")
    for name, key, cond, lim in [
        ("F-02 화면 평면 (추적 프레임)", "plane_ms", lambda r: r.get("plane_mode") == "track", 8),
        ("F-02 화면 평면 (M1 재검출 프레임)", "plane_ms", lambda r: r.get("plane_mode") != "track", None),
        ("F-08 손끝(가짜 손, M4 제외)", "tip_ms", lambda r: True, 25),
        ("키프레임 판단", "keyframe_ms", lambda r: True, None),
        ("F-03 화면 읽기(펼치기+M2)", "read_ms", lambda r: True, 300),
        ("프레임 전체", "total_ms", lambda r: "read_ms" not in r, 66.7),
    ]:
        med, p95, n = stat(key, cond)
        ok = "" if lim is None else ("통과" if p95 <= lim else "초과")
        print(f"{name:<28} {med:7.1f} {p95:7.1f} ms  {'' if lim is None else f'≤{lim:g} ms'} {ok}  (n={n})")

    if a.m1_data:
        raw, ref, rnet = corner_accuracy(a.m1, a.m1_data, device=a.device)
        for name, e in (("M1", raw), ("M1+테두리 보정", ref), ("M1+M1-R(기본)", rnet)):
            if len(e) == 0:
                continue
            print(f"꼭짓점 오차 {name:<14} 중앙값 {np.median(e) * 100:5.2f}%  90% {np.percentile(e, 90) * 100:5.2f}%"
                  f"  2% 이내 비율 {(e < 0.02).mean() * 100:5.1f}%  (기준: 2% 이내)")


if __name__ == "__main__":
    main()
