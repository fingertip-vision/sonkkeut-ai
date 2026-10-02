"""
손끝길 화면 읽기 파이프라인 데모 (F-02 → F-03)

카메라 사진 한 장을 받아
  1) M1으로 화면 네 꼭짓점을 찾고
  2) 호모그래피로 화면을 정면으로 편 다음
  3) M2로 버튼·메뉴·가격·탭을 찾아
결과 이미지를 저장한다.

  python demo_pipeline.py --m1 models/m1_screen_corners.pt --m2 models/m2_screen_elements.pt --img photo.jpg
"""
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

COLORS = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 140, 255), (170, 0, 170)]


def flatten(img, kpts, long_side=960):
    tl, tr, br, bl = kpts
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    s = long_side / max(w, h)
    W, H = int(w * s), int(h * s)
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(img, M, (W, H)), M


def run(m1_path, m2_path, img_path, out_path, conf=0.25):
    img = cv2.imread(img_path)
    m1, m2 = YOLO(m1_path), YOLO(m2_path)
    r1 = m1.predict(img, conf=conf, verbose=False)[0]
    if r1.keypoints is None or len(r1.keypoints) == 0:
        raise SystemExit("화면을 찾지 못했습니다")
    kp = r1.keypoints.xy[0].cpu().numpy()
    flat, _ = flatten(img, kp)
    r2 = m2.predict(flat, conf=conf, verbose=False)[0]
    vis_flat = flat.copy()
    for b, c in zip(r2.boxes.xyxy.cpu().numpy(), r2.boxes.cls.cpu().numpy().astype(int)):
        cv2.rectangle(vis_flat, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), COLORS[c % 5], 3)
        cv2.putText(vis_flat, m2.names[c], (int(b[0]), int(b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[c % 5], 2)
    vis = img.copy()
    cv2.polylines(vis, [kp.astype(np.int32)], True, (0, 0, 255), 4)
    for i, p in enumerate(kp):
        cv2.circle(vis, tuple(p.astype(int)), 12, [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255)][i], -1)
    h = 900
    a = cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h))
    b = cv2.resize(vis_flat, (int(vis_flat.shape[1] * h / vis_flat.shape[0]), h))
    cv2.imwrite(out_path, np.hstack([a, np.full((h, 20, 3), 255, np.uint8), b]))
    print(f"saved {out_path}: {len(r2.boxes)} elements")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--m2", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="demo_result.jpg")
    a = ap.parse_args()
    run(a.m1, a.m2, a.img, a.out)
