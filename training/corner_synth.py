"""
손끝길 화면 꼭짓점 합성 데이터 생성기 (M1: 화면 평면 추정용)

kiosk_synth.py가 그린 키오스크 화면을 키오스크 본체(베젤) 안에 넣고,
스마트폰으로 비스듬히 찍은 것처럼 원근 변환해 배경 위에 합성한다.
화면 네 꼭짓점의 위치를 알고 변환했기 때문에 키포인트 라벨이 자동으로 생긴다.

라벨 형식: YOLO pose
  class cx cy w h  x_TL y_TL v  x_TR y_TR v  x_BR y_BR v  x_BL y_BL v
  (꼭짓점 순서: 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래, v=2 보임 / 0 프레임 밖)

사용 예
  python corner_synth.py --out data/m1 --n-train 3000 --n-val 400
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from kiosk_synth import find_fonts, photometric, render_screen

FRAME_W, FRAME_H = 960, 1280  # 세로로 든 스마트폰 카메라 프레임


BG_FILES = []  # --bg-dir 로 실제 매장 사진 폴더를 주면 여기에 채워진다


def background(w, h):
    """배경: 실제 사진이 있으면 무작위로 잘라 쓰고, 없으면 절차적으로 만든다"""
    if BG_FILES and random.random() < 0.8:
        im = cv2.imread(random.choice(BG_FILES))
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            s = max(w / im.shape[1], h / im.shape[0]) * random.uniform(1.0, 1.5)
            im = cv2.resize(im, (int(im.shape[1] * s) + 1, int(im.shape[0] * s) + 1))
            y0, x0 = random.randint(0, im.shape[0] - h), random.randint(0, im.shape[1] - w)
            return im[y0:y0 + h, x0:x0 + w].copy()
    c1 = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
    c2 = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
    t = np.linspace(0, 1, h)[:, None, None]
    bg = (c1 * (1 - t) + c2 * t).repeat(w, axis=1)
    for _ in range(random.randint(3, 12)):
        x1, y1 = random.randint(-100, w), random.randint(-100, h)
        x2, y2 = x1 + random.randint(40, w // 2), y1 + random.randint(40, h // 2)
        col = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(bg, (x1, y1), (x2, y2), col, -1)
    bg += np.random.normal(0, random.uniform(2, 12), bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def kiosk_with_bezel(screen):
    """화면 둘레에 키오스크 베젤과 본체를 붙인다. 화면 꼭짓점 좌표도 함께 돌려준다."""
    s = np.asarray(screen)
    h, w = s.shape[:2]
    bz = int(min(w, h) * random.uniform(0.02, 0.08))
    extra_bottom = int(h * random.uniform(0.0, 0.25))  # 카드 투입구 등 본체 아래쪽
    body_col = random.choice([(20, 20, 20), (235, 235, 235), (60, 60, 70), (200, 200, 205)])
    K = np.full((h + 2 * bz + extra_bottom, w + 2 * bz, 3), body_col, np.uint8)
    K[bz:bz + h, bz:bz + w] = s
    if extra_bottom > 20:  # 카드 투입구 흉내
        cy = bz * 2 + h + extra_bottom // 2
        cv2.rectangle(K, (w // 3, cy - 6), (w // 3 * 2, cy + 6), (40, 40, 40), -1)
    corners = np.float32([[bz, bz], [bz + w, bz], [bz + w, bz + h], [bz, bz + h]])
    return K, corners


def random_homography(src_w, src_h):
    """카메라 프레임 안에 키오스크를 비스듬히 놓는 원근 변환"""
    scale = random.uniform(0.45, 1.05) * min(FRAME_W / src_w, FRAME_H / src_h)
    w, h = src_w * scale, src_h * scale
    cx = FRAME_W / 2 + random.uniform(-0.25, 0.25) * FRAME_W
    cy = FRAME_H / 2 + random.uniform(-0.2, 0.2) * FRAME_H
    dst = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    # 원근(기울임): 꼭짓점마다 다른 크기로 흔든다
    jitter = random.uniform(0.0, 0.18)
    dst += np.random.uniform(-jitter, jitter, dst.shape).astype(np.float32) * np.float32([w, h])
    ang = np.deg2rad(random.uniform(-20, 20))
    R = np.float32([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = dst @ R.T + np.float32([cx, cy])
    src = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
    return cv2.getPerspectiveTransform(src, dst)


def make_sample(fonts):
    screen, _, _ = render_screen(fonts)
    screen = screen.resize((screen.width // 2, screen.height // 2))
    K, corners = kiosk_with_bezel(screen)
    H = random_homography(K.shape[1], K.shape[0])
    bg = background(FRAME_W, FRAME_H)
    warped = cv2.warpPerspective(K, H, (FRAME_W, FRAME_H))
    mask = cv2.warpPerspective(np.full(K.shape[:2], 255, np.uint8), H, (FRAME_W, FRAME_H))
    frame = np.where(mask[..., None] > 0, warped, bg)
    pts = cv2.perspectiveTransform(corners[None], H)[0]
    img = photometric(Image.fromarray(frame))
    return img, pts


def pose_line(pts, w, h):
    vis = [2 if 0 <= x < w and 0 <= y < h else 0 for x, y in pts]
    if sum(v > 0 for v in vis) < 3:
        return None  # 꼭짓점이 둘 이상 잘리면 학습에서 뺀다
    xs, ys = np.clip(pts[:, 0], 0, w - 1), np.clip(pts[:, 1], 0, h - 1)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    kp = " ".join(f"{x / w:.6f} {y / h:.6f} {v}" if v else "0 0 0" for (x, y), v in zip(pts, vis))
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {kp}"


def build(out, n_train, n_val, fonts, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    out = Path(out)
    for split, n in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        i = 0
        while i < n:
            img, pts = make_sample(fonts)
            line = pose_line(pts, img.width, img.height)
            if line is None:
                continue
            stem = f"{split}_{i:05d}"
            img.save(out / "images" / split / f"{stem}.jpg", quality=random.randint(60, 95))
            (out / "labels" / split / f"{stem}.txt").write_text(line)
            i += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: screen\n", encoding="utf-8")
    print(f"done: {out} train={n_train} val={n_val}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/m1")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--font-dir", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bg-dir", default=None, help="실제 매장·실내 사진 폴더(선택). 넣으면 현실감이 크게 올라간다")
    a = ap.parse_args()
    if a.bg_dir:
        BG_FILES.extend(str(p) for p in Path(a.bg_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"background photos: {len(BG_FILES)}")
    build(a.out, a.n_train, a.n_val, find_fonts(a.font_dir), seed=a.seed)
