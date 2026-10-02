"""
M1-R 꼭짓점 정밀 보정망 (2단계) — 데이터 생성 · 학습 · 내보내기

왜 2단계인가
  M1(YOLOv8n-pose)은 프레임 전체(640px)를 보고 꼭짓점 네 개를 한꺼번에 회귀한다.
  화면이 프레임의 대부분을 차지하면 꼭짓점이 격자 중심에서 멀어 오차가 커지고(화면 폭의 약 5%),
  4점 키포인트의 기본 OKS sigma(0.25)가 느슨해 학습도 그 이상 정밀해지려 하지 않는다.
  그래서 M1은 '대략 어디'만 맡기고, 꼭짓점마다 주변을 확대한 작은 패치를 보고
  '정확히 어디'를 다시 맞히는 작은 CNN을 따로 둔다(얼굴 랜드마크 등에서 흔히 쓰는 거친→정밀 방식).

패치 만들기
  - 크기는 화면 변 길이의 25%로 잡아(화면 크기와 무관하게 같은 배율) 64×64로 줄인다
  - 네 꼭짓점을 뒤집어 모두 '왼쪽 위 꼭짓점'처럼 보이게 맞춘다(TR은 좌우, BL은 상하, BR은 둘 다)
    → 한 모델이 네 꼭짓점을 모두 처리하고 데이터도 4배로 쓴다
  - 학습 때는 정답에서 일부러 흔든 위치(화면 폭의 ±8%)를 중심으로 잘라, 그 흔들림을 되돌리는 법을 배운다

  python corner_refiner.py gen   --out data/refine --n 3000
  python corner_refiner.py train --data data/refine --epochs 30
  python corner_refiner.py eval  --m1 ../models/m1_screen_corners.pt --data data/m1
"""
import argparse
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PATCH = 64
SCALE = 0.25  # 패치 한 변 = 화면 평균 변 길이 × 0.25
JITTER = 0.08  # 학습 때 중심을 흔드는 범위 (화면 평균 변 길이 대비)
FLIPS = [(False, False), (True, False), (True, True), (False, True)]  # TL, TR, BR, BL → TL 모양으로


def mean_side(c):
    c = np.asarray(c, np.float32)
    return float(np.mean([np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)]))


def crop_patch(gray, center, size, k):
    """center 주변 size×size를 잘라 PATCH×PATCH로, 꼭짓점 k를 TL 모양으로 뒤집어 돌려준다."""
    fx, fy = FLIPS[k]
    s = PATCH / size
    # 출력 (u, v) ← 입력 (x, y): u = s*(±(x - cx)) + PATCH/2
    ax, ay = (-s if fx else s), (-s if fy else s)
    M = np.float32([[ax, 0, PATCH / 2 - ax * center[0]], [0, ay, PATCH / 2 - ay * center[1]]])
    return cv2.warpAffine(gray, M, (PATCH, PATCH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def offset_to_image(center, size, k, d):
    """정규화된 예측 오프셋 d(TL 모양 패치 기준, 패치 크기 단위) → 원본 영상 좌표"""
    fx, fy = FLIPS[k]
    return np.float32([center[0] + (-d[0] if fx else d[0]) * size, center[1] + (-d[1] if fy else d[1]) * size])


# ---------------- 데이터 ----------------
def gen(out, n, seed, per_corner=4):
    import corner_synth
    import kiosk_synth

    random.seed(seed)
    np.random.seed(seed)
    fonts = kiosk_synth.find_fonts()
    X, Y = [], []
    t0 = time.time()
    i = 0
    while i < n:
        img, pts = corner_synth.make_sample(fonts)
        g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        if not all(0 <= x < w and 0 <= y < h for x, y in pts):
            continue
        side = mean_side(pts)
        size = SCALE * side
        for k in range(4):
            for _ in range(per_corner):
                jit = np.random.uniform(-JITTER, JITTER, 2) * side
                c = pts[k] + jit
                X.append(crop_patch(g, c, size, k))
                fx, fy = FLIPS[k]
                d = (pts[k] - c) / size
                Y.append([-d[0] if fx else d[0], -d[1] if fy else d[1]])
        i += 1
        if i % 200 == 0:
            print(f"{i}/{n}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, X=np.array(X, np.uint8), Y=np.array(Y, np.float32))
    print("saved", out, len(X))


# ---------------- 모델 ----------------
def build_model():
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))

    return nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 96),  # 64 → 4
                         nn.Flatten(), nn.Linear(96 * 16, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))


def augment(x):
    """패치 단위 밝기·대비·노이즈 흔들기 (uint8 N,64,64 → float N,1,64,64)"""
    x = x.astype(np.float32) / 255.0
    n = len(x)
    a = np.random.uniform(0.6, 1.4, (n, 1, 1)).astype(np.float32)
    b = np.random.uniform(-0.2, 0.2, (n, 1, 1)).astype(np.float32)
    x = (x - 0.5) * a + 0.5 + b
    neg = np.random.rand(n) < 0.3  # 밝은 베젤 / 어두운 화면 같은 반전된 대비
    x[neg] = 1 - x[neg]
    x += np.random.normal(0, np.random.uniform(0, 0.04), x.shape).astype(np.float32)
    return x[:, None]


def train(data, out, epochs=30, bs=256, lr=2e-3, init=None):
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    files = data.split(",")
    X = np.concatenate([np.load(f)["X"] for f in files])
    Y = np.concatenate([np.load(f)["Y"] for f in files])
    idx = np.random.default_rng(0).permutation(len(X))
    nv = max(256, len(X) // 20)
    vi, ti = idx[:nv], idx[nv:]
    model = build_model()
    if init:
        model.load_state_dict(torch.load(init, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(ti) // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xv = torch.from_numpy(X[vi].astype(np.float32)[:, None] / 255.0)
    Yv = torch.from_numpy(Y[vi])
    best = 1e9
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(ti)
        t0 = time.time()
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            xb = torch.from_numpy(augment(X[b]))
            yb = torch.from_numpy(Y[b])
            loss = torch.nn.functional.smooth_l1_loss(model(xb), yb, beta=0.02)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            err = (model(Xv) - Yv).norm(dim=1)  # 패치 크기 단위
        # 패치 = 변 길이 × 0.25 이므로 변 길이 대비 오차 = err × 0.25
        e = float(err.median()) * SCALE * 100
        print(f"epoch {ep + 1}/{epochs}  loss {loss.item():.4f}  val 중앙값 {e:.2f}% (변 길이 대비)  {time.time() - t0:.0f}s",
              flush=True)
        if e < best:
            best = e
            torch.save(model.state_dict(), out)
    print("best", best, "saved", out)


def export_onnx(weights, out):
    import torch

    model = build_model()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    torch.onnx.export(model, torch.zeros(4, 1, PATCH, PATCH), out, input_names=["patches"], output_names=["offsets"],
                      dynamic_axes={"patches": {0: "n"}, "offsets": {0: "n"}}, opset_version=13, dynamo=False)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "train", "export"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--data", default="data/refine.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--weights", default="../models/m1r_corner_refiner.pt")
    ap.add_argument("--init", default=None, help="이어서 학습할 가중치")
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a.out or "data/refine.npz", a.n, a.seed)
    elif a.cmd == "train":
        train(a.data, a.out or a.weights, a.epochs, lr=a.lr, init=a.init)
    else:
        export_onnx(a.weights, a.out or a.weights.replace(".pt", ".onnx"))


if __name__ == "__main__":
    main()
