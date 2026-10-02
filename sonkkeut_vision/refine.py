"""
꼭짓점 보정 (실험 기능, 기본 꺼짐: ScreenPlaneEstimator(refine=True)로 켠다)

M1(YOLOv8n-pose)은 꼭짓점을 한 점씩 회귀하므로 위치가 화면 폭의 수 % 흔들린다.
화면 테두리는 영상에서 가장 긴 직선이므로, M1이 준 사각형의 네 변 각각에 대해
  ① 변을 따라 48개 지점에서 변에 수직 방향으로 밝기 변화가 큰 곳(후보 최대 3개)을 찾고
  ② 후보들 중 가장 많은 점을 지나면서 원래 변과 거의 평행한 직선을 RANSAC으로 고른 뒤
  ③ 이웃한 직선의 교점을 새 꼭짓점으로 삼는다.
합성 검증 세트에서 '오차 2% 이내' 비율이 2% → 28%로 늘지만, 베젤 바깥 테두리를 화면 테두리로
잘못 고르는 경우가 있어 중앙값은 거의 그대로다(4.8% → 4.6%). 실제 사진으로 확인한 뒤 켤지 정한다.
"""

from __future__ import annotations

import cv2
import numpy as np


def _inter(l1, l2):
    """(점, 방향)으로 나타낸 두 직선의 교점"""
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s, _ = np.linalg.solve(A, p2 - p1)
    return p1 + s * d1


def refine_corners(gray, corners, search=0.06, n=48, iters=300, tol=1.5, seed=0):
    """return (보정된 corners (4,2) float32, 성공 여부)"""
    rng = np.random.default_rng(seed)
    c = np.asarray(corners, np.float64)
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]
    ctr = c.mean(0)
    lines = []
    for k in range(4):
        p, q = c[k], c[(k + 1) % 4]
        L = np.linalg.norm(q - p)
        if L < 20:
            return c.astype(np.float32), False
        d = (q - p) / L
        nn = np.array([-d[1], d[0]])
        if np.dot(ctr - (p + q) / 2, nn) > 0:
            nn = -nn
        r = max(6, search * L)
        offs = np.arange(-r, r + 1)
        cand = []
        for a in np.linspace(0.08, 0.92, n):
            b = p + (q - p) * a
            ray = b + offs[:, None] * nn
            xs, ys = ray[:, 0], ray[:, 1]
            ok = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
            if ok.sum() < len(offs) * 0.6:
                continue
            xi, yi = xs[ok].astype(int), ys[ok].astype(int)
            pr = np.abs(gx[yi, xi] * nn[0] + gy[yi, xi] * nn[1])
            rr = ray[ok]
            # 이 광선 위의 밝기 변화 극대점 (화면 테두리, 베젤 바깥 테두리, 화면 안 선 등) 상위 3개
            lm = [
                j
                for j in range(1, len(pr) - 1)
                if pr[j] >= pr[j - 1] and pr[j] >= pr[j + 1] and pr[j] >= max(8, 0.3 * pr.max())
            ]
            lm = sorted(lm, key=lambda j: -pr[j])[:3]
            cand += [rr[j] for j in lm]
        if len(cand) < n * 0.5:
            return c.astype(np.float32), False
        P = np.array(cand)
        best = None
        for _ in range(iters):
            i, j = rng.choice(len(P), 2, replace=False)
            v = P[j] - P[i]
            nv = np.linalg.norm(v)
            if nv < L * 0.2:
                continue
            v /= nv
            if abs(np.dot(v, d)) < np.cos(np.radians(6)):  # 원래 변과 6도 넘게 기울면 버림
                continue
            nrm = np.array([-v[1], v[0]])
            dist = np.abs((P - P[i]) @ nrm)
            inl = dist < tol
            # 지지하는 점이 가장 많은 직선, 같으면 원래 변에 더 가까운 직선
            key = (int(inl.sum()), -abs(np.dot(P[i] - p, nn)))
            if best is None or key > best[0]:
                best = (key, inl)
        if best is None:
            return c.astype(np.float32), False
        inl = best[1]
        if inl.sum() < n * 0.4:
            return c.astype(np.float32), False
        vx, vy, x0, y0 = cv2.fitLine(P[inl].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    out = []
    for k in range(4):
        x = _inter(lines[(k - 1) % 4], lines[k])
        if x is None:
            return c.astype(np.float32), False
        out.append(x)
    out = np.array(out, np.float32)
    sc = np.mean([np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)])
    if np.linalg.norm(out - c, axis=1).max() > 0.08 * sc or not cv2.isContourConvex(out.reshape(-1, 1, 2)):
        return c.astype(np.float32), False
    return out, True
