"""
F-02 화면 평면 추정

왜 '검출 + 추적' 두 단계인가
  M1(YOLOv8n-pose)은 프레임당 수십 ms가 걸려 매 프레임 돌리면 손끝 유도(15FPS) 예산을 다 쓴다.
  화면은 평평하므로 직전 프레임과 지금 프레임 사이의 움직임은 '호모그래피 하나'로 설명된다.
  그래서 화면 안의 특징점 수십 개를 광류(Lucas-Kanade)로 따라가 그 호모그래피를 RANSAC으로 구하고,
  네 꼭짓점을 함께 옮긴다. 이 계산은 수 ms면 끝난다.
  손가락이 화면을 가려도 RANSAC이 가려진 점을 이상치로 버리므로 추적이 흔들리지 않는다.
  추적 품질(정상치 비율)이 떨어지거나 일정 프레임이 지나면 M1으로 다시 검출해 누적 오차를 지운다.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .refine import refine_corners
from .schema import PlaneState

UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def homography_from_corners(corners):
    """꼭짓점 4개(카메라 픽셀) → (H: 픽셀→0~1, 화면 가로/세로 비)"""
    c = np.asarray(corners, np.float32)
    H = cv2.getPerspectiveTransform(c, UNIT_SQUARE)
    tl, tr, br, bl = c
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return H.astype(np.float64), float(w / max(h, 1e-6))


def is_valid_quad(corners, frame_shape, min_area_ratio=0.02):
    """볼록한 사각형이고 너무 작지 않은지. 잘못된 검출로 화면이 뒤집히는 것을 막는다."""
    c = np.asarray(corners, np.float32)
    if not cv2.isContourConvex(c.reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(c)
    return area >= min_area_ratio * frame_shape[0] * frame_shape[1]


class ScreenPlaneEstimator:
    """
    update(frame) 을 매 프레임 부르면 PlaneState(없으면 None)와 안내 문구(hint)를 돌려준다.

    model: ultralytics YOLO pose 모델 또는 경로
    """

    def __init__(self, model, conf=0.25, kp_conf=0.5, redetect_every=30, min_inliers=25,
                 min_inlier_ratio=0.6, lost_after=3.0, imgsz=640, device=None, refine=False, reframe_tol=0.08,
                 refiner="auto"):
        if isinstance(model, str):
            from ultralytics import YOLO
            model = YOLO(model)
        self.model = model
        self.conf, self.kp_conf = conf, kp_conf
        self.redetect_every = redetect_every
        self.min_inliers, self.min_inlier_ratio = min_inliers, min_inlier_ratio
        self.lost_after = lost_after
        self.imgsz, self.device = imgsz, device
        self.refine = refine  # 테두리 직선으로 꼭짓점 보정 (refine.py)
        self.reframe_tol = reframe_tol  # 재검출값이 추적값과 변 길이의 8% 넘게 다를 때만 기준을 바꾼다
        # M1-R 꼭짓점 정밀 보정망(corner_net.py). "auto"면 models/m1r_corner_refiner.pt가 있을 때만 쓴다.
        if refiner == "auto":
            from .corner_net import DEFAULT_REFINER, CornerRefiner
            refiner = CornerRefiner(DEFAULT_REFINER) if os.path.exists(DEFAULT_REFINER) else None
        elif isinstance(refiner, str):
            from .corner_net import CornerRefiner
            refiner = CornerRefiner(refiner)
        self.refiner = refiner
        self.reset()

    def reset(self):
        self.state = None
        self._prev_gray = None
        self._pts = None  # 추적 중인 특징점 (N,1,2)
        self._since_detect = 0
        self._frame_idx = 0
        self._last_seen = None
        self.last_timing = {}

    # ---------- 검출 ----------
    def detect(self, frame):
        """M1으로 꼭짓점 검출. (corners, conf, kp_conf(4,)) 또는 None"""
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        kp = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
        kc = r.keypoints.conf
        kc = kc[i].cpu().numpy() if kc is not None else np.ones(4, np.float32)
        box_conf = float(r.boxes.conf[i])
        return kp, box_conf * float(kc.mean()), kc

    @staticmethod
    def missing_hint(kp, kc, frame_shape, kp_conf=0.5, margin=4):
        """보이지 않는 꼭짓점이 있으면 휴대폰을 어느 쪽으로 옮길지 문구로 돌려준다."""
        h, w = frame_shape[:2]
        miss = [(kc[j] < kp_conf) or not (margin <= kp[j, 0] < w - margin and margin <= kp[j, 1] < h - margin)
                for j in range(4)]
        if not any(miss):
            return None
        # 잘린 꼭짓점이 프레임 어느 쪽으로 나갔는지 본다. 그쪽으로 휴대폰을 옮기면 된다.
        left = right = top = bottom = False
        for j in range(4):
            if not miss[j]:
                continue
            x, y = kp[j]
            out = False
            if x < margin:
                left = out = True
            if x >= w - margin:
                right = out = True
            if y < margin:
                top = out = True
            if y >= h - margin:
                bottom = out = True
            if not out:  # 프레임 안인데 신뢰도만 낮음(가려짐·반사) → 꼭짓점 위치(0 왼쪽 위 …)로 판단
                left |= j in (0, 3)
                right |= j in (1, 2)
                top |= j in (0, 1)
                bottom |= j in (2, 3)
        if (left and right) or (top and bottom):
            return "휴대폰을 조금 뒤로 빼 주세요"
        parts = [w_ for w_, f in (("왼쪽", left), ("오른쪽", right), ("위", top), ("아래", bottom)) if f]
        josa = "으로" if parts[-1].endswith("쪽") else "로"  # 받침 있는 '쪽' 뒤에는 '으로'
        return f"휴대폰을 조금 {' '.join(parts)}{josa}"

    # ---------- 추적 ----------
    def _seed_points(self, gray, corners):
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        # 화면 테두리 바로 안쪽은 베젤과 섞이므로 조금 줄인다
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
        return cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=8, mask=mask)

    def _track(self, gray):
        if self._pts is None or len(self._pts) < self.min_inliers:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._pts, None,
                                              winSize=(21, 21), maxLevel=3)
        ok = st.reshape(-1) == 1
        if ok.sum() < self.min_inliers:
            return None
        p0, p1 = self._pts[ok], nxt[ok]
        M, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if M is None:
            return None
        inl = inl.reshape(-1).astype(bool)
        ratio = inl.sum() / len(self._pts)
        if inl.sum() < self.min_inliers or ratio < self.min_inlier_ratio:
            return None
        corners = cv2.perspectiveTransform(self.state.corners.reshape(1, 4, 2).astype(np.float64), M)[0]
        self._pts = p1[inl].reshape(-1, 1, 2)
        return corners.astype(np.float32), ratio

    # ---------- 매 프레임 ----------
    def update(self, frame, t=None):
        """return (PlaneState | None, hint 문구 | None)"""
        t = time.monotonic() if t is None else t
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._frame_idx += 1
        hint = None
        prev = self.state
        tracked = self._track(gray) if prev is not None else None
        if tracked is not None and not is_valid_quad(tracked[0], gray.shape):
            tracked = None
        need_detect = tracked is None or self._since_detect >= self.redetect_every
        det = self.detect(frame) if need_detect else None
        if det is not None:
            kp, conf, kc = det
            hint = self.missing_hint(kp, kc, gray.shape, self.kp_conf)
            if hint is None and self.refiner is not None:
                kp = self.refiner(gray, kp)
            if hint is None and self.refine:
                kp, _ = refine_corners(gray, kp)
            if hint is not None or not is_valid_quad(kp, gray.shape):
                det = None

        if tracked is not None:
            corners, ratio = tracked
            mode = "track"
            if det is not None:
                # 주기적 재검출: 추적 결과와 거의 같으면 추적 결과를 그대로 쓴다.
                # M1 자체가 화면 폭의 수 % 흔들리므로, 매번 검출값으로 바꾸면 기준 좌표계가 튀어서
                # 이미 읽어 둔 버튼 좌표(0~1)가 실제 버튼과 어긋난다. 크게 다를 때(추적이 틀어짐)만 바꾼다.
                side = float(np.mean([np.linalg.norm(corners[(k + 1) % 4] - corners[k]) for k in range(4)]))
                if np.linalg.norm(kp - corners, axis=1).max() > self.reframe_tol * side:
                    self._adopt(gray, kp, conf, reframe_from=PlaneState(corners, homography_from_corners(corners)[0],
                                                                       prev.conf, prev.aspect, "track"))
                    mode = "reframe"
                else:
                    self._since_detect = 0
                    self.state = self._make(corners, max(conf, prev.conf * 0.98), "track")
                    mode = "verify"
            else:
                # 추적 신뢰도는 마지막 검출 신뢰도에서 서서히 낮춘다
                self.state = self._make(corners, prev.conf * (0.98 + 0.02 * ratio), "track")
                self._since_detect += 1
            if len(self._pts) < 80:  # 특징점이 줄어들면 다시 뿌린다
                self._pts = self._seed_points(gray, self.state.corners)
        elif det is not None:
            self._adopt(gray, kp, conf, reframe_from=prev)
            mode = "detect"
        else:
            self.state, self._pts = None, None
            mode = "detect"
        self.last_timing = {"plane_ms": (time.perf_counter() - t0) * 1000, "plane_mode": mode}
        self._prev_gray = gray
        if self.state is not None:
            self._last_seen = t
        elif hint is None:
            if self._last_seen is None:
                self._last_seen = t
            if t - self._last_seen >= self.lost_after:
                hint = "화면이 보이지 않습니다. 천천히 움직여 주세요"
        return self.state, hint

    def _make(self, corners, conf, source):
        H, aspect = homography_from_corners(corners)
        return PlaneState(np.asarray(corners, np.float32), H, conf, aspect, source, self._frame_idx)

    def _adopt(self, gray, corners, conf, reframe_from=None):
        """새 검출값을 기준 좌표계로 삼는다. 이전 기준이 있었다면 reframed로 표시해
        파이프라인이 읽어 둔 버튼 좌표를 새 좌표계로 옮기고 화면을 다시 읽게 한다."""
        self.state = self._make(corners, conf, "detect")
        if reframe_from is not None:
            self.state.reframed = True
            self.state.ref_prev = reframe_from
        self._pts = self._seed_points(gray, self.state.corners)
        self._since_detect = 0

    def force_redetect(self):
        self._since_detect = self.redetect_every
