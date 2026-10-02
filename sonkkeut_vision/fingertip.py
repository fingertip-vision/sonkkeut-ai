"""
F-08 손끝 추적

M4는 직접 학습하지 않고 MediaPipe Hand Landmarker(손 관절 21점)를 쓴다.
손 관절 데이터는 수십만 장 규모로 이미 잘 학습된 모델이 있고, 우리가 풀 문제는
'손끝이 화면의 어디에 있나'이지 '손을 찾는 법'이 아니기 때문이다.

처리 순서
  ① 손 관절 21점 추정 → ② 검지 끝(8번) 추출 → ③ F-02의 H로 화면 0~1 좌표로 변환
  ④ 최근 5프레임 이동 평균으로 떨림 완화
손이 여러 개면 '화면에 가장 가까운 손'(검지 끝이 화면 안에 있거나 화면 경계에 가장 가까운 손)을 고른다.
"""
from __future__ import annotations

import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

from .schema import Fingertip

INDEX_TIP = 8
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
DEFAULT_HAND_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "models", "hand_landmarker.task")


def ensure_hand_model(path=DEFAULT_HAND_MODEL):
    """hand_landmarker.task(약 7.8MB)가 없으면 내려받는다. 첫 실행 때 한 번만 필요하다."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"손 관절 모델 내려받는 중: {HAND_MODEL_URL}")
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


class MediaPipeHandSource:
    """카메라 프레임 → [(관절 21점 픽셀 좌표 (21,2), 신뢰도), ...]"""

    def __init__(self, model_path=None, num_hands=2, min_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        path = ensure_hand_model(model_path or DEFAULT_HAND_MODEL)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def __call__(self, frame, t):
        h, w = frame.shape[:2]
        ts = max(int(t * 1000), self._last_ts + 1)  # VIDEO 모드는 시각이 계속 커져야 한다
        self._last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(img, ts)
        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            hands.append((pts, float(hd[0].score) if hd else 1.0))
        return hands

    def close(self):
        self.landmarker.close()


def pointing_score(pts):
    """검지를 펴서 가리키는 자세인지 0~1로 평가한다.

    MediaPipe는 검지를 접은 손에도 21점을 내놓는데, 이때 '검지 끝(8번)'은 실제 손가락 끝이 아니라
    접힌 마디 근처에 찍혀 실제 손끝과 수 cm 어긋난다(실제 손 사진 실험에서 잘못된 '누르세요'의 원인).
    검지 길이(5→8)가 손바닥 길이(0→9)의 0.6배 이하면 접힌 것으로, 0.8배 이상이면 편 것으로 본다.
    """
    palm = np.linalg.norm(pts[9] - pts[0])
    if palm < 1e-6:
        return 0.0
    ext = np.linalg.norm(pts[8] - pts[5]) / palm
    return float(np.clip((ext - 0.6) / 0.2, 0.0, 1.0))


def _dist_to_unit_square(x, y):
    """화면 0~1 좌표의 점이 화면 사각형에서 얼마나 떨어졌나 (안이면 0)"""
    dx = max(0.0 - x, 0.0, x - 1.0)
    dy = max(0.0 - y, 0.0, y - 1.0)
    return float(np.hypot(dx, dy))


class FingertipTracker:
    """
    hand_source: frame, t → [(pts(21,2) 픽셀, score)] 를 돌려주는 호출 가능 객체.
                 실제로는 MediaPipeHandSource, 테스트에서는 가짜 손을 넣는다.
    """

    def __init__(self, hand_source, smooth=5, jump_reset=0.25, max_outside=0.15):
        self.source = hand_source
        self.smooth = smooth
        self.jump_reset = jump_reset  # 한 프레임에 화면 높이의 25% 이상 튀면 평균을 버린다(다른 손으로 바뀜)
        self.max_outside = max_outside  # 화면 밖 15%까지는 손끝으로 인정(화면 가장자리 버튼 유도용)
        self.hist = deque(maxlen=smooth)
        self.last_seen = None
        self.last_hand_px = None
        self.last_timing = {}

    def reset(self):
        self.hist.clear()
        self.last_seen = None

    def select_hand(self, hands, plane):
        best, best_key = None, None
        for pts, score in hands:
            tip = plane.to_screen(pts[INDEX_TIP:INDEX_TIP + 1])[0]
            d = _dist_to_unit_square(*tip)
            key = (d, -score)
            if best_key is None or key < best_key:
                best, best_key = (pts, score, tip, d), key
        return best

    def hand_box_screen(self, plane, pts=None, pad=0.3):
        """손 전체를 감싸는 화면 0~1 박스 (키프레임 비교에서 손 영역을 빼는 데 쓴다)
        관절점은 손가락 '중심선'이라 실제 손 윤곽보다 좁으므로 박스 크기의 30%만큼 넓힌다."""
        pts = self.last_hand_px if pts is None else pts
        if pts is None or plane is None:
            return None
        s = plane.to_screen(pts)
        (x1, y1), (x2, y2) = s.min(0), s.max(0)
        px, py = (x2 - x1) * pad + 0.02, (y2 - y1) * pad + 0.02
        x1, y1, x2, y2 = np.clip([x1 - px, y1 - py, x2 + px, y2 + py], -1, 2)
        return float(x1), float(y1), float(x2), float(y2)

    def update(self, frame, plane, t=None):
        t = time.monotonic() if t is None else t
        t0 = time.perf_counter()
        hands = self.source(frame, t) if plane is not None else []
        t1 = time.perf_counter()
        sel = self.select_hand(hands, plane) if hands else None
        if sel is None or sel[3] > self.max_outside:
            self.last_hand_px = sel[0] if sel is not None else None
            if self.last_seen is None:
                self.last_seen = t
            lost = t - self.last_seen
            if lost > 0.3:
                self.hist.clear()
            self.last_timing = {"m4_ms": (t1 - t0) * 1000, "tip_ms": (time.perf_counter() - t0) * 1000}
            return Fingertip(pos=None, lost_for=lost)
        pts, score, tip, d = sel
        self.last_hand_px = pts
        if self.hist and np.hypot(*(np.asarray(self.hist[-1]) - tip)) > self.jump_reset:
            self.hist.clear()
        self.hist.append(tuple(tip))
        pos = tuple(float(v) for v in np.mean(self.hist, axis=0))
        self.last_seen = t
        self.last_timing = {"m4_ms": (t1 - t0) * 1000, "tip_ms": (time.perf_counter() - t0) * 1000}
        pt = pointing_score(pts)
        return Fingertip(pos=pos, conf=score * pt, image_px=tuple(map(float, pts[INDEX_TIP])), lost_for=0.0,
                         inside_screen=d == 0.0, pointing=pt)
