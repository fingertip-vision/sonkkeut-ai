"""
키프레임 판단 — 무거운 화면 읽기(F-03~F-05)를 '화면이 바뀌었을 때만' 돌리기 위한 장치

펼친 화면을 작은 흑백 이미지(64칸 기준)로 줄여 직전 키프레임과 비교한다.
카메라 좌표가 아니라 펼친 화면끼리 비교하므로 휴대폰이 흔들려도 '화면 내용이 바뀐 것'만 잡힌다.
손가락이 화면을 가리는 것도 변화로 보이면 안 되므로 손 영역은 비교에서 뺀다.
화면 전환 애니메이션 도중에 읽으면 반쯤 바뀐 화면을 읽게 되므로,
'바뀌었고(changed) + 몇 프레임 동안 멈춰 있을 때(stable)' 키프레임을 낸다.
"""
from __future__ import annotations

import cv2
import numpy as np


class KeyframeDetector:
    def __init__(self, size=64, change_thr=12.0, stable_thr=4.0, stable_frames=3, min_valid=0.4):
        self.size = size
        self.change_thr, self.stable_thr = change_thr, stable_thr
        self.stable_frames, self.min_valid = stable_frames, min_valid
        self.reset()

    def reset(self):
        self.key = None  # 마지막 키프레임 (작은 흑백)
        self.prev = None
        self.stable_count = 0
        self.keyframe_id = 0
        self.last_change = 0.0

    def _small(self, flat):
        h, w = flat.shape[:2]
        s = self.size / max(h, w)
        g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY) if flat.ndim == 3 else flat
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)

    def _diff(self, a, b, mask):
        if a.shape != b.shape:
            return 255.0  # 화면 비율이 바뀜 = 다른 화면
        d = np.abs(a - b)
        if mask is not None:
            if mask.mean() < self.min_valid:
                return None  # 대부분 가려져 판단 불가
            return float(d[mask].mean())
        return float(d.mean())

    def hand_mask(self, small_shape, hand_boxes):
        """hand_boxes: 화면 0~1 좌표 (x1,y1,x2,y2) 목록 → 비교에 쓸 칸 True"""
        m = np.ones(small_shape, bool)
        h, w = small_shape
        for x1, y1, x2, y2 in hand_boxes or []:
            m[max(0, int(y1 * h) - 1):min(h, int(y2 * h) + 2), max(0, int(x1 * w) - 1):min(w, int(x2 * w) + 2)] = False
        return m

    def update(self, flat, hand_boxes=None, force=False):
        """return (is_keyframe, change_score)"""
        cur = self._small(flat)
        mask = self.hand_mask(cur.shape, hand_boxes)
        if self.key is None or force:
            self._commit(cur)
            return True, 0.0
        change = self._diff(cur, self.key, mask)
        motion = self._diff(cur, self.prev, mask) if self.prev is not None else 0.0
        self.prev = cur
        if change is None or motion is None:
            self.stable_count = 0
            return False, 0.0
        self.last_change = change
        self.stable_count = self.stable_count + 1 if motion < self.stable_thr else 0
        if change > self.change_thr and self.stable_count >= self.stable_frames:
            self._commit(cur)
            return True, change
        return False, change

    def changed_since_key(self, flat, hand_boxes=None):
        """누름 결과 확인(F-10)용: 지금 화면이 마지막 키프레임과 다른가"""
        if self.key is None:
            return False
        cur = self._small(flat)
        v = self._diff(cur, self.key, self.hand_mask(cur.shape, hand_boxes))
        return v is not None and v > self.change_thr

    def _commit(self, cur):
        self.key = cur
        self.prev = cur
        self.stable_count = 0
        self.keyframe_id += 1
