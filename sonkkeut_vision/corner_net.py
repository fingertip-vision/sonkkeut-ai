"""
M1-R 꼭짓점 정밀 보정망 추론 (학습: training/corner_refiner.py)

M1이 준 대략의 꼭짓점마다 주변 패치(화면 변 길이의 25%)를 64×64로 잘라 작은 CNN에 넣고,
정확한 꼭짓점까지의 오프셋을 받아 옮긴다. 이를 2번 반복한다(옮긴 위치에서 다시 자르면 더 정확해짐).
네 패치를 한 번에 넣으므로 CPU에서도 1~2 ms 정도다.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)

from corner_refiner import PATCH, SCALE, build_model, crop_patch, mean_side, offset_to_image  # noqa: E402

DEFAULT_REFINER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                               "m1r_corner_refiner.pt")


class CornerRefiner:
    def __init__(self, weights=DEFAULT_REFINER, iters=2, max_shift=0.12):
        import torch

        self.torch = torch
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()
        self.iters, self.max_shift = iters, max_shift

    def __call__(self, gray, corners):
        """return 보정된 corners (4,2). 보정량이 비정상적으로 크면 원래 값을 돌려준다."""
        c0 = np.asarray(corners, np.float32)
        c = c0.copy()
        side = mean_side(c)
        size = SCALE * side
        for _ in range(self.iters):
            x = np.stack([crop_patch(gray, c[k], size, k) for k in range(4)]).astype(np.float32)[:, None] / 255.0
            with self.torch.no_grad():
                d = self.model(self.torch.from_numpy(x)).numpy()
            c = np.stack([offset_to_image(c[k], size, k, d[k]) for k in range(4)])
        if np.linalg.norm(c - c0, axis=1).max() > self.max_shift * side:
            return c0
        if not cv2.isContourConvex(c.reshape(-1, 1, 2).astype(np.float32)):
            return c0
        return c.astype(np.float32)


__all__ = ["CornerRefiner", "DEFAULT_REFINER", "PATCH"]
