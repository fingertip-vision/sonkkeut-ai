"""
F-03 화면 요소 인식

카메라 영상이 아니라 '펼친 화면'에서 M2를 돌린다.
기울어진 영상에서는 같은 버튼도 각도마다 모양이 달라 학습 데이터가 훨씬 많이 필요하지만,
펼친 화면에서는 모든 키오스크가 정면 사진처럼 보이므로 합성 데이터만으로도 잘 맞는다.
결과 좌표는 펼친 화면 기준 0~1이라 휴대폰이 움직여도 다시 계산할 필요가 없다.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .schema import Element

NAMES = {0: "tab", 1: "menu", 2: "price", 3: "button", 4: "back"}


def flat_matrix(W, H, margin=0.0):
    """화면 0~1 좌표 → 펼친 이미지 픽셀. margin만큼 화면 바깥도 함께 담는다."""
    s = 1.0 + 2 * margin
    return np.array([[W / s, 0, W * margin / s], [0, H / s, H * margin / s], [0, 0, 1]], np.float64)


def flatten(frame, plane, long_side=960, margin=0.0):
    """카메라 프레임 → 정면으로 펼친 화면 이미지

    margin: 꼭짓점이 화면 안쪽으로 조금 잘못 잡혀도 가장자리 버튼이 잘리지 않도록
            화면 바깥을 변 길이의 margin 비율만큼 더 담는다(요소 인식에서는 0.04 사용).
    """
    W, H = plane.flat_size(long_side)
    return cv2.warpPerspective(frame, flat_matrix(W, H, margin) @ plane.H, (W, H), flags=cv2.INTER_LINEAR)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def containment(inner, outer):
    """inner 면적 중 outer 안에 든 비율"""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def clean_overlaps(dets, same_iou=0.5, cross_iou=0.8):
    """겹친 박스 정리.
    같은 종류끼리 IoU 0.5 이상이면 신뢰도 높은 쪽만 남긴다(중복 검출).
    다른 종류끼리 거의 같은 자리(IoU 0.8 이상)면 하나만 남긴다(버튼/메뉴 혼동).
    가격이 메뉴 카드 안에 '포함'된 경우는 겹침이 아니라 구조이므로 둘 다 남긴다.
    """
    dets = sorted(dets, key=lambda d: -d[2])
    keep = []
    for box, kind, conf in dets:
        dup = False
        for kb, kk, _ in keep:
            v = iou(box, kb)
            if (kk == kind and v >= same_iou) or (kk != kind and v >= cross_iou):
                dup = True
                break
        if not dup:
            keep.append((box, kind, conf))
    return keep


def reading_order(dets, row_tol=0.5):
    """위→아래, 같은 줄이면 왼쪽→오른쪽. 화면 읽기 모드(F-12)와 id 안정성에 쓴다."""
    dets = sorted(dets, key=lambda d: (d[0][1] + d[0][3]) / 2)
    rows, cur = [], []
    for d in dets:
        if cur:
            ref_h = np.mean([c[0][3] - c[0][1] for c in cur])
            ref_y = np.mean([(c[0][1] + c[0][3]) / 2 for c in cur])
            if abs((d[0][1] + d[0][3]) / 2 - ref_y) > row_tol * ref_h:
                rows.append(cur)
                cur = []
        cur.append(d)
    if cur:
        rows.append(cur)
    return [d for row in rows for d in sorted(row, key=lambda d: d[0][0])]


class ElementDetector:
    def __init__(self, model, conf=0.3, imgsz=640, long_side=960, device=None, margin=0.04):
        if isinstance(model, str):
            from ultralytics import YOLO
            model = YOLO(model)
        self.model = model
        self.conf, self.imgsz, self.long_side, self.device = conf, imgsz, long_side, device
        self.margin = margin
        names = getattr(model, "names", None) or NAMES
        self.names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        self.last_timing = {}

    def detect_flat(self, flat, margin=0.0):
        """펼친 화면 이미지 → Element 목록 (좌표는 화면 0~1, margin 바깥 영역은 0 미만·1 초과 가능)"""
        r = self.model.predict(flat, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        h, w = flat.shape[:2]
        k, m = 1.0 + 2 * margin, margin
        dets = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            box = (b[0] / w * k - m, b[1] / h * k - m, b[2] / w * k - m, b[3] / h * k - m)
            if box[2] <= 0 or box[3] <= 0 or box[0] >= 1 or box[1] >= 1:
                continue  # 화면 바깥(베젤·배경)에서 잡힌 것
            dets.append((box, self.names.get(int(c), str(c)), float(s)))
        dets = reading_order(clean_overlaps(dets))
        els = [Element(id=f"e{i + 1}", kind=k, box=tuple(float(v) for v in b), conf=s)
               for i, (b, k, s) in enumerate(dets)]
        # 포함 관계: 가격이 메뉴 카드 안에 있으면 parent로 묶는다 (F-05가 메뉴명-가격을 짝지을 때 쓴다)
        for e in els:
            if e.kind != "price":
                continue
            best = max((m for m in els if m.kind == "menu"), key=lambda m: containment(e.box, m.box), default=None)
            if best is not None and containment(e.box, best.box) > 0.6:
                e.parent = best.id
        return els

    def detect(self, frame, plane):
        """카메라 프레임 + 화면 평면 → (Element 목록, 펼친 화면 이미지)"""
        t0 = time.perf_counter()
        flat = flatten(frame, plane, self.long_side, self.margin)
        t1 = time.perf_counter()
        els = self.detect_flat(flat, self.margin)
        self.last_timing = {"flatten_ms": (t1 - t0) * 1000, "m2_ms": (time.perf_counter() - t1) * 1000}
        return els, flat

    @staticmethod
    def crops(flat, elements, pad=0.008, margin=0.04):
        """F-04(문자 인식, 노현석)에 넘길 요소별 잘라낸 이미지. {id: BGR 이미지}
        flat은 detect()가 돌려준 이미지(margin 포함)여야 한다."""
        h, w = flat.shape[:2]
        k = 1.0 + 2 * margin
        out = {}
        for e in elements:
            x1, y1, x2, y2 = [(v + margin) / k for v in e.box]
            X1, Y1 = int(max(0, (x1 - pad) * w)), int(max(0, (y1 - pad) * h))
            X2, Y2 = int(min(w, (x2 + pad) * w)), int(min(h, (y2 + pad) * h))
            if X2 > X1 and Y2 > Y1:
                out[e.id] = flat[Y1:Y2, X1:X2].copy()
        return out
