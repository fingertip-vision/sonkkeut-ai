"""
가짜 카메라 시뮬레이터 — 실제 키오스크·손 없이 전체 흐름을 시험하기 위한 장치

  - 화면: training/kiosk_synth.py로 그린 키오스크 화면 (정답 박스를 알고 있음)
  - 카메라: 휴대폰을 든 손이 천천히 흔들리는 것처럼 호모그래피가 매 프레임 조금씩 변함
  - 손: 정해진 화면 좌표에 검지 끝이 오도록 손가락 모양을 그리고, 관절 21점을 돌려줌
        (MediaPipe 대신 들어가는 가짜 손. 손이 화면을 가리는 효과도 그대로 생긴다)
통합 테스트, 지연 측정(scripts/bench.py), 카메라 없이 데모 영상 만들기에 쓴다.
"""
from __future__ import annotations

import math
import os
import random
import sys

import cv2
import numpy as np

_TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)

import kiosk_synth  # noqa: E402

FRAME_W, FRAME_H = 960, 1280


def render(kind="menu", seed=0, w=720, h=1280):
    """키오스크 화면 하나 → (BGR 이미지, [(종류, x1,y1,x2,y2 0~1)])"""
    random.seed(seed)
    np.random.seed(seed)
    fonts = kiosk_synth.find_fonts()
    p = kiosk_synth.Painter(w, h, fonts, random.choice(kiosk_synth.PALETTES))
    fn = dict((n, f) for n, f, _ in kiosk_synth.SCREENS)[kind]
    fn(p)
    names = {v: k for k, v in kiosk_synth.CID.items()}
    labels = [(names[c], x1 / w, y1 / h, x2 / w, y2 / h) for c, x1, y1, x2, y2 in p.labels]
    return cv2.cvtColor(np.asarray(p.img), cv2.COLOR_RGB2BGR), labels


class FakeKioskCamera:
    def __init__(self, screen_bgr, seed=0, scale=0.8, tilt=0.08, shake=6.0, hand_drawer=None, condition=None):
        rng = np.random.default_rng(seed)
        self.condition = condition  # None | blur | glare | dark | shake | far | steep (CONDITIONS 참고)
        self.crng = np.random.default_rng(seed + 7)
        if condition == "shake":
            shake = 25.0
        elif condition == "far":
            scale = 0.45
        elif condition == "steep":
            tilt = 0.18
        self.screen = screen_bgr
        self.bezel = 30
        self.t0_jitter = rng.uniform(-tilt, tilt, (4, 2))
        self.phase = rng.uniform(0, 2 * math.pi, 4)
        self.scale, self.shake = scale, shake
        self.bg = self._background(rng)
        self.hand_drawer = hand_drawer  # None이면 그림으로 그린 손, RealHandCompositor면 실제 손 사진

    def _background(self, rng):
        bg = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        c1, c2 = rng.integers(40, 200, 3), rng.integers(40, 200, 3)
        for y in range(FRAME_H):
            a = y / FRAME_H
            bg[y] = (c1 * (1 - a) + c2 * a).astype(np.uint8)
        for _ in range(12):
            x, y = rng.integers(0, FRAME_W), rng.integers(0, FRAME_H)
            cv2.rectangle(bg, (int(x), int(y)), (int(x + rng.integers(30, 200)), int(y + rng.integers(30, 200))),
                          tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        return bg

    def set_screen(self, screen_bgr):
        self.screen = screen_bgr

    def corners(self, t):
        """시각 t의 화면 네 꼭짓점 (카메라 픽셀)"""
        h, w = self.screen.shape[:2]
        s = self.scale * min(FRAME_W / (w + 2 * self.bezel), FRAME_H / (h + 2 * self.bezel))
        cw, ch = w * s, h * s
        base = np.float32([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
        base += (self.t0_jitter * np.float32([cw, ch])).astype(np.float32)
        wob = np.float32([[math.sin(0.9 * t + self.phase[0]), math.cos(0.7 * t + self.phase[1])]]) * self.shake
        rot = math.radians(2.0 * math.sin(0.5 * t + self.phase[2]))
        R = np.float32([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        return base @ R.T + np.float32([FRAME_W / 2, FRAME_H / 2]) + wob

    def frame(self, t, finger_scr=None):
        """시각 t의 카메라 프레임과 정답 정보를 만든다. finger_scr: 검지 끝 화면 좌표(0~1) 또는 None"""
        h, w = self.screen.shape[:2]
        b = self.bezel
        K = np.full((h + 2 * b, w + 2 * b, 3), 25, np.uint8)
        K[b:b + h, b:b + w] = self.screen
        cs = self.corners(t)
        src_screen = np.float32([[b, b], [b + w, b], [b + w, b + h], [b, b + h]])
        M = cv2.getPerspectiveTransform(src_screen, cs)
        img = cv2.warpPerspective(K, M, (FRAME_W, FRAME_H))
        mask = cv2.warpPerspective(np.full(K.shape[:2], 255, np.uint8), M, (FRAME_W, FRAME_H))
        img = np.where(mask[..., None] > 0, img, self.bg)
        H_scr2img = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), cs)
        hand = None
        if finger_scr is not None:
            tip = cv2.perspectiveTransform(np.float32([[finger_scr]]), H_scr2img)[0, 0]
            if self.hand_drawer is None:
                hand = draw_hand(img, tip)
            else:
                screen_h = float(np.linalg.norm(cs[3] - cs[0]))
                hand = self.hand_drawer(img, tip, screen_h)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = degrade(img, self.condition, t, self.crng)
        return img, {"corners": cs, "H_scr2img": H_scr2img, "hand": hand}


CONDITIONS = {
    None: "기본",
    "blur": "초점 흐림 + 움직임 번짐",
    "glare": "화면 반사광(움직이는 밝은 얼룩)",
    "dark": "어두운 매장 + 센서 노이즈",
    "shake": "손떨림 4배",
    "far": "멀리서 촬영(화면이 프레임의 45%)",
    "steep": "비스듬히 촬영(원근 2배)",
}


def degrade(img, condition, t, rng):
    """촬영 조건 열화 (견고성 평가용)"""
    if condition == "blur":
        k = 9
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k  # 가로 움직임 번짐
        img = cv2.filter2D(cv2.GaussianBlur(img, (0, 0), 2.0), -1, ker)
    elif condition == "glare":
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w * (0.5 + 0.3 * math.sin(0.6 * t)), h * (0.4 + 0.2 * math.cos(0.4 * t))
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.18 * w) ** 2)))
        img = np.clip(img.astype(np.float32) + (g * 170)[..., None], 0, 255).astype(np.uint8)
    elif condition == "dark":
        a = img.astype(np.float32) * 0.35 + rng.normal(0, 8, img.shape)
        img = np.clip(a, 0, 255).astype(np.uint8)
    return img


def draw_hand(img, tip):
    """검지 끝이 tip(픽셀)에 오는 손 모양을 그리고 관절 21점(픽셀)을 돌려준다"""
    tip = np.float32(tip)
    wrist = tip + np.float32([60, 330])
    skin = (150, 180, 225)
    cv2.circle(img, tuple(int(v) for v in (wrist + np.float32([0, -60]))), 70, skin, -1)
    pts = np.zeros((21, 2), np.float32)
    pts[0] = wrist
    # 검지(5~8)는 tip까지 곧게, 나머지 손가락은 접힌 채 손바닥 근처
    for j, a in zip(range(5, 9), np.linspace(0.45, 1.0, 4)):
        pts[j] = wrist + (tip - wrist) * a
    cv2.line(img, tuple(int(v) for v in pts[5]), tuple(int(v) for v in tip), skin, 26)
    cv2.circle(img, tuple(int(v) for v in tip), 13, skin, -1)
    for f, off in zip((1, 9, 13, 17), (-70, 25, 50, 75)):
        for k in range(4):
            pts[f + k] = wrist + np.float32([off * 0.8, -80 - 15 * k])
    return pts


class RealHandCompositor:
    """실제 손 사진(검지를 편 손)을 잘라 검지 끝이 원하는 위치에 오도록 프레임에 붙인다.

    손 사진·관절 정답은 Ultralytics hand-keypoints 데이터셋(관절 21점, MediaPipe 순서)을 쓴다.
      python scripts/eval_real_hand.py --hands /경로/hand-keypoints  (데이터셋은 레포에 포함하지 않음)
    손 모양 마스크는 관절점으로 만든다: 손가락 뼈마디를 굵은 선으로, 손바닥은 볼록 다각형으로 칠한 뒤 가장자리를 흐린다.
    """

    def __init__(self, hands_dir, seed=0, max_hands=200, mode="rect", scale=0.33):
        import glob

        self.mode, self.scale = mode, scale
        rng = random.Random(seed)
        items = []
        for f in sorted(glob.glob(os.path.join(hands_dir, "labels", "*", "*.txt"))):
            rows = [r for r in open(f).read().split("\n") if r.strip()]
            if len(rows) != 1:
                continue
            v = np.array(rows[0].split()[5:], float).reshape(21, 3)
            if (v[:, 2] == 0).any():
                continue
            p = v[:, :2]
            idx, mid, palm = (np.linalg.norm(p[8] - p[5]), np.linalg.norm(p[12] - p[9]), np.linalg.norm(p[9] - p[0]))
            # 검지만 펴서 위를 가리키는 손 (검지 끝이 가장 위, 가운데 손가락은 접힘)
            if p[8, 1] > p[:, 1].min() + 1e-6 or idx < 0.9 * palm or mid > 0.6 * idx:
                continue
            if p[8, 1] < 0.03 or p[0, 1] > 0.97:
                continue
            img_path = f.replace("labels", "images").replace(".txt", ".jpg")
            # 사무실 배경 사진(엄지척 등 라벨이 틀린 경우가 많음)은 빼고, 단색 벽 앞 손 사진만 쓴다.
            # 직접 눈으로 확인한 60장에서 경계선 밀도 11 미만이 '검지를 편 손'과 정확히 일치했다.
            g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if g is None or cv2.Canny(g, 60, 150).mean() >= 11:
                continue
            items.append((img_path, p))
        rng.shuffle(items)
        self.items = items[:max_hands]
        if not self.items:
            raise ValueError(f"검지를 편 손 사진을 찾지 못했습니다: {hands_dir}")
        self.rng = rng
        self.current = None

    def pick(self):
        self.current = self.rng.randrange(len(self.items))

    def __call__(self, img, tip_px, screen_h_px):
        if self.current is None:
            self.pick()
        path, p = self.items[self.current]
        src = cv2.imread(path)
        h, w = src.shape[:2]
        P = p * [w, h]
        if self.mode == "mask":
            # 마스크: 손가락 뼈마디 선 + 손바닥 다각형 (배경을 지운 손 모양)
            mask = np.zeros((h, w), np.uint8)
            width = max(6, int(0.7 * np.linalg.norm(P[5] - P[9])))
            for chain in ([0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
                for a_, b_ in zip(chain, chain[1:]):
                    cv2.line(mask, tuple(P[a_].astype(int)), tuple(P[b_].astype(int)), 255, width)
                cv2.circle(mask, tuple(P[chain[-1]].astype(int)), width // 2, 255, -1)
            cv2.fillConvexPoly(mask, cv2.convexHull(P[[0, 1, 2, 5, 9, 13, 17]].astype(np.int32)), 255)
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
        else:
            # 사진 전체를 붙이되 가장자리 12%만 서서히 흐리게 (손 주변 맥락을 그대로 유지)
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32) / (0.12 * min(h, w))
            alpha = np.clip(d, 0, 1)
        # 크기: 손목~검지 끝이 화면 높이의 약 1/3 (키오스크 앞에서 손을 뻗었을 때)
        s = self.scale * screen_h_px / max(np.linalg.norm(P[8] - P[0]), 1.0)
        M = np.float32([[s, 0, tip_px[0] - s * P[8, 0]], [0, s, tip_px[1] - s * P[8, 1]]])
        H_, W_ = img.shape[:2]
        warped = cv2.warpAffine(src, M, (W_, H_), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(alpha, M, (W_, H_), flags=cv2.INTER_LINEAR)[..., None]
        img[:] = (warped * a + img * (1 - a)).astype(np.uint8)
        return (P * s + M[:, 2]).astype(np.float32)


class FakeHandSource:
    """FakeKioskCamera.frame()이 그린 손의 관절을 그대로 돌려준다 (MediaPipe 대체)"""

    def __init__(self):
        self.current = None

    def __call__(self, frame, t):
        return [] if self.current is None else [(self.current, 0.95)]


STEP = {"far": 0.035, "near": 0.012}  # 가짜 사용자가 안내 한 번에 손을 옮기는 거리(화면 높이 대비)
VEC = {"right": (1, 0), "up_right": (1, -1), "up": (0, -1), "up_left": (-1, -1), "left": (-1, 0),
       "down_left": (-1, 1), "down": (0, 1), "down_right": (1, 1)}


def _est_to_true(plane, H_scr2img):
    """추정 화면 좌표의 박스 → 실제 화면 좌표의 박스 (시뮬레이터만 정답 H를 알기 때문에 가능)"""
    Hinv = np.linalg.inv(H_scr2img)

    def f(box):
        x1, y1, x2, y2 = box
        img = plane.to_image([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        ph = np.hstack([img, np.ones((4, 1))]) @ Hinv.T
        q = ph[:, :2] / ph[:, 2:3]
        return (*q.min(0), *q.max(0))

    return f


def run_scenario(pipe, cam, hand, next_screen, labels, seed=0, fps=15, max_s=20.0, pick=None, on_frame=None):
    """
    가짜 사용자가 안내를 따라 목표 버튼을 누르는 시나리오 1회.
      pick(elements, rng) → 목표 Element (기본: 메뉴·버튼 중 무작위)
      on_frame(img, result, pipe) → 프레임마다 호출 (영상 저장 등)
    return dict: pressed_at, press_inside_gt, verdict, events, plane_err, elem_recall
    """
    from .elements import iou

    rng = random.Random(seed)
    finger = pressed_at = target_gt = None
    log = {"events": [], "verdict": None, "plane_err": [], "pressed_at": None, "press_inside_gt": None,
           "elem_recall": None, "target": None}
    for i in range(int(max_s * fps)):
        t = i / fps
        img, info = cam.frame(t, finger)
        if hand is not None:  # 가짜 손이면 정답 관절을 넘기고, 실제 MediaPipe면 영상만 보고 찾는다
            hand.current = info["hand"]
        r = pipe.process(img, t)
        if on_frame:
            on_frame(img, r, pipe)
        if r.plane is not None:
            sw = np.linalg.norm(info["corners"][1] - info["corners"][0])
            log["plane_err"].append(float(np.linalg.norm(r.plane.corners - info["corners"], axis=1).max() / sw))
        if r.keyframe and pipe.guide.target is None and pressed_at is None:
            # 검출 박스는 '추정한' 화면 좌표라 정답(실제 화면 좌표)과 비교하려면 카메라 영상을 거쳐 옮겨야 한다
            to_true = _est_to_true(r.plane, info["H_scr2img"])
            true_boxes = [to_true(e.box) for e in pipe.elements]
            # F-03 재현율: 정답 박스 중 같은 종류가 IoU 0.5 이상으로 잡힌 비율
            hit = sum(any(e.kind == k and iou(tb, b) >= 0.5 for e, tb in zip(pipe.elements, true_boxes))
                      for k, *b in labels)
            log["elem_recall"] = hit / max(1, len(labels))
            cands = [e for e in pipe.elements if e.kind in ("menu", "button")]
            tgt = pick(cands, rng) if pick else rng.choice(cands)
            pipe.set_target(tgt.id, expect={"changed": True})
            log["target"] = (tgt.id, tgt.kind)
            tb = to_true(tgt.box)
            target_gt = max(labels, key=lambda lb: iou(lb[1:], tb) if lb[0] == tgt.kind else 0)
            finger = (rng.uniform(0.3, 0.7), 0.97)  # 손은 화면 아래쪽에서 들어온다
        ev = r.event
        if ev is not None:
            log["events"].append((round(t, 3), ev.type, ev.dir, ev.distance, ev.speak))
            if ev.type == "direction" and ev.dir and finger is not None:
                vx, vy = VEC[ev.dir]
                n = math.hypot(vx, vy)
                # 사람 손은 정확하지 않다: 방향 ±15도, 거리 ±30% 흔들림
                ang = math.atan2(vy, vx) + math.radians(rng.uniform(-15, 15))
                s = STEP[ev.distance] * rng.uniform(0.7, 1.3)
                finger = (finger[0] + math.cos(ang) * s / pipe.guide.aspect, finger[1] + math.sin(ang) * s)
            if ev.type in ("no_hand", "reset") and finger is not None and ev.speak:
                # "검지를 화면 앞으로" / "가운데에서 다시" → 손을 화면 가운데 쪽으로 옮긴다
                finger = (finger[0] + (0.5 - finger[0]) * 0.5, finger[1] + (0.5 - finger[1]) * 0.5)
            if ev.type == "press" and pressed_at is None:
                pressed_at = t
                x1, y1, x2, y2 = target_gt[1:]
                log["press_inside_gt"] = bool(x1 <= finger[0] <= x2 and y1 <= finger[1] <= y2)
        if pressed_at is not None and finger is not None and t - pressed_at >= 0.2:
            cam.set_screen(next_screen)  # 눌러서 키오스크 화면이 바뀜
            finger = None
        if r.verdict is not None:
            log["verdict"] = (round(t, 3), r.verdict.result)
            break
    log["pressed_at"] = pressed_at
    return log
