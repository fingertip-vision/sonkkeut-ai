"""
파이썬(기준 구현) ↔ Kotlin(앱) 동등성 확인용 정답 데이터 만들기

  python android/parity/make_golden.py        # android/parity/golden/ 에 저장
  bash android/parity/run_tests.sh            # Kotlin 쪽 테스트 실행 (JDK 17+ 필요)

파이썬 sonkkeut_vision의 각 부품에 같은 입력을 넣고 출력(정답)을 JSON으로 남긴다.
Kotlin 테스트는 같은 입력을 Kotlin 코드에 넣어 출력이 같은지 비교한다.
  - guide     : 무작위 손끝 움직임 시퀀스 → 안내 이벤트 (유형·방향·거리·음성·진동)
  - keyframe  : 작은 화면 이미지 시퀀스 + 손 영역 → 키프레임 여부
  - verify    : 누르기 전·후 화면 구조 + 기대 결과 → 판정
  - detect    : M2 ONNX 원시 출력 → Ultralytics가 낸 박스 (Kotlin 해석기와 비교)
  - pose      : M1 ONNX 원시 출력 → Ultralytics가 낸 꼭짓점
  - elements  : 검출 박스 → 화면 요소(겹침 정리, 읽는 순서, parent)
  - geometry  : 꼭짓점 → 호모그래피·화면 비율, 꼭짓점 안내 문구, 검지 자세 점수
"""
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from sonkkeut_vision.elements import clean_overlaps, reading_order, containment  # noqa: E402
from sonkkeut_vision.fingertip import pointing_score  # noqa: E402
from sonkkeut_vision.guidance import Guide  # noqa: E402
from sonkkeut_vision.keyframe import KeyframeDetector  # noqa: E402
from sonkkeut_vision.plane import ScreenPlaneEstimator, homography_from_corners  # noqa: E402
from sonkkeut_vision.schema import Element, Fingertip  # noqa: E402
from sonkkeut_vision.verify import PressVerifier  # noqa: E402

OUT = os.path.join(HERE, "golden")


def ev_dict(ev):
    if ev is None:
        return None
    return {"type": ev.type, "dir": ev.dir, "distance": ev.distance, "speak": ev.speak, "vibe_hz": ev.vibe_hz}


def gen_guide(n_seq=40, steps=90, seed=0):
    rng = random.Random(seed)
    cases = []
    for s in range(n_seq):
        x1, y1 = rng.uniform(0.05, 0.7), rng.uniform(0.05, 0.8)
        box = [x1, y1, x1 + rng.uniform(0.05, 0.28), y1 + rng.uniform(0.04, 0.16)]
        aspect = rng.choice([1.0, 0.5625, 1.7778, rng.uniform(0.4, 2.0)])
        conf = rng.choice([0.9, 0.9, 0.9, 0.4])
        g = Guide()
        g.set_target(Element("e7", "menu", tuple(box), conf), aspect)
        x, y = rng.uniform(0, 1), rng.uniform(0.6, 1.0)
        tx, ty = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        steps_out = []
        t = 0.0
        lost = 0.0
        for i in range(steps):
            t += rng.choice([1 / 15, 1 / 15, 1 / 15, 0.1, 0.2])
            r = rng.random()
            if r < 0.06:
                lost += 0.3
                tip = {"pos": None, "conf": 0.0, "lost_for": lost, "pointing": 1.0}
            else:
                lost = 0.0
                # 목표 쪽으로 다가가되 흔들림
                x += (tx - x) * rng.uniform(0.0, 0.35) + rng.gauss(0, 0.02)
                y += (ty - y) * rng.uniform(0.0, 0.35) + rng.gauss(0, 0.02)
                pointing = 1.0 if rng.random() > 0.05 else rng.uniform(0, 0.6)
                tip = {"pos": [x, y], "conf": rng.choice([0.95, 0.95, 0.95, 0.3]), "lost_for": 0.0, "pointing": pointing}
            tconf = None if rng.random() > 0.05 else rng.uniform(0.2, 0.9)
            ft = Fingertip(pos=tuple(tip["pos"]) if tip["pos"] else None, conf=tip["conf"], lost_for=tip["lost_for"],
                           pointing=tip["pointing"])
            ev = g.update(ft, t, target_conf=tconf)
            steps_out.append({"t": t, "tip": tip, "target_conf": tconf, "event": ev_dict(ev)})
        cases.append({"box": box, "aspect": aspect, "conf": conf, "steps": steps_out})
    return cases


def gen_keyframe(n_seq=12, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for s in range(n_seq):
        h, w = (96, 54) if s % 2 == 0 else (54, 96)

        def screen(k):
            r = np.random.default_rng(1000 * s + k)
            img = np.full((h, w), r.integers(180, 250), np.uint8)
            for _ in range(10):
                y, x = r.integers(0, h - 10), r.integers(0, w - 10)
                img[y:y + r.integers(4, 12), x:x + r.integers(4, 14)] = r.integers(0, 200)
            return img

        kf = KeyframeDetector()
        cur = 0
        frames = []
        for i in range(40):
            if rng.random() < 0.08:
                cur += 1
            img = screen(cur).copy()
            hands = None
            if rng.random() < 0.4:  # 손가락이 화면을 가림
                bx, by = rng.uniform(0, 0.7), rng.uniform(0, 0.6)
                X1, Y1 = int(bx * w), int(by * h)
                img[Y1:Y1 + h // 3, X1:X1 + w // 5] = 200
                hands = [[bx, by, bx + 0.2, by + 0.33]]
            img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
            force = bool(rng.random() < 0.03)
            is_kf, change = kf.update(img, hands, force=force)
            changed = kf.changed_since_key(img, hands)
            frames.append({"img": img.flatten().tolist(), "w": w, "h": h, "hands": hands, "force": force,
                           "is_kf": bool(is_kf), "change": float(change), "changed_since": bool(changed)})
        cases.append(frames)
    return cases


def gen_verify():
    S = lambda t, **kw: {"screen_type": t, "elements": [], **kw}  # noqa: E731
    combos = [
        (S("menu"), {"screen_type": "option"}, S("option")),
        (S("menu"), {"screen_type": "option"}, S("cart")),
        (S("option", cart_count=1), {"cart_delta": 1, "success_speak": "담겼습니다"}, S("menu", cart_count=2)),
        (S("option", cart_count=1), {"cart_delta": 1}, S("menu", cart_count=1)),
        (S("menu"), {"screen_type": "option"}, S("unknown")),
        (S("cart"), {"screen_type": "payment"}, S("start")),
        (S("menu"), None, S("option")),
        (S("menu"), {"screen_type_not": "menu"}, S("menu")),
        (S("option"), {"selected": "e12"}, S("option", selected=["e12", "e3"])),
        (S("option"), {"selected": "e12"}, S("option")),
        (S("unknown"), {"changed": True, "screen_type": "cart"}, S("unknown")),
    ]
    out = []
    for before, exp, after in combos:
        v = PressVerifier()
        v.arm(0.0, before, exp)
        r = v.judge(after)
        out.append({"before": before, "expect": exp, "after": after, "result": r.result, "reason": r.reason, "speak": r.speak})
    return out


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, left = round(dh - 0.1), round(dw - 0.1)
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = rs
    return out


def gen_models(n=6):
    import onnxruntime as ort
    from ultralytics import YOLO

    from sonkkeut_vision.sim import FakeKioskCamera, render

    os.makedirs(os.path.join(OUT, "bin"), exist_ok=True)
    mob = os.path.join(ROOT, "android", "react-native-sonkkeut", "android", "src", "main", "assets", "sonkkeut")
    det_cases, pose_cases = [], []
    s2 = ort.InferenceSession(os.path.join(mob, "m2_screen_elements_int8.onnx"), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(os.path.join(mob, "m1_screen_corners_int8.onnx"), providers=["CPUExecutionProvider"])
    y2 = YOLO(os.path.join(mob, "m2_screen_elements_int8.onnx"), task="detect")
    y1 = YOLO(os.path.join(mob, "m1_screen_corners_int8.onnx"), task="pose")
    for i in range(n):
        kind = ["menu", "option", "cart"][i % 3]
        screen, _ = render(kind, seed=500 + i)
        if i % 2:
            screen = cv2.resize(screen, (960, 540)) if i % 4 == 1 else screen[: screen.shape[0] * 3 // 4]
        x = letterbox(screen)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = s2.run(None, {s2.get_inputs()[0].name: x})[0]
        name = f"bin/m2_{i}.f32"
        raw.astype("<f4").tofile(os.path.join(OUT, name))
        r = y2.predict(screen, imgsz=640, conf=0.3, verbose=False)[0]
        det_cases.append({"raw": name, "shape": list(raw.shape), "w": screen.shape[1], "h": screen.shape[0],
                          "boxes": r.boxes.xyxy.cpu().numpy().tolist(), "cls": r.boxes.cls.cpu().numpy().astype(int).tolist(),
                          "conf": r.boxes.conf.cpu().numpy().tolist()})
        cam = FakeKioskCamera(screen if screen.shape[0] > screen.shape[1] else render(kind, seed=600 + i)[0], seed=i)
        frame, info = cam.frame(0.0, None)
        x = letterbox(frame)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = s1.run(None, {s1.get_inputs()[0].name: x})[0]
        name = f"bin/m1_{i}.f32"
        raw.astype("<f4").tofile(os.path.join(OUT, name))
        r = y1.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        j = int(np.argmax(r.boxes.conf.cpu().numpy()))
        pose_cases.append({"raw": name, "shape": list(raw.shape), "w": frame.shape[1], "h": frame.shape[0],
                           "kp": r.keypoints.xy[j].cpu().numpy().tolist(), "kc": r.keypoints.conf[j].cpu().numpy().tolist(),
                           "score": float(r.boxes.conf[j])})
    return det_cases, pose_cases


def gen_elements(n=30, seed=0):
    rng = random.Random(seed)
    cases = []
    names = ["tab", "menu", "price", "button", "back"]
    for _ in range(n):
        w, h = rng.choice([(540, 960), (960, 540), (600, 960)])
        margin = 0.04
        dets = []
        # 메뉴 격자 + 안의 가격 + 버튼, 일부는 중복·바깥
        for r_ in range(rng.randint(1, 4)):
            for c in range(rng.randint(1, 3)):
                x1, y1 = 40 + c * w * 0.3, 100 + r_ * h * 0.18
                mb = [x1, y1, x1 + w * 0.27, y1 + h * 0.16]
                dets.append((mb, 1, rng.uniform(0.4, 0.99)))
                pb = [mb[0] + 10, mb[3] - 30, mb[0] + 80, mb[3] - 8]
                dets.append((pb, 2, rng.uniform(0.4, 0.99)))
                if rng.random() < 0.3:
                    dets.append(([v + rng.uniform(-5, 5) for v in mb], rng.choice([1, 3]), rng.uniform(0.3, 0.9)))
        dets.append(([5, 2, 60, 20], 4, 0.8))
        dets.append(([w * 0.6, h * 0.92, w * 0.95, h * 0.99], 3, 0.9))
        if rng.random() < 0.5:
            dets.append(([0, 0, 10, 8], 3, 0.5))  # 바깥 쪽 (margin 영역)
        k, m = 1 + 2 * margin, margin
        raw = []
        for b, c, s in dets:
            box = (b[0] / w * k - m, b[1] / h * k - m, b[2] / w * k - m, b[3] / h * k - m)
            if box[2] <= 0 or box[3] <= 0 or box[0] >= 1 or box[1] >= 1:
                continue
            raw.append((box, names[c], float(s)))
        ordered = reading_order(clean_overlaps(raw))
        els = [Element(f"e{i + 1}", kk, tuple(bb), ss) for i, (bb, kk, ss) in enumerate(ordered)]
        for e in els:
            if e.kind != "price":
                continue
            best = max((mm for mm in els if mm.kind == "menu"), key=lambda mm: containment(e.box, mm.box), default=None)
            if best is not None and containment(e.box, best.box) > 0.6:
                e.parent = best.id
        cases.append({"w": w, "h": h, "margin": margin,
                      "dets": [{"box": list(b), "cls": c, "score": s} for b, c, s in dets],
                      "elements": [{"id": e.id, "kind": e.kind, "box": list(e.box), "parent": e.parent} for e in els]})
    return cases


def gen_geometry(seed=0):
    rng = np.random.default_rng(seed)
    homs = []
    for _ in range(20):
        c = np.float32([[100, 120], [600, 90], [640, 900], [80, 860]]) + rng.uniform(-60, 60, (4, 2)).astype(np.float32)
        H, aspect = homography_from_corners(c)
        homs.append({"corners": c.tolist(), "H": H.tolist(), "aspect": aspect})
    hints = []
    for _ in range(40):
        kp = rng.uniform(-40, 1000, (4, 2)).astype(np.float32)
        kc = rng.choice([0.1, 0.9, 0.9, 0.9], 4)
        hints.append({"kp": kp.tolist(), "kc": kc.tolist(), "w": 960, "h": 1280,
                      "hint": ScreenPlaneEstimator.missing_hint(kp, kc, (1280, 960))})
    pts = []
    for _ in range(20):
        p = rng.uniform(0, 300, (21, 2)).astype(np.float32)
        pts.append({"points": p.tolist(), "score": pointing_score(p)})
    return {"homography": homs, "hints": hints, "pointing": pts}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {
        "guide": gen_guide(),
        "keyframe": gen_keyframe(),
        "verify": gen_verify(),
        "elements": gen_elements(),
        "geometry": gen_geometry(),
    }
    det, pose = gen_models()
    data["detect"], data["pose"] = det, pose
    for k, v in data.items():
        with open(os.path.join(OUT, f"{k}.json"), "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False)
    print({k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
