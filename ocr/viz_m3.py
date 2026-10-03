"""인식 결과 시각화 — 같은 화면을 공식 가중치 / 튜닝 모델로 읽어 나란히 그린다.
초록 = 정답과 일치, 빨강 = 틀림, 주황 = 맞았지만 uncertain 표시.
  python viz_m3.py --rec-dir models/kiosk_rec_v1_mid --out runs/viz
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_m3 import KioskOCR

F = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 20)
FH = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 30)


def norm(s):
    return (s or "").replace(" ", "")


def draw(img, gt, pred, title):
    h, w = img.shape[:2]
    im = Image.fromarray(img[:, :, ::-1]).convert("RGB")
    d = ImageDraw.Draw(im)
    ok_n = 0
    for g, p in zip(gt["elements"], pred):
        if g["kind"] == "price":
            ok = p.get("price") == g.get("price")
            shown = f"{p.get('price')}"
        else:
            ok = norm(p["text"]) == norm(g["text"]) and (g.get("price") is None or p.get("price") == g["price"])
            shown = p["text"] or "(못 읽음)"
            if p.get("price"):
                shown += f" / {p['price']}"
        ok_n += ok
        col = (255, 150, 0) if ok and p["uncertain"] else (0, 200, 0) if ok else (230, 0, 0)
        x1, y1, x2, y2 = g["box"][0] * w, g["box"][1] * h, g["box"][2] * w, g["box"][3] * h
        d.rectangle((x1, y1, x2, y2), outline=col, width=4)
        tb = d.textbbox((x1 + 4, y2 - 28), shown, font=F)
        d.rectangle((tb[0] - 3, tb[1] - 3, tb[2] + 3, tb[3] + 3), fill=col)
        d.text((x1 + 4, y2 - 28), shown, font=F, fill=(255, 255, 255))
    n = len(gt["elements"])
    head = Image.new("RGB", (w, 56), (25, 25, 25))
    ImageDraw.Draw(head).text((14, 10), f"{title}   {ok_n}/{n} 정답", font=FH, fill=(255, 255, 255))
    out = Image.new("RGB", (w, h + 56))
    out.paste(head, (0, 0)), out.paste(im, (0, 56))
    return np.array(out)[:, :, ::-1], ok_n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", required=True)
    ap.add_argument("--out", default="runs/viz")
    ap.add_argument("--picks", default="synth_val:2,synth_val:5,synth_val:29,synth_unseen:2,synth_unseen:4,synth_unseen:8")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base, tuned = KioskOCR(), KioskOCR(rec_model_dir=a.rec_dir)
    for pick in a.picks.split(","):
        ds, i = pick.split(":")
        jf = Path("data") / ds / "images" / f"{int(i):05d}.json"
        gt = json.load(open(jf, encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(jf.with_suffix(".jpg")), np.uint8), cv2.IMREAD_COLOR)
        els = [{k: e[k] for k in ("id", "kind", "box")} for e in gt["elements"]]
        lv = ["깨끗", "보통", "나쁨"][gt["degrade"]]
        A, an, n = draw(img, gt, base.read_screen(img, els), "공식 가중치")
        B, bn, _ = draw(img, gt, tuned.read_screen(img, els), "튜닝 모델")
        gap = np.full((A.shape[0], 16, 3), 255, np.uint8)
        pair = np.hstack([A, gap, B])
        name = f"{ds}_{int(i):05d}_{lv}.jpg"
        cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out / name))
        print(f"{name}: 공식 {an}/{n} -> 튜닝 {bn}/{n}")


if __name__ == "__main__":
    main()
