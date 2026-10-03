"""AIHub 관광 음식메뉴판(71553) 실사진에서 글자 줄을 잘라 인식 모델만 비교한다 — 첫 '진짜 카메라' 측정.
라벨 ocr 박스(x,y,w,h = 원본 대비 %)를 축 정렬로 자르고, 3도 넘게 기운 박스는 뺀다.
  python real_menu_eval.py --src D:/sonkkeutgil/data_aihub/menu_gj --out data/real_menu_gj
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ocr_m3 import REC_MODEL


def build(src, out):
    (out / "crops").mkdir(parents=True, exist_ok=True)
    labels, skip = [], {"rot": 0, "size": 0, "tiny": 0}
    for jf in sorted((src / "labels").glob("*.json")):
        d = json.load(open(jf, encoding="utf-8"))
        m = d["meta"]
        ip = src / "images" / m["file_name"].replace(".jpg", "-tf.jpg")
        if not ip.exists():
            continue
        im = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")
        W, H = m["image_original_width"], m["image_original_height"]
        if im.size != (W, H):
            skip["size"] += 1
            continue
        for k, a in enumerate(d["annotations"]):
            o = a.get("ocr")
            if not o or not o.get("text"):
                continue
            if abs(o.get("rotation") or 0) > 3:
                skip["rot"] += 1
                continue
            x1, y1 = o["x"] / 100 * W, o["y"] / 100 * H
            x2, y2 = x1 + o["width"] / 100 * W, y1 + o["height"] / 100 * H
            if x2 - x1 < 8 or y2 - y1 < 8:
                skip["tiny"] += 1
                continue
            name = f"{jf.stem}_{k:03d}.jpg"
            im.crop((int(x1), int(y1), int(x2), int(y2))).save(out / "crops" / name, quality=92)
            labels.append((name, o["text"]))
    open(out / "labels.txt", "w", encoding="utf-8").write("\n".join(f"crops/{n}\t{t}" for n, t in labels) + "\n")
    print(f"crops {len(labels)}  skip {skip}")
    return labels


def norm(s):
    return s.replace(" ", "")


def run(model_dir, labels, out):
    from paddleocr import TextRecognition
    rec = TextRecognition(model_name=REC_MODEL, model_dir=model_dir)
    ok, preds = 0, []
    paths = [str(out / "crops" / n) for n, _ in labels]
    for i in range(0, len(paths), 64):
        for r in rec.predict(paths[i:i + 64], batch_size=64):
            preds.append(r["rec_text"])
    for (n, t), p in zip(labels, preds):
        ok += norm(p) == norm(t)
    return ok / len(labels), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/sonkkeutgil/data_aihub/menu_gj")
    ap.add_argument("--out", default="data/real_menu_gj")
    ap.add_argument("--models", default="base,models/kiosk_rec_v1")
    a = ap.parse_args()
    out = Path(a.out)
    labels = build(Path(a.src), out)
    dict_chars = set(open("PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt", encoding="utf-8").read().split("\n")) | {" "}
    oov = sum(any(c not in dict_chars for c in t) for _, t in labels)
    print(f"사전에 없는 글자가 섞인 줄 {oov}/{len(labels)} (이 줄들은 원래 맞힐 수 없음)")
    res, allp = {}, {}
    for m in a.models.split(","):
        acc, preds = run(None if m == "base" else m, labels, out)
        res[m], allp[m] = round(acc, 4), preds
        print(f"{m}: 줄 정답률 {acc * 100:.1f}%")
    ms = list(allp)
    rows = ["file\tgt\t" + "\t".join(ms)]
    for i, (n, t) in enumerate(labels):
        if any(norm(allp[m][i]) != norm(t) for m in ms):
            rows.append(f"{n}\t{t}\t" + "\t".join(allp[m][i] for m in ms))
    Path("runs").mkdir(exist_ok=True)
    open("runs/errors_real_menu_gj.tsv", "w", encoding="utf-8").write("\n".join(rows))
    json.dump({"n": len(labels), "oov_lines": oov, "acc": res}, open("runs/eval_real_menu_gj.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
