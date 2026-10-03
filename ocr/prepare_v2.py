"""v2 인식 학습 데이터 — v1 이 합성에 과적합돼 실사진에서 나빠진 것(91.3 → 83.5%)을 고치기 위해 실데이터를 섞는다.

  실사진  : 대전·울산 메뉴판(71553) 글자 줄 → 가게 단위 85/15 학습/검증, 야외 간판(105) 글자 → 이미지 단위 90/10
  합성    : v1 키오스크 줄 40k 로 축소 + 일반 줄 30k (소문자·기호·실제 메뉴/상품 어휘)
  시험    : 광주 메뉴판(data/real_menu_gj)은 학습·검증 어디에도 넣지 않는다
결과: data/v2_train.txt, data/v2_val.txt (data_dir=../data/ 기준 경로)
"""
import csv
import glob
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import make_synth as ms
from real_menu_eval import build

AIHUB = Path("D:/sonkkeutgil/aihub/71553")
RAW = Path("D:/sonkkeutgil/data_aihub")
DATA = Path("data")
random.seed(7), np.random.seed(7)
DICT = set(open("PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt", encoding="utf-8").read().split("\n")) | {" "}


def ok_text(t):
    return 0 < len(t) <= 25 and all(c in DICT for c in t)


def extract(tag, dst):
    for z in AIHUB.rglob(f"*_{tag}.*.zip"):
        sub = "images" if Path(z).name.startswith("VS") else "labels"
        out = RAW / dst / sub
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            continue
        for i in zipfile.ZipFile(z).infolist():
            if not i.is_dir() and i.file_size:
                (out / Path(i.filename).name).write_bytes(zipfile.ZipFile(z).read(i))


def real_menu(region):
    out = DATA / f"real_menu_{region}"
    labels = build(RAW / f"menu_{region}", out)
    by_store = {}
    for n, t in labels:
        if ok_text(t):
            by_store.setdefault(n.split("_")[3], []).append(f"{out.name}/crops/{n}\t{t}")
    stores = sorted(by_store)
    random.shuffle(stores)
    k = max(1, int(len(stores) * 0.15))
    val = [l for s in stores[:k] for l in by_store[s]]
    tr = [l for s in stores[k:] for l in by_store[s]]
    print(f"menu {region}: 가게 {len(stores)}  train {len(tr)}  val {len(val)}")
    return tr, val


def outdoor():
    out = DATA / "real_outdoor" / "crops"
    out.mkdir(parents=True, exist_ok=True)
    tr, val = [], []
    for jf in sorted(glob.glob(str(RAW / "outdoor_add/labels/*.json"))):
        d = json.load(open(jf, encoding="utf-8"))
        ip = RAW / "outdoor_add/images" / d["images"][0]["file_name"]
        img = None
        dst = val if random.random() < 0.1 else tr
        for a in d["annotations"]:
            t = a["text"].strip()
            x, y, w, h = a["bbox"]
            if t == "xxx" or not ok_text(t) or w < 8 or h < 8:
                continue
            if img is None:
                img = cv2.imdecode(np.fromfile(str(ip), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    break
            name = f"o{Path(jf).stem.split('_')[-1]}_{a['id']:03d}.jpg"
            c = img[max(0, y):y + h, max(0, x):x + w]
            if c.size == 0:
                continue
            cv2.imencode(".jpg", c)[1].tofile(str(out / name))
            dst.append(f"real_outdoor/crops/{name}\t{t}")
    print(f"outdoor: train {len(tr)}  val {len(val)}")
    return tr, val


def vocab():
    """일반 줄에 쓸 실제 어휘: 대전·울산 메뉴 라벨 메뉴명 + 주문 QA 상품명 (광주는 시험셋이라 제외)."""
    words = set()
    for region in ("dj", "us"):
        for jf in glob.glob(str(RAW / f"menu_{region}/labels/*.json")):
            for a in json.load(open(jf, encoding="utf-8"))["annotations"]:
                w = (a.get("menu_information") or {}).get("ko") or ""
                if ok_text(w):
                    words.add(w)
    for f in glob.glob(str(RAW / "order_qa/train/*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            for w in (r.get("상품명") or "").split(","):
                w = w.strip()
                if ok_text(w):
                    words.add(w)
    print(f"실제 어휘 {len(words)}개")
    return sorted(words)


LOWER = "abcdefghijklmnopqrstuvwxyz"
SYM = ["~", "+", "-", "/", "&", ".", "(", ")", "·", ":", "%", "!", "#"]


def generic_text(words):
    r = random.random()
    if r < 0.45:
        t = random.choice(words)
        if random.random() < 0.3:
            t += random.choice([f"({random.randint(1, 6)}인)", f" {random.randint(1, 20)}p", " set", " x2", " 1+1",
                                f" {random.randint(1, 3)}인분", " (ICE)", " (HOT)", " L", " R"])
    elif r < 0.6:
        t = "".join(random.choices(LOWER, k=random.randint(2, 8)))
        if random.random() < 0.5:
            t = t.capitalize()
    elif r < 0.72:
        a, b = random.randint(1, 15), random.randint(2, 20)
        t = random.choice([f"{a}~{b}세", f"{a}+{b}", f"{a}/{b}", f"{a}-{b}", f"{a}:{b:02d}", f"{a}%", f"{a}.{b}"])
    elif r < 0.84:
        t = random.choice(words) + random.choice(SYM) + random.choice(words)
        t = t[:25]
    elif r < 0.92:
        t = random.choice(["", "+", "W", "₩"]) + f"{random.randint(1, 300) * 100:,}" + random.choice(["", "원", " 원", ".-"])
    else:
        t = random.choice(ms.HANGUL)
    return ms.rec_label(t)


def generic_lines(n, words):
    out = DATA / "gen_v2" / "crops"
    out.mkdir(parents=True, exist_ok=True)
    fonts = ms.FONTS + ["C:/Windows/Fonts/malgunsl.ttf"]  # 시험용 holdout NotoSerif 는 넣지 않는다
    lines = []
    for k in range(n):
        t = generic_text(words)
        if not ok_text(t):
            continue
        bg, fg, accent, card, btn = random.choice(ms.THEMES)
        bgc, fgc = random.choice([(bg, fg), (bg, accent), (card, fg), (btn, (255, 255, 255)),
                                  ((255, 255, 255), (0, 0, 0)), ((20, 20, 20), (255, 255, 255))])
        f = ms.font(random.randint(24, 60), random.choice(fonts))
        x1, y1, x2, y2 = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), t, font=f)
        pad = 40
        im = Image.new("RGB", (x2 - x1 + 2 * pad, y2 - y1 + 2 * pad), bgc)
        ImageDraw.Draw(im).text((pad - x1, pad - y1), t, font=f, fill=fgc)
        arr = ms.degrade(np.array(im)[:, :, ::-1].copy(), random.choice([0, 1, 1, 2]))
        c = ms.crop_line(arr, (pad, pad, pad + x2 - x1, pad + y2 - y1), 0)
        if c is None:
            continue
        name = f"v{k:06d}.jpg"
        cv2.imencode(".jpg", c)[1].tofile(str(out / name))
        lines.append(f"gen_v2/crops/{name}\t{t}")
    print(f"generic v2: {len(lines)}")
    return lines


def main():
    extract("06", "menu_dj"), extract("07", "menu_us")
    dj_tr, dj_val = real_menu("dj")
    us_tr, us_val = real_menu("us")
    od_tr, od_val = outdoor()
    gen = generic_lines(30000, vocab())
    syn = open(DATA / "rec_train/labels_clean.txt", encoding="utf-8").read().splitlines()
    syn = [l for l in random.sample(syn, 40000) if ok_text(l.split("\t", 1)[1])]
    real = dj_tr + us_tr
    train = syn + gen + real * 3 + od_tr * 2
    random.shuffle(train)
    val = dj_val + us_val + od_val
    open(DATA / "v2_train.txt", "w", encoding="utf-8").write("\n".join(train) + "\n")
    open(DATA / "v2_val.txt", "w", encoding="utf-8").write("\n".join(val) + "\n")
    print(f"train {len(train)} (합성 {len(syn)}, 일반 {len(gen)}, 메뉴판 {len(real)}x3, 간판 {len(od_tr)}x2)  val {len(val)}")


if __name__ == "__main__":
    main()
