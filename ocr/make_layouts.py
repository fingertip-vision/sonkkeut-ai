"""학습에 쓰지 않은 키오스크 인터페이스로 평가셋 만들기 (M3 일반화 확인용, 학습에는 쓰지 않는다).

  list : 가로 1280x720, 왼쪽 세로 카테고리 + 목록형 메뉴(이름 왼쪽·가격 오른쪽) + 하단 합계/버튼
  cart : 가로 1280x720, 위 탭 + 4x2 카드 + 오른쪽 장바구니 패널(담은 항목·합계·결제)
--fonts holdout : 학습에 안 쓴 폰트(Noto Serif KR, 맑은 고딕 Semilight)
--vocab unseen  : 학습에 안 쓴 한식·분식 어휘
테마에는 학습에 없던 고대비(검정/노랑)·녹색 계열을 섞는다. 라벨 형식은 make_synth 와 같다.

  python make_layouts.py --layout list --n 150 --out data/lay_list --seed 21
  python make_layouts.py --layout cart --n 150 --out data/lay_cart_hard --seed 22 --vocab unseen --fonts holdout
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import make_synth as ms

W, H = 1280, 720
HOLDOUT_FONTS = ["C:/Windows/Fonts/NotoSerifKR-VF.ttf", "C:/Windows/Fonts/malgunsl.ttf"]
NEW_THEMES = [((0, 0, 0), (255, 230, 0), (255, 255, 255), (28, 28, 28), (0, 110, 60)),     # 고대비
              ((245, 250, 240), (20, 50, 20), (200, 30, 30), (225, 240, 220), (0, 120, 70)),  # 녹색
              ((250, 250, 250), (40, 40, 40), (210, 0, 30), (255, 255, 255), (220, 30, 40))]  # 버거집 빨강
STORE = ["손끝카페", "MENU", "오늘의 메뉴", "주문하기", "Self Order", "어서오세요"]


class Canvas:
    def __init__(self):
        ms.LINES.clear()
        self.theme = random.choice(ms.THEMES + NEW_THEMES + NEW_THEMES)
        self.img = Image.new("RGB", (W, H), self.theme[0])
        self.d = ImageDraw.Draw(self.img)
        self.els = []

    def add(self, kind, text, box, price=None):
        x1, y1, x2, y2 = box
        e = {"id": f"e{len(self.els) + 1}", "kind": kind, "text": text,
             "box": [round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4)]}
        if price is not None:
            e["price"] = price
        self.els.append(e)

    def right(self, x2, y, text, size, fill):
        f = ms.font(size)
        ms.put(self.d, (x2 - self.d.textlength(text, font=f), y), text, f, fill)

    def done(self, st):
        return np.array(self.img)[:, :, ::-1].copy(), st, self.els


def label(cur, name):
    return random.choice(ms.PREFIX) + name if cur in ("커피", "음료") else name


def draw_list():
    c = Canvas()
    bg, fg, accent, card, btn = c.theme
    d = c.d
    d.rectangle((0, 0, W, 70), fill=btn)
    ms.put(d, (24, 12), random.choice(STORE), ms.font(36), (255, 255, 255))
    tabs = list(ms.TABS)
    cur = random.choice(tabs)
    for i, t in enumerate(tabs):
        y1 = 90 + i * 100
        d.rounded_rectangle((16, y1, 236, y1 + 84), 12, fill=btn if t == cur else card)
        ms.fit_text(d, (22, y1 + 20), t, 36, 208, (255, 255, 255) if t == cur else fg)
        c.add("tab", t, (16, y1, 236, y1 + 84))
    names = random.sample(ms.TABS[cur], min(random.choice([5, 6, 7]), len(ms.TABS[cur])))
    rh = 520 // len(names)
    for k, name in enumerate(names):
        y1 = 86 + k * rh
        d.rectangle((260, y1 + 4, W - 20, y1 + rh - 4), fill=card)
        lab = label(cur, name)
        size = min(40, rh - 26)
        ms.fit_text(d, (284, y1 + (rh - size) / 2 - 6), lab, size, 640, fg, center=False)
        p = random.choice(range(2500, 9500, 500))
        c.right(W - 44, y1 + (rh - size) / 2 - 6, ms.price_str(p), size, accent)
        c.add("menu", lab, (260, y1 + 4, W - 20, y1 + rh - 4), p)
    total = random.choice(range(4500, 40000, 500))
    ms.put(d, (284, 630), f"합계 {total:,}원", ms.font(38), fg)
    c.add("price", f"합계 {total:,}원", (270, 615, 700, 690), total)
    for i, b in enumerate(random.sample(ms.BUTTONS, 2)):
        x1 = 760 + i * 250
        d.rounded_rectangle((x1, 612, x1 + 230, 700), 14, fill=btn)
        ms.fit_text(d, (x1 + 6, 632), b, 38, 218, (255, 255, 255))
        c.add("button", b, (x1, 612, x1 + 230, 700))
    return c.done("menu")


def draw_cart():
    c = Canvas()
    bg, fg, accent, card, btn = c.theme
    d = c.d
    tabs = list(ms.TABS)
    cur = random.choice(tabs)
    tw = 880 // len(tabs)
    for i, t in enumerate(tabs):
        x1 = 16 + i * tw
        d.rounded_rectangle((x1, 16, x1 + tw - 12, 80), 30, fill=btn if t == cur else card)
        ms.fit_text(d, (x1 + 6, 30), t, 32, tw - 24, (255, 255, 255) if t == cur else fg)
        c.add("tab", t, (x1, 16, x1 + tw - 12, 80))
    names = random.sample(ms.TABS[cur], min(8, len(ms.TABS[cur])))
    cw, ch = 205, 300
    for k, name in enumerate(names):
        r, col = divmod(k, 4)
        x1, y1 = 16 + col * (cw + 15), 100 + r * (ch + 12)
        d.rounded_rectangle((x1, y1, x1 + cw, y1 + ch), 14, fill=card)
        d.ellipse((x1 + 40, y1 + 14, x1 + cw - 40, y1 + 140), fill=tuple(random.randint(80, 230) for _ in range(3)))
        lab = label(cur, name)
        ms.fit_text(d, (x1 + 6, y1 + 160), lab, 30, cw - 12, fg)
        p = random.choice(range(2500, 9500, 500))
        ms.fit_text(d, (x1 + 6, y1 + 212), ms.price_str(p), 28, cw - 12, accent)
        c.add("menu", lab, (x1, y1, x1 + cw, y1 + ch), p)
    px = 905
    d.rectangle((px, 0, W, H), fill=card)
    ms.put(d, (px + 20, 20), random.choice(["주문 내역", "장바구니", "선택한 메뉴"]), ms.font(34), fg)
    total = 0
    pool = [n for v in ms.TABS.values() for n in v]
    for k, name in enumerate(random.sample(pool, random.randint(1, 4))):
        q = random.randint(1, 3)
        p = random.choice(range(2500, 9500, 500)) * q
        total += p
        y1 = 80 + k * 100
        ms.fit_text(d, (px + 20, y1 + 8), f"{name} x{q}", 28, 330, fg, center=False)
        c.right(W - 24, y1 + 52, f"{p:,}원", 28, accent)
        c.add("cart_item", f"{name} x{q}", (px + 10, y1, W - 10, y1 + 92), p)
    ms.put(d, (px + 20, 520), f"총 {total:,}원", ms.font(40), fg)
    c.add("price", f"총 {total:,}원", (px + 10, 505, W - 10, 580), total)
    b = random.choice(["결제하기", "주문하기", "카드결제"])
    d.rounded_rectangle((px + 20, 600, W - 20, 700), 16, fill=btn)
    ms.fit_text(d, (px + 26, 624), b, 44, W - px - 52, (255, 255, 255))
    c.add("button", b, (px + 20, 600, W - 20, 700))
    return c.done("cart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["list", "cart"], required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--vocab", choices=["cafe", "unseen"], default="cafe")
    ap.add_argument("--fonts", choices=["train", "holdout"], default="train")
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed)
    if a.vocab == "unseen":
        ms.TABS = ms.TABS_UNSEEN
    if a.fonts == "holdout":
        ms.FONTS = HOLDOUT_FONTS
    out = Path(a.out) / "images"
    out.mkdir(parents=True, exist_ok=True)
    draw = draw_list if a.layout == "list" else draw_cart
    for i in range(a.n):
        img, st, els = draw()
        level = i % 3
        cv2.imencode(".jpg", ms.degrade(img, level))[1].tofile(str(out / f"{i:05d}.jpg"))
        json.dump({"screen_type": st, "layout": a.layout, "keyframe_id": i, "degrade": level, "elements": els},
                  open(out / f"{i:05d}.json", "w", encoding="utf-8"), ensure_ascii=False)
    menu = sorted({p + m for k, v in ms.TABS.items() for m in v for p in (ms.PREFIX if k in ("커피", "음료") else [""])})
    json.dump(menu, open(Path(a.out) / "menu_dict.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{a.n} {a.layout} screens -> {a.out}")


if __name__ == "__main__":
    main()
