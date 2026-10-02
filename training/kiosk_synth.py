"""
손끝길 합성 키오스크 화면 생성기 (M2: 화면 요소 탐지용)

키오스크 화면을 무작위로 그리면서, 그린 요소의 위치를 그대로 YOLO 라벨로 저장한다.
버튼을 '알고 그리기' 때문에 사람이 라벨링할 필요가 없다.

클래스
  0 tab     카테고리 탭 (커피, 디저트 ...)
  1 menu    메뉴 카드 전체 (누르는 영역)
  2 price   가격 글자 영역
  3 button  동작 버튼 (장바구니, 결제하기, HOT/ICE, 담기 ...)
  4 back    뒤로가기/처음으로

화면 종류: menu(메뉴), option(옵션), cart(장바구니), payment(결제)
화면 종류는 meta.csv에 함께 저장해 두어, 이후 화면 분류기(F-05) 학습에 쓴다.

사용 예
  python kiosk_synth.py --out data/m2 --n-train 4000 --n-val 500
"""
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLASSES = ["tab", "menu", "price", "button", "back"]
CID = {c: i for i, c in enumerate(CLASSES)}

STORES = ["카페 손끝", "오늘커피", "메가빈", "블루하우스", "달빛다방", "커피온", "빈스토리", "햇살카페",
          "버거팩토리", "김밥천국", "분식왕", "치킨온", "누들하우스"]
CATS = ["커피", "논커피", "티", "디저트", "스무디", "에이드", "베이커리", "시즌메뉴", "프라페", "주스",
        "버거", "세트", "사이드", "음료", "추천메뉴", "신메뉴", "분식", "식사"]
MENUS = ["아메리카노", "카페라떼", "바닐라라떼", "카푸치노", "카라멜마끼아또", "콜드브루", "에스프레소",
         "녹차라떼", "초코라떼", "고구마라떼", "딸기라떼", "자몽에이드", "레몬에이드", "청포도에이드",
         "아이스티", "캐모마일", "얼그레이", "페퍼민트", "치즈케이크", "티라미수", "크로플", "마카롱",
         "베이글", "스콘", "망고스무디", "딸기스무디", "요거트스무디", "불고기버거", "치즈버거",
         "새우버거", "감자튀김", "치즈스틱", "콜라", "사이다", "참치김밥", "라볶이", "떡볶이", "돈까스"]
OPT_GROUPS = [["HOT", "ICE"], ["Regular", "Large"], ["톨", "그란데", "벤티"], ["샷 추가", "시럽 추가"],
              ["얼음 적게", "얼음 보통", "얼음 많이"], ["단품", "세트"], ["매장", "포장"]]
ACTION_BTNS = ["장바구니", "결제하기", "주문하기", "선택 완료", "담기", "전체 삭제", "카드 결제", "간편 결제"]
BACK_TXT = ["← 이전", "처음으로", "< 뒤로", "홈", "취소"]

PALETTES = [  # (배경, 패널, 글자, 강조)
    ((250, 250, 250), (255, 255, 255), (30, 30, 30), (230, 80, 40)),
    ((245, 240, 232), (255, 252, 246), (60, 40, 30), (120, 70, 40)),
    ((30, 30, 34), (48, 48, 54), (240, 240, 240), (255, 196, 0)),
    ((235, 244, 255), (255, 255, 255), (20, 40, 80), (0, 110, 220)),
    ((20, 60, 50), (34, 84, 70), (240, 250, 240), (250, 200, 80)),
    ((255, 246, 246), (255, 255, 255), (60, 20, 30), (220, 30, 80)),
]

FONT_CANDIDATES = [
    # (경로, ttc 인덱스) — Linux / Windows / macOS 순서로 찾는다
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 1),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc", 1),
    ("C:/Windows/Fonts/malgun.ttf", 0),
    ("C:/Windows/Fonts/malgunbd.ttf", 0),
    ("C:/Windows/Fonts/gulim.ttc", 0),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
]


def find_fonts(extra_dir=None):
    found = [(p, i) for p, i in FONT_CANDIDATES if os.path.exists(p)]
    if extra_dir:
        for p in Path(extra_dir).glob("*.[ot]t[fc]"):
            found.append((str(p), 0))
    if not found:
        raise SystemExit("한글 폰트를 찾지 못했습니다. --font-dir 로 .ttf/.otf 폴더를 지정하세요.")
    return found


class Painter:
    """그리면서 라벨을 모으는 도우미"""

    def __init__(self, w, h, fonts, pal):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), pal[0])
        self.d = ImageDraw.Draw(self.img)
        self.fonts = fonts
        self.font_path = random.choice(fonts)
        self.pal = pal
        self.labels = []  # (cls, x1, y1, x2, y2)

    def font(self, size):
        p, i = self.font_path
        return ImageFont.truetype(p, max(8, int(size)), index=i)

    def text(self, xy, s, size, fill, anchor="la"):
        f = self.font(size)
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return self.d.textbbox(xy, s, font=f, anchor=anchor)

    def label(self, cls, box, pad=0):
        x1, y1, x2, y2 = box
        self.labels.append((CID[cls], x1 - pad, y1 - pad, x2 + pad, y2 + pad))

    def button(self, box, s, cls="button", filled=True, size=None):
        x1, y1, x2, y2 = box
        r = random.choice([0, 6, 12, (y2 - y1) // 2])
        fill = self.pal[3] if filled else self.pal[1]
        outline = self.pal[3]
        self.d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=2)
        tcol = (255, 255, 255) if filled and sum(self.pal[3]) < 600 else self.pal[2]
        size = size or (y2 - y1) * random.uniform(0.32, 0.45)
        self.text(((x1 + x2) / 2, (y1 + y2) / 2), s, size, tcol, anchor="mm")
        self.label(cls, box)


def fmt_price(v):
    return f"{v:,}원" if random.random() < 0.8 else f"₩{v:,}"


def draw_header(p, h_ratio=0.08, back=True):
    hh = int(p.h * h_ratio)
    p.d.rectangle((0, 0, p.w, hh), fill=p.pal[3] if random.random() < 0.5 else p.pal[1])
    p.text((p.w / 2, hh / 2), random.choice(STORES), hh * 0.4, p.pal[2] if random.random() < 0.5 else (255, 255, 255), "mm")
    if back and random.random() < 0.7:
        bw, bh = int(p.w * random.uniform(0.14, 0.2)), int(hh * 0.6)
        x1, y1 = int(p.w * 0.02), (hh - bh) // 2
        p.button((x1, y1, x1 + bw, y1 + bh), random.choice(BACK_TXT), cls="back", filled=False)
    return hh


def draw_tabs(p, y0):
    n = random.randint(3, 6)
    th = int(p.h * random.uniform(0.045, 0.065))
    gap = int(p.w * 0.01)
    tw = (p.w - gap * (n + 1)) // n
    cats = random.sample(CATS, n)
    sel = random.randrange(n)
    for i, c in enumerate(cats):
        x1 = gap + i * (tw + gap)
        box = (x1, y0, x1 + tw, y0 + th)
        if i == sel:
            p.d.rounded_rectangle(box, radius=8, fill=p.pal[3])
            p.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), c, th * 0.42, (255, 255, 255), "mm")
        else:
            p.d.rounded_rectangle(box, radius=8, fill=p.pal[1], outline=p.pal[2] if random.random() < 0.3 else None)
            p.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), c, th * 0.42, p.pal[2], "mm")
        p.label("tab", box)
    return y0 + th + gap


def draw_menu_grid(p, y0, y1):
    cols = random.randint(2, 4) if p.w < p.h else random.randint(3, 5)
    rows = random.randint(2, 4)
    gap = int(p.w * random.uniform(0.015, 0.03))
    cw = (p.w - gap * (cols + 1)) // cols
    ch = (y1 - y0 - gap * (rows + 1)) // rows
    names = random.sample(MENUS, cols * rows)
    for r in range(rows):
        for c in range(cols):
            if random.random() < 0.06:
                continue  # 빈 칸
            x = gap + c * (cw + gap)
            y = y0 + gap + r * (ch + gap)
            box = (x, y, x + cw, y + ch)
            p.d.rounded_rectangle(box, radius=random.choice([0, 8, 16]), fill=p.pal[1],
                                  outline=(200, 200, 200) if random.random() < 0.5 else None)
            # 상품 이미지 자리: 무작위 도형
            ih = int(ch * random.uniform(0.45, 0.6))
            cx, cy = x + cw / 2, y + ih / 2 + ch * 0.05
            rad = min(cw, ih) * random.uniform(0.3, 0.45)
            col = tuple(random.randint(60, 230) for _ in range(3))
            if random.random() < 0.5:
                p.d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=col)
            else:
                p.d.rounded_rectangle((cx - rad * 0.8, cy - rad, cx + rad * 0.8, cy + rad), radius=10, fill=col)
            name_size = min(ch * 0.1, cw / max(4, len(names[r * cols + c])) * 1.1)
            p.text((x + cw / 2, y + ih + ch * 0.08), names[r * cols + c], name_size, p.pal[2], "mt")
            pb = p.text((x + cw / 2, y + ih + ch * 0.08 + name_size * 1.5), fmt_price(random.randrange(15, 120) * 100),
                        name_size * 0.9, p.pal[3], "mt")
            p.label("price", pb, pad=2)
            p.label("menu", box)


def draw_bottom_bar(p, y0):
    p.d.rectangle((0, y0, p.w, p.h), fill=p.pal[1])
    p.text((p.w * 0.04, y0 + (p.h - y0) * 0.3), f"총 {random.randint(0, 5)}개  {fmt_price(random.randrange(0, 300) * 100)}",
           (p.h - y0) * 0.18, p.pal[2])
    n = random.randint(1, 3)
    bw = int(p.w * random.uniform(0.2, 0.3))
    bh = int((p.h - y0) * random.uniform(0.45, 0.65))
    by = p.h - bh - int((p.h - y0) * 0.12)
    for i, s in enumerate(random.sample(ACTION_BTNS, n)):
        x2 = p.w - int(p.w * 0.03) - i * (bw + int(p.w * 0.02))
        p.button((x2 - bw, by, x2, by + bh), s, filled=(i == 0))


def screen_menu(p):
    y = draw_header(p)
    y = draw_tabs(p, y + int(p.h * 0.01))
    bar = int(p.h * random.uniform(0.82, 0.88))
    draw_menu_grid(p, y, bar)
    draw_bottom_bar(p, bar)


def screen_option(p):
    if random.random() < 0.5:  # 메뉴 화면 위의 모달
        screen_menu(p)
        p.labels.clear()
        overlay = Image.new("RGBA", p.img.size, (0, 0, 0, random.randint(90, 160)))
        p.img = Image.alpha_composite(p.img.convert("RGBA"), overlay).convert("RGB")
        p.d = ImageDraw.Draw(p.img)
        mx, my = int(p.w * random.uniform(0.06, 0.15)), int(p.h * random.uniform(0.12, 0.22))
        area = (mx, my, p.w - mx, p.h - my)
    else:
        draw_header(p)
        area = (0, int(p.h * 0.09), p.w, p.h)
    ax1, ay1, ax2, ay2 = area
    p.d.rounded_rectangle(area, radius=16, fill=p.pal[1])
    aw, ah = ax2 - ax1, ay2 - ay1
    p.text((ax1 + aw / 2, ay1 + ah * 0.06), random.choice(MENUS), ah * 0.045, p.pal[2], "mt")
    y = ay1 + ah * 0.16
    for grp in random.sample(OPT_GROUPS, random.randint(2, 3)):
        p.text((ax1 + aw * 0.06, y), random.choice(["온도", "사이즈", "옵션", "선택", "얼음", "수령 방법"]), ah * 0.03, p.pal[2])
        y += ah * 0.05
        n = len(grp)
        gap = aw * 0.03
        bw = (aw * 0.88 - gap * (n - 1)) / n
        bh = ah * random.uniform(0.07, 0.1)
        for i, s in enumerate(grp):
            x1 = ax1 + aw * 0.06 + i * (bw + gap)
            p.button((int(x1), int(y), int(x1 + bw), int(y + bh)), s, filled=(i == 0 and random.random() < 0.5))
        y += bh + ah * 0.05
    bh = ah * 0.09
    by = ay2 - bh - ah * 0.04
    p.button((int(ax1 + aw * 0.06), int(by), int(ax1 + aw * 0.47), int(by + bh)), "취소", cls="back", filled=False)
    p.button((int(ax1 + aw * 0.53), int(by), int(ax1 + aw * 0.94), int(by + bh)), random.choice(["담기", "선택 완료", "장바구니 담기"]))


def screen_cart(p):
    y = draw_header(p)
    p.text((p.w * 0.05, y + p.h * 0.02), "장바구니", p.h * 0.035, p.pal[2])
    y += int(p.h * 0.08)
    rh = int(p.h * random.uniform(0.07, 0.09))
    for _ in range(random.randint(1, 5)):
        p.d.rounded_rectangle((p.w * 0.04, y, p.w * 0.96, y + rh), radius=8, fill=p.pal[1])
        p.text((p.w * 0.07, y + rh / 2), random.choice(MENUS), rh * 0.3, p.pal[2], "lm")
        pb = p.text((p.w * 0.55, y + rh / 2), fmt_price(random.randrange(15, 120) * 100), rh * 0.28, p.pal[3], "lm")
        p.label("price", pb, pad=2)
        bs = int(rh * 0.55)
        bx = int(p.w * 0.74)
        p.button((bx, y + (rh - bs) // 2, bx + bs, y + (rh + bs) // 2), "−", filled=False, size=bs * 0.6)
        p.text((bx + bs * 1.6, y + rh / 2), str(random.randint(1, 4)), rh * 0.3, p.pal[2], "mm")
        bx2 = int(bx + bs * 2.2)
        p.button((bx2, y + (rh - bs) // 2, bx2 + bs, y + (rh + bs) // 2), "+", filled=False, size=bs * 0.6)
        y += rh + int(p.h * 0.012)
    draw_bottom_bar(p, int(p.h * 0.86))


def screen_payment(p):
    draw_header(p, back=False)
    p.text((p.w / 2, p.h * 0.3), random.choice(["카드를 넣어주세요", "결제 수단을 선택하세요", "결제 금액"]), p.h * 0.04, p.pal[2], "mm")
    pb = p.text((p.w / 2, p.h * 0.4), fmt_price(random.randrange(20, 300) * 100), p.h * 0.05, p.pal[3], "mm")
    p.label("price", pb, pad=3)
    bw, bh = int(p.w * 0.38), int(p.h * 0.08)
    p.button((int(p.w * 0.08), int(p.h * 0.6), int(p.w * 0.08) + bw, int(p.h * 0.6) + bh), "카드 결제")
    p.button((int(p.w * 0.54), int(p.h * 0.6), int(p.w * 0.54) + bw, int(p.h * 0.6) + bh), "간편 결제", filled=False)
    p.button((int(p.w * 0.3), int(p.h * 0.82), int(p.w * 0.7), int(p.h * 0.82) + bh), "취소", cls="back", filled=False)


SCREENS = [("menu", screen_menu, 0.6), ("option", screen_option, 0.2), ("cart", screen_cart, 0.12), ("payment", screen_payment, 0.08)]


def render_screen(fonts, w=None, h=None):
    if w is None:
        if random.random() < 0.7:
            w, h = 720, 1280
        else:
            w, h = 1280, 720
    p = Painter(w, h, fonts, random.choice(PALETTES))
    r, acc = random.random(), 0
    for name, fn, prob in SCREENS:
        acc += prob
        if r <= acc:
            fn(p)
            return p.img, p.labels, name
    screen_menu(p)
    return p.img, p.labels, "menu"


def photometric(img):
    """카메라로 찍힌 듯한 열화: 흐림, 밝기, 노이즈, 반사광, 압축"""
    a = np.asarray(img).astype(np.float32)
    a = a * random.uniform(0.6, 1.25) + random.uniform(-30, 30)
    if random.random() < 0.5:  # 반사광
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        rad = random.uniform(0.15, 0.5) * max(w, h)
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2)))
        a += (g * random.uniform(40, 140))[..., None]
    a += np.random.normal(0, random.uniform(0, 10), a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0)))
    return img


def yolo_lines(labels, w, h):
    out = []
    for c, x1, y1, x2, y2 in labels:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        out.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def build(out, n_train, n_val, fonts, long_side=960, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    out = Path(out)
    rows = []
    for split, n in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels, kind = render_screen(fonts)
            img = photometric(img)
            s = long_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
            labels = [(c, x1 * s, y1 * s, x2 * s, y2 * s) for c, x1, y1, x2, y2 in labels]
            stem = f"{split}_{i:05d}"
            img.save(out / "images" / split / f"{stem}.jpg", quality=random.randint(60, 95))
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo_lines(labels, img.width, img.height)))
            rows.append((split, stem, kind))
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("split", "stem", "screen_type")] + rows)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    print(f"done: {out} train={n_train} val={n_val}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/m2")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--font-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.n_train, a.n_val, find_fonts(a.font_dir), seed=a.seed)
