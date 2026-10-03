"""합성 키오스크 화면 생성기 — 실제 촬영 500장이 모이기 전 M3 스모크 테스트용.

펼친 화면(F-02 출력)을 흉내 낸다: 세로 720x1280, 탭·메뉴 카드·옵션·하단 버튼.
카메라 열화(잔여 원근, 해상도 손실, 흐림, 반사광, 노이즈, JPEG)를 섞는다.
라벨은 명세 6장 화면 구조 JSON 형식(elements: id/kind/text/price/box).

  python make_synth.py --n 300 --out data/synth_val --seed 1                       # 평가용 화면
  python make_synth.py --n 300 --out data/synth_unseen --seed 2 --vocab unseen     # 학습에 없는 메뉴로 일반화 확인
  python make_synth.py --n 3000 --out data/rec_train --seed 10 --rec --generic 20000  # 인식 모델 학습용 줄 이미지
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 1280
FONTS = ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/NotoSansKR-VF.ttf",
         ("C:/Windows/Fonts/gulim.ttc", 0), ("C:/Windows/Fonts/batang.ttc", 0)]
TABS = {"커피": ["아메리카노", "카페라떼", "카푸치노", "바닐라라떼", "카라멜마끼아또", "카페모카", "콜드브루", "에스프레소",
               "헤이즐넛라떼", "돌체라떼", "아인슈페너", "연유라떼"],
        "음료": ["딸기라떼", "초코라떼", "녹차라떼", "고구마라떼", "레몬에이드", "자몽에이드", "청포도에이드", "딸기스무디",
               "망고스무디", "플레인요거트스무디", "아이스초코", "밀크티"],
        "티": ["캐모마일", "페퍼민트", "얼그레이", "유자차", "레몬차", "생강차", "자몽허니블랙티", "복숭아아이스티"],
        "디저트": ["치즈케이크", "티라미수", "크루아상", "베이글", "마카롱", "허니브레드", "초코머핀", "소금빵"]}
# 학습에 쓰지 않는 메뉴 — 일반화 확인용 (--vocab unseen)
TABS_UNSEEN = {"식사": ["김치찌개", "된장찌개", "제육덮밥", "비빔밥", "돈까스", "순두부찌개", "불고기정식", "육개장",
                      "오므라이스", "카레라이스", "냉면", "칼국수"],
               "분식": ["떡볶이", "김밥", "참치김밥", "라볶이", "쫄면", "순대", "튀김모둠", "어묵탕", "만두", "치즈라면"],
               "음료": ["콜라", "사이다", "제로콜라", "환타", "식혜", "수정과", "생수", "매실주스"],
               "사이드": ["감자튀김", "치즈볼", "계란찜", "공기밥", "군만두", "샐러드", "주먹밥", "콘치즈"]}
PREFIX = ["", "", "", "아이스 ", "(ICE) ", "HOT ", "따뜻한 "]
OPTIONS = [("HOT", None), ("ICE", None), ("톨", None), ("그란데", "+500원"), ("벤티", "+1,000원"),
           ("샷 추가", "+500원"), ("시럽 추가", "+300원"), ("휘핑 추가", "+500원"), ("디카페인", "+1,000원"), ("얼음 적게", None)]
BUTTONS = ["장바구니", "결제하기", "처음으로", "이전", "다음", "담기", "취소", "주문하기", "전체삭제", "포장", "매장"]
THEMES = [((250, 250, 250), (30, 30, 30), (200, 40, 40), (235, 235, 235), (60, 120, 220)),
          ((30, 30, 35), (240, 240, 240), (255, 200, 60), (55, 55, 65), (230, 90, 40)),
          ((255, 248, 235), (60, 40, 20), (180, 60, 20), (240, 225, 200), (110, 70, 40)),
          ((235, 245, 255), (20, 40, 80), (220, 50, 50), (255, 255, 255), (0, 90, 170))]


LINES = []  # 이번 화면에 그린 글자 줄 (text, (x1,y1,x2,y2))


def put(d, xy, text, font, fill):
    d.text(xy, text, font=font, fill=fill)
    LINES.append((text, d.textbbox(xy, text, font=font)))


def font(size, spec=None):
    f = spec or random.choice(FONTS)
    return ImageFont.truetype(f[0], size, index=f[1]) if isinstance(f, tuple) else ImageFont.truetype(f, size)


def price_str(v):
    return random.choice([f"{v:,}원", f"{v:,}", f"₩{v:,}", f"{v:,} 원"])


def fit_text(d, xy, text, size, box_w, fill, center=True):
    """박스 폭에 맞을 때까지 글자 크기를 줄여 그린다(키오스크의 긴 메뉴명 처리 흉내)."""
    spec = random.choice(FONTS)
    while True:
        f = font(size, spec)
        tw = d.textlength(text, font=f)
        if tw <= box_w or size <= 14:
            break
        size -= 2
    put(d, (xy[0] + (box_w - tw) / 2 if center else xy[0], xy[1]), text, f, fill)


def nb(x1, y1, x2, y2):
    return [round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4)]


def draw_screen():
    LINES.clear()
    bg, fg, accent, card, btn = random.choice(THEMES)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    els, n = [], 0

    def add(kind, text, box, price=None):
        nonlocal n
        n += 1
        e = {"id": f"e{n}", "kind": kind, "text": text, "box": nb(*box)}
        if price is not None:
            e["price"] = price
        els.append(e)

    tabs = list(TABS)
    cur = random.choice(tabs)
    tw = W // len(tabs)
    for i, t in enumerate(tabs):
        x1, y1, x2, y2 = i * tw + 6, 90, (i + 1) * tw - 6, 160
        d.rounded_rectangle((x1, y1, x2, y2), 14, fill=btn if t == cur else card)
        fit_text(d, (x1, y1 + 14), t, 34, x2 - x1 - 10, (255, 255, 255) if t == cur else fg)
        add("tab", t, (x1, y1, x2, y2))
    put(d, (24, 24), random.choice(["어서오세요", "메뉴를 선택해 주세요", "MENU", "주문하실 메뉴를 골라주세요"]), font(36), fg)

    screen_type = random.choice(["menu", "menu", "option"])
    if screen_type == "menu":
        cols = random.choice([2, 3])
        rows = 3 if cols == 3 else random.choice([2, 3])
        cw, ch = (W - 24 * (cols + 1)) // cols, (900 - 24 * (rows + 1)) // rows
        names = random.sample(TABS[cur], min(cols * rows, len(TABS[cur])))
        for k, name in enumerate(names):
            r, c = divmod(k, cols)
            x1, y1 = 24 + c * (cw + 24), 190 + r * (ch + 24)
            x2, y2 = x1 + cw, y1 + ch
            d.rounded_rectangle((x1, y1, x2, y2), 18, fill=card)
            ph = int(ch * 0.55)
            d.rounded_rectangle((x1 + 14, y1 + 12, x2 - 14, y1 + ph), 12,
                                fill=tuple(random.randint(90, 220) for _ in range(3)))
            label = random.choice(PREFIX) + name if cur in ("커피", "음료") else name
            fit_text(d, (x1 + 8, y1 + ph + 12), label, 32 if cols == 3 else 40, cw - 16, fg)
            p = random.choice(range(2500, 7500, 500))
            fit_text(d, (x1 + 8, y1 + ph + 60 if cols == 3 else y1 + ph + 70), price_str(p), 28 if cols == 3 else 34,
                     cw - 16, accent)
            add("menu", label, (x1, y1, x2, y2), p)
    else:
        name = random.choice(TABS[cur])
        put(d, (40, 200), name, font(52), fg)
        base = random.choice(range(2500, 7500, 500))
        put(d, (40, 280), price_str(base), font(40), accent)
        add("menu", name, (30, 190, W - 30, 340), base)
        opts = random.sample(OPTIONS, 6)
        for k, (o, extra) in enumerate(opts):
            r, c = divmod(k, 2)
            x1, y1 = 40 + c * 330, 380 + r * 170
            x2, y2 = x1 + 310, y1 + 140
            d.rounded_rectangle((x1, y1, x2, y2), 16, fill=card, outline=btn, width=3)
            fit_text(d, (x1 + 8, y1 + 22), o, 40, 294, fg)
            if extra:
                fit_text(d, (x1 + 8, y1 + 80), extra, 30, 294, accent)
            add("button", o, (x1, y1, x2, y2), int(extra.strip("+원").replace(",", "")) if extra else None)

    bs = random.sample(BUTTONS, 3)
    bw = (W - 24 * 4) // 3
    for i, b in enumerate(bs):
        x1, y1 = 24 + i * (bw + 24), 1150
        d.rounded_rectangle((x1, y1, x1 + bw, y1 + 100), 16, fill=btn)
        fit_text(d, (x1 + 6, y1 + 26), b, 38, bw - 12, (255, 255, 255))
        add("button", b, (x1, y1, x1 + bw, y1 + 100))
    if random.random() < 0.5:
        total = random.choice(range(4500, 30000, 500))
        put(d, (40, 1095), f"합계 {total:,}원", font(36), fg)
        add("price", f"합계 {total:,}원", (30, 1085, 400, 1145), total)
    return np.array(img)[:, :, ::-1].copy(), screen_type, els, list(LINES)


def degrade(img, level):
    """level 0=깨끗 1=보통 2=나쁨"""
    if level == 0:
        return img
    h, w = img.shape[:2]
    j = 0.006 * level
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32(np.random.uniform(-j, j, (4, 2)) * [w, h])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    s = random.uniform(0.45, 0.75) if level == 1 else random.uniform(0.3, 0.5)  # 카메라 속 화면 해상도
    img = cv2.resize(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    k = random.choice([1, 3, 3, 5][: 2 + level])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = img.astype(np.float32)
    img = img * random.uniform(0.75, 1.15) + random.uniform(-25, 25)
    if random.random() < 0.3 * level:  # 반사광
        yy, xx = np.mgrid[:h, :w]
        cx, cy, r = random.uniform(0, w), random.uniform(0, h), random.uniform(120, 400)
        img += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * random.uniform(60, 140 * level))[..., None]
    img += np.random.normal(0, 3 + 3 * level, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = random.randint(55, 85) if level == 1 else random.randint(35, 60)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


HANGUL = [bytes([a, b]).decode("euc-kr") for a in range(0xB0, 0xC9) for b in range(0xA1, 0xFF)]  # KS X 1001 2350자


def rec_label(t):
    return t.replace("₩", "W")  # ₩ 는 korean 사전에 없음 → 모델은 W 로 읽도록


def crop_line(img, box, level):
    """검출기가 잘라 주는 것처럼 여백을 흔들어 줄 이미지를 자른다."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    px = random.uniform(-0.05, 0.5) * h
    py = random.uniform(-0.05, 0.3) * h
    H_, W_ = img.shape[:2]
    a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
    c, d = min(W_, int(x2 + px)), min(H_, int(y2 + py))
    return img[b:d, a:c] if c - a > 4 and d - b > 4 else None


def generic_line():
    """키오스크 어휘에 과적합되지 않도록 섞는 임의 한글 줄 (한 글자 포함)."""
    r = random.random()
    if r < 0.15:
        t = random.choice(HANGUL)
    elif r < 0.8:
        t = " ".join("".join(random.choices(HANGUL, k=random.randint(1, 5))) for _ in range(random.randint(1, 3)))
    elif r < 0.9:
        t = random.choice(["", "+", "W"]) + f"{random.randint(1, 300) * 100:,}" + random.choice(["", "원", " 원"])
    else:
        t = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=random.randint(1, 6)))
    bg, fg, accent, card, btn = random.choice(THEMES)
    bgc, fgc = random.choice([(bg, fg), (bg, accent), (card, fg), (btn, (255, 255, 255))])
    f = font(random.randint(26, 56))
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x1, y1, x2, y2 = tmp.textbbox((0, 0), t, font=f)
    pad = 40
    im = Image.new("RGB", (x2 - x1 + 2 * pad, y2 - y1 + 2 * pad), bgc)
    ImageDraw.Draw(im).text((pad - x1, pad - y1), t, font=f, fill=fgc)
    arr = degrade(np.array(im)[:, :, ::-1].copy(), random.choice([0, 1, 1, 2, 2]))
    return t, crop_line(arr, (pad, pad, pad + x2 - x1, pad + y2 - y1), 0)


def main():
    global TABS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/synth_val")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vocab", choices=["cafe", "unseen"], default="cafe")
    ap.add_argument("--rec", action="store_true", help="화면 대신 인식 학습용 줄 이미지 + 라벨 목록을 만든다")
    ap.add_argument("--generic", type=int, default=0, help="--rec 일 때 섞을 임의 한글 줄 수")
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed)
    if a.vocab == "unseen":
        TABS = TABS_UNSEEN
    out = Path(a.out)
    sub = out / ("crops" if a.rec else "images")
    sub.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(a.n):
        img, st, els, lines = draw_screen()
        level = random.choice([0, 1, 1, 2, 2]) if a.rec else i % 3
        img = degrade(img, level)
        if a.rec:
            for j, (t, box) in enumerate(lines):
                c = crop_line(img, box, level)
                if c is not None and t.strip():
                    name = f"{i:05d}_{j:02d}.jpg"
                    cv2.imencode(".jpg", c)[1].tofile(str(sub / name))
                    labels.append(f"{out.name}/crops/{name}\t{rec_label(t)}")
            continue
        cv2.imencode(".jpg", img)[1].tofile(str(sub / f"{i:05d}.jpg"))
        json.dump({"screen_type": st, "keyframe_id": i, "degrade": level, "elements": els},
                  open(sub / f"{i:05d}.json", "w", encoding="utf-8"), ensure_ascii=False)
    if a.rec:
        for k in range(a.generic):
            t, c = generic_line()
            if c is not None:
                name = f"g{k:06d}.jpg"
                cv2.imencode(".jpg", c)[1].tofile(str(sub / name))
                labels.append(f"{out.name}/crops/{name}\t{rec_label(t)}")
        random.shuffle(labels)
        open(out / "labels.txt", "w", encoding="utf-8").write("\n".join(labels) + "\n")
        print(f"{len(labels)} lines -> {out / 'labels.txt'}")
        return
    menu = sorted({p + m for k, v in TABS.items() for m in v for p in (PREFIX if k in ("커피", "음료") else [""])})
    json.dump(menu, open(out / "menu_dict.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{a.n} screens -> {out}")


if __name__ == "__main__":
    main()
