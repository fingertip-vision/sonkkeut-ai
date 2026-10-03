"""가짜 키오스크 시뮬레이터 — 누르면 화면이 넘어가는 목업 키오스크 3종 (F-05·F-07·F-10 자동 테스트용).

화면: start(기타) → menu → option → (다시 menu) → cart(주문 확인) → payment(결제)
브랜드(흐름) 3종 = 명세의 '목업 키오스크 3개 흐름' 대역
  A : 세로 720x1280, 위 탭 + 3x2 카드 + 페이지 넘김, 온도는 옵션 화면에서 고름
  B : 가로 1280x720, 왼쪽 세로 탭 + 목록형, '아이스 ○○'가 별도 메뉴(온도 옵션 없음)
  C : 가로 1280x720, 위 탭 + 4x2 카드 + 오른쪽 장바구니 패널, 수량을 옵션 화면에서 고름
render() 는 (BGR 이미지, 정답 요소 목록)을 돌려주고, press(요소 id) 로 상태가 바뀐다.
정답 요소는 F-03 출력 형식(id·kind·box)에 F-04 정답(text·price·qty)을 더한 것.
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ocr"))
import make_synth as ms  # noqa: E402  (폰트·테마·글자 그리기 재사용)

CATALOG = {
    "커피": [("아메리카노", 4500), ("카페라떼", 5000), ("카푸치노", 5000), ("바닐라라떼", 5500), ("카라멜마끼아또", 5800),
           ("카페모카", 5500), ("콜드브루", 5000), ("에스프레소", 4000), ("헤이즐넛라떼", 5500), ("돌체라떼", 5800)],
    "음료": [("딸기라떼", 5800), ("초코라떼", 5500), ("녹차라떼", 5500), ("레몬에이드", 5500), ("자몽에이드", 5500),
           ("청포도에이드", 5500), ("딸기스무디", 6000), ("망고스무디", 6000)],
    "티": [("캐모마일", 4500), ("페퍼민트", 4500), ("얼그레이", 4500), ("유자차", 5000), ("레몬차", 5000), ("복숭아아이스티", 4800)],
    "디저트": [("치즈케이크", 6000), ("티라미수", 6500), ("크루아상", 3800), ("베이글", 3500), ("마카롱", 2500), ("소금빵", 3500)],
}
HAS_TEMP = {"커피", "음료", "티"}
SIZES = [("톨", 0), ("그란데", 500), ("벤티", 1000)]
EXTRAS = [("샷 추가", 500), ("시럽 추가", 300)]


def category_of(name):
    for c, items in CATALOG.items():
        for n, p in items:
            if n == name:
                return c, p
    raise KeyError(name)


class Kiosk:
    def __init__(self, brand="A", seed=0):
        self.brand = brand
        self.rng = random.Random(seed)
        self.W, self.H = (720, 1280) if brand == "A" else (1280, 720)
        self.per_page = {"A": 6, "B": 6, "C": 8}[brand]
        self.theme = self.rng.choice(ms.THEMES)
        self.font_spec = self.rng.choice(ms.FONTS)
        self.screen, self.tab, self.page = "start", "커피", 0
        self.cart = []  # [{"name","temp","size","extras","qty","unit"}]
        self.cur = None  # 옵션 화면에서 고르는 중인 항목
        self.actions = {}

    # ---------- 메뉴 목록 (B 는 아이스 메뉴를 따로 둔다) ----------
    def items(self, tab):
        out = []
        for n, p in CATALOG[tab]:
            out.append((n, p))
            if self.brand == "B" and tab in ("커피", "음료") and n not in ("에스프레소",):
                out.append(("아이스 " + n, p + 500))
        return out

    # ---------- 상태 전이 ----------
    def press(self, eid):
        act = self.actions.get(eid)
        if act is None:
            return False
        kind = act[0]
        if kind == "start":
            self.screen = "menu"
        elif kind == "tab":
            self.tab, self.page = act[1], 0
        elif kind == "page":
            self.page += act[1]
        elif kind == "menu":
            name, price = act[1], act[2]
            base = name.replace("아이스 ", "")
            cat = category_of(base)[0]
            temp = "ice" if name.startswith("아이스 ") else None
            self.cur = {"name": name, "temp": temp, "size": None, "extras": [], "qty": 1, "unit": price, "cat": cat}
            self.screen = "option"
        elif kind == "opt":
            g, v, extra = act[1], act[2], act[3]
            if g == "extras":
                if v not in self.cur["extras"]:
                    self.cur["extras"].append(v)
                    self.cur["unit"] += extra
            else:
                prev = self.cur.get(g)
                if g == "size" and prev:
                    self.cur["unit"] -= dict(SIZES)[prev]
                self.cur[g] = v
                if g == "size":
                    self.cur["unit"] += extra
        elif kind == "qty":
            self.cur["qty"] = max(1, self.cur["qty"] + act[1])
        elif kind == "add":
            c = dict(self.cur)
            if c["cat"] in HAS_TEMP and c["temp"] is None:
                c["temp"] = "hot"  # 기본값
            if c["cat"] in ("커피", "음료") and c["size"] is None:
                c["size"] = "톨"
            self.cart.append(c)
            self.cur, self.screen = None, "menu"
        elif kind == "cancel":
            self.cur, self.screen = None, "menu"
        elif kind == "cart_qty":
            it = self.cart[act[1]]
            it["qty"] += act[2]
            if it["qty"] <= 0:
                self.cart.pop(act[1])
        elif kind == "goto":
            self.screen = act[1]
        return True

    def total(self):
        return sum(c["unit"] * c["qty"] for c in self.cart)

    # ---------- 그리기 ----------
    def render(self):
        ms.LINES.clear()
        self.actions = {}
        self.els = []
        bg, fg, accent, card, btn = self.theme
        self.img = Image.new("RGB", (self.W, self.H), bg)
        self.d = ImageDraw.Draw(self.img)
        getattr(self, f"_r_{self.screen}")()
        return np.array(self.img)[:, :, ::-1].copy(), self.els

    def _font(self, size):
        return ms.font(size, self.font_spec)

    def _text(self, xy, text, size, fill, w=None, center=False):
        f = self._font(size)
        if w:
            while self.d.textlength(text, font=f) > w and size > 14:
                size -= 2
                f = self._font(size)
        tw = self.d.textlength(text, font=f)
        x = xy[0] + (w - tw) / 2 if (w and center) else xy[0]
        ms.put(self.d, (x, xy[1]), text, f, fill)

    def _el(self, kind, text, box, action=None, **kw):
        x1, y1, x2, y2 = box
        e = {"id": f"e{len(self.els) + 1}", "kind": kind, "text": text,
             "box": [round(x1 / self.W, 4), round(y1 / self.H, 4), round(x2 / self.W, 4), round(y2 / self.H, 4)]}
        e.update({k: v for k, v in kw.items() if v is not None})
        self.els.append(e)
        if action:
            self.actions[e["id"]] = action
        return e

    def _button(self, box, text, action, kind="button", size=36, fill=None, tcol=(255, 255, 255), **kw):
        bg, fg, accent, card, btn = self.theme
        self.d.rounded_rectangle(box, 14, fill=fill or btn)
        x1, y1, x2, y2 = box
        self._text((x1 + 6, y1 + (y2 - y1 - size) / 2 - 4), text, size, tcol, w=x2 - x1 - 12, center=True)
        return self._el(kind, text, box, action, **kw)

    def _label(self, xy, text, size, kind="text", fill=None, w=None, **kw):
        bg, fg, accent, card, btn = self.theme
        self._text(xy, text, size, fill or fg, w=w)
        f = self._font(size)
        tw = min(self.d.textlength(text, font=f), w or 10 ** 6)
        return self._el(kind, text, (xy[0] - 6, xy[1] - 4, xy[0] + tw + 6, xy[1] + size * 1.35), **kw)

    def _r_start(self):
        W, H = self.W, self.H
        self._label((W * 0.12, H * 0.25), "화면을 터치해 주세요", 48)
        self._label((W * 0.12, H * 0.25 + 80), "주문하실 곳을 선택하세요", 34)
        bw = int(W * 0.34)
        self._button((int(W * 0.12), int(H * 0.55), int(W * 0.12) + bw, int(H * 0.55) + 160), "매장", ("start",), size=48)
        self._button((int(W * 0.54), int(H * 0.55), int(W * 0.54) + bw, int(H * 0.55) + 160), "포장", ("start",), size=48)

    def _tabs(self, boxes):
        bg, fg, accent, card, btn = self.theme
        for t, box in zip(CATALOG, boxes):
            on = t == self.tab
            self._button(box, t, ("tab", t), kind="tab", size=34, fill=btn if on else card,
                         tcol=(255, 255, 255) if on else fg)

    def _page_slice(self):
        its = self.items(self.tab)
        n_pages = (len(its) + self.per_page - 1) // self.per_page
        self.page = min(self.page, n_pages - 1)
        return its[self.page * self.per_page:(self.page + 1) * self.per_page], n_pages

    def _card(self, box, name, price, name_size=32):
        bg, fg, accent, card, btn = self.theme
        x1, y1, x2, y2 = box
        self.d.rounded_rectangle(box, 16, fill=card)
        ph = int((y2 - y1) * 0.5)
        self.d.rounded_rectangle((x1 + 14, y1 + 12, x2 - 14, y1 + ph), 12, fill=(150, 150, 170))
        self._text((x1 + 8, y1 + ph + 14), name, name_size, fg, w=x2 - x1 - 16, center=True)
        self._text((x1 + 8, y1 + ph + 14 + name_size + 14), f"{price:,}원", name_size - 4, accent, w=x2 - x1 - 16,
                   center=True)
        self._el("menu", name, box, ("menu", name, price), price=price)

    def _paging(self, n_pages, prev_box, next_box):
        if self.page > 0:
            self._button(prev_box, "이전 페이지", ("page", -1), size=30)
        if self.page < n_pages - 1:
            self._button(next_box, "다음 페이지", ("page", 1), size=30)

    def _r_menu(self):
        W, H = self.W, self.H
        bg, fg, accent, card, btn = self.theme
        its, n_pages = self._page_slice()
        if self.brand == "A":
            tw = W // 4
            self._tabs([(i * tw + 6, 90, (i + 1) * tw - 6, 160) for i in range(4)])
            cw, ch = (W - 24 * 4) // 3, 300
            for k, (n, p) in enumerate(its):
                r, c = divmod(k, 3)
                x1, y1 = 24 + c * (cw + 24), 190 + r * (ch + 24)
                self._card((x1, y1, x1 + cw, y1 + ch), n, p, 30)
            self._paging(n_pages, (24, 840, 340, 920), (380, 840, W - 24, 920))
            cnt = sum(c["qty"] for c in self.cart)
            self._label((40, 960), f"담은 수량 {cnt}개  합계 {self.total():,}원", 34, kind="price", price=self.total())
            self._button((24, 1150, 340, 1250), "처음으로", ("goto", "start"), size=36)
            self._button((380, 1150, W - 24, 1250), "장바구니", ("goto", "cart"), size=36)
        elif self.brand == "B":
            self._tabs([(16, 90 + i * 100, 236, 174 + i * 100) for i in range(4)])
            rh = 76
            for k, (n, p) in enumerate(its):
                y1 = 90 + k * (rh + 8)
                self.d.rectangle((260, y1, W - 20, y1 + rh), fill=card)
                self._text((284, y1 + 18), n, 34, fg, w=600)
                f = self._font(34)
                ps = f"{p:,}원"
                self._text((W - 44 - self.d.textlength(ps, font=f), y1 + 18), ps, 34, accent)
                self._el("menu", n, (260, y1, W - 20, y1 + rh), ("menu", n, p), price=p)
            self._paging(n_pages, (260, 600, 520, 690), (540, 600, 800, 690))
            self._button((830, 600, 1040, 690), "처음으로", ("goto", "start"), size=34)
            self._button((1060, 600, W - 20, 690), "주문하기", ("goto", "cart"), size=34)
        else:  # C
            tw = 880 // 4
            self._tabs([(16 + i * tw, 16, 16 + (i + 1) * tw - 12, 80) for i in range(4)])
            cw, ch = 205, 250
            for k, (n, p) in enumerate(its):
                r, c = divmod(k, 4)
                x1, y1 = 16 + c * (cw + 15), 100 + r * (ch + 12)
                self._card((x1, y1, x1 + cw, y1 + ch), n, p, 28)
            self._paging(n_pages, (16, 630, 300, 700), (320, 630, 600, 700))
            px = 905
            self.d.rectangle((px, 0, W, H), fill=card)
            self._label((px + 20, 20), "주문 내역", 32)
            for k, it in enumerate(self.cart[:4]):
                y1 = 80 + k * 100
                self._label((px + 20, y1 + 6), f"{it['name']} x{it['qty']}", 26, kind="cart_item", w=330,
                            price=it["unit"] * it["qty"], qty=it["qty"])
            self._label((px + 20, 500), f"총 {self.total():,}원", 36, kind="price", price=self.total())
            self._button((px + 20, 600, W - 20, 700), "결제하기", ("goto", "cart"), size=40)

    def _r_option(self):
        W, H = self.W, self.H
        c = self.cur
        bg, fg, accent, card, btn = self.theme
        port = self.brand == "A"
        x0 = 40
        self._label((x0, 40), c["name"], 48, kind="title")
        self._label((x0, 110), f"{c['unit']:,}원", 36, kind="price", price=c["unit"], fill=accent)
        groups = []
        if c["cat"] in HAS_TEMP and not c["name"].startswith("아이스 "):
            groups.append(("temp", "온도", [("HOT", "hot", 0), ("ICE", "ice", 0)]))
        if c["cat"] in ("커피", "음료"):
            groups.append(("size", "사이즈", [(s, s, e) for s, e in SIZES]))
        if c["cat"] == "커피":
            groups.append(("extras", "추가", [(n, n, e) for n, e in EXTRAS]))
        y = 190 if port else 170
        step = 170 if port else 175
        bw = 200 if port else 230
        for g, title, opts in groups:
            self._label((x0, y), title, 30, kind="text")
            for k, (lab, val, extra) in enumerate(opts):
                x1 = x0 + k * (bw + 16)
                chosen = (c.get(g) == val) or (g == "extras" and val in c["extras"])
                text = lab if not extra else f"{lab} +{extra:,}원"
                self._button((x1, y + 50, x1 + bw, y + 140), text, ("opt", g, val, extra), size=28,
                             fill=btn if chosen else card, tcol=(255, 255, 255) if chosen else fg,
                             selected=chosen, price=extra or None)
            y += step
        if self.brand == "C":  # 가로 화면: 수량은 오른쪽 열
            qx, qy = 920, 200
            self._label((qx, qy), "수량", 30)
            self._button((qx, qy + 50, qx + 90, qy + 140), "-", ("qty", -1), size=40)
            self._label((qx + 120, qy + 70), str(c["qty"]), 40, kind="text", qty=c["qty"])
            self._button((qx + 190, qy + 50, qx + 280, qy + 140), "+", ("qty", 1), size=40)
        if port:
            by = H - 130
            self._button((40, by, W // 2 - 20, by + 100), "취소", ("cancel",), size=38)
            self._button((W // 2 + 20, by, W - 40, by + 100), "담기", ("add",), size=38)
        else:
            self._button((920, 440, W - 40, 540), "취소", ("cancel",), size=38)
            self._button((920, 580, W - 40, 680), "담기", ("add",), size=38)

    def _r_cart(self):
        W, H = self.W, self.H
        bg, fg, accent, card, btn = self.theme
        self._label((40, 30), "주문 확인", 44, kind="title")
        rh = 110 if self.brand == "A" else 90
        for k, it in enumerate(self.cart[:6]):
            y1 = 110 + k * (rh + 10)
            self.d.rectangle((30, y1, W - 30, y1 + rh), fill=card)
            opt = " ".join(x for x in [{"hot": "HOT", "ice": "ICE"}.get(it["temp"] or ""), it["size"] or ""] + it["extras"] if x)
            self._label((50, y1 + 10), it["name"], 32, kind="cart_item", w=W * 0.45, qty=it["qty"],
                        price=it["unit"] * it["qty"], options=opt or None)
            if opt:
                self._label((50, y1 + 52), opt, 24, kind="text")
            qx = int(W * 0.55)
            self._button((qx, y1 + 15, qx + 70, y1 + rh - 15), "-", ("cart_qty", k, -1), size=36, item=k)
            self._label((qx + 95, y1 + 25), str(it["qty"]), 36, kind="text", qty=it["qty"], item=k)
            self._button((qx + 150, y1 + 15, qx + 220, y1 + rh - 15), "+", ("cart_qty", k, 1), size=36, item=k)
        self._label((40, H - 230), f"총 결제 금액 {self.total():,}원", 40, kind="price", price=self.total())
        self._button((30, H - 130, W // 2 - 15, H - 30), "메뉴 추가", ("goto", "menu"), size=38)
        self._button((W // 2 + 15, H - 130, W - 30, H - 30), "결제하기", ("goto", "payment"), size=38)

    def _r_payment(self):
        W, H = self.W, self.H
        self._label((60, H * 0.2), f"결제 금액 {self.total():,}원", 48, kind="price", price=self.total())
        self._label((60, H * 0.2 + 90), "카드를 넣어 주세요", 44)
        self._label((60, H * 0.2 + 160), "IC카드를 투입구에 꽂아 주세요", 30)
        self._button((60, H - 160, W // 2 - 20, H - 50), "이전", ("goto", "cart"), size=38)
        self._button((W // 2 + 20, H - 160, W - 60, H - 50), "결제 취소", ("goto", "menu"), size=38)
