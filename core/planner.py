"""F-07 버튼 순서 계획 — 주문 의도 + 화면 구조(F-05) + 진행 상태 → 다음에 누를 버튼 하나와 예고 문장.

화면 종류별 규칙 (명세 F-07)
  other(시작) : 매장/포장 선택
  menu        : 남은 항목의 메뉴 → 없으면 다음 페이지 → 안 가 본 탭 → 다 담았으면 장바구니/결제
  option      : 온도 → 사이즈 → 추가 → (수량 +) → 담기
  cart        : 항목별 수량 맞추기(+/−) → 빠진 항목 있으면 메뉴 추가 → 결제하기
  payment     : 끝 (S6)
의도 JSON(명세 6장): {"items":[{"menu","qty","options":{"temp","size","extras"},"status"}], "dine"?}
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

try:
    from .screen_struct import n
except ImportError:  # preserve direct script execution used by the original experiments
    from screen_struct import n

TEMP_WORDS = {"hot": ["hot", "따뜻하게", "따뜻한", "핫"], "ice": ["ice", "차갑게", "아이스", "시원하게"]}
ICE_PREFIX = ["아이스", "(ice)", "ice", "아이스)"]
CHECKOUT = ["결제하기", "장바구니", "주문하기", "주문확인", "결제"]
ADD = ["담기", "장바구니담기", "선택완료", "추가하기"]
TEMP_KO = {"hot": "따뜻한", "ice": "차가운"}


@dataclass
class Progress:
    current: int | None = None           # 옵션 화면에서 고르는 중인 의도 항목 번호
    variant_temp: bool = False           # '아이스 ○○'처럼 메뉴 자체가 온도를 정했나
    opt_done: set = field(default_factory=set)
    qty_pressed: int = 0
    tabs_tried: list = field(default_factory=list)
    pages_moved: int = 0
    last: dict | None = None             # 마지막으로 누른 요소 (F-10 확인용)
    cur_tab: str | None = None           # 마지막으로 누른 탭 (첫 화면의 탭은 모름)
    seen: dict = field(default_factory=dict)  # 본 적 있는 메뉴 → 그때의 탭
    relook: int = 0                      # 같은 화면을 다시 읽어 본 횟수 (누르면 0)


# 메뉴명으로 탭 추측 — 위에서부터 먼저 맞는 규칙 (딸기라떼는 '라떼'보다 '딸기'가 먼저)
TAB_GUESS = [
    (["에이드", "스무디", "주스", "쥬스", "딸기", "초코", "녹차", "말차", "고구마", "밀크티", "요거트", "프라페"], ["음료", "논커피", "noncoffee", "beverage", "drink"]),
    (["캐모마일", "페퍼민트", "얼그레이", "루이보스", "아이스티", "블랙티", "유자차", "레몬차", "생강차", "자몽차", "차"], ["티", "차", "tea"]),
    (["케이크", "티라미수", "크루아상", "베이글", "마카롱", "빵", "머핀", "쿠키", "와플", "스콘", "디저트"], ["디저트", "베이커리", "dessert", "bakery"]),
    (["아메리카노", "라떼", "카푸치노", "에스프레소", "모카", "콜드브루", "마끼아또", "아인슈페너", "커피"], ["커피", "coffee"]),
]


def guess_tabs(menu, tabs):
    m = n(menu)
    for keys, names in TAB_GUESS:
        if any(m.endswith(k) if k == "차" else k in m for k in keys):
            hit = [t for t in tabs if any(x in n(t["text"]) for x in names)]
            if hit:
                return hit
    return []


def _find(els, words, kinds=("button", "back", "tab")):
    for w in words:
        for e in els:
            if e["kind"] in kinds and n(e["text"]) == n(w):
                return e
    return None


def _name_score(text, target):
    a, b = n(text), n(target)
    return 100.0 if a == b else fuzz.ratio(a, b)


def _strip_ice(t):
    s = n(t)
    for p in ICE_PREFIX:
        if s.startswith(p):
            return s[len(p):], True
    if s.endswith("(ice)"):
        return s[:-5], True
    return s, False


def match_menu(els, item, cut=72.0):
    """의도 항목에 맞는 메뉴 요소. 차가운 주문이면 '아이스 ○○' 메뉴를 먼저 찾는다."""
    want_ice = (item.get("options") or {}).get("temp") == "ice"
    best, best_s, best_ice = None, cut, False
    for e in els:
        if e["kind"] != "menu":
            continue
        base, is_ice = _strip_ice(e["text"])
        s = _name_score(base, item["menu"])
        if is_ice and not want_ice:
            continue                      # 따뜻한 주문에 아이스 메뉴는 고르지 않는다
        s += 5 if is_ice == want_ice else 0
        if s >= best_s:
            best, best_s, best_ice = e, s, is_ice
    return best, best_ice


def _opt_hit(text, word):
    a, b = n(text).split("+")[0], n(word)
    return a == b or (len(b) >= 2 and fuzz.ratio(a, b) >= 75)


def _row_buttons(els, ref):
    """ref 와 같은 줄(세로로 겹침)에 있는 버튼들, 왼쪽부터."""
    cy = (ref["box"][1] + ref["box"][3]) / 2
    return sorted([b for b in els if b["kind"] == "button" and b["box"][1] <= cy <= b["box"][3]],
                  key=lambda b: b["box"][0])


def _plus_minus(els, ref, sym):
    """수량 버튼: 글자로 먼저, 못 읽었으면 위치로(같은 줄 작은 버튼 둘 중 왼쪽 '-', 오른쪽 '+')."""
    row = _row_buttons(els, ref)
    by_text = [b for b in row if n(b["text"]) in ((sym,) if sym == "+" else ("-", "−", "ㅡ", "_"))]
    if by_text:
        return by_text[0]
    small = [b for b in row if (b["box"][2] - b["box"][0]) < 0.15]
    if len(small) >= 2:
        return small[-1] if sym == "+" else small[0]
    return None


def _pending(intent):
    return [i for i, it in enumerate(intent["items"]) if it.get("status", "pending") == "pending"]


def _say(e, text=None):
    return f"{text or e['text']}{'을' if _has_jong(text or e['text']) else '를'} 누르겠습니다"


def _has_jong(s):
    c = s.strip()[-1:] or "가"
    return "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0


def plan(screen: dict, intent: dict, pg: Progress) -> dict | None:
    els, st = screen["elements"], screen["screen_type"]

    def go(e, say=None, why=""):
        pg.last = {"id": e["id"], "text": e["text"], "screen": st, "why": why}
        pg.relook = 0
        return {"target_id": e["id"], "say": say or _say(e), "why": why}

    if st == "payment":
        return None
    if st == "other":
        dine = intent.get("dine", "매장")
        e = _find(els, [dine, "매장", "먹고가기", "시작", "주문하기"])
        if e is None:
            btns = [x for x in els if x["kind"] == "button"]
            e = max(btns, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]), default=None)
        return go(e, why="start") if e else None

    pend = _pending(intent)
    if st == "menu":
        if not pend:
            e = _find(els, CHECKOUT)
            return go(e, why="checkout") if e else None
        tabs = [x for x in els if x["kind"] == "tab"]
        for x in els:
            if x["kind"] == "menu" and pg.cur_tab:
                pg.seen[_strip_ice(x["text"])[0]] = pg.cur_tab
        item = intent["items"][pend[0]]
        e, is_ice = match_menu(els, item)
        if e:
            pg.current, pg.variant_temp = pend[0], is_ice
            pg.opt_done, pg.qty_pressed, pg.tabs_tried, pg.pages_moved = set(), 0, [], 0
            return go(e, why="menu")

        if pg.relook < 1:   # 흐린 프레임일 수 있다 → 넘기기 전에 한 번 더 읽는다
            pg.relook += 1
            return {"target_id": None, "say": "잠시 멈춰 주세요", "why": "relook"}

        def press_tab(t):
            pg.tabs_tried.append(n(t["text"]))
            pg.cur_tab, pg.pages_moved = n(t["text"]), 0
            return go(t, f"{t['text']} 탭부터 누르겠습니다", why="tab")

        known = pg.seen.get(n(item["menu"]))
        if known and known != pg.cur_tab:
            t = next((x for x in tabs if n(x["text"]) == known), None)
            if t and known not in pg.tabs_tried:
                return press_tab(t)
        for t in guess_tabs(item["menu"], tabs):
            if n(t["text"]) not in pg.tabs_tried and n(t["text"]) != pg.cur_tab:
                return press_tab(t)
            break
        nxt = _find(els, ["다음 페이지", "다음", "▶", ">"])
        if nxt and pg.pages_moved < 5:
            pg.pages_moved += 1
            return go(nxt, f"{item['menu']}이 이 화면에 없어 다음 페이지로 넘기겠습니다", why="page")
        for t in tabs:
            if n(t["text"]) not in pg.tabs_tried and n(t["text"]) != pg.cur_tab:
                return press_tab(t)
        return {"target_id": None, "say": f"{item['menu']}을 찾지 못했습니다", "why": "not_found"}

    if st == "option":
        if pg.current is None:
            e = _find(els, ["취소"])
            return go(e, why="unknown_option") if e else None
        item = intent["items"][pg.current]
        opts = item.get("options") or {}
        want = []
        if opts.get("temp") and not pg.variant_temp:
            want.append(("temp", TEMP_WORDS[opts["temp"]]))
        if opts.get("size"):
            want.append(("size", [opts["size"]]))
        for x in opts.get("extras") or []:
            want.append((f"extra:{x}", [x]))
        for key, words in want:
            if key in pg.opt_done:
                continue
            for e in els:
                if e["kind"] == "button" and any(_opt_hit(e["text"], w) for w in words):
                    pg.opt_done.add(key)
                    return go(e, why=key)
            if pg.relook < 1:
                pg.relook += 1
                return {"target_id": None, "say": "잠시 멈춰 주세요", "why": "relook"}
            pg.opt_done.add(key)          # 다시 읽어도 없는 옵션은 건너뛴다
        qlab = next((x for x in els if x["kind"] == "text" and x.get("qty") is None and n(x["text"]).isdigit()), None)
        plus = _find(els, ["+"]) or (qlab and _plus_minus(els, qlab, "+"))
        if plus and pg.qty_pressed < item.get("qty", 1) - 1:
            pg.qty_pressed += 1
            return go(plus, "수량을 하나 늘리겠습니다", why="qty")
        e = _find(els, ADD)
        return go(e, why="add") if e else None

    if st == "cart":
        rows = [x for x in els if x["kind"] == "cart_item"]
        for r in rows:
            base, _ = _strip_ice(r["text"])
            want = next((it for it in intent["items"] if _name_score(base, it["menu"]) >= 85), None)
            if r.get("qty") is None and pg.relook < 1:   # 수량 숫자를 못 읽었다
                pg.relook += 1
                return {"target_id": None, "say": "잠시 멈춰 주세요", "why": "relook"}
            have, need = r.get("qty") or 1, (want or {}).get("qty", 0)
            if have != need:
                sym = "+" if have < need else "-"
                btn = _plus_minus(els, r, sym)
                if btn:
                    word = "늘리겠습니다" if sym == "+" else "줄이겠습니다"
                    return go(btn, f"{r['text']} 수량을 {word}", why="cart_qty")
        if pend:
            e = _find(els, ["메뉴 추가", "추가 주문", "메뉴추가"])
            return go(e, why="more") if e else None
        e = _find(els, ["결제하기", "결제", "주문하기"])
        return go(e, why="pay") if e else None
    return None


def after_press(prev: dict, new: dict, intent: dict, pg: Progress) -> str:
    """F-10 대역(시뮬레이션용 간이 판정): 누른 뒤 화면 변화로 진행 상태를 갱신한다."""
    last = pg.last or {}
    if last.get("why") == "add" and new["screen_type"] in ("menu", "cart") and pg.current is not None:
        it = intent["items"][pg.current]
        it["status"] = "added"
        it["added_qty"] = 1 + pg.qty_pressed
        pg.current = None
        return "added"
    if last.get("why") == "menu" and new["screen_type"] != "option":
        pg.current = None
        return "menu_failed"
    return "ok"
