"""F-05 화면 구조화 — F-03 요소 목록 + F-04 문자 → 화면 구조 JSON(명세 6장).

입력 요소: {"id", "kind", "box", "conf"?} (F-03) + {"text", "price"?, "qty"?, "conf_ocr"?, "uncertain"?} (F-04)
처리: ① 요소와 텍스트 결합(뒤로가기 재분류) ② 화면 종류 분류 ③ 직전 화면과 비교해 변경 여부·차이
출력: {"screen_type", "keyframe_id", "elements", "changed", "diff", "cart", "total", "hint"}
screen_type: menu | option | cart | payment | other   (분류 불확실 → other, F-12 화면 읽기 제안은 앱 쪽)
"""
from __future__ import annotations

import re
from collections import Counter

BACK_WORDS = {"이전", "뒤로", "뒤로가기", "돌아가기"}
# '결제 금액'은 주문 확인 화면에도 나오므로 결제 화면 판단에 쓰지 않는다
PAY_WORDS = ["카드를넣어", "ic카드", "카드를투입", "결제수단", "결제방법", "삼성페이", "바코드를", "카드를꽂아"]
CART_WORDS = ["주문확인", "주문내역확인", "장바구니목록", "선택하신메뉴"]
OPTION_WORDS = {"온도", "사이즈", "크기", "추가", "옵션", "수량", "hot", "ice", "샷추가", "톨", "그란데", "벤티",
                "레귤러", "라지", "따뜻하게", "차갑게"}
ADD_WORDS = {"담기", "장바구니담기", "선택완료", "추가하기"}
START_WORDS = ["화면을터치", "터치해", "주문하실곳", "매장", "포장"]
TOTAL_WORDS = ("합계", "총", "결제금액", "주문금액")
_KEEP = ("id", "kind", "text", "price", "qty", "box", "conf", "uncertain")


def n(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()


def _kind(e: dict) -> str:
    k = e.get("kind", "button")
    if k == "button" and n(e.get("text")) in BACK_WORDS:
        return "back"
    return k


def classify(els: list[dict]) -> tuple[str, str | None]:
    kinds = Counter(e["kind"] for e in els)
    texts = [n(e.get("text")) for e in els]
    joined = "|".join(texts)
    menus = kinds["menu"]
    if menus == 0 and (kinds["cart_item"] >= 1 and any(t in ("+", "-", "−") for t in texts)
                       or any(w in joined for w in CART_WORDS)):
        return "cart", None
    if menus == 0 and any(w in joined for w in PAY_WORDS):
        return "payment", None
    # 옵션 없는 메뉴(디저트 등)도 '담기'가 있으면 옵션 화면
    opt_hits = sum(t in OPTION_WORDS or t.split("+")[0] in OPTION_WORDS for t in texts)
    if menus <= 1 and any(t in ADD_WORDS for t in texts) and (opt_hits >= 1 or kinds["menu"] + kinds["title"] >= 1
                                                              or any(e.get("price") for e in els)):
        return "option", None
    if menus >= 2 or (kinds["tab"] >= 2 and menus >= 1):
        return "menu", None
    if menus == 0 and sum(any(w in t for w in START_WORDS) for t in texts) >= 2:
        return "other", "start"
    return "other", None


def _attach_row_qty(els):
    """장바구니 줄의 수량은 메뉴명과 떨어진 숫자 글자다 → 같은 줄(세로 겹침)의 숫자를 qty 로 붙인다."""
    nums = [e for e in els if e["kind"] in ("text", "price") and re.fullmatch(r"\d{1,2}", n(e["text"]))]
    for r in els:
        if r["kind"] != "cart_item" or r.get("qty"):
            continue
        cy = (r["box"][1] + r["box"][3]) / 2
        row = [x for x in nums if x["box"][1] <= cy <= x["box"][3]
               or r["box"][1] <= (x["box"][1] + x["box"][3]) / 2 <= r["box"][3]]
        if row:
            r["qty"] = int(n(min(row, key=lambda x: abs((x["box"][1] + x["box"][3]) / 2 - cy))["text"]))


def _total(els):
    for e in els:
        if e["kind"] == "price" and e.get("price") is not None and any(w in n(e.get("text")) for w in TOTAL_WORDS):
            return e["price"]
    return None


def structure(elements: list[dict], keyframe_id: int = 0, prev: dict | None = None) -> dict:
    els = []
    for e in elements:
        o = {k: e[k] for k in _KEEP if k in e and e[k] is not None}
        o["kind"] = _kind(e)
        o["text"] = (e.get("text") or "").strip()
        els.append(o)
    _attach_row_qty(els)
    st, hint = classify(els)
    cart = [{"text": e["text"], "qty": e.get("qty"), "price": e.get("price")} for e in els if e["kind"] == "cart_item"]
    out = {"screen_type": st, "keyframe_id": keyframe_id, "elements": els, "cart": cart, "total": _total(els)}
    if hint:
        out["hint"] = hint
    if prev is None:
        out["changed"], out["diff"] = True, None
    else:
        a, b = Counter(n(e["text"]) for e in prev["elements"]), Counter(n(e["text"]) for e in els)
        inter, union = sum((a & b).values()), sum((a | b).values()) or 1
        out["changed"] = st != prev["screen_type"] or inter / union < 0.85
        out["diff"] = {"added": sorted((b - a).elements()), "removed": sorted((a - b).elements()),
                       "similarity": round(inter / union, 3),
                       "cart_count": (sum(c["qty"] or 1 for c in cart), sum(c["qty"] or 1 for c in prev["cart"])),
                       "total": (out["total"], prev["total"])}
    return out
