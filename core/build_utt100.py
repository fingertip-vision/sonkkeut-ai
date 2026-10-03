"""F-06 주문 발화 100개 세트 (명세 7장 '주문 발화 100개 세트') — 텍스트 + 정답 의도.
시뮬레이터 메뉴판 기준. 어르신 말투·군말·숫자 표기·속성 위치·여러 항목·'씩'·매장/포장·메뉴판에 없는 메뉴를 섞는다.
  python build_utt100.py   → data/utt100.json
"""
import json
import random

from kiosk_sim import CATALOG, HAS_TEMP, category_of

rng = random.Random(100)
NUMW = {1: ["한", "1"], 2: ["두", "2"], 3: ["세", "3"]}
FILL = ["", "", "", "음 ", "저기요 ", "어 ", "그 ", "저 "]
POLITE = ["주세요", "주세요", "줘요", "주문할게요", "할게요", "부탁해요", "주시겠어요", "이요", "요"]
TEMPW = {"hot": ["따뜻한", "뜨거운", "따뜻하게", "핫으로"], "ice": ["아이스", "차가운", "시원한", "아이스로", "차갑게"]}
SIZEW = {"그란데": ["그란데", "중간 사이즈", "그란데 사이즈"], "벤티": ["벤티", "큰 사이즈", "제일 큰 걸로", "벤티로"]}
DINEW = {"매장": ["먹고 갈게요", "매장에서 먹을게요", "여기서 먹고 가요"], "포장": ["포장해 주세요", "포장이요", "가져갈게요"]}
NOT_ON_MENU = ["망고주스", "생과일 키위주스", "팥빙수", "아이스크림", "블루베리 머핀", "핫도그", "토스트", "밀크쉐이크",
               "수박주스", "와플"]


def unit(name):
    return "개" if category_of(name)[0] == "디저트" else "잔"


def phrase(it):
    nm, q, o = it["menu"], it["qty"], it["options"]
    t = rng.choice(TEMPW[o["temp"]]) if o.get("temp") else ""
    sz = rng.choice(SIZEW[o["size"]]) if o.get("size") else ""
    ex = "샷 추가해서" if o.get("extras") else ""
    qw = f"{rng.choice(NUMW[q])} {unit(nm)}" if rng.random() < 0.8 else (f"{rng.choice(NUMW[q])}{unit(nm)}")
    if q == 1 and rng.random() < 0.3:
        qw = "하나"
    form = rng.randint(0, 3)
    if form == 0 or not t:      # 따뜻한 아메리카노 그란데 두 잔
        parts = [t if t and not t.endswith(("로", "게")) else "", nm, t if t.endswith(("로", "게")) else "", sz, ex, qw]
    elif form == 1:             # 아메리카노 아이스로 두 잔
        parts = [nm, t, sz, ex, qw]
    elif form == 2:             # 아메리카노 두 잔 아이스로
        parts = [nm, qw, t, sz, ex]
    else:                       # 아이스 큰 사이즈로 아메리카노 두 잔
        parts = [t, sz, nm, ex, qw]
    return " ".join(p for p in parts if p)


def make_item():
    nm = rng.choice([n for c in CATALOG.values() for n, _ in c])
    cat = category_of(nm)[0]
    o = {}
    if cat in HAS_TEMP and rng.random() < 0.8:
        o["temp"] = rng.choice(["hot", "ice"])
    if cat in ("커피", "음료") and rng.random() < 0.35:
        o["size"] = rng.choice(["그란데", "벤티"])
    if cat == "커피" and rng.random() < 0.2:
        o["extras"] = ["샷 추가"]
    return {"menu": nm, "qty": rng.choice([1, 1, 1, 2, 2, 3]), "options": o}


def main():
    out = []
    for i in range(100):
        if i >= 90:   # 메뉴판에 없는 메뉴 → 의도 없음 + 비슷한 메뉴 제안이 정답
            nm = NOT_ON_MENU[i - 90]
            out.append({"id": i, "text": f"{rng.choice(FILL)}{nm} {rng.choice(['하나', '한 잔', '두 개'])} {rng.choice(POLITE)}",
                        "items": [], "dine": None, "kind": "not_on_menu"})
            continue
        n = rng.choice([1, 1, 1, 2, 2, 3])
        items = []
        while len(items) < n:
            it = make_item()
            if all(x["menu"] != it["menu"] for x in items):
                items.append(it)
        if n == 2 and rng.random() < 0.25:   # '하나씩'
            q = rng.choice([1, 2])
            for it in items:
                it["qty"] = q
            body = f"{items[0]['menu']}{rng.choice(['랑', '하고', ' 그리고'])} {items[1]['menu']} {'하나씩' if q == 1 else '두 잔씩'}"
            for it in items:
                it["options"] = {}
            kind = "each"
        else:
            ps = [phrase(it) for it in items]
            body = ps[0]
            for p in ps[1:]:
                j = rng.choice([" 하고 ", " 그리고 ", ", ", "랑 "])
                if j == "랑 " and (ord(body[-1]) - 0xAC00) % 28:
                    j = "이랑 "
                body += j + p
            kind = f"items{n}"
        dine = rng.choice([None, None, "매장", "포장"])
        tail = rng.choice(POLITE)
        text = f"{rng.choice(FILL)}{body} {tail}" + (f" {rng.choice(DINEW[dine])}" if dine else "")
        out.append({"id": i, "text": text.replace("  ", " ").strip(), "items": items, "dine": dine, "kind": kind})
    json.dump(out, open("data/utt100.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for x in out[:12] + out[90:93]:
        print(x["text"])


if __name__ == "__main__":
    main()
