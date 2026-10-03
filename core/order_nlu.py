"""F-06 음성 주문 이해 (규칙 부분, M6) — 발화 텍스트 → 주문 의도 JSON(명세 6장) + 확인 문장.

처리 (명세 F-06 ②~⑤)
  ② 현재 화면 메뉴명·매장 메뉴 사전으로 단어 보정 : 자모 단위 유사도로 '아메리 카노'·'까페라떼'도 잡는다
  ③ 규칙 매칭 : 메뉴·수량·온도·크기·추가·매장/포장
  ④ 규칙으로 안 풀리면 소형 언어모델 (llm_fallback 자리만 있음 — source="rule" 이 아니면 거기로)
  ⑤ "따뜻한 아메리카노 두 잔 맞나요?" 확인 문장
예외: 메뉴를 못 찾으면 비슷한 메뉴 2개 제안, 아무것도 못 알아들으면 다시 말해 달라고
"""
from __future__ import annotations

import re

from functools import lru_cache

from rapidfuzz import fuzz, process

# ---------- 한글 자모 ----------
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def jamo(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out += [_CHO[c // 588], _JUNG[(c % 588) // 28], _JONG[c % 28].strip()]
        else:
            out.append(ch)
    return "".join(out)


def nsp(s: str) -> str:
    return re.sub(r"[\s.,!?~·]+", "", s or "").lower()


# ---------- 어휘 ----------
NUM = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "석": 3, "네": 4, "넷": 4, "넉": 4, "다섯": 5,
       "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5}
# 음성인식 변형: '세 잔'→'새 잔', '두 잔'→'투잔'·'도전', '잔'→'젠'·'찬'·'전'
NUM.update({"새": 3, "투": 2, "도": 2})
COUNTER = r"(?:잔|젠|쟌|찬|전|장|개|게|컵|조각|병|그릇|인분|판|세트|봉지|캔)"
QTY_RE = re.compile(r"(\d{1,2}|열|아홉|여덟|일곱|여섯|다섯|하나|한|둘|두|셋|세|새|석|넷|네|넉|투|도)\s*" + COUNTER + r"?\s*(씩)?"
                    r"|(\d{1,2}|하나|둘|셋|넷)\s*(씩)")
TEMP = {"hot": ["따뜻", "따듯", "뜨거운", "뜨겁게", "뜨끈", "핫", "하스로", "하수로", "화수로", "핫스", "hot", "데워"],
        "ice": ["아이스", "차가운", "차갑게", "시원한", "시원하게", "얼음", "ice", "아이스로", "콜드"]}
SIZE = {"톨": ["톨", "작은", "스몰", "레귤러", "기본 사이즈", "보통"],
        "그란데": ["그란데", "중간", "미디엄", "그랑데"],
        "벤티": ["벤티", "큰", "라지", "제일 큰", "가장 큰", "큰 거", "큰걸"]}
EXTRA = {"샷 추가": ["샷 추가", "샷추가", "샤 추가", "차 추가", "샷 하나 더", "진하게", "투샷", "더블샷"],
         "시럽 추가": ["시럽 추가", "시럽추가", "달게", "시럽 넣어"]}
DINE = {"매장": ["먹고 갈", "먹고갈", "먹고 가", "먹고 발", "매장", "여기서", "드시고"], "포장": ["포장", "가져갈", "가지고 갈", "테이크아웃", "들고 갈"]}
SPLIT_RE = re.compile(r"\s*(?:,|그리고|하고|이랑|랑|\s와\s|\s과\s|또|더하고)\s*")
NUM_KO = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열"}


def _josa(word: str, with_jong: str, without: str) -> str:
    c = (word or "가")[-1]
    return with_jong if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else without


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else NUM.get(tok, 1)


# ---------- 메뉴 찾기 ----------
@lru_cache(maxsize=32)
def _menu_index(menus: tuple) -> dict:
    """메뉴 글자 수 → [(메뉴명, 자모)] — 매장 사전이 커도 C 로 일괄 비교하도록."""
    idx = {}
    for m in menus:
        k = nsp(m)
        if len(k) >= 3:          # 두 글자 이하는 정확 일치로만 ('한과'·'콘' 오탐 방지)
            idx.setdefault(len(k), []).append((m, jamo(k)))
    return idx


def find_menus(text: str, menus: list[str], cut: float = 77.0, screen: tuple = ()):
    """발화에서 메뉴 언급 위치를 찾는다. 반환 [(시작, 끝, 메뉴명, 점수)] — 원문 글자 위치 기준.
    1) 띄어쓰기 무시 정확 일치  2) 자모 유사도로 음성인식 오류 보정 ('까페라떼', '아메리 카노')"""
    raw = text
    idx = [i for i, ch in enumerate(raw) if not ch.isspace()]          # 공백 제거 문자열 → 원문 위치
    s = "".join(raw[i] for i in idx).lower()
    found = []
    for m in sorted(menus, key=lambda x: -len(nsp(x))):
        k = nsp(m)
        if len(k) < 2:          # 한 글자 메뉴는 오탐이 많다
            continue
        p = s.find(k)
        while p != -1:
            span = (idx[p], idx[p + len(k) - 1] + 1)
            if not any(a < span[1] and span[0] < b for a, b, _, _ in found):
                found.append((span[0], span[1], m, 100.0))
            p = s.find(k, p + 1)
    # 자모 유사도: 아직 덮이지 않은 구간에서 메뉴 길이 ±1 글자 창으로
    covered = lambda a, b: any(x < b and a < y for x, y, _, _ in found)  # noqa: E731
    cands = []
    by_len = _menu_index(tuple(menus))
    for L in range(2, len(s) + 1):
        pool = [x for d in (-1, 0, 1) for x in by_len.get(L + d, [])]   # 메뉴 길이 ±1 글자 창
        if not pool:
            continue
        names, jams = [x[0] for x in pool], [x[1] for x in pool]
        for p in range(0, len(s) - L + 1):
            for _, sc, i in process.extract(jamo(s[p:p + L]), jams, scorer=fuzz.ratio,
                                            score_cutoff=min(cut, 75 if screen else cut), limit=3):
                # 세 글자는 엄격하게('사이즈'→'사이다' 방지). 단 지금 화면에 보이는 메뉴는 후보가 적어 조금 느슨하게
                if len(nsp(names[i])) == 3 and sc < (75 if names[i] in screen else 85):
                    continue
                cands.append((sc, p, L, names[i]))
    for sc, p, L, m in sorted(cands, key=lambda x: (-x[0], -x[2])):
        span = (idx[p], idx[p + L - 1] + 1)
        if not covered(*span):
            found.append((span[0], span[1], m, round(sc, 1)))
    return sorted(found)


# 없는 메뉴를 말했을 때 비슷한 메뉴 고르기: 같은 종류(과일 음료·커피·차·디저트) > 같은 재료 > 글자 모양
SUGGEST_GROUPS = [("주스", "쥬스", "에이드", "스무디", "프라페", "쉐이크", "셰이크"),
                  ("아메리카노", "라떼", "커피", "모카", "브루", "에스프레소", "마끼아또", "카푸치노"),
                  ("차", "티", "캐모마일", "페퍼민트", "얼그레이"),
                  ("케이크", "빵", "머핀", "마카롱", "크루아상", "베이글", "티라미수", "와플", "토스트", "쿠키", "스콘")]
FLAVORS = ("망고", "딸기", "자몽", "레몬", "청포도", "포도", "키위", "블루베리", "복숭아", "유자", "초코", "바닐라",
           "녹차", "말차", "고구마", "치즈", "카라멜", "헤이즐넛", "사과", "수박", "바나나")


def _group(s):
    return next((g for g, keys in enumerate(SUGGEST_GROUPS) if any(k in s for k in keys)), None)


def suggest(word: str, menus: list[str], k: int = 2) -> list[str]:
    w = nsp(word)
    jw, gw = jamo(w), _group(w)
    fl = {f for f in FLAVORS if f in w}

    def score(m):
        mm = nsp(m)
        return (40 if gw is not None and _group(mm) == gw else 0) + 30 * len(fl & {f for f in FLAVORS if f in mm})             + fuzz.partial_ratio(jw, jamo(mm)) * 0.3
    return sorted(menus, key=lambda m: -score(m))[:k]


# ---------- 속성 ----------
def _attrs(seg: str) -> dict:
    a = {}
    low = seg.lower()
    for t, words in TEMP.items():
        if any(w in low for w in words):
            a["temp"] = t
    for sz, words in SIZE.items():
        if any(re.search(r"(?<![가-힣])" + re.escape(w), low) for w in words):
            a["size"] = sz
    ex = [e for e, words in EXTRA.items() if any(w in low for w in words)]
    if ex:
        a["extras"] = ex
    q = None
    for m in QTY_RE.finditer(seg):
        tok = m.group(1) or m.group(3)
        has_counter = re.match(r".*" + COUNTER, m.group(0) or "") or m.group(2) or m.group(4)
        if tok in ("도", "투", "새") and not has_counter:
            continue
        if tok and (has_counter or tok.isdigit() or tok in ("하나", "둘", "셋", "넷")):
            q = _num(tok)
            if m.group(2) or m.group(4):
                a["each"] = True
    if q:
        a["qty"] = q
    return a


FOOD = ("케이크", "티라미수", "크루아상", "베이글", "마카롱", "빵", "머핀", "쿠키", "와플", "스콘", "샌드위치", "토스트")


def _is_food(menu: str) -> bool:
    """음식(디저트)에는 온도·크기·샷을 붙이지 않는다 ('아이스' 가 옆 음료에서 새어 들어오는 것 방지)."""
    return any(f in nsp(menu) for f in FOOD)


def parse(text: str, menus: list[str], store_menus: list[str] | None = None) -> dict:
    """menus: 현재 화면의 메뉴명(우선), store_menus: 매장 메뉴 사전(F-14, 화면에 없는 메뉴도)."""
    allm = list(dict.fromkeys((menus or []) + (store_menus or [])))
    out = {"items": [], "source": "rule", "text": text}
    for d, words in DINE.items():
        if any(w in text for w in words):
            out["dine"] = d
    hits = find_menus(text, allm, screen=tuple(menus or []))
    if not hits:
        out["source"] = "none"
        out["say"] = "다시 말씀해 주세요"
        guess = re.sub(r"(주세요|줘요|주문|할게요|하나|한잔|두잔|요)", " ", text).strip()
        if guess:
            out["suggest"] = suggest(guess, allm)
            if re.search(COUNTER + r"|주세요|줘|할게", text):   # 주문은 했는데 메뉴가 없다 → 비슷한 메뉴 제안
                a, b = out["suggest"]
                out["say"] = f"말씀하신 메뉴가 없습니다. {a}{_josa(a, '이나', '나')} {b}{_josa(b, '은', '는')} 어떠세요?"
        return out
    # 메뉴별 구간: 두 메뉴 사이는 '앞 메뉴의 수량'까지(그 뒤 연결어 전 속성도) 앞 메뉴 몫, 나머지는 뒤 메뉴 몫.
    # 쉼표·'하고'가 음성인식에서 빠져도 수량이 경계가 된다. '전부/모두'의 온도·크기는 전체에.
    k = len(hits)
    pre, suf = [""] * k, [""] * k
    pre[0], suf[-1] = text[:hits[0][0]], text[hits[-1][1]:]
    for i in range(k - 1):
        reg = text[hits[i][1]:hits[i + 1][0]]
        q = next((m for m in QTY_RE.finditer(reg) if _attrs(m.group(0)).get("qty")), None)
        cut = q.end() if q else None
        sp = [m for m in SPLIT_RE.finditer(reg) if cut is None or m.start() >= cut]
        if sp:
            cut = sp[-1].end()
        if cut is None:
            cut = len(reg)            # 단서가 없으면 앞 메뉴에 ('아메리카노 차가운 카페라떼')
        suf[i], pre[i + 1] = reg[:cut], reg[cut:]
    glob = {}
    if re.search(r"(전부|모두|다\s|둘\s?다|모든)", text):
        g = _attrs(re.sub(r"|".join(re.escape(text[h[0]:h[1]]) for h in hits), " ", text))
        glob = {x: v for x, v in g.items() if x in ("temp", "size")}
    for i, (a, b, m, sc) in enumerate(hits):
        at_s, at_p = _attrs(suf[i]), _attrs(pre[i])
        at = {**at_p, **at_s}
        opts = {x: v for x, v in {**glob, **at}.items() if x in ("temp", "size", "extras")}
        if _is_food(m):
            opts = {}
        q = at_s.get("qty") or at_p.get("qty") or 1
        out["items"].append({"menu": m, "qty": q, "options": opts, "status": "pending", "score": sc,
                             "on_screen": m in (menus or [])})
    # '아이스 ○○'가 별도 메뉴로 잡혔으면 기본 메뉴 + 온도로 정리 (화면에 그 메뉴가 있으면 플래너가 다시 찾는다)
    for it in out["items"]:
        if it["menu"].startswith("아이스 ") and it["menu"][4:] in allm:
            it["menu"], it["options"]["temp"] = it["menu"][4:], "ice"
    # '○○랑 △△ 두 잔씩' → 수량 없는 앞 항목에도 같은 수량
    if "씩" in text:
        each = [it["qty"] for it in out["items"] if it["qty"] > 1]
        if each:
            for it in out["items"]:
                it["qty"] = max(it["qty"], each[0])
    out["say"] = confirm(out)
    return out


def confirm(intent: dict) -> str:
    parts = []
    for it in intent["items"]:
        o = it["options"]
        t = {"hot": "따뜻한 ", "ice": "아이스 "}.get(o.get("temp"), "")
        sz = f" {o['size']}" if o.get("size") else ""
        ex = "".join(f" {e}" for e in o.get("extras", []))
        unit = "개" if any(k in it["menu"] for k in ("케이크", "빵", "마카롱", "베이글", "크루아상", "티라미수", "머핀")) else "잔"
        parts.append(f"{t}{it['menu']}{sz}{ex} {NUM_KO.get(it['qty'], str(it['qty']))} {unit}")
    tail = {"매장": ", 드시고 가시는 거", "포장": ", 포장"}.get(intent.get("dine"), "")
    return ", ".join(parts) + tail + " 맞나요?"


def llm_fallback(text: str, menus: list[str]) -> dict | None:
    """④ 규칙으로 안 풀린 발화 → 소형 언어모델. 아직 모델 미정(온디바이스 후보 조사 필요) — 자리만."""
    return None
