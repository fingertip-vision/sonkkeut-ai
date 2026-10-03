"""F-05·F-07 통합 시뮬레이션 — 가짜 키오스크 3종에서 주문을 끝까지 자동으로 진행하고 잰다.

  python run_sim.py --episodes 100                  # 인식 완벽(정답 글자) 가정
  python run_sim.py --episodes 60 --perception ocr  # 화면을 열화시켜 실제 OCR(v2)로 읽기 (요소 박스는 정답 = F-03 완벽 가정)

지표 (명세 수용 기준)
  F-05 화면 종류 분류 정확도  ≥ 95%
  F-07 다음 버튼 정답률       ≥ 95%  (시뮬레이터 내부 상태로 구한 '정답 버튼 집합'에 들면 정답)
  주문 성공률 = 결제 화면 도달 + 장바구니가 의도와 정확히 일치
"""
import argparse
import copy
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from kiosk_sim import CATALOG, HAS_TEMP, Kiosk, category_of
from planner import Progress, after_press, plan
from screen_struct import structure

TRUE_TYPE = {"start": "other", "menu": "menu", "option": "option", "cart": "cart", "payment": "payment"}


def random_intent(rng):
    names = rng.sample([n for c in CATALOG.values() for n, _ in c], rng.randint(1, 3))
    items = []
    for nm in names:
        cat = category_of(nm)[0]
        o = {}
        if cat in HAS_TEMP:
            o["temp"] = rng.choice(["hot", "ice", "ice"])
        if cat in ("커피", "음료") and rng.random() < 0.5:
            o["size"] = rng.choice(["그란데", "벤티"])
        if cat == "커피" and rng.random() < 0.3:
            o["extras"] = ["샷 추가"]
        items.append({"menu": nm, "qty": rng.choice([1, 1, 2, 3]), "options": o, "status": "pending"})
    return {"items": items, "dine": rng.choice(["매장", "포장"]), "source": "sim"}


# ---------- 정답(오라클): 시뮬레이터 내부 상태로 '맞는 다음 버튼' 집합을 구한다 ----------
def _cart_key(c):
    return (c["name"].replace("아이스 ", ""), c["temp"], c["size"], tuple(c["extras"]))


def _want_key(it):
    o = it["options"]
    cat = category_of(it["menu"])[0]
    temp = o.get("temp") or ("hot" if cat in HAS_TEMP else None)
    size = o.get("size") or ("톨" if cat in ("커피", "음료") else None)
    return (it["menu"], temp, size, tuple(o.get("extras") or []))


def oracle(k: Kiosk, intent, els):
    ids = lambda pred: {e["id"] for e in els if pred(e)}  # noqa: E731
    act = k.actions
    if k.screen == "start":
        return ids(lambda e: e["text"] == intent["dine"])
    in_cart = Counter()
    for c in k.cart:
        in_cart[_cart_key(c)] += c["qty"]
    missing = [it for it in intent["items"] if in_cart[_want_key(it)] == 0]
    if k.screen == "menu":
        if not missing:
            return ids(lambda e: act.get(e["id"], ("",))[0] == "goto" and act[e["id"]][1] == "cart")
        good = set()
        for it in missing:
            cat = category_of(it["menu"])[0]
            ice = it["options"].get("temp") == "ice"
            names = [("아이스 " + it["menu"]) if (ice and k.brand == "B" and cat in ("커피", "음료")
                                                and it["menu"] != "에스프레소") else it["menu"]]
            pos = [i for i, (nm, _) in enumerate(k.items(cat)) if nm in names]
            if not pos:
                continue
            page = pos[0] // k.per_page
            if cat != k.tab:
                good |= ids(lambda e: act.get(e["id"]) == ("tab", cat))
            elif page != k.page:
                good |= ids(lambda e: act.get(e["id"]) == ("page", 1 if page > k.page else -1))
            else:
                good |= ids(lambda e: act.get(e["id"], ("",))[0] == "menu" and act[e["id"]][1] in names)
        return good
    if k.screen == "option":
        cur = k.cur
        tgt = next((it for it in missing if it["menu"] == cur["name"].replace("아이스 ", "")), None)
        if tgt is None:
            return ids(lambda e: act.get(e["id"]) == ("cancel",))
        o, good = tgt["options"], set()
        if o.get("temp") and not cur["name"].startswith("아이스 ") and cur["temp"] != o["temp"]:
            good |= ids(lambda e: act.get(e["id"], ("",))[:3] == ("opt", "temp", o["temp"]))
        if o.get("size") and cur["size"] != o["size"]:
            good |= ids(lambda e: act.get(e["id"], ("",))[:3] == ("opt", "size", o["size"]))
        for x in o.get("extras") or []:
            if x not in cur["extras"]:
                good |= ids(lambda e: act.get(e["id"], ("",))[:3] == ("opt", "extras", x))
        if k.brand == "C" and cur["qty"] < tgt["qty"]:
            good |= ids(lambda e: act.get(e["id"]) == ("qty", 1))
        return good or ids(lambda e: act.get(e["id"]) == ("add",))
    if k.screen == "cart":
        good = set()
        want = {_want_key(it): it["qty"] for it in intent["items"]}
        for i, c in enumerate(k.cart):
            need = want.get(_cart_key(c), 0)
            if c["qty"] != need:
                good |= ids(lambda e: act.get(e["id"]) == ("cart_qty", i, 1 if c["qty"] < need else -1))
        if good:
            return good
        if missing:
            return ids(lambda e: act.get(e["id"]) == ("goto", "menu"))
        return ids(lambda e: act.get(e["id"]) == ("goto", "payment"))
    return set()


def success(k, intent):
    if k.screen != "payment":
        return False
    have = Counter()
    for c in k.cart:
        have[_cart_key(c)] += c["qty"]
    return have == Counter({_want_key(it): it["qty"] for it in intent["items"]})


# ---------- 인식 ----------
class OCRPerception:
    def __init__(self, rec_dir, seed):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ocr"))
        import make_synth as ms
        from ocr_m3 import KioskOCR
        self.ms, self.rng = ms, random.Random(seed)
        self.ocr = KioskOCR(rec_model_dir=rec_dir)

    def __call__(self, img, gt):
        lvl = self.rng.choice([0, 1, 1, 2])
        els = [{k: e[k] for k in ("id", "kind", "box")} for e in gt]
        return self.ocr.read_screen(self.ms.degrade(img, lvl), els)


def gt_perception(img, gt):
    """인식 완벽 가정. 단 화면에 글자로 없는 정보(장바구니 줄의 qty 속성)는 주지 않는다 — F-05 가 직접 찾아야 한다."""
    out = []
    for e in gt:
        o = {k: e[k] for k in ("id", "kind", "box", "text", "price") if k in e}
        if "qty" in e and e["kind"] != "cart_item":
            o["qty"] = e["qty"]
        if e["kind"] == "cart_item" and " x" in e["text"]:
            o["qty"] = e["qty"]  # 'x2' 처럼 글자에 수량이 있으면 OCR 후처리가 뽑는 것과 같다
        out.append(o)
    return out


def speech_episodes(asr_json):
    """주문 발화 100개 세트(메뉴판에 있는 90개)의 음성인식 결과 → (정답 의도, F-06 이 이해한 의도)."""
    from order_nlu import parse
    menus = [n for c in CATALOG.values() for n, _ in c]
    hyp = {x["id"]: x["hyp"] for x in json.load(open(asr_json, encoding="utf-8"))}
    eps = []
    for u in json.load(open(Path(__file__).parent / "data/utt100.json", encoding="utf-8")):
        if not u["items"]:
            continue
        truth = {"items": [dict(it, status="pending") for it in u["items"]], "dine": u["dine"] or "매장"}
        got = parse(hyp[u["id"]], menus)
        heard = {"items": [{k: it[k] for k in ("menu", "qty", "options", "status")} for it in got["items"]],
                 "dine": got.get("dine") or "매장"}
        eps.append((truth, heard))
    return eps


def run(episodes, perception, seed, log_fail, speech=None):
    rng = random.Random(seed)
    if speech:
        episodes = len(speech)
    stats = defaultdict(Counter)
    fails, t_plan = [], []
    for ep in range(episodes):
        brand = "ABC"[ep % 3]
        k = Kiosk(brand, seed=seed * 1000 + ep)
        if speech:
            truth, intent = copy.deepcopy(speech[ep][0]), copy.deepcopy(speech[ep][1])
        else:
            intent = random_intent(rng)
            truth = copy.deepcopy(intent)
        pg, prev, trace = Progress(), None, []
        for step in range(60):
            img, gt = k.render()
            t0 = time.perf_counter()
            screen = structure(perception(img, gt), step, prev)
            st_ok = screen["screen_type"] == TRUE_TYPE[k.screen]
            stats[brand]["type_ok"] += st_ok
            stats[brand]["type_n"] += 1
            if k.screen == "payment":
                break
            good = oracle(k, truth, gt)
            p = plan(screen, intent, pg)
            t_plan.append((time.perf_counter() - t0) * 1000)
            if p and p.get("why") == "relook":       # 같은 화면을 다시 찍어 읽는다 (누르지 않음)
                stats[brand]["relook"] += 1
                continue
            tid = p and p.get("target_id")
            ok = tid in good
            stats[brand]["step_ok"] += ok
            stats[brand]["step_n"] += 1
            trace.append(f"{k.screen}:{screen['screen_type']} -> {p and p.get('why')}:"
                         f"{next((e['text'] for e in gt if e['id'] == tid), None)} {'OK' if ok else 'X'}")
            if not tid:
                break
            k.press(tid)
            img2, gt2 = k.render()
            new = structure(perception(img2, gt2), step, screen)
            after_press(screen, new, intent, pg)
            prev = screen
        s = success(k, truth)
        stats[brand]["succ"] += s
        stats[brand]["ep"] += 1
        stats[brand]["steps"] += len(trace)
        if not s:
            fails.append({"ep": ep, "brand": brand, "intent": truth["items"], "trace": trace[-12:]})
    tot = sum(stats.values(), Counter())
    res = {}
    for b in list("ABC") + ["all"]:
        c = tot if b == "all" else stats[b]
        res[b] = {"screen_type_acc": round(c["type_ok"] / max(1, c["type_n"]), 4),
                  "next_button_acc": round(c["step_ok"] / max(1, c["step_n"]), 4),
                  "order_success": round(c["succ"] / max(1, c["ep"]), 4),
                  "avg_steps": round(c["steps"] / max(1, c["ep"]), 1), "relook_per_ep": round(c["relook"] / max(1, c["ep"]), 1),
                  "episodes": c["ep"]}
    res["perception_plus_plan_ms"] = round(sum(t_plan) / max(1, len(t_plan)), 1)
    if log_fail:
        json.dump(fails, open(log_fail, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=90)
    ap.add_argument("--perception", choices=["gt", "ocr"], default="gt")
    ap.add_argument("--rec-dir", default=str(Path(__file__).resolve().parent.parent / "ocr/models/kiosk_rec_v2"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speech", default=None, help="음성인식 결과 json (runs/asr_*.json) — 말로 주문한 의도로 진행")
    a = ap.parse_args()
    per = gt_perception if a.perception == "gt" else OCRPerception(a.rec_dir, a.seed)
    Path("runs").mkdir(exist_ok=True)
    tag = a.perception + ("_speech" if a.speech else "")
    res = run(a.episodes, per, a.seed, f"runs/fails_{tag}.json", speech_episodes(a.speech) if a.speech else None)
    json.dump(res, open(f"runs/sim_{tag}.json", "w", encoding="utf-8"), indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
