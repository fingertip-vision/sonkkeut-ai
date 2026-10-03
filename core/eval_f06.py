"""F-06 평가.
  python eval_f06.py text            # 100개 세트, 정답 텍스트로 의도 추출 정확도 (수용 기준 95%)
  python eval_f06.py qa              # AIHub 소상공인 주문 질의응답(카페) 주문 발화에서 상품명 추출 재현율
  python eval_f06.py asr --model small   # TTS 음성 → 음성인식 → 의도. 단어 오류율·의도 정확도·지연
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from kiosk_sim import CATALOG
from order_nlu import nsp, parse

MENUS = [n for c in CATALOG.values() for n, _ in c]
# 음성인식에 미리 알려 줄 말: 화면 메뉴명 + 옵션 단어 ('벤티'가 '팬티'로 들리는 것 방지)
HOTWORDS = MENUS + ["톨", "그란데", "벤티", "아이스", "따뜻한", "샷 추가", "시럽 추가", "포장", "매장", "잔", "개"]


def key(items):
    return sorted((it["menu"], it["qty"], (it["options"] or {}).get("temp") or "", (it["options"] or {}).get("size") or "",
                   tuple((it["options"] or {}).get("extras") or [])) for it in items)


def judge(gt, r):
    ok_items = key(gt["items"]) == key(r["items"])
    ok_dine = gt.get("dine") is None or gt["dine"] == r.get("dine")
    return ok_items and ok_dine


def eval_text(utts, texts=None):
    ok, errs = 0, []
    for u, t in zip(utts, texts or [u["text"] for u in utts]):
        r = parse(t, MENUS)
        good = judge(u, r)
        ok += good
        if not good:
            errs.append({"id": u["id"], "text": t, "gt": key(u["items"]), "got": key(r["items"]), "say": r["say"]})
    return ok / len(utts), errs


def eval_qa():
    rows = [r for f in Path("../data_aihub/order_qa").glob("*/카페_*.csv")
            for r in csv.DictReader(open(f, encoding="utf-8-sig"))]
    orders = [r for r in rows if r["인텐트"] == "주문_제품_요청" and r["발화자"] == "c" and r["상품명"]]
    vocab = sorted({p.strip() for r in orders for p in r["상품명"].split("|") if p.strip()})
    hit = tot = exact = 0
    for r in orders:
        want = {nsp(p) for p in r["상품명"].split("|") if p.strip()}
        got = {nsp(it["menu"]) for it in parse(r["발화문"], [], vocab)["items"]}
        hit += len(want & got)
        tot += len(want)
        exact += want == got
    print(f"주문 발화 {len(orders)}개, 상품명 사전 {len(vocab)}개")
    print(f"상품명 재현율 {hit / tot * 100:.1f}%  발화 단위 정확 일치 {exact / len(orders) * 100:.1f}%")


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def norm_words(s):
    return re.sub(r"[^\w\s]", " ", s).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["text", "qa", "asr"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--audio", default="data/utt100_wav")
    ap.add_argument("--hotwords", action="store_true", help="화면 메뉴명을 음성인식에 미리 알려 준다")
    a = ap.parse_args()
    utts = json.load(open("data/utt100.json", encoding="utf-8"))
    if a.mode == "text":
        acc, errs = eval_text(utts)
        print(f"의도 추출 정확도(정답 텍스트) {acc * 100:.1f}%  틀림 {len(errs)}")
        for e in errs:
            print(json.dumps(e, ensure_ascii=False))
    elif a.mode == "qa":
        eval_qa()
    else:
        from asr import ASR
        asr = ASR(a.model)
        res = {}
        for cond in sorted(p.name for p in Path(a.audio).iterdir() if p.is_dir()):
            hyps, lat, E, N = [], [], 0, 0
            for u in utts:
                wav = Path(a.audio) / cond / f"{u['id']:03d}.wav"
                t0 = time.perf_counter()
                hyp = asr(str(wav), HOTWORDS if a.hotwords else None)
                r = parse(hyp, MENUS)
                lat.append((time.perf_counter() - t0) * 1000)
                hyps.append(hyp)
                e, n = wer(" ".join(norm_words(u["text"])), " ".join(norm_words(hyp)))
                E, N = E + e, N + n
            acc, errs = eval_text(utts, hyps)
            lat.sort()
            res[cond] = {"wer": round(E / N, 3), "intent_acc": round(acc, 3), "lat_ms_mean": round(sum(lat) / len(lat)),
                         "lat_ms_p95": round(lat[int(len(lat) * 0.95)])}
            print(cond, res[cond])
            json.dump([{"id": u["id"], "ref": u["text"], "hyp": h} for u, h in zip(utts, hyps)],
                      open(f"runs/asr_{a.model}{'_hw' if a.hotwords else ''}_{cond}.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        json.dump(res, open(f"runs/f06_asr_{a.model}{'_hw' if a.hotwords else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
