"""M3 평가 — 수용 기준: 메뉴명 완전 일치 95% 이상 (F-04).

정답 요소 박스(F-03가 완벽하다고 가정)를 그대로 넣고 문자만 평가한다.
  python eval_m3.py --data data/synth_val            # 사전 보정 끔/켬 둘 다
  python eval_m3.py --data data/real_v1 --no-dict     # 실제 촬영셋 (같은 json 형식)
결과는 runs/eval_<data이름>.json, 틀린 사례는 runs/errors_<data이름>.tsv
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ocr_m3 import KioskOCR


def norm(s):
    return (s or "").replace(" ", "")


def evaluate(ocr, files, tag):
    st = defaultdict(lambda: [0, 0])  # key -> [맞음, 전체]
    errs, lat = [], []
    for jf in files:
        gt = json.load(open(jf, encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(jf.with_suffix(".jpg")), np.uint8), cv2.IMREAD_COLOR)
        els = [{k: e[k] for k in ("id", "kind", "box")} for e in gt["elements"]]
        t = time.perf_counter()
        pred = ocr.read_screen(img, els)
        lat.append((time.perf_counter() - t) * 1000)
        lvl = gt.get("degrade", "-")
        for g, p in zip(gt["elements"], pred):
            k = g["kind"]
            if k == "price":
                ok = p.get("price") == g["price"]
                st["price(합계)"][0] += ok; st["price(합계)"][1] += 1
                continue
            ok_strict = p["text"] == g["text"]
            ok = norm(p["text"]) == norm(g["text"])
            for key in (k, f"{k}@lv{lvl}"):
                st[key][0] += ok; st[key][1] += 1
            if k == "menu":
                st["menu(띄어쓰기까지)"][0] += ok_strict; st["menu(띄어쓰기까지)"][1] += 1
            if "price" in g:
                pk = "price(메뉴)" if k == "menu" else "price(옵션)"
                st[pk][0] += p.get("price") == g["price"]; st[pk][1] += 1
            st["uncertain표시율"][0] += p["uncertain"]; st["uncertain표시율"][1] += 1
            if p["uncertain"]:
                st["틀린것중 uncertain"][0] += not ok
                st["uncertain중 실제로 틀림"][0] += not ok; st["uncertain중 실제로 틀림"][1] += 1
            if not ok:
                st["틀린것중 uncertain"][1] += 1
                errs.append(f"{jf.stem}\tlv{lvl}\t{k}\t{g['text']}\t{p['text']}\t{p['conf_ocr']}\t{'|'.join(p['raw'])}")
    res = {k: {"acc": round(a / n, 4), "n": n} for k, (a, n) in sorted(st.items()) if n}
    lat = np.array(lat[1:] if len(lat) > 1 else lat)
    res["latency_ms"] = {"mean": round(float(lat.mean()), 1), "p95": round(float(np.percentile(lat, 95)), 1)}
    print(f"\n== {tag} ==")
    for k, v in res.items():
        print(f"  {k:24s} {v}")
    return res, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_val")
    ap.add_argument("--no-dict", action="store_true")
    ap.add_argument("--conf", type=float, default=0.80)
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--rec-dir", default=None, help="파인튜닝 인식 모델 폴더 (export 결과)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 꼬리표")
    a = ap.parse_args()
    data = Path(a.data)
    files = sorted((data / "images").glob("*.json"))
    dict_path = data / "menu_dict.json"
    menu = json.load(open(dict_path, encoding="utf-8")) if dict_path.exists() else None

    ocr = KioskOCR(conf_thresh=a.conf, device=a.device, rec_model_dir=a.rec_dir)
    out = {}
    out["no_dict"], errs = evaluate(ocr, files, "사전 보정 없음")
    if menu and not a.no_dict:
        ocr.menu_dict = menu
        out["with_dict"], errs = evaluate(ocr, files, f"매장 메뉴 사전 보정 ({len(menu)}개)")
    Path("runs").mkdir(exist_ok=True)
    json.dump(out, open(f"runs/eval_{data.name}{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"runs/errors_{data.name}{a.tag}.tsv", "w", encoding="utf-8").write(
        "file\tlevel\tkind\tgt\tpred\tconf\traw\n" + "\n".join(errs))
    print(f"\n틀린 사례 {len(errs)}건 -> runs/errors_{data.name}{a.tag}.tsv")


if __name__ == "__main__":
    main()
