"""F-06 '후보 여러 개 + 다시 묻기 + 가까이 안내' 효과 측정 (주문 100개, v3 Whisper, GPU).

1차 시도: clean / noisy15 / noisy5 (기존 음성)
다시 물었을 때 2차 시도(시뮬레이션):
  same  : 같은 소음 세기, 다른 소음 조각 (그냥 한 번 더 말함)
  close : 안내대로 폰을 절반 거리로 → 목소리 +6dB (역제곱 법칙 가정, 실측 아님)
지표(주문 90개 + 메뉴판에 없는 10개): 맞게 확인 / 틀리게 확인 / 다시 물음(2번 다 실패)
기준(확신 threshold)별로 다시 계산하려고 후보 정보를 저장한다 → runs/voice_<cond>.json
"""
import json
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np
import torch

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
CORE = Path("D:/sonkkeutgil/core")
sys.path.insert(0, str(CORE))
from transcribe_utt100 import MENUS  # noqa: E402  (core/eval_f06 은 화면 렌더링 의존성이 있어 쓰지 않는다)


def _key(items):
    return sorted((it["menu"], it["qty"], (it["options"] or {}).get("temp") or "", (it["options"] or {}).get("size") or "",
                   tuple((it["options"] or {}).get("extras") or [])) for it in items)


def judge(gt, r):   # core/eval_f06.judge 와 같은 기준
    return _key(gt["items"]) == _key(r["items"]) and (gt.get("dine") is None or gt["dine"] == r.get("dine"))

from voice_order import VoiceOrder  # noqa: E402


def rd(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


CLEAN = {int(p.stem): rd(p) for p in sorted((CORE / "data/utt100_wav/clean").glob("*.wav"))}


def remix(i, snr, seed):
    """기존 noisy15/5 와 같은 방식(주변 손님 대화 4개 + 저역 잡음)으로 다른 소음 조각을 섞는다."""
    rng, nrng = random.Random(seed * 1000 + i), np.random.default_rng(seed * 1000 + i)
    x = np.concatenate([np.zeros(4000, np.float32), CLEAN[i], np.zeros(4000, np.float32)])
    bab = np.zeros_like(x)
    for j in rng.sample([k for k in CLEAN if k != i], 4):
        y = CLEAN[j]
        off = rng.randint(-len(y) // 2, len(x) // 2)
        a, b = max(0, off), min(len(x), off + len(y))
        bab[a:b] += y[a - off:b - off] * rng.uniform(0.5, 1)
    wn = np.cumsum(nrng.normal(0, 1, len(x)))
    wn -= np.convolve(wn, np.ones(400) / 400, "same")
    noise = bab / (np.std(bab) + 1e-6) + 0.5 * wn / (np.std(wn) + 1e-6)
    noise *= np.std(x) / (10 ** (snr / 20)) / (np.std(noise) + 1e-6)
    return np.clip(x + noise, -1, 1).astype(np.float32)


def _score(utt, r):
    ok = (r["decision"] in ("reask", "suggest")) if not utt["items"] else None
    if r.get("intent") is not None:
        ok = judge(utt, {"items": r["intent"]["items"], "dine": r["intent"].get("dine")})
    return {"decision": r["decision"], "conf": r.get("confidence", 0.0), "right": bool(ok), "snr": r["snr_db"],
            "say": r["say"], "text": r.get("text"), "top1": r["hypotheses"][0]}


def run(vo, utt, audio):
    cached = vo.nbest(audio, MENUS + __import__("voice_order").OPTION_WORDS)
    out = _score(utt, vo.understand(audio, MENUS, cached=cached))
    out["top1_only"] = _score(utt, vo.understand(audio, MENUS, use_nbest=False, cached=cached))
    return out


def main():
    vo = VoiceOrder(str(Path(__file__).parent / "ct2/whisper-elder-v3"), threshold=0.0)  # 기준은 나중에 바꿔 가며 계산
    utts = json.load(open(CORE / "data/utt100.json", encoding="utf-8"))
    base_snr = {"clean": None, "noisy15": 15, "noisy5": 5}
    for cond, snr in base_snr.items():
        rows = []
        for u in utts:
            first = run(vo, u, rd(CORE / f"data/utt100_wav/{cond}/{u['id']:03d}.wav"))
            row = {"id": u["id"], "on_menu": bool(u["items"]), "first": first}
            if snr is not None:
                row["same"] = run(vo, u, remix(u["id"], snr, 7))
                row["close"] = run(vo, u, remix(u["id"], snr + 6, 8))
            else:
                row["same"] = row["close"] = first
            rows.append(row)
        json.dump(rows, open(f"runs/voice_{cond}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(cond, "done", flush=True)


def report():
    for cond in ("clean", "noisy15", "noisy5"):
        rows = json.load(open(f"runs/voice_{cond}.json", encoding="utf-8"))
        print(f"\n[{cond}]  맞게 확인 / 틀리게 확인 / 다시 물음  (100개)")
        base = [r["first"]["top1_only"] for r in rows]
        bo = ["right" if (b["right"] and b["decision"] != "reask") else ("reask" if b["decision"] == "reask" else "wrong")
              for b in base]
        print(f"  기존(후보 1개, 항상 확인)    | 1번 시도: {bo.count('right'):3d} / {bo.count('wrong'):3d} / {bo.count('reask'):3d}")
        for th in (0.0, 0.5, 0.6, 0.7, 0.8):
            def outcome(r):
                if r["decision"] == "suggest":
                    return "right" if r["right"] else "wrong"
                if r["decision"] == "reask" or r["conf"] < th:
                    return "reask"
                return "right" if r["right"] else "wrong"
            one = [outcome(r["first"]) for r in rows]
            for retry in ("same", "close"):
                two = [o if o != "reask" else outcome(r[retry]) for o, r in zip(one, rows)]
                if retry == "same":
                    print(f"  기준 {th:.1f} | 1번 시도: {one.count('right'):3d} / {one.count('wrong'):3d} / {one.count('reask'):3d}"
                          f" | 다시 말하면: {two.count('right'):3d} / {two.count('wrong'):3d} / {two.count('reask'):3d}", end="")
                else:
                    print(f" | 가까이 말하면: {two.count('right'):3d} / {two.count('wrong'):3d} / {two.count('reask'):3d}")


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
