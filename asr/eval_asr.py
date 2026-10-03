"""음성인식 평가 — 실제 어르신 키오스크 음성(AIHub 94 검증 2,000개 중 일부) + 실제 매장 소음 섞기.

  python eval_asr.py --model small                    # 공식 Whisper
  python eval_asr.py --model ct2/whisper-elder-v1     # 파인튜닝 후 CTranslate2 변환본
지표: CER(글자 오류율, 띄어쓰기·문장부호 무시). 조건: clean / cafe10(매장 소음 SNR 10dB) / cafe5
"""
import argparse
import json
import os
import random
import re
import time
import wave
from pathlib import Path

import numpy as np

import torch  # noqa: F401  (torch/lib 의 cuBLAS·cuDNN DLL 을 ctranslate2 가 쓰도록)
os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
from faster_whisper import WhisperModel  # noqa: E402

D = Path(__file__).parent / "data"


def rd(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class Noise:
    def __init__(self, folder, seed=0):
        self.files = sorted(Path(folder).glob("*.wav"))
        self.rng = random.Random(seed)
        self.cache = {}

    def mix(self, x, snr):
        f = self.rng.choice(self.files)
        if f not in self.cache:
            self.cache[f] = rd(f)
        n = self.cache[f]
        s = self.rng.randint(0, max(0, len(n) - len(x) - 1))
        n = n[s:s + len(x)]
        if len(n) < len(x):
            n = np.pad(n, (0, len(x) - len(n)))
        n = n * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(n ** 2)) + 1e-8)) / (10 ** (snr / 20))
        return np.clip(x + n, -1, 1).astype(np.float32)


SINO = "영일이삼사오육칠팔구"
NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
          20: "스물", 30: "서른", 40: "마흔", 50: "쉰", 60: "예순", 70: "일흔", 80: "여든", 90: "아흔"}
NATIVE_UNIT = ("시", "개", "잔", "장", "명", "살", "마리", "번째", "병", "권", "대", "그릇", "가지", "분이")


def sino(n):
    if n == 0:
        return "영"
    out = ""
    for v, u in ((10000, "만"), (1000, "천"), (100, "백"), (10, "십")):
        q, n = divmod(n, v)
        if q:
            out += ("" if q == 1 else sino(q)) + u
    return out + (SINO[n] if n else "")


def native(n):
    if n == 20:
        return "스무"
    t, o = divmod(n, 10)
    return NATIVE.get(t * 10, "") + (NATIVE[o] if o else "") if n < 100 else sino(n)


def num2kor(s):
    """'6시 20분 2장' → '여섯시 이십분 두장' (라벨이 숫자를 한글로 적으므로 받아쓰기 쪽을 맞춘다)."""
    def rep(m):
        n, unit = int(m.group(1).replace(",", "")), m.group(2)
        return (native(n) if unit.startswith(NATIVE_UNIT) and n < 100 else sino(n)) + unit
    return re.sub(r"(\d[\d,]*)\s*([가-힣]*)", rep, s)


def norm(s):
    return re.sub(r"[^\w]|_", "", num2kor(s or ""))


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", default="elder", choices=["elder", "sd"],
                    help="elder: 어르신 음성(+소음 섞기) / sd: 실제 매장 소음 속 실제 대화(시험용 녹음 40개)")
    a = ap.parse_args()
    if a.set == "sd":
        sd = [json.loads(l) for l in open(D / "sd_eval.jsonl", encoding="utf-8")]
        rec = lambda r: r["audio"].split("/")[-1].rsplit("_", 1)[0]  # noqa: E731
        sel = set(sorted({rec(r) for r in sd})[:20])            # 학습 중 체크포인트 선택에 쓴 녹음은 뺀다
        rows = [r for r in sd if rec(r) not in sel][: a.n]
    else:
        rows = [json.loads(l) for l in open(D / "elder_val.jsonl", encoding="utf-8")][: a.n]
    seen = {json.loads(l)["text"] for l in open(D / "elder_train.jsonl", encoding="utf-8")}
    m = WhisperModel(a.model, device=a.device, compute_type="float16" if a.device == "cuda" else "int8")
    noise = Noise(D / "noise_eval")
    res, dump = {}, []
    conds = (("real", None),) if a.set == "sd" else (("clean", None), ("cafe10", 10), ("cafe5", 5))
    for cond, snr in conds:
        E = N = Eu = Nu = 0
        lat = []
        for r in rows:
            x = rd(D / r["audio"])
            if snr is not None:
                x = noise.mix(x, snr)
            t0 = time.perf_counter()
            segs, _ = m.transcribe(x, language="ko", beam_size=1, condition_on_previous_text=False)
            hyp = " ".join(s.text.strip() for s in segs)
            lat.append(time.perf_counter() - t0)
            e, n = cer(r["text"], hyp)
            E, N = E + e, N + n
            if r["text"] not in seen:          # 학습 대본에 없는 문장만 (외운 효과 제외)
                Eu, Nu = Eu + e, Nu + n
            if len(dump) < 40 and cond != "clean" or len(dump) < 20:
                dump.append({"cond": cond, "ref": r["text"], "hyp": hyp})
        res[cond] = {"cer": round(E / N, 4), "cer_unseen_sent": round(Eu / max(1, Nu), 4),
                     "lat_s": round(sum(lat) / len(lat), 3), "n": len(rows),
                     "n_unseen": sum(r["text"] not in seen for r in rows)}
        print(cond, res[cond], flush=True)
    tag = (a.tag or Path(a.model).name) + ("_sd" if a.set == "sd" else "")
    Path("runs").mkdir(exist_ok=True)
    json.dump({"result": res, "samples": dump}, open(f"runs/eval_{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
