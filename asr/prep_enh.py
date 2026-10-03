"""소음 제거를 건 학습·검증 데이터 미리 만들기 (v3_1 nr / v3_2 rnnoise / v3_3 df).

v3 와 같은 조건: 어르신 학습 57,568(60% 매장소음 섞기, SNR 0~20, 30% 볼륨 0.3~1.0) + 실제 소음 대화 16,567.
소음 섞는 위치·세기는 문장 번호로 고정 → 세 방식이 똑같이 섞인 소리에서 출발한다(공정 비교).
그 위에 소음 제거 → data_enh/<kind>/... 저장. 1에폭만 돌리므로 미리 한 번 섞는 것과 매번 섞는 것이 같다.
검증(체크포인트 선택): 어르신 1400~1699 깨끗·소음10dB(seed 1), 실제 소음 대화 선택용 300 — 모두 같은 소음 제거.
  python prep_enh.py nr        # CPU 병렬
  python prep_enh.py df        # GPU
"""
import json
import random
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

D = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "data_enh"


def rd(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def wr(p, x):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(np.clip(x * 32768, -32768, 32767).astype(np.int16).tobytes())


NOISE = sorted((D / "noise_train").glob("*.wav"))
NOISE_EVAL = sorted((D / "noise_eval").glob("*.wav"))
_cache = {}


def _noise(files, n, rng):
    f = rng.choice(files)
    if f not in _cache:
        if len(_cache) > 30:
            _cache.pop(next(iter(_cache)))
        _cache[f] = rd(f)
    z = _cache[f]
    s = rng.randint(0, max(0, len(z) - n - 1))
    z = z[s:s + n]
    return np.pad(z, (0, n - len(z))) if len(z) < n else z


def mix(x, files, snr, rng, vol=True):
    z = _noise(files, len(x), rng)
    z = z * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(z ** 2)) + 1e-8)) / (10 ** (snr / 20))
    y = np.clip(x + z, -1, 1).astype(np.float32)
    if vol and rng.random() < 0.3:
        y = (y * rng.uniform(0.3, 1.0)).astype(np.float32)
    return y


def source(task):
    """(출력 이름, 입력 소리) — 소음 섞기까지. task = (set, idx, row)"""
    name, i, r = task
    x = rd(D / r["audio"])
    if name == "elder_train":
        rng = random.Random(1_000_003 * 7 + i)
        if rng.random() < 0.6:
            x = mix(x, NOISE, rng.uniform(0, 20), rng)
    elif name == "val_cafe10":
        x = mix(x, NOISE_EVAL, 10, random.Random(1_000_003 + i), vol=False)
    return x


_enh = None


def work(args):
    kind, tasks = args
    global _enh
    if _enh is None:
        from enh_exp import Enh
        _enh = Enh(kind)
    rows = []
    for name, i, r in tasks:
        dst = OUT / kind / name / Path(r["audio"]).name
        if not dst.exists():
            wr(dst, _enh(source((name, i, r))))
        rows.append((name, i, dict(r, audio=f"../{OUT.name}/{kind}/{name}/{dst.name}", noisy=True)))
    return rows


def main():
    kind = sys.argv[1]
    load = lambda f: [json.loads(l) for l in open(D / f, encoding="utf-8")]  # noqa: E731
    sd_eval = load("sd_eval.jsonl")
    rec = lambda r: r["audio"].split("/")[-1].rsplit("_", 1)[0]  # noqa: E731
    sel = set(sorted({rec(r) for r in sd_eval})[:20])
    val = load("elder_val.jsonl")[1400:1700]
    sets = {"elder_train": load("elder_train.jsonl"), "sd_train": load("sd_train.jsonl"),
            "val_clean": val, "val_cafe10": val, "val_sd": [r for r in sd_eval if rec(r) in sel][:300]}
    tasks = [(n, i, r) for n, rows in sets.items() for i, r in enumerate(rows)]
    for n in sets:
        (OUT / kind / n).mkdir(parents=True, exist_ok=True)
    workers = 1 if kind == "df" else 12
    chunks = [tasks[k::workers * 8] for k in range(workers * 8)]
    out = {n: [None] * len(rows) for n, rows in sets.items()}
    done = 0
    if workers == 1:
        it = map(work, [(kind, c) for c in chunks])
    else:
        ex = ProcessPoolExecutor(workers)
        it = ex.map(work, [(kind, c) for c in chunks])
    for rows in it:
        for n, i, r in rows:
            out[n][i] = r
        done += len(rows)
        print(f"{kind} {done}/{len(tasks)}", flush=True)
    for n, rows in out.items():
        with open(OUT / kind / f"{n}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(kind, "DONE", flush=True)


if __name__ == "__main__":
    main()
