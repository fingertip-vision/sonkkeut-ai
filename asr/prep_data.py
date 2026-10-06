"""Whisper 파인튜닝 데이터 준비 — zip 에서 바로 읽어 16kHz 모노로 변환 (zip 은 그대로 둔다).

  elder_val   : AIHub 94 키오스크 노인 Validation 중 2,000개 (화자 골고루) — 평가 전용
  elder_train : AIHub 94 키오스크 노인 Training 원천_6 전부 (57,568개)
  noise_train / noise_eval : AIHub 568 카페·음식점 업소소음 중 소음만 녹음(_SN) — 녹음 단위로 겹치지 않게 300 / 60개
출력: data/<이름>/*.wav + data/<이름>.jsonl  {"audio", "text", "speaker", "age", "dialect", "env"}
"""
import io
import json
import random
import wave
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

AI = Path("D:/sonkkeutgil/aihub")
OUT = Path("D:/sonkkeutgil/asr/data")
random.seed(0)


def _name(i):
    n = i.filename
    if not i.flag_bits & 0x800:
        try:
            n = n.encode("cp437").decode("cp949")
        except Exception:
            pass
    return n


def _to16k(raw: bytes):
    w = wave.open(io.BytesIO(raw))
    sr, ch = w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
    return np.clip(x, -32768, 32767).astype(np.int16)


def _write(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(x.tobytes())


def _job(args):
    zpath, members, outdir = args
    z = zipfile.ZipFile(zpath)
    done = []
    for m in members:
        dst = Path(outdir) / Path(m).name
        if not dst.exists():
            try:
                _write(dst, _to16k(z.read(m)))
            except Exception as e:  # 깨진 파일은 건너뛴다
                print("skip", m, e)
                continue
        done.append(m)
    return done


def convert(zpath, members, outdir, workers=8):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    chunks = [members[i::workers * 4] for i in range(workers * 4)]
    ok = set()
    with ProcessPoolExecutor(workers) as ex:
        for d in ex.map(_job, [(str(zpath), c, str(outdir)) for c in chunks if c]):
            ok.update(d)
    return ok


def labels(lzip):
    z = zipfile.ZipFile(lzip)
    out = {}
    for i in z.infolist():
        if i.filename.endswith(".json"):
            d = json.loads(z.read(i).decode("utf-8-sig"))
            sp = d.get("화자정보", {})
            out[Path(_name(i)).stem] = {"text": d["전사정보"]["LabelText"].strip(), "speaker": sp.get("SpeakerName"),
                                        "age": sp.get("Age"), "dialect": sp.get("Dialect"),
                                        "env": d.get("환경정보", {}).get("NoiseEnviron")}
    return out


def elder(split, n=None):
    tag = "validation" if split == "val" else "training"
    wz = next(AI.glob(f"94/**/3.키오스크_원천_*_명령어(노인)_{tag}.zip"))
    lz = next(AI.glob(f"94/**/3.키오스크_라벨링_명령어(노년)_{tag}.zip"))
    lab = labels(lz)
    members = [i.filename for i in zipfile.ZipFile(wz).infolist() if i.filename.endswith(".wav")]
    members = [m for m in members if Path(m).stem in lab]
    if n:   # 화자별로 골고루
        by = defaultdict(list)
        for m in members:
            by[lab[Path(m).stem]["speaker"]].append(m)
        for v in by.values():
            random.shuffle(v)
        pick, k = [], 0
        while len(pick) < n:
            for v in by.values():
                if k < len(v) and len(pick) < n:
                    pick.append(v[k])
            k += 1
        members = pick
    name = f"elder_{split}"
    ok = convert(wz, members, OUT / name)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for m in members:
            if m in ok:
                f.write(json.dumps({"audio": f"{name}/{Path(m).name}", **lab[Path(m).stem]}, ensure_ascii=False) + "\n")
    print(name, len(ok), "화자", len({lab[Path(m).stem]["speaker"] for m in ok}))


def noise():
    nz = next(AI.glob("568/**/TS3_04*.zip"))
    sn = sorted(i.filename for i in zipfile.ZipFile(nz).infolist() if i.filename.endswith("_SN.wav"))
    random.shuffle(sn)
    for name, part in (("noise_train", sn[:300]), ("noise_eval", sn[300:360])):
        ok = convert(nz, part, OUT / name, workers=6)
        print(name, len(ok))


if __name__ == "__main__":
    elder("val", 2000)
    noise()
    elder("train")
