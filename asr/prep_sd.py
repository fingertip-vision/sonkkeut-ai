"""AIHub 568 카페·음식점 업소소음 — 소음 속 실제 대화(_SD)를 발화 단위로 잘라 16kHz 로.

라벨(TL.zip)의 dialogs: speakerText, startTime, endTime(초, 정수). 경계가 거칠어 앞뒤 0.3초 여유를 둔다.
평가용: noise_eval 에 쓴 녹음(_SN)과 같은 녹음 60개의 _SD → sd_eval (학습과 녹음 단위로 분리)
학습용: 나머지 840개 → sd_train
출력: data/sd_{train,eval}/*.wav + data/sd_{train,eval}.jsonl {"audio","text","spl","place"}
"""
import io
import json
import re
import wave
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

AI = Path("D:/sonkkeutgil/aihub/568")
OUT = Path("D:/sonkkeutgil/asr/data")


def clean(t):
    t = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", t)        # 비언어 표기 제거
    return re.sub(r"\s+", " ", t).strip()


def job(args):
    zpath, member, segs, outdir = args
    w = wave.open(io.BytesIO(zipfile.ZipFile(zpath).read(member)))
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if w.getnchannels() > 1:
        x = x.reshape(-1, w.getnchannels()).mean(1)
    g = np.gcd(sr, 16000)
    x = resample_poly(x, 16000 // g, sr // g)
    rows = []
    stem = Path(member).stem
    for k, (s, e, text, meta) in enumerate(segs):
        a, b = max(0, int((s - 0.3) * 16000)), min(len(x), int((e + 0.3) * 16000))
        if b - a < 8000 or b - a > 29 * 16000:
            continue
        name = f"{stem}_{k:03d}.wav"
        with wave.open(str(Path(outdir) / name), "wb") as o:
            o.setnchannels(1), o.setsampwidth(2), o.setframerate(16000)
            o.writeframes(np.clip(x[a:b], -32768, 32767).astype(np.int16).tobytes())
        rows.append({"audio": f"{Path(outdir).name}/{name}", "text": text, **meta})
    return rows


def main():
    tl = zipfile.ZipFile(next(AI.glob("**/TL.zip")))
    labels = {}
    for i in tl.infolist():
        if i.filename.endswith("_SD.json") and Path(i.filename).name.startswith("04_"):   # 카페·음식점
            d = json.loads(tl.read(i).decode("utf-8-sig"))
            ti = (d.get("typeInfo") or [{}])[0]
            meta = {"spl": ti.get("avgnoisespl"), "place": ti.get("place")}
            segs = []
            for dl in d.get("dialogs", []):
                t = clean(dl.get("speakerText", ""))
                try:
                    s, e = float(dl["startTime"]), float(dl["endTime"])
                except (KeyError, ValueError):
                    continue
                if t and e > s:
                    segs.append((s, e, t, meta))
            labels[Path(i.filename).stem] = segs
    nz = next(AI.glob("**/TS3_04*.zip"))
    sd = [i.filename for i in zipfile.ZipFile(nz).infolist() if i.filename.endswith("_SD.wav")]
    eval_ids = {p.stem.replace("_SN", "") for p in (OUT / "noise_eval").glob("*.wav")}
    print("SD 녹음", len(sd), "라벨 있는 것", sum(Path(m).stem in labels for m in sd), "평가 녹음", len(eval_ids))
    for split in ("eval", "train"):
        mem = [m for m in sd if Path(m).stem in labels and (Path(m).stem.replace("_SD", "") in eval_ids) == (split == "eval")]
        outdir = OUT / f"sd_{split}"
        outdir.mkdir(parents=True, exist_ok=True)
        rows = []
        with ProcessPoolExecutor(6) as ex:
            for r in ex.map(job, [(str(nz), m, labels[Path(m).stem], str(outdir)) for m in mem]):
                rows += r
        with open(OUT / f"sd_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"sd_{split}: 녹음 {len(mem)}  발화 {len(rows)}")


if __name__ == "__main__":
    main()
