"""소음 제거를 음성인식 앞에 붙이면 나아지는가 — v3 Whisper 로 비교.

방식: none / nr(noisereduce 스펙트럼 게이팅, 안드로이드 기본 NS 와 비슷한 계열) / rnnoise / df(DeepFilterNet3)
시험: sd_real(실제 소음 속 대화 300) · elder_cafe5(어르신+매장소음 5dB 300) · elder_clean(300, 깨끗한 음성 손상 확인)
      · utt100 noisy15/noisy5 (TTS 주문 → 의도 정확도는 core 에서 채점)
결과: runs/enh_<방식>.json, core/runs/asr_enh-<방식>_hw_<cond>.json
"""
import dataclasses
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
# DeepFilterNet 은 옛 torchaudio.backend 를 찾는다 → 빈 모듈로 대체
_m, _c = types.ModuleType("torchaudio.backend"), types.ModuleType("torchaudio.backend.common")


@dataclasses.dataclass
class AudioMetaData:
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = ""


_c.AudioMetaData, _m.common = AudioMetaData, _c
sys.modules["torchaudio.backend"], sys.modules["torchaudio.backend.common"] = _m, _c

from faster_whisper import WhisperModel  # noqa: E402

from eval_asr import D, Noise, cer, rd  # noqa: E402
from transcribe_utt100 import CORE, HOT  # noqa: E402


class Enh:
    def __init__(self, kind):
        self.kind = kind
        if kind == "df":
            from df.enhance import enhance, init_df
            self.model, self.state, _ = init_df()
            self.enhance = enhance
        elif kind == "nr":
            import noisereduce
            self.nr = noisereduce

    def __call__(self, x16):
        if self.kind == "none":
            return x16
        if self.kind == "nr":
            return self.nr.reduce_noise(y=x16, sr=16000, stationary=False, prop_decrease=0.8).astype(np.float32)
        x48 = resample_poly(x16, 3, 1).astype(np.float32)
        if self.kind == "df":
            y48 = self.enhance(self.model, self.state, torch.from_numpy(x48)[None]).squeeze(0).numpy()
        else:  # rnnoise
            from pyrnnoise import RNNoise
            r = RNNoise(48000)
            pcm = np.clip(x48 * 32767, -32768, 32767).astype(np.int16)
            outs = [f for _, f in r.denoise_chunk(pcm, partial=True)]
            y48 = np.concatenate([np.atleast_2d(o) for o in outs], axis=-1)[0].astype(np.float32) / 32768
        y = resample_poly(y48, 1, 3).astype(np.float32)
        return np.pad(y, (0, max(0, len(x16) - len(y))))[: len(x16)]


def asr(m, x, hot=False):
    segs, _ = m.transcribe(x, language="ko", beam_size=1, condition_on_previous_text=False,
                           hotwords=" ".join(HOT) if hot else None)
    return " ".join(s.text.strip() for s in segs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kinds", nargs="*", default=["none", "nr", "rnnoise", "df"])
    ap.add_argument("--model", default="ct2/whisper-elder-v3")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    m = WhisperModel(a.model, device="cuda", compute_type="float16")
    sd = [json.loads(l) for l in open(D / "sd_eval.jsonl", encoding="utf-8")]
    rec = lambda r: r["audio"].split("/")[-1].rsplit("_", 1)[0]  # noqa: E731
    sel = set(sorted({rec(r) for r in sd})[:20])
    sd = [r for r in sd if rec(r) not in sel][:300]
    elder = [json.loads(l) for l in open(D / "elder_val.jsonl", encoding="utf-8")][:300]
    utts = json.load(open(CORE / "data/utt100.json", encoding="utf-8"))
    for kind in a.kinds:
        tagk = f"{a.tag}-{kind}" if a.tag else kind
        enh, res = Enh(kind), {}
        noise = Noise(D / "noise_eval", seed=0)                     # 방식마다 같은 소음 조각
        sets = {"sd_real": [(rd(D / r["audio"]), r["text"]) for r in sd],
                "elder_cafe5": [(noise.mix(rd(D / r["audio"]), 5), r["text"]) for r in elder],
                "elder_clean": [(rd(D / r["audio"]), r["text"]) for r in elder]}
        for name, items in sets.items():
            E = N = 0
            te = 0.0
            for x, ref in items:
                t0 = time.perf_counter()
                y = enh(x)
                te += time.perf_counter() - t0
                e, n = cer(ref, asr(m, y))
                E, N = E + e, N + n
            res[name] = {"cer": round(E / N, 4), "enh_ms_per_s_audio": round(te / sum(len(x) for x, _ in items) * 16000 * 1000, 1)}
            print(a.tag or "v3", kind, name, res[name], flush=True)
        for cond in ("noisy15", "noisy5"):
            out = []
            for u in utts:
                x = rd(CORE / f"data/utt100_wav/{cond}/{u['id']:03d}.wav")
                out.append({"id": u["id"], "ref": u["text"], "hyp": asr(m, enh(x), hot=True)})
            json.dump(out, open(CORE / f"runs/asr_enh-{tagk}_hw_{cond}.json", "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(res, open(f"runs/enh_{tagk}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
