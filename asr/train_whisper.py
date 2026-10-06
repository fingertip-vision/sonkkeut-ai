"""Whisper 파인튜닝 — 어르신 키오스크 명령 음성(AIHub 94) + 실제 매장 소음(AIHub 568) 섞기.

  python train_whisper.py --base openai/whisper-small --out out/whisper-elder-v1
검증(체크포인트 선택): elder_val 1400~1699 (300개, 깨끗 + 매장 소음 10dB)
최종 시험은 eval_asr.py 가 elder_val 0~599 로 따로 잰다 (선택에 안 쓴 부분)
과적합·망각 방지: 낮은 학습률(1e-5), 2에폭 이내, 소음 섞기 60%
"""
import argparse
import json
import random
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

D = Path(__file__).parent / "data"


def rd(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class ElderSet(torch.utils.data.Dataset):
    def __init__(self, rows, proc, noise_dir=None, p_noise=0.0, fixed_snr=None, seed=0):
        self.rows, self.proc, self.p, self.fixed = rows, proc, p_noise, fixed_snr
        self.noise = sorted(Path(noise_dir).glob("*.wav")) if noise_dir else []
        self.seed = seed
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _noise(self, n, rng):
        f = rng.choice(self.noise)
        if f not in self._cache:
            if len(self._cache) > 40:
                self._cache.pop(next(iter(self._cache)))
            self._cache[f] = rd(f)
        z = self._cache[f]
        s = rng.randint(0, max(0, len(z) - n - 1))
        z = z[s:s + n]
        return np.pad(z, (0, n - len(z))) if len(z) < n else z

    def __getitem__(self, i):
        r = self.rows[i]
        x = rd(D / r["audio"])
        rng = random.Random(self.seed * 1_000_003 + i) if self.fixed is not None else random
        if self.noise and not r.get("noisy") and (self.fixed is not None or rng.random() < self.p):
            snr = self.fixed if self.fixed is not None else rng.uniform(0, 20)
            z = self._noise(len(x), rng)
            z = z * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(z ** 2)) + 1e-8)) / (10 ** (snr / 20))
            x = np.clip(x + z, -1, 1).astype(np.float32)
            if rng.random() < 0.3:          # 마이크 거리·볼륨 차이
                x = (x * rng.uniform(0.3, 1.0)).astype(np.float32)
        feat = self.proc.feature_extractor(x, sampling_rate=16000).input_features[0]
        lab = self.proc.tokenizer(r["text"]).input_ids
        return {"input_features": feat, "labels": lab}


@dataclass
class Collator:
    proc: WhisperProcessor
    start_id: int

    def __call__(self, feats):
        x = self.proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        y = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (y[:, 0] == self.start_id).all():
            y = y[:, 1:]
        x["labels"] = y
        return x


def norm(s):
    return re.sub(r"[^\w]|_", "", s or "")


def char_err(refs, hyps):
    E = N = 0
    for r, h in zip(refs, hyps):
        r, h = norm(r), norm(h)
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        E, N = E + d[len(h)], N + max(1, len(r))
    return E / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--out", default="out/whisper-elder-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--freeze-decoder", action="store_true",
                    help="인코더(소리)만 학습 — 디코더(어휘·hotwords 활용)를 지켜 망각을 막는다")
    ap.add_argument("--extra", nargs="*", default=[], help="추가 학습 manifest (이미 시끄러운 실제 음성 — 소음 안 섞음)")
    ap.add_argument("--select", default="cafe10", choices=["cafe10", "sd"], help="체크포인트 고르는 기준")
    ap.add_argument("--resume", default=None, help="이어서 학습할 체크포인트 폴더 (또는 'last')")
    a = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(a.base, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base)
    model.generation_config.language, model.generation_config.task = "korean", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if a.freeze_decoder:
        for p in model.model.decoder.parameters():
            p.requires_grad = False
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"디코더 고정: 학습 파라미터 {n_tr / 1e6:.0f}M / {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)

    train = [json.loads(l) for l in open(D / "elder_train.jsonl", encoding="utf-8")]
    val = [json.loads(l) for l in open(D / "elder_val.jsonl", encoding="utf-8")][1400:1700]
    ds_train = ElderSet(train, proc, D / "noise_train", p_noise=0.6)
    for m in a.extra:
        ex = [dict(json.loads(l), noisy=True) for l in open(D / f"{m}.jsonl", encoding="utf-8")]
        print(f"추가 {m}: {len(ex)}", flush=True)
        train += ex
    ds_val = {"clean": ElderSet(val, proc),
              "cafe10": ElderSet(val, proc, D / "noise_eval", fixed_snr=10, seed=1)}
    if (D / "sd_eval.jsonl").exists():   # 실제 소음 속 대화: 녹음 앞 20개는 선택용, 나머지는 eval_asr 시험용
        sd = [json.loads(l) for l in open(D / "sd_eval.jsonl", encoding="utf-8")]
        recs = sorted({r["audio"].split("/")[-1].rsplit("_", 1)[0] for r in sd})[:20]
        ds_val["sd"] = ElderSet([r for r in sd if r["audio"].split("/")[-1].rsplit("_", 1)[0] in recs][:300], proc)

    def metrics(pred):
        ids = pred.predictions
        lab = np.where(pred.label_ids != -100, pred.label_ids, proc.tokenizer.pad_token_id)
        return {"cer": char_err(proc.batch_decode(lab, skip_special_tokens=True),
                                proc.batch_decode(ids, skip_special_tokens=True))}

    steps_per_epoch = len(train) // (a.bs * 2)
    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=2,
        learning_rate=a.lr, warmup_steps=300, max_steps=int(steps_per_epoch * a.epochs),
        fp16=True, eval_strategy="steps", eval_steps=400, save_steps=400, save_total_limit=3,
        per_device_eval_batch_size=32, predict_with_generate=True, generation_max_length=96,
        logging_steps=50, load_best_model_at_end=True, metric_for_best_model=f"eval_{a.select}_cer",
        greater_is_better=False, dataloader_num_workers=6, report_to="none", remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                        data_collator=Collator(proc, model.config.decoder_start_token_id), compute_metrics=metrics,
                        processing_class=proc.feature_extractor)
    if not a.resume:
        print("step-0 eval", tr.evaluate(), flush=True)
    tr.train(resume_from_checkpoint=(True if a.resume == "last" else a.resume))
    tr.save_model(a.out + "/best")
    proc.save_pretrained(a.out + "/best")
    print("final", tr.evaluate(), flush=True)


if __name__ == "__main__":
    main()
