#!/usr/bin/env bash
# v2: 디코더 고정(인코더만) 1에폭 → 변환 → 어르신 시험 + 주문100(힌트 있음/없음)
cd /d/sonkkeutgil/asr
export PYTHONIOENCODING=utf-8
RUN=whisper-elder-v2
.venv/Scripts/python train_whisper.py --base openai/whisper-small --out out/$RUN --epochs 1 --freeze-decoder > train_v2.log 2>&1
echo "TRAIN_EXIT $?" >> train_v2.log
.venv/Scripts/ct2-transformers-converter --model out/$RUN/best --output_dir ct2/$RUN --quantization float16 \
    --copy_files tokenizer_config.json preprocessor_config.json --force > after_v2.log 2>&1
.venv/Scripts/python eval_asr.py --model ct2/$RUN --tag $RUN 2>&1 | grep -E "^(clean|cafe)" >> after_v2.log
.venv/Scripts/python transcribe_utt100.py --model ct2/$RUN --tag $RUN 2>&1 | grep -E "^(clean|noisy)" >> after_v2.log
.venv/Scripts/python transcribe_utt100.py --model ct2/$RUN --tag $RUN --no-hot 2>&1 | grep -E "^(clean|noisy)" >> after_v2.log
(cd ../core && ../ocr/.venv/Scripts/python - <<'PY'
import json
from eval_f06 import eval_text
utts = json.load(open("data/utt100.json", encoding="utf-8"))
for t in ("base_small", "whisper-elder-v1", "whisper-elder-v2"):
    for hw in ("hw", "nohw"):
        row = []
        for c in ("clean", "noisy15", "noisy5"):
            h = {x["id"]: x["hyp"] for x in json.load(open(f"runs/asr_{t}_{hw}_{c}.json", encoding="utf-8"))}
            row.append(f"{eval_text(utts, [h[u['id']] for u in utts])[0] * 100:.0f}%")
        print(f"  주문100 {t} {hw}: {row}")
PY
) >> after_v2.log 2>&1
echo "ALL_DONE" >> after_v2.log
