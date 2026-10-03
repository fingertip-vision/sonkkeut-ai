#!/usr/bin/env bash
# v3: 디코더 고정 + 어르신 음성(매장소음 섞기) + 실제 매장소음 속 대화 → 1에폭, 실제 소음 대화로 체크포인트 선택
cd /d/sonkkeutgil/asr
export PYTHONIOENCODING=utf-8
RUN=whisper-elder-v3
.venv/Scripts/python train_whisper.py --base openai/whisper-small --out out/$RUN --epochs 1 --freeze-decoder \
    --extra sd_train --select sd > train_v3.log 2>&1
echo "TRAIN_EXIT $?" >> train_v3.log
.venv/Scripts/ct2-transformers-converter --model out/$RUN/best --output_dir ct2/$RUN --quantization float16 \
    --copy_files tokenizer_config.json preprocessor_config.json --force > after_v3.log 2>&1
for M in "small base_small" "ct2/whisper-elder-v2 whisper-elder-v2" "ct2/$RUN $RUN"; do
  set -- $M
  echo "== $2" >> after_v3.log
  .venv/Scripts/python eval_asr.py --model $1 --tag $2 --set sd 2>&1 | grep -E "^real" >> after_v3.log
  [ "$2" = "$RUN" ] && .venv/Scripts/python eval_asr.py --model $1 --tag $2 2>&1 | grep -E "^(clean|cafe)" >> after_v3.log
done
.venv/Scripts/python transcribe_utt100.py --model ct2/$RUN --tag $RUN 2>&1 | grep -E "^(clean|noisy)" >> after_v3.log
.venv/Scripts/python transcribe_utt100.py --model ct2/$RUN --tag $RUN --no-hot 2>&1 | grep -E "^(clean|noisy)" >> after_v3.log
(cd ../core && ../ocr/.venv/Scripts/python - <<'PY'
import json
from eval_f06 import eval_text
utts = json.load(open("data/utt100.json", encoding="utf-8"))
for t in ("base_small", "whisper-elder-v2", "whisper-elder-v3"):
    for hw in ("hw", "nohw"):
        row = []
        for c in ("clean", "noisy15", "noisy5"):
            h = {x["id"]: x["hyp"] for x in json.load(open(f"runs/asr_{t}_{hw}_{c}.json", encoding="utf-8"))}
            row.append(f"{eval_text(utts, [h[u['id']] for u in utts])[0] * 100:.0f}%")
        print(f"  주문100 {t} {hw}: {row}")
PY
) >> after_v3.log 2>&1
echo "ALL_DONE" >> after_v3.log
