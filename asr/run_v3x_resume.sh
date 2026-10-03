#!/usr/bin/env bash
# v3x 재시작: v3_1(nr)·v3_2(rnnoise) 학습 → DF 데이터 끝나면 v3_3(df) → 평가 (prep 은 이미 끝났거나 진행 중)
cd /d/sonkkeutgil/asr
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
LOG=v3x.log
log() { echo "[$(date +%T)] $*" >> $LOG; }
log "재시작 (--enh-root 옵션 누락 수정)"
for pair in "v3_1 nr" "v3_2 rnnoise" "v3_3 df"; do
  set -- $pair; V=$1; K=$2; RUN=whisper-elder-$V
  while [ ! -f data_enh/$K/val_sd.jsonl ] || [ ! -f data_enh/$K/elder_train.jsonl ]; do sleep 60; done
  log "학습 $V ($K) 시작"
  $PY train_whisper.py --base openai/whisper-small --out out/$RUN --epochs 1 --freeze-decoder \
      --enh-root data_enh/$K --select sd > train_$V.log 2>&1
  log "학습 $V 끝 $?"
  .venv/Scripts/ct2-transformers-converter --model out/$RUN/best --output_dir ct2/$RUN --quantization float16 \
      --copy_files tokenizer_config.json preprocessor_config.json --force > /dev/null 2>&1
  $PY enh_exp.py --model ct2/$RUN --tag $V $K 2>&1 | grep -E "^$V " >> $LOG
done
(cd ../core && ../ocr/.venv/Scripts/python - <<'PY'
import json
from eval_f06 import eval_text
utts = json.load(open("data/utt100.json", encoding="utf-8"))
for t in ("none", "nr", "rnnoise", "df", "v3_1-nr", "v3_2-rnnoise", "v3_3-df"):
    row = []
    for c in ("noisy15", "noisy5"):
        h = {x["id"]: x["hyp"] for x in json.load(open(f"runs/asr_enh-{t}_hw_{c}.json", encoding="utf-8"))}
        row.append(f"{c} {eval_text(utts, [h[u['id']] for u in utts])[0] * 100:.0f}%")
    print("주문100 의도", t, row)
PY
) >> $LOG 2>&1
log "ALL_DONE"
