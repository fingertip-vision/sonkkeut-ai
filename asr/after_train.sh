#!/usr/bin/env bash
# 파인튜닝 끝난 Whisper → CTranslate2 변환 → 어르신 목소리 시험 + 주문 100개 의도 정확도 (튜닝 전과 비교)
#   bash after_train.sh whisper-elder-v1
cd "$(dirname "$0")"
RUN=${1:-whisper-elder-v1}
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
.venv/Scripts/ct2-transformers-converter --model out/$RUN/best --output_dir ct2/$RUN --quantization float16 \
    --copy_files tokenizer_config.json preprocessor_config.json --force 2>&1 | tail -1
cp out/$RUN/best/tokenizer.json ct2/$RUN/ 2>/dev/null
for M in small ct2/$RUN; do
  T=$( [ "$M" = small ] && echo base_small || echo $RUN )
  $PY eval_asr.py --model $M --tag $T 2>&1 | grep -E "^(clean|cafe)"
  $PY transcribe_utt100.py --model $M --tag $T 2>&1 | grep -E "^(clean|noisy)"
  (cd ../core && ../ocr/.venv/Scripts/python - "$T" <<'EOF'
import json, sys
from eval_f06 import eval_text
t = sys.argv[1]
utts = json.load(open("data/utt100.json", encoding="utf-8"))
for c in ("clean", "noisy15", "noisy5"):
    h = {x["id"]: x["hyp"] for x in json.load(open(f"runs/asr_{t}_hw_{c}.json", encoding="utf-8"))}
    acc, _ = eval_text(utts, [h[u["id"]] for u in utts])
    print(f"  주문100 {c}: 의도 {acc * 100:.0f}%")
EOF
  )
done
