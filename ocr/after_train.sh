#!/usr/bin/env bash
# 학습 끝난 인식 모델을 export 하고 두 평가셋(학습 어휘 / 처음 보는 어휘)에서 기준선과 비교한다.
#   bash after_train.sh kiosk_rec_v1
set -e
cd "$(dirname "$0")"
RUN=${1:-kiosk_rec_v1}
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
(cd PaddleOCR && $PY tools/export_model.py -c ../configs/$RUN.yml \
    -o Global.pretrained_model=../output/$RUN/best_accuracy Global.save_inference_dir=../models/$RUN) 2>&1 | tail -2
for D in synth_val synth_unseen; do
  .venv/Scripts/python eval_m3.py --data data/$D --rec-dir models/$RUN --tag _$RUN 2>&1 | sed -n '/== /,$p' | grep -vE "@lv|uncertain"
done
