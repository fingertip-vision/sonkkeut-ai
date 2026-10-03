#!/usr/bin/env bash
# v2 export → 광주 실사진 시험(공식/v1/v2) + 합성 평가셋 회귀 확인
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
(cd PaddleOCR && ../.venv/Scripts/python tools/export_model.py -c ../configs/kiosk_rec_v2.yml \
    -o Global.pretrained_model=../output/kiosk_rec_v2/best_accuracy Global.save_inference_dir=../models/kiosk_rec_v2) 2>&1 | tail -1
.venv/Scripts/python real_menu_eval.py --models base,models/kiosk_rec_v1,models/kiosk_rec_v2 2>&1 | grep -E "줄 정답률|사전에 없는|crops"
for D in synth_val synth_unseen lay_list_hard lay_cart_hard; do
  .venv/Scripts/python eval_m3.py --data data/$D --rec-dir models/kiosk_rec_v2 --tag _v2 2>&1 \
    | sed -n '/== 매장/,$p' | grep -E "^  (menu|tab|button|cart_item) " | tr -s ' ' | tr '\n' ' '; echo " <- $D v2"
done
