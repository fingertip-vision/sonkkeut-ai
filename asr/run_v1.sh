#!/usr/bin/env bash
# 도구의 백그라운드 시간 제한을 피하려고 별도 프로세스로 띄운다: 이어서 학습 → 변환·평가
cd /d/sonkkeutgil/asr
export PYTHONIOENCODING=utf-8
.venv/Scripts/python train_whisper.py --base openai/whisper-small --out out/whisper-elder-v1 --resume last >> train_v1.log 2>&1
echo "TRAIN_EXIT $?" >> train_v1.log
bash after_train.sh whisper-elder-v1 > after_v1.log 2>&1
echo "ALL_DONE" >> after_v1.log
