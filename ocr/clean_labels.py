"""반사광 등으로 글자가 사라진 줄 이미지를 학습 목록에서 뺀다 (라벨 잡음 제거).
  python clean_labels.py data/rec_train/labels.txt data/rec_train/labels_clean.txt
"""
import sys

import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
keep, drop = [], 0
for line in open(src, encoding="utf-8").read().splitlines():
    p, t = line.split("\t", 1)
    g = cv2.imdecode(np.fromfile("data/" + p, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None or g.std() < 8 or len(t) > 25:
        drop += 1
        continue
    keep.append(line)
open(dst, "w", encoding="utf-8").write("\n".join(keep) + "\n")
print(f"keep {len(keep)}  drop {drop}")
