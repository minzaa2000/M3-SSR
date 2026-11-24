#!/bin/bash
# 一键跑四个数据集的某个模型： bash run_all.sh sasrec 42
MODEL=${1:-sasrec}
SEED=${2:-42}
for DS in Hvideo_E Hvideo_V Hamazon_M Hamazon_B; do
  echo "===== $DS / $MODEL / seed=$SEED ====="
  python main.py --dataset $DS --model $MODEL --seed $SEED --epochs 50
done
