#!/usr/bin/env bash
set -euo pipefail
SEEDS=(42 123 2024 3407 5678)
DATASETS=(beauty sports toys movielens1m)
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python src/train.py --config "configs/isdsrec/${ds}.yaml" --seed "$seed"
  done
done
