#!/usr/bin/env bash
set -euo pipefail
SEEDS=(42 123 2024 3407 5678)
DATASETS=(beauty sports toys movielens1m)
declare -A GRIDS
GRIDS[gamma]="0 0.2 0.4 0.6 0.8 1.0"
GRIDS[lambda_noise]="0 0.1 0.3 0.5 0.7 1.0"
GRIDS[lambda_diff]="0 0.05 0.1 0.2 0.4 0.6 0.8"
# Post-selection sensitivity grid from Figure 4 / Table S6:
GRIDS[t_inf]="0 10 20 40 60 80 100"
for ds in "${DATASETS[@]}"; do
  for param in gamma lambda_noise lambda_diff t_inf; do
    for value in ${GRIDS[$param]}; do
      for seed in "${SEEDS[@]}"; do
        out="outputs/${ds}/sensitivity/${param}_${value}/seed_${seed}"
        # t_inf is inference-only after a selected trained checkpoint in the paper; do not retrain it.
        if [[ "$param" == "t_inf" ]]; then
          ck="outputs/${ds}/seed_${seed}/best.pt"
          python src/evaluate.py --config "configs/isdsrec/${ds}.yaml" --checkpoint "$ck" --split test --set "model.t_inf=${value}" --output "${out}/test_per_user.jsonl"
        else
          python src/train.py --config "configs/isdsrec/${ds}.yaml" --seed "$seed" --set "model.${param}=${value}" --output-dir "$out"
          python src/evaluate.py --config "configs/isdsrec/${ds}.yaml" --checkpoint "${out}/best.pt" --split test --set "model.${param}=${value}" --output "${out}/test_per_user.jsonl"
        fi
      done
    done
  done
done
