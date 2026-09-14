#!/usr/bin/env bash
set -euo pipefail
SEEDS=(42 123 2024 3407 5678)
DATASETS=(beauty sports toys movielens1m)
SUPPORTED=(noise_matched_uniform shuffled_recency reversed_recency clean_input_transformer schedule_only denoiser_only attention_only)
UNRESOLVED=(position_schedule heteroscedastic_dae temporal_gte_sasrec)

for ds in "${DATASETS[@]}"; do
  for ab in "${SUPPORTED[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python src/train.py --config "configs/isdsrec/${ds}.yaml" \
        --ablation "configs/ablations/${ab}.yaml" --seed "$seed" \
        --output-dir "outputs/${ds}/ablations/${ab}/seed_${seed}"
    done
  done
done

echo "Skipped unresolved exact-reproduction controls: ${UNRESOLVED[*]}"
echo "See IMPLEMENTATION_NOTES.md. Fill the original experiment details before running them."
