#!/usr/bin/env bash
set -euo pipefail
for ds in beauty sports toys movielens1m; do
  ck="outputs/${ds}/seed_42/best.pt"
  python src/evaluate.py --config "configs/isdsrec/${ds}.yaml" --checkpoint "$ck" --split test --benchmark > "outputs/${ds}/efficiency_seed42.json"
done
