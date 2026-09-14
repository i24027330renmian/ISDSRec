# ISDSRec Reproducibility Reference Implementation

This repository is a paper-derived reference implementation for **ISDSRec**. It is organized to mirror the manuscript's reproducibility workflow: preprocessing, cached GTE semantics, training/evaluation, ablation controls, stratified/sensitivity analysis, efficiency measurement, and statistical analysis.

## Scope and provenance

The implementation is grounded in the manuscript's stated methodology and experimental protocol. In particular, it follows these paper-specified choices:

- iterative 5-core filtering after exact `(user, item, timestamp)` deduplication;
- stable chronological ordering and leave-one-out split (last=test, penultimate=validation);
- all valid next-item prefixes from the training partition;
- maximum retained history length 50 and latent dimension 128;
- frozen `Alibaba-NLP/gte-Qwen2-7B-instruct` item encoder with cached embeddings;
- `Linear(3584,128) + LayerNorm(128)` semantic projection and dimension-wise gated ID/semantic fusion;
- continuous local-gap encoding with `Linear(1,64) -> GELU -> Linear(64,128)`;
- recency prior `r = exp(-gamma * normalized_log_recency)`;
- linear diffusion schedule, interaction-specific modulation, clipping at `beta_up=0.05`, one sampled diffusion step per training instance, epsilon prediction, and analytical recovery;
- two-layer Pre-LayerNorm bidirectional Transformer denoiser (d=128, 4 heads, FFN=512, dropout=0.1);
- recency-biased additive long-term attention, final recovered state as short-term intent, and a six-statistic adaptive scalar fusion gate;
- 100 uniformly sampled training negatives per instance;
- full-catalog validation/test ranking;
- five reporting seeds: `42, 123, 2024, 3407, 5678`;
- 20 independent Gaussian-noise evaluations for inference-noise configurations;
- Student-t run-level confidence intervals, paired user-level bootstrap with 5,000 resamples, Holm correction, and separate test-time noise variation.


## Environment

The manuscript reports Python 3.10, PyTorch 2.5.1, CUDA 12.4, and one NVIDIA RTX 4090 for the experiments. Other packages in `requirements.txt` are dependencies of this reference implementation; they should not be interpreted as the exact original environment unless independently verified.

## Data preparation

Raw data are not redistributed in this repository. See `data/README.md`.

Example Amazon pipeline:

```bash
python data/preprocessing/preprocess_amazon.py \
  --reviews /path/to/reviews_Beauty.json.gz \
  --metadata /path/to/meta_Beauty.json.gz \
  --out-dir data/processed/beauty

python data/preprocessing/build_splits.py \
  --interactions data/processed/beauty/interactions_raw.csv \
  --items data/processed/beauty/items_raw.csv \
  --out-dir data/processed/beauty

python data/preprocessing/build_training_prefixes.py \
  --split-file data/processed/beauty/interactions_split.csv \
  --out-dir data/processed/beauty

python data/preprocessing/compute_statistics.py \
  --train-instances data/processed/beauty/train_instances.jsonl \
  --split-file data/processed/beauty/interactions_split.csv \
  --out-dir data/processed/beauty \
  --gamma 0.8
```

MovieLens-1M:

```bash
python data/preprocessing/preprocess_movielens.py \
  --ratings /path/to/ratings.dat \
  --movies /path/to/movies.dat \
  --out-dir data/processed/movielens1m
```

Then run the same `build_splits.py`, `build_training_prefixes.py`, and `compute_statistics.py` steps.

## Cache GTE item embeddings

```bash
python scripts/cache_gte_embeddings.py \
  --items data/processed/beauty/items_mapped.csv \
  --output data/processed/beauty/gte_embeddings.npy
```

The default encoder is `Alibaba-NLP/gte-Qwen2-7B-instruct`. The script uses last-valid-token pooling and L2 normalization, matching the manuscript description.

## Train and evaluate

```bash
python src/train.py --config configs/isdsrec/beauty.yaml --seed 42
python src/evaluate.py --config configs/isdsrec/beauty.yaml \
  --checkpoint outputs/beauty/seed_42/best.pt --split test
```

Batch reproduction:

```bash
bash scripts/run_main_experiments.sh
bash scripts/run_ablations.sh
bash scripts/run_stratified_analysis.sh
bash scripts/run_sensitivity.sh
bash scripts/run_efficiency.sh
```

## Output conventions

Training writes checkpoints and JSON summaries beneath `outputs/`. Evaluation can write per-user metrics so that the bootstrap and stratified analyses are reproducible without recomputing model scores.

## Repository layout

```text
README.md
requirements.txt
IMPLEMENTATION_NOTES.md
configs/
  isdsrec/
  ablations/
data/
  README.md
  preprocessing/
src/
  model/
  train.py
  evaluate.py
  metrics.py
scripts/
statistics/
```
