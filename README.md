# Dual-Attention BiGRU for Human Activity Recognition

A CNN-free, pure recurrent + attention architecture for HAR from wearable/inertial sensor data, evaluated with an emphasis on **efficiency accounting** and **evaluation rigor** — two dimensions largely missing from recent dual-attention HAR literature.

## Novelty

Recent HAR papers (Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer) report accuracy-only comparisons on single train/test splits. No reviewed CNN-free BiGRU + dual-attention model reports parameter/FLOPs/latency accounting alongside subject-independent, multi-seed, statistically validated evaluation across UCI-HAR, WISDM, and PAMAP2. This project fills that combined gap.

See `docs/research_gap_notes.md` for the full gap analysis and `report/literature_review.md` for the complete literature review.

## Models compared (4 total)

| # | Model | Role |
|---|---|---|
| 1 | Plain BiGRU | Baseline 1 |
| 2 | BiLSTM + single attention | Baseline 2 (reproduces prior literature) |
| 3 | CNN + BiGRU | Reference "heavy" baseline for the efficiency claim |
| 4 | Dual-Attention BiGRU (channel + temporal) | Proposed model |

## Datasets

UCI-HAR, WISDM, PAMAP2 — evaluated independently in native mode (own sampling rate/window/channels); optional harmonized mode for a supplementary cross-dataset transfer experiment (see `docs/research_gap_notes.md` §3).

## Evaluation

- Official train/test split (comparability with literature) **and** subject-independent (LOSO) k-fold (headline result).
- ≥3 seeds per fold, mean ± std reporting.
- Params, FLOPs, and inference latency logged for every model.
- Wilcoxon signed-rank / paired t-test significance testing vs. each baseline.

## Running the full pipeline

```bash
bash scripts/run_all.sh                  # venv -> download -> preprocess -> smoke -> train -> profile -> stats -> evaluate
PARALLEL=1 bash scripts/run_all.sh       # same, training the 3 datasets concurrently
```

The script creates `.venv` and installs `requirements.txt` on first run (and again whenever it
changes). On a machine with an NVIDIA GPU it installs the CUDA build of PyTorch and aborts if CUDA
isn't usable rather than silently training on the CPU. On Windows, run it from **Git Bash**.

`scripts/run_all.sh` is safe to re-run: downloads are skipped if present, and every finished
training run is skipped (resume is keyed on model/variant/dataset/split/fold/seed in
`results/tables/raw_results.csv`). Run only some steps with e.g. `STEPS="train" DATASETS="pamap2"`.

| Step | Command | Output |
|---|---|---|
| Download | `python data/download_data.py --dataset all` | `data/raw/` |
| Preprocess | `python -m src.preprocessing --dataset all` | `data/processed/<ds>/{X,y,subject}.npy`, `meta.json` |
| Train | `python -m src.train --config configs/<ds>_config.yaml --model <model> [--dual_attn_variant all]` | `results/tables/raw_results.csv`, `results/preds/*.npz`, `checkpoints/` |
| Efficiency | `python -m src.profile_efficiency` | `results/tables/efficiency.csv` (params, FLOPs, CPU/GPU latency) |
| Significance | `python -m src.stats_test` | `results/tables/significance.csv` |
| Tables & figures | `python -m src.evaluate` | `results/tables/master_comparison.{csv,md}`, `results/tables/confusion/`, `results/figures/` |

Protocol details:
- Early stopping monitors a **subject-wise** validation set (10% of each fold's training subjects), never test subjects.
- Labels use one dataset-wide mapping; normalization is fit on each fold's training subjects only.
- Latency is measured in one controlled pass (`profile_efficiency.py`, 1 CPU thread by default), not during training.
- PAMAP2: acc + gyro of all 3 IMUs (18 channels), downsampled 100 → 50 Hz, subject 109 excluded (8-fold LOSO).
- The notebooks are for exploration; use `src.train` for anything that goes into the results table.

Results produced before the 2026-09-22 pipeline fixes are archived in `results/archive/2026-09-22_pre_fix/` (see its README).

## Status

Project underway — see week-by-week timeline in the project plan. Repo committed weekly as provenance evidence.

## Repository structure

See project plan for the full folder layout (`src/`, `data/`, `configs/`, `results/`, `report/`, `docs/`).
