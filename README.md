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

## Status

Project underway — see week-by-week timeline in the project plan. Repo committed weekly as provenance evidence.

## Repository structure

See project plan for the full folder layout (`src/`, `data/`, `configs/`, `results/`, `report/`, `docs/`).
