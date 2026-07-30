# Literature Review

_Last updated: Week 1 (fresh re-search conducted July 2026)_

## 1. Foundation papers (original 10-paper review)

| # | Topic | Algorithm | Dataset | Key Contribution | Limitation / Gap |
|---|---|---|---|---|---|
| 1 | Attention-based RNN with information-gain sensor placement | BiLSTM + Attention | UCI Opportunity Challenge | Sensor-selection + attention reweighting | Dataset-specific, no cross-dataset test, no smartphone-only setup |
| 2 | Attention-based Residual BiLSTM | Residual BiLSTM + Attention | UCI-HAR, WISDM, KU-HAR | Fixes confusion between similar activities (walking vs. walking-upstairs) | No cross-subject/cross-dataset transfer study |
| 3 | GRU-based Hierarchical Attention | GRU + Hierarchical Attention | Nurse-activity dataset | Handles class imbalance | Cross-subject accuracy much lower (~66%); small, non-public dataset |
| 4 | Lightweight Attention-GRU for WiFi CSI HAR | GRU + Self-Attention + pruning | WiFi CSI (ARIL etc.) | 252M → 0.058M params, ~98.9% accuracy | Only RF/CSI signals, not IMU; unclear if pruning transfers |
| 5 | GRU with Dual Attention (channel + temporal) | BiGRU + Dual Attention | Custom multi-sensor dataset | Splits "where to look" (channel) from "when to look" (temporal) | Not tested on UCI-HAR/WISDM/PAMAP2; no ablation isolating each attention type |
| 6 | Transformer for HAR (no RNN/CNN) | Vanilla Transformer Encoder | Smartphone accel+gyro | Attention alone matches CNN+LSTM | Generic positional encoding, no data-efficiency analysis |
| 7 | CNN-LSTM with Self-Attention | CNN + LSTM + Self-Attention | H-Activity, MHEALTH, UCI-HAR | 99.93% / 98.76% / 93.11% | Gap between own data (99.9%) and UCI-HAR (93%) suggests overfitting |
| 8 | Multi-branch CNN-BiLSTM | CNN + BiLSTM | Multiple public datasets | Strong cited baseline | Reference-only; CNN-containing "gold standard" |
| 9 | CNN-BiLSTM Parallel + Attention (noisy data) | CNN + BiLSTM + Attention | UCI-HAR, WISDM | Robust to noisy/missing sensor data | Unclear if pure RNN+Attention can match this without CNN |
| 10 | Multi-Agent Attention-CNN-BiLSTM (ensemble) | CNN + BiLSTM + Attention + ML | Disability-assistance dataset | Ensemble accuracy gains | Deployment complexity, no latency/resource analysis |

**Pattern:** the strongest recent baselines (rows 7–10) are CNN+RNN hybrids; pure RNN+Attention papers (rows 1, 2, 3, 5, 6) are fewer, mostly single-dataset, and rarely report cross-dataset generalization.

## 2. Newly identified papers (found on re-search, July 2026) — critical addition

These directly overlap with the project's originally-claimed novelty and must be explicitly differentiated from in the final report.

| Paper | Venue / Date | Architecture | Datasets | Result |
|---|---|---|---|---|
| Dual-INABA (Dual Inception-Attention-BiGRU-Attention) | Scientific Reports, Dec 2025 | Inception + Attention + BiGRU + Attention, wavelet-transform preprocessing | PAMAP2, UCI-HAR, WISDM + self-collected WSMC-HAR | 98.15% / 98.89% / 96.78% accuracy; ablations isolate BiGRU vs. Inception contribution |
| TGA-HAR | Sensors (MDPI), Sept 2025 | TCN + BiGRU + Attention | WISDM, USC-HAD, PAMAP2, Real-World IDLab | Robust generalization across sensor configurations |
| MHAT-Net | MDPI, April 2026 | Parallel BiGRU + Transformer dual-pathway | Smartphone inertial (motionless activities) | Targets weak-signature stationary activities |
| CNN-BiGRU-Transformer | Applied Sciences, Nov 2025 | Multi-branch CNN + BiGRU + Transformer | Wearable sensor data | Improved long-term temporal dependency capture |
| AttnFusionGRU | ScienceDirect, Sept 2025 | Dual-branch attention + SqueezeNet + BiGRU | UCF101, Human3.6M | Spatial + temporal dual attention (vision-based, not IMU) |

Cross-dataset transfer (originally proposed as the "key novelty result") is also a mature, active subfield well beyond simple fine-tuning: Deep CORAL domain adaptation, self-supervised cross-domain pretraining (UniMotion, CrossHAR), and data-mixture optimization (HAR-DoReMi, ~6.5% average accuracy improvement over prior state-of-the-art). Naive train-then-fine-tune is no longer a novel contribution in this space.

## 3. Explicit differentiation (required per Minimum Bar item #4)

- **vs. Dual-INABA:** shares the dual-attention + BiGRU idea but adds an Inception branch (not CNN-free) and reports accuracy only — no params/FLOPs/latency accounting, no LOSO cross-validation, no significance testing.
- **vs. TGA-HAR:** uses a TCN, not pure RNN+Attention; robustness claim is about sensor configuration generalization, not efficiency-vs-accuracy tradeoff reporting.
- **vs. MHAT-Net:** targets a narrower activity subset (stationary/weak-signature) and uses a Transformer pathway, not evaluated on the standard UCI-HAR/WISDM/PAMAP2 triad with LOSO rigor.
- **vs. CNN-BiGRU-Transformer:** explicitly CNN-containing; this project's headline question (can a CNN-free model match it at lower cost) is not addressed by this paper itself.
- **vs. AttnFusionGRU:** vision-based (video), not IMU/wearable sensor data — different modality entirely, included here only because "dual attention + GRU" as a phrase overlaps.

## 4. See also

`docs/research_gap_notes.md` for the full gap table and final novelty statement.
