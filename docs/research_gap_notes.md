# Research Gap Notes

_Last updated: Week 1_

## 1. Gaps identified in the literature

| Gap | Status in literature | Still open? |
|---|---|---|
| Dual attention (channel + temporal) + BiGRU backbone | Closed — Dual-INABA (Dec 2025) does this on the exact 3 benchmark datasets | No |
| Cross-dataset transfer with fine-tuning as sole novelty | Closed / surpassed — Deep CORAL, CrossHAR, HAR-DoReMi are far more sophisticated | No |
| CNN-free (pure RNN+Attention) model compared against CNN-hybrids on params, FLOPs, and inference latency, not just accuracy | Not directly reported by Dual-INABA, TGA-HAR, or CNN-BiGRU-Transformer — these report accuracy only | **Yes** |
| Evaluation using subject-independent (leave-subjects-out) cross-validation, multiple seeds, and statistical significance testing across all 3 benchmark datasets | Most reviewed papers (including new ones) report single random-split accuracy; no paired significance testing found | **Yes** |

**Conclusion:** the project's novelty is repositioned from architecture novelty ("we combined X and Y first") to methodological novelty ("we are the first to report this specific efficiency-and-rigor combination for a CNN-free dual-attention BiGRU on these three benchmarks"). This is a defensible, currently-unclaimed gap.

## 2. Final novelty statement

Recent HAR literature (Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer) reports accuracy-only comparisons on single train/test splits. No reviewed CNN-free BiGRU + dual-attention model reports parameter/FLOPs/latency accounting alongside subject-independent, multi-seed, statistically validated evaluation across UCI-HAR, WISDM, and PAMAP2 — this project addresses that combined efficiency-and-rigor gap.

**The two novelty pillars:**

1. **Efficiency accounting** — Dual-Attention BiGRU vs. a CNN+BiGRU reference baseline and other baselines, compared on accuracy, params, FLOPs, and inference latency, all reimplemented under identical conditions (not copied from literature tables).
2. **Evaluation rigor** — Subject-independent (LOSO) cross-validation, ≥3 seeds per fold, mean±std reporting, and paired statistical significance testing (Wilcoxon signed-rank) — instead of a single random-split accuracy number.

## 2.1 Architectural novelty layered on top of the methodological gap

Section 2's conclusion repositioned novelty from architecture to methodology
because plain "channel + temporal dual attention + BiGRU" is a closed gap
(Dual-INABA, Dec 2025). To avoid the proposed model being a pure
reimplementation of already-published architecture, `dual_attention_bigru.py`
adds one new, well-defined mechanism on top of that closed design:

**Spectral-Gated Channel Attention.** Every dual-attention HAR paper found in
the review (Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer,
AttnFusionGRU) computes channel attention purely from learned time-domain
statistics of the window. None inject a frequency-domain prior. This project's
`SpectralChannelGate` (`src/attention_modules.py`) computes, per sensor
channel and per window, (a) the fraction of spectral power inside the human
gait/limb-swing band (0.3-3.5 Hz, via a single `torch.fft.rfft` — no CNN, no
extra learned transform) and (b) normalized spectral entropy, then maps that
2-value descriptor through a tiny MLP to a per-channel gate that multiplies
the learned channel-attention weights. This lets the model up-weight channels
carrying real periodic locomotion signal and down-weight near-DC/static
channels — a distinction a purely time-domain learned attention has no direct
way to make from a single window.

This is implemented as an ablatable, switchable component
(`use_spectral_gate` in `DualAttentionBiGRU`, exposed via
`--dual_attn_variant` in `train.py`: `full` = proposed model, `plain` =
learned-channel-attention-only ablation arm equivalent to the closed
Dual-INABA-style baseline, `channel_only` / `temporal_only` isolate each
attention axis). The `full` vs. `plain` delta is this project's headline
architectural result, reported alongside the efficiency-and-rigor pillars
below.

## 3. Data harmonization decision (decided Week 1, not deferred to Week 3)

**Core evaluation (native mode) does not require cross-dataset harmonization.** The 4×3 comparison table and LOSO results are computed per-dataset, using each dataset's own optimal sampling rate, window size, and channel set. This is the submission-critical path and carries no harmonization risk.

**Harmonization is required only for the optional Week-10 bonus cross-dataset transfer experiment**, and is scoped as follows:

- **Sampling rate: 20 Hz** (the lowest of the three, so all datasets are downsampled, never upsampled — avoids manufacturing fake precision via interpolation and avoids aliasing artifacts). UCI-HAR (50→20Hz) and PAMAP2 (100→20Hz) are downsampled; WISDM (~20Hz) stays native.
- **Window size: 2.56s (~51 samples at 20Hz), 50% overlap** — anchored on UCI-HAR's established window *duration*, not its sample count, since durations transfer across sampling rates and sample counts don't.
- **Channel set: triaxial accelerometer only (x, y, z)** — the true lowest common denominator across all three datasets. (Check which WISDM release is downloaded — a newer version includes gyroscope; if used, accel-only remains the primary harmonized set with accel+gyro offered as a secondary richer variant. Not a blocker either way.)
- **Units/normalization:** normalize to g (or m/s²) explicitly; z-score normalize per-channel using **training-fold statistics only** (never test-fold, to avoid leakage) — this rule applies in native mode too.

**Sensor placement mismatch is a limitation, not a solved problem.** UCI-HAR is waist-worn, WISDM is pocket/wrist-worn, PAMAP2 has three locations (chest/hand/ankle). No amount of resampling makes a waist accelerometer signal physically equivalent to a chest one. **Decision: report this mismatch explicitly as a documented limitation/confound** of the (optional) cross-dataset result, rather than attempting a fix that quietly papers over a real physical difference. This is the more defensible choice for a paper and costs no engineering time.

## 4. Expected outcomes (from project plan, restated here for tracking)

1. Complete, reproducible 4-model × 3-dataset comparison table (accuracy, macro-F1, params, FLOPs, latency), no placeholder cells.
2. Subject-independent (LOSO) cross-validation results, mean ± std over ≥3 seeds, every model on every dataset.
3. At least one statistical significance test (Wilcoxon or paired t-test) with reported p-values, Dual-Attention BiGRU vs. each baseline.
4. Evidence (or disproof) that a CNN-free dual-attention BiGRU can match/approach CNN-hybrid accuracy at meaningfully lower params/FLOPs/latency — the headline result either way.
5. Related-work section explicitly naming and differentiating from Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer, and AttnFusionGRU.
6. Public GitHub repo with weekly commit history matching the timeline, as provenance evidence.
7. (Stretch) supplementary cross-dataset transfer result, framed as exploratory, not core.
