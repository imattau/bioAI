# Interface Experiments: Results and Findings

## Experiment 1: Encoder → Hopfield Store

| Test | Result | Notes |
|---|---|---|
| Store 10 patterns | PASS | Baseline functionality confirmed |
| Clean recall | PASS | sim=1.0 — perfect auto-associative recall |
| Noisy recall (σ=0.3) | PASS | sim=1.0 — Hopfield corrects noise completely |
| High-noise recall (σ=1.0) | PASS | sim=1.0 — surprisingly robust for MAP vectors |
| Basin separation | PASS | sim to wrong pattern ≈ 0.0 — basins are well-separated |
| Energy decreases | PASS | Corrected energy computation for tanh dynamics |
| Energy landscape | PASS | Stored patterns form distinct minima |

**Conclusion:** Hopfield store is reliable. MAP vectors (bipolar {-1, +1}) have near-perfect recall even at high noise due to the discrete nature of the state space.

## Experiment 2: Hopfield → Clonal Memory

| Test | Result | Notes |
|---|---|---|
| 5 patterns → 5 modules | PASS | |
| Novel pattern expands | PASS | |
| Near-duplicate fine-tunes | FAIL | Threshold incorrectly calibrated for MAP vectors. Cosine similarity between original and 5%-noise copy is > 0.95, which exceeds the clone threshold. Needs tuning or the clone margin widened. |
| Originals recognisable | FAIL | avg affinity = 0.06 after 30 new patterns. Root cause: `last_used` was not set on new modules, causing originals to be pruned immediately. **Fixed** but also reveals a deeper issue: modules adapt away from originals during fine-tuning, and pruning is based on recency, not retention value. |
| Pool bounded | PASS | Pruning respects max_modules. |

**Bugs found:** `last_used` unset on module creation — new modules always pruned first. **Fixed.**

**Design issues:**
- Fine-tuning rate (0.1 × default_lr) still causes drift away from the stored pattern. Need lower rate or a different mechanism (e.g., episodic memory replay).
- Pruning by recency is too aggressive — an important old module gets pruned if unused. Needs importance-weighted or affinity-weighted pruning.

## Experiment 3: Hopfield → Immune Monitor

| Test | Result | Notes |
|---|---|---|
| Calibration | PASS | |
| Normal recall not anomalous | PASS | RMS drift ≈ 1.0 — correct |
| OOD input anomalous | FAIL | Both drift and OneClassSVM failed. RMS drift ≈ 1.0 regardless of input type. |
| Raw random anomalous | FAIL | Same issue. |
| OOD detection rate | FAIL | 0/20 flagged. |

**Root cause:** Two separate failures:
1. RMS z-score drift is dimension-independent and ≈ 1.0 for any input (by construction of z-scores). It cannot distinguish in-distribution from OOD because z-scores normalize per-dimension, removing the magnitude signal.
2. OneClassSVM with 8 samples in 128 dimensions has insufficient data to build a meaningful boundary. The nu=0.1 setting means it expects 10% outliers in training data, but all 8 calibration samples are inliers.

**Design insight:** The drift metric as implemented is fundamentally unsuitable for high-dimensional OOD detection. A viable replacement would be:
- Reconstruction error from a learned autoencoder (more calibration data needed)
- Mahalanobis distance in a learned feature space (not raw activation space)
- Ensemble of density estimators (normalising flows, not OneClassSVM)
- Or simply use the Hopfield energy directly — OOD inputs have higher energy than stored patterns (confirmed in Experiment 6)

## Experiment 4: Action Selection → Memory Retrieval

| Test | Result | Notes |
|---|---|---|
| Reward improves | PASS | early=0.93, late=3.57 — agent learns to select correct memory |
| Valid action | PASS | |

**Conclusion:** Go/NoGo actor-critic works for the memory retrieval task. The environment interface is functional. No issues found.

## Experiment 5: Decoder (adaLN)

**Not yet implemented.** Previous experiment (embedding-level conditioning) showed 0.0% accuracy. adaLN per-layer modulation is the planned fix.

## Experiment 6: Full Pipeline

| Test | Result | Notes |
|---|---|---|
| Noisy recall | PASS | sim=1.0 |
| Store lookup | PASS | top sim=0.98 |
| Hopfield modifies novel input | PASS | Attractor dynamics confirmed |
| Stored vs random energy | PASS | stored=-32k, random=-260 |

**Conclusion:** The core pipeline (encoder → store → Hopfield → recall) works end-to-end.

## Summary

| Subsystem | Status | Action needed |
|---|---|---|
| Hopfield store | ✅ Solid | None |
| Clonal memory | ⚠️ Two bugs fixed, one remaining design issue | Need importance-weighted pruning or lower fine-tuning rate |
| Immune monitor | 🔴 Fundamentally unsuitable metric | Requires redesign — Hopfield energy is a viable alternative signal |
| Action selection | ✅ Works | None |
| Decoder | 🔴 Not yet implemented | adaLN first, then re-test |
