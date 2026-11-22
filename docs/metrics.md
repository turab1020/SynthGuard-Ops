# Metrics specification (Deliverable 1)

Notation: **R** = real train (22,775 rows), **H** = real hold-out (9,762 rows), **S** = a synthetic
dataset sampled from a generator fitted on R, with |S| = |H|. Positive class = `>50K`.
All thresholds and weights live in `params.yaml`; metric names below are the MLflow metric keys.

## 1. Fidelity (does S look like R?)
SDMetrics single-table `QualityReport` on (R, S) with the SDV metadata from `prepare`.
- `column_shapes` — mean over columns of: KSComplement (numeric) = 1 − KS statistic;
  TVComplement (categorical) = 1 − TVD, where TVD = ½ Σ_c |p_R(c) − p_S(c)|.
- `column_pair_trends` — mean over column pairs of CorrelationSimilarity (numeric pairs:
  1 − |ρ_R − ρ_S| / 2) or ContingencySimilarity (other pairs: 1 − TVD of the joint table).
- `quality_score` = mean(column_shapes, column_pair_trends). Range 0–1, higher is better.
- Diagnostics also saved: per-column KS / TVD, and `corr_diff` = mean |ρ_R − ρ_S| over numeric pairs (Pearson).

## 2. Utility (is S useful for training models?)
One preprocessing pipeline per training source: one-hot encode categoricals
(`handle_unknown="ignore"`), standard-scale numerics, **fitted on the training source only**.
Classifiers: Random Forest and XGBoost with the hyperparameters in `params.yaml`, `random_state=seed`.
- **TRTR** (reference): train on R, test on H.
- **TSTR**: train on S, test on the same H.
- Reported per model m ∈ {rf, xgb}: `tstr_m_precision`, `tstr_m_recall`, `tstr_m_f1`, `tstr_m_roc_auc`
  (and `trtr_m_*` on the reference run).
- `utility_ratio` = TSTR_RF ROC-AUC / TRTR_RF ROC-AUC. ≈ 1 means S is as useful as real data.

## 3. Privacy (does S leak R?) — empirical risk estimates, not guarantees
Shared encoding: one-hot categoricals + min-max scale numerics, **fitted on R**; Euclidean distance.
- **DCR**: for each row, distance to its nearest neighbour in R.
  `dcr_median_synth_train` = median DCR(S→R); reference = median DCR(H→R).
  `dcr_ratio` = median DCR(S→R) / median DCR(H→R). ≈ 1 good; ≪ 1 suggests memorisation.
- `exact_match_rate` = fraction of S rows identical (all 15 columns) to some row of R.
- **Distance-based MIA**: draw n = `mia_sample_size` members from R and n non-members from H
  (seeded). Attack score for a record x = −min distance from x to S (closer ⇒ "member").
  `mia_auc` = ROC-AUC of that score for member vs non-member (0.5 = attacker guesses).
  `mia_advantage` = max over thresholds of (TPR − FPR).

These numbers measure one specific attacker on one sample; a low value is evidence of lower risk,
**not** a formal privacy guarantee (unlike differential privacy).

## 4. Cost
`fit_seconds`, `sample_seconds` (wall clock), `peak_memory_mb` (process RSS peak during fit).

## 5. Selection (registering the top generator)
1. **Privacy gate** — pass iff `mia_auc ≤ 0.60` **and** `exact_match_rate ≤ 0.01`.
2. **Score** — `selection_score = 0.5 · quality_score + 0.5 · utility_ratio`.
3. **Pool** — the candidates that pass the gate (tag `privacy_gate=passed`). If none passes, the
   pool is every candidate and the registered model is tagged `privacy_gate=failed`.
4. **Near-ties** — candidates in the pool whose `selection_score` is within
   `selection.tie_tolerance` (0.02, inclusive) of the pool's best are treated as tied.
5. **Winner** — among the tied, the lowest `fit_seconds` (cheapest to train); then the higher
   `selection_score`, then the lower `mia_auc`.

Why a tolerance: retraining CTGAN/TVAE is bit-identical on one machine, but arm64 (macOS) and
x86_64 (Linux/Docker) CPUs produce slightly different networks. The measured drift in
`selection_score` was up to 0.0087, enough to swap two candidates that differ by 0.001. Scores
closer than the reproducibility tolerance (0.02) are not a meaningful difference, so the cheaper
model is preferred; that choice is the same on every platform.
