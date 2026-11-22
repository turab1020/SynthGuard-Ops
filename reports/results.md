# Results: generator comparison (Deliverable 1)

Source: `metrics/evaluation.json`, produced by the Docker fresh-clone reference run on x86_64 Linux. Real-data reference (TRTR, train on real, test on hold-out): Random Forest ROC-AUC 0.9091, XGBoost ROC-AUC 0.9335.

| Generator | Selection score | Privacy gate | Quality | Utility ratio | TSTR RF ROC-AUC | TSTR XGB ROC-AUC | MIA AUC | DCR ratio | Exact match | Fit time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **TVAE (registered)** | 0.9080 | passed | 0.8441 | 0.9719 | 0.8836 | 0.8854 | 0.4983 | 0.6267 | 0.0000 | 73.50 |
| CTGAN | 0.9001 | passed | 0.8457 | 0.9545 | 0.8677 | 0.8757 | 0.4907 | 12.9606 | 0.0000 | 342.92 |
| Gaussian Copula (baseline) | 0.8827 | passed | 0.8420 | 0.9234 | 0.8395 | 0.8277 | 0.4992 | 13.0881 | 0.0000 | 4.77 |
| Independent marginal (baseline) | 0.6841 | passed | 0.8787 | 0.4896 | 0.4451 | 0.5099 | 0.5024 | 13.1747 | 0.0000 | 0.01 |

## Why TVAE was registered

1. **Privacy gate:** all four candidates pass (MIA AUC ≤ 0.60 and exact-match rate ≤ 0.01).
2. **Score:** `0.5 × quality + 0.5 × utility_ratio`. TVAE is highest at 0.9080, then CTGAN at 0.9001.
3. **Tie rule:** scores within 0.02 of the best count as tied. TVAE and CTGAN are tied, and TVAE trains faster (73.5 s vs 342.9 s). Gaussian Copula is 0.0253 below the best, so it is not tied.

## Reading the results

- **Quality alone would mislead:** the marginal sampler has the highest quality score (0.8787) because it copies each column's distribution, but it loses the relationships between columns, so its utility ratio is only 0.4896.
- **TVAE's DCR ratio is 0.6267, below 1.** Its rows sit closer to training rows than real hold-out rows do, which is a mild memorisation signal. Its exact-match rate is 0.0000 and its MIA AUC is 0.4983 (about chance), so neither test shows a leak. DCR is not part of the D1 gate.
- **Privacy numbers are empirical risk estimates** from one attacker on one sample. They are not formal privacy guarantees.
- **Scores vary slightly by CPU type** (within ±0.02). See the reproducibility section of the README.
