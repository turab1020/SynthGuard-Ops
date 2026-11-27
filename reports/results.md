# Results: generator comparison

Source: `metrics/evaluation.json`, produced by the Docker fresh-clone reference run on x86_64 Linux. Real-data reference (TRTR, train on real, test on hold-out): Random Forest ROC-AUC 0.9091, XGBoost ROC-AUC 0.9335.

| Generator | Selection score | Privacy gate | Quality | Utility ratio | TSTR RF ROC-AUC | TSTR XGB ROC-AUC | MIA AUC | DCR ratio | Exact match | Fit time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **TVAE (registered)** | 0.9058 | passed | 0.8434 | 0.9682 | 0.8802 | 0.8844 | 0.4914 | 0.6297 | 0.0000 | 61.17 |
| CTGAN | 0.8984 | passed | 0.8380 | 0.9587 | 0.8716 | 0.8754 | 0.4967 | 12.9558 | 0.0000 | 210.52 |
| Gaussian Copula (baseline) | 0.8827 | passed | 0.8420 | 0.9234 | 0.8395 | 0.8277 | 0.4992 | 13.0881 | 0.0000 | 4.14 |
| Independent marginal (baseline) | 0.6841 | passed | 0.8787 | 0.4896 | 0.4451 | 0.5099 | 0.5024 | 13.1747 | 0.0000 | 0.01 |

## Why TVAE was registered

1. **Privacy gate:** all four candidates pass (MIA AUC ≤ 0.60 and exact-match rate ≤ 0.01).
2. **Score:** `0.5 × quality + 0.5 × utility_ratio`. TVAE is highest at 0.9058, then CTGAN at 0.8984.
3. **Tie rule:** scores within 0.02 of the best count as tied. TVAE and CTGAN are tied, and TVAE trains faster (61.2 s vs 210.5 s). Gaussian Copula is 0.0231 below the best, so it is not tied.

## Reading the results

- **Quality alone would mislead:** the marginal sampler has the highest quality score (0.8787) because it copies each column's distribution, but it loses the relationships between columns, so its utility ratio is only 0.4896.
- **TVAE's DCR ratio is 0.6297, below 1.** Its rows sit closer to training rows than real hold-out rows do, which is a mild memorisation signal. Its exact-match rate is 0.0000 and its MIA AUC is 0.4914 (about chance), so neither test shows a leak. DCR is not part of the gate.
- **Privacy numbers are empirical risk estimates** from one attacker on one sample. They are not formal privacy guarantees.
- **Scores vary slightly by CPU type** (within ±0.02). See the reproducibility section of the README.
