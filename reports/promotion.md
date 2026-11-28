## Champion vs challenger: PROMOTED

Both scored on the same fixed real hold-out (positive class `>50K`).

| Metric | Champion (v1) | Challenger (v2) | Δ |
|---|---|---|---|
| roc_auc | 0.8844 | 0.8783 | -0.0061 |
| f1 | 0.6222 | 0.6073 | -0.0149 |
| accuracy | 0.8305 | 0.8175 | -0.0130 |
| precision | 0.6718 | 0.6304 | -0.0414 |
| recall | 0.5795 | 0.5859 | +0.0064 |

Rule: challenger ROC-AUC ≥ champion − 0.01, and the challenger's generator passed the privacy gate (it is `passed`).

`champion` → v2; v1 kept as `previous` (rollback target).
