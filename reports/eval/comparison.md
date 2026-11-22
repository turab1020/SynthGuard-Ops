| generator | selection_score | privacy_gate | quality_score | column_shapes | column_pair_trends | utility_ratio | tstr_rf_roc_auc | tstr_xgb_roc_auc | mia_auc | mia_advantage | exact_match_rate | dcr_ratio | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tvae | 0.9080 | passed | 0.8441 | 0.8740 | 0.8142 | 0.9719 | 0.8836 | 0.8854 | 0.4983 | 0.0115 | 0.0000 | 0.6267 | 73.4998 |
| ctgan | 0.9001 | passed | 0.8457 | 0.8935 | 0.7980 | 0.9545 | 0.8677 | 0.8757 | 0.4907 | 0.0050 | 0.0000 | 12.9606 | 342.9162 |
| gaussian_copula | 0.8827 | passed | 0.8420 | 0.9157 | 0.7682 | 0.9234 | 0.8395 | 0.8277 | 0.4992 | 0.0135 | 0.0000 | 13.0881 | 4.7745 |
| marginal | 0.6841 | passed | 0.8787 | 0.9938 | 0.7637 | 0.4896 | 0.4451 | 0.5099 | 0.5024 | 0.0205 | 0.0000 | 13.1747 | 0.0112 |

Selected: **tvae** (privacy gate passed).
Tied within 0.02 of the top score: ctgan, tvae; the fastest to train among them wins.
