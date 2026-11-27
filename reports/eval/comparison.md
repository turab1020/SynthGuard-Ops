| generator | selection_score | privacy_gate | quality_score | column_shapes | column_pair_trends | utility_ratio | tstr_rf_roc_auc | tstr_xgb_roc_auc | mia_auc | mia_advantage | exact_match_rate | dcr_ratio | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tvae | 0.9058 | passed | 0.8434 | 0.8802 | 0.8067 | 0.9682 | 0.8802 | 0.8844 | 0.4914 | 0.0135 | 0.0000 | 0.6297 | 61.1700 |
| ctgan | 0.8984 | passed | 0.8380 | 0.8863 | 0.7897 | 0.9587 | 0.8716 | 0.8754 | 0.4967 | 0.0095 | 0.0000 | 12.9558 | 210.5210 |
| gaussian_copula | 0.8827 | passed | 0.8420 | 0.9157 | 0.7682 | 0.9234 | 0.8395 | 0.8277 | 0.4992 | 0.0135 | 0.0000 | 13.0881 | 4.1371 |
| marginal | 0.6841 | passed | 0.8787 | 0.9938 | 0.7637 | 0.4896 | 0.4451 | 0.5099 | 0.5024 | 0.0205 | 0.0000 | 13.1747 | 0.0108 |

Selected: **tvae** (privacy gate passed).
Tied within 0.02 of the top score: ctgan, tvae; the fastest to train among them wins.
