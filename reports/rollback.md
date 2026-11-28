## Rollback

| | Served now | After rollback |
|---|---|---|
| classifier version | v2 | v1 |
| trained on | synthetic:tvae | synthetic:tvae |
| hold-out ROC-AUC (benchmark) | 0.8783 | 0.8844 |

Restored `deploy/model/` and `metrics/classifier.json` from `2c1c0b5` (the state before `2379240`).
Merging this deploys it through deploy.yml (tests + eval gate, then Render).
