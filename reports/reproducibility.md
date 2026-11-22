# Reproducibility check (Deliverable 1)

**Result: PASS.** A fresh clone running `docker compose run --rm pipeline` registered the same model (TVAE) and reproduced all four committed selection scores to 6 decimals.

## Fresh-clone check

GitHub Actions run `45016077578`, commit `b77df6c`, clean checkout, x86_64 Linux, 13 min 0 s. Tolerance: ±0.02 on selection scores.

| Generator | Committed score | Fresh-clone score | Difference | Privacy gate |
|---|---|---|---|---|
| TVAE (registered) | 0.908011 | 0.908011 | 0.000000 | passed |
| CTGAN | 0.900101 | 0.900101 | 0.000000 | passed |
| Gaussian Copula | 0.882685 | 0.882685 | 0.000000 | passed |
| Independent marginal | 0.684147 | 0.684147 | 0.000000 | passed |

Run IDs, timestamps and fit times change on every run and are not compared.

## Across CPU types

Repeated runs on one CPU type are identical. Different CPU types use different low-level numeric routines, so scores are **not bit-identical across CPU models**. They stayed within the ±0.02 tolerance:

| Environment | CTGAN | TVAE | Largest difference from committed |
|---|---|---|---|
| AMD EPYC 9V45 / 9V74 | 0.897160 | 0.905247 | 0.0029 |
| Intel Xeon Platinum 8573C | 0.900956 | 0.905794 | 0.0022 |
| macOS arm64 (native) | 0.905879 | 0.904732 | 0.0058 |
| Intel Xeon 8573C, AVX2 forced (diagnostic) | 0.903353 | 0.895534 | 0.0125 |

CTGAN and TVAE are closer than this variation, so `selection.tie_tolerance: 0.02` treats them as tied and selects the faster to train. That gives TVAE in every environment above.

## Limits

- The rows in the second table retrain and score CTGAN and TVAE only; the macOS row also comes from a full pipeline run.
- Gaussian Copula is 0.0253 below the committed top score. If a machine's top score fell below about 0.9027, Gaussian Copula would join the tie and win as the fastest. The closest case measured was 0.9029.
