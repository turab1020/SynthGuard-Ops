# Reproducibility check

**Result: PASS.** A fresh clone running `docker compose run --rm pipeline` registered the same model (TVAE) and reproduced all four committed selection scores to 6 decimals.

## Fresh-clone check

Reference: GitHub Actions run `45848855837` (Docker fresh clone, x86_64 Linux), committed in `metrics/`.
Independent check: run `45850821198`, a separate clean checkout of the same commit. Tolerance: ±0.02.

| Generator | Committed score | Fresh-clone score | Difference | Privacy gate |
|---|---|---|---|---|
| TVAE (registered) | 0.905794 | 0.905794 | 0.000000 | passed |
| CTGAN | 0.898364 | 0.898364 | 0.000000 | passed |
| Gaussian Copula (baseline) | 0.882685 | 0.882685 | 0.000000 | passed |
| Independent marginal (baseline) | 0.684147 | 0.684147 | 0.000000 | passed |

Run IDs, timestamps and fit times change on every run and are not compared.

## Across CPU types

Repeated runs on one CPU type are identical. Different CPU types use different low-level numeric routines, so scores are **not bit-identical across CPU models**; they stay within the ±0.02 tolerance and TVAE is selected in every case.

Committed reference (TVAE / CTGAN): TVAE **0.905794**, CTGAN **0.898364**. An independent fresh-clone run reproduced all four selection scores of this reference to 6 decimals.

| Environment | CTGAN score (Δ vs reference) | TVAE score (Δ vs reference) | Selected by the tie rule |
|---|---|---|---|
| **Reference** (Docker fresh clone, x86_64 Linux; reproduced exactly by an independent run) | 0.898364 | 0.905794 | TVAE |
| Docker fresh clone, two other runners | 0.893697 (−0.0047) | 0.905247 (−0.0005) | TVAE |
| D1 reference run (AMD EPYC 7763) | 0.900101 (+0.0017) | 0.908011 (+0.0022) | TVAE |
| AMD EPYC 9V45 / 9V74 ¹ | 0.897160 (−0.0012) | 0.905247 (−0.0005) | TVAE |
| Intel Xeon Platinum 8573C ¹ | 0.900956 (+0.0026) | 0.905794 (−0.0000) | TVAE |
| macOS arm64 (native, no Docker) | 0.905879 (+0.0075) | 0.904732 (−0.0011) | TVAE |
| Intel Xeon 8573C, AVX2 forced (diagnostic) ¹ | 0.903353 (+0.0050) | 0.895534 (−0.0103) | TVAE |

¹ Retrains and scores CTGAN and TVAE only (the rule applied to those two scores); the other rows are full pipeline runs.

- Largest difference from the reference: **0.0103**, inside the ±0.02 tolerance.
- Largest native Mac vs Linux gap: **0.0087** (CTGAN, 0.905879 on macOS arm64 vs 0.897160 on AMD EPYC).
- CTGAN and TVAE are closer than this CPU-to-CPU variation, so `selection.tie_tolerance: 0.02` in `params.yaml` treats them as tied and picks the faster to train (TVAE, 4–5× faster on every CPU measured).
- Known limit: Gaussian Copula (0.8827) is 0.0231 below the reference top score. If a machine's top score fell below about 0.9027, it would join the tie and win as the fastest. The closest case measured was 0.9034 (0.0207 above it).
