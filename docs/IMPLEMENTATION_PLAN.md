# SynthGuard-Ops — Implementation Plan (Deliverable 1)

Deadline: **Sun Nov 22 2026, 11:59 PM PKT**. Target: submission-ready Sat Nov 21.
Task IDs match the team tracker. Owner = whose branch / who reviews the PR.
Each task lists **Build**, **Done when** (acceptance criteria) and **Maps to** (D1 requirement).

---

## 0. Target repository layout

```
synthguard-ops/
├── CLAUDE.md
├── README.md
├── pyproject.toml            # package + deps + ruff + pytest config
├── requirements.lock         # pip freeze after a clean install (pinned)
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .dockerignore  .gitignore  .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── params.yaml               # every tunable value lives here
├── dvc.yaml  dvc.lock  .dvc/config
├── data/
│   ├── raw/                  # DVC output (adult.data)
│   ├── processed/            # DVC output (train.csv, holdout.csv, metadata.json)
│   └── synthetic/            # DVC output (<generator>.csv)
├── models/                   # DVC output (<generator>.pkl)
├── metrics/                  # small JSON metric files (git-tracked, `dvc metrics`)
├── reports/                  # validation reports, quality reports, plots
├── src/synthguard/
│   ├── config.py             # load params.yaml + smoke overrides, seeding helper
│   ├── data/{schema.py, download.py, validate.py, prepare.py}
│   ├── generators/{base.py, marginal.py, sdv_models.py}
│   ├── evaluation/{fidelity.py, utility.py, privacy.py, score.py}
│   ├── tracking/{mlflow_utils.py, pyfunc_model.py, registry.py}
│   └── stages/{download.py, validate.py, prepare.py, baselines.py, generate.py,
│               evaluate.py, register.py}       # thin CLI entry points used by dvc.yaml
├── tests/                    # pytest, all runnable in smoke mode in < 2 min
├── notebooks/01_eda.ipynb
├── docs/                     # this plan, PROGRESS.md, walkthrough/, metrics.md, decisions
└── submission/               # MLflow exports + screenshots for Classroom
```

## 1. DVC pipeline (dvc.yaml)

```
download ─► validate ─► prepare ─► baselines ─► generate ─► evaluate ─► register
```

| Stage | Command | deps | outs / metrics |
|---|---|---|---|
| download | `python -m synthguard.stages.download` | params: data.url, data.sha256 | `data/raw/adult.data` |
| validate | `python -m synthguard.stages.validate` | raw data, schema.py | `reports/validation/raw.json` (fails pipeline on error) |
| prepare | `python -m synthguard.stages.prepare` | raw data, params: data.* , seed | `data/processed/{train,holdout}.csv`, `metadata.json`; metrics `metrics/prepare.json` |
| baselines | `python -m synthguard.stages.baselines` | processed data, params: generators.marginal/gaussian_copula, utility | `models/{marginal,gaussian_copula}.pkl`, `data/synthetic/…`; metrics `metrics/baselines.json`, `metrics/runs.json` |
| generate | `python -m synthguard.stages.generate` | processed data, **`metrics/baselines.json`** (forces baselines first), params: generators.ctgan/tvae | `models/{ctgan,tvae}.pkl`, `data/synthetic/…`; updates `metrics/runs.json` |
| evaluate | `python -m synthguard.stages.evaluate` | all synthetic csvs, processed data, params: evaluation.* | `metrics/evaluation.json`, `reports/eval/` |
| register | `python -m synthguard.stages.register` | `metrics/evaluation.json`, models, params: selection.* | `metrics/registered.json` |

Notes
- Small JSON files in `metrics/` are git-tracked (`cache: false`) so `dvc metrics show/diff` works.
- `metrics/runs.json` maps generator name → MLflow run_id so `evaluate` logs metrics into the
  same run as training (`mlflow.start_run(run_id=...)`).
- The grader needs **no DVC remote credentials**: `dvc repro` rebuilds everything from the public
  URL. The remote is only for the team to share cached outputs.

## 2. params.yaml (starting values)

```yaml
seed: 42
data:
  url: https://archive.ics.uci.edu/static/public/2/adult.zip
  member: adult.data
  sha256: "<fill in after first download in T04, then pin>"
  target: income
  test_size: 0.30
  min_rows: 30000          # raw-data quality check
  max_null_fraction: 0.10
generators:
  marginal: {}
  gaussian_copula: {}
  ctgan: {epochs: 100, batch_size: 500}
  tvae:  {epochs: 100, batch_size: 500}
evaluation:
  utility:
    random_forest: {n_estimators: 200, max_depth: null}
    xgboost: {n_estimators: 300, max_depth: 6, learning_rate: 0.1}
  privacy:
    mia_sample_size: 2000   # members and non-members each
selection:
  weights: {quality: 0.5, utility: 0.5}
  gate: {mia_auc_max: 0.60, exact_match_max: 0.01}
smoke:                      # used when SYNTHGUARD_SMOKE=1
  rows: 1000
  epochs: 1
```

## 3. MLflow conventions

- Tracking URI `sqlite:///mlflow.db` (set in `mlflow_utils.py`, overridable by `MLFLOW_TRACKING_URI`).
- Experiment `synthguard-d1`. Run name = generator name.
- Params: all generator hyperparameters + `seed` + `smoke`.
- Tags: `git_commit`, `data_train_md5`, `data_holdout_md5`, `dvc_lock_md5`, `generator`,
  `role` (`baseline` | `primary` | `reference`), `stage`.
- Artifacts: `params.yaml`, `dvc.lock`, quality report JSON, sample of 100 synthetic rows,
  validation report.
- Metrics (after evaluate): `quality_score`, `column_shapes`, `column_pair_trends`,
  `tstr_rf_{precision,recall,f1,roc_auc}`, `tstr_xgb_*`, `utility_ratio`,
  `dcr_median_synth_train`, `dcr_ratio`, `exact_match_rate`, `mia_auc`, `mia_advantage`,
  `fit_seconds`, `sample_seconds`, `peak_memory_mb`, `selection_score`.
- Reference run `trtr_reference`: RF and XGBoost trained on real train, tested on hold-out.
- Registry: model `synthguard-generator`, `mlflow.pyfunc` wrapper whose `predict(df)` reads
  `n_rows` from the input and returns sampled rows. Version tags: `generator`, `selection_score`,
  `privacy_gate`, `git_commit`, `data_train_md5`. Set alias `champion` on the newest version.

---

## Day 1 — Mon Nov 16 · Setup & Data

### T01 · Turab · GitHub repo + collaborators — MANUAL (Turab), then Claude Code wires it up
Follow docs/REPO_SETUP.md steps 1–5 on github.com. Claude Code then: `git init -b main`,
check `git config user.name/user.email` are Turab's, create `.gitignore` first, make the initial
scaffold commit on `main`, `git remote add origin https://github.com/turab1020/SynthGuard-Ops.git`, and push. Set Turab's git identity yourself if missing; add `handoff/` to `.gitignore`.
**Done when:** teammates accepted invites (done Nov 16); initial commit visible on GitHub; `handoff/` ignored; no AI attribution in `git log`.

### T02 · Turab · Repo scaffold  — branch `turab/scaffold`
**Build:** layout above, `pyproject.toml` (deps: pandas, numpy, scikit-learn, xgboost, sdv,
sdmetrics, torch (CPU), mlflow, dvc, pandera, pyyaml, requests, psutil; dev: pytest, ruff,
pre-commit, jupyter), Makefile targets, `.gitignore`, pre-commit (ruff), `config.py` with
`load_params()` + `set_seed()` (random, numpy, torch) + smoke override. After a clean install,
write `requirements.lock` with `pip freeze`.
**Done when:** fresh venv → `make setup && make lint test` green (a trivial config test).

### T03 · Turab · DVC + MLflow init — branch `turab/scaffold`
**Build:** `dvc init`; a DVC remote (default: local folder outside the repo, documented in
README; Google Drive/DagsHub optional later); `mlflow_utils.py` (`get_tracking_uri`,
`start_candidate_run`, `log_common_tags` computing git commit + file md5s); `make ui`.
**Done when:** a dummy run appears in `make ui`; `dvc status` works. **Maps to:** versioned data, tracking.

### T04 · Nufil · Download raw data + EDA — branch `nufil/data`
**Build:** `download` stage: fetch zip, extract `adult.data`, verify SHA-256 (fail loudly on
mismatch; first run prints the hash to pin in `params.yaml`). `notebooks/01_eda.ipynb`:
dtypes, `?` counts per column, class balance (~24% `>50K`), duplicate count, numeric ranges.
Save key findings to `docs/walkthrough/T04-eda.md`.
**Done when:** `dvc repro download` produces tracked raw data; EDA notebook runs top to bottom.
**Maps to:** Versioned data.

### T05 · Nufil · Validation stage (pandera) — branch `nufil/data`
**Build:** `schema.py` with `RAW_SCHEMA` and `PROCESSED_SCHEMA`:
- exact column set and order: `age, workclass, fnlwgt, education, education_num, marital_status,
  occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week,
  native_country, income`
- dtypes; ranges (age 17–90, education_num 1–16, hours 1–99, capital_* ≥ 0, fnlwgt > 0)
- allowed category sets (taken from the UCI documentation, plus `"Unknown"`)
- dataframe checks: rows ≥ `min_rows`, per-column null fraction ≤ `max_null_fraction`,
  `income` has both classes.
`validate` stage writes `reports/validation/raw.json` and exits non-zero on failure
(use `lazy=True` to report all errors at once). `prepare` re-validates its outputs with
`PROCESSED_SCHEMA`.
**Done when:** valid data passes; `tests/test_validation.py` shows each corruption type
(missing column, wrong dtype, out-of-range, unknown category, too many nulls, too few rows)
is caught. **Maps to:** schema & quality checks before training.

### T06 · Nufil · Prepare stage + leakage-free split — branch `nufil/data`
**Build:** strip whitespace, `?`→`"Unknown"`, normalise `income` labels, **drop exact duplicates
before splitting**, stratified 70/30 split (`seed`), save train/holdout CSVs, build SDV metadata
JSON (categorical vs numerical columns explicit). Write `metrics/prepare.json`
(row counts, duplicates removed, class balance per split).
**Done when:** `tests/test_split.py` proves: no shared rows between train and hold-out, class
ratio within ±1% across splits, deterministic with the same seed. **Maps to:** leakage-free splits.

### T07 · Faizan · Metrics spec — branch `faizan/metrics-spec`
**Build:** `docs/metrics.md`: exact definitions + formulas for quality score (SDMetrics
QualityReport), TRTR/TSTR protocol, utility_ratio, DCR and dcr_ratio, exact-match rate,
distance-based MIA (AUC, advantage = max(TPR−FPR)), selection score and gate. Note that privacy
numbers are empirical risk estimates, not guarantees.
**Done when:** file merged after Faizan's review.

**CHECKPOINT 1 —** summarise for Turab, then continue: `dvc dag`, `dvc repro` up to `prepare`, test output,
`metrics/prepare.json`. Open PR `turab/scaffold`; handoff bundles ready for T04–T06 (Nufil) and T07 (Faizan).

---

## Day 2 — Tue Nov 17 · Baselines & Generators

### T08 · Nufil · Generator wrapper + baselines — branch `nufil/generators`
**Build:** `generators/base.py`: `Generator` interface (`fit(df)`, `sample(n)`, `save`, `load`)
that records `fit_seconds`, `sample_seconds`, `peak_memory_mb`. `marginal.py`: independent
per-column sampler (categorical frequencies; numeric via empirical quantiles) — the rule-based
baseline. `sdv_models.py`: Gaussian Copula, CTGAN, TVAE wrappers (check installed SDV API).
`baselines` stage: TRTR reference run, then marginal + Gaussian Copula runs; each run logs params,
tags, timing, sample artifact; synthetic size = hold-out size. Writes `metrics/baselines.json`
and `metrics/runs.json`.
**Done when:** MLflow shows `trtr_reference`, `marginal`, `gaussian_copula` runs with full
tags. **Maps to:** baseline logged before primary model.

### T09 · Nufil · CTGAN + TVAE — branch `nufil/generators`
**Build:** `generate` stage (depends on `metrics/baselines.json`). Seeds set before fit.
Log as `role=primary`. Default 100 epochs (expect tens of minutes on CPU; run in background,
smoke mode for iteration).
**Done when:** both runs in MLflow; synthetic CSVs + pickled models tracked by DVC.

### T10 · Faizan · Fidelity — branch `faizan/evaluation`
**Build:** `fidelity.py`: SDMetrics single-table QualityReport (overall, column shapes,
column pair trends) + per-column KS (numeric) / TVD (categorical) + correlation-matrix
difference for numeric columns. Save report details JSON.
**Done when:** function returns a flat metric dict; unit test on identical data gives ≈1.0.

### T11 · Faizan · Utility — branch `faizan/evaluation`
**Build:** `utility.py`: one shared preprocessing (one-hot categoricals, scale numerics) fitted
on the training source; RF and XGBoost; TRTR (real train → hold-out) and TSTR (synthetic → same
hold-out); precision, recall, F1, ROC-AUC (positive class `>50K`); `utility_ratio`.
**Done when:** TRTR ROC-AUC is in a sensible range (~0.85–0.92); test on tiny data passes.

### T12 · Turab · Dockerfile — branch `turab/docker`
**Build:** `python:3.11-slim`, git installed, CPU-only torch (PyTorch CPU index), install
`requirements.lock` + package, non-root user, `WORKDIR /app`. `.dockerignore` excludes data,
models, mlruns, venv.
**Done when:** `docker build` succeeds on Apple Silicon and `docker run … make test` passes.

**CHECKPOINT 2 —** summarise (MLflow runs, PRs opened), continue.

---

## Day 3 — Wed Nov 18 · Pipeline & Registry

### T13 · Faizan · Privacy — branch `faizan/evaluation`
**Build:** `privacy.py` on a shared encoding (one-hot + min-max scale, fitted on train):
- DCR: nearest-neighbour distance synth→train; same for hold-out→train;
  `dcr_ratio = median(synth→train) / median(holdout→train)` (≈1 is good; ≪1 = memorisation).
- `exact_match_rate`: fraction of synthetic rows identical to a training row.
- Distance-based MIA: sample `mia_sample_size` members (train) and non-members (hold-out);
  score = −distance to nearest synthetic row; report ROC-AUC and advantage.
**Done when:** unit tests: copying train → exact_match_rate≈1 and MIA AUC high; independent noise → MIA AUC ≈0.5.

### T14 · Faizan · Evaluate stage + score — branch `faizan/evaluation`
**Build:** `evaluate` stage: for every candidate compute fidelity, utility, privacy; log into
its MLflow run (by run_id); compute `selection_score` and gate result (`score.py`); write
`metrics/evaluation.json` (one entry per generator) and comparison table to `reports/eval/`.
**Done when:** `dvc repro` runs through evaluate; `dvc metrics show` prints all candidates.

### T15 · Nufil · Params-driven generate + seeds — branch `nufil/generators`
**Build:** confirm every hyperparameter comes from `params.yaml` and is listed in `dvc.yaml`
`params:`; changing `generators.ctgan.epochs` re-runs only generate → evaluate → register.
**Done when:** demonstrated with `dvc status` / `dvc repro` output in the walkthrough.

### T16 · Turab · Registry — branch `turab/registry`
**Build:** `pyfunc_model.py` (wrapper loading the pickled synthesizer as an artifact; signature
with `n_rows` input); `register` stage: apply gate + pick top score, log/registers it as
`synthguard-generator` with version tags + alias `champion`; write `metrics/registered.json`
(generator, version, selection_score, key metrics).
**Done when:** MLflow Models page shows the version; loading via
`mlflow.pyfunc.load_model("models:/synthguard-generator@champion")` and predicting 10 rows works
(add this as a test in smoke mode). **Maps to:** register top-performing model.

### T17 · All · MILESTONE
Fresh clone in a temp dir → `make setup && make reproduce` → registered model + metrics.

**CHECKPOINT 3 —** summarise, push, open PRs, continue (stack on unmerged branches if needed).

---

## Day 4 — Thu Nov 19 · One-command run, CI, tests

### T18 · Turab · One command — branch `turab/one-command`
**Build:** `docker-compose.yml` with service `pipeline` (build `.`, mount `.:/app`, command
`make reproduce`) and service `mlflow` (`mlflow ui` on port 5000 against the same `mlflow.db`).
README "Reproduce in one command" section.
**Done when:** on a fresh clone, `docker compose run --rm pipeline` reproduces
`metrics/registered.json` within tolerance (see T22). **Maps to:** reproducible execution.

### T19 · Turab · CI — branch `turab/ci`
**Build:** `.github/workflows/ci.yml`: Python 3.11, cache pip, `make lint`, `make test`,
`make smoke` (SYNTHGUARD_SMOKE=1, writes to a temp MLflow DB). Keep under ~10 min.
**Done when:** CI green on the PR.

### T20 · Nufil · Data/generator tests — branch `nufil/tests`
Validation corruption tests (if not complete), split/leakage tests, generator output
shape/columns/dtypes/category sets in smoke mode.

### T21 · Faizan · Metric tests — branch `faizan/tests`
Known-answer tests for fidelity, utility, privacy, score and gate logic.

**CHECKPOINT 4 —** summarise, push, open PRs, continue.

---

## Day 5 — Fri Nov 20 · Verify & Evidence

### T22 · Nufil · Fresh-clone reproducibility check
Clone into a temp directory, run the one command, compare `metrics/registered.json` with the
committed one. Tolerance: ±0.02 on scores (CTGAN/TVAE training on CPU is not bit-exact; document
why). Record the result in README.

### T23 · Faizan · Results table
`reports/results.md`: table of all candidates (quality, utility ratio, ROC-AUC, MIA AUC,
dcr_ratio, exact-match, fit time), which one was registered and why. Numbers from `metrics/`.

### T24 · Turab · Submission evidence + README
`make export`: script exporting all runs of `synthguard-d1` to `submission/mlflow_runs.csv` and
registered model versions to `submission/registered_models.json`. Turab takes screenshots
(runs list, one run's params/metrics/tags, Models page) into `submission/screenshots/`.
README: overview, architecture (pipeline DAG), how each D1 requirement is met (table with file
links), one-command reproduction, team & ownership, AI-assistance statement if the course
requires it.

**CHECKPOINT 5 —** summarise, push, open PRs, continue.

---

## Day 6 — Sat Nov 21 · Submit

### T25 · All · Buffer — fix issues, review and merge all PRs.
### T26 · Turab · Submit (manual)
Classroom: repo link, DVC config files (`dvc.yaml`, `dvc.lock`, `.dvc/config`, `params.yaml`),
MLflow screenshots/exports. Make sure the repo is shared with the instructor if private.

---

## Deferred to Deliverable 2 (do not build now)
Selection gates config + Pareto ranking, Optuna closed loop, single-metric vs SynthGuard
experiment, FastAPI service, Streamlit dashboard, full docker-compose stack, final report/slides.

## Risks and fallbacks
- **CTGAN/TVAE slow on CPU:** use smoke mode while developing; reduce epochs to 50 if a full
  run exceeds ~45 min; record the change in `params.yaml` and the walkthrough.
- **UCI download blocked/changed:** fall back to `sklearn.datasets.fetch_openml("adult", version=2)`
  inside the same stage, normalised to the same schema; pin its hash instead.
- **SDV/MLflow API differences:** check installed versions first; adapt wrappers, not the design.
- **Docker on Apple Silicon:** use multi-arch base image; CPU torch wheels for aarch64.
