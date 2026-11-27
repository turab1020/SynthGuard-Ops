# SynthGuard-Ops 🛡️

> **Quality-Gated Synthetic Tabular Data Pipeline**  
> MLOps Course Project — FAST National University (BSCS 2023)  
> Status: **Deliverable 1** (Completed) | **Deliverable 2** (In Progress)

---

## 📌 Project Overview

**SynthGuard-Ops** is an end-to-end, production-grade MLOps pipeline designed to generate, validate, and evaluate synthetic tabular data using the UCI **Adult Income** dataset. 

Synthetic tabular data must satisfy three critical requirements before enterprise release:
1. **Fidelity:** Statistical alignment and correlation preservation with real data.
2. **Utility:** Downstream machine learning performance parity (Train on Synthetic, Test on Real — TSTR vs. TRTR).
3. **Privacy:** Protection against empirical inference attacks (Membership Inference Attack AUC, Distance to Closest Record, and exact-match rate).

SynthGuard-Ops enforces strict quality and privacy gates, tracks every candidate generator run in **MLflow**, versions code and datasets with **Git + DVC**, and registers the champion generator to the MLflow Model Registry.

---

## 👥 Team & Ownership

| Team Member | Role | Workstream & Ownership |
|---|---|---|
| **Turab Haider** | MLOps & Platform Lead | Project scaffolding, DVC pipeline configuration, MLflow tracking & Model Registry, Dockerization, CI/CD, Deliverable 1 submission. |
| **Muhammad Nufil Anwer** | Data & Generation Lead | Data ingestion & integrity checks, Pandera schema validation, leakage-free data splits, synthetic generators (Marginal, Gaussian Copula, CTGAN, TVAE), downstream income classifier (`train_classifier`), traffic simulation (`monitoring/simulate.py`), Streamlit UI (`ui/`). |
| **Muhammad Faizan** | Evaluation & Metrics Lead | Metrics formalization, fidelity scoring (SDMetrics), downstream utility evaluation (TSTR/TRTR), privacy risk assessment (MIA, DCR), quality gating. |

---

## 🔄 Pipeline Architecture (DVC DAG)

```mermaid
flowchart LR
    A["download<br/>(UCI Adult + SHA-256)"] --> B["validate<br/>(Pandera Schemas)"]
    B --> C["prepare<br/>(Clean + Stratified Split)"]
    C --> D["baselines<br/>(Marginal + Copula)"]
    D --> E["generate<br/>(CTGAN + TVAE)"]
    D --> F["evaluate<br/>(Fidelity + Utility + Privacy)"]
    E --> F
    F --> G["register<br/>(Quality Gate + Champion Model)"]
    G --> H["train_classifier<br/>(XGBoost on Champion Synth)"]

    classDef completed fill:#d4edda,stroke:#28a745,stroke-width:2px;
    class A,B,C,D,E,F,G,H completed;
```

---

## 🌐 Live deployment (Deliverable 2)

**Live API:** https://synthguard-ops.onrender.com · interactive docs: https://synthguard-ops.onrender.com/docs

| Endpoint | What it does |
|---|---|
| `GET /health` | Service status, model version, deployed commit |
| `POST /predict` | Income prediction (`>50K` / `<=50K`), probability, confidence and TreeSHAP explanation (top features) |
| `GET /metrics` | Prometheus metrics: requests, latency, errors, predictions, drift |
| `GET /generate?n=` | Synthetic rows from the champion generator; **off on Render** (needs torch/SDV), on locally with `make serve` |

```bash
curl -s https://synthguard-ops.onrender.com/health
curl -s -X POST https://synthguard-ops.onrender.com/predict -H 'Content-Type: application/json' -d '{
  "age": 45, "workclass": "Private", "fnlwgt": 190000, "education": "Masters", "education_num": 14,
  "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial", "relationship": "Husband",
  "race": "White", "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 50,
  "native_country": "United-States"}'
```

- **Hosting:** Render free Docker web service (Oregon), built from `Dockerfile.serve` (no torch/SDV, fits 512 MB).
  The free instance sleeps when idle: the first request after a sleep takes about a minute.
  Warm it up with `python scripts/smoke_test_api.py --url https://synthguard-ops.onrender.com`.
- **Deploys only through the gate:** on every merge to `main`, `deploy.yml` runs lint, tests, data validation,
  the evaluation gate and a 512 MB image smoke test; only then it calls Render's deploy hook, waits until the
  live `/health` reports the new commit, and smoke-tests the live URL. Render auto-deploy is off.
- **Streamlit UI (`ui/app.py`, `Dockerfile.ui`):** interactive web frontend built dynamically from `/openapi.json` (14-field form, live `>50K` / `<=50K` probability, confidence, log-odds TreeSHAP bar chart, sleeping-instance wake retries, and local synthetic row generation tab):
  ```bash
  pip install -r ui/requirements.txt
  streamlit run ui/app.py                                           # against live Render API
  SYNTHGUARD_API_URL=http://localhost:8000 streamlit run ui/app.py  # against local API (enables Generate tab)
  ```

---

## 📈 Observability (local demo stack)

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

| Service | URL | What it shows |
|---|---|---|
| API (same image as Render) | http://localhost:8000/docs | live predictions; requests logged to `monitoring/logs/requests.jsonl` for drift checks |
| Prometheus | http://localhost:9090 | scrapes the local API **and the live Render service** (`/metrics`) |
| Grafana | http://localhost:3000 | dashboard "SynthGuard serving": request rate, p50/p95/p99 latency, 5xx error rate, predictions by label, prediction drift, input drift share |

Metrics exported by the API: `synthguard_requests_total`, `synthguard_request_latency_seconds`,
`synthguard_errors_total`, `synthguard_predictions_total`, `synthguard_prediction_probability`, `synthguard_drift_share`.

```bash
# Send 400 normal vs. 400 drifted requests from unseen UCI adult.test rows
python scripts/simulate_traffic.py send --url http://localhost:8000 --mode normal --n 400
python scripts/simulate_traffic.py send --url http://localhost:8000 --mode drifted --n 400
```

---

## 🎯 Deliverable 1: Requirements and Results

### How each requirement is met

| Requirement | How it is met | Where to check |
|---|---|---|
| Versioned data | DVC stages download, prepare and generate the data; `dvc.lock` pins every hash; the raw file has a pinned SHA-256 | [`dvc.yaml`](dvc.yaml), [`dvc.lock`](dvc.lock), [`.dvc/config`](.dvc/config), [`params.yaml`](params.yaml) |
| Schema and quality checks before training | pandera checks columns, dtypes, ranges, categories, nulls and row count; the pipeline stops on bad input | [`schema.py`](src/synthguard/data/schema.py), [`test_validation.py`](tests/test_validation.py) |
| Leakage-free split | Exact duplicates removed **before** a stratified 70/30 split (seed 42); generators see train only; a test asserts zero shared rows | [`prepare.py`](src/synthguard/data/prepare.py), [`test_split.py`](tests/test_split.py) |
| Baseline before the primary model | The `baselines` stage (independent marginal, Gaussian Copula, real-data TRTR reference) must finish before `generate` (CTGAN, TVAE) | [`baselines.py`](src/synthguard/stages/baselines.py), [`dvc.yaml`](dvc.yaml) |
| Experiment tracking | One MLflow run per candidate with hyperparameters, git commit, data hashes and all metrics | [`mlflow_runs.csv`](submission/mlflow_runs.csv), [`mlflow_utils.py`](src/synthguard/tracking/mlflow_utils.py) |
| Registered top model | Privacy gate, then selection score and tie rule; winner registered as `synthguard-generator@champion` | [`registered_models.json`](submission/registered_models.json), [`registered.json`](metrics/registered.json), [`register.py`](src/synthguard/stages/register.py) |
| One-command reproduction | `docker compose run --rm pipeline` rebuilds everything from a fresh clone | "Reproduce in one command" below, [`docker-compose.yml`](docker-compose.yml), [`reproducibility.md`](reports/reproducibility.md) |

### Results

Registered model: **TVAE**, `synthguard-generator` version 1, alias `champion`. Real-data reference (TRTR) Random Forest ROC-AUC: 0.9091. Serving classifier trained only on TVAE's synthetic data: hold-out ROC-AUC 0.8844 (`metrics/classifier.json`, the production benchmark).

| Generator | Selection score | Quality | Utility ratio | MIA AUC | DCR ratio | Exact match | Fit time (s) |
|---|---|---|---|---|---|---|---|
| **TVAE (registered)** | 0.9058 | 0.8434 | 0.9682 | 0.4914 | 0.6297 | 0.0000 | 61.17 |
| CTGAN | 0.8984 | 0.8380 | 0.9587 | 0.4967 | 12.9558 | 0.0000 | 210.52 |
| Gaussian Copula (baseline) | 0.8827 | 0.8420 | 0.9234 | 0.4992 | 13.0881 | 0.0000 | 4.14 |
| Independent marginal (baseline) | 0.6841 | 0.8787 | 0.4896 | 0.5024 | 13.1747 | 0.0000 | 0.01 |

- All four candidates pass the privacy gate (MIA AUC ≤ 0.60, exact-match rate ≤ 0.01).
- `selection_score = 0.5 × quality + 0.5 × utility_ratio`. Scores within 0.02 of the best are tied, and the faster to train wins. TVAE and CTGAN are tied; TVAE is both higher here and faster.
- **TVAE's DCR ratio is 0.6297 (below 1):** its rows sit closer to training rows than real hold-out rows do, a mild memorisation signal. Its exact-match rate is 0.0000 and its MIA AUC is 0.4914 (about chance), so neither test shows a leak. DCR is reported but is not part of the gate. These are empirical risk estimates, not formal privacy guarantees.
- These are the Deliverable 2 reference numbers. The Deliverable 1 submission evidence in `submission/` comes from the D1 reference run (TVAE 0.9080); both select TVAE.

---

## ✅ Work Completed So Far

### 1. Platform & Tooling Scaffolding (`T01` – `T03`, `T12`)
- **Repository & Environment:** Initialized Python 3.11 environment with `pyproject.toml`, pinned dependencies (`requirements.lock`), code formatting and linting rules (`ruff`), and pre-commit hooks.
- **Data Version Control (DVC):** Initialized DVC tracking with autostage disabled to preserve strict multi-author git attribution.
- **Experiment Tracking (MLflow):** Implemented `synthguard.tracking.mlflow_utils` with automatic tagging of Git commit SHAs, DVC data MD5 hashes, and candidate metadata. Local tracking store backed by SQLite (`mlflow.db`).
- **Containerization (`T12`):** Multi-stage CPU-only `Dockerfile` and `.dockerignore` for containerized execution.

### 2. Raw Data Ingestion & EDA (`T04` — Merged via PR #3)
- **Download Stage:** Automated download of the UCI Adult dataset from the official UCI repository with cryptographic SHA-256 verification (`src/synthguard/data/download.py` & `src/synthguard/stages/download.py`).
- **Exploratory Data Analysis:** Authored `notebooks/01_eda.ipynb` covering:
  - Feature distributions and missing value analysis (`?` characters).
  - Class imbalance quantification (~24.08% positive `>50K` class).
  - Duplicate detection (24 exact duplicate rows identified in raw data).

### 3. Schema Validation Stage (`T05` — Merged via PR #3)
- **Pandera Schemas:** Defined strict `RAW_SCHEMA` and `PROCESSED_SCHEMA` in `src/synthguard/data/schema.py` enforcing:
  - Column ordering, exact data types, and non-negative constraints.
  - Value ranges (e.g., `age` [17–90], `hours_per_week` [1–99]).
  - Categorical domain constraints and handling of missing categories (`"Unknown"`).
  - DataFrame-level health checks: minimum row count ($\ge 30,000$), maximum null tolerance ($\le 10\%$).
- **Validation Stage:** `src/synthguard/stages/validate.py` executes lazy validation, dumps errors to `reports/validation/raw.json`, and halts pipeline execution on corruption.
- **Unit Tests:** `tests/test_validation.py` asserting that schema checks reliably catch invalid dtypes, out-of-range values, missing columns, and unseen categories.

### 4. Leakage-Free Data Preparation & Stratified Split (`T06` — Merged via PR #3)
- **Data Cleansing:** Stripped whitespace from categorical strings, transformed missing markers (`?` $\rightarrow$ `"Unknown"`), and normalized target labels.
- **Leakage Prevention:** Removed the 24 duplicate records **prior** to splitting to avoid train/test contamination.
- **Stratified Split:** Split data into 70% training (`train.csv`) and 30% holdout (`holdout.csv`) with `seed=42`.
  - Raw dataset: **32,561** rows.
  - Cleaned dataset: **32,537** rows.
  - Train set: **22,775** rows (positive rate: **24.09%**).
  - Holdout set: **9,762** rows (positive rate: **24.09%**).
- **Metadata & Tracking:** Generated SDV-compatible table metadata and logged split summary to `metrics/prepare.json`.

### 5. Metrics Specification & Evaluation Modules (`T07`, `T10`, `T11`, `T13`)
- **Metrics Specification (`T07`):** Authored `docs/metrics.md` detailing formal mathematical definitions for Quality Score, TRTR/TSTR protocol, utility ratio, Distance to Closest Record (DCR), exact-match rate, distance-based Membership Inference Attack (MIA AUC & advantage), and composite scoring.
- **Fidelity Module (`T10`):** `src/synthguard/evaluation/fidelity.py` computing SDMetrics QualityReport, shape/trend adherence, KS-test, total variation distance, and correlation matrix delta.
- **Utility Module (`T11`):** `src/synthguard/evaluation/utility.py` training Random Forest and XGBoost classifiers on real vs. synthetic data to benchmark TSTR vs. TRTR ROC-AUC, F1-score, and utility ratio.
- **Privacy Module (`T13`):** `src/synthguard/evaluation/privacy.py` computing empirical privacy metrics including 5th-percentile DCR, DCR ratio ($\text{DCR}_{\text{synth}} / \text{DCR}_{\text{real}}$), exact match count, and distance-based MIA AUC using holdout non-members.

### 6. Synthetic Data Generators (`T08`, `T09` — Merged via PR #13)
- **Generator Interface (`T08`):** `src/synthguard/generators/base.py` defining standard `Generator` protocol (`fit`, `sample`, `save`, `load`) with built-in telemetry tracking fit time, sample time, and peak memory usage (`psutil`).
- **Baseline Models (`T08`):** 
  - `IndependentMarginalSampler`: Fast empirical distribution baseline.
  - `GaussianCopulaSynthesizer`: Classical multivariate statistical copula baseline.
  - `baselines` stage: Fits and logs `trtr_reference`, `marginal`, and `gaussian_copula` candidate runs to MLflow (`metrics/baselines.json`, `metrics/runs_baselines.json`).
- **Deep Generative Synthesizers (`T09`):**
  - Integrated SDV 2.2 deep models: **CTGAN** and **TVAE** (`src/synthguard/generators/sdv_models.py`).
  - `generate` stage (`src/synthguard/stages/generate.py`): Depends on `baselines` outputs, trains CTGAN (100 epochs) and TVAE (100 epochs), and outputs synthetic datasets (`metrics/primary.json`, `metrics/runs_primary.json`).
  - Benchmark timings on CPU: CTGAN fit ~155.0 s, TVAE ~37.4 s, Gaussian Copula ~3.35 s, Marginal ~0.008 s for 9,762 synthetic rows.
- **DVC Pipeline Integration:** Pinned full DAG through `generate` stage in `dvc.yaml` and `dvc.lock`.

### 7. Parameter Wiring & Generator Pipeline Tests (`T15`, `T20` — Merged via PR #19)
- **Hyperparameter & DVC Tracking Wiring (`T15`):** Added comprehensive wiring checks in `tests/test_params_wiring.py`:
  - Verified that every parameter declared in `dvc.yaml` exists in `params.yaml`.
  - Confirmed stage-specific tracking: `generate` tracks `seed` and `generators.ctgan`/`tvae`, `baselines` tracks baseline parameters, and data stages track their respective parameters.
  - Enforced that all CTGAN/TVAE hyperparameter keys in `params.yaml` map directly to valid SDV `CTGANSynthesizer` and `TVAESynthesizer` arguments to catch configuration typos early.
  - Verified parameter pass-through from `params.yaml` directly to model initializations.
- **Training Runner & Generate-Stage Guards (`T20`):** Added robust tests in `tests/test_runner.py`:
  - Seeded smoke subsampling (1,000 rows) and synthetic sample size invariance.
  - Verified `train_candidate` produces expected outputs (CSV data, model pkl) and logs MLflow runs with tags, parameters, and telemetry metrics.
  - Guaranteed reproducibility of Gaussian Copula generation and SDV model serialization/deserialization round-trips.
  - Tested fail-fast guard in `src/synthguard/stages/generate.py` that strictly requires `metrics/baselines.json` before training commences.
- **Planted Bugs & DVC Granular Invalidation:** Demonstrated that editing `generators.ctgan.epochs` in `params.yaml` invalidates only the `generate` stage in `dvc status`. Verified that deliberately planted bugs (untracked params, `epoch` typo, missing baselines guard) reliably trigger immediate test failures.

### 8. Fresh-Clone Reproducibility Audit (`T22` — Merged via PR #26)
- **Zero-Config Reproduction Audit:** Verified that running `docker compose run --rm pipeline` from a fresh clone reproduces all outputs and registers the exact same champion model (**TVAE**).
- **Metric Invariance:** All four candidate generator selection scores matched the committed metrics to 6 decimal places (maximum absolute difference: `0.000000` vs tolerance $\pm 0.02$).
- **Cross-Architecture Tolerance Analysis:** Evaluated floating-point drift across CPU architectures (AMD EPYC, Intel Xeon, macOS arm64). Even under forced AVX2 conditions, the largest difference observed was `0.0103`, comfortably within the `0.02` tie tolerance. Documented full audit findings in `reports/reproducibility.md`.

### 9. Downstream Income Classifier on Champion Synthetic Data (`D2-02` — Merged via PR #28)
- **Serving Classifier Pipeline (`src/synthguard/serving/classifier.py`):** Implemented an XGBoost income classifier pipeline (`OneHotEncoder(handle_unknown="ignore")` + `StandardScaler` fitted strictly on the training data), evaluated on the fixed real hold-out split across ROC-AUC, F1, accuracy, precision, and recall.
- **DVC Stage (`train_classifier`):** Added `src/synthguard/stages/train_classifier.py` downstream of `register`. Trains exclusively on the champion generator's synthetic rows (**TVAE**), logs an MLflow run with `cloudpickle` serialization, registers `synthguard-classifier@champion`, and writes the production benchmark to `metrics/classifier.json`.
- **Hyperparameters & Unit Tests:** Configured `classifier` hyperparameters in `params.yaml` and added tests in `tests/test_classifier.py` verifying hold-out scoring, determinism, unseen category handling, and single-class training data rejection.

### 10. Serving API, Explainability, Gated CI/CD & Drift Monitoring (`PR #29` – `PR #39`)
- **Explainability & Evaluation Gate (`PR #29`):** Added TreeSHAP per-request explanations (`src/synthguard/serving/explain.py`) and champion-vs-challenger evaluation gating (`src/synthguard/serving/gate.py`, `scripts/eval_gate.py`).
- **FastAPI Service & Render Deployment (`PR #30`, `PR #31`, `PR #33`):** Built lightweight `Dockerfile.serve` and `src/synthguard/serving/app.py` (`/health`, `/predict`, `/metrics`, `/generate`), gated PR workflow (`pr-gate.yml`), and automated gated deployment (`deploy.yml`) to Render.
- **Observability & Evidently Drift Detection (`PR #34`, `PR #37` – `PR #39`):** Configured local Prometheus + Grafana stack (`docker-compose.monitoring.yml`), live request logging/exposure, Evidently feature & prediction drift detection (`src/synthguard/monitoring/drift.py`, `scripts/check_drift.py`), and the `drift-check.yml` workflow.

### 11. Normal & Drifted Traffic Simulation (`D2-10` — Merged via PR #40)
- **Leakage-Free Traffic Generator (`src/synthguard/monitoring/simulate.py`):** Downloads and verifies UCI `adult.test` (pinned SHA-256; rows never seen in training or hold-out), cleans records identically to the training pipeline, and generates seeded `normal` or `drifted` batches (`age +15`, `hours_per_week ×1.4`, `education_num +2` capped to valid Pandera bounds, plus 60% white-collar occupation shift).
- **Simulation & Retraining Batch CLI (`scripts/simulate_traffic.py`):** Supports `send` (POSTs simulated requests to `/predict` and summarizes status/label rates) and `batch` (exports JSONL request bodies or labeled `--csv` batches for downstream retraining).
- **Unit Tests (`tests/test_simulate.py`):** Verified deterministic seeded sampling, documented covariate shifts, schema-valid API payloads, `adult.test` parsing, and both CLI modes.

### 12. Streamlit UI with Live Predictions & SHAP Explanations (`D2-07` — Merged via PR #41)
- **Schema-Driven HTTP Client (`ui/client.py`):** Pure `requests`-based API client that builds all 14 form field specifications dynamically from the API's `/openapi.json` (enum dropdowns, numeric min/max bounds, schema defaults), wakes sleeping Render instances automatically, and surfaces field-level `422` validation errors cleanly.
- **Interactive Prediction & SHAP Frontend (`ui/app.py`):** Displays predicted income class, $P(\text{income} > 50\text{K})$, confidence, and a directional log-odds TreeSHAP contribution chart (`top-k` features + `all other features` summing to the logit), plus a conditional **Generate** tab when running against a local generator-enabled API.
- **Isolated Container & Tests (`Dockerfile.ui`, `ui/requirements.txt`, `tests/test_ui_client.py`):** Separate 512 MB Render-ready container (`$PORT`, `SYNTHGUARD_API_URL`) keeping `requirements.lock` and the API image untouched, verified by 8 in-process integration tests against the real FastAPI app.

---

## 🗺️ Project Roadmap & Status

### Deliverable 2 (Serving, Monitoring & Production Pipeline)

| Task ID | Owner | Workstream | Description | Status |
|---|---|---|---|---|
| **D2-02** | Nufil | Serving / ML | Train downstream XGBoost income classifier on champion synthetic data (`train_classifier` stage), register `synthguard-classifier@champion`, and record `metrics/classifier.json`. | **Done** (PR #28) |
| **D2 — Explain & Gate** | Faizan | Serving / Eval | TreeSHAP feature explanations (`serving/explain.py`) and champion-challenger evaluation gate (`serving/gate.py`). | **Done** (PR #29) |
| **D2 — Serving API & CD** | Turab | Serving / Platform | FastAPI inference service (`serving/app.py`), `Dockerfile.serve`, PR eval gate (`pr-gate.yml`), and gated Render CD (`deploy.yml`). | **Done** (PR #30, #31, #33) |
| **D2-07** | Nufil | UI / Frontend | Schema-driven Streamlit UI (`ui/app.py`, `ui/client.py`), `Dockerfile.ui`, live predictions with TreeSHAP bar charts, and API integration tests (`tests/test_ui_client.py`). | **Done** (PR #41) |
| **D2 — Observability & Drift** | Turab & Faizan | Monitoring | Prometheus + Grafana stack (`docker-compose.monitoring.yml`), Evidently drift check (`monitoring/drift.py`), and `drift-check.yml`. | **Done** (PR #34, #37–#39) |
| **D2-10** | Nufil | Monitoring | Simulate normal & drifted live traffic and labeled retraining batches from unseen UCI `adult.test` (`monitoring/simulate.py`, `scripts/simulate_traffic.py`). | **Done** (PR #40) |
| **D2 — Retraining Loop** | Team | Pipeline / CI | Automated/triggered retraining workflow appending new labeled batch, re-evaluating through quality/eval gates, and promoting updated model. | Pending |
| **D2 — Final Demo & Submission** | Team | Submission | End-to-end live drift & retraining demonstration, documentation polish, and Deliverable 2 submission. | Pending |

### Deliverable 1 (Completed)

| Task ID | Owner | Workstream | Description | Status |
|---|---|---|---|---|
| **T14** | Faizan | Evaluation | Implement `evaluate` stage: run candidate evaluations, compute composite selection score, and apply privacy gate. | **Done** (PR #15, PR #21) |
| **T15** | Nufil | Generation | Ensure generator parameters and RNG seeds are fully controlled via `params.yaml` with verified DVC repro triggers. | **Done** (PR #19) |
| **T16** | Turab | Platform | Implement MLflow Model Registry stage: register champion generator under alias `champion`. | **Done** (PR #11, PR #14) |
| **T17** | All | Milestone | Full end-to-end pipeline run from raw download through champion model registration. | **Done** (PR #22) |
| **T18** | Turab | Platform | Configure `docker-compose.yml` for zero-configuration, single-command pipeline reproduction. | **Done** (PR #10) |
| **T19** | Turab | Platform | Setup GitHub Actions CI pipeline running linting, unit tests, determinism, and repro checks. | **Done** (PR #9, PR #17) |
| **T20** | Nufil | Data/Generation | Expand unit tests for synthetic data distributions, column structures, and generator determinism. | **Done** (PR #19) |
| **T21** | Faizan | Evaluation | Unit tests with synthetic known-answer fixtures for composite score and gate logic. | **Done** (PR #16) |
| **T22** | Nufil | Verification | Fresh-clone reproduction audit to verify score reproducibility within tolerance ($\pm 0.02$). | **Done** (PR #26) |
| **T23** | Faizan | Documentation | Compile model comparison and selection evidence table in `reports/results.md`. | **Done** (PR #25) |
| **T24** | Turab | Platform | Generate MLflow run exports (`submission/mlflow_runs.csv`) and UI screenshots. | **Done** (PR #12, PR #27) |
| **T25** | All | Buffer | Final code review, pre-commit audit, and branch cleanup. | **Done** |
| **T26** | Turab | Submission | Package deliverables and submit to Google Classroom. | **Done** |

---

## 📂 Repository Structure

```
SynthGuard-Ops/
├── .dockerignore                     # Docker build exclusions
├── Dockerfile                        # Multi-stage CPU pipeline container
├── Dockerfile.serve                  # Lightweight 512 MB FastAPI serving image
├── Dockerfile.ui                     # Lightweight 512 MB Streamlit UI image (D2-07)
├── docker-compose.yml                # One-command pipeline reproduction
├── docker-compose.monitoring.yml     # Local API + Prometheus + Grafana stack
├── .dvc/                             # DVC internal configuration
├── deploy/model/                     # Exported classifier artifact & metadata for serving
├── data/                             # Versioned data artifacts (DVC managed)
│   ├── raw/                          # adult.data (download stage)
│   ├── processed/                    # train.csv, holdout.csv, metadata.json (prepare stage)
│   └── synthetic/                    # Generated synthetic datasets (<model>.csv)
├── docs/                             # Project documentation
│   └── metrics.md                    # Formal metric specifications (T07)
├── metrics/                          # Tracked lightweight metric summaries
│   ├── prepare.json                  # Dataset split and duplicate counts (T06)
│   ├── baselines.json                # Marginal & Gaussian Copula timings (T08)
│   ├── runs_baselines.json           # MLflow run IDs for baselines (T08)
│   ├── primary.json                  # CTGAN & TVAE timings (T09)
│   ├── runs_primary.json             # MLflow run IDs for primary models (T09)
│   ├── evaluation.json               # Candidate fidelity, utility & privacy scores
│   ├── registered.json               # Champion generator metadata
│   └── classifier.json               # Production classifier hold-out benchmark (D2-02)
├── models/                           # Serialized generator artifacts (DVC managed)
├── monitoring/                       # Prometheus config, Grafana dashboards & request logs
├── notebooks/                        # Jupyter notebooks for data analysis
│   └── 01_eda.ipynb                  # Exploratory Data Analysis (T04)
├── reports/                          # Generated evaluation and validation reports
│   ├── eval/                         # Model comparison table & metrics (T23)
│   ├── validation/                   # Pandera raw and processed schema check results
│   ├── results.md                    # Generator comparison summary (T23)
│   └── reproducibility.md            # Fresh-clone reproduction audit report (T22)
├── scripts/                          # Operational & CI/CD helper scripts
│   ├── check_drift.py                # Evidently drift check runner
│   ├── eval_gate.py                  # Champion-challenger evaluation gate CLI
│   ├── export_models.py              # Export champion classifier into deploy/model/
│   ├── simulate_traffic.py           # Send live traffic or generate normal/drifted batches (D2-10)
│   └── smoke_test_api.py             # API health, predict & metrics smoke test
├── src/synthguard/                   # Core Python package
│   ├── config.py                     # Parameter loader & seed management
│   ├── data/                         # Ingestion, schema, and processing modules
│   │   ├── download.py
│   │   ├── schema.py
│   │   ├── validate.py
│   │   └── prepare.py
│   ├── generators/                   # Synthetic data models (T08, T09)
│   │   ├── base.py                   # Generator abstract base class
│   │   ├── marginal.py               # Independent marginal sampler
│   │   ├── runner.py                 # Fit & telemetry runner
│   │   └── sdv_models.py             # Gaussian Copula, CTGAN, TVAE wrappers
│   ├── evaluation/                   # Evaluation modules (T10, T11, T13)
│   │   ├── fidelity.py               # SDMetrics quality score & distributions
│   │   ├── utility.py                # TRTR vs TSTR (RF & XGBoost)
│   │   ├── privacy.py                # DCR, exact match, MIA AUC
│   │   └── score.py                  # Composite selection score & tie-breaking
│   ├── monitoring/                   # Live drift detection & traffic simulation
│   │   ├── __init__.py
│   │   ├── drift.py                  # Evidently feature & prediction drift analysis
│   │   └── simulate.py               # Seeded normal & drifted batches from adult.test (D2-10)
│   ├── serving/                      # Inference API, classifier, explainability & gating
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI service + Prometheus instrumentation
│   │   ├── classifier.py             # XGBoost pipeline + real hold-out evaluation (D2-02)
│   │   ├── explain.py                # TreeSHAP feature attribution
│   │   └── gate.py                   # Champion-challenger promotion gate
│   ├── stages/                       # Thin CLI entry points invoked by DVC
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── prepare.py
│   │   ├── baselines.py
│   │   ├── generate.py
│   │   ├── evaluate.py
│   │   ├── register.py
│   │   └── train_classifier.py       # Champion synthetic-data classifier stage (D2-02)
│   └── tracking/                     # MLflow experiment tracking & registry utilities
│       ├── export.py
│       ├── mlflow_utils.py
│       ├── pyfunc_model.py
│       └── registry.py
├── tests/                            # Test suite
│   ├── conftest.py
│   ├── test_app.py
│   ├── test_classifier.py            # Synthetic-trained classifier & hold-out tests (D2-02)
│   ├── test_config.py
│   ├── test_drift.py
│   ├── test_evaluate_stage.py
│   ├── test_explain.py
│   ├── test_export.py
│   ├── test_fidelity.py
│   ├── test_gate.py
│   ├── test_generators.py
│   ├── test_metric_known_answers.py
│   ├── test_mlflow_utils.py
│   ├── test_params_wiring.py         # Hyperparameters & DVC wiring tests (T15)
│   ├── test_privacy.py
│   ├── test_register_stage.py
│   ├── test_registry.py
│   ├── test_runner.py                # Generator training runner & stage guards (T20)
│   ├── test_score.py
│   ├── test_simulate.py              # Normal/drifted traffic simulation & CLI tests (D2-10)
│   ├── test_split.py
│   ├── test_ui_client.py             # Streamlit UI client & OpenAPI schema tests (D2-07)
│   ├── test_utility.py
│   └── test_validation.py
├── ui/                               # Streamlit web frontend (D2-07)
│   ├── app.py                        # Live predictions, TreeSHAP chart & local Generate tab
│   ├── client.py                     # Schema-driven API client & wake retry handler
│   └── requirements.txt              # Isolated UI dependencies (streamlit, requests)
├── dvc.yaml                          # DVC pipeline stage definitions (download -> train_classifier)
├── dvc.lock                          # Pinned stage inputs, outputs, and hashes
├── Makefile                          # Command runners (setup, lint, test, reproduce)
├── params.yaml                       # Pipeline hyperparameters & configuration
├── pyproject.toml                    # Package metadata & tool settings
└── requirements.lock                 # Pinned dependencies
```

---

## 🚀 How to Run & Verify

### ▶️ Reproduce in one command (Docker)

Requirements: Docker (Desktop on macOS/Windows, Engine on Linux) and internet access for the
UCI download. No Python setup, DVC remote or credentials needed.

```bash
git clone https://github.com/turab1020/SynthGuard-Ops.git
cd SynthGuard-Ops
docker compose run --rm pipeline
```

This builds the CPU-only image (first time only), mounts the clone at `/app` and runs
`make reproduce` (`dvc repro` + `dvc metrics show`) inside it. Every output lands in the clone:
`data/`, `models/`, `metrics/*.json`, `reports/`, `dvc.lock` and the MLflow store `mlflow.db`.

```bash
docker compose up mlflow        # MLflow UI on http://localhost:5001  (Ctrl+C to stop)
```

Notes
- DVC only re-runs a stage when its code, data or parameters changed. The raw-data validation
  report is committed, so after the download produces the same SHA-256 the `validate` stage is
  reported as unchanged and its recorded result (`reports/validation/raw.json`) is reused.
  Force every stage with `docker compose run --rm pipeline dvc repro --force`.
- On Linux, if your user id is not 1000, run `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline`
  so files written into the clone are owned by you.
- Without Docker: `make setup && make reproduce`.

### 🔁 What "reproduces" means here

**The one command reproduces the registered model choice (TVAE), with selection scores within ±0.02 across CPU types.** Scores are not bit-identical across CPU models: repeated runs on one machine are identical, but different CPUs use different low-level numeric routines, so CTGAN and TVAE train to slightly different networks.

Committed reference: Docker fresh-clone run on GitHub Actions (x86_64 Linux), TVAE **0.905794**, CTGAN **0.898364**. An independent fresh-clone run reproduced all four selection scores of this reference to 6 decimals.

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

### 1. Environment Setup
```bash
# Create venv and install package in editable mode with dev dependencies
make setup
```

### 2. Run Tests & Code Quality Checks
```bash
# Run Ruff linting and formatting verification
make lint

# Execute pytest suite across all test modules
make test
```

### 3. Reproduce Pipeline Stages (via DVC)
```bash
# Reproduce data and generator stages (download -> validate -> prepare -> baselines -> generate)
dvc repro generate

# View pipeline DAG
dvc dag

# Inspect logged metrics
dvc metrics show
```

### 4. MLflow Tracking UI
```bash
# Launch MLflow tracking UI on http://localhost:5001 (macOS AirPlay uses port 5000)
make ui
# View candidate runs: trtr_reference, marginal, gaussian_copula, ctgan, tvae
```
