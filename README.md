# SynthGuard-Ops 🛡️

> **Quality-Gated Synthetic Tabular Data Pipeline**  
> MLOps Course Project — FAST National University (BSCS 2023)  
> Target: **Deliverable 1** (Due: Sun Nov 22, 2026)

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
| **Muhammad Nufil Anwer** | Data & Generation Lead | Data ingestion & integrity checks, Pandera schema validation, leakage-free data splits, synthetic generators (Marginal, Gaussian Copula, CTGAN, TVAE). |
| **Muhammad Faizan** | Evaluation & Metrics Lead | Metrics formalization, fidelity scoring (SDMetrics), downstream utility evaluation (TSTR/TRTR), privacy risk assessment (MIA, DCR), quality gating. |

---

## 🔄 Pipeline Architecture (DVC DAG)

```mermaid
flowchart LR
    A["download\n(UCI Adult + SHA-256)"] --> B["validate\n(Pandera Schemas)"]
    B --> C["prepare\n(Clean + Stratified Split)"]
    C --> D["baselines\n(Marginal + Copula)"]
    D --> E["generate\n(CTGAN + TVAE)"]
    E -.-> F["evaluate\n(Fidelity + Utility + Privacy)"]
    F -.-> G["register\n(Quality Gate + Champion Model)"]

    classDef completed fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef pending fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,stroke-dasharray: 5 5;
    class A,B,C,D,E completed;
    class F,G pending;
```

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

---

## ⏳ Pending Work (Roadmap to Deliverable 1)

| Task ID | Owner | Workstream | Description | Status |
|---|---|---|---|---|
| **T14** | Faizan | Evaluation | Implement `evaluate` stage: run candidate evaluations, compute composite selection score, and apply privacy gate. | In progress |
| **T15** | Nufil | Generation | Ensure generator parameters and RNG seeds are fully controlled via `params.yaml` with verified DVC repro triggers. | To do |
| **T16** | Turab | Platform | Implement MLflow Model Registry stage: register champion generator under alias `champion`. | To do |
| **T17** | All | Milestone | Full end-to-end pipeline run from raw download through champion model registration. | To do |
| **T18** | Turab | Platform | Configure `docker-compose.yml` for zero-configuration, single-command pipeline reproduction. | To do |
| **T19** | Turab | Platform | Setup GitHub Actions CI pipeline running linting, unit tests, and smoke pipeline execution. | To do |
| **T20** | Nufil | Data/Generation | Expand unit tests for synthetic data distributions, column structures, and generator determinism. | To do |
| **T21** | Faizan | Evaluation | Unit tests with synthetic known-answer fixtures for composite score and gate logic. | To do |
| **T22** | Nufil | Verification | Fresh-clone reproduction audit to verify score reproducibility within tolerance ($\pm 0.02$). | To do |
| **T23** | Faizan | Documentation | Compile model comparison and selection evidence table in `reports/results.md`. | To do |
| **T24** | Turab | Platform | Generate MLflow run exports (`submission/mlflow_runs.csv`) and UI screenshots. | To do |
| **T25** | All | Buffer | Final code review, pre-commit audit, and branch cleanup. | To do |
| **T26** | Turab | Submission | Package deliverables and submit to Google Classroom. | To do |

---

## 📂 Repository Structure

```
SynthGuard-Ops/
├── .dockerignore                     # Docker build exclusions
├── Dockerfile                        # Multi-stage CPU container
├── .dvc/                             # DVC internal configuration
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
│   └── runs_primary.json             # MLflow run IDs for primary models (T09)
├── models/                           # Serialized generator artifacts (DVC managed)
├── notebooks/                        # Jupyter notebooks for data analysis
│   └── 01_eda.ipynb                  # Exploratory Data Analysis (T04)
├── reports/                          # Generated evaluation and validation reports
│   └── validation/                   # Pandera raw and processed schema check results
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
│   │   └── privacy.py                # DCR, exact match, MIA AUC
│   ├── stages/                       # Thin CLI entry points invoked by DVC
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── prepare.py
│   │   ├── baselines.py
│   │   └── generate.py
│   └── tracking/                     # MLflow experiment tracking utilities
│       └── mlflow_utils.py
├── tests/                            # Test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_fidelity.py
│   ├── test_generators.py
│   ├── test_mlflow_utils.py
│   ├── test_privacy.py
│   ├── test_split.py
│   ├── test_utility.py
│   └── test_validation.py
├── dvc.yaml                          # DVC pipeline stage definitions (download -> generate)
├── dvc.lock                          # Pinned stage inputs, outputs, and hashes
├── Makefile                          # Command runners (setup, lint, test, reproduce)
├── params.yaml                       # Pipeline hyperparameters & configuration
├── pyproject.toml                    # Package metadata & tool settings
└── requirements.lock                 # Pinned dependencies
```

---

## 🚀 How to Run & Verify

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
# Launch MLflow tracking server locally on port 5000
make ui
# View candidate runs: trtr_reference, marginal, gaussian_copula, ctgan, tvae
```
