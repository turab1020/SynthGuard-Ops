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
3. **Privacy:** Protection against empirical inference attacks (Membership Inference Attack AUC and exact match rate).

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
    C -.-> D["baselines\n(Marginal + Copula)"]
    D -.-> E["generate\n(CTGAN + TVAE)"]
    E -.-> F["evaluate\n(Fidelity + Utility + Privacy)"]
    F -.-> G["register\n(Quality Gate + Champion Model)"]

    classDef completed fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef pending fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,stroke-dasharray: 5 5;
    class A,B,C completed;
    class D,E,F,G pending;
```

---

## ✅ Work Completed So Far

### 1. Platform & Tooling Scaffolding (`T01` – `T03`)
- **Repository & Environment:** Initialized Python 3.11 environment with `pyproject.toml`, pinned dependencies (`requirements.lock`), code formatting and linting rules (`ruff`), and pre-commit hooks.
- **Data Version Control (DVC):** Initialized DVC tracking with autostage disabled to preserve strict multi-author git attribution.
- **Experiment Tracking (MLflow):** Implemented `synthguard.tracking.mlflow_utils` with automatic tagging of Git commit SHAs, DVC data MD5 hashes, and candidate metadata. Local tracking store backed by SQLite (`mlflow.db`).

### 2. Raw Data Download & EDA (`T04` — Merged via PR #3)
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
- **DVC Pipeline Integration:** Pinned pipeline dependencies and reproducible outputs in `dvc.yaml` and `dvc.lock`.

---

## ⏳ Pending Work (Roadmap to Deliverable 1)

| Task ID | Owner | Workstream | Description | Status |
|---|---|---|---|---|
| **T07** | Faizan | Evaluation | Write formal metric specification (`docs/metrics.md`) for fidelity, utility, and privacy. | In progress |
| **T08** | Nufil | Generation | Implement base synthesizer wrapper, Independent Marginal Sampler, and Gaussian Copula baseline stage. | To do |
| **T09** | Nufil | Generation | Implement deep generative synthesizers (CTGAN & TVAE) via SDV. | To do |
| **T10** | Faizan | Evaluation | Build Fidelity evaluation module using SDMetrics (`QualityReport`, shape/trend adherence). | To do |
| **T11** | Faizan | Evaluation | Build Downstream ML Utility module (TSTR vs. TRTR using Random Forest and XGBoost). | To do |
| **T12** | Turab | Platform | Author multi-stage `Dockerfile` and runtime environment. | To do |
| **T13** | Faizan | Evaluation | Build Privacy evaluation module (Distance to Closest Record, Exact-Match rate, MIA AUC). | To do |
| **T14** | Faizan | Evaluation | Implement `evaluate` stage: run candidate evaluations, compute composite selection score, apply gate. | To do |
| **T15** | Nufil | Generation | Ensure generator parameters and RNG seeds are fully controlled via `params.yaml` with DVC repro triggers. | To do |
| **T16** | Turab | Platform | Implement MLflow Model Registry stage: register champion generator under alias `champion`. | To do |
| **T17** | All | Milestone | Full end-to-end pipeline run from raw download to champion model registration. | To do |
| **T18** | Turab | Platform | Configure `docker-compose.yml` for zero-configuration, single-command pipeline reproduction. | To do |
| **T19** | Turab | Platform | Setup GitHub Actions CI pipeline running linting, unit tests, and smoke pipeline execution. | To do |
| **T20** | Nufil | Data/Generation | Expand unit tests for synthetic data distributions, column structures, and generator determinism. | To do |
| **T21** | Faizan | Evaluation | Unit tests with synthetic known-answer fixtures for fidelity, utility, and privacy metrics. | To do |
| **T22** | Nufil | Verification | Fresh-clone reproduction audit to verify score reproducibility within tolerance ($\pm 0.02$). | To do |
| **T23** | Faizan | Documentation | Compile model comparison and selection evidence table in `reports/results.md`. | To do |
| **T24** | Turab | Platform | Generate MLflow run exports (`submission/mlflow_runs.csv`) and UI screenshots. | To do |
| **T25** | All | Buffer | Final code review, pre-commit audit, and branch cleanup. | To do |
| **T26** | Turab | Submission | Package deliverables and submit to Google Classroom. | To do |

---

## 📂 Repository Structure

```
SynthGuard-Ops/
├── .dvc/                             # DVC internal configuration
├── data/                             # Versioned data artifacts (DVC managed)
│   ├── raw/                          # adult.data (download stage)
│   └── processed/                    # train.csv, holdout.csv, metadata.json (prepare stage)
├── metrics/                          # Tracked lightweight metric summaries
│   └── prepare.json                  # Dataset split and duplicate counts
├── notebooks/                        # Jupyter notebooks for data analysis
│   └── 01_eda.ipynb                  # Exploratory Data Analysis
├── reports/                          # Generated evaluation and validation reports
│   └── validation/                   # Pandera raw and processed schema check results
├── src/synthguard/                   # Core Python package
│   ├── config.py                     # Parameter loader & seed management
│   ├── data/                         # Data ingestion, schema, and processing modules
│   │   ├── download.py
│   │   ├── schema.py
│   │   ├── validate.py
│   │   └── prepare.py
│   ├── generators/                   # Synthetic data models (Marginal, Copula, CTGAN, TVAE)
│   ├── evaluation/                   # Fidelity, utility, and privacy evaluation modules
│   ├── stages/                       # Thin CLI entry points invoked by DVC
│   │   ├── download.py
│   │   ├── validate.py
│   │   └── prepare.py
│   └── tracking/                     # MLflow experiment tracking utilities
│       └── mlflow_utils.py
├── tests/                            # Test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_mlflow_utils.py
│   ├── test_split.py
│   └── test_validation.py
├── dvc.yaml                          # DVC pipeline stage definitions
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

# Execute pytest suite
make test
```

### 3. Reproduce the Data Pipeline Stages (via DVC)
```bash
# Reproduce pipeline up to the prepare stage
dvc repro prepare

# View pipeline DAG
dvc dag

# Inspect logged metrics
dvc metrics show
```

### 4. MLflow Tracking UI
```bash
# Launch MLflow tracking server locally on port 5000
make ui
```

