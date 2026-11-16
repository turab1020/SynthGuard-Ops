# Project Deliverable 1: Reproducible Training & Experimentation Pipeline

Course: MLOps (FAST NU Lahore, BSCS 2023) · 100 points · **Due Nov 22, 11:59 PM**

Before building a frontend or setting up cloud autoscaling, your team must prove that your core machine learning pipeline is solvable, leak-free, and reproducible. Submit your GitHub repository containing the following components:

**Versioned Data & Validation:** Track your raw and processed datasets using DVC (or an equivalent tool). Include automated schema and quality checks that catch corrupted or missing inputs before training starts.

**Leakage-Free Splits & Baselines:** Enforce strict splits suited to your problem (such as time-ordered splits for financial data or speaker-disjoint splits for audio). Log a simple rule-based or classical baseline before training your primary model.

**Experiment Tracking & Registry:** Log all runs in MLflow (or Weights & Biases), capturing hyperparameters, dataset versions, and evaluation metrics. Register your top-performing model as a versioned artifact.

**Reproducible Execution:** Package your training pipeline in a script or Dockerfile so we can pull your repository, run one command, and reproduce your registered metrics.

**What to Submit:** Your GitHub repository link, DVC configuration files, and screenshots or exported logs of your MLflow runs and registered model.

Any related query must be dropped as a private comment to this assignment.

## How SynthGuard-Ops maps to these requirements

| Requirement | Where it is met | Tasks |
|---|---|---|
| Versioned data | DVC stages `download`, `prepare`, `baselines`, `generate` | T03, T04, T06 |
| Schema & quality checks before training | pandera `validate` stage + processed-data check | T05 |
| Leakage-free split | dedupe → stratified 70/30; generators see train only; overlap test | T06, T20 |
| Baseline before primary model | `baselines` stage (marginal, Gaussian Copula, TRTR) precedes `generate` | T08 |
| MLflow: hyperparameters, data versions, metrics | one run per candidate with params, md5 tags, metrics | T03, T08–T14 |
| Register top model | `register` stage → `synthguard-generator@champion` | T16 |
| One command reproduces registered metrics | `docker compose run --rm pipeline` | T18, T22 |
| Submission evidence | `submission/` exports + screenshots | T24, T26 |
