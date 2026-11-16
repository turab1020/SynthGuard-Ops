# T02 — Repo scaffold

**What was built:** `src/synthguard/` package (src layout, empty sub-packages for data,
generators, evaluation, tracking, stages), `pyproject.toml` (deps + ruff + pytest config),
`Makefile`, ruff pre-commit hooks, `params.yaml` (every tunable value), `config.py`, and
`tests/test_config.py`. `requirements.lock` is a `pip freeze` of the installed env (262 pins,
incl. sdv 2.2.0, sdmetrics 0.33.0, mlflow 3.18.0, dvc 3.67.1, pandera 0.36.0, torch 2.15.0).

**Why this design**
- *src layout* forces tests to import the installed package, catching packaging bugs early.
- *params.yaml* is the single source of tunables; DVC tracks it so a change re-runs only affected stages.
- `load_params()` applies smoke mode (`SYNTHGUARD_SMOKE=1` → 1 epoch, `smoke_rows=1000`) in one place.
- `set_seed()` seeds `random`, NumPy and torch so runs are repeatable.
- `pyproject.toml` lists loose deps (what we need); `requirements.lock` pins exact versions (what we ran).

**Run / verify**
```bash
make setup      # creates .venv, installs package + dev tools, installs git hook
make lint test  # ruff clean, 4 tests pass
make lock       # regenerate requirements.lock after changing deps
```

**Teacher questions**
- *Why both pyproject deps and a lock file?* pyproject says what's compatible; the lock reproduces
  the exact environment (used by Docker in T12).
- *How does smoke mode work?* One env var flips `load_params()` to tiny settings, so CI and tests
  exercise the full pipeline in minutes.
- *Why seed torch too?* CTGAN/TVAE are torch networks; unseeded weight init changes results run to run.
