VENV ?= .venv
PY ?= $(VENV)/bin/python
BIN ?= $(VENV)/bin

# macOS: torch bundles its own OpenMP (libomp). Make XGBoost load that same copy so only one
# OpenMP runtime is in the process; two copies crash when torch and XGBoost run together.
ifeq ($(shell uname),Darwin)
OMP_ENV := DYLD_LIBRARY_PATH=$(CURDIR)/$(VENV)/lib/python3.11/site-packages/torch/lib
endif

.PHONY: setup lint format test reproduce smoke ui lock export serve serve-docker rollback \
	rollback-registry

setup:
	python3.11 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"
	$(BIN)/pre-commit install

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(OMP_ENV) $(BIN)/pytest

# DVC runs each stage as `python -m ...`: put the project's interpreter first on PATH.
reproduce:
	PATH="$(abspath $(BIN)):$$PATH" $(OMP_ENV) $(BIN)/dvc repro
	$(BIN)/dvc metrics show

smoke:
	PATH="$(abspath $(BIN)):$$PATH" $(OMP_ENV) SYNTHGUARD_SMOKE=1 $(BIN)/dvc repro

ui:
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

lock:
	$(PY) -m pip freeze --exclude-editable > requirements.lock

# MLflow evidence for submission: submission/mlflow_runs.csv + registered_models.json
export:
	$(PY) -m synthguard.tracking.export

# Local API with /generate on (needs torch + SDV, i.e. the dev venv): http://localhost:8000/docs
serve:
	SYNTHGUARD_ENABLE_GENERATE=1 $(OMP_ENV) $(BIN)/uvicorn synthguard.serving.app:app --port 8000

# The Render image, locally, with Render's memory limit
serve-docker:
	docker build -f Dockerfile.serve -t synthguard-serve .
	docker run --rm --memory=512m -e PORT=8000 -p 8000:8000 synthguard-serve

# Roll back the served model to the one before the last promotion (TO=<git ref> to pick one).
# Restores deploy/model + metrics/classifier.json; commit on a rollback/* branch and merge the PR:
# deploy.yml re-runs the gate and redeploys. In GitHub: Actions -> Rollback -> Run workflow.
rollback:
	$(PY) scripts/rollback.py $(if $(TO),--to $(TO))

# Same in the local MLflow registry: champion -> previous; then re-export deploy/model.
rollback-registry:
	$(OMP_ENV) $(PY) scripts/rollback.py --registry
	$(OMP_ENV) $(PY) scripts/export_models.py
