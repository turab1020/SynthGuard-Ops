VENV ?= .venv
PY ?= $(VENV)/bin/python
BIN ?= $(VENV)/bin

# macOS: torch bundles its own OpenMP (libomp). Make XGBoost load that same copy so only one
# OpenMP runtime is in the process; two copies crash when torch and XGBoost run together.
ifeq ($(shell uname),Darwin)
OMP_ENV := DYLD_LIBRARY_PATH=$(CURDIR)/$(VENV)/lib/python3.11/site-packages/torch/lib
endif

.PHONY: setup lint format test reproduce smoke ui lock export

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

reproduce:
	$(OMP_ENV) $(BIN)/dvc repro
	$(BIN)/dvc metrics show

smoke:
	$(OMP_ENV) SYNTHGUARD_SMOKE=1 $(BIN)/dvc repro

ui:
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

lock:
	$(PY) -m pip freeze --exclude-editable > requirements.lock

# MLflow evidence for submission: submission/mlflow_runs.csv + registered_models.json
export:
	$(PY) -m synthguard.tracking.export
