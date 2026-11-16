VENV ?= .venv
PY := $(VENV)/bin/python
BIN := $(VENV)/bin

.PHONY: setup lint format test reproduce smoke ui lock

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
	$(BIN)/pytest

reproduce:
	$(BIN)/dvc repro
	$(BIN)/dvc metrics show

smoke:
	SYNTHGUARD_SMOKE=1 $(BIN)/dvc repro

ui:
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

lock:
	$(PY) -m pip freeze --exclude-editable > requirements.lock
