VENV ?= .venv
PY ?= $(VENV)/bin/python
BIN ?= $(VENV)/bin

# macOS: torch bundles its own OpenMP (libomp). Make XGBoost load that same copy so only one
# OpenMP runtime is in the process; two copies crash when torch and XGBoost run together.
ifeq ($(shell uname),Darwin)
OMP_ENV := DYLD_LIBRARY_PATH=$(CURDIR)/$(VENV)/lib/python3.11/site-packages/torch/lib
endif

# Linux x86_64: use one numeric code path so results do not depend on which CPU model runs them
# (measured: bit-identical across AMD EPYC 7763, 9V74 and 9V45; Intel Xeon still differs slightly).
ifeq ($(shell uname -s)-$(shell uname -m),Linux-x86_64)
REPRO_ENV := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_CBWR=AVX2,STRICT \
	OPENBLAS_CORETYPE=Haswell \
	NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
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

# DVC runs each stage as `python -m ...`: put the project's interpreter first on PATH.
reproduce:
	PATH="$(abspath $(BIN)):$$PATH" $(OMP_ENV) $(REPRO_ENV) $(BIN)/dvc repro
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
