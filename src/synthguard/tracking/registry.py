"""Register a chosen generator in the MLflow Model Registry as `synthguard-generator`."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from synthguard.config import PROJECT_ROOT
from synthguard.tracking.mlflow_utils import setup_mlflow
from synthguard.tracking.pyfunc_model import ARTIFACT_KEY, GeneratorModel, signature

MODEL_NAME = "synthguard-generator"
ALIAS = "champion"
MODEL_ARTIFACT = "generator_model"


def register_generator(
    pickle_path: str | Path,
    run_id: str,
    version_tags: dict[str, str],
    model_name: str = MODEL_NAME,
    alias: str = ALIAS,
) -> str:
    """Log the pyfunc model into the candidate's existing run, register it, tag it, alias it.

    Returns the new model version number (as a string).
    """
    setup_mlflow()
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT,
            python_model=GeneratorModel(),
            artifacts={ARTIFACT_KEY: str(pickle_path)},
            signature=signature(),
            input_example=pd.DataFrame({"n_rows": [5]}),
            code_paths=[str(PROJECT_ROOT / "src" / "synthguard")],
        )
    version = mlflow.register_model(info.model_uri, model_name)
    client = MlflowClient()
    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, version.version, key, str(value))
    client.set_registered_model_alias(model_name, alias, version.version)
    return str(version.version)


def load_champion(model_name: str = MODEL_NAME, alias: str = ALIAS):
    setup_mlflow()
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
