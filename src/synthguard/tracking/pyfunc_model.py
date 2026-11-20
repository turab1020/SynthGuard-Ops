"""MLflow pyfunc wrapper: a registered generator that samples synthetic rows on request.

Usage after registration:
    model = mlflow.pyfunc.load_model("models:/synthguard-generator@champion")
    rows = model.predict(pd.DataFrame({"n_rows": [10]}))
"""

from __future__ import annotations

import pickle

import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from synthguard.data.schema import COLUMNS, NUMERIC

ARTIFACT_KEY = "generator"


class GeneratorModel(mlflow.pyfunc.PythonModel):
    """Loads a pickled generator (any object with `sample(n) -> DataFrame`)."""

    def load_context(self, context) -> None:
        with open(context.artifacts[ARTIFACT_KEY], "rb") as f:
            self.generator = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if "n_rows" not in model_input.columns or len(model_input) == 0:
            raise ValueError("input must be a DataFrame with an `n_rows` column")
        n = int(model_input["n_rows"].iloc[0])
        if n <= 0:
            raise ValueError(f"n_rows must be positive, got {n}")
        return self.generator.sample(n)[COLUMNS].reset_index(drop=True)


def signature() -> ModelSignature:
    inputs = Schema([ColSpec("long", "n_rows")])
    outputs = Schema([ColSpec("long" if c in NUMERIC else "string", c) for c in COLUMNS])
    return ModelSignature(inputs=inputs, outputs=outputs)
