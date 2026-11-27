"""DVC stage: train the serving classifier on the champion generator's synthetic data.

Registers it as `synthguard-classifier` with alias `champion` and writes the production benchmark
(metrics/classifier.json): the classifier's scores on the fixed real hold-out. In a retraining run
(SYNTHGUARD_REGISTER_ALIAS=challenger) the alias is `challenger` and the scores go to
metrics/classifier_challenger.json; the production benchmark is left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from synthguard.config import PROJECT_ROOT, load_params, register_alias, set_seed
from synthguard.data.schema import COLUMNS
from synthguard.generators.runner import SYNTH_DIR, load_processed
from synthguard.serving.classifier import evaluate, train
from synthguard.tracking.mlflow_utils import common_tags, start_candidate_run

MODEL_NAME = "synthguard-classifier"
BENCHMARK = PROJECT_ROOT / "metrics" / "classifier.json"


def scores_path(alias: str) -> Path:
    """metrics/classifier.json for the champion (the production benchmark), else
    metrics/classifier_<alias>.json."""
    return BENCHMARK if alias == "champion" else BENCHMARK.with_name(f"classifier_{alias}.json")


def main() -> dict:
    params = load_params()
    alias = register_alias()
    seed = params["seed"]
    set_seed(seed)
    registered = json.loads((PROJECT_ROOT / "metrics" / "registered.json").read_text())
    generator = registered["generator"]
    synthetic = pd.read_csv(SYNTH_DIR / f"{generator}.csv")[COLUMNS]
    _, holdout, _ = load_processed()

    pipeline = train(synthetic, params["classifier"], seed)
    scores = evaluate(pipeline, holdout)

    with start_candidate_run("classifier", role="serving", stage="train_classifier") as run:
        mlflow.log_params({f"xgb_{k}": v for k, v in params["classifier"].items()})
        mlflow.log_params({"seed": seed, "trained_on": f"synthetic:{generator}",
                           "train_rows": len(synthetic)})  # fmt: skip
        mlflow.log_metrics({f"holdout_{k}": v for k, v in scores.items()})
        mlflow.set_tags({"source_generator": generator,
                         "source_generator_version": registered["version"]})  # fmt: skip
        example = synthetic.drop(columns="income").head(3)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="classifier_model",
            input_example=example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id
    version = mlflow.register_model(info.model_uri, MODEL_NAME)
    client = MlflowClient()
    tags = {"source_generator": generator, "holdout_roc_auc": f"{scores['roc_auc']:.6f}",
            "git_commit": common_tags()["git_commit"]}  # fmt: skip
    for k, v in tags.items():
        client.set_model_version_tag(MODEL_NAME, version.version, k, v)
    client.set_registered_model_alias(MODEL_NAME, alias, version.version)

    benchmark = {"model_name": MODEL_NAME, "alias": alias, "version": str(version.version),
                 "run_id": run_id, "trained_on": f"synthetic:{generator}", **scores}  # fmt: skip
    scores_path(alias).write_text(json.dumps(benchmark, indent=2))
    print(f"classifier v{version.version} @{alias}: hold-out ROC-AUC {scores['roc_auc']:.4f} "
          f"(trained on {generator} synthetic data)")  # fmt: skip
    return benchmark


if __name__ == "__main__":
    main()
