"""Export the @champion classifier from the MLflow registry for serving.

Writes deploy/model/classifier.joblib and deploy/model/meta.json (classifier and generator summary,
library versions). Render builds from the repo without an MLflow server, so these are committed.

usage: python scripts/export_models.py [--out deploy/model]
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas
import sklearn
import xgboost
from mlflow import MlflowClient

from synthguard.config import PROJECT_ROOT
from synthguard.tracking.mlflow_utils import setup_mlflow

CLASSIFIER = "synthguard-classifier"
GENERATOR = "synthguard-generator"
ALIAS = "champion"


def build_meta(client: MlflowClient) -> dict:
    clf = client.get_model_version_by_alias(CLASSIFIER, ALIAS)
    gen = client.get_model_version_by_alias(GENERATOR, ALIAS)
    benchmark = json.loads((PROJECT_ROOT / "metrics" / "classifier.json").read_text())
    evaluation = json.loads((PROJECT_ROOT / "metrics" / "evaluation.json").read_text())
    g = evaluation["candidates"][gen.tags["generator"]]
    keep = ("selection_score", "quality_score", "utility_ratio", "mia_auc", "exact_match_rate",
            "dcr_ratio", "privacy_gate")  # fmt: skip
    return {
        "classifier": {"name": CLASSIFIER, "version": str(clf.version), "run_id": clf.run_id,
                       "trained_on": benchmark.get("trained_on"),
                       "holdout": {k: benchmark[k] for k in ("roc_auc", "f1", "accuracy",
                                                            "precision", "recall")}},
        "generator": {"name": gen.tags["generator"], "version": str(gen.version),
                      **{k: g[k] for k in keep}},
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "xgboost": xgboost.__version__, "pandas": pandas.__version__},
        "git_commit": clf.tags.get("git_commit"),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "deploy" / "model"))
    out = Path(ap.parse_args(argv).out)
    setup_mlflow()
    pipeline = mlflow.sklearn.load_model(f"models:/{CLASSIFIER}@{ALIAS}")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "classifier.joblib")
    meta = build_meta(MlflowClient())
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"exported {CLASSIFIER} v{meta['classifier']['version']} -> {out}")
    return out


if __name__ == "__main__":
    main()
