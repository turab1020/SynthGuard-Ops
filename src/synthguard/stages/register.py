"""DVC stage: register the generator chosen by `evaluate` as `synthguard-generator@champion`.

In a retraining run (SYNTHGUARD_REGISTER_ALIAS=challenger) the alias is `challenger` instead.

The decision (privacy gate, then highest selection score) is made once in the evaluate stage
and recorded in metrics/evaluation.json; this stage carries it out and records the result.
"""

from __future__ import annotations

import json
from pathlib import Path

from synthguard.config import PROJECT_ROOT, register_alias
from synthguard.tracking.mlflow_utils import common_tags
from synthguard.tracking.registry import MODEL_NAME, register_generator

METRICS_DIR = PROJECT_ROOT / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"
KEY_METRICS = [
    "selection_score", "quality_score", "utility_ratio", "tstr_rf_roc_auc",
    "mia_auc", "mia_advantage", "exact_match_rate", "dcr_ratio", "fit_seconds",
]  # fmt: skip


def plan_registration(evaluation: dict, run_ids: dict, tags: dict) -> dict:
    """What to register: the selected generator, its run, its version tags and key metrics."""
    selection = evaluation["selection"]
    name = selection["generator"]
    if name not in run_ids:
        raise KeyError(f"no MLflow run id recorded for selected generator {name!r}")
    metrics = evaluation["candidates"][name]
    return {
        "generator": name,
        "run_id": run_ids[name],
        "version_tags": {
            "generator": name,
            "selection_score": f"{selection['selection_score']:.6f}",
            "privacy_gate": selection["privacy_gate"],
            "git_commit": tags["git_commit"],
            "data_train_md5": tags["data_train_md5"],
        },
        "metrics": {k: metrics[k] for k in KEY_METRICS if k in metrics},
    }


def main(metrics_dir: Path = METRICS_DIR, models_dir: Path = MODELS_DIR) -> dict:
    evaluation = json.loads((metrics_dir / "evaluation.json").read_text())
    run_ids = {}
    for f in ("runs_baselines.json", "runs_primary.json"):
        run_ids.update(json.loads((metrics_dir / f).read_text()))
    plan = plan_registration(evaluation, run_ids, common_tags())
    alias = register_alias()
    version = register_generator(
        models_dir / f"{plan['generator']}.pkl", plan["run_id"], plan["version_tags"], alias=alias
    )
    registered = {
        "model_name": MODEL_NAME,
        "alias": alias,
        "version": version,
        "model_uri": f"models:/{MODEL_NAME}@{alias}",
        "generator": plan["generator"],
        "privacy_gate": plan["version_tags"]["privacy_gate"],
        "run_id": plan["run_id"],
        **plan["metrics"],
    }
    (metrics_dir / "registered.json").write_text(json.dumps(registered, indent=2))
    gate = registered["privacy_gate"]
    name = plan["generator"]
    print(f"registered {MODEL_NAME} v{version} @{alias} = {name} (privacy gate {gate})")
    return registered


if __name__ == "__main__":
    main()
