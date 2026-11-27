"""Champion vs challenger in the MLflow registry: score both on the fixed hold-out, then decide.

Rule (same as the evaluation gate): the challenger is promoted if its hold-out ROC-AUC is at most
`tolerance` below the champion's and its source generator passed the privacy gate. Promotion moves
`champion` to the challenger and keeps the old champion as `previous` (for rollback); otherwise the
champion stays and the challenger is tagged as rejected. The same applies to the generator aliases.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from synthguard.serving.classifier import evaluate
from synthguard.serving.gate import check
from synthguard.tracking.mlflow_utils import setup_mlflow

CLASSIFIER = "synthguard-classifier"
GENERATOR = "synthguard-generator"
CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"
METRICS = ("roc_auc", "f1", "accuracy", "precision", "recall")


def version_for(client: MlflowClient, name: str, alias: str):
    """The model version behind `name@alias`, or None if the alias does not exist."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def score(client: MlflowClient, alias: str, holdout: pd.DataFrame) -> dict:
    """Load synthguard-classifier@alias and score it on the hold-out."""
    mv = version_for(client, CLASSIFIER, alias)
    if mv is None:
        raise LookupError(f"no {CLASSIFIER}@{alias} in the registry")
    pipeline = mlflow.sklearn.load_model(f"models:/{CLASSIFIER}@{alias}")
    return {"alias": alias, "version": str(mv.version), **evaluate(pipeline, holdout)}


def challenger_privacy(client: MlflowClient) -> str:
    """Privacy gate of the generator behind the challenger ('unknown' blocks promotion)."""
    gen = version_for(client, GENERATOR, CHALLENGER)
    return gen.tags.get("privacy_gate", "unknown") if gen is not None else "unknown"


def decide(champion: dict, challenger: dict, tolerance: float, privacy_gate: str) -> dict:
    result = check(challenger, champion, tolerance, privacy_gate)
    return {
        "promote": result["passed"],
        "reasons": result["reasons"],
        "roc_auc_delta": result["roc_auc_delta"],
        "tolerance": tolerance,
        "privacy_gate": privacy_gate,
        "champion": champion,
        "challenger": challenger,
    }


def _move(client: MlflowClient, name: str) -> None:
    old, new = version_for(client, name, CHAMPION), version_for(client, name, CHALLENGER)
    if new is None:
        return
    if old is not None:
        client.set_registered_model_alias(name, PREVIOUS, old.version)
        client.set_model_version_tag(name, old.version, "promotion", "replaced")
    client.set_registered_model_alias(name, CHAMPION, new.version)
    client.delete_registered_model_alias(name, CHALLENGER)
    client.set_model_version_tag(name, new.version, "promotion", "promoted")


def apply(client: MlflowClient, decision: dict) -> None:
    """Carry out the decision in the registry (aliases + tags)."""
    version = decision["challenger"]["version"]
    delta = f"{decision['roc_auc_delta']:+.6f}"
    client.set_model_version_tag(CLASSIFIER, version, "promotion_roc_auc_delta", delta)
    if decision["promote"]:
        _move(client, CLASSIFIER)
        _move(client, GENERATOR)
        return
    client.set_model_version_tag(CLASSIFIER, version, "promotion", "rejected")
    client.set_model_version_tag(CLASSIFIER, version, "promotion_reasons",
                                 "; ".join(decision["reasons"]))  # fmt: skip


def seed_champion(export_dir: str | Path, benchmark: dict) -> str:
    """Register an exported classifier (deploy/model) as synthguard-classifier@champion.

    Used where no registry exists yet (e.g. a CI retrain run): the deployed model becomes the
    champion the challenger is compared against.
    """
    setup_mlflow()
    export_dir = Path(export_dir)
    pipeline = joblib.load(export_dir / "classifier.joblib")
    with mlflow.start_run(run_name="deployed-champion") as run:
        mlflow.set_tags({"role": "serving", "source": "deploy/model"})
        mlflow.log_metrics({f"holdout_{k}": benchmark[k] for k in METRICS if k in benchmark})
        info = mlflow.sklearn.log_model(
            pipeline,
            name="classifier_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    mv = mlflow.register_model(info.model_uri, CLASSIFIER)
    client = MlflowClient()
    client.set_model_version_tag(CLASSIFIER, mv.version, "source", "deploy/model")
    client.set_model_version_tag(CLASSIFIER, mv.version, "deployed_version",
                                 str(benchmark.get("version", "?")))  # fmt: skip
    client.set_model_version_tag(CLASSIFIER, mv.version, "source_run", run.info.run_id)
    client.set_registered_model_alias(CLASSIFIER, CHAMPION, mv.version)
    return str(mv.version)


def benchmark_for(decision: dict, trained_on: str | None) -> dict:
    """The new production benchmark (metrics/classifier.json) after a promotion."""
    c = decision["challenger"]
    return {"model_name": CLASSIFIER, "alias": CHAMPION, "version": c["version"],
            "trained_on": trained_on, **{k: c[k] for k in METRICS}}  # fmt: skip


def report_markdown(decision: dict) -> str:
    champ, chall = decision["champion"], decision["challenger"]
    verdict = "PROMOTED" if decision["promote"] else "KEPT CHAMPION (challenger rejected)"
    lines = [
        f"## Champion vs challenger: {verdict}",
        "",
        "Both scored on the same fixed real hold-out (positive class `>50K`).",
        "",
        f"| Metric | Champion (v{champ['version']}) | Challenger (v{chall['version']}) | Δ |",
        "|---|---|---|---|",
    ]
    for k in METRICS:
        lines.append(f"| {k} | {champ[k]:.4f} | {chall[k]:.4f} | {chall[k] - champ[k]:+.4f} |")
    lines += [
        "",
        f"Rule: challenger ROC-AUC ≥ champion − {decision['tolerance']}, and the challenger's "
        f"generator passed the privacy gate (it is `{decision['privacy_gate']}`).",
    ]
    if decision["promote"]:
        lines.append(f"\n`champion` → v{chall['version']}; v{champ['version']} kept as `previous` "
                     "(rollback target).")  # fmt: skip
    else:
        lines += ["", "Reasons:"] + [f"- {r}" for r in decision["reasons"]]
        lines.append(f"\n`champion` stays v{champ['version']}.")
    return "\n".join(lines) + "\n"
