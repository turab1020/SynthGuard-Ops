"""DVC stage: score every candidate for fidelity, utility and privacy; pick the winner.

Metrics are logged into each candidate's existing MLflow run (run ids from the baselines and
generate stages); TRTR metrics go into the `trtr_reference` run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

# macOS only: torch (pulled in by sdmetrics via fidelity.py) and XGBoost each ship an OpenMP
# runtime. Loading XGBoost's first lets both coexist in this stage, which never trains torch
# models; the other order segfaults. No effect on Linux. Must stay above the synthguard imports.
import xgboost  # noqa: F401
from mlflow.exceptions import MlflowException

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.data.schema import COLUMNS
from synthguard.evaluation.fidelity import evaluate_fidelity
from synthguard.evaluation.privacy import evaluate_privacy
from synthguard.evaluation.score import score_candidate, select
from synthguard.evaluation.utility import evaluate_trtr, evaluate_tstr
from synthguard.generators.runner import SYNTH_DIR, load_processed, training_frame
from synthguard.tracking.mlflow_utils import setup_mlflow, start_candidate_run

STAGE = "evaluate"
METRICS_DIR = PROJECT_ROOT / "metrics"
REPORT_DIR = PROJECT_ROOT / "reports" / "eval"
EVAL_PATH = METRICS_DIR / "evaluation.json"
CANDIDATES = {
    "marginal": "baseline",
    "gaussian_copula": "baseline",
    "ctgan": "primary",
    "tvae": "primary",
}
TABLE_COLUMNS = [
    "selection_score", "privacy_gate", "quality_score", "column_shapes", "column_pair_trends",
    "utility_ratio", "tstr_rf_roc_auc", "tstr_xgb_roc_auc", "mia_auc", "mia_advantage",
    "exact_match_rate", "dcr_ratio", "fit_seconds",
]  # fmt: skip


def load_json(name: str) -> dict:
    return json.loads((METRICS_DIR / name).read_text())


def resume_run(name: str, role: str, run_ids: dict):
    """Resume the candidate's training run; if it isn't in this tracking store, start a new one."""
    run_id = run_ids.get(name)
    if run_id:
        setup_mlflow()
        try:
            mlflow.get_run(run_id)
            return start_candidate_run(name, role=role, stage=STAGE, run_id=run_id)
        except MlflowException:
            print(f"WARNING: run {run_id} for {name} not found in this MLflow store; new run")
    return start_candidate_run(name, role=role, stage=STAGE)


def evaluate_candidate(
    synth: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    metadata: dict,
    trtr: dict,
    params: dict,
) -> tuple[dict, dict]:
    """All metrics for one synthetic dataset -> (flat metrics, fidelity details)."""
    ev = params["evaluation"]
    fidelity, details = evaluate_fidelity(train, synth, metadata)
    utility = evaluate_tstr(synth, holdout, trtr, ev["utility"], params["seed"])
    privacy = evaluate_privacy(
        synth, train, holdout, ev["privacy"]["mia_sample_size"], params["seed"]
    )
    return {**fidelity, **utility, **privacy}, details


def write_table(candidates: dict, selection: dict) -> pd.DataFrame:
    table = pd.DataFrame(candidates).T.reindex(columns=TABLE_COLUMNS)
    table = table.loc[selection["ranking"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "comparison.csv", index_label="generator")
    fmt = table.copy()
    for c in fmt.columns:
        if c != "privacy_gate":
            fmt[c] = fmt[c].map(lambda v: f"{v:.4f}")
    lines = [
        "| generator | " + " | ".join(fmt.columns) + " |",
        "|" + "---|" * (len(fmt.columns) + 1),
        *[f"| {g} | " + " | ".join(row) + " |" for g, row in fmt.iterrows()],
        "",
        f"Selected: **{selection['generator']}** (privacy gate {selection['privacy_gate']}).",
    ]
    (REPORT_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    return table


def main() -> None:
    params = load_params()
    train, holdout, metadata = load_processed()
    fit_df = training_frame(train, params)  # exactly the rows the generators were trained on
    run_ids = {**load_json("runs_baselines.json"), **load_json("runs_primary.json")}
    costs = {**load_json("baselines.json"), **load_json("primary.json")}

    trtr = evaluate_trtr(fit_df, holdout, params["evaluation"]["utility"], params["seed"])
    with resume_run("trtr_reference", "reference", run_ids):
        mlflow.log_metrics(trtr)
    print("trtr:", {k: round(v, 4) for k, v in trtr.items()}, flush=True)

    candidates = {}
    for name, role in CANDIDATES.items():
        synth = pd.read_csv(SYNTH_DIR / f"{name}.csv")[COLUMNS]
        metrics, details = evaluate_candidate(synth, fit_df, holdout, metadata, trtr, params)
        scored = score_candidate(metrics, params["selection"])
        scored["fit_seconds"] = costs.get(name, {}).get("fit_seconds", float("nan"))
        candidates[name] = scored
        with resume_run(name, role, run_ids):
            mlflow.log_metrics({k: v for k, v in scored.items() if isinstance(v, float)})
            mlflow.set_tag("privacy_gate", scored["privacy_gate"])
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / f"{name}_fidelity_report.json"
                p.write_text(json.dumps(details, indent=2, default=str))
                mlflow.log_artifact(str(p), artifact_path="reports")
        print(
            f"{name}: score={scored['selection_score']:.4f} gate={scored['privacy_gate']}",
            flush=True,
        )

    selection = select(candidates)
    write_table(candidates, selection)
    EVAL_PATH.write_text(
        json.dumps({"trtr": trtr, "candidates": candidates, "selection": selection}, indent=2)
    )
    print(f"selected: {selection['generator']} (gate {selection['privacy_gate']})")


if __name__ == "__main__":
    main()
