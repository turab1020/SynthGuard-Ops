"""Export MLflow evidence for submission: all runs (CSV) and registered model versions (JSON).

Local details are removed so the files can be shared: artifact locations, the OS user name and
source-file paths are dropped, and any remaining absolute path to the project is made relative.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from synthguard.config import PROJECT_ROOT
from synthguard.tracking.mlflow_utils import EXPERIMENT_NAME, setup_mlflow

OUT_DIR = PROJECT_ROOT / "submission"
RUNS_CSV = OUT_DIR / "mlflow_runs.csv"
MODELS_JSON = OUT_DIR / "registered_models.json"
DROP_COLUMNS = ["artifact_uri", "tags.mlflow.user", "tags.mlflow.source.name"]
DROP_VERSION_FIELDS = ["source", "storage_location", "user_id", "run_link"]


def _relativize(value, root: str):
    if isinstance(value, str) and root in value:
        return value.replace(root, ".")
    return value


def export_runs(experiment: str = EXPERIMENT_NAME, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """One row per run: run info, params.*, metrics.*, tags.* (newest first)."""
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])
    runs = runs.drop(columns=[c for c in DROP_COLUMNS if c in runs.columns])
    for col in runs.columns:
        if runs[col].dtype == object:
            runs[col] = runs[col].map(lambda v: _relativize(v, str(root)))
    return runs


def export_registered_models(root: Path = PROJECT_ROOT) -> list[dict]:
    client = MlflowClient()
    out = []
    for rm in client.search_registered_models():
        versions = []
        for mv in client.search_model_versions(f"name = '{rm.name}'"):
            v = {
                "version": str(mv.version),
                "run_id": mv.run_id,
                "status": mv.status,
                "created": pd.to_datetime(mv.creation_timestamp, unit="ms", utc=True).isoformat(),
                "aliases": sorted(
                    a for a, ver in rm.aliases.items() if str(ver) == str(mv.version)
                ),
                "tags": dict(mv.tags),
            }
            versions.append({k: _relativize(val, str(root)) for k, val in v.items()})
        versions.sort(key=lambda v: int(v["version"]))
        out.append({"name": rm.name, "aliases": dict(rm.aliases), "versions": versions})
    return out


def main(out_dir: Path = OUT_DIR) -> None:
    setup_mlflow()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = export_runs()
    runs.to_csv(out_dir / RUNS_CSV.name, index=False)
    models = export_registered_models()
    (out_dir / MODELS_JSON.name).write_text(json.dumps(models, indent=2, default=str))
    print(f"{len(runs)} runs -> {out_dir / RUNS_CSV.name}")
    print(
        f"{sum(len(m['versions']) for m in models)} model versions -> {out_dir / MODELS_JSON.name}"
    )


if __name__ == "__main__":
    main()
