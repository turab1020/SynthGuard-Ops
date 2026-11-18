"""Train one candidate generator, save its outputs and log it as an MLflow run."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from synthguard.config import PROJECT_ROOT, set_seed
from synthguard.generators.base import Generator
from synthguard.tracking.mlflow_utils import start_candidate_run

PROCESSED = PROJECT_ROOT / "data" / "processed"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv(PROCESSED / "train.csv")
    holdout = pd.read_csv(PROCESSED / "holdout.csv")
    metadata = json.loads((PROCESSED / "metadata.json").read_text())
    return train, holdout, metadata


def training_frame(train: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Full train set, or a seeded subsample of `smoke_rows` in smoke mode."""
    if params.get("is_smoke"):
        n = min(params["smoke_rows"], len(train))
        return train.sample(n=n, random_state=params["seed"]).reset_index(drop=True)
    return train


def n_synthetic(holdout: pd.DataFrame, params: dict) -> int:
    """Synthetic size = hold-out size (capped at smoke_rows in smoke mode)."""
    n = len(holdout)
    return min(n, params["smoke_rows"]) if params.get("is_smoke") else n


def log_artifacts(sample: pd.DataFrame) -> None:
    for name in ("params.yaml", "dvc.lock"):
        if (PROJECT_ROOT / name).exists():
            mlflow.log_artifact(str(PROJECT_ROOT / name), artifact_path="config")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "synthetic_sample.csv"
        sample.head(100).to_csv(p, index=False)
        mlflow.log_artifact(str(p), artifact_path="samples")


def train_candidate(
    gen: Generator, train: pd.DataFrame, n_rows: int, role: str, stage: str, params: dict
) -> dict:
    """Fit, sample, save model + CSV, log the run. Returns a summary incl. run_id."""
    set_seed(gen.seed)
    with start_candidate_run(gen.name, role=role, stage=stage) as run:
        mlflow.log_params(
            {
                **{f"gen_{k}": v for k, v in gen.params.items()},
                "seed": gen.seed,
                "smoke": params.get("is_smoke", False),
                "train_rows": len(train),
                "synthetic_rows": n_rows,
            }
        )
        gen.fit(train)
        synth = gen.sample(n_rows)
        gen.save(MODELS_DIR / f"{gen.name}.pkl")
        SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        synth.to_csv(SYNTH_DIR / f"{gen.name}.csv", index=False)
        costs = {k: round(v, 4) for k, v in gen.costs.items()}
        mlflow.log_metrics(costs)
        log_artifacts(synth)
        return {"run_id": run.info.run_id, "rows": n_rows, **costs}
