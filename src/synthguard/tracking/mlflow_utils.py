"""MLflow helpers: tracking URI, candidate runs and reproducibility tags."""

from __future__ import annotations

import hashlib
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow

from synthguard.config import PROJECT_ROOT

EXPERIMENT_NAME = "synthguard-d1"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def get_tracking_uri() -> str:
    """MLFLOW_TRACKING_URI if set, else the project's local SQLite DB."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def setup_mlflow(experiment: str = EXPERIMENT_NAME) -> str:
    """Point MLflow at our tracking store and select the experiment. Returns experiment id."""
    mlflow.set_tracking_uri(get_tracking_uri())
    return mlflow.set_experiment(experiment).experiment_id


def file_md5(path: str | Path) -> str:
    """md5 of a file, or 'missing' if it does not exist."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Current HEAD commit, or 'unknown' outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def common_tags() -> dict[str, str]:
    """Tags linking a run to the exact code and data versions."""
    processed = PROJECT_ROOT / "data" / "processed"
    return {
        "git_commit": git_commit(),
        "data_train_md5": file_md5(processed / "train.csv"),
        "data_holdout_md5": file_md5(processed / "holdout.csv"),
        "dvc_lock_md5": file_md5(PROJECT_ROOT / "dvc.lock"),
    }


def log_common_tags() -> dict[str, str]:
    tags = common_tags()
    mlflow.set_tags(tags)
    return tags


@contextmanager
def start_candidate_run(generator: str, role: str, stage: str, run_id: str | None = None):
    """Start (or resume, if run_id is given) the MLflow run for one generator candidate."""
    setup_mlflow()
    kwargs = {"run_id": run_id} if run_id else {"run_name": generator}
    with mlflow.start_run(**kwargs) as run:
        if run_id is None:
            mlflow.set_tags({"generator": generator, "role": role, "stage": stage})
            log_common_tags()
        yield run
