"""DVC stage: TRTR reference run + the two baseline generators (marginal, Gaussian Copula).

Must run before the CTGAN/TVAE `generate` stage, which depends on metrics/baselines.json.
"""

import json

import mlflow

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.generators.marginal import MarginalSampler
from synthguard.generators.runner import (
    load_processed,
    n_synthetic,
    train_candidate,
    training_frame,
)
from synthguard.generators.sdv_models import GaussianCopulaGenerator
from synthguard.tracking.mlflow_utils import start_candidate_run

STAGE = "baselines"
BASELINES_PATH = PROJECT_ROOT / "metrics" / "baselines.json"
RUNS_PATH = PROJECT_ROOT / "metrics" / "runs_baselines.json"


def log_reference_run(params: dict, train_rows: int, holdout_rows: int) -> str:
    """TRTR reference: real train -> real hold-out. Its metrics are added by `evaluate`."""
    utility = params["evaluation"]["utility"]
    with start_candidate_run("trtr_reference", role="reference", stage=STAGE) as run:
        mlflow.log_params(
            {
                **{f"rf_{k}": v for k, v in utility["random_forest"].items()},
                **{f"xgb_{k}": v for k, v in utility["xgboost"].items()},
                "seed": params["seed"],
                "smoke": params["is_smoke"],
                "train_rows": train_rows,
                "holdout_rows": holdout_rows,
            }
        )
        return run.info.run_id


def main() -> None:
    params = load_params()
    seed = params["seed"]
    train, holdout, metadata = load_processed()
    fit_df = training_frame(train, params)
    n = n_synthetic(holdout, params)
    g = params["generators"]

    runs = {"trtr_reference": log_reference_run(params, len(fit_df), len(holdout))}
    summary = {}
    for gen in (
        MarginalSampler(seed=seed, **g["marginal"]),
        GaussianCopulaGenerator(metadata, seed=seed, **g["gaussian_copula"]),
    ):
        result = train_candidate(gen, fit_df, n, role="baseline", stage=STAGE, params=params)
        runs[gen.name] = result.pop("run_id")
        summary[gen.name] = result
        print(f"{gen.name}: {result}")

    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(summary, indent=2))
    RUNS_PATH.write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
