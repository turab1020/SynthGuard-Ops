"""DVC stage: train the primary generators (CTGAN, TVAE).

Depends on metrics/baselines.json, so DVC always runs the baselines stage first.
"""

import json

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.generators.runner import (
    load_processed,
    n_synthetic,
    train_candidate,
    training_frame,
)
from synthguard.generators.sdv_models import CTGANGenerator, TVAEGenerator

STAGE = "generate"
BASELINES_PATH = PROJECT_ROOT / "metrics" / "baselines.json"
PRIMARY_PATH = PROJECT_ROOT / "metrics" / "primary.json"
RUNS_PATH = PROJECT_ROOT / "metrics" / "runs_primary.json"


def main() -> None:
    if not BASELINES_PATH.exists():
        raise SystemExit("metrics/baselines.json missing: run the baselines stage first")
    params = load_params()
    seed = params["seed"]
    train, holdout, metadata = load_processed()
    fit_df = training_frame(train, params)
    n = n_synthetic(holdout, params)
    g = params["generators"]

    runs, summary = {}, {}
    for gen in (
        CTGANGenerator(metadata, seed=seed, **g["ctgan"]),
        TVAEGenerator(metadata, seed=seed, **g["tvae"]),
    ):
        result = train_candidate(gen, fit_df, n, role="primary", stage=STAGE, params=params)
        runs[gen.name] = result.pop("run_id")
        summary[gen.name] = result
        print(f"{gen.name}: {result}", flush=True)

    PRIMARY_PATH.write_text(json.dumps(summary, indent=2))
    RUNS_PATH.write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
