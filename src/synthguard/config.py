"""Load params.yaml, apply smoke-mode overrides, and seed every RNG."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
SMOKE_ENV = "SYNTHGUARD_SMOKE"
ALIAS_ENV = "SYNTHGUARD_REGISTER_ALIAS"


def is_smoke() -> bool:
    """True when SYNTHGUARD_SMOKE is set to a truthy value."""
    return os.environ.get(SMOKE_ENV, "").strip().lower() in {"1", "true", "yes"}


def register_alias() -> str:
    """Registry alias the register/train_classifier stages assign: `champion` (default), or e.g.
    `challenger` in a retraining run, so a retrained model never replaces the champion directly."""
    return os.environ.get(ALIAS_ENV, "").strip() or "champion"


def load_params(path: str | Path = PARAMS_PATH, smoke: bool | None = None) -> dict[str, Any]:
    """Read params.yaml. In smoke mode, cap epochs and add `smoke_rows` for fast runs."""
    with open(path) as f:
        params = yaml.safe_load(f)
    params = copy.deepcopy(params)
    smoke = is_smoke() if smoke is None else smoke
    params["is_smoke"] = smoke
    if smoke:
        cfg = params.get("smoke", {})
        for gen in params.get("generators", {}).values():
            if isinstance(gen, dict) and "epochs" in gen:
                gen["epochs"] = cfg.get("epochs", 1)
        params["smoke_rows"] = cfg.get("rows", 1000)
    return params


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
