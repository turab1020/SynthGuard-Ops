import random

import numpy as np

from synthguard.config import load_params, set_seed


def test_load_params_defaults():
    p = load_params(smoke=False)
    assert p["seed"] == 42
    assert p["data"]["target"] == "income"
    assert p["generators"]["ctgan"]["epochs"] == 100
    assert p["is_smoke"] is False


def test_smoke_overrides():
    p = load_params(smoke=True)
    assert p["generators"]["ctgan"]["epochs"] == 1
    assert p["generators"]["tvae"]["epochs"] == 1
    assert p["smoke_rows"] == 1000


def test_smoke_env(monkeypatch):
    monkeypatch.setenv("SYNTHGUARD_SMOKE", "1")
    assert load_params()["is_smoke"] is True


def test_set_seed_is_deterministic():
    set_seed(7)
    a = (random.random(), np.random.rand())
    set_seed(7)
    assert (random.random(), np.random.rand()) == a
