import numpy as np
import pandas as pd
import pytest

from synthguard.data.schema import CATEGORIES, COLUMNS


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Small valid raw-style frame (uses '?' for some missing categories)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(17, 91, n),
            "fnlwgt": rng.integers(10_000, 500_000, n),
            "education_num": rng.integers(1, 17, n),
            "capital_gain": rng.integers(0, 5_000, n),
            "capital_loss": rng.integers(0, 500, n),
            "hours_per_week": rng.integers(1, 100, n),
        }
    )
    for c, cats in CATEGORIES.items():
        df[c] = rng.choice(cats, n)
    df.loc[::25, "workclass"] = "?"
    df["income"] = np.where(np.arange(n) % 4 == 0, ">50K", "<=50K")
    return df[COLUMNS].astype({c: "int64" for c in df.select_dtypes("number").columns})


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()
