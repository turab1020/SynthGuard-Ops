import json

import pandas as pd
import pytest

from synthguard.data.prepare import clean, sdv_metadata
from synthguard.evaluation.fidelity import corr_diff, evaluate_fidelity, ks_statistic, tvd


def shuffle_columns(df, seed=0):
    """Permute every column independently: keeps each marginal, destroys all relationships."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return df.apply(lambda s: s.to_numpy()[rng.permutation(len(s))]).reset_index(drop=True)


@pytest.fixture
def real(raw_df):
    return clean(raw_df)


def test_identical_data_scores_one(real):
    metrics, details = evaluate_fidelity(real, real.copy(), sdv_metadata())
    assert metrics["quality_score"] == pytest.approx(1.0, abs=0.01)
    assert metrics["column_shapes"] == pytest.approx(1.0, abs=0.01)
    assert metrics["corr_diff"] == pytest.approx(0.0, abs=1e-9)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in details["per_column"].values())
    json.dumps(details)  # report must be JSON-serialisable


def test_independent_columns_keeps_shapes_but_loses_pairs(real):
    synth = shuffle_columns(real)
    m, _ = evaluate_fidelity(real, synth, sdv_metadata())
    assert set(m) == {"quality_score", "column_shapes", "column_pair_trends", "corr_diff"}
    assert m["column_shapes"] > m["column_pair_trends"] - 0.2
    assert 0 <= m["quality_score"] <= 1


def test_known_answers():
    a = pd.Series(["x", "x", "y", "y"])
    b = pd.Series(["x", "x", "x", "x"])
    assert tvd(a, b) == pytest.approx(0.5)
    assert ks_statistic(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) == 0
    df = pd.DataFrame({"u": [1, 2, 3, 4], "v": [2, 4, 6, 8]})
    flipped = df.assign(v=df["v"][::-1].to_numpy())
    assert corr_diff(df, flipped, ["u", "v"]) == pytest.approx(2.0)
