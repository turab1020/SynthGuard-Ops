import numpy as np
import pandas as pd
import pytest
from conftest import make_raw

from synthguard.data.prepare import clean
from synthguard.evaluation.privacy import (
    evaluate_privacy,
    exact_match_rate,
    fit_encoder,
    nearest_distances,
)


@pytest.fixture
def split():
    df = clean(make_raw(n=1200, seed=1)).drop_duplicates().reset_index(drop=True)
    return df.iloc[:800].reset_index(drop=True), df.iloc[800:].reset_index(drop=True)


@pytest.fixture
def unrelated():
    """Synthetic data that never saw train: drawn from the same process with another seed."""
    return clean(make_raw(n=600, seed=99))


def test_copy_of_train_is_flagged(split):
    train, holdout = split
    m = evaluate_privacy(train.copy(), train, holdout, sample_size=300, seed=0)
    assert m["exact_match_rate"] == pytest.approx(1.0)
    assert m["dcr_median_synth_train"] == pytest.approx(0.0)
    assert m["dcr_ratio"] == pytest.approx(0.0)
    assert m["mia_auc"] > 0.95
    assert m["mia_advantage"] > 0.9


def test_unrelated_data_is_not_flagged(split, unrelated):
    train, holdout = split
    m = evaluate_privacy(unrelated, train, holdout, sample_size=300, seed=0)
    assert m["exact_match_rate"] < 0.01
    assert m["mia_auc"] == pytest.approx(0.5, abs=0.07)
    assert m["dcr_ratio"] == pytest.approx(1.0, abs=0.15)


def test_partial_leak_sits_in_between(split, unrelated):
    train, holdout = split
    leaky = pd.concat([train.iloc[:300], unrelated.iloc[:300]], ignore_index=True)
    m = evaluate_privacy(leaky, train, holdout, sample_size=300, seed=0)
    assert m["exact_match_rate"] == pytest.approx(0.5, abs=0.01)
    assert 0.6 < m["mia_auc"] < 0.95


def test_exact_match_known_answer():
    train = clean(make_raw(n=50, seed=3))
    other = clean(make_raw(n=50, seed=4))
    synth = pd.concat([train.iloc[:10], other.iloc[:30]], ignore_index=True)
    assert exact_match_rate(synth, train) == pytest.approx(10 / 40)


def test_nearest_distance_known_answer():
    train = clean(make_raw(n=50, seed=5))
    enc = fit_encoder(train)
    moved = train.iloc[[0]].copy()
    lo, hi = train["age"].min(), train["age"].max()
    moved["age"] = lo if moved["age"].iloc[0] != lo else hi  # change one numeric value only
    expected = abs(moved["age"].iloc[0] - train["age"].iloc[0]) / (hi - lo)
    d = nearest_distances(enc, moved, train)[0]
    assert d <= expected + 1e-9  # nearest row is at most this far (the original row)
    assert nearest_distances(enc, train.iloc[[0]], train)[0] == pytest.approx(0.0)


def test_mia_is_seeded(split, unrelated):
    train, holdout = split
    a = evaluate_privacy(unrelated, train, holdout, sample_size=200, seed=7)
    b = evaluate_privacy(unrelated, train, holdout, sample_size=200, seed=7)
    assert a == b
    assert np.isfinite(list(a.values())).all()
