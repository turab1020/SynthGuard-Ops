"""Known-answer tests for the fidelity, utility and privacy metrics.

Each test builds an input whose correct metric value is known exactly (or its direction is
certain), so a wrong formula, a sign error or a mixed-up argument makes it fail.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_raw

from synthguard.data.prepare import clean, sdv_metadata
from synthguard.data.schema import CATEGORICAL, NUMERIC
from synthguard.evaluation.fidelity import corr_diff, evaluate_fidelity, ks_statistic, tvd
from synthguard.evaluation.privacy import evaluate_privacy, exact_match_rate
from synthguard.evaluation.utility import evaluate_trtr, evaluate_tstr

PARAMS = {
    "random_forest": {"n_estimators": 20, "max_depth": None},
    "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
}


@pytest.fixture
def real():
    return clean(make_raw(n=600, seed=11)).drop_duplicates().reset_index(drop=True)


def shuffle_columns(df, seed=0):
    rng = np.random.default_rng(seed)
    return df.apply(lambda s: s.to_numpy()[rng.permutation(len(s))]).reset_index(drop=True)


# ---------------------------------------------------------------- fidelity


def test_tvd_and_ks_of_disjoint_distributions_are_one():
    assert tvd(pd.Series(["a", "a", "b"]), pd.Series(["c", "d", "d"])) == pytest.approx(1.0)
    assert ks_statistic(pd.Series([1, 2, 3]), pd.Series([10, 11, 12])) == pytest.approx(1.0)


def test_tvd_half_overlap():
    # p = {a: 1}, q = {a: .5, b: .5} -> TVD = 0.5 * (0.5 + 0.5) = 0.5
    assert tvd(pd.Series(["a"] * 4), pd.Series(["a", "a", "b", "b"])) == pytest.approx(0.5)


def test_shuffled_columns_keep_every_marginal_exactly(real):
    """A column permutation has identical marginals: shapes = 1, every KS/TVD = 0."""
    metrics, details = evaluate_fidelity(real, shuffle_columns(real), sdv_metadata())
    assert metrics["column_shapes"] == pytest.approx(1.0)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in details["per_column"].values())


def test_corr_diff_ignores_positive_rescaling(real):
    rescaled = real.copy()
    rescaled[NUMERIC] = rescaled[NUMERIC] * 3 + 7
    assert corr_diff(real, rescaled) == pytest.approx(0.0, abs=1e-12)


def test_corr_diff_sign_flip():
    df = pd.DataFrame({"u": [1.0, 2, 3, 4, 5], "v": [1.0, 3, 2, 5, 4]})
    rho = df.corr().loc["u", "v"]
    flipped = df.assign(v=-df["v"])
    assert corr_diff(df, flipped, ["u", "v"]) == pytest.approx(2 * abs(rho))
    # mirror case (real correlation negative): the distance must still be positive
    assert corr_diff(flipped, df, ["u", "v"]) == pytest.approx(2 * abs(rho))


# ---------------------------------------------------------------- utility


@pytest.fixture
def learnable(real):
    df = real.copy()
    df["income"] = (df["education_num"] >= 12).map({True: ">50K", False: "<=50K"})
    return df.iloc[:350].reset_index(drop=True), df.iloc[350:].reset_index(drop=True)


def test_utility_metric_names(learnable):
    train, test = learnable
    trtr = evaluate_trtr(train, test, PARAMS, seed=0)
    tstr = evaluate_tstr(train, test, trtr, PARAMS, seed=0)
    stats = ["precision", "recall", "f1", "roc_auc"]
    assert set(trtr) == {f"trtr_{m}_{s}" for m in ("rf", "xgb") for s in stats}
    assert set(tstr) == {f"tstr_{m}_{s}" for m in ("rf", "xgb") for s in stats} | {"utility_ratio"}


def test_utility_is_deterministic(learnable):
    train, test = learnable
    assert evaluate_trtr(train, test, PARAMS, seed=3) == evaluate_trtr(train, test, PARAMS, seed=3)


def test_unseen_category_in_test_set_is_ignored(learnable):
    """Preprocessing is fitted on the training source only; a new category must not crash."""
    train, test = learnable
    test = test.copy()
    test.loc[test.index[:5], "occupation"] = "Never-seen-before"
    m = evaluate_trtr(train, test, PARAMS, seed=0)
    assert 0.0 <= m["trtr_rf_roc_auc"] <= 1.0


# ---------------------------------------------------------------- privacy


@pytest.fixture
def split(real):
    return real.iloc[:400].reset_index(drop=True), real.iloc[400:].reset_index(drop=True)


def test_exact_match_ignores_numeric_dtype(split):
    """A float-typed copy of train (39.0 vs 39) is still an exact copy."""
    train, _ = split
    as_float = train.copy()
    as_float[NUMERIC] = as_float[NUMERIC].astype(float)
    assert exact_match_rate(as_float, train) == pytest.approx(1.0)


def test_exact_match_counts_duplicate_synthetic_rows(split):
    train, holdout = split
    synth = pd.concat([train.iloc[[0, 0]], holdout.iloc[[0]]], ignore_index=True)
    assert exact_match_rate(synth, train) == pytest.approx(2 / 3)


def test_exact_match_needs_every_column(split):
    train, _ = split
    near = train.iloc[[0]].copy()
    other = [c for c in train["race"].unique() if c != near["race"].iloc[0]][0]
    near["race"] = other
    assert exact_match_rate(near, train.iloc[[0]]) == 0.0


def test_synthetic_equal_to_holdout(split):
    """synth == hold-out: DCR ratio is exactly 1; the attacker flags NON-members (AUC << 0.5)."""
    train, holdout = split
    m = evaluate_privacy(holdout.copy(), train, holdout, sample_size=150, seed=0)
    assert m["dcr_ratio"] == pytest.approx(1.0)
    assert m["dcr_median_synth_train"] == pytest.approx(m["dcr_median_holdout_train"])
    assert m["mia_auc"] < 0.1


def test_categories_only_in_synthetic_do_not_crash(split):
    train, holdout = split
    synth = holdout.copy()
    synth.loc[synth.index[:3], CATEGORICAL[0]] = "Brand-new-category"
    m = evaluate_privacy(synth, train, holdout, sample_size=100, seed=0)
    assert np.isfinite(m["mia_auc"])
