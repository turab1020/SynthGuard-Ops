import pytest

from synthguard.data.prepare import clean
from synthguard.evaluation.utility import evaluate_trtr, evaluate_tstr, score

PARAMS = {
    "random_forest": {"n_estimators": 20, "max_depth": None},
    "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
}


@pytest.fixture
def data(raw_df):
    df = clean(raw_df)
    # make income learnable from education_num so utility is measurable
    df["income"] = (df["education_num"] >= 12).map({True: ">50K", False: "<=50K"})
    return df.iloc[:300], df.iloc[300:]


def test_trtr_learns_signal(data):
    train, test = data
    m = evaluate_trtr(train, test, PARAMS, seed=0)
    assert m["trtr_rf_roc_auc"] > 0.9
    assert m["trtr_xgb_roc_auc"] > 0.9
    assert {"trtr_rf_precision", "trtr_rf_recall", "trtr_rf_f1"} <= set(m)


def test_tstr_real_copy_ratio_is_one(data):
    train, test = data
    trtr = evaluate_trtr(train, test, PARAMS, seed=0)
    tstr = evaluate_tstr(train.copy(), test, trtr, PARAMS, seed=0)
    assert tstr["utility_ratio"] == pytest.approx(1.0)


def test_independent_columns_loses_utility(data):
    train, test = data
    trtr = evaluate_trtr(train, test, PARAMS, seed=0)
    synth = shuffle_columns(train)
    tstr = evaluate_tstr(synth, test, trtr, PARAMS, seed=0)
    assert tstr["utility_ratio"] < 0.8  # columns independent -> no signal


def test_single_class_synthetic(data):
    train, test = data
    trtr = evaluate_trtr(train, test, PARAMS, seed=0)
    tstr = evaluate_tstr(train.assign(income="<=50K"), test, trtr, PARAMS, seed=0)
    assert tstr["tstr_rf_roc_auc"] == 0.5


def test_score_known_answer():
    pass


def shuffle_columns(df, seed=0):
    """Permute every column independently: keeps each marginal, destroys all relationships."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return df.apply(lambda s: s.to_numpy()[rng.permutation(len(s))]).reset_index(drop=True)

    s = score(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]))
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["roc_auc"] == 0.75
