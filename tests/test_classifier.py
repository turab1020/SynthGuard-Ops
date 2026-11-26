import numpy as np
import pytest
from conftest import make_raw

from synthguard.data.prepare import clean
from synthguard.serving.classifier import evaluate, train

PARAMS = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}


@pytest.fixture
def data():
    df = clean(make_raw(n=800, seed=5)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    return df.iloc[:500].reset_index(drop=True), df.iloc[500:].reset_index(drop=True)


def test_trains_and_scores_on_holdout(data):
    synth, holdout = data
    scores = evaluate(train(synth, PARAMS, seed=0), holdout)
    assert scores["roc_auc"] > 0.95
    assert set(scores) == {"roc_auc", "f1", "accuracy", "precision", "recall",
                           "holdout_rows", "positive_rate"}  # fmt: skip
    assert scores["holdout_rows"] == len(holdout)


def test_is_deterministic(data):
    synth, holdout = data
    a = train(synth, PARAMS, seed=1).predict_proba(holdout.drop(columns="income"))
    b = train(synth, PARAMS, seed=1).predict_proba(holdout.drop(columns="income"))
    assert np.array_equal(a, b)


def test_preprocessing_is_fitted_on_synthetic_only(data):
    synth, holdout = data
    holdout = holdout.copy()
    holdout.loc[holdout.index[:3], "occupation"] = "Not-in-training-data"
    scores = evaluate(train(synth, PARAMS, seed=0), holdout)  # unseen category must not crash
    assert 0 <= scores["roc_auc"] <= 1


def test_single_class_training_data_is_rejected(data):
    synth, _ = data
    with pytest.raises(ValueError, match="single income class"):
        train(synth.assign(income="<=50K"), PARAMS, seed=0)
