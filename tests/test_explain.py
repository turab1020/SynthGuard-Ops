import numpy as np
import pytest
import xgboost as xgb
from conftest import make_raw

from synthguard.data.prepare import clean
from synthguard.serving.classifier import train
from synthguard.serving.explain import FEATURES, contributions, explain

PARAMS = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.2}


@pytest.fixture(scope="module")
def model_and_rows():
    df = clean(make_raw(n=700, seed=6)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where((df["education_num"] >= 12) & (df["age"] > 30), ">50K", "<=50K")
    pipe = train(df.iloc[:500], PARAMS, seed=0)
    return pipe, df.iloc[500:520].drop(columns="income").reset_index(drop=True)


def test_contributions_plus_base_equal_model_output(model_and_rows):
    pipe, x = model_and_rows
    contrib, base = contributions(pipe, x)
    transformed = pipe.named_steps["prep"].transform(x[FEATURES])
    margin = (
        pipe.named_steps["model"]
        .get_booster()
        .predict(xgb.DMatrix(transformed), output_margin=True)
    )
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), margin, atol=1e-4)
    proba = pipe.predict_proba(x)[:, 1]
    got = [e["probability"] for e in explain(pipe, x)]
    np.testing.assert_allclose(got, proba, atol=1e-5)


def test_one_contribution_per_original_feature(model_and_rows):
    pipe, x = model_and_rows
    contrib, _ = contributions(pipe, x)
    assert list(contrib.columns) == FEATURES and len(FEATURES) == 14
    assert not contrib.isna().any().any()


def test_top_features_are_sorted_and_complete(model_and_rows):
    pipe, x = model_and_rows
    e = explain(pipe, x.iloc[[0]], top_k=3)[0]
    mags = [abs(t["contribution"]) for t in e["top_features"]]
    assert mags == sorted(mags, reverse=True) and len(mags) == 3
    largest = max(abs(v) for v in e["all_contributions"].values())
    assert mags[0] == pytest.approx(largest)  # the top feature is the overall largest
    assert e["top_features"][0]["feature"] in FEATURES
    total = sum(t["contribution"] for t in e["top_features"]) + e["other_features"]
    assert total == pytest.approx(sum(e["all_contributions"].values()), abs=1e-6)
    assert e["logit"] == pytest.approx(e["base_value"] + total, abs=1e-6)


def test_signal_features_dominate(model_and_rows):
    pipe, x = model_and_rows
    contrib, _ = contributions(pipe, x)
    mean_abs = contrib.abs().mean().sort_values(ascending=False)
    assert set(mean_abs.index[:2]) == {"education_num", "age"}


def test_deterministic(model_and_rows):
    pipe, x = model_and_rows
    assert explain(pipe, x) == explain(pipe, x)
