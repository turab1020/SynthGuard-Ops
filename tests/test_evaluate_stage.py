import numpy as np
import pandas as pd
from conftest import make_raw

from synthguard.config import load_params
from synthguard.data.prepare import clean, sdv_metadata
from synthguard.evaluation.utility import evaluate_trtr
from synthguard.stages.evaluate import TABLE_COLUMNS, evaluate_candidate

EXPECTED = {
    "quality_score", "column_shapes", "column_pair_trends", "corr_diff",
    "tstr_rf_precision", "tstr_rf_recall", "tstr_rf_f1", "tstr_rf_roc_auc",
    "tstr_xgb_precision", "tstr_xgb_recall", "tstr_xgb_f1", "tstr_xgb_roc_auc",
    "utility_ratio", "dcr_median_synth_train", "dcr_median_holdout_train", "dcr_ratio",
    "exact_match_rate", "mia_auc", "mia_advantage",
}  # fmt: skip


def small_params():
    p = load_params(smoke=True)
    p["evaluation"]["utility"]["random_forest"]["n_estimators"] = 20
    p["evaluation"]["utility"]["xgboost"]["n_estimators"] = 20
    p["evaluation"]["privacy"]["mia_sample_size"] = 100
    return p


def test_evaluate_candidate_returns_every_metric():
    df = clean(make_raw(n=600, seed=21)).drop_duplicates().reset_index(drop=True)
    train, holdout = df.iloc[:400], df.iloc[400:]
    p = small_params()
    trtr = evaluate_trtr(train, holdout, p["evaluation"]["utility"], p["seed"])
    rng = np.random.default_rng(0)
    synth = train.apply(lambda s: s.to_numpy()[rng.permutation(len(s))])
    metrics, details = evaluate_candidate(synth, train, holdout, sdv_metadata(), trtr, p)
    assert set(metrics) == EXPECTED
    assert all(np.isfinite(v) for v in metrics.values())
    assert "per_column" in details
    # every table column except the added ones comes from evaluate_candidate
    added = {"selection_score", "privacy_gate", "fit_seconds"}
    assert set(TABLE_COLUMNS) - added <= set(metrics)


def test_copy_of_train_fails_the_gate_in_the_stage_pipeline():
    from synthguard.evaluation.score import score_candidate

    df = clean(make_raw(n=600, seed=22)).drop_duplicates().reset_index(drop=True)
    train, holdout = df.iloc[:400], df.iloc[400:]
    p = small_params()
    trtr = evaluate_trtr(train, holdout, p["evaluation"]["utility"], p["seed"])
    metrics, _ = evaluate_candidate(train.copy(), train, holdout, sdv_metadata(), trtr, p)
    scored = score_candidate(metrics, p["selection"])
    assert scored["privacy_gate"] == "failed"
    assert isinstance(pd.Series(scored).get("selection_score"), float)
