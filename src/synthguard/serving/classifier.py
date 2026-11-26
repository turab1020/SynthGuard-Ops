"""Income classifier trained only on the champion generator's synthetic data.

The model users query live. It never sees real training rows; it is evaluated on the fixed real
hold-out, and those scores are the production benchmark (metrics/classifier.json).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from synthguard.evaluation.utility import POSITIVE, preprocessor, split_xy


def build_pipeline(params: dict, seed: int) -> Pipeline:
    """One-hot + scaling (fitted on the training data) followed by XGBoost."""
    model = XGBClassifier(**params, random_state=seed, n_jobs=1, eval_metric="logloss")
    return Pipeline([("prep", preprocessor()), ("model", model)])


def train(synthetic: pd.DataFrame, params: dict, seed: int) -> Pipeline:
    x, y = split_xy(synthetic)
    if len(np.unique(y)) < 2:
        raise ValueError("synthetic training data has a single income class")
    return build_pipeline(params, seed).fit(x, y)


def evaluate(pipeline: Pipeline, holdout: pd.DataFrame) -> dict[str, float]:
    """Scores on the real hold-out (positive class >50K, threshold 0.5)."""
    x, y = split_xy(holdout)
    proba = pipeline.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "holdout_rows": int(len(holdout)),
        "positive_rate": float((holdout["income"] == POSITIVE).mean()),
    }
