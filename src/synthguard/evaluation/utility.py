"""Utility: train classifiers on real or synthetic data, test on the real hold-out.

TRTR = train real, test real (reference). TSTR = train synthetic, test real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from synthguard.data.schema import CATEGORICAL, NUMERIC, TARGET

POSITIVE = ">50K"
FEATURES_CAT = [c for c in CATEGORICAL if c != TARGET]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[NUMERIC + FEATURES_CAT], (df[TARGET] == POSITIVE).astype(int).to_numpy()


def preprocessor() -> ColumnTransformer:
    """One-hot categoricals (unknown categories ignored) + standard-scaled numerics."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )


def make_models(params: dict, seed: int) -> dict[str, object]:
    rf = dict(params["random_forest"])
    xgb = dict(params["xgboost"])
    return {
        "rf": RandomForestClassifier(**rf, random_state=seed, n_jobs=-1),
        "xgb": XGBClassifier(**xgb, random_state=seed, n_jobs=-1, eval_metric="logloss"),
    }


def score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def train_test(train: pd.DataFrame, test: pd.DataFrame, params: dict, seed: int) -> dict:
    """Fit each model on `train` (preprocessing fitted on `train` only), score on `test`.

    Returns {"rf": {...}, "xgb": {...}}. If `train` has one class only, the model can only
    predict that class: scored as a constant predictor (ROC-AUC 0.5).
    """
    x_tr, y_tr = split_xy(train)
    x_te, y_te = split_xy(test)
    results = {}
    for name, model in make_models(params, seed).items():
        if len(np.unique(y_tr)) < 2:
            proba = np.full(len(y_te), float(y_tr[0]))
        else:
            pipe = Pipeline([("prep", preprocessor()), ("model", model)])
            pipe.fit(x_tr, y_tr)
            proba = pipe.predict_proba(x_te)[:, 1]
        results[name] = score(y_te, proba)
    return results


def flatten(prefix: str, results: dict) -> dict[str, float]:
    return {f"{prefix}_{m}_{k}": v for m, r in results.items() for k, v in r.items()}


def evaluate_trtr(train: pd.DataFrame, holdout: pd.DataFrame, params: dict, seed: int) -> dict:
    return flatten("trtr", train_test(train, holdout, params, seed))


def evaluate_tstr(
    synth: pd.DataFrame, holdout: pd.DataFrame, trtr: dict, params: dict, seed: int
) -> dict:
    """TSTR metrics + utility_ratio = TSTR RF ROC-AUC / TRTR RF ROC-AUC."""
    out = flatten("tstr", train_test(synth, holdout, params, seed))
    out["utility_ratio"] = out["tstr_rf_roc_auc"] / trtr["trtr_rf_roc_auc"]
    return out
