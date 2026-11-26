"""Per-prediction explanations: exact TreeSHAP contributions from XGBoost.

XGBoost computes TreeSHAP itself (`pred_contribs=True`), so no extra package is needed. The model
sees one-hot / scaled columns; contributions are summed back to the 14 original features, so each
explanation reads "occupation", not "cat__occupation_Exec-managerial".
In log-odds: base_value + sum(contributions) = the model's raw score, and
probability = sigmoid(raw score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from synthguard.data.schema import NUMERIC
from synthguard.evaluation.utility import FEATURES_CAT

FEATURES = NUMERIC + FEATURES_CAT


def column_owners(pipeline: Pipeline) -> list[str]:
    """Original feature name for every transformed column, in the model's column order."""
    prep = pipeline.named_steps["prep"]
    owners: list[str] = []
    for name, transformer, cols in prep.transformers_:
        if name == "num":
            owners += list(cols)
        elif name == "cat":
            for col, cats in zip(cols, transformer.categories_, strict=True):
                owners += [col] * len(cats)
    return owners


def contributions(pipeline: Pipeline, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(per-row contributions per original feature in log-odds, per-row base values)."""
    transformed = pipeline.named_steps["prep"].transform(x[FEATURES])
    booster = pipeline.named_steps["model"].get_booster()
    raw = booster.predict(xgb.DMatrix(transformed), pred_contribs=True)
    owners = column_owners(pipeline)
    per_col = pd.DataFrame(raw[:, :-1], columns=range(len(owners)))
    grouped = per_col.T.groupby(np.array(owners)).sum().T
    return grouped.reindex(columns=FEATURES), raw[:, -1]


def explain(pipeline: Pipeline, x: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """One explanation per row: probability, base value and the top-k features by |contribution|."""
    contrib, base = contributions(pipeline, x)
    out = []
    for i in range(len(x)):
        row = contrib.iloc[i]
        logit = float(base[i] + row.sum())
        order = row.abs().sort_values(ascending=False).index
        top = [
            {"feature": f, "value": _plain(x.iloc[i][f]), "contribution": float(row[f])}
            for f in order[:top_k]
        ]
        out.append(
            {
                "probability": float(1 / (1 + np.exp(-logit))),
                "base_value": float(base[i]),
                "logit": logit,
                "top_features": top,
                "other_features": float(row[order[top_k:]].sum()),
                "all_contributions": {f: float(row[f]) for f in FEATURES},
            }
        )
    return out


def _plain(v):
    return v.item() if hasattr(v, "item") else v
