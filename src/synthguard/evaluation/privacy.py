"""Privacy risk: distance to closest record, exact matches, distance-based membership inference.

These are empirical risk estimates for one attacker, not formal privacy guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from synthguard.data.schema import CATEGORICAL, COLUMNS, NUMERIC


def fit_encoder(train: pd.DataFrame) -> ColumnTransformer:
    """One-hot categoricals + min-max numerics, fitted on real train only (all 15 columns)."""
    enc = ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return enc.fit(train[COLUMNS])


def nearest_distances(
    encoder: ColumnTransformer, queries: pd.DataFrame, reference: pd.DataFrame
) -> np.ndarray:
    """Euclidean distance from each query row to its nearest row in `reference`."""
    nn = NearestNeighbors(n_neighbors=1).fit(encoder.transform(reference[COLUMNS]))
    dist, _ = nn.kneighbors(encoder.transform(queries[COLUMNS]))
    return dist[:, 0]


def _row_keys(df: pd.DataFrame) -> list[tuple]:
    """Hashable row keys: numerics compared as numbers (39 == 39.0), categoricals as text."""
    num = df[NUMERIC].astype("float64").to_numpy().tolist()
    cat = df[CATEGORICAL].astype(str).to_numpy().tolist()
    return [tuple(n) + tuple(c) for n, c in zip(num, cat, strict=True)]


def exact_match_rate(synth: pd.DataFrame, train: pd.DataFrame) -> float:
    """Fraction of synthetic rows identical (all columns) to at least one training row."""
    train_rows = set(_row_keys(train))
    return float(np.mean([row in train_rows for row in _row_keys(synth)]))


def membership_inference(
    encoder: ColumnTransformer,
    synth: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    """Attacker scores a record by -distance to its nearest synthetic row (closer = 'member').

    Members: `n` rows of train; non-members: `n` rows of hold-out, n = min(sample_size, both sizes).
    Returns ROC-AUC (0.5 = guessing) and advantage = max(TPR - FPR).
    """
    n = min(sample_size, len(train), len(holdout))
    members = train.sample(n=n, random_state=seed)
    non_members = holdout.sample(n=n, random_state=seed)
    records = pd.concat([members, non_members], ignore_index=True)
    labels = np.r_[np.ones(n), np.zeros(n)]
    scores = -nearest_distances(encoder, records, synth)
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "mia_auc": float(roc_auc_score(labels, scores)),
        "mia_advantage": float(np.max(tpr - fpr)),
    }


def evaluate_privacy(
    synth: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    encoder = fit_encoder(train)
    synth_dcr = nearest_distances(encoder, synth, train)
    holdout_dcr = nearest_distances(encoder, holdout, train)
    median_synth = float(np.median(synth_dcr))
    median_holdout = float(np.median(holdout_dcr))
    return {
        "dcr_median_synth_train": median_synth,
        "dcr_median_holdout_train": median_holdout,
        "dcr_ratio": median_synth / median_holdout if median_holdout > 0 else float("nan"),
        "exact_match_rate": exact_match_rate(synth, train),
        **membership_inference(encoder, synth, train, holdout, sample_size, seed),
    }
