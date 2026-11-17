"""Clean raw Adult data and make a leakage-free stratified train/hold-out split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from synthguard.data.schema import (
    CATEGORICAL,
    MISSING_CLEAN,
    MISSING_RAW,
    NUMERIC,
    TARGET,
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, map '?' to 'Unknown', normalise income labels (drop trailing '.')."""
    out = df.copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype(str).str.strip().replace(MISSING_RAW, MISSING_CLEAN)
    out[TARGET] = out[TARGET].str.rstrip(".")
    for c in NUMERIC:
        out[c] = out[c].astype("int64")
    return out


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Drop exact duplicates FIRST, then stratified split. Returns (train, holdout, n_dropped)."""
    deduped = df.drop_duplicates().reset_index(drop=True)
    n_dropped = len(df) - len(deduped)
    train, holdout = train_test_split(
        deduped, test_size=test_size, stratify=deduped[TARGET], random_state=seed
    )
    return train.reset_index(drop=True), holdout.reset_index(drop=True), n_dropped


def sdv_metadata(table_name: str = "adult") -> dict:
    """SDV single-table metadata dict with explicit categorical / numerical sdtypes."""
    columns = {c: {"sdtype": "numerical", "computer_representation": "Int64"} for c in NUMERIC}
    columns.update({c: {"sdtype": "categorical"} for c in CATEGORICAL})
    ordered = {c: columns[c] for c in [*NUMERIC, *CATEGORICAL]}
    return {"METADATA_SPEC_VERSION": "V1", "tables": {table_name: {"columns": ordered}}}


def positive_rate(df: pd.DataFrame) -> float:
    return float((df[TARGET] == ">50K").mean())
