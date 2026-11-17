"""Fidelity: how closely synthetic data matches the real training data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdmetrics.reports.single_table import QualityReport

from synthguard.data.schema import CATEGORICAL, NUMERIC


def ks_statistic(real: pd.Series, synth: pd.Series) -> float:
    return float(ks_2samp(real, synth).statistic)


def tvd(real: pd.Series, synth: pd.Series) -> float:
    """Total variation distance between two categorical distributions."""
    p = real.value_counts(normalize=True)
    q = synth.value_counts(normalize=True)
    cats = p.index.union(q.index)
    return float(0.5 * (p.reindex(cats, fill_value=0) - q.reindex(cats, fill_value=0)).abs().sum())


def corr_diff(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str] = NUMERIC) -> float:
    """Mean |Pearson_real - Pearson_synth| over distinct numeric column pairs."""
    r = real[cols].corr().to_numpy()
    s = synth[cols].corr().to_numpy()
    iu = np.triu_indices(len(cols), k=1)
    return float(np.nanmean(np.abs(r[iu] - s[iu])))


def evaluate_fidelity(
    real: pd.DataFrame, synth: pd.DataFrame, metadata: dict
) -> tuple[dict[str, float], dict]:
    """Return (flat metrics for MLflow, details for the JSON report).

    `metadata` is the SDV multi-table dict written by `prepare`; its single table is used.
    """
    table_meta = next(iter(metadata["tables"].values()))
    report = QualityReport()
    report.generate(real, synth, table_meta, verbose=False)
    props = report.get_properties().set_index("Property")["Score"]

    per_column = {
        **{f"ks_{c}": ks_statistic(real[c], synth[c]) for c in NUMERIC},
        **{f"tvd_{c}": tvd(real[c], synth[c]) for c in CATEGORICAL},
    }
    metrics = {
        "quality_score": float(report.get_score()),
        "column_shapes": float(props["Column Shapes"]),
        "column_pair_trends": float(props["Column Pair Trends"]),
        "corr_diff": corr_diff(real, synth),
    }
    details = {
        "metrics": metrics,
        "per_column": per_column,
        "column_shapes": report.get_details("Column Shapes").to_dict(orient="records"),
        "column_pair_trends": report.get_details("Column Pair Trends")
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }
    return metrics, details
