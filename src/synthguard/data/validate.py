"""Read raw Adult data and validate it, collecting every error at once."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from synthguard.data.schema import COLUMNS


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read adult.data (no header, comma + space separated). '?' is kept as a string."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=[],
        keep_default_na=False,
    )
    obj = df.select_dtypes("object").columns
    df[obj] = df[obj].apply(lambda s: s.str.strip())
    return df


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> dict:
    """Validate lazily. Returns a JSON-able report with `passed` and a list of errors."""
    report = {"schema": schema.name, "rows": int(len(df)), "columns": list(df.columns)}
    try:
        schema.validate(df, lazy=True)
        report.update(passed=True, n_errors=0, errors=[])
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases
        errors = [
            {
                "column": None if pd.isna(r.column) else str(r.column),
                "check": str(r.check),
                "failure_case": None if pd.isna(r.failure_case) else str(r.failure_case),
            }
            for r in fc.head(200).itertuples()
        ]
        report.update(passed=False, n_errors=int(len(fc)), errors=errors)
    return report
