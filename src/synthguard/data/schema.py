"""pandera schemas for raw and processed UCI Adult data.

Allowed categories follow the UCI documentation (adult.names). Raw data uses "?" for
missing values; the prepare stage turns those into "Unknown".
"""

from __future__ import annotations

import pandera.pandas as pa

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]  # fmt: skip

NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC]
TARGET = "income"

CATEGORIES: dict[str, list[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
        "State-gov", "Without-pay", "Never-worked",
    ],
    "education": [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
        "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate",
        "5th-6th", "Preschool",
    ],
    "marital_status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
        "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried",
    ],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native_country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
        "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
        "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
        "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
        "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands",
    ],
    "income": ["<=50K", ">50K"],
}  # fmt: skip

RANGES = {
    "age": (17, 90),
    "education_num": (1, 16),
    "hours_per_week": (1, 99),
}

MISSING_RAW = "?"
MISSING_CLEAN = "Unknown"


def _numeric_column(name: str) -> pa.Column:
    checks = []
    if name in RANGES:
        lo, hi = RANGES[name]
        checks.append(pa.Check.in_range(lo, hi))
    elif name == "fnlwgt":
        checks.append(pa.Check.gt(0))
    else:  # capital_gain, capital_loss
        checks.append(pa.Check.ge(0))
    return pa.Column("int64", checks=checks, nullable=True, name=name)


def _categorical_column(name: str, missing_token: str) -> pa.Column:
    allowed = list(CATEGORIES[name])
    if name != TARGET:
        allowed.append(missing_token)
    return pa.Column(str, checks=pa.Check.isin(allowed), nullable=True, name=name)


def build_schema(
    missing_token: str, min_rows: int, max_null_fraction: float, name: str
) -> pa.DataFrameSchema:
    cols = {
        c: _numeric_column(c) if c in NUMERIC else _categorical_column(c, missing_token)
        for c in COLUMNS
    }
    return pa.DataFrameSchema(
        cols,
        strict=True,
        ordered=True,
        coerce=False,
        name=name,
        checks=[
            pa.Check(lambda df: len(df) >= min_rows, error=f"fewer than {min_rows} rows"),
            pa.Check(
                lambda df: bool((df.isna().mean() <= max_null_fraction).all()),
                error=f"a column has null fraction > {max_null_fraction}",
            ),
            pa.Check(
                lambda df: df[TARGET].nunique() == 2, error="income must contain both classes"
            ),
        ],
    )


def raw_schema(min_rows: int, max_null_fraction: float) -> pa.DataFrameSchema:
    return build_schema(MISSING_RAW, min_rows, max_null_fraction, "RAW_SCHEMA")


def processed_schema(min_rows: int, max_null_fraction: float = 0.0) -> pa.DataFrameSchema:
    return build_schema(MISSING_CLEAN, min_rows, max_null_fraction, "PROCESSED_SCHEMA")
