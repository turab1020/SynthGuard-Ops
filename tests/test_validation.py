import numpy as np
import pytest

from synthguard.data.prepare import clean
from synthguard.data.schema import processed_schema, raw_schema
from synthguard.data.validate import validate

SCHEMA = raw_schema(min_rows=100, max_null_fraction=0.10)


def failed(df, schema=SCHEMA):
    report = validate(df, schema)
    assert report["passed"] is False
    return report


def test_valid_raw_passes(raw_df):
    assert validate(raw_df, SCHEMA)["passed"]


def test_valid_processed_passes(raw_df):
    assert validate(clean(raw_df), processed_schema(min_rows=100))["passed"]


def test_raw_question_mark_not_allowed_after_cleaning(raw_df):
    # processed schema rejects '?', proving prepare must map it to 'Unknown'
    failed(raw_df, processed_schema(min_rows=100))


def test_missing_column(raw_df):
    r = failed(raw_df.drop(columns=["occupation"]))
    assert any("occupation" in str(e) for e in r["errors"])


def test_extra_column(raw_df):
    failed(raw_df.assign(extra=1))


def test_wrong_dtype(raw_df):
    failed(raw_df.assign(age=raw_df["age"].astype(str)))


@pytest.mark.parametrize(
    "col,value", [("age", 150), ("education_num", 0), ("hours_per_week", 120), ("fnlwgt", -5)]
)
def test_out_of_range(raw_df, col, value):
    df = raw_df.copy()
    df.loc[0, col] = value
    r = failed(df)
    assert any(e["column"] == col for e in r["errors"])


def test_unknown_category(raw_df):
    df = raw_df.copy()
    df.loc[0, "race"] = "Martian"
    r = failed(df)
    assert any(e["failure_case"] == "Martian" for e in r["errors"])


def test_too_many_nulls(raw_df):
    df = raw_df.copy()
    df.loc[df.index[:100], "occupation"] = np.nan  # 25% > 10%
    failed(df)


def test_too_few_rows(raw_df):
    failed(raw_df.head(50))


def test_single_class_target(raw_df):
    failed(raw_df.assign(income="<=50K"))


def test_lazy_reports_all_errors(raw_df):
    df = raw_df.copy()
    df.loc[0, "age"] = 200
    df.loc[1, "race"] = "Martian"
    cols = {e["column"] for e in failed(df)["errors"]}
    assert {"age", "race"} <= cols
