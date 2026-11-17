import pandas as pd

from synthguard.data.prepare import clean, positive_rate, sdv_metadata, split
from synthguard.data.schema import COLUMNS


def prepared(raw_df):
    # add exact duplicates on purpose to prove they are removed before splitting
    df = pd.concat([raw_df, raw_df.head(30)], ignore_index=True)
    return clean(df)


def test_no_shared_rows(raw_df):
    train, holdout, n_dup = split(prepared(raw_df), 0.3, 42)
    assert n_dup >= 30
    shared = train.merge(holdout, how="inner", on=COLUMNS)
    assert len(shared) == 0


def test_class_ratio_preserved(raw_df):
    train, holdout, _ = split(prepared(raw_df), 0.3, 42)
    assert abs(positive_rate(train) - positive_rate(holdout)) <= 0.01


def test_split_sizes(raw_df):
    train, holdout, n_dup = split(prepared(raw_df), 0.3, 42)
    total = len(train) + len(holdout)
    assert total == len(raw_df) + 30 - n_dup
    assert abs(len(holdout) / total - 0.3) < 0.01


def test_deterministic(raw_df):
    a = split(prepared(raw_df), 0.3, 42)
    b = split(prepared(raw_df), 0.3, 42)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_clean_maps_unknown_and_labels(raw_df):
    df = raw_df.copy()
    df.loc[0, "income"] = ">50K."
    out = clean(df)
    assert "?" not in out.values
    assert (out["workclass"] == "Unknown").any()
    assert set(out["income"]) == {"<=50K", ">50K"}


def test_sdv_metadata_loads():
    from sdv.metadata import Metadata

    md = Metadata.load_from_dict(sdv_metadata())
    md.validate()
    cols = md.tables["adult"].columns
    assert cols["age"]["sdtype"] == "numerical"
    assert cols["income"]["sdtype"] == "categorical"
