import importlib.util
import json
from pathlib import Path

import pytest
from conftest import make_raw

from synthguard.data.prepare import clean
from synthguard.data.schema import CATEGORIES, RANGES
from synthguard.monitoring.simulate import (
    WHITE_COLLAR,
    load_test,
    make_batch,
    read_raw_test,
)
from synthguard.serving.app import PersonFeatures
from synthguard.serving.explain import FEATURES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    return clean(make_raw(n=800, seed=5)).drop_duplicates().reset_index(drop=True)


def test_batches_are_seeded_and_sized(source):
    a = make_batch(source, "normal", 300, seed=1)
    assert len(a) == 300 and list(a.columns) == list(source.columns)
    assert a.equals(make_batch(source, "normal", 300, seed=1))
    assert not a.equals(make_batch(source, "normal", 300, seed=2))


def test_drifted_batch_shifts_the_documented_columns(source):
    normal = make_batch(source, "normal", 400, seed=3)
    drifted = make_batch(source, "drifted", 400, seed=3)
    assert drifted["age"].mean() > normal["age"].mean() + 10
    assert drifted["hours_per_week"].mean() > normal["hours_per_week"].mean() * 1.2
    assert drifted["education_num"].mean() > normal["education_num"].mean() + 1
    assert drifted["occupation"].isin(WHITE_COLLAR).mean() > 0.6
    assert (drifted[["fnlwgt", "sex", "race"]] == normal[["fnlwgt", "sex", "race"]]).all().all()


@pytest.mark.parametrize("mode", ["normal", "drifted"])
def test_every_row_is_a_valid_api_request(source, mode):
    batch = make_batch(source, mode, 400, seed=4)
    for col, (lo, hi) in RANGES.items():
        if col in batch:
            assert batch[col].between(lo, hi).all(), col
    for col in FEATURES:
        if col in CATEGORIES:
            assert batch[col].isin(CATEGORIES[col] + ["Unknown"]).all(), col
    for row in batch[FEATURES].head(50).to_dict(orient="records"):
        PersonFeatures(**row)


def test_unknown_mode_is_rejected(source):
    with pytest.raises(ValueError, match="mode"):
        make_batch(source, "sideways", 10, seed=0)


RAW_TEST = (
    "|1x3 Cross validator\n"
    "25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, "
    "0, 0, 40, United-States, <=50K.\n"
    "38, ?, 89814, HS-grad, 9, Married-civ-spouse, ?, Husband, White, Male, 0, 0, 50, "
    "United-States, >50K.\n\n"
)


def test_adult_test_format_is_parsed(tmp_path):
    p = tmp_path / "adult.test"
    p.write_text(RAW_TEST)
    raw = read_raw_test(p)
    assert len(raw) == 2 and raw["income"].tolist() == ["<=50K.", ">50K."]
    df = load_test(p)
    assert list(df.columns) == [*FEATURES, "income"] and df.loc[1, "workclass"] == "Unknown"
    assert df["income"].tolist() == ["<=50K", ">50K"]
    assert not (tmp_path / "adult.test.body").exists()


def test_batch_command_writes_jsonl(tmp_path):
    p = tmp_path / "adult.test"
    p.write_text(RAW_TEST)
    spec = importlib.util.spec_from_file_location("sim", ROOT / "scripts" / "simulate_traffic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "b.jsonl"
    assert mod.main(["batch", "--mode", "drifted", "--n", "5", "--source", str(p),
                     "--out", str(out)]) == 0  # fmt: skip
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 5 and set(rows[0]) == set(FEATURES)


def test_csv_batch_keeps_labels_for_retraining(tmp_path):
    p = tmp_path / "adult.test"
    p.write_text(RAW_TEST)
    spec = importlib.util.spec_from_file_location("sim", ROOT / "scripts" / "simulate_traffic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "new_batch.csv"
    assert mod.main(["batch", "--mode", "drifted", "--n", "6", "--source", str(p),
                     "--csv", str(out)]) == 0  # fmt: skip
    df = __import__("pandas").read_csv(out)
    assert len(df) == 6 and list(df.columns) == [*FEATURES, "income"]
