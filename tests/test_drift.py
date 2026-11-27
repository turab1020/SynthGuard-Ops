import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_raw

from synthguard.data.prepare import clean
from synthguard.monitoring.drift import (
    PREDICTION,
    check_drift,
    reference_sample,
    requests_frame,
)
from synthguard.serving.app import DriftStatus
from synthguard.serving.classifier import train
from synthguard.serving.explain import FEATURES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    df = clean(make_raw(n=2000, seed=3)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    ref_part, cur_part = df.iloc[:1200], df.iloc[1200:].reset_index(drop=True)
    pipe = train(ref_part, {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3}, seed=0)
    reference = reference_sample(ref_part, pipe, n=1000, seed=0)
    return pipe, reference, cur_part


def scored(pipe, df):
    out = df[FEATURES].copy()
    out[PREDICTION] = pipe.predict_proba(out)[:, 1]
    return out


def shifted(df):
    d = df.copy()
    d["age"] = (d["age"] + 15).clip(upper=90)
    d["hours_per_week"] = (d["hours_per_week"] * 1.4).clip(upper=99).astype(int)
    d["education_num"] = (d["education_num"] + 2).clip(upper=16)
    return d


def test_reference_is_a_seeded_sample_with_predictions(setup):
    pipe, reference, _ = setup
    assert list(reference.columns) == [*FEATURES, PREDICTION] and len(reference) == 1000
    assert reference[PREDICTION].between(0, 1).all()


def test_same_distribution_raises_no_alert(setup, tmp_path):
    pipe, reference, cur = setup
    r = check_drift(reference, scored(pipe, cur), threshold=0.25, html_path=tmp_path / "r.html")
    assert not r.alert and r.share_of_drifted_columns < 0.25
    assert (tmp_path / "r.html").stat().st_size > 0


def test_shifted_traffic_raises_an_alert(setup):
    pipe, reference, cur = setup
    r = check_drift(reference, scored(pipe, shifted(cur)), threshold=0.25)
    assert r.alert and r.dataset_drift
    assert {"age", "hours_per_week", "education_num"} <= set(r.drifted_columns)
    assert r.number_of_columns == len(FEATURES)


def test_threshold_decides_dataset_drift(setup):
    pipe, reference, cur = setup
    current = scored(pipe, shifted(cur))
    assert check_drift(reference, current, threshold=0.05).dataset_drift
    assert not check_drift(reference, current, threshold=0.9).dataset_drift


def test_too_few_requests_and_missing_columns_are_rejected(setup):
    pipe, reference, cur = setup
    with pytest.raises(ValueError, match="at least"):
        check_drift(reference, scored(pipe, cur.head(5)), threshold=0.25, min_rows=30)
    with pytest.raises(ValueError, match="missing"):
        requests_frame([{"age": 30}])


def test_api_payload_is_accepted_by_the_drift_endpoint_model(setup):
    pipe, reference, cur = setup
    r = check_drift(reference, scored(pipe, shifted(cur)), threshold=0.25)
    status = DriftStatus(**r.api_payload())
    assert status.alert is True and status.share_of_drifted_columns == r.share_of_drifted_columns


def _script():
    spec = importlib.util.spec_from_file_location(
        "check_drift", ROOT / "scripts" / "check_drift.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_writes_outputs_reports_and_dispatches_on_alert(setup, tmp_path, monkeypatch):
    pipe, _, cur = setup
    import joblib

    joblib.dump(pipe, tmp_path / "m.joblib")
    train_csv = tmp_path / "train.csv"
    real = clean(make_raw(n=2000, seed=3)).drop_duplicates().reset_index(drop=True)
    real.iloc[:1200].to_csv(train_csv, index=False)
    reqs = tmp_path / "req.jsonl"
    reqs.write_text("".join(json.dumps(r) + "\n" for r in
                            scored(pipe, shifted(cur)).to_dict(orient="records")))  # fmt: skip
    mod = _script()
    calls = []
    monkeypatch.setattr(mod.requests, "post", lambda url, **kw: calls.append(url) or _Ok())
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    code = mod.main(["--requests-file", str(reqs), "--out-dir", str(tmp_path / "out"),
                     "--model", str(tmp_path / "m.joblib"), "--train", str(train_csv),
                     "--dispatch"])  # fmt: skip
    assert code == mod.ALERT_EXIT
    out = json.loads((tmp_path / "out" / "drift.json").read_text())
    assert out["alert"] is True and (tmp_path / "out" / "report.html").exists()
    assert calls == ["https://api.github.com/repos/o/r/dispatches"]


class _Ok:
    def raise_for_status(self):
        return None
