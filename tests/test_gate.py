import json

import numpy as np
import pytest
from conftest import make_raw

from synthguard.data.prepare import clean
from synthguard.serving.classifier import evaluate, train
from synthguard.serving.gate import check, summary_markdown

BENCH = {"roc_auc": 0.88, "f1": 0.63, "accuracy": 0.82, "precision": 0.63, "recall": 0.62}


def cand(auc):
    return {**BENCH, "roc_auc": auc}


@pytest.mark.parametrize(
    "auc,passed", [(0.88, True), (0.90, True), (0.87, True), (0.8699, False), (0.70, False)]
)
def test_roc_auc_rule_with_inclusive_boundary(auc, passed):
    assert check(cand(auc), BENCH, max_drop=0.01, privacy_gate="passed")["passed"] is passed


def test_privacy_gate_failure_blocks_even_a_better_model():
    r = check(cand(0.95), BENCH, max_drop=0.01, privacy_gate="failed")
    assert r["passed"] is False and "privacy" in r["reasons"][0]


def test_summary_states_the_decision():
    r = check(cand(0.80), BENCH, 0.01, "passed")
    text = summary_markdown(cand(0.80), BENCH, r, 0.01)
    assert "BLOCKED" in text and "| roc_auc | 0.8800 | 0.8000 | -0.0800 |" in text
    assert "PASSED" in summary_markdown(BENCH, BENCH, check(BENCH, BENCH, 0.01, "passed"), 0.01)


def test_a_deliberately_worse_model_is_blocked():
    """Production model vs a candidate trained on label-shuffled data: the gate blocks it."""
    df = clean(make_raw(n=900, seed=8)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    synth, holdout = df.iloc[:600], df.iloc[600:]
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}
    benchmark = evaluate(train(synth, params, seed=0), holdout)
    rng = np.random.default_rng(0)
    broken = synth.assign(income=rng.permutation(synth["income"].to_numpy()))
    worse = evaluate(train(broken, params, seed=0), holdout)
    assert check(worse, benchmark, 0.01, "passed")["passed"] is False
    assert check(benchmark, benchmark, 0.01, "passed")["passed"] is True


def test_script_exit_codes(tmp_path, monkeypatch):
    """End to end through scripts/eval_gate.py with a saved candidate."""
    import importlib.util

    import joblib

    df = clean(make_raw(n=900, seed=9)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    synth, holdout = df.iloc[:600], df.iloc[600:].reset_index(drop=True)
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}
    good = train(synth, params, seed=0)
    joblib.dump(good, tmp_path / "good.joblib")
    bench = evaluate(good, holdout)
    (tmp_path / "bench.json").write_text(json.dumps(bench))
    (tmp_path / "eval.json").write_text(json.dumps({"selection": {"privacy_gate": "passed"}}))
    spec = importlib.util.spec_from_file_location("eval_gate", "scripts/eval_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setattr(gate, "load_processed", lambda: (synth, holdout, {}))
    args = [
        "--benchmark",
        str(tmp_path / "bench.json"),
        "--evaluation",
        str(tmp_path / "eval.json"),
    ]
    assert gate.main(["--candidate", str(tmp_path / "good.joblib"), *args]) == 0
    worse = dict(bench, roc_auc=bench["roc_auc"] + 0.05)  # benchmark better than candidate
    (tmp_path / "bench.json").write_text(json.dumps(worse))
    out = tmp_path / "gate.md"
    assert (
        gate.main(["--candidate", str(tmp_path / "good.joblib"), *args, "--summary", str(out)]) == 1
    )
    assert "BLOCKED" in out.read_text()
