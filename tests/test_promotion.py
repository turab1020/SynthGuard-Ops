"""Champion vs challenger: decision rule, registry aliases/tags, and scripts/compare_promote.py."""

import importlib.util
import json
import pickle

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from conftest import make_raw
from mlflow import MlflowClient
from stub_generator import StubGenerator

from synthguard.data.prepare import clean
from synthguard.serving.classifier import evaluate, train
from synthguard.tracking import promotion, registry
from synthguard.tracking.mlflow_utils import start_candidate_run

PARAMS = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}
SCORES = {"roc_auc": 0.88, "f1": 0.63, "accuracy": 0.82, "precision": 0.63, "recall": 0.62}


@pytest.fixture(scope="module")
def data():
    df = clean(make_raw(n=900, seed=8)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    synth, holdout = df.iloc[:600], df.iloc[600:].reset_index(drop=True)
    good = train(synth, PARAMS, seed=0)
    rng = np.random.default_rng(0)
    worse = train(synth.assign(income=rng.permutation(synth["income"].to_numpy())), PARAMS, 0)
    return synth, holdout, good, worse


@pytest.fixture
def registry_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


def export(tmp_path, pipeline, holdout):
    d = tmp_path / "deploy"
    d.mkdir(exist_ok=True)
    joblib.dump(pipeline, d / "classifier.joblib")
    bench = {"version": "1", **evaluate(pipeline, holdout)}
    (tmp_path / "classifier.json").write_text(json.dumps(bench))
    return d, bench


def register_challenger(pipeline, synth, privacy="passed"):
    """Like a retrain run: a generator and a classifier registered with alias `challenger`."""
    with start_candidate_run("stub", role="primary", stage="test") as run:
        run_id = run.info.run_id
    pkl = mlflow.get_tracking_uri().removeprefix("sqlite:///").replace("mlflow.db", "stub.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(StubGenerator(synth), f)
    registry.register_generator(pkl, run_id, {"generator": "stub", "privacy_gate": privacy},
                                alias=promotion.CHALLENGER)  # fmt: skip
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            pipeline,
            name="classifier_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    mv = mlflow.register_model(info.model_uri, promotion.CLASSIFIER)
    client = MlflowClient()
    client.set_model_version_tag(promotion.CLASSIFIER, mv.version, "source_generator", "stub")
    client.set_registered_model_alias(promotion.CLASSIFIER, promotion.CHALLENGER, mv.version)
    return str(mv.version)


def load_script():
    spec = importlib.util.spec_from_file_location("compare_promote", "scripts/compare_promote.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "auc,promote", [(0.88, True), (0.90, True), (0.87, True), (0.8699, False), (0.70, False)]
)
def test_rule_same_as_eval_gate_with_inclusive_boundary(auc, promote):
    d = promotion.decide({**SCORES, "version": "1"}, {**SCORES, "roc_auc": auc, "version": "2"},
                         0.01, "passed")  # fmt: skip
    assert d["promote"] is promote


def test_privacy_failure_blocks_a_better_challenger():
    d = promotion.decide({**SCORES, "version": "1"}, {**SCORES, "roc_auc": 0.95, "version": "2"},
                         0.01, "failed")  # fmt: skip
    assert d["promote"] is False and "privacy" in d["reasons"][0]
    assert "KEPT CHAMPION" in promotion.report_markdown(d)


def test_promotion_moves_aliases_and_keeps_previous(registry_dir, data):
    synth, holdout, good, _ = data
    deploy, bench = export(registry_dir, good, holdout)
    assert promotion.seed_champion(deploy, bench) == "1"
    assert register_challenger(good, synth) == "2"
    client = MlflowClient()
    champ = promotion.score(client, "champion", holdout)
    chall = promotion.score(client, "challenger", holdout)
    assert champ["roc_auc"] == pytest.approx(bench["roc_auc"])
    d = promotion.decide(champ, chall, 0.01, promotion.challenger_privacy(client))
    assert d["promote"] is True
    promotion.apply(client, d)
    by = promotion.version_for
    assert str(by(client, promotion.CLASSIFIER, "champion").version) == "2"
    assert str(by(client, promotion.CLASSIFIER, "previous").version) == "1"
    assert by(client, promotion.CLASSIFIER, "challenger") is None
    assert str(by(client, promotion.GENERATOR, "champion").version) == "1"
    assert by(client, promotion.CLASSIFIER, "champion").tags["promotion"] == "promoted"
    assert by(client, promotion.CLASSIFIER, "previous").tags["promotion"] == "replaced"


def test_worse_challenger_is_rejected_and_champion_stays(registry_dir, data):
    synth, holdout, good, worse = data
    deploy, bench = export(registry_dir, good, holdout)
    promotion.seed_champion(deploy, bench)
    register_challenger(worse, synth)
    client = MlflowClient()
    d = promotion.decide(promotion.score(client, "champion", holdout),
                         promotion.score(client, "challenger", holdout), 0.01,
                         promotion.challenger_privacy(client))  # fmt: skip
    assert d["promote"] is False and d["roc_auc_delta"] < -0.01
    promotion.apply(client, d)
    assert str(promotion.version_for(client, promotion.CLASSIFIER, "champion").version) == "1"
    rejected = client.get_model_version(promotion.CLASSIFIER, "2")
    assert rejected.tags["promotion"] == "rejected" and "ROC-AUC" in rejected.tags[
        "promotion_reasons"]  # fmt: skip


def test_script_promotes_and_writes_the_new_benchmark(registry_dir, data, monkeypatch):
    synth, holdout, good, _ = data
    deploy, _ = export(registry_dir, good, holdout)
    register_challenger(good, synth)
    script = load_script()
    monkeypatch.setattr(script, "load_processed", lambda: (synth, holdout, {}))
    out = registry_dir / "reports"
    bench_path = registry_dir / "classifier.json"
    args = ["--seed-champion", str(deploy), "--benchmark", str(bench_path), "--out-dir", str(out),
            "--tolerance", "0.01"]  # fmt: skip
    assert script.main(args) == 0
    assert "PROMOTED" in (out / "promotion.md").read_text()
    new = json.loads(bench_path.read_text())
    assert new["alias"] == "champion" and new["version"] == "1"  # challenger got v1, champion v2
    assert new["trained_on"] == "synthetic:stub"


def test_script_keeps_champion_with_exit_3_and_leaves_benchmark(registry_dir, data, monkeypatch):
    synth, holdout, good, worse = data
    deploy, _ = export(registry_dir, good, holdout)
    register_challenger(worse, synth)
    script = load_script()
    monkeypatch.setattr(script, "load_processed", lambda: (synth, holdout, {}))
    bench_path = registry_dir / "classifier.json"
    before = bench_path.read_text()
    args = ["--seed-champion", str(deploy), "--benchmark", str(bench_path),
            "--out-dir", str(registry_dir / "reports")]  # fmt: skip
    assert script.main(args) == script.KEPT_EXIT
    assert bench_path.read_text() == before
    report = json.loads((registry_dir / "reports" / "promotion.json").read_text())
    assert report["promote"] is False and report["reasons"]


def test_dry_run_changes_nothing(registry_dir, data, monkeypatch):
    synth, holdout, good, _ = data
    deploy, bench = export(registry_dir, good, holdout)
    promotion.seed_champion(deploy, bench)
    register_challenger(good, synth)
    script = load_script()
    monkeypatch.setattr(script, "load_processed", lambda: (synth, holdout, {}))
    args = ["--dry-run", "--benchmark", str(registry_dir / "classifier.json"),
            "--out-dir", str(registry_dir / "reports")]  # fmt: skip
    assert script.main(args) == 0
    client = MlflowClient()
    assert str(promotion.version_for(client, promotion.CLASSIFIER, "champion").version) == "1"
    assert str(promotion.version_for(client, promotion.CLASSIFIER, "challenger").version) == "2"


def test_missing_challenger_cannot_compare(registry_dir, data, monkeypatch):
    synth, holdout, good, _ = data
    deploy, _ = export(registry_dir, good, holdout)
    script = load_script()
    monkeypatch.setattr(script, "load_processed", lambda: (synth, holdout, {}))
    args = ["--seed-champion", str(deploy), "--benchmark", str(registry_dir / "classifier.json"),
            "--out-dir", str(registry_dir / "reports")]  # fmt: skip
    assert script.main(args) == 1


def test_seed_only_registers_the_deployed_model_as_v1(registry_dir, data):
    synth, holdout, good, _ = data
    deploy, _ = export(registry_dir, good, holdout)
    args = ["--seed-champion", str(deploy), "--seed-only",
            "--benchmark", str(registry_dir / "classifier.json")]  # fmt: skip
    assert load_script().main(args) == 0
    mv = promotion.version_for(MlflowClient(), promotion.CLASSIFIER, "champion")
    assert str(mv.version) == "1" and mv.tags["source"] == "deploy/model"
    assert not (registry_dir / "reports").exists()
