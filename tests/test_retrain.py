"""Retraining (D2-12): challenger alias for the stages, and validating + appending a new batch."""

import importlib.util
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from conftest import make_raw
from mlflow import MlflowClient
from stub_generator import StubGenerator
from test_register_stage import evaluation

from synthguard.config import register_alias
from synthguard.data.prepare import clean
from synthguard.data.schema import COLUMNS
from synthguard.stages import register
from synthguard.stages.train_classifier import BENCHMARK, scores_path
from synthguard.tracking.mlflow_utils import start_candidate_run
from synthguard.tracking.registry import MODEL_NAME


def load_script():
    spec = importlib.util.spec_from_file_location("append_batch", "scripts/append_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def split(tmp_path):
    df = clean(make_raw(n=600, seed=3)).drop_duplicates().reset_index(drop=True)[COLUMNS]
    train, holdout, batch = df.iloc[:300], df.iloc[300:400], df.iloc[400:500].copy()
    paths = {}
    for name, frame in (("train", train), ("holdout", holdout)):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    return paths, train, holdout, batch


def run(script, tmp_path, paths, batch):
    batch.to_csv(tmp_path / "batch.csv", index=False)
    args = ["--batch", str(tmp_path / "batch.csv"), "--train", str(paths["train"]),
            "--holdout", str(paths["holdout"]),
            "--report", str(tmp_path / "report.json")]  # fmt: skip
    return script.main(args)


def test_alias_is_champion_unless_set(monkeypatch):
    monkeypatch.delenv("SYNTHGUARD_REGISTER_ALIAS", raising=False)
    assert register_alias() == "champion"
    monkeypatch.setenv("SYNTHGUARD_REGISTER_ALIAS", " ")
    assert register_alias() == "champion"
    monkeypatch.setenv("SYNTHGUARD_REGISTER_ALIAS", "challenger")
    assert register_alias() == "challenger"


def test_challenger_scores_never_overwrite_the_production_benchmark():
    assert scores_path("champion") == BENCHMARK
    assert scores_path("challenger").name == "classifier_challenger.json"
    assert scores_path("challenger") != BENCHMARK


def test_register_stage_uses_the_challenger_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("SYNTHGUARD_REGISTER_ALIAS", "challenger")
    metrics_dir, models_dir = tmp_path / "metrics", tmp_path / "models"
    metrics_dir.mkdir()
    models_dir.mkdir()
    with start_candidate_run("tvae", role="primary", stage="test") as r:
        run_id = r.info.run_id
    with open(models_dir / "tvae.pkl", "wb") as f:
        pickle.dump(StubGenerator(clean(make_raw(n=100))), f)
    (metrics_dir / "evaluation.json").write_text(json.dumps(evaluation()))
    (metrics_dir / "runs_baselines.json").write_text(json.dumps({}))
    (metrics_dir / "runs_primary.json").write_text(json.dumps({"tvae": run_id}))

    out = register.main(metrics_dir=metrics_dir, models_dir=models_dir)

    assert out["alias"] == "challenger"
    assert out["model_uri"] == f"models:/{MODEL_NAME}@challenger"
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(MODEL_NAME, "challenger").version) == "1"
    assert "champion" not in client.get_registered_model(MODEL_NAME).aliases


def test_valid_batch_is_appended(tmp_path, split):
    paths, train, _, batch = split
    assert run(load_script(), tmp_path, paths, batch) == 0
    merged = pd.read_csv(paths["train"])
    assert len(merged) == len(train) + len(batch) and list(merged.columns) == COLUMNS
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] and report["appended"] == len(batch)


def test_holdout_rows_and_duplicates_are_dropped(tmp_path, split):
    paths, train, holdout, batch = split
    batch = pd.concat([batch, holdout.iloc[:7], train.iloc[:5], batch.iloc[:3]])
    assert run(load_script(), tmp_path, paths, batch) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["dropped_in_holdout"] == 7
    assert report["dropped_duplicates"] == 8
    assert report["appended"] == 100
    merged = pd.read_csv(paths["train"])
    on = list(COLUMNS)
    assert merged.merge(holdout, on=on).empty  # the hold-out stays unseen


def test_invalid_batch_is_rejected_and_train_unchanged(tmp_path, split):
    paths, _, _, batch = split
    before = paths["train"].read_text()
    bad = batch.assign(age=np.where(np.arange(len(batch)) == 0, 5, batch["age"]))
    assert run(load_script(), tmp_path, paths, bad) == 1
    assert paths["train"].read_text() == before
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] is False and any(e["column"] == "age" for e in report["errors"])


def test_batch_missing_a_column_is_rejected(tmp_path, split):
    paths, _, _, batch = split
    assert run(load_script(), tmp_path, paths, batch.drop(columns="income")) == 1
    assert "income" in json.loads((tmp_path / "report.json").read_text())["errors"][0]
