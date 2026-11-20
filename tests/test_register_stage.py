import json
import pickle

import mlflow
import pandas as pd
import pytest
from conftest import make_raw
from mlflow import MlflowClient
from stub_generator import StubGenerator

from synthguard.data.prepare import clean
from synthguard.data.schema import COLUMNS
from synthguard.stages import register
from synthguard.tracking.mlflow_utils import start_candidate_run
from synthguard.tracking.registry import ALIAS, MODEL_NAME

TAGS = {"git_commit": "abc123", "data_train_md5": "d41d8cd9"}


def evaluation(selected="tvae", gate="passed"):
    cand = {
        "selection_score": 0.9, "quality_score": 0.85, "utility_ratio": 0.95, "mia_auc": 0.5,
        "exact_match_rate": 0.0, "privacy_gate": gate, "fit_seconds": 3.0,
    }  # fmt: skip
    return {
        "candidates": {"ctgan": {**cand, "selection_score": 0.8}, "tvae": cand},
        "selection": {"generator": selected, "privacy_gate": gate, "selection_score": 0.9},
    }


def test_plan_uses_the_recorded_selection():
    plan = register.plan_registration(evaluation(), {"ctgan": "r1", "tvae": "r2"}, TAGS)
    assert plan["generator"] == "tvae" and plan["run_id"] == "r2"
    assert plan["version_tags"] == {
        "generator": "tvae", "selection_score": "0.900000", "privacy_gate": "passed",
        "git_commit": "abc123", "data_train_md5": "d41d8cd9",
    }  # fmt: skip
    assert plan["metrics"]["utility_ratio"] == 0.95


def test_failed_gate_is_carried_into_the_tags():
    plan = register.plan_registration(evaluation(gate="failed"), {"tvae": "r2"}, TAGS)
    assert plan["version_tags"]["privacy_gate"] == "failed"


def test_missing_run_id_is_an_error():
    with pytest.raises(KeyError):
        register.plan_registration(evaluation(), {"ctgan": "r1"}, TAGS)


def test_stage_registers_champion_and_writes_registered_json(tmp_path, monkeypatch):
    """Smoke-sized end-to-end: evaluation.json -> registry -> champion predicts 10 rows."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    metrics_dir, models_dir = tmp_path / "metrics", tmp_path / "models"
    metrics_dir.mkdir()
    models_dir.mkdir()
    with start_candidate_run("tvae", role="primary", stage="test") as run:
        run_id = run.info.run_id
    with open(models_dir / "tvae.pkl", "wb") as f:
        pickle.dump(StubGenerator(clean(make_raw(n=100))), f)
    (metrics_dir / "evaluation.json").write_text(json.dumps(evaluation()))
    (metrics_dir / "runs_baselines.json").write_text(json.dumps({}))
    (metrics_dir / "runs_primary.json").write_text(json.dumps({"tvae": run_id}))

    out = register.main(metrics_dir=metrics_dir, models_dir=models_dir)

    saved = json.loads((metrics_dir / "registered.json").read_text())
    assert saved == out
    assert saved["generator"] == "tvae" and saved["version"] == "1"
    assert saved["model_uri"] == f"models:/{MODEL_NAME}@{ALIAS}"
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    assert mv.run_id == run_id and mv.tags["privacy_gate"] == "passed"
    rows = mlflow.pyfunc.load_model(saved["model_uri"]).predict(pd.DataFrame({"n_rows": [10]}))
    assert len(rows) == 10 and list(rows.columns) == COLUMNS
