"""scripts/rollback.py: restore the previously served model from git, or move registry aliases."""

import importlib.util
import json
import subprocess

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from synthguard.tracking.mlflow_utils import setup_mlflow


def load_script():
    spec = importlib.util.spec_from_file_location("rollback", "scripts/rollback.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = load_script()


def alias(client, name):
    aliases = client.get_registered_model(rb.CLASSIFIER).aliases
    return client.get_model_version_by_alias(rb.CLASSIFIER, name) if name in aliases else None


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def serve(repo, version, auc, msg):
    (repo / "deploy" / "model").mkdir(parents=True, exist_ok=True)
    (repo / "metrics").mkdir(exist_ok=True)
    meta = {"classifier": {"version": version, "trained_on": f"synthetic:v{version}"}}
    (repo / "deploy" / "model" / "meta.json").write_text(json.dumps(meta))
    (repo / "deploy" / "model" / "classifier.joblib").write_text(f"model {version}")
    (repo / "metrics" / "classifier.json").write_text(json.dumps({"roc_auc": auc}))
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    first = serve(tmp_path, "1", 0.88, "model v1")
    (tmp_path / "README.md").write_text("unrelated change")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "docs")
    second = serve(tmp_path, "2", 0.885, "promote v2")
    return tmp_path, first, second


def test_git_rollback_restores_the_model_before_the_last_promotion(repo):
    path, first, second = repo
    result = rb.git_rollback(cwd=path)
    assert result["from"]["version"] == "2" and result["to"]["version"] == "1"
    assert result["undone_commit"] == second
    assert (path / "deploy" / "model" / "classifier.joblib").read_text() == "model 1"
    assert json.loads((path / "metrics" / "classifier.json").read_text())["roc_auc"] == 0.88
    assert (path / "README.md").read_text() == "unrelated change"  # nothing else touched
    text = rb.report_markdown(result)
    assert "v2 | v1" in text and second[:7] in text


def test_git_rollback_to_an_explicit_ref(repo):
    path, first, _ = repo
    result = load_script().git_rollback(to=first, cwd=path)
    assert result["restored_from"] == first and result["undone_commit"] is None
    assert result["to"]["version"] == "1"


def test_git_rollback_without_history_fails(tmp_path):
    git(tmp_path, "init", "-q")
    with pytest.raises(SystemExit, match="nothing to roll back"):
        load_script().last_model_change(cwd=tmp_path)


def register(version_alias):
    X, y = np.zeros((4, 1)), np.array([0, 1, 0, 1])
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            DummyClassifier().fit(X, y),
            name="m",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    mv = mlflow.register_model(info.model_uri, rb.CLASSIFIER)
    MlflowClient().set_registered_model_alias(rb.CLASSIFIER, version_alias, mv.version)


def test_registry_rollback_moves_champion_back_to_previous(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    setup_mlflow()
    register("previous")
    register("champion")
    moved = load_script().registry_rollback()
    assert moved[rb.CLASSIFIER] == {"from": "2", "to": "1"}
    client = MlflowClient()
    assert str(alias(client, "champion").version) == "1"
    assert alias(client, "previous") is None
    assert client.get_model_version(rb.CLASSIFIER, "2").tags["promotion"] == "rolled_back"


def test_registry_rollback_needs_a_previous_version(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    setup_mlflow()
    register("champion")
    with pytest.raises(SystemExit, match="nothing to roll back"):
        load_script().registry_rollback()
