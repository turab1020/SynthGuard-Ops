"""scripts/export_models.py (registry -> deploy/model) and scripts/smoke_test_api.py."""

import importlib.util
import json
import socket
import threading
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pytest
import uvicorn
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from synthguard.tracking.mlflow_utils import setup_mlflow


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ export_models.py
def register(name, tags=None):
    X, y = np.zeros((4, 1)), np.array([0, 1, 0, 1])
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            DummyClassifier().fit(X, y),
            name="m",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    mv = mlflow.register_model(info.model_uri, name, tags=tags)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)
    return mv


def test_export_writes_the_champion_and_its_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    setup_mlflow()
    exp = load("export_models")
    root = tmp_path / "repo"
    (root / "metrics").mkdir(parents=True)
    bench = {"roc_auc": 0.88, "f1": 0.62, "accuracy": 0.83, "precision": 0.67, "recall": 0.58,
             "trained_on": "synthetic:tvae"}  # fmt: skip
    (root / "metrics" / "classifier.json").write_text(json.dumps(bench))
    cand = {"selection_score": 0.9, "quality_score": 0.84, "utility_ratio": 0.96,
            "mia_auc": 0.49, "exact_match_rate": 0.0, "dcr_ratio": 0.63,
            "privacy_gate": "passed", "extra": 1}  # fmt: skip
    (root / "metrics" / "evaluation.json").write_text(json.dumps({"candidates": {"tvae": cand}}))
    monkeypatch.setattr(exp, "PROJECT_ROOT", root)
    register(exp.GENERATOR, tags={"generator": "tvae"})
    register(exp.CLASSIFIER)
    register(exp.CLASSIFIER)  # v2 is the champion now

    out = exp.main(["--out", str(tmp_path / "model")])

    meta = json.loads((out / "meta.json").read_text())
    assert meta["classifier"]["version"] == "2"
    assert meta["classifier"]["holdout"]["roc_auc"] == 0.88
    assert meta["classifier"]["trained_on"] == "synthetic:tvae"
    assert meta["generator"]["name"] == "tvae" and meta["generator"]["privacy_gate"] == "passed"
    assert "extra" not in meta["generator"]
    assert set(meta["libraries"]) == {"python", "scikit-learn", "xgboost", "pandas"}
    assert joblib.load(out / "classifier.joblib").predict(np.zeros((2, 1))).shape == (2,)
    text = (out / "meta.json").read_text()
    assert str(tmp_path) not in text and "artifact_uri" not in text  # no local paths shipped


# ------------------------------------------------------------------ smoke_test_api.py
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api():
    """The real API (committed deploy/model) served by uvicorn on a free local port."""
    from synthguard.serving.app import app

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_smoke_test_passes_against_the_real_api(live_api, capsys):
    assert load("smoke_test_api").main(["--url", live_api + "/", "--wait", "10"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "/predict invalid -> 422" in out


def test_smoke_test_fails_when_nothing_answers():
    url = f"http://127.0.0.1:{free_port()}"
    assert load("smoke_test_api").main(["--url", url, "--wait", "0"]) == 1


def test_smoke_test_fails_when_a_check_fails(live_api, monkeypatch):
    smoke = load("smoke_test_api")
    real = smoke.call

    def broken_metrics(url, body=None, timeout=30):
        return (500, "") if url.endswith("/metrics") else real(url, body, timeout)

    monkeypatch.setattr(smoke, "call", broken_metrics)
    assert smoke.main(["--url", live_api, "--wait", "10"]) == 1
