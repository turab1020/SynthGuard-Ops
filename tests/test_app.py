import json
import pickle
import subprocess
import sys

import joblib
import numpy as np
import pytest
from conftest import make_raw
from fastapi.testclient import TestClient
from stub_generator import StubGenerator

from synthguard.data.prepare import clean
from synthguard.serving import app as service
from synthguard.serving.classifier import train
from synthguard.serving.explain import FEATURES

PERSON = {
    "age": 45, "workclass": "Private", "fnlwgt": 190000, "education": "Masters",
    "education_num": 14, "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 0,
    "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States",
}  # fmt: skip


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    df = clean(make_raw(n=600, seed=4)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    pipe = train(df, {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}, seed=0)
    joblib.dump(pipe, d / "classifier.joblib")
    meta = {"classifier": {"version": "7", "trained_on": "synthetic:tvae"},
            "generator": {"name": "stub", "selection_score": 0.9}}  # fmt: skip
    (d / "meta.json").write_text(json.dumps(meta))
    with open(d / "generator.pkl", "wb") as f:
        pickle.dump(StubGenerator(df), f)
    return d, pipe


@pytest.fixture
def client(model_dir, monkeypatch, tmp_path):
    d, _ = model_dir
    monkeypatch.setattr(service, "MODEL_DIR", d)
    monkeypatch.setenv("SYNTHGUARD_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    service.load_model.cache_clear()
    service._load_generator.cache_clear()
    return TestClient(service.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "7", "trained_on": "synthetic:tvae",
                        "generate_enabled": False}  # fmt: skip


def test_predict_returns_label_probability_and_shap(client, model_dir):
    _, pipe = model_dir
    r = client.post("/predict", json=PERSON)
    assert r.status_code == 200
    body = r.json()
    import pandas as pd

    expected = pipe.predict_proba(pd.DataFrame([PERSON])[FEATURES])[0, 1]
    assert body["probability_gt_50k"] == pytest.approx(expected, abs=1e-5)
    assert body["label"] == (">50K" if expected >= 0.5 else "<=50K")
    assert body["confidence"] == pytest.approx(max(expected, 1 - expected), abs=1e-5)
    assert body["model_version"] == "7"
    top = body["explanation"]["top_features"]
    assert len(top) == 5 and top[0]["feature"] in FEATURES
    assert client.post("/predict?top_k=2", json=PERSON).json()["explanation"]["top_features"][1]


@pytest.mark.parametrize(
    "change",
    [{"age": 150}, {"race": "Martian"}, {"hours_per_week": 0}, {"fnlwgt": -1}, {"extra": 1}],
)
def test_invalid_input_is_rejected(client, change):
    assert client.post("/predict", json={**PERSON, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in PERSON.items() if k != "occupation"}
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_category_value_is_allowed(client):
    assert client.post("/predict", json={**PERSON, "occupation": "Unknown"}).status_code == 200


def test_metrics_count_requests_latency_and_predictions(client):
    client.post("/predict", json=PERSON)
    client.post("/predict", json={**PERSON, "age": 150})
    text = client.get("/metrics").text
    assert 'synthguard_requests_total{endpoint="/predict",method="POST",status="200"}' in text
    assert 'synthguard_requests_total{endpoint="/predict",method="POST",status="422"}' in text
    assert "synthguard_request_latency_seconds_bucket" in text
    assert "synthguard_predictions_total" in text and "synthguard_prediction_probability" in text
    assert "synthguard_errors_total" in text and "synthguard_drift_share" in text


def test_request_log_records_inputs_and_prediction(client, tmp_path):
    client.post("/predict", json=PERSON)
    line = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[-1])
    assert line["age"] == 45 and line["prediction"] in (">50K", "<=50K")


def test_generate_is_off_by_default(client):
    r = client.get("/generate?n=3")
    assert r.status_code == 404 and "SYNTHGUARD_ENABLE_GENERATE" in r.json()["detail"]


def test_generate_when_enabled(client, model_dir, monkeypatch):
    d, _ = model_dir
    monkeypatch.setenv("SYNTHGUARD_ENABLE_GENERATE", "1")
    monkeypatch.setenv("SYNTHGUARD_GENERATOR_PATH", str(d / "generator.pkl"))
    r = client.get("/generate?n=4")
    assert r.status_code == 200 and len(r.json()["rows"]) == 4
    assert client.get("/generate?n=5000").status_code == 422


def test_serving_import_does_not_load_torch_or_sdv():
    code = ("import sys, synthguard.serving.app; "
            "bad=[m for m in ('torch','sdv','ctgan','rdt') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")  # fmt: skip
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
