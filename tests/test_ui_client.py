"""ui/client.py against the real FastAPI app (in-process), so the UI form always matches the API."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import requests
from conftest import make_raw
from fastapi.testclient import TestClient

from synthguard.data.prepare import clean
from synthguard.serving import app as service
from synthguard.serving.classifier import train
from synthguard.serving.explain import FEATURES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import client as ui  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    df = clean(make_raw(n=600, seed=4)).drop_duplicates().reset_index(drop=True)
    df["income"] = np.where(df["education_num"] >= 12, ">50K", "<=50K")
    pipe = train(df, {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}, seed=0)
    joblib.dump(pipe, d / "classifier.joblib")
    (d / "meta.json").write_text(json.dumps({"classifier": {"version": "7"}}))
    return d


@pytest.fixture
def api(model_dir, monkeypatch):
    monkeypatch.setattr(service, "MODEL_DIR", model_dir)
    monkeypatch.delenv("SYNTHGUARD_REQUEST_LOG", raising=False)
    service.load_model.cache_clear()
    yield ui.Api("http://testserver", session=TestClient(service.app))
    service.load_model.cache_clear()


def defaults(specs):
    return {s.name: s.default for s in specs}


def test_field_specs_cover_every_input_with_valid_defaults(api):
    specs = ui.field_specs(api.openapi())
    assert [s.name for s in specs] == list(api.openapi()["components"]["schemas"][
        "PersonFeatures"]["properties"])  # fmt: skip
    assert sorted(s.name for s in specs) == sorted(FEATURES)
    by = {s.name: s for s in specs}
    assert by["workclass"].kind == "category" and "Private" in by["workclass"].options
    assert by["age"].kind == "int" and (by["age"].minimum, by["age"].maximum) == (17, 90)
    assert by["fnlwgt"].minimum == 1  # exclusiveMinimum 0 -> smallest allowed integer
    for s in specs:
        if s.kind == "category":
            assert s.default in s.options
        else:
            assert s.minimum is None or s.default >= s.minimum
            assert s.maximum is None or s.default <= s.maximum


def test_predict_with_form_defaults(api):
    out = api.predict(defaults(ui.field_specs(api.openapi())), top_k=4)
    assert out["label"] in {">50K", "<=50K"}
    assert 0 <= out["probability_gt_50k"] <= 1
    assert out["model_version"] == "7"
    rows = ui.contribution_rows(out["explanation"])
    assert 4 <= len(rows) <= 5
    total = out["explanation"]["base_value"] + sum(r["contribution"] for r in rows)
    assert total == pytest.approx(out["explanation"]["logit"], abs=1e-3)


def test_predict_rejects_bad_input_with_a_readable_message(api):
    values = defaults(ui.field_specs(api.openapi()))
    values["age"] = 5
    with pytest.raises(ValueError, match="age"):
        api.predict(values)


def test_info_reports_whether_generate_is_enabled(api, monkeypatch):
    monkeypatch.setenv("SYNTHGUARD_ENABLE_GENERATE", "0")
    assert api.info()["generate_enabled"] is False
    monkeypatch.setenv("SYNTHGUARD_ENABLE_GENERATE", "1")
    assert api.info()["generate_enabled"] is True


def test_contribution_rows_direction_and_other_bar():
    exp = {"top_features": [{"feature": "age", "value": 45, "contribution": 0.4},
                            {"feature": "sex", "value": "Male", "contribution": -0.1}],
           "other_features": -0.05}  # fmt: skip
    rows = ui.contribution_rows(exp)
    assert [r["feature"] for r in rows] == ["age = 45", "sex = Male", "all other features"]
    assert [r["direction"] for r in rows] == ["towards >50K", "towards <=50K", "towards <=50K"]
    assert len(ui.contribution_rows({**exp, "other_features": 0.0})) == 2


class FlakySession:
    """Fails like a sleeping Render instance, then answers."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.ConnectionError("waking up")
        return FakeResponse({"status": "ok"})


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_wake_retries_until_the_api_answers():
    s = FlakySession(failures=2)
    assert ui.Api("http://x", session=s).wake(timeout=5, pause=0) == {"status": "ok"}
    assert s.calls == 3


def test_wake_gives_up_after_the_timeout():
    with pytest.raises(requests.ConnectionError):
        ui.Api("http://x", session=FlakySession(failures=10**6)).wake(timeout=0, pause=0.01)


def test_api_url_from_env(monkeypatch):
    monkeypatch.setenv("SYNTHGUARD_API_URL", "http://localhost:8000/")
    assert ui.api_url() == "http://localhost:8000"
    monkeypatch.delenv("SYNTHGUARD_API_URL")
    assert ui.api_url() == ui.DEFAULT_API
