import pickle

import mlflow
import pandas as pd
import pytest
from conftest import make_raw
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from stub_generator import StubGenerator

from synthguard.data.prepare import clean
from synthguard.data.schema import COLUMNS, NUMERIC
from synthguard.tracking import registry
from synthguard.tracking.mlflow_utils import start_candidate_run

TAGS = {"generator": "stub", "selection_score": 0.9, "privacy_gate": "passed"}


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


@pytest.fixture
def stub_pickle(tracking):
    path = tracking / "stub.pkl"
    with open(path, "wb") as f:
        pickle.dump(StubGenerator(clean(make_raw(n=100))), f)
    return path


def candidate_run(name="stub"):
    with start_candidate_run(name, role="baseline", stage="test") as run:
        return run.info.run_id


def test_register_and_load_champion(stub_pickle):
    run_id = candidate_run()
    version = registry.register_generator(stub_pickle, run_id, TAGS)
    assert version == "1"

    model = registry.load_champion()
    rows = model.predict(pd.DataFrame({"n_rows": [10]}))
    assert len(rows) == 10
    assert list(rows.columns) == COLUMNS
    for c in NUMERIC:
        assert pd.api.types.is_integer_dtype(rows[c])

    mv = MlflowClient().get_model_version_by_alias(registry.MODEL_NAME, registry.ALIAS)
    assert str(mv.version) == "1"
    assert mv.run_id == run_id
    assert mv.tags["generator"] == "stub"
    assert mv.tags["selection_score"] == "0.9"
    assert mv.tags["privacy_gate"] == "passed"


def test_model_is_logged_in_the_candidate_run(stub_pickle):
    run_id = candidate_run()
    registry.register_generator(stub_pickle, run_id, TAGS)
    models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run_id}'", output_format="list"
    )
    assert [m.name for m in models] == [registry.MODEL_ARTIFACT]


def test_champion_alias_moves_to_newest_version(stub_pickle):
    registry.register_generator(stub_pickle, candidate_run("a"), TAGS)
    second = registry.register_generator(
        stub_pickle, candidate_run("b"), {**TAGS, "generator": "b"}
    )
    assert second == "2"
    mv = MlflowClient().get_model_version_by_alias(registry.MODEL_NAME, registry.ALIAS)
    assert str(mv.version) == "2"
    assert mv.tags["generator"] == "b"


@pytest.mark.parametrize("bad", [pd.DataFrame({"rows": [5]}), pd.DataFrame({"n_rows": [0]})])
def test_predict_rejects_bad_input(stub_pickle, bad):
    registry.register_generator(stub_pickle, candidate_run(), TAGS)
    model = registry.load_champion()
    with pytest.raises((MlflowException, ValueError)):
        model.predict(bad)
