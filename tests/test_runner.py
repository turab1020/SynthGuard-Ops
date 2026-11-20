"""Smoke-sized tests of the shared training runner and the generate stage guard."""

import pickle

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from synthguard.config import load_params
from synthguard.data.prepare import clean, sdv_metadata
from synthguard.data.schema import COLUMNS
from synthguard.generators import runner
from synthguard.generators.marginal import MarginalSampler
from synthguard.generators.sdv_models import GaussianCopulaGenerator


@pytest.fixture
def train_df(raw_df):
    return clean(raw_df)


def test_training_frame_smoke_is_a_seeded_subsample():
    train = pd.DataFrame({"x": range(5000)})
    smoke = load_params(smoke=True)
    a, b = runner.training_frame(train, smoke), runner.training_frame(train, smoke)
    assert len(a) == smoke["smoke_rows"] == 1000
    pd.testing.assert_frame_equal(a, b)
    assert runner.training_frame(train, load_params(smoke=False)) is train


def test_n_synthetic_equals_holdout_size_capped_in_smoke():
    holdout = pd.DataFrame({"x": range(9762)})
    assert runner.n_synthetic(holdout, load_params(smoke=False)) == 9762
    assert runner.n_synthetic(holdout, load_params(smoke=True)) == 1000


def test_train_candidate_writes_outputs_and_logs_run(train_df, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(runner, "SYNTH_DIR", tmp_path / "synthetic")
    monkeypatch.setattr(runner, "MODELS_DIR", tmp_path / "models")
    params = load_params(smoke=True)

    result = runner.train_candidate(
        MarginalSampler(seed=3), train_df, 150, role="baseline", stage="test", params=params
    )

    synth = pd.read_csv(tmp_path / "synthetic" / "marginal.csv")
    assert len(synth) == 150 and list(synth.columns) == COLUMNS
    with open(tmp_path / "models" / "marginal.pkl", "rb") as f:
        assert len(pickle.load(f).sample(5)) == 5
    assert set(result) == {"run_id", "rows", "fit_seconds", "sample_seconds", "peak_memory_mb"}

    run = mlflow.get_run(result["run_id"])
    assert run.info.run_name == "marginal"
    assert {"generator": "marginal", "role": "baseline", "stage": "test"}.items() <= (
        run.data.tags.items()
    )
    assert {"git_commit", "data_train_md5", "data_holdout_md5", "dvc_lock_md5"} <= set(
        run.data.tags
    )
    assert run.data.params["seed"] == "3" and run.data.params["synthetic_rows"] == "150"
    assert {"fit_seconds", "sample_seconds", "peak_memory_mb"} <= set(run.data.metrics)
    client = MlflowClient()
    sample = client.download_artifacts(result["run_id"], "samples/synthetic_sample.csv", tmp_path)
    assert len(pd.read_csv(sample)) == 100
    assert "params.yaml" in [
        a.path.split("/")[-1] for a in client.list_artifacts(result["run_id"], "config")
    ]


def test_gaussian_copula_is_reproducible(train_df):
    """Same data + seed -> identical rows. (SDV samples with its own fixed seed, 73251, so the
    Gaussian Copula output does not depend on our `seed`; our seed drives CTGAN/TVAE training.)"""

    def sample(seed):
        return GaussianCopulaGenerator(sdv_metadata(), seed=seed).fit(train_df).sample(50)

    pd.testing.assert_frame_equal(sample(7), sample(7))


def test_sdv_generator_save_load_roundtrip(train_df, tmp_path):
    gen = GaussianCopulaGenerator(sdv_metadata(), seed=2).fit(train_df)
    gen.save(tmp_path / "gc.pkl")
    loaded = GaussianCopulaGenerator.load(tmp_path / "gc.pkl")
    pd.testing.assert_frame_equal(gen.sample(30), loaded.sample(30))


def test_generate_refuses_to_run_without_baselines(tmp_path, monkeypatch):
    from synthguard.stages import generate

    monkeypatch.setattr(generate, "BASELINES_PATH", tmp_path / "missing.json")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    def must_not_train(*_):
        pytest.fail("generate went past the baselines guard and started loading data")

    # if the guard ever breaks, fail fast instead of training on real data
    monkeypatch.setattr(generate, "load_processed", must_not_train)
    with pytest.raises(SystemExit, match="baselines"):
        generate.main()
