import pandas as pd
import pytest

from synthguard.data.prepare import clean, sdv_metadata
from synthguard.data.schema import CATEGORICAL, COLUMNS, NUMERIC, processed_schema
from synthguard.data.validate import validate
from synthguard.generators.base import Generator
from synthguard.generators.marginal import MarginalSampler
from synthguard.generators.sdv_models import GaussianCopulaGenerator


@pytest.fixture
def train_df(raw_df):
    return clean(raw_df)


def check_output(synth: pd.DataFrame, train: pd.DataFrame, n: int):
    assert len(synth) == n
    assert list(synth.columns) == COLUMNS
    for c in NUMERIC:
        assert pd.api.types.is_integer_dtype(synth[c]), c
    for c in CATEGORICAL:
        assert set(synth[c]) <= set(train[c]), c
    # synthetic data must satisfy the same processed schema as real data
    assert validate(synth, processed_schema(min_rows=1))["passed"]


def test_marginal_sampler(train_df):
    gen = MarginalSampler(seed=1).fit(train_df)
    synth = gen.sample(300)
    check_output(synth, train_df, 300)
    assert {"fit_seconds", "sample_seconds", "peak_memory_mb"} <= set(gen.costs)
    for c in NUMERIC:
        assert synth[c].between(train_df[c].min(), train_df[c].max()).all()


def test_marginal_is_seeded(train_df):
    a = MarginalSampler(seed=3).fit(train_df).sample(50)
    b = MarginalSampler(seed=3).fit(train_df).sample(50)
    pd.testing.assert_frame_equal(a, b)


def test_gaussian_copula(train_df):
    gen = GaussianCopulaGenerator(sdv_metadata(), seed=1).fit(train_df)
    check_output(gen.sample(200), train_df, 200)


def test_save_load_roundtrip(train_df, tmp_path):
    gen = MarginalSampler(seed=5).fit(train_df)
    gen.save(tmp_path / "m.pkl")
    loaded = Generator.load(tmp_path / "m.pkl")
    pd.testing.assert_frame_equal(gen.sample(20), loaded.sample(20))


@pytest.mark.parametrize("cls_name", ["CTGANGenerator", "TVAEGenerator"])
def test_neural_generators_smoke(train_df, cls_name):
    from synthguard.generators import sdv_models

    cls = getattr(sdv_models, cls_name)
    gen = cls(sdv_metadata(), seed=1, epochs=1, batch_size=100).fit(train_df)
    check_output(gen.sample(100), train_df, 100)
