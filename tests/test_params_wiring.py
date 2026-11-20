"""params.yaml <-> dvc.yaml <-> code: every tunable value is tracked, so DVC re-runs exactly
the stages a parameter change affects."""

import inspect

import pytest
import yaml
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

from synthguard.config import PARAMS_PATH, PROJECT_ROOT, load_params
from synthguard.data.prepare import sdv_metadata
from synthguard.generators.sdv_models import CTGANGenerator, TVAEGenerator

DVC = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text())["stages"]
PARAMS = yaml.safe_load(PARAMS_PATH.read_text())


def stage_params(stage: str) -> set[str]:
    return set(DVC[stage].get("params", []))


def lookup(key: str):
    node = PARAMS
    for part in key.split("."):
        node = node[part]
    return node


def test_every_tracked_param_exists_in_params_yaml():
    for stage in DVC.values():
        for key in stage.get("params", []):
            lookup(key)  # KeyError -> a dvc.yaml param that params.yaml doesn't define


def test_generator_stages_track_their_hyperparameters():
    assert {"seed", "generators.ctgan", "generators.tvae"} <= stage_params("generate")
    assert {"seed", "generators.marginal", "generators.gaussian_copula"} <= stage_params(
        "baselines"
    )
    # a CTGAN/TVAE change must not invalidate the baselines
    assert not {"generators.ctgan", "generators.tvae"} & stage_params("baselines")


def test_data_params_are_tracked_by_the_data_stages():
    assert {"data.url", "data.member", "data.sha256"} <= stage_params("download")
    assert {"data.min_rows", "data.max_null_fraction"} <= stage_params("validate")
    assert {"seed", "data.test_size", "data.target"} <= stage_params("prepare")


@pytest.mark.parametrize("key,synth", [("ctgan", CTGANSynthesizer), ("tvae", TVAESynthesizer)])
def test_hyperparameters_are_real_sdv_arguments(key, synth):
    """A typo in params.yaml (e.g. `epoch`) must fail here, not be silently ignored."""
    accepted = set(inspect.signature(synth.__init__).parameters) - {"self", "metadata"}
    assert set(PARAMS["generators"][key]) <= accepted


@pytest.mark.parametrize("cls,key", [(CTGANGenerator, "ctgan"), (TVAEGenerator, "tvae")])
def test_wrapper_passes_params_through(cls, key, raw_df):
    from synthguard.data.prepare import clean

    params = {**load_params(smoke=True)["generators"][key], "batch_size": 100}
    gen = cls(sdv_metadata(), seed=1, **params).fit(clean(raw_df))
    model = gen.model._model  # ctgan stores _epochs/_batch_size, tvae epochs/batch_size
    epochs = getattr(model, "_epochs", getattr(model, "epochs", None))
    batch = getattr(model, "_batch_size", getattr(model, "batch_size", None))
    assert epochs == params["epochs"] == 1
    assert batch == 100
