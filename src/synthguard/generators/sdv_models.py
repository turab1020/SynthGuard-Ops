"""Wrappers around SDV 2.x single-table synthesizers.

SDV 2.x takes and returns data as {table_name: DataFrame}, even for a single table;
these wrappers hide that so every Generator works on plain DataFrames.
"""

from __future__ import annotations

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from synthguard.config import set_seed
from synthguard.generators.base import Generator


class SDVGenerator(Generator):
    synth_cls: type = GaussianCopulaSynthesizer

    def __init__(self, metadata: dict, seed: int = 42, **params):
        super().__init__(seed=seed, **params)
        self.metadata_dict = metadata

    def _fit(self, df: pd.DataFrame) -> None:
        set_seed(self.seed)
        metadata = Metadata.load_from_dict(self.metadata_dict)
        kwargs = dict(self.params)
        if self.synth_cls is not GaussianCopulaSynthesizer:
            kwargs.setdefault("enable_gpu", False)
        self.model = self.synth_cls(metadata, **kwargs)
        self.table = next(iter(self.metadata_dict["tables"]))
        self.model.fit({self.table: df})

    def _sample(self, n: int) -> pd.DataFrame:
        set_seed(self.seed)
        out = self.model.sample(num_rows=n)
        return out[self.table] if isinstance(out, dict) else out


class GaussianCopulaGenerator(SDVGenerator):
    name = "gaussian_copula"
    synth_cls = GaussianCopulaSynthesizer


class CTGANGenerator(SDVGenerator):
    name = "ctgan"
    synth_cls = CTGANSynthesizer


class TVAEGenerator(SDVGenerator):
    name = "tvae"
    synth_cls = TVAESynthesizer
