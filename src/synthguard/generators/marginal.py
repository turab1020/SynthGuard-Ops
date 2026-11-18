"""Rule-based baseline: sample every column independently from its own marginal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthguard.data.schema import NUMERIC
from synthguard.generators.base import Generator


class MarginalSampler(Generator):
    """Categorical columns: observed frequencies. Numeric columns: empirical quantiles.

    Keeps each column's distribution but destroys every relationship between columns,
    which makes it a useful lower bound for fidelity and utility.
    """

    name = "marginal"

    def _fit(self, df: pd.DataFrame) -> None:
        self.marginals = {}
        for c in df.columns:
            if c in NUMERIC:
                self.marginals[c] = ("num", np.sort(df[c].to_numpy()))
            else:
                freq = df[c].value_counts(normalize=True)
                self.marginals[c] = ("cat", (freq.index.to_numpy(), freq.to_numpy()))

    def _sample(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        out = {}
        for c, (kind, m) in self.marginals.items():
            if kind == "num":
                # inverted_cdf returns observed values, so integer columns stay integer
                out[c] = np.quantile(m, rng.random(n), method="inverted_cdf").astype(m.dtype)
            else:
                values, probs = m
                out[c] = rng.choice(values, size=n, p=probs)
        return pd.DataFrame(out)
