"""A tiny stand-in generator for registry tests (resamples rows it was given)."""

import pandas as pd


class StubGenerator:
    name = "stub"

    def __init__(self, rows: pd.DataFrame, seed: int = 0):
        self.rows = rows.reset_index(drop=True)
        self.seed = seed

    def sample(self, n: int) -> pd.DataFrame:
        return self.rows.sample(n=n, replace=True, random_state=self.seed).reset_index(drop=True)
