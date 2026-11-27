"""Simulated live traffic for the drift demo, built from UCI `adult.test`.

`adult.test` rows are real people the pipeline never saw (training and the hold-out both
come from `adult.data`), so normal batches are an honest "no drift" baseline. Drifted
batches apply a fixed, seeded shift: older, more educated, longer hours, and an occupation
mix skewed towards white-collar jobs. All values stay inside the API's allowed ranges.
Labels are carried over unchanged, so a drifted batch can also be used as "new data" for
retraining (data/new_batch.csv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synthguard.config import PROJECT_ROOT
from synthguard.data.download import download
from synthguard.data.prepare import clean
from synthguard.data.schema import TARGET
from synthguard.data.validate import read_raw
from synthguard.serving.explain import FEATURES

TEST_MEMBER = "adult.test"
TEST_SHA256 = "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "adult.test"
MODES = ("normal", "drifted")
# drifted: this share of rows gets an occupation drawn from WHITE_COLLAR
OCCUPATION_SHIFT = 0.6
WHITE_COLLAR = ["Exec-managerial", "Prof-specialty", "Tech-support"]


def load_test(path: Path = TEST_PATH, url: str | None = None) -> pd.DataFrame:
    """Read (downloading once if needed) and clean adult.test: 14 features + income."""
    if not path.exists():
        if url is None:
            from synthguard.config import load_params

            url = load_params()["data"]["url"]
        download(url, TEST_MEMBER, TEST_SHA256, path)
    raw = read_raw_test(path)
    return clean(raw)[[*FEATURES, TARGET]].reset_index(drop=True)


def read_raw_test(path: Path) -> pd.DataFrame:
    """adult.test starts with a junk line ('|1x3 Cross validator') and labels end with '.'."""
    lines = path.read_text().splitlines()
    body = [ln for ln in lines[1:] if ln.strip()]
    tmp = path.with_name(path.name + ".body")
    tmp.write_text("\n".join(body) + "\n")
    try:
        return read_raw(tmp)
    finally:
        tmp.unlink()


def shift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the drift: age +15 (≤90), hours ×1.4 (≤99), education_num +2 (≤16), occupations."""
    d = df.copy()
    d["age"] = (d["age"] + 15).clip(upper=90)
    d["hours_per_week"] = (d["hours_per_week"] * 1.4).round().clip(upper=99).astype("int64")
    d["education_num"] = (d["education_num"] + 2).clip(upper=16)
    moved = rng.random(len(d)) < OCCUPATION_SHIFT
    d.loc[moved, "occupation"] = rng.choice(WHITE_COLLAR, size=int(moved.sum()))
    return d


def make_batch(source: pd.DataFrame, mode: str, n: int, seed: int) -> pd.DataFrame:
    """n rows sampled (seeded) from `source`; shifted when mode == 'drifted'."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(source), size=n, replace=n > len(source))
    batch = source.iloc[idx].reset_index(drop=True)
    return shift(batch, rng) if mode == "drifted" else batch
