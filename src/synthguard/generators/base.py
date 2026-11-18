"""Common interface for every synthetic data generator, with cost measurement."""

from __future__ import annotations

import pickle
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import psutil


class _PeakRSS:
    """Track the peak resident memory of this process while the block runs."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20


class Generator(ABC):
    """fit(df) -> sample(n). Records fit_seconds, sample_seconds and peak_memory_mb."""

    name: str = "generator"

    def __init__(self, seed: int = 42, **params):
        self.seed = seed
        self.params = params
        self.costs: dict[str, float] = {}

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def _sample(self, n: int) -> pd.DataFrame: ...

    def fit(self, df: pd.DataFrame) -> Generator:
        self.columns = list(df.columns)
        with _PeakRSS() as mem:
            start = time.perf_counter()
            self._fit(df)
            self.costs["fit_seconds"] = time.perf_counter() - start
        self.costs["peak_memory_mb"] = mem.peak_mb
        return self

    def sample(self, n: int) -> pd.DataFrame:
        start = time.perf_counter()
        out = self._sample(n)[self.columns]
        self.costs["sample_seconds"] = time.perf_counter() - start
        return out.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> Generator:
        with open(path, "rb") as f:
            return pickle.load(f)
