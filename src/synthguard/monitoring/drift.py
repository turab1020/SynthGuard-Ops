"""Input and prediction drift of live traffic against the real training data (Evidently).

Reference = a seeded sample of the real training split (data/processed/train.csv) with the
served model's predicted probability added. Current = the API's recent requests
(GET /monitoring/requests). An alert is raised when the share of drifted input columns
reaches ``drift.threshold`` (params.yaml) or the predicted-probability distribution drifts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DataDriftTable
from evidently.report import Report

from synthguard.data.schema import CATEGORICAL, NUMERIC, TARGET
from synthguard.serving.explain import FEATURES

PREDICTION = "probability_gt_50k"


@dataclass
class DriftResult:
    share_of_drifted_columns: float
    number_of_drifted_columns: int
    number_of_columns: int
    dataset_drift: bool
    prediction_drift_score: float | None
    prediction_drift: bool
    drifted_columns: list[str]
    reference_rows: int
    current_rows: int

    @property
    def alert(self) -> bool:
        return self.dataset_drift or self.prediction_drift

    def to_dict(self) -> dict:
        return {**asdict(self), "alert": self.alert}

    def api_payload(self) -> dict:
        """Body for POST /monitoring/drift."""
        return {
            "share_of_drifted_columns": self.share_of_drifted_columns,
            "dataset_drift": self.dataset_drift,
            "prediction_drift_score": self.prediction_drift_score,
            "alert": self.alert,
        }


def reference_sample(train: pd.DataFrame, pipeline, n: int, seed: int) -> pd.DataFrame:
    """Seeded sample of real training rows (features only) plus the model's prediction."""
    sample = train.drop(columns=[TARGET], errors="ignore")
    sample = sample.sample(n=min(n, len(sample)), random_state=seed).reset_index(drop=True)
    sample = sample[FEATURES].copy()
    sample[PREDICTION] = pipeline.predict_proba(sample[FEATURES])[:, 1]
    return sample


def requests_frame(records: list[dict]) -> pd.DataFrame:
    """Recent API requests -> DataFrame with the same columns as the reference."""
    df = pd.DataFrame.from_records(records)
    missing = [c for c in [*FEATURES, PREDICTION] if c not in df.columns]
    if missing:
        raise ValueError(f"request records are missing columns: {missing}")
    return df[[*FEATURES, PREDICTION]].reset_index(drop=True)


def _report() -> Report:
    return Report(
        metrics=[
            DataDriftTable(columns=FEATURES),
            ColumnDriftMetric(column_name=PREDICTION),
        ]
    )


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float,
    html_path: Path | None = None,
    min_rows: int = 30,
) -> DriftResult:
    """Run Evidently on reference vs current; optionally save the HTML report."""
    if len(current) < min_rows:
        raise ValueError(f"need at least {min_rows} current rows, got {len(current)}")
    mapping = ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=[c for c in NUMERIC if c in FEATURES] + [PREDICTION],
        categorical_features=[c for c in CATEGORICAL if c in FEATURES],
    )
    report = _report()
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))
    dataset, prediction = (m["result"] for m in report.as_dict()["metrics"])
    share = float(dataset["share_of_drifted_columns"])
    drifted = [c for c, r in dataset["drift_by_columns"].items() if r["drift_detected"]]
    return DriftResult(
        share_of_drifted_columns=share,
        number_of_drifted_columns=int(dataset["number_of_drifted_columns"]),
        number_of_columns=int(dataset["number_of_columns"]),
        dataset_drift=share >= threshold,
        prediction_drift_score=float(prediction["drift_score"]),
        prediction_drift=bool(prediction["drift_detected"]),
        drifted_columns=sorted(drifted),
        reference_rows=len(reference),
        current_rows=len(current),
    )
