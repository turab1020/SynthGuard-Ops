"""FastAPI inference service: live predictions with TreeSHAP explanations and Prometheus metrics.

Environment:
  SYNTHGUARD_MODEL_DIR        exported model folder (default: <repo>/deploy/model)
  SYNTHGUARD_ENABLE_GENERATE  "1" to enable /generate (needs torch + SDV; off on Render)
  SYNTHGUARD_GENERATOR_PATH   pickled generator for /generate (default: models/<champion>.pkl)
  SYNTHGUARD_REQUEST_LOG      optional JSONL file, one line per prediction (for drift checks)
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, create_model

from synthguard.data.schema import CATEGORIES, MISSING_CLEAN, RANGES
from synthguard.serving.explain import FEATURES, explain

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("SYNTHGUARD_MODEL_DIR", PROJECT_ROOT / "deploy" / "model"))


def generate_enabled() -> bool:
    return os.environ.get("SYNTHGUARD_ENABLE_GENERATE", "0").strip().lower() in {"1", "true", "yes"}


# ------------------------------------------------------------------ request schema (from pandera)
def _field(name: str):
    if name in CATEGORIES:
        allowed = tuple(CATEGORIES[name]) + (MISSING_CLEAN,)
        return (Literal.__getitem__(allowed), ...)
    if name in RANGES:
        lo, hi = RANGES[name]
        return (int, Field(..., ge=lo, le=hi))
    if name == "fnlwgt":
        return (int, Field(..., gt=0))
    return (int, Field(..., ge=0))  # capital_gain, capital_loss


EXAMPLE = {
    "age": 45, "workclass": "Private", "fnlwgt": 190000, "education": "Masters",
    "education_num": 14, "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 0,
    "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States",
}  # fmt: skip


PersonFeatures = create_model(
    "PersonFeatures",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]}),
    **{f: _field(f) for f in FEATURES},
)


class Contribution(BaseModel):
    feature: str
    value: str | int | float
    contribution: float


class Explanation(BaseModel):
    base_value: float
    logit: float
    top_features: list[Contribution]
    other_features: float


class Prediction(BaseModel):
    label: Literal[">50K", "<=50K"]
    probability_gt_50k: float
    confidence: float
    model_version: str
    explanation: Explanation


# ------------------------------------------------------------------ model + metrics
@lru_cache(maxsize=1)
def load_model():
    pipeline = joblib.load(MODEL_DIR / "classifier.joblib")
    meta_path = MODEL_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipeline, meta


REGISTRY = CollectorRegistry()
REQUESTS = Counter("synthguard_requests_total", "HTTP requests", ["endpoint", "method", "status"],
                   registry=REGISTRY)  # fmt: skip
ERRORS = Counter("synthguard_errors_total", "Requests that ended in a 5xx error", ["endpoint"],
                 registry=REGISTRY)  # fmt: skip
LATENCY = Histogram("synthguard_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
                    registry=REGISTRY)  # fmt: skip
PREDICTIONS = Counter("synthguard_predictions_total", "Predictions by label", ["label"],
                      registry=REGISTRY)  # fmt: skip
PROBABILITY = Histogram("synthguard_prediction_probability", "Predicted P(income > 50K)",
                        buckets=[i / 10 for i in range(1, 11)], registry=REGISTRY)  # fmt: skip
DRIFT_SHARE = Gauge("synthguard_drift_share", "Share of drifted input columns (latest check)",
                    registry=REGISTRY)  # fmt: skip

app = FastAPI(
    title="SynthGuard-Ops inference API",
    description="Income classifier trained only on synthetic data, with TreeSHAP explanations.",
    version="2.0",
)


@app.middleware("http")
async def observe(request: Request, call_next):
    endpoint = request.url.path if request.url.path in _ROUTES else "other"
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ERRORS.labels(endpoint).inc()
        REQUESTS.labels(endpoint, request.method, "500").inc()
        raise
    LATENCY.labels(endpoint).observe(time.perf_counter() - start)
    REQUESTS.labels(endpoint, request.method, str(response.status_code)).inc()
    if response.status_code >= 500:
        ERRORS.labels(endpoint).inc()
    return response


@app.get("/")
def root():
    links = {"docs": "/docs", "health": "/health", "predict": "/predict", "metrics": "/metrics"}
    return {"service": "synthguard-ops", **links, "generate_enabled": generate_enabled()}


@app.get("/health")
def health():
    try:
        _, meta = load_model()
    except Exception as exc:  # model missing or unreadable
        raise HTTPException(503, f"model not loaded: {exc}") from exc
    commit = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    return {"status": "ok", "model_version": str(meta.get("classifier", {}).get("version", "?")),
            "trained_on": meta.get("classifier", {}).get("trained_on"),
            "generate_enabled": generate_enabled(), "commit": commit}  # fmt: skip


@app.post("/predict", response_model=Prediction)
def predict(person: PersonFeatures, top_k: int = Query(5, ge=1, le=14)):  # type: ignore[valid-type]
    pipeline, meta = load_model()
    row = pd.DataFrame([person.model_dump()])[FEATURES]
    e = explain(pipeline, row, top_k=top_k)[0]
    p = e["probability"]
    label = ">50K" if p >= 0.5 else "<=50K"
    PREDICTIONS.labels(label).inc()
    PROBABILITY.observe(p)
    _log_request(person.model_dump(), label, p)
    return {
        "label": label, "probability_gt_50k": p, "confidence": max(p, 1 - p),
        "model_version": str(meta.get("classifier", {}).get("version", "?")),
        "explanation": {k: e[k] for k in ("base_value", "logit", "top_features", "other_features")},
    }  # fmt: skip


@app.get("/generate")
def generate(n: int = Query(10, ge=1, le=1000)):
    if not generate_enabled():
        raise HTTPException(404, "/generate is disabled here (set SYNTHGUARD_ENABLE_GENERATE=1)")
    gen = _load_generator()
    rows = gen.sample(n)
    _, meta = load_model()
    return {"generator": meta.get("generator", {}), "rows": rows.to_dict(orient="records")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


_ROUTES = {"/", "/health", "/predict", "/generate", "/metrics", "/docs", "/openapi.json"}
for _route in ("/health", "/predict", "/generate", "/metrics"):
    ERRORS.labels(_route)  # export 0 from the start, so dashboards show 0% instead of "no data"


@lru_cache(maxsize=1)
def _load_generator():
    import pickle  # torch + SDV are imported here, only when /generate is enabled

    default = (
        PROJECT_ROOT / "models" / f"{load_model()[1].get('generator', {}).get('name', 'tvae')}.pkl"
    )
    path = Path(os.environ.get("SYNTHGUARD_GENERATOR_PATH", default))
    with open(path, "rb") as f:
        return pickle.load(f)


def _log_request(features: dict, label: str, probability: float) -> None:
    path = os.environ.get("SYNTHGUARD_REQUEST_LOG")
    if not path:
        return
    record = {"ts": datetime.now(UTC).isoformat(), **features,
              "prediction": label, "probability_gt_50k": probability}  # fmt: skip
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
