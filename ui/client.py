"""HTTP client for the SynthGuard-Ops API, used by the Streamlit UI (ui/app.py).

Only needs `requests`: the form is built from the API's own OpenAPI schema, so the UI never
imports the model, pandera or the training code, and it always matches the deployed API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

DEFAULT_API = "https://synthguard-ops.onrender.com"
WAKE_TIMEOUT = 180  # seconds; a sleeping free Render instance takes about a minute to start
TIMEOUT = 30


def api_url() -> str:
    return os.environ.get("SYNTHGUARD_API_URL", DEFAULT_API).rstrip("/")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str  # "category" or "int"
    options: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    default: str | int | None = None


def field_specs(openapi: dict) -> list[FieldSpec]:
    """One FieldSpec per /predict input, in the API's order, with the schema example as default."""
    schema = openapi["components"]["schemas"]["PersonFeatures"]
    example = (schema.get("examples") or [{}])[0]
    specs = []
    for name, prop in schema["properties"].items():
        label = prop.get("title", name)
        if "enum" in prop:
            options = tuple(prop["enum"])
            default = example.get(name, options[0])
            specs.append(FieldSpec(name, label, "category", options=options, default=default))
            continue
        lo = prop.get("minimum")
        if lo is None and "exclusiveMinimum" in prop:
            lo = prop["exclusiveMinimum"] + 1
        hi = prop.get("maximum")
        lo = None if lo is None else int(lo)
        hi = None if hi is None else int(hi)
        default = example.get(name, lo if lo is not None else 0)
        specs.append(FieldSpec(name, label, "int", minimum=lo, maximum=hi, default=int(default)))
    return specs


def contribution_rows(explanation: dict) -> list[dict]:
    """SHAP bars for the chart: the top features plus one bar for all other features."""
    rows = [
        {
            "feature": f"{c['feature']} = {c['value']}",
            "contribution": float(c["contribution"]),
            "direction": "towards >50K" if c["contribution"] >= 0 else "towards <=50K",
        }
        for c in explanation["top_features"]
    ]
    other = float(explanation.get("other_features", 0.0))
    if other:
        direction = "towards >50K" if other >= 0 else "towards <=50K"
        rows.append(
            {"feature": "all other features", "contribution": other, "direction": direction}
        )
    return rows


class Api:
    def __init__(self, base: str | None = None, session: requests.Session | None = None):
        self.base = (base or api_url()).rstrip("/")
        self.http = session or requests.Session()

    def _get(self, path: str, timeout: float = TIMEOUT, **params) -> dict:
        r = self.http.get(f"{self.base}{path}", params=params or None, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def wake(self, timeout: float = WAKE_TIMEOUT, pause: float = 5.0) -> dict:
        """GET /health until it answers (a sleeping Render instance needs time to start)."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self._get("/health", timeout=min(60, timeout))
            except requests.RequestException:
                if time.monotonic() + pause > deadline:
                    raise
                time.sleep(pause)

    def info(self) -> dict:
        return self._get("/")

    def openapi(self) -> dict:
        return self._get("/openapi.json")

    def predict(self, features: dict, top_k: int = 5) -> dict:
        r = self.http.post(f"{self.base}/predict", params={"top_k": top_k}, json=features,
                           timeout=TIMEOUT)  # fmt: skip
        if r.status_code == 422:
            raise ValueError(_validation_message(r.json()))
        r.raise_for_status()
        return r.json()

    def generate(self, n: int = 10) -> dict:
        return self._get("/generate", timeout=120, n=n)


def _validation_message(body: dict) -> str:
    detail = body.get("detail")
    if not isinstance(detail, list):
        return str(detail)
    parts = []
    for d in detail:
        where = ".".join(str(p) for p in d.get("loc", []) if p != "body")
        parts.append(f"{where}: {d.get('msg')}")
    return "; ".join(parts)
