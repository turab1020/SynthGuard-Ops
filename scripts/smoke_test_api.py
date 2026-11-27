"""Smoke-test a running SynthGuard API: /health, one valid and one invalid /predict, /metrics.

usage: python scripts/smoke_test_api.py --url http://localhost:8000 [--wait 120]
Waits for the service (a sleeping Render instance can take ~1 min to wake). Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PERSON = {
    "age": 45, "workclass": "Private", "fnlwgt": 190000, "education": "Masters",
    "education_num": 14, "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 0,
    "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States",
}  # fmt: skip


def call(url: str, body: dict | None = None, timeout: float = 30) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--wait", type=float, default=120)
    args = ap.parse_args(argv)
    base = args.url.rstrip("/")
    deadline = time.time() + args.wait
    while True:
        try:
            status, body = call(base + "/health", timeout=10)
            if status == 200:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            status, body = 0, "unreachable"
        if time.time() > deadline:
            print(f"FAIL /health: {status} {body[:200]}")
            return 1
        time.sleep(3)
    print("ok   /health", body)
    checks = []
    status, body = call(base + "/predict", PERSON)
    ok = status == 200 and "top_features" in body
    checks.append(ok)
    print("ok  " if ok else "FAIL", "/predict valid", status, body[:160])
    status, _ = call(base + "/predict", {**PERSON, "age": 150})
    checks.append(status == 422)
    print("ok  " if status == 422 else "FAIL", "/predict invalid ->", status)
    status, body = call(base + "/metrics")
    ok = status == 200 and "synthguard_requests_total" in body
    checks.append(ok)
    print("ok  " if ok else "FAIL", "/metrics", status)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
