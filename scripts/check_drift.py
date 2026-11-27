"""Drift check of live traffic against the real training data; closes the monitoring loop.

usage: python scripts/check_drift.py (--url https://... | --requests-file requests.jsonl)
                                     [--out-dir reports/drift] [--report] [--dispatch]

1. Fetch the API's recent predictions (GET /monitoring/requests) or read a JSONL file.
2. Run Evidently vs a seeded sample of data/processed/train.csv scored by deploy/model.
3. Write <out-dir>/drift.json and <out-dir>/report.html.
4. --report: POST the result to /monitoring/drift (needs SYNTHGUARD_DRIFT_TOKEN), so the
   Prometheus gauges and the Grafana dashboard show it.
5. --dispatch: on an alert, send a GitHub repository_dispatch "drift-alert" (needs
   GITHUB_TOKEN and GITHUB_REPOSITORY), which starts the retraining workflow.
Exit code 0 = no alert, 3 = alert (so a workflow can branch on it), other = error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.monitoring.drift import check_drift, reference_sample, requests_frame

ALERT_EXIT = 3
MODEL = PROJECT_ROOT / "deploy" / "model" / "classifier.joblib"
TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv"


def fetch_requests(url: str, limit: int) -> list[dict]:
    resp = requests.get(f"{url.rstrip('/')}/monitoring/requests", params={"limit": limit},
                        timeout=120)  # fmt: skip
    resp.raise_for_status()
    return resp.json()["requests"]


def read_requests(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report_to_api(url: str, payload: dict, token: str) -> None:
    resp = requests.post(f"{url.rstrip('/')}/monitoring/drift", json=payload,
                         headers={"X-Drift-Token": token}, timeout=60)  # fmt: skip
    resp.raise_for_status()


def dispatch_alert(repo: str, token: str, result: dict) -> None:
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/dispatches",
        json={"event_type": "drift-alert", "client_payload": {"drift": result}},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=60,
    )
    resp.raise_for_status()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--requests-file", type=Path)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "drift")
    ap.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "deploy" / "model" / "classifier.joblib"
    )
    ap.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train.csv")
    ap.add_argument("--report", action="store_true", help="POST the result to the API")
    ap.add_argument("--dispatch", action="store_true", help="repository_dispatch on alert")
    args = ap.parse_args(argv)

    params = load_params()
    cfg = params["drift"]
    if args.report and not args.url:
        ap.error("--report needs --url")

    records = (fetch_requests(args.url, cfg["window"]) if args.url
               else read_requests(args.requests_file))  # fmt: skip
    current = requests_frame(records)
    pipeline = joblib.load(args.model)
    reference = reference_sample(pd.read_csv(args.train), pipeline, cfg["reference_rows"],
                                 params["seed"])  # fmt: skip
    result = check_drift(reference, current, cfg["threshold"],
                         html_path=args.out_dir / "report.html",
                         min_rows=cfg["min_requests"])  # fmt: skip
    out = result.to_dict()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "drift.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

    if args.report:
        token = os.environ.get("SYNTHGUARD_DRIFT_TOKEN")
        if not token:
            print("SYNTHGUARD_DRIFT_TOKEN is not set; not reporting to the API", file=sys.stderr)
        else:
            report_to_api(args.url, result.api_payload(), token)
            print("reported to the API")
    if result.alert and args.dispatch:
        repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
        if not (repo and token):
            print("GITHUB_REPOSITORY/GITHUB_TOKEN not set; no dispatch", file=sys.stderr)
        else:
            dispatch_alert(repo, token, out)
            print(f"drift-alert dispatched to {repo}")
    return ALERT_EXIT if result.alert else 0


if __name__ == "__main__":
    sys.exit(main())
