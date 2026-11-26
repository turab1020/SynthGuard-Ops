"""Evaluation gate for CI/CD: block a candidate classifier that is worse than production.

usage: python scripts/eval_gate.py --candidate <model.joblib | models:/... URI>
                                   [--benchmark metrics/classifier.json]
                                   [--evaluation metrics/evaluation.json] [--summary gate.md]
Exit code 0 = passed, 1 = blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.generators.runner import load_processed
from synthguard.serving.classifier import evaluate
from synthguard.serving.gate import check, summary_markdown


def load_candidate(ref: str):
    if ref.endswith(".joblib"):
        return joblib.load(ref)
    import mlflow.sklearn

    return mlflow.sklearn.load_model(ref)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--benchmark", default=str(PROJECT_ROOT / "metrics" / "classifier.json"))
    ap.add_argument("--evaluation", default=str(PROJECT_ROOT / "metrics" / "evaluation.json"))
    ap.add_argument("--summary")
    args = ap.parse_args(argv)

    max_drop = load_params()["gate"]["max_drop"]
    benchmark = json.loads(Path(args.benchmark).read_text())
    privacy = json.loads(Path(args.evaluation).read_text())["selection"]["privacy_gate"]
    _, holdout, _ = load_processed()
    candidate = evaluate(load_candidate(args.candidate), holdout)
    result = check(candidate, benchmark, max_drop, privacy)
    text = summary_markdown(candidate, benchmark, result, max_drop)
    print(text)
    if args.summary:
        Path(args.summary).write_text(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
