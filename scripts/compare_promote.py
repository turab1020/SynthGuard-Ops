"""Compare synthguard-classifier@champion with @challenger on the fixed hold-out; promote or keep.

Writes reports/promotion.md and reports/promotion.json, tags both versions in MLflow, and moves the
aliases. After a promotion it also writes the new production benchmark (metrics/classifier.json).

usage: python scripts/compare_promote.py [--tolerance 0.01] [--dry-run]
       [--seed-champion deploy/model [--seed-only]]   # empty registry (CI): deployed = champion
exit code: 0 = promoted, 3 = champion kept (not an error), 1 = could not compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlflow import MlflowClient

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.generators.runner import load_processed
from synthguard.tracking import promotion
from synthguard.tracking.mlflow_utils import setup_mlflow

KEPT_EXIT = 3
BENCHMARK = PROJECT_ROOT / "metrics" / "classifier.json"
REPORTS = PROJECT_ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, help="allowed ROC-AUC drop (default: gate.max_drop)")
    ap.add_argument("--seed-champion", help="register this exported model as @champion first")
    ap.add_argument("--seed-only", action="store_true", help="only seed the champion, then stop")
    ap.add_argument("--benchmark", default=str(BENCHMARK))
    ap.add_argument("--out-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="decide and report, change nothing")
    args = ap.parse_args(argv)

    tolerance = args.tolerance
    if tolerance is None:
        tolerance = load_params()["gate"]["max_drop"]
    setup_mlflow()
    client = MlflowClient()
    benchmark_path = Path(args.benchmark)
    if args.seed_champion:
        seeded = promotion.seed_champion(args.seed_champion, json.loads(benchmark_path.read_text()))
        print(f"seeded {promotion.CLASSIFIER}@champion = v{seeded} from {args.seed_champion}")
    if args.seed_only:
        return 0

    _, holdout, _ = load_processed()
    try:
        champion = promotion.score(client, promotion.CHAMPION, holdout)
        challenger = promotion.score(client, promotion.CHALLENGER, holdout)
    except LookupError as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 1
    decision = promotion.decide(champion, challenger, tolerance,
                                promotion.challenger_privacy(client))  # fmt: skip
    trained_on = None
    mv = promotion.version_for(client, promotion.CLASSIFIER, promotion.CHALLENGER)
    if mv is not None and mv.tags.get("source_generator"):
        trained_on = f"synthetic:{mv.tags['source_generator']}"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = promotion.report_markdown(decision)
    (out / "promotion.md").write_text(text)
    (out / "promotion.json").write_text(json.dumps({**decision, "dry_run": args.dry_run}, indent=2))
    print(text)
    if args.dry_run:
        return 0 if decision["promote"] else KEPT_EXIT
    promotion.apply(client, decision)
    if decision["promote"]:
        new = promotion.benchmark_for(decision, trained_on)
        benchmark_path.write_text(json.dumps(new, indent=2) + "\n")
        print(f"new production benchmark -> {benchmark_path}")
        return 0
    return KEPT_EXIT


if __name__ == "__main__":
    sys.exit(main())
