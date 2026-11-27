"""Validate a new labelled batch and append it to the training split (retraining, D2-12).

The batch must pass the same pandera schema as the processed data. Rows identical to a hold-out row
are dropped, so the fixed hold-out stays unseen; rows already in train are dropped as duplicates.
The hold-out itself is never changed.

usage: python scripts/append_batch.py --batch data/new_batch.csv
       [--train data/processed/train.csv] [--holdout data/processed/holdout.csv]
       [--report reports/validation/new_batch.json]
exit code: 0 = appended, 1 = batch failed validation (nothing written)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from synthguard.config import PROJECT_ROOT
from synthguard.data.schema import COLUMNS, processed_schema
from synthguard.data.validate import validate

PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORT = PROJECT_ROOT / "reports" / "validation" / "new_batch.json"


def merge(
    train: pd.DataFrame, holdout: pd.DataFrame, batch: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """train + the batch rows that are neither in the hold-out nor already in train."""
    key = list(COLUMNS)
    seen_holdout = batch.merge(holdout[key].drop_duplicates(), on=key, how="left", indicator=True)
    in_holdout = (seen_holdout["_merge"] == "both").to_numpy()
    fresh = batch[~in_holdout].drop_duplicates()
    seen_train = fresh.merge(train[key].drop_duplicates(), on=key, how="left", indicator=True)
    new = fresh[(seen_train["_merge"] == "left_only").to_numpy()]
    merged = pd.concat([train[key], new[key]], ignore_index=True)
    stats = {"batch_rows": int(len(batch)), "dropped_in_holdout": int(in_holdout.sum()),
             "dropped_duplicates": int(len(batch) - in_holdout.sum() - len(new)),
             "appended": int(len(new)), "train_before": int(len(train)),
             "train_after": int(len(merged))}  # fmt: skip
    return merged, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--train", type=Path, default=PROCESSED / "train.csv")
    ap.add_argument("--holdout", type=Path, default=PROCESSED / "holdout.csv")
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args(argv)

    batch = pd.read_csv(args.batch)
    missing = [c for c in COLUMNS if c not in batch.columns]
    report = {"batch": str(args.batch)}
    if missing:
        report.update(passed=False, errors=[f"missing columns: {missing}"])
    else:
        batch = batch[COLUMNS]
        report.update(validate(batch, processed_schema(min_rows=1)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not report["passed"]:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"new batch FAILED validation -> {args.report}", file=sys.stderr)
        return 1

    merged, stats = merge(pd.read_csv(args.train), pd.read_csv(args.holdout), batch)
    merged.to_csv(args.train, index=False)
    args.report.write_text(json.dumps({**report, **stats}, indent=2))
    print(f"new batch valid: appended {stats['appended']} of {stats['batch_rows']} rows "
          f"({stats['dropped_in_holdout']} in hold-out, {stats['dropped_duplicates']} duplicates); "
          f"train {stats['train_before']} -> {stats['train_after']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
