"""DVC stage: clean + split raw data; validate outputs; write metrics/prepare.json."""

import json
import sys

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.data.prepare import clean, positive_rate, sdv_metadata, split
from synthguard.data.schema import COLUMNS, processed_schema
from synthguard.data.validate import read_raw, validate
from synthguard.stages.download import RAW_PATH

OUT_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_PATH = PROJECT_ROOT / "metrics" / "prepare.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "processed.json"


def main() -> int:
    p = load_params()
    d = p["data"]
    raw = read_raw(RAW_PATH)
    cleaned = clean(raw)
    train, holdout, n_dup = split(cleaned, d["test_size"], p["seed"])

    reports = {}
    for name, df in [("train", train), ("holdout", holdout)]:
        reports[name] = validate(df[COLUMNS], processed_schema(min_rows=1))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(reports, indent=2))
    if not all(r["passed"] for r in reports.values()):
        print(f"Processed validation FAILED -> {REPORT_PATH}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    holdout.to_csv(OUT_DIR / "holdout.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(json.dumps(sdv_metadata(), indent=2))

    metrics = {
        "raw_rows": len(raw),
        "duplicates_removed": n_dup,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_positive_rate": round(positive_rate(train), 4),
        "holdout_positive_rate": round(positive_rate(holdout), 4),
        "unknown_cells_train": int((train == "Unknown").sum().sum()),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
