"""DVC stage: validate raw data; write reports/validation/raw.json; exit 1 on failure."""

import json
import sys

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.data.schema import raw_schema
from synthguard.data.validate import read_raw, validate
from synthguard.stages.download import RAW_PATH

REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "raw.json"


def main() -> int:
    d = load_params()["data"]
    report = validate(read_raw(RAW_PATH), raw_schema(d["min_rows"], d["max_null_fraction"]))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    status = "PASSED" if report["passed"] else f"FAILED ({report['n_errors']} errors)"
    print(f"Raw validation {status} -> {REPORT_PATH}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
