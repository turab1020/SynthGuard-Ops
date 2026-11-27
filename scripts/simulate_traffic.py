"""Simulate live traffic for the drift demo (rows from UCI adult.test, never seen in training).

usage:
  python scripts/simulate_traffic.py batch --mode normal|drifted [--n 500] [--seed 42]
                                           [--out batch.jsonl | --csv data/new_batch.csv]
  python scripts/simulate_traffic.py send  --url https://... --mode normal|drifted
                                           [--n 500] [--seed 42]
`batch` writes one JSON object per line (the 14 /predict fields), or with --csv a CSV that
also keeps the real income label (the new batch for retraining). `send` POSTs each row to
/predict and prints a summary; then run scripts/check_drift.py against the same URL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

from synthguard.monitoring.simulate import MODES, load_test, make_batch
from synthguard.serving.explain import FEATURES


def send(url: str, rows: list[dict], timeout: float = 60) -> Counter:
    counts: Counter = Counter()
    with requests.Session() as s:
        for row in rows:
            r = s.post(f"{url.rstrip('/')}/predict", json=row, timeout=timeout)
            counts[r.status_code] += 1
            if r.ok:
                counts[r.json()["label"]] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batch", "send"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=MODES, required=True)
        p.add_argument("--n", type=int, default=500)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--source", type=Path, help="adult.test path (default: data/raw)")
    out = sub.choices["batch"].add_mutually_exclusive_group()
    out.add_argument("--out", type=Path)
    out.add_argument("--csv", type=Path, help="labelled CSV (new batch for retraining)")
    sub.choices["send"].add_argument("--url", required=True)
    args = ap.parse_args(argv)

    source = load_test(args.source) if args.source else load_test()
    batch = make_batch(source, args.mode, args.n, args.seed)
    if args.cmd == "batch" and args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        batch.to_csv(args.csv, index=False)
        print(f"wrote {len(batch)} labelled {args.mode} rows to {args.csv}")
        return 0
    rows = batch[FEATURES].to_dict(orient="records")
    if args.cmd == "batch":
        text = "".join(json.dumps(r) + "\n" for r in rows)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"wrote {len(rows)} {args.mode} rows to {args.out}")
        else:
            sys.stdout.write(text)
        return 0

    start = time.time()
    counts = send(args.url, rows)
    took = time.time() - start
    print(json.dumps({"mode": args.mode, "sent": len(rows), "seconds": round(took, 1),
                      "responses": {str(k): v for k, v in counts.items()}}, indent=2))  # fmt: skip
    return 0 if counts.get(200, 0) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
