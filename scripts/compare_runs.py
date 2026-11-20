"""Compare determinism-check results with each other and with the committed pipeline outputs.

usage: python scripts/compare_runs.py <result.json> [<result.json> ...]
Prints a markdown table (also appended to $GITHUB_STEP_SUMMARY when set).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def committed() -> dict:
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())["stages"]["generate"]["outs"]
    csv_md5 = {Path(o["path"]).stem: o["md5"] for o in lock if o["path"].endswith(".csv")}
    ev = json.loads((ROOT / "metrics" / "evaluation.json").read_text())["candidates"]
    return {
        g: {"csv_md5": csv_md5[g], "selection_score": ev[g]["selection_score"]} for g in csv_md5
    }


def main() -> None:
    runs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    ref = committed()
    header = "| run | mode | threads | gen | csv md5 | = committed? | score | Δ vs committed |"
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for g in ("ctgan", "tvae"):
            x = r[g]
            same = "yes" if x["csv_md5"] == ref[g]["csv_md5"] else "no"
            delta = x["selection_score"] - ref[g]["selection_score"]
            score = x["selection_score"]
            lines.append(
                f"| {r['label']} | {r['mode']} | {r['torch_threads']} | {g} | "
                f"`{x['csv_md5'][:10]}` | {same} | {score:.6f} | {delta:+.6f} |"
            )
    for g in ("ctgan", "tvae"):
        hashes = {r["label"]: r[g]["csv_md5"] for r in runs}
        lines.append(f"\n{g}: all runs identical to each other: {len(set(hashes.values())) == 1}")
    for r in runs:
        winner = max(("ctgan", "tvae"), key=lambda g: r[g]["selection_score"])
        margin = abs(r["ctgan"]["selection_score"] - r["tvae"]["selection_score"])
        lines.append(f"{r['label']}: higher score = {winner} (margin {margin:.6f})")
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
