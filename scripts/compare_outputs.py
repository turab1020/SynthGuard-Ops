"""After a fresh-clone run: compare produced metrics with the committed ones (git HEAD).

usage: python scripts/compare_outputs.py [tolerance]   (default 0.02)
Exit 1 if any candidate's selection_score differs by more than the tolerance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02


def committed(path: str) -> dict:
    return json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"]))


def main() -> int:
    old, new = committed("metrics/evaluation.json"), json.load(open("metrics/evaluation.json"))
    reg_old, reg_new = (
        committed("metrics/registered.json"),
        json.load(open("metrics/registered.json")),
    )
    lines = ["| generator | committed score | fresh score | Δ | gate committed → fresh |",
             "|---|---|---|---|---|"]  # fmt: skip
    worst = 0.0
    for g, c in old["candidates"].items():
        f = new["candidates"][g]
        d = f["selection_score"] - c["selection_score"]
        worst = max(worst, abs(d))
        cs, fs = c["selection_score"], f["selection_score"]
        gates = f"{c['privacy_gate']} → {f['privacy_gate']}"
        lines.append(f"| {g} | {cs:.6f} | {fs:.6f} | {d:+.6f} | {gates} |")
    lines += [
        "",
        f"Selected: committed **{old['selection']['generator']}**, "
        f"fresh **{new['selection']['generator']}**",
        f"Registered: committed {reg_old['generator']} (v{reg_old['version']}), "
        f"fresh {reg_new['generator']} (v{reg_new['version']})",
        f"Max |Δ score| = {worst:.6f} (tolerance {TOL}) → {'PASS' if worst <= TOL else 'FAIL'}",
    ]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text + "\n")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    sys.exit(main())
