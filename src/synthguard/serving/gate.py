"""Evaluation gate: a candidate classifier may replace the production one only if it is not worse.

Both are scored on the same fixed real hold-out. The candidate fails if its ROC-AUC is more than
`max_drop` below the production benchmark, or if the generator behind it failed the privacy gate.
"""

from __future__ import annotations


def check(candidate: dict, benchmark: dict, max_drop: float, privacy_gate: str) -> dict:
    """Return {passed, reasons, roc_auc_delta}. Boundary: a drop of exactly max_drop passes."""
    delta = candidate["roc_auc"] - benchmark["roc_auc"]
    reasons = []
    if delta < -max_drop - 1e-12:
        reasons.append(
            f"ROC-AUC {candidate['roc_auc']:.4f} is {-delta:.4f} below the benchmark "
            f"{benchmark['roc_auc']:.4f} (allowed drop {max_drop})"
        )
    if privacy_gate != "passed":
        reasons.append(f"privacy gate of the source generator is '{privacy_gate}'")
    return {"passed": not reasons, "reasons": reasons, "roc_auc_delta": delta}


def summary_markdown(candidate: dict, benchmark: dict, result: dict, max_drop: float) -> str:
    status = "PASSED" if result["passed"] else "BLOCKED"
    lines = [
        f"## Evaluation gate: {status}",
        "",
        "Fixed real hold-out, positive class `>50K`.",
        "",
        "| Metric | Production benchmark | Candidate | Δ |",
        "|---|---|---|---|",
    ]
    for k in ("roc_auc", "f1", "accuracy", "precision", "recall"):
        if k in candidate and k in benchmark:
            d = candidate[k] - benchmark[k]
            lines.append(f"| {k} | {benchmark[k]:.4f} | {candidate[k]:.4f} | {d:+.4f} |")
    lines += ["", f"Rule: candidate ROC-AUC ≥ benchmark − {max_drop}, and privacy gate passed."]
    lines += [f"- {r}" for r in result["reasons"]]
    return "\n".join(lines) + "\n"
