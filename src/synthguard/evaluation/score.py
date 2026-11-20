"""Selection: privacy gate, combined score, and choosing the generator to register."""

from __future__ import annotations


def selection_score(metrics: dict, weights: dict) -> float:
    """score = w_quality * quality_score + w_utility * utility_ratio."""
    return (
        weights["quality"] * metrics["quality_score"]
        + weights["utility"] * metrics["utility_ratio"]
    )


def passes_gate(metrics: dict, gate: dict) -> bool:
    """Privacy gate: MIA AUC <= mia_auc_max AND exact_match_rate <= exact_match_max."""
    return (
        metrics["mia_auc"] <= gate["mia_auc_max"]
        and metrics["exact_match_rate"] <= gate["exact_match_max"]
    )


def rank(candidates: dict[str, dict]) -> list[str]:
    """Best first: highest selection_score, then lower mia_auc, then lower fit_seconds."""
    return sorted(
        candidates,
        key=lambda g: (
            -candidates[g]["selection_score"],
            candidates[g]["mia_auc"],
            candidates[g].get("fit_seconds", float("inf")),
        ),
    )


def select(candidates: dict[str, dict]) -> dict:
    """Pick the top-ranked candidate that passed the gate; if none passed, the top overall.

    Each candidate dict must already hold `selection_score`, `privacy_gate` ("passed"/"failed"),
    `mia_auc` and optionally `fit_seconds`.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    passed = {g: m for g, m in candidates.items() if m["privacy_gate"] == "passed"}
    pool = passed or candidates
    winner = rank(pool)[0]
    return {
        "generator": winner,
        "privacy_gate": "passed" if passed else "failed",
        "selection_score": candidates[winner]["selection_score"],
        "ranking": rank(candidates),
        "passed_gate": sorted(passed),
    }


def score_candidate(metrics: dict, selection_params: dict) -> dict:
    """Add `selection_score` and `privacy_gate` to a candidate's metric dict (returns a copy)."""
    out = dict(metrics)
    out["selection_score"] = selection_score(metrics, selection_params["weights"])
    out["privacy_gate"] = "passed" if passes_gate(metrics, selection_params["gate"]) else "failed"
    return out
