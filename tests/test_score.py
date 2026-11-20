import pytest

from synthguard.evaluation.score import passes_gate, rank, score_candidate, select, selection_score

SELECTION = {
    "weights": {"quality": 0.5, "utility": 0.5},
    "gate": {"mia_auc_max": 0.60, "exact_match_max": 0.01},
}


def cand(quality, utility, mia=0.5, exact=0.0, fit=1.0):
    m = {"quality_score": quality, "utility_ratio": utility, "mia_auc": mia,
         "exact_match_rate": exact, "fit_seconds": fit}  # fmt: skip
    return score_candidate(m, SELECTION)


def test_selection_score_known_answer():
    assert selection_score({"quality_score": 0.8, "utility_ratio": 0.9}, SELECTION["weights"]) == (
        pytest.approx(0.85)
    )
    weights = {"quality": 0.25, "utility": 0.75}
    assert selection_score({"quality_score": 0.8, "utility_ratio": 0.4}, weights) == (
        pytest.approx(0.5)
    )


@pytest.mark.parametrize(
    "mia,exact,expected",
    [
        (0.60, 0.01, True),  # boundaries are inclusive (<=)
        (0.6001, 0.0, False),
        (0.5, 0.0101, False),
        (0.9, 0.5, False),
        (0.5, 0.0, True),
    ],
)
def test_gate(mia, exact, expected):
    assert passes_gate({"mia_auc": mia, "exact_match_rate": exact}, SELECTION["gate"]) is expected


def test_highest_score_that_passes_the_gate_wins():
    c = {
        "leaky": cand(0.99, 0.99, mia=0.9),  # best score but fails the gate
        "good": cand(0.85, 0.95),
        "ok": cand(0.80, 0.90),
    }
    s = select(c)
    assert s["generator"] == "good"
    assert s["privacy_gate"] == "passed"
    assert s["passed_gate"] == ["good", "ok"]
    assert s["ranking"] == ["leaky", "good", "ok"]
    assert s["selection_score"] == pytest.approx(0.9)


def test_nothing_passes_best_overall_tagged_failed():
    c = {"a": cand(0.8, 0.8, mia=0.7), "b": cand(0.9, 0.9, exact=0.5)}
    s = select(c)
    assert s["generator"] == "b"
    assert s["privacy_gate"] == "failed"
    assert s["passed_gate"] == []


def test_ties_break_on_mia_then_fit_time():
    c = {
        "slow": cand(0.8, 0.8, mia=0.50, fit=100.0),
        "fast": cand(0.8, 0.8, mia=0.50, fit=1.0),
        "riskier": cand(0.8, 0.8, mia=0.55, fit=0.1),
    }
    assert rank(c) == ["fast", "slow", "riskier"]
    assert select(c)["generator"] == "fast"


def test_score_candidate_adds_fields_without_mutating():
    m = {"quality_score": 0.8, "utility_ratio": 0.9, "mia_auc": 0.5, "exact_match_rate": 0.0}
    out = score_candidate(m, SELECTION)
    assert out["privacy_gate"] == "passed" and out["selection_score"] == pytest.approx(0.85)
    assert "selection_score" not in m


def test_select_needs_candidates():
    with pytest.raises(ValueError):
        select({})
