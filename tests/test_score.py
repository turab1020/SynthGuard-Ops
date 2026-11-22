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


def test_exact_ties_go_to_the_faster_candidate():
    c = {
        "slow": cand(0.8, 0.8, mia=0.45, fit=100.0),
        "fast": cand(0.8, 0.8, mia=0.55, fit=1.0),
    }
    assert select(c)["generator"] == "fast"
    assert select(c)["tied"] == ["fast", "slow"]


def test_rank_orders_by_score_then_mia_then_fit():
    c = {
        "slow": cand(0.8, 0.8, mia=0.50, fit=100.0),
        "fast": cand(0.8, 0.8, mia=0.50, fit=1.0),
        "riskier": cand(0.8, 0.8, mia=0.55, fit=0.1),
    }
    assert rank(c) == ["fast", "slow", "riskier"]


def test_within_tolerance_the_faster_candidate_wins():
    c = {"ctgan": cand(0.86, 0.952, fit=155.0), "tvae": cand(0.84, 0.97, fit=37.0)}
    assert select(c, tie_tolerance=0.0)["generator"] == "ctgan"  # 0.906 vs 0.905: no tie
    s = select(c, tie_tolerance=0.02)
    assert s["generator"] == "tvae" and s["tied"] == ["ctgan", "tvae"]
    assert s["tie_tolerance"] == 0.02


def test_outside_tolerance_the_higher_score_wins():
    c = {"best": cand(0.90, 0.90, fit=500.0), "fast": cand(0.85, 0.85, fit=1.0)}
    s = select(c, tie_tolerance=0.02)
    assert s["generator"] == "best" and s["tied"] == ["best"]


def test_tolerance_boundary_is_inclusive():
    c = {"best": cand(0.90, 0.90, fit=500.0), "fast": cand(0.88, 0.88, fit=1.0)}
    assert select(c, tie_tolerance=0.02)["generator"] == "fast"  # exactly 0.02 below
    assert select(c, tie_tolerance=0.0199)["generator"] == "best"


def test_tie_only_among_candidates_that_pass_the_gate():
    c = {"good": cand(0.90, 0.90, fit=500.0), "leaky_fast": cand(0.90, 0.90, mia=0.9, fit=1.0)}
    s = select(c, tie_tolerance=0.02)
    assert s["generator"] == "good" and s["tied"] == ["good"]


@pytest.mark.parametrize(
    "scores",
    [
        {"ctgan": 0.905879, "tvae": 0.904732},  # macOS arm64 run
        {"ctgan": 0.897160, "tvae": 0.905247},  # Linux x86_64 / Docker run
    ],
)
def test_measured_near_tie_selects_the_same_generator_on_both_platforms(scores):
    c = {
        g: {"selection_score": s, "privacy_gate": "passed", "mia_auc": 0.49, "fit_seconds": f}
        for (g, s), f in zip(scores.items(), [155.0, 37.0], strict=True)
    }
    assert select(c, tie_tolerance=0.02)["generator"] == "tvae"


def test_score_candidate_adds_fields_without_mutating():
    m = {"quality_score": 0.8, "utility_ratio": 0.9, "mia_auc": 0.5, "exact_match_rate": 0.0}
    out = score_candidate(m, SELECTION)
    assert out["privacy_gate"] == "passed" and out["selection_score"] == pytest.approx(0.85)
    assert "selection_score" not in m


def test_select_needs_candidates():
    with pytest.raises(ValueError):
        select({})
