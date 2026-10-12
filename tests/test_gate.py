"""Promotion gate decisions (criteria version 2), on hand-made comparisons."""

from retention.evaluation.bootstrap import Comparison
from retention.evaluation.gate import decide


def vs_champion(difference: float, ci_low: float) -> Comparison:
    return Comparison(0.6 + difference, 0.6, difference, ci_low, difference + 0.02, 0.5)


def test_first_model_is_promoted_if_it_meets_the_criteria() -> None:
    decision = decide(passed_criteria=True, vs_champion=None)
    assert decision.promote and "no current champion" in decision.reasons


def test_a_tie_with_the_champion_is_promoted() -> None:
    """Retrained on fresh data and not worse: replace the stale champion."""
    assert decide(True, vs_champion(0.0, -0.004)).promote


def test_a_challenger_clearly_worse_than_the_champion_is_kept_out() -> None:
    decision = decide(True, vs_champion(-0.03, -0.05))
    assert not decision.promote
    assert any(reason.startswith("worse than the champion") for reason in decision.reasons)


def test_failing_the_shipping_criteria_blocks_promotion_even_if_better_than_champion() -> None:
    assert not decide(False, vs_champion(0.05, 0.02)).promote
