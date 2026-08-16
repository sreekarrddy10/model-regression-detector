"""Statistics tests.

These functions decide whether merges are blocked, so each is checked against a
hand-computable value rather than only against itself.
"""

from __future__ import annotations

import pytest

from mrd.stats import (
    ewma,
    is_flaky,
    majority_pass,
    mcnemar_exact,
    pass_at_k,
    pass_hat_k,
    quadratic_weighted_kappa,
    spearman,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #


def test_no_discordant_pairs_is_no_evidence() -> None:
    assert mcnemar_exact(0, 0) == 1.0


def test_symmetric_discordance_is_no_evidence() -> None:
    """Equal regressions and improvements is exactly the null hypothesis."""
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_two_of_eighty_flipped_is_not_significant() -> None:
    """The question the guide's flat 3% threshold cannot answer."""
    # 2 regressed, 0 improved: two-sided exact binomial = 2 * 0.5^2 = 0.5
    assert mcnemar_exact(2, 0) == pytest.approx(0.5)


def test_ten_versus_one_is_significant() -> None:
    # Two-sided exact binomial on 11 discordant pairs, 10 in one direction.
    assert mcnemar_exact(10, 1) == pytest.approx(2 * (11 + 1) / 2**11)
    assert mcnemar_exact(10, 1) < 0.05


def test_direction_does_not_change_the_p_value() -> None:
    """A two-sided test is symmetric; direction is judged separately."""
    assert mcnemar_exact(9, 2) == pytest.approx(mcnemar_exact(2, 9))


# --------------------------------------------------------------------------- #
# Weighted kappa
# --------------------------------------------------------------------------- #


def test_perfect_agreement_is_one() -> None:
    scores = [1, 2, 3, 4, 5, 3, 2]
    assert quadratic_weighted_kappa(scores, scores) == pytest.approx(1.0)


def test_inverted_agreement_is_negative() -> None:
    assert quadratic_weighted_kappa([1, 2, 4, 5], [5, 4, 2, 1]) < 0


def test_near_misses_cost_less_than_opposite_verdicts() -> None:
    """Quadratic weighting is why a 5-vs-4 is not a 5-vs-1."""
    truth = [1, 2, 3, 4, 5]
    near = quadratic_weighted_kappa(truth, [2, 3, 4, 5, 4])
    far = quadratic_weighted_kappa(truth, [5, 4, 3, 2, 1])
    assert near > far


def test_constant_rater_yields_zero_not_a_pass() -> None:
    """Kappa is undefined with no variance; failing the floor is the safe direction."""
    assert quadratic_weighted_kappa([5, 5, 5, 5], [5, 5, 5, 5]) == 0.0
    assert quadratic_weighted_kappa([1, 2, 3, 4], [3, 3, 3, 3]) == 0.0


def test_empty_input_is_zero() -> None:
    assert quadratic_weighted_kappa([], []) == 0.0


def test_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        quadratic_weighted_kappa([1, 2], [1])


# --------------------------------------------------------------------------- #
# Spearman
# --------------------------------------------------------------------------- #


def test_monotonic_relationship_is_one() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_inverse_relationship_is_minus_one() -> None:
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_constant_input_is_zero_not_nan() -> None:
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_single_sample_is_zero() -> None:
    assert spearman([1], [1]) == 0.0


# --------------------------------------------------------------------------- #
# Repeat aggregation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "flags,expected",
    [
        ((True, True, True), True),
        ((True, True, False), True),
        ((True, False, False), False),
        ((False, False, False), False),
        ((), False),
    ],
)
def test_majority_pass(flags: tuple[bool, ...], expected: bool) -> None:
    """A case must fail at least twice of three to count as regressed."""
    assert majority_pass(flags) is expected


def test_flaky_detection() -> None:
    assert is_flaky((True, False, True))
    assert not is_flaky((True, True, True))
    assert not is_flaky((False, False, False))


def test_pass_at_k_and_pass_hat_k() -> None:
    assert pass_at_k((False, False, True))
    assert not pass_at_k((False, False, False))
    assert pass_hat_k((True, True, True))
    assert not pass_hat_k((True, True, False))
    assert not pass_hat_k(())


# --------------------------------------------------------------------------- #
# EWMA
# --------------------------------------------------------------------------- #


def test_ewma_of_constant_series_is_that_constant() -> None:
    assert ewma([0.9] * 7) == pytest.approx(0.9)


def test_ewma_weights_recent_values_more() -> None:
    declining = ewma([1.0, 1.0, 1.0, 0.5])
    assert declining < 0.9
    assert declining == pytest.approx(0.3 * 0.5 + 0.7 * 1.0)


def test_ewma_of_empty_series_is_zero() -> None:
    assert ewma([]) == 0.0
