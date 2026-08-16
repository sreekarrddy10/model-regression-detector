"""Statistics for the gate.

Every function here decides whether a merge is blocked, so each one is tested
against a hand-verifiable case as well as its degenerate inputs. Where a
statistic is undefined, these functions fail toward "no evidence of agreement"
rather than toward a passing number.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

from scipy.stats import ConstantInputWarning, binomtest, spearmanr


def mcnemar_exact(regressed: int, improved: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail outcomes.

    Only discordant pairs carry information: cases that passed on both runs, or
    failed on both, tell us nothing about whether the change mattered. Under the
    null hypothesis each discordant pair is equally likely to fall either way, so
    this is an exact two-sided binomial test on the discordant count.

    Answers the question the flat-threshold approach cannot: "2 of 80 flipped -
    is that signal or noise?"
    """
    discordant = regressed + improved
    if discordant == 0:
        return 1.0
    return float(binomtest(regressed, discordant, 0.5).pvalue)


def quadratic_weighted_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    """Agreement between two ordinal raters, corrected for chance.

    Quadratically weighted, so a 5-vs-4 disagreement counts far less than 5-vs-1 -
    the right choice for a 1-5 quality scale where near-misses are not equivalent
    to opposite verdicts.

    Returns 0.0 when either rater gave a constant score. Kappa is genuinely
    undefined there (no variance to explain), and 0.0 fails the calibration floor,
    which is the safe direction: an untested judge must not pass as calibrated.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"rater length mismatch: {len(rater_a)} vs {len(rater_b)}")
    if not rater_a:
        return 0.0

    ratings = list(range(min_rating, max_rating + 1))
    index = {r: i for i, r in enumerate(ratings)}
    size = len(ratings)
    span = (size - 1) ** 2
    total = len(rater_a)

    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(rater_a, rater_b, strict=True):
        observed[index[a]][index[b]] += 1 / total

    hist_a = [0.0] * size
    hist_b = [0.0] * size
    for a, b in zip(rater_a, rater_b, strict=True):
        hist_a[index[a]] += 1 / total
        hist_b[index[b]] += 1 / total

    numerator = 0.0
    denominator = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / span
            numerator += weight * observed[i][j]
            denominator += weight * hist_a[i] * hist_b[j]

    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def spearman(rater_a: Sequence[float], rater_b: Sequence[float]) -> float:
    """Rank correlation. Returns 0.0 when undefined (constant input)."""
    if len(rater_a) < 2:
        return 0.0
    with warnings.catch_warnings():
        # A constant rater is an expected, handled case here - the flat-holdout
        # warning in dataset.report is what surfaces it to the author.
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(list(rater_a), list(rater_b)).statistic
    return 0.0 if math.isnan(float(rho)) else float(rho)


def majority_pass(flags: Sequence[bool]) -> bool:
    """True when a case passes in more than half its repeats.

    This is what separates a regression from a flake: with N=3, a case must fail
    at least twice to count as regressed.
    """
    if not flags:
        return False
    return sum(flags) * 2 > len(flags)


def is_flaky(flags: Sequence[bool]) -> bool:
    """True when repeats of the same case disagree with each other."""
    return len(set(flags)) > 1


def pass_at_k(flags: Sequence[bool]) -> bool:
    """At least one success in k attempts."""
    return any(flags)


def pass_hat_k(flags: Sequence[bool]) -> bool:
    """All k attempts succeed. The bar for release-critical paths."""
    return bool(flags) and all(flags)


def ewma(values: Sequence[float], *, alpha: float = 0.3) -> float:
    """Exponentially weighted moving average, most recent value last.

    Catches gradual degradation that no single run-to-run diff would flag.
    """
    if not values:
        return 0.0
    current = values[0]
    for value in values[1:]:
        current = alpha * value + (1 - alpha) * current
    return current
