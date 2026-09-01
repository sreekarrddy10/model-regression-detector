"""Gate behavior.

Each test names the operational question it answers, because these thresholds
are what a teammate will argue with when CI blocks their PR.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mrd.compare import (
    IncomparableRuns,
    Thresholds,
    Verdict,
    compare,
    evaluate,
    evaluate_first_run,
    measure,
    percentile,
)

from .engine_fixtures import make_case, make_outcome

pytestmark = pytest.mark.unit

CASES = tuple(make_case(i) for i in range(10))
CRITICAL_CASES = (make_case(0, critical=True), *(make_case(i) for i in range(1, 10)))

ALL_PASS = {f"tc_{i:04d}": (True, True, True) for i in range(10)}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def test_accuracy_uses_majority_of_repeats() -> None:
    outcome = make_outcome({"tc_0000": (True, True, False), "tc_0001": (True, False, False)})
    metrics = measure(outcome)

    assert metrics.case_count == 2
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.flaky_cases == ("tc_0000", "tc_0001")


def test_pass_at_k_and_hat_k_reported() -> None:
    outcome = make_outcome({"tc_0000": (False, True, True), "tc_0001": (True, True, True)})
    metrics = measure(outcome)

    assert metrics.pass_at_k_rate == pytest.approx(1.0)
    assert metrics.pass_hat_k_rate == pytest.approx(0.5)
    assert metrics.pass_at_1 == pytest.approx(0.5)


def test_empty_run_measures_cleanly() -> None:
    assert measure(make_outcome({})).case_count == 0


def test_percentile_nearest_rank() -> None:
    assert percentile([10, 20, 30, 40, 50], 0.95) == 50.0
    assert percentile([], 0.95) == 0.0


# --------------------------------------------------------------------------- #
# Comparability
# --------------------------------------------------------------------------- #


def test_different_ground_truth_refuses_to_diff() -> None:
    """A quiet label edit must not be reportable as a model improvement."""
    baseline = make_outcome(ALL_PASS, dataset_hash="hash-a")
    candidate = make_outcome(ALL_PASS, run_id="run-2", dataset_hash="hash-b")

    with pytest.raises(IncomparableRuns, match="different ground truth"):
        compare(baseline, candidate, CASES)


# --------------------------------------------------------------------------- #
# Signal versus noise
# --------------------------------------------------------------------------- #


def test_identical_runs_pass() -> None:
    result = evaluate(
        compare(make_outcome(ALL_PASS), make_outcome(ALL_PASS, run_id="run-2"), CASES)
    )
    assert result.verdict is Verdict.PASS
    assert result.exit_code == 0


def test_single_repeat_flip_is_flaky_not_a_regression() -> None:
    """One bad sample of three is noise; the case still passes by majority."""
    candidate = make_outcome({**ALL_PASS, "tc_0003": (True, False, True)}, run_id="run-2")
    comparison = compare(make_outcome(ALL_PASS), candidate, CASES)

    assert comparison.regressed == ()
    assert comparison.newly_flaky == ("tc_0003",)
    assert evaluate(comparison).verdict is Verdict.WARN


def test_two_of_three_repeats_failing_is_a_regression() -> None:
    candidate = make_outcome({**ALL_PASS, "tc_0003": (False, False, True)}, run_id="run-2")
    comparison = compare(make_outcome(ALL_PASS), candidate, CASES)

    assert comparison.regressed == ("tc_0003",)


def test_small_regression_warns_but_does_not_block() -> None:
    """One case of twenty: real, but under the effect floor and not significant."""
    cases = tuple(make_case(i) for i in range(20))
    baseline_flags = {f"tc_{i:04d}": (True, True, True) for i in range(20)}
    candidate_flags = {**baseline_flags, "tc_0003": (False, False, False)}

    comparison = compare(
        make_outcome(baseline_flags), make_outcome(candidate_flags, run_id="run-2"), cases
    )
    report = evaluate(comparison)

    assert comparison.accuracy_delta == pytest.approx(-0.05)
    assert comparison.mcnemar_p == pytest.approx(1.0)
    assert report.verdict is Verdict.WARN
    assert any("accuracy fell" in w for w in report.warnings)


def test_large_regression_blocks_on_effect_size() -> None:
    broken = {**ALL_PASS}
    for i in range(3):
        broken[f"tc_{i:04d}"] = (False, False, False)
    report = evaluate(compare(make_outcome(ALL_PASS), make_outcome(broken, run_id="run-2"), CASES))

    assert report.verdict is Verdict.BLOCK
    assert report.exit_code == 1
    assert any("past the 8% block threshold" in b for b in report.blocking)


def test_significant_regression_blocks_even_under_the_effect_floor() -> None:
    """Six regressions and no improvements is significant regardless of headline size."""
    cases = tuple(make_case(i) for i in range(100))
    baseline_flags = {f"tc_{i:04d}": (True, True, True) for i in range(100)}
    candidate_flags = dict(baseline_flags)
    for i in range(6):
        candidate_flags[f"tc_{i:04d}"] = (False, False, False)

    comparison = compare(
        make_outcome(baseline_flags), make_outcome(candidate_flags, run_id="run-2"), cases
    )
    report = evaluate(comparison)

    assert comparison.accuracy_delta > -0.08, "effect size alone would not block"
    assert comparison.mcnemar_p < 0.05
    assert report.verdict is Verdict.BLOCK
    assert any("McNemar" in b for b in report.blocking)


def test_equal_regressions_and_improvements_do_not_block() -> None:
    """Churn is not degradation."""
    candidate = dict(ALL_PASS)
    baseline = dict(ALL_PASS)
    baseline["tc_0001"] = (False, False, False)
    candidate["tc_0002"] = (False, False, False)

    comparison = compare(make_outcome(baseline), make_outcome(candidate, run_id="run-2"), CASES)
    assert comparison.mcnemar_p == pytest.approx(1.0)
    assert not comparison.net_negative


# --------------------------------------------------------------------------- #
# Blocking signals
# --------------------------------------------------------------------------- #


def test_one_critical_regression_blocks_alone() -> None:
    candidate = make_outcome({**ALL_PASS, "tc_0000": (False, False, False)}, run_id="run-2")
    report = evaluate(compare(make_outcome(ALL_PASS), candidate, CRITICAL_CASES))

    assert report.verdict is Verdict.BLOCK
    assert any("critical case(s) regressed: tc_0000" in b for b in report.blocking)


def test_schema_failure_blocks() -> None:
    candidate = make_outcome(
        {**ALL_PASS, "tc_0004": (False, False, False)}, run_id="run-2", schema_valid=False
    )
    report = evaluate(compare(make_outcome(ALL_PASS), candidate, CASES))

    assert report.verdict is Verdict.BLOCK
    assert any("response schema" in b for b in report.blocking)


def test_uncalibrated_judge_blocks_the_run() -> None:
    """An uninterpretable quality number is worse than no quality number."""
    report = evaluate(
        compare(make_outcome(ALL_PASS), make_outcome(ALL_PASS, run_id="run-2"), CASES),
        judge_calibrated=False,
    )
    assert report.verdict is Verdict.BLOCK
    assert any("failed calibration" in b for b in report.blocking)


# --------------------------------------------------------------------------- #
# Judge: advisory unless confirmed
# --------------------------------------------------------------------------- #


def test_unconfirmed_judge_drop_only_warns() -> None:
    """One model's opinion is not evidence."""
    comparison = compare(
        make_outcome(ALL_PASS, judge_score=5),
        make_outcome(ALL_PASS, run_id="run-2", judge_score=3),
        CASES,
    )
    report = evaluate(comparison, judge_confirmations=1)

    assert report.verdict is Verdict.WARN
    assert any("advisory only" in w for w in report.warnings)


def test_confirmed_judge_drop_blocks() -> None:
    comparison = compare(
        make_outcome(ALL_PASS, judge_score=5),
        make_outcome(ALL_PASS, run_id="run-2", judge_score=3),
        CASES,
    )
    report = evaluate(comparison, judge_confirmations=3)

    assert report.verdict is Verdict.BLOCK
    assert any("confirmed across 3 independent seeds" in b for b in report.blocking)


def test_judge_drop_inside_the_warn_band_never_blocks() -> None:
    """A 0.4-point slip is worth telling someone about, not worth blocking on."""
    comparison = compare(
        make_outcome(ALL_PASS, judge_score=5),
        make_outcome(ALL_PASS, run_id="run-2", judge_score=[5, 5, 4, 4, 5]),
        CASES,
    )
    report = evaluate(comparison, judge_confirmations=3)

    assert comparison.judge_delta == pytest.approx(-0.4)
    assert report.verdict is Verdict.WARN
    assert any("summary quality fell 0.40 points" in w for w in report.warnings)


def test_missing_judge_scores_are_not_an_error() -> None:
    comparison = compare(make_outcome(ALL_PASS), make_outcome(ALL_PASS, run_id="run-2"), CASES)
    assert comparison.judge_delta is None
    assert evaluate(comparison).verdict is Verdict.PASS


# --------------------------------------------------------------------------- #
# Cost, latency, drift
# --------------------------------------------------------------------------- #


def test_latency_regression_warns_never_blocks() -> None:
    comparison = compare(
        make_outcome(ALL_PASS, latency_ms=100),
        make_outcome(ALL_PASS, run_id="run-2", latency_ms=200),
        CASES,
    )
    report = evaluate(comparison)

    assert report.verdict is Verdict.WARN
    assert any("p95 latency rose" in w for w in report.warnings)


def test_cost_regression_warns() -> None:
    comparison = compare(
        make_outcome(ALL_PASS, cost_usd=0.001),
        make_outcome(ALL_PASS, run_id="run-2", cost_usd=0.002),
        CASES,
    )
    assert any("cost per case rose" in w for w in evaluate(comparison).warnings)


def test_unpriced_run_skips_the_cost_dimension() -> None:
    comparison = compare(
        make_outcome(ALL_PASS, cost_usd=None),
        make_outcome(ALL_PASS, run_id="run-2", cost_usd=None),
        CASES,
    )
    assert comparison.cost_ratio is None
    assert not any("cost" in w for w in evaluate(comparison).warnings)


def test_slow_drift_warns_though_no_single_run_regressed() -> None:
    """The failure mode per-run diffs are structurally blind to.

    Baseline and candidate fail the same two cases, so nothing regressed and no
    single diff would ever fire - but the trend has been sagging for six runs.
    """
    sagging = {**ALL_PASS, "tc_0008": (False,) * 3, "tc_0009": (False,) * 3}
    comparison = compare(make_outcome(sagging), make_outcome(sagging, run_id="run-2"), CASES)
    report = evaluate(comparison, recent_accuracy=[0.95, 0.93, 0.91, 0.89, 0.87, 0.85])

    assert comparison.regressed == ()
    assert comparison.accuracy_delta == 0.0
    assert report.verdict is Verdict.WARN
    assert any("gradual drift" in w for w in report.warnings)


def test_healthy_trend_does_not_warn() -> None:
    comparison = compare(make_outcome(ALL_PASS), make_outcome(ALL_PASS, run_id="run-2"), CASES)
    report = evaluate(comparison, recent_accuracy=[0.97, 0.98, 0.99, 1.0])

    assert report.verdict is Verdict.PASS


def test_thresholds_are_configurable() -> None:
    candidate = make_outcome({**ALL_PASS, "tc_0003": (False, False, False)}, run_id="run-2")
    comparison = compare(make_outcome(ALL_PASS), candidate, CASES)

    strict = evaluate(comparison, thresholds=Thresholds(accuracy_block=-0.05))
    assert strict.verdict is Verdict.BLOCK

    lenient = evaluate(comparison, thresholds=Thresholds(accuracy_warn=-0.5, accuracy_block=-0.9))
    assert lenient.verdict is Verdict.PASS


# --------------------------------------------------------------------------- #
# First run: no baseline, but the checks that need none still apply
# --------------------------------------------------------------------------- #


def test_first_run_blocks_when_every_attempt_failed_at_the_provider() -> None:
    """The bug this exists for: a CI run whose calls all returned 401 was
    reported PASS and recorded as the baseline, so the next run compared
    against 0% accuracy and would have read as a large improvement."""
    outcome = make_outcome(ALL_PASS)
    broken = replace(
        outcome,
        results=tuple(replace(r, error="401 Unauthorized") for r in outcome.results),
    )
    report = evaluate_first_run(broken)
    assert report.verdict is Verdict.BLOCK
    assert "measured nothing" in " ".join(report.blocking)


def test_first_run_blocks_an_uncalibrated_judge() -> None:
    report = evaluate_first_run(make_outcome(ALL_PASS), judge_calibrated=False)
    assert report.verdict is Verdict.BLOCK
    assert "failed calibration" in " ".join(report.blocking)


def test_first_run_blocks_an_empty_run() -> None:
    empty = replace(make_outcome(ALL_PASS), results=())
    assert evaluate_first_run(empty).verdict is Verdict.BLOCK


def test_first_run_blocks_below_the_sanity_floor() -> None:
    """Not a quality bar - a prompt scoring under half on the golden set is a
    broken baseline whatever the cause."""
    mostly_failing = {f"tc_{i:04d}": (i < 3, i < 3, i < 3) for i in range(10)}
    assert evaluate_first_run(make_outcome(mostly_failing)).verdict is Verdict.BLOCK


def test_a_healthy_first_run_still_passes() -> None:
    report = evaluate_first_run(make_outcome(ALL_PASS))
    assert report.verdict is Verdict.PASS
    assert report.blocking == ()
