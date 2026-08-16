from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mrd.compare import GateReport, Verdict, compare, evaluate
from mrd.graders.calibration import Calibration
from mrd.report import html, model

from .engine_fixtures import make_case, make_outcome

pytestmark = pytest.mark.unit

CASES = (make_case(0, critical=True), *(make_case(i) for i in range(1, 10)))
ALL_PASS = {f"tc_{i:04d}": (True, True, True) for i in range(10)}
GENERATED = datetime(2026, 8, 16, 12, 30)


def build(
    candidate_flags: dict[str, tuple[bool, ...]],
    *,
    baseline_flags: dict[str, tuple[bool, ...]] | None = None,
    calibration: Calibration | None = None,
    trend: tuple[float, ...] = (),
    report_url: str | None = None,
    **outcome_kwargs: object,
) -> model.ReportData:
    candidate = make_outcome(candidate_flags, run_id="run-2", **outcome_kwargs)  # type: ignore[arg-type]
    baseline = (
        None if baseline_flags is None else make_outcome(baseline_flags, **outcome_kwargs)  # type: ignore[arg-type]
    )
    comparison = None if baseline is None else compare(baseline, candidate, CASES)
    gate = (
        GateReport(verdict=Verdict.PASS)
        if comparison is None
        else evaluate(comparison, recent_accuracy=trend)
    )
    return model.build(
        candidate,
        CASES,
        gate=gate,
        generated_at=GENERATED,
        baseline=baseline,
        comparison=comparison,
        trend=trend,
        calibration=calibration,
        report_url=report_url,
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def test_regressions_improvements_and_flakes_are_all_surfaced() -> None:
    candidate = {
        **ALL_PASS,
        "tc_0003": (False, False, False),
        "tc_0005": (True, False, True),
    }
    baseline = {**ALL_PASS, "tc_0007": (False, False, False)}
    data = build(candidate, baseline_flags=baseline)

    assert [d.case_id for d in data.regressed] == ["tc_0003"]
    assert [d.case_id for d in data.improved] == ["tc_0007"]
    assert [d.case_id for d in data.flaky] == ["tc_0005"]


def test_critical_regressions_sort_first() -> None:
    """They are why the gate fired, so they lead the report."""
    candidate = {
        **ALL_PASS,
        "tc_0000": (False, False, False),
        "tc_0002": (False, False, False),
    }
    data = build(candidate, baseline_flags=ALL_PASS)

    assert [d.case_id for d in data.regressed] == ["tc_0000", "tc_0002"]
    assert data.regressed[0].critical


def test_diff_carries_both_runs_attempts() -> None:
    data = build({**ALL_PASS, "tc_0003": (False, False, False)}, baseline_flags=ALL_PASS)
    diff = data.regressed[0]

    assert len(diff.baseline) == 3
    assert len(diff.candidate) == 3
    assert diff.baseline_verdict == "pass"
    assert diff.candidate_verdict == "fail"
    assert diff.notes  # the "why this case exists" line


def test_first_run_has_no_baseline() -> None:
    data = build(ALL_PASS)

    assert not data.has_baseline
    assert "No baseline to compare against yet" in data.headline
    assert data.diffs == ()


def test_headline_reports_no_change() -> None:
    data = build(ALL_PASS, baseline_flags=ALL_PASS)
    assert "No cases changed" in data.headline


def test_headline_counts_both_directions() -> None:
    candidate = {**ALL_PASS, "tc_0003": (False,) * 3}
    baseline = {**ALL_PASS, "tc_0007": (False,) * 3}
    assert "1 regression and 1 improvement" in build(candidate, baseline_flags=baseline).headline


# --------------------------------------------------------------------------- #
# Sparkline
# --------------------------------------------------------------------------- #


def test_sparkline_needs_at_least_two_points() -> None:
    """A one-point trend is a dot pretending to be information."""
    assert html.sparkline([]) == ""
    assert html.sparkline([0.95]) == ""


def test_sparkline_is_self_contained_svg() -> None:
    svg = html.sparkline([0.95, 0.93, 0.90, 0.88])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "http" not in svg
    assert "polyline" in svg


def test_sparkline_colours_by_the_drift_floor() -> None:
    assert "#b42318" in html.sparkline([0.95, 0.85], floor=0.90)
    assert "#067647" in html.sparkline([0.85, 0.95], floor=0.90)


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #


def test_report_is_self_contained() -> None:
    """It is opened from a CI artifact on a laptop with no network."""
    page = html.render(build({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS))

    assert "<!doctype html>" in page.lower()
    assert "<script" not in page.lower()
    assert 'src="http' not in page
    assert 'href="http' not in page
    assert "cdn" not in page.lower()


def test_report_leads_with_the_verdict_and_its_reasons() -> None:
    broken = {**ALL_PASS}
    for i in range(3):
        broken[f"tc_{i:04d}"] = (False,) * 3
    data = build(broken, baseline_flags=ALL_PASS)
    page = html.render(data)

    assert data.gate.verdict is Verdict.BLOCK
    assert 'class="verdict BLOCK"' in page
    assert "Blocking." in page
    assert "critical case(s) regressed" in page


def test_report_shows_the_statistics_it_decided_on() -> None:
    """A reader must be able to audit the gate, not just obey it."""
    page = html.render(build({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS))

    assert "McNemar exact" in page
    assert "discordant pair" in page
    assert "majority of its 3 repeats" in page


def test_report_renders_both_sides_of_a_regression() -> None:
    page = html.render(build({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS))

    assert "tc_0003" in page
    assert "Baseline — pass" in page
    assert "This run — fail" in page
    assert "Why this case exists" in page


def test_report_marks_uncalibrated_judge_numbers_as_uninterpreted() -> None:
    page = html.render(build(ALL_PASS, baseline_flags=ALL_PASS, calibration=None))
    assert "has not been measured against human scores" in page


def test_report_shows_calibration_when_present() -> None:
    calibration = Calibration(
        kappa=0.81, spearman=0.79, sample_count=20, scored_count=20, floor=0.6
    )
    page = html.render(build(ALL_PASS, baseline_flags=ALL_PASS, calibration=calibration))

    assert "0.81" in page
    assert "agrees with human scores" in page


def test_report_escapes_case_content() -> None:
    """Case text is data; a golden case containing markup must not become markup."""
    hostile = make_case(0)
    data = model.build(
        make_outcome({"tc_0000": (True, True, True)}),
        (
            type(hostile)(
                **{
                    **hostile.model_dump(),
                    "input_email": "<script>alert('x')</script>",
                }
            ),
        ),
        gate=GateReport(verdict=Verdict.PASS),
        generated_at=GENERATED,
    )
    page = html.render(data)

    assert "<script>alert" not in page


def test_first_run_report_renders() -> None:
    page = html.render(build(ALL_PASS))
    assert "No baseline to compare against yet" in page
    assert "No regressions, improvements or newly flaky cases." in page


def test_trend_section_appears_only_with_enough_history() -> None:
    without = html.render(build(ALL_PASS, baseline_flags=ALL_PASS, trend=(0.95,)))
    assert "Accuracy trend" not in without

    with_trend = html.render(
        build(ALL_PASS, baseline_flags=ALL_PASS, trend=(0.95, 0.94, 0.93, 0.92))
    )
    assert "Accuracy trend" in with_trend
    assert "90% drift" in with_trend


def test_write_produces_a_readable_file(tmp_path: Path) -> None:
    path = html.write(build(ALL_PASS, baseline_flags=ALL_PASS), tmp_path / "out" / "report.html")

    assert path.exists()
    assert path.read_text(encoding="utf-8").lstrip().startswith("<!doctype html>")
