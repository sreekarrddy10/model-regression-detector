"""E2E fixtures.

Reports are generated from the same fixtures the unit tests use, so a browser
test and a string test can never be asserting about different pages.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from mrd.compare import GateReport, Verdict, compare, evaluate
from mrd.graders.calibration import Calibration
from mrd.report import html, model

from ..engine_fixtures import make_case, make_outcome

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "e2e"
GENERATED = datetime(2026, 8, 16, 12, 30)

CASES = (make_case(0, critical=True), *(make_case(i) for i in range(1, 10)))
ALL_PASS = {f"tc_{i:04d}": (True, True, True) for i in range(10)}


def build_report(
    candidate_flags: dict[str, tuple[bool, ...]],
    *,
    baseline_flags: dict[str, tuple[bool, ...]] | None = None,
    calibration: Calibration | None = None,
    trend: tuple[float, ...] = (),
    **outcome_kwargs: object,
) -> model.ReportData:
    candidate = make_outcome(candidate_flags, run_id="run-2", **outcome_kwargs)  # type: ignore[arg-type]
    baseline = (
        None
        if baseline_flags is None
        else make_outcome(baseline_flags, **outcome_kwargs)  # type: ignore[arg-type]
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
    )


def write_report(data: model.ReportData, path: Path) -> Path:
    return html.write(data, path)


@pytest.fixture(scope="session", autouse=True)
def artifacts_dir() -> Iterator[Path]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    yield ARTIFACTS


@pytest.fixture(scope="session")
def blocked_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A BLOCK verdict: two critical regressions, a flake, drift, cost and latency."""
    broken = {
        **ALL_PASS,
        "tc_0000": (False, False, False),
        "tc_0002": (False, False, False),
        "tc_0005": (True, False, True),
    }
    data = build_report(
        broken,
        baseline_flags=ALL_PASS,
        calibration=Calibration(
            kappa=0.78, spearman=0.81, sample_count=20, scored_count=20, floor=0.60
        ),
        trend=(0.98, 0.96, 0.94, 0.92, 0.90, 0.88),
    )
    return write_report(data, tmp_path_factory.mktemp("blocked") / "report.html")


@pytest.fixture(scope="session")
def passing_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = build_report(ALL_PASS, baseline_flags=ALL_PASS)
    return write_report(data, tmp_path_factory.mktemp("passing") / "report.html")


@pytest.fixture(scope="session")
def first_run_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = build_report(ALL_PASS)
    return write_report(data, tmp_path_factory.mktemp("first") / "report.html")


@pytest.fixture
def console_errors(page: object) -> Iterator[list[str]]:
    """Collect console errors and page exceptions for the duration of a test."""
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)  # type: ignore[attr-defined]
    page.on("pageerror", lambda e: errors.append(str(e)))  # type: ignore[attr-defined]
    yield errors
