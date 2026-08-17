"""Browser tests for the diff report.

Every test here asserts something a substring check on the HTML cannot: that an
element is actually visible, that computed styles differ, that the SVG has real
geometry, that nothing overflows the viewport. The unit tests already prove the
right strings are present - these prove the page is legible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from .conftest import ARTIFACTS
from .pages.report_page import ReportPage

pytestmark = pytest.mark.e2e


# --------------------------------------------------------------------------- #
# Self-containment
# --------------------------------------------------------------------------- #


def test_report_loads_with_no_network_at_all(
    page: Page, blocked_report: Path, console_errors: list[str]
) -> None:
    """It is opened from a CI artifact on a laptop offline."""
    requests: list[str] = []
    page.on("request", lambda r: requests.append(r.url))
    # Abort only off-disk schemes; routing "**" would abort the file:// load itself.
    page.route("http://**", lambda route: route.abort())
    page.route("https://**", lambda route: route.abort())

    report = ReportPage(page).open(blocked_report)

    expect(report.verdict).to_be_visible()
    assert all(
        url.startswith("file://") for url in requests
    ), f"the report reached off-disk: {[u for u in requests if not u.startswith('file://')]}"
    assert console_errors == []


def test_no_scripts_execute(page: Page, blocked_report: Path) -> None:
    ReportPage(page).open(blocked_report)
    assert page.evaluate("document.scripts.length") == 0


# --------------------------------------------------------------------------- #
# The verdict is the first thing a reader sees
# --------------------------------------------------------------------------- #


def test_blocked_verdict_is_visible_and_styled(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)

    expect(report.verdict_tag).to_be_visible()
    expect(report.verdict_tag).to_have_text("BLOCK")
    assert "BLOCK" in report.verdict_class()

    # Red, not merely "some colour". A verdict that reads as neutral is a
    # verdict people skim past.
    colour = page.evaluate("getComputedStyle(document.querySelector('.verdict .tag')).color")
    red, green, blue = _rgb(colour)
    assert red > green and red > blue, f"BLOCK tag is not red: {colour}"


def test_passing_verdict_reads_green(page: Page, passing_report: Path) -> None:
    ReportPage(page).open(passing_report)
    colour = page.evaluate("getComputedStyle(document.querySelector('.verdict .tag')).color")
    red, green, blue = _rgb(colour)
    assert green > red, f"PASS tag is not green: {colour}"


def test_every_blocking_reason_is_rendered(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)

    expect(report.blocking_reasons).to_have_count(2)
    expect(report.blocking_reasons.first).to_contain_text("critical case(s) regressed")
    for i in range(report.blocking_reasons.count()):
        expect(report.blocking_reasons.nth(i)).to_be_visible()


def test_verdict_sits_above_the_fold(page: Page, blocked_report: Path) -> None:
    """The reader decides in three seconds; scrolling for the verdict loses them."""
    page.set_viewport_size({"width": 1280, "height": 800})
    report = ReportPage(page).open(blocked_report)

    box = report.verdict.bounding_box()
    assert box is not None
    assert box["y"] + box["height"] < 800, "verdict is below the fold at 1280x800"


# --------------------------------------------------------------------------- #
# Case detail
# --------------------------------------------------------------------------- #


def test_critical_regressions_render_first(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)

    expect(report.cases.first.locator("h3")).to_have_text("tc_0000")
    expect(report.cases.first.locator(".pill.critical")).to_be_visible()


def test_a_regressed_case_shows_both_runs(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)
    case = report.case("tc_0000")

    expect(case.locator(".col").nth(0)).to_contain_text("Baseline — pass")
    expect(case.locator(".col").nth(1)).to_contain_text("This run — fail")
    expect(case.locator(".attempt")).to_have_count(6)  # 3 repeats each side


def test_failed_attempts_are_visually_distinct(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)
    case = report.case("tc_0000")

    ok_border = case.locator(".attempt.ok").first.evaluate("el => getComputedStyle(el).borderColor")
    fail_border = case.locator(".attempt.fail").first.evaluate(
        "el => getComputedStyle(el).borderColor"
    )
    assert ok_border != fail_border, "pass and fail attempts are indistinguishable"


def test_flaky_case_is_surfaced_and_labelled(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)
    expect(report.case("tc_0005").locator(".pill.flaky")).to_be_visible()


def test_case_notes_are_rendered(page: Page, blocked_report: Path) -> None:
    """The 'why this case exists' line is the payoff for requiring notes."""
    report = ReportPage(page).open(blocked_report)
    expect(report.case("tc_0000")).to_contain_text("Why this case exists")


# --------------------------------------------------------------------------- #
# The trend chart
# --------------------------------------------------------------------------- #


def test_trend_svg_draws_real_geometry(page: Page, blocked_report: Path) -> None:
    """An SVG that renders at zero height is invisible but passes a string check."""
    report = ReportPage(page).open(blocked_report)

    expect(report.trend_svg).to_be_visible()
    box = report.trend_svg.bounding_box()
    assert box is not None
    assert box["width"] > 100 and box["height"] > 20, f"trend chart collapsed: {box}"

    points = page.locator("svg polyline").get_attribute("points") or ""
    assert len(points.split()) == 6, "one vertex per run in the trend"


def test_declining_trend_is_drawn_in_the_alarm_colour(page: Page, blocked_report: Path) -> None:
    ReportPage(page).open(blocked_report)
    stroke = page.locator("svg polyline").evaluate("el => el.getAttribute('stroke')")
    red, green, blue = _rgb_hex(stroke)
    assert red > green and red > blue, f"a trend below the floor is not red: {stroke}"


def test_no_trend_section_without_history(page: Page, passing_report: Path) -> None:
    report = ReportPage(page).open(passing_report)
    expect(report.section("Accuracy trend")).to_have_count(0)


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #


def test_scorecard_reports_every_gate_dimension(page: Page, blocked_report: Path) -> None:
    report = ReportPage(page).open(blocked_report)

    for label in ["Accuracy", "pass^3", "Summary quality", "p95 latency", "Cost per case"]:
        expect(report.scorecard_row(label)).to_be_visible()


def test_scorecard_shows_baseline_and_current_side_by_side(
    page: Page, blocked_report: Path
) -> None:
    report = ReportPage(page).open(blocked_report)
    cells = report.scorecard_row("Accuracy").locator("td")

    expect(cells).to_have_count(4)
    expect(cells.nth(1)).to_have_text("100.0%")
    # Two cases regressed outright; the third only flakes, so it still passes by
    # majority and does not move accuracy.
    expect(cells.nth(2)).to_have_text("80.0%")
    expect(cells.nth(3)).to_have_text("-20.0%")


def test_first_run_renders_without_a_baseline(page: Page, first_run_report: Path) -> None:
    report = ReportPage(page).open(first_run_report)

    expect(report.verdict).to_be_visible()
    expect(report.scorecard_row("Accuracy").locator("td").nth(1)).to_have_text("—")
    expect(report.cases).to_have_count(0)


# --------------------------------------------------------------------------- #
# Themes and viewports
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_report_is_legible_in_both_colour_schemes(
    page: Page, blocked_report: Path, scheme: str
) -> None:
    """Dark mode is not decoration - CI artifacts open in whatever the reader uses."""
    page.emulate_media(color_scheme=scheme)
    report = ReportPage(page).open(blocked_report)

    background = _rgb(report.body_background())
    foreground = _rgb(report.body_color())
    assert (
        _contrast(foreground, background) > 4.5
    ), f"{scheme} mode contrast is below WCAG AA: {foreground} on {background}"
    page.screenshot(path=str(ARTIFACTS / f"report-{scheme}.png"), full_page=True)


@pytest.mark.parametrize(
    "name,width,height",
    [("desktop", 1280, 900), ("laptop", 1024, 768), ("mobile", 390, 844)],
)
def test_nothing_overflows_the_viewport(
    page: Page, blocked_report: Path, name: str, width: int, height: int
) -> None:
    """A table that scrolls the whole page sideways is unreadable on a phone."""
    page.set_viewport_size({"width": width, "height": height})
    report = ReportPage(page).open(blocked_report)

    assert (
        report.horizontal_overflow() <= 0
    ), f"the page scrolls {report.horizontal_overflow()}px sideways at {width}px"
    page.screenshot(path=str(ARTIFACTS / f"report-{name}.png"), full_page=True)


def test_case_columns_stack_on_narrow_screens(page: Page, blocked_report: Path) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    report = ReportPage(page).open(blocked_report)

    columns = report.case("tc_0000").locator(".col")
    first = columns.nth(0).bounding_box()
    second = columns.nth(1).bounding_box()
    assert first is not None and second is not None
    assert second["y"] > first["y"], "side-by-side columns did not stack on mobile"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rgb(value: str) -> tuple[float, float, float]:
    parts = value[value.index("(") + 1 : value.index(")")].split(",")
    return tuple(float(p.strip()) for p in parts[:3])  # type: ignore[return-value]


def _rgb_hex(value: str) -> tuple[float, float, float]:
    raw = value.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _contrast(fg: tuple[float, float, float], bg: tuple[float, float, float]) -> float:
    """WCAG relative-contrast ratio."""

    def luminance(colour: tuple[float, float, float]) -> float:
        channels = []
        for raw in colour:
            c = raw / 255
            channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    light, dark = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (light + 0.05) / (dark + 0.05)
