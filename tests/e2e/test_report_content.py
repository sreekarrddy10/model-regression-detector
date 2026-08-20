"""Browser tests for hostile and pathological case content.

Golden cases are hand-authored, but the emails inside them are quoted from real
customers: pasted stack traces, base64 blobs, 40-line threads, occasionally
markup. All of it renders into the report. These tests use content the short
fixture emails cannot expose.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from mrd.compare import GateReport, Verdict
from mrd.dataset.schema import GoldenCase
from mrd.report import html, model

from ..engine_fixtures import make_outcome
from .pages.report_page import ReportPage

pytestmark = pytest.mark.e2e

STAMP = datetime(2026, 8, 17)


def _case(**overrides: object) -> GoldenCase:
    payload: dict[str, object] = {
        "id": "tc_0001",
        "input_email": "Ordinary email.",
        "expected_category": "billing",
        "expected_summary": "Ordinary summary.",
        "difficulty": "easy",
        "critical": False,
        "source": "handwritten",
        "notes": "Ordinary note.",
        "added_at": STAMP,
    }
    payload.update(overrides)
    return GoldenCase.model_validate(payload)


# Flaky by default, deliberately: the report only renders cases needing
# attention, so an all-passing case produces a page with no case detail at all -
# and a layout assertion against that page passes without testing anything.
def _render(
    case: GoldenCase, tmp_path: Path, *, flags: tuple[bool, ...] = (True, False, True)
) -> Path:
    data = model.build(
        make_outcome({case.id: flags}),
        (case,),
        gate=GateReport(verdict=Verdict.PASS),
        generated_at=STAMP,
    )
    return html.write(data, tmp_path / "report.html")


# --------------------------------------------------------------------------- #
# Hostile content
# --------------------------------------------------------------------------- #


def test_markup_in_case_content_does_not_execute(page: Page, tmp_path: Path) -> None:
    """Case text is data. A customer quoting HTML must not become HTML."""
    executed: list[str] = []
    page.on("dialog", lambda d: (executed.append(d.message), d.dismiss()))
    page.on("pageerror", lambda e: executed.append(str(e)))

    hostile = _case(
        input_email="<script>window.__pwned = 1; alert('xss')</script>",
        expected_summary='<img src=x onerror="window.__pwned = 2">',
        notes="<iframe src='https://evil.example'></iframe>",
    )
    report = ReportPage(page).open(_render(hostile, tmp_path))

    assert page.evaluate("window.__pwned === undefined"), "injected script ran"
    assert page.evaluate("document.scripts.length") == 0
    assert page.locator("iframe").count() == 0
    assert page.locator("img").count() == 0
    assert executed == []
    # The text is still shown - escaped, not swallowed. A reader debugging this
    # case needs to see what the customer actually sent.
    expect(report.case("tc_0001")).to_contain_text("<script>")


def test_no_off_disk_request_from_case_content(page: Page, tmp_path: Path) -> None:
    off_disk: list[str] = []
    page.on("request", lambda r: None if r.url.startswith("file://") else off_disk.append(r.url))

    hostile = _case(
        input_email='<link rel=stylesheet href="https://evil.example/x.css">',
        notes='<img src="https://evil.example/track.gif">',
    )
    ReportPage(page).open(_render(hostile, tmp_path))

    assert off_disk == [], f"case content caused an off-disk fetch: {off_disk}"


# --------------------------------------------------------------------------- #
# Pathological content
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("width", [1280, 390])
def test_an_unbroken_token_does_not_break_the_layout(
    page: Page, tmp_path: Path, width: int
) -> None:
    """A pasted base64 blob or a long URL has no spaces to wrap on.

    `white-space: pre-wrap` wraps on whitespace; without `overflow-wrap` a single
    long token runs straight off the page and takes the whole document with it.
    """
    page.set_viewport_size({"width": width, "height": 900})
    blob = "A" * 600
    report = ReportPage(page).open(_render(_case(input_email=f"Log attached: {blob}"), tmp_path))

    assert report.horizontal_overflow() <= 0, (
        f"a {len(blob)}-char unbroken token scrolls the page "
        f"{report.horizontal_overflow()}px sideways at {width}px"
    )


def test_a_long_url_in_a_summary_does_not_break_the_layout(page: Page, tmp_path: Path) -> None:
    page.set_viewport_size({"width": 390, "height": 900})
    url = "https://app.example.com/very/deep/path?" + "&".join(f"k{i}=v{i}" for i in range(40))
    report = ReportPage(page).open(
        _render(_case(expected_summary=f"Customer linked {url}"), tmp_path)
    )

    assert report.horizontal_overflow() <= 0


def test_a_long_email_thread_stays_readable(page: Page, tmp_path: Path) -> None:
    thread = "\n".join(
        f"> On day {i}, the customer wrote about their billing issue." for i in range(60)
    )
    report = ReportPage(page).open(_render(_case(input_email=thread), tmp_path))

    expect(report.case("tc_0001")).to_be_visible()
    assert report.horizontal_overflow() <= 0


def test_wide_content_scrolls_inside_its_own_box_not_the_page(page: Page, tmp_path: Path) -> None:
    """If something must overflow, it overflows locally - never the document."""
    page.set_viewport_size({"width": 390, "height": 900})
    ReportPage(page).open(_render(_case(input_email="X" * 400), tmp_path))

    body_overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert body_overflow <= 0


def test_the_fixture_actually_renders_case_detail(page: Page, tmp_path: Path) -> None:
    """Guard against the vacuous-pass trap the tests above fell into once.

    The report shows only cases needing attention. If a fixture produces a page
    with no case detail, every assertion about case content above it passes
    without testing anything.
    """
    report = ReportPage(page).open(_render(_case(input_email="MARKER-TEXT"), tmp_path))

    expect(report.cases).to_have_count(1)
    expect(report.case("tc_0001")).to_contain_text("MARKER-TEXT")
