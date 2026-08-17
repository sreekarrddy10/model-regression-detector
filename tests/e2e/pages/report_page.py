"""Page object for the diff report.

The report is a single self-contained file loaded over file://, so there is no
server, no navigation and no network to wait on. What the page object buys here
is a single place where the report's structure is named - if the template
changes, one file moves rather than every assertion.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Locator, Page


class ReportPage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.verdict = page.locator(".verdict")
        self.verdict_tag = page.locator(".verdict .tag")
        self.headline = page.locator(".verdict .headline")
        self.blocking_reasons = page.locator("ul.reasons li.block")
        self.all_reasons = page.locator("ul.reasons li")
        self.scorecard = page.locator("table").first
        self.cases = page.locator("article.case")
        self.critical_pills = page.locator(".pill.critical")
        self.trend_svg = page.locator("svg")
        self.footer = page.locator("footer")

    def open(self, path: Path) -> ReportPage:
        self.page.goto(path.resolve().as_uri())
        # The page has no scripts and no async work; once the DOM is parsed it
        # is final. Waiting on networkidle here would only mask a regression
        # where the report started fetching something.
        self.page.wait_for_load_state("domcontentloaded")
        return self

    def case(self, case_id: str) -> Locator:
        return self.cases.filter(has=self.page.locator("h3", has_text=case_id))

    def section(self, heading: str) -> Locator:
        return self.page.locator("h2", has_text=heading)

    def scorecard_row(self, label: str) -> Locator:
        return self.page.locator("tbody tr").filter(has_text=label).first

    def verdict_class(self) -> str:
        return self.verdict.get_attribute("class") or ""

    def body_background(self) -> str:
        return self.page.evaluate("getComputedStyle(document.body).backgroundColor")

    def body_color(self) -> str:
        return self.page.evaluate("getComputedStyle(document.body).color")

    def horizontal_overflow(self) -> int:
        """Pixels the document scrolls sideways. Anything above zero is a bug."""
        return self.page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
