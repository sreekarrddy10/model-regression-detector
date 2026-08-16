from __future__ import annotations

import json
from typing import Any

import pytest

from mrd.alerts import slack
from mrd.compare import Verdict

from .test_report import ALL_PASS, build

pytestmark = pytest.mark.unit


def texts(payload: dict[str, Any]) -> str:
    """Every string in the message, flattened, for substring assertions."""
    return json.dumps(payload)


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #


def test_verdict_leads_the_message() -> None:
    broken = {**ALL_PASS}
    for i in range(3):
        broken[f"tc_{i:04d}"] = (False,) * 3
    data = build(broken, baseline_flags=ALL_PASS)
    payload = slack.build_payload(data)

    assert data.gate.verdict is Verdict.BLOCK
    assert payload["blocks"][0]["type"] == "header"
    assert "BLOCK" in payload["blocks"][0]["text"]["text"]
    assert ":no_entry:" in payload["blocks"][0]["text"]["text"]


def test_fallback_text_is_set_for_notifications() -> None:
    """Slack shows `text` in the push notification; an empty one is a silent alert."""
    payload = slack.build_payload(build(ALL_PASS, baseline_flags=ALL_PASS))
    assert payload["text"].startswith("Eval PASS:")


@pytest.mark.parametrize(
    "verdict_flags,icon",
    [
        (ALL_PASS, ":white_check_mark:"),
        ({**ALL_PASS, "tc_0003": (True, False, True)}, ":warning:"),
        ({**ALL_PASS, "tc_0000": (False, False, False)}, ":no_entry:"),
    ],
)
def test_icon_matches_the_verdict(verdict_flags: dict[str, tuple[bool, ...]], icon: str) -> None:
    payload = slack.build_payload(build(verdict_flags, baseline_flags=ALL_PASS))
    assert icon in payload["blocks"][0]["text"]["text"]


def test_headline_numbers_are_present() -> None:
    payload = slack.build_payload(
        build({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS)
    )
    flat = texts(payload)

    assert "Accuracy" in flat
    assert "McNemar p" in flat
    assert "1 regressed / 0 improved" in flat


def test_blocking_reasons_are_included() -> None:
    payload = slack.build_payload(
        build({**ALL_PASS, "tc_0000": (False,) * 3}, baseline_flags=ALL_PASS)
    )
    assert "critical case(s) regressed" in texts(payload)


def test_regressed_cases_are_named_with_what_changed() -> None:
    payload = slack.build_payload(
        build({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS)
    )
    flat = texts(payload)

    assert "tc_0003" in flat
    assert "expected" in flat


def test_truncation_is_stated_never_silent() -> None:
    """A reader who cannot tell 5 regressions from 40 will under-react."""
    broken = {**ALL_PASS}
    for i in range(9):
        broken[f"tc_{i:04d}"] = (False,) * 3
    payload = slack.build_payload(build(broken, baseline_flags=ALL_PASS))

    assert "and 4 more" in texts(payload)


def test_short_lists_are_not_annotated() -> None:
    payload = slack.build_payload(
        build({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS)
    )
    assert "more, see the full report" not in texts(payload)


def test_report_link_is_included_when_known() -> None:
    payload = slack.build_payload(
        build(ALL_PASS, baseline_flags=ALL_PASS, report_url="https://ci.example/report.html")
    )
    assert "<https://ci.example/report.html|full diff report>" in texts(payload)


def test_clean_run_produces_a_short_message() -> None:
    payload = slack.build_payload(build(ALL_PASS, baseline_flags=ALL_PASS))
    flat = texts(payload)

    assert ":white_check_mark:" in flat
    assert "No cases changed" in flat
    assert "Regressions" not in flat


# --------------------------------------------------------------------------- #
# Credential handling
# --------------------------------------------------------------------------- #


def test_webhook_urls_are_redacted() -> None:
    message = "failed posting to https://hooks.slack.com/services/T000/B111/xxxxSECRETxxxx now"
    redacted = slack.redact(message)

    assert "SECRET" not in redacted
    assert "[redacted]" in redacted


def test_missing_webhook_is_skipped_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forks and local runs must not fail because nobody wired up Slack."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert slack.send(build(ALL_PASS, baseline_flags=ALL_PASS)) is False


def test_send_posts_json_to_the_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: int = 0) -> FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["headers"] = request.headers
        return FakeResponse()

    monkeypatch.setattr(slack.urllib.request, "urlopen", fake_urlopen)

    sent = slack.send(
        build(ALL_PASS, baseline_flags=ALL_PASS),
        webhook_url="https://hooks.slack.com/services/T/B/secret",
    )

    assert sent is True
    assert captured["url"].startswith("https://hooks.slack.com/")
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["body"]["blocks"][0]["type"] == "header"


def test_non_200_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 500

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(slack.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    with pytest.raises(slack.SlackError, match="HTTP 500"):
        slack.send(
            build(ALL_PASS, baseline_flags=ALL_PASS), webhook_url="https://hooks.slack.com/x"
        )


def test_unreachable_slack_does_not_leak_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("refused by https://hooks.slack.com/services/T/B/SECRET")

    monkeypatch.setattr(slack.urllib.request, "urlopen", boom)

    with pytest.raises(slack.SlackError) as exc:
        slack.send(
            build(ALL_PASS, baseline_flags=ALL_PASS),
            webhook_url="https://hooks.slack.com/services/T/B/SECRET",
        )

    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://hooks.slack.com/x", "hooks.slack.com/x"]
)
def test_non_https_webhooks_are_refused(url: str) -> None:
    """A misconfigured webhook must not become a file read or a cleartext post."""
    with pytest.raises(slack.SlackError, match="must use https"):
        slack.send(build(ALL_PASS, baseline_flags=ALL_PASS), webhook_url=url)


def test_calibration_appears_in_the_alert() -> None:
    from mrd.graders.calibration import Calibration

    payload = slack.build_payload(
        build(
            ALL_PASS,
            baseline_flags=ALL_PASS,
            calibration=Calibration(
                kappa=0.77, spearman=0.8, sample_count=20, scored_count=20, floor=0.6
            ),
        )
    )
    assert "0.77" in texts(payload)
