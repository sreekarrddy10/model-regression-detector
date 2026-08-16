"""Slack alerting.

The payload is built separately from sending so the message can be asserted in
tests without a network stub, and so a formatting mistake cannot only be
discovered in production.

The webhook URL is a credential: it is read from the environment, never logged,
and redacted out of any error this module raises.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..compare import Verdict
from ..report.model import ReportData

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10
MAX_LISTED_CASES = 5

_ICON = {Verdict.PASS: ":white_check_mark:", Verdict.WARN: ":warning:", Verdict.BLOCK: ":no_entry:"}
_WEBHOOK_RE = re.compile(r"https://hooks\.slack\.com/\S+")


class SlackError(RuntimeError):
    """Slack rejected the message, or could not be reached."""


def redact(text: str) -> str:
    """Strip webhook URLs out of anything destined for a log or an exception."""
    return _WEBHOOK_RE.sub("https://hooks.slack.com/[redacted]", text)


def _fields(data: ReportData) -> list[dict[str, str]]:
    metrics = data.metrics
    comparison = data.comparison
    rows = [
        (
            "Accuracy",
            f"{metrics.accuracy:.1%}"
            + (f"  ({comparison.accuracy_delta:+.1%})" if comparison else ""),
        ),
        ("Cases", f"{metrics.case_count} × {data.run.repeats} repeats"),
        ("Prompt", f"`{data.run.prompt_version}`"),
        ("Dataset", f"`{data.run.dataset_version}`"),
    ]
    if comparison:
        rows.append(
            (
                "Changed",
                f"{len(comparison.regressed)} regressed / {len(comparison.improved)} improved",
            )
        )
        rows.append(("McNemar p", f"{comparison.mcnemar_p:.4f}"))
    if data.calibration:
        rows.append(("Judge κ", f"{data.calibration.kappa:.2f}"))
    if metrics.cost_per_case is not None:
        rows.append(("Cost/case", f"${metrics.cost_per_case:.4f}"))

    return [{"type": "mrkdwn", "text": f"*{label}*\n{value}"} for label, value in rows]


def build_payload(data: ReportData) -> dict[str, Any]:
    """Block Kit message.

    Leads with the verdict and the reasons it fired, because the reader is
    deciding in three seconds whether this needs them right now.
    """
    verdict = data.gate.verdict
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{_ICON[verdict]} Eval {verdict}: {data.run.prompt_version}",
                "emoji": True,
            },
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": data.headline}},
        {"type": "section", "fields": _fields(data)},
    ]

    reasons = [f"• *{r}*" for r in data.gate.blocking] + [f"• {r}" for r in data.gate.warnings]
    if reasons:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(reasons)}})

    regressed = data.regressed
    if regressed:
        listed = regressed[:MAX_LISTED_CASES]
        lines = [
            f"`{d.case_id}`{' *critical*' if d.critical else ''} — "
            f"expected `{d.expected_category}`, got `{d.candidate_categories}`"
            for d in listed
        ]
        # Never silently truncate: a reader who cannot tell 5 from 40 will
        # under-react to the larger failure.
        if len(regressed) > len(listed):
            lines.append(f"_…and {len(regressed) - len(listed)} more, see the full report._")
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Regressions*\n" + "\n".join(lines)},
            }
        )

    context = f"run `{data.run.run_id}` · commit `{data.run.git_sha[:12]}` · {data.run.tier} tier"
    if data.report_url:
        context += f" · <{data.report_url}|full diff report>"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": context}]})

    return {"text": f"Eval {verdict}: {data.headline}", "blocks": blocks}


def send(data: ReportData, *, webhook_url: str | None = None) -> bool:
    """Post the alert. Returns False when no webhook is configured.

    A missing webhook is not an error: local runs and forks should not fail
    because nobody wired up Slack.
    """
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        logger.info("SLACK_WEBHOOK_URL not set; skipping alert")
        return False

    # Only https is opened. Without this, a misconfigured or attacker-supplied
    # SLACK_WEBHOOK_URL of file:// or a custom scheme would be dereferenced here.
    scheme = urllib.parse.urlparse(url).scheme
    if scheme != "https":
        raise SlackError(f"webhook URL must use https, got {scheme or 'no'} scheme")

    payload = json.dumps(build_payload(data)).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(  # noqa: S310  # nosec B310
            request, timeout=TIMEOUT_SECONDS
        ) as response:
            if response.status != 200:
                raise SlackError(f"Slack returned HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise SlackError(f"could not reach Slack: {redact(str(exc))}") from None
    return True
