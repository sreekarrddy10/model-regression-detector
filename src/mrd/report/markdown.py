"""PR comment rendering.

The comment carries a hidden marker so CI can find and update its own previous
comment instead of appending a new one to every push. A PR with eleven stacked
eval comments is one nobody reads.
"""

from __future__ import annotations

from ..compare import Verdict
from .model import ReportData

MARKER = "<!-- mrd-eval-report -->"

_BADGE = {
    Verdict.PASS: "✅ **PASS**",
    Verdict.WARN: "⚠️ **WARN**",
    Verdict.BLOCK: "⛔ **BLOCK**",
}

MAX_LISTED_CASES = 10


def _delta(value: float | None, *, pct: bool = True, invert: bool = False) -> str:
    if value is None:
        return "—"
    arrow = ""
    if abs(value) > 1e-9:
        worse = value > 0 if invert else value < 0
        arrow = " 🔻" if worse else " 🔹"
    return (f"{value:+.1%}" if pct else f"{value:+.2f}") + arrow


def _row(label: str, baseline: str, current: str, delta: str) -> str:
    return f"| {label} | {baseline} | {current} | {delta} |"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def _scorecard(data: ReportData) -> list[str]:
    """The four dimensions the gate reasons about, baseline against current."""
    c = data.comparison
    base = c.baseline if c else None
    now = data.metrics

    return [
        "| Metric | Baseline | This run | Δ |",
        "|---|---:|---:|---:|",
        _row(
            "Accuracy",
            _pct(base.accuracy if base else None),
            _pct(now.accuracy),
            _delta(c.accuracy_delta if c else None),
        ),
        _row(
            f"pass^{data.run.repeats}",
            _pct(base.pass_hat_k_rate if base else None),
            _pct(now.pass_hat_k_rate),
            "—",
        ),
        _row(
            "Summary quality",
            _score(base.judge_mean if base else None),
            _score(now.judge_mean),
            _delta(c.judge_delta if c else None, pct=False),
        ),
        _row(
            "p95 latency",
            _ms(base.p95_latency_ms if base else None),
            _ms(now.p95_latency_ms),
            _delta(c.latency_ratio if c else None, invert=True),
        ),
        _row(
            "Cost / case",
            _usd(base.cost_per_case if base else None),
            _usd(now.cost_per_case),
            _delta(c.cost_ratio if c else None, invert=True),
        ),
    ]


def render(data: ReportData) -> str:
    """Markdown body for a sticky PR comment."""
    c = data.comparison
    lines = [
        MARKER,
        f"## {_BADGE[data.gate.verdict]} — eval `{data.run.prompt_version}` on "
        f"dataset `{data.run.dataset_version}`",
        "",
        data.headline,
        "",
    ]

    if data.gate.blocking:
        lines.append("### Blocking")
        lines += [f"- **{reason}**" for reason in data.gate.blocking]
        lines.append("")
    if data.gate.warnings:
        lines.append("### Warnings")
        lines += [f"- {reason}" for reason in data.gate.warnings]
        lines.append("")

    lines += _scorecard(data) + [""]

    if c:
        lines += [
            f"{len(c.regressed)} regressed · {len(c.improved)} improved · "
            f"McNemar exact `p = {c.mcnemar_p:.4f}` on "
            f"{len(c.regressed) + len(c.improved)} discordant pair(s). "
            f"A case counts as regressed only when it fails a majority of "
            f"{data.run.repeats} repeats.",
            "",
        ]

    if data.regressed:
        listed = data.regressed[:MAX_LISTED_CASES]
        lines += [
            "### Regressed cases",
            "",
            "| Case | Expected | Got | Difficulty |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| `{d.case_id}`{' ⚠️ critical' if d.critical else ''} | `{d.expected_category}` "
            f"| `{d.candidate_categories}` | {d.difficulty} |"
            for d in listed
        ]
        if len(data.regressed) > len(listed):
            lines.append(
                f"\n_…and {len(data.regressed) - len(listed)} more. "
                "Full detail in the report artifact._"
            )
        lines.append("")

    if data.calibration:
        state = "trusted" if data.calibration.passed else "**not trusted**"
        lines.append(
            f"Judge calibration: κ = `{data.calibration.kappa:.2f}` over "
            f"{data.calibration.scored_count} human-scored summaries — {state}."
        )
    else:
        lines.append(
            "Judge calibration: not run. Any summary-quality number above is uninterpreted."
        )

    lines += [
        "",
        f"<sub>run `{data.run.run_id}` · {data.run.tier} tier · {data.metrics.case_count} cases × "
        f"{data.run.repeats} repeats · commit `{data.run.git_sha[:12]}`"
        + (f" · [full report]({data.report_url})" if data.report_url else "")
        + "</sub>",
    ]

    return "\n".join(lines) + "\n"
