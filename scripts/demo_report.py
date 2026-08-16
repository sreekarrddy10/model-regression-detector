#!/usr/bin/env python3
"""Generate a sample diff report and Slack payload from a scripted scenario.

This is the Phase 4 proof artifact: it exercises the whole reporting path -
runner, graders, gate, HTML, Slack - with no network and no API key, and writes
a report a human can open.

The cases here are DEMO data, deliberately kept out of data/golden/. Ground truth
stays hand-authored.

Usage:
    python scripts/demo_report.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mrd.alerts import slack  # noqa: E402
from mrd.compare import compare, evaluate  # noqa: E402
from mrd.dataset.loader import Dataset  # noqa: E402
from mrd.dataset.schema import GoldenCase  # noqa: E402
from mrd.graders.calibration import Calibration  # noqa: E402
from mrd.prompts import PromptConfig  # noqa: E402
from mrd.report import html, model  # noqa: E402
from mrd.runner import RunConfig, run  # noqa: E402
from tests.engine_fixtures import ScriptedProvider  # noqa: E402

STAMP = datetime(2026, 8, 16, 12, 30, tzinfo=UTC)
OUT = ROOT / "docs" / "sample-report.html"

DEMO = [
    (
        "tc_0001",
        "You charged my card twice for the September invoice. Please refund the duplicate.",
        "billing",
        "Customer was charged twice in September and wants the duplicate refunded.",
        "easy",
        True,
        "Core billing path. If duplicate-charge refunds misroute, money is at stake.",
    ),
    (
        "tc_0002",
        "The sync broke after your last release and it double-billed our customers. "
        "We want the overcharge back today.",
        "billing",
        "Customer wants a refund for overcharges caused by a sync bug.",
        "ambiguous",
        True,
        "Tie-break rule 1: asks for money back, so billing even though a bug caused it.",
    ),
    (
        "tc_0003",
        "webhook 502s since tues. nothing changed our end. sync is behind ~6h now",
        "technical",
        "Customer reports intermittent webhook 502 errors leaving sync six hours behind.",
        "easy",
        False,
        "Lowercase, abbreviated, no punctuation - realistic engineer-to-support tone.",
    ),
    (
        "tc_0004",
        "Great, another 'improvement' that logs me out every twenty minutes. Love it.",
        "account",
        "Customer is being logged out every twenty minutes and wants it fixed.",
        "adversarial",
        False,
        "Sarcasm: literal reading suggests praise. Tie-break rule 2 puts sign-in under account.",
    ),
    (
        "tc_0005",
        "Hi — we run a consultancy in Lisbon and would like to discuss becoming a reseller.",
        "general",
        "Customer wants to discuss a reseller partnership.",
        "easy",
        False,
        "Partnership enquiry: must not be misfiled as an account request.",
    ),
    (
        "tc_0006",
        "?",
        "general",
        "Customer sent an empty message with no discernible request.",
        "adversarial",
        True,
        "Tie-break rule 4. Degenerate input must not crash or hallucinate a category.",
    ),
]


def demo_cases() -> tuple[GoldenCase, ...]:
    return tuple(
        GoldenCase(
            id=cid,
            input_email=email,
            expected_category=category,  # type: ignore[arg-type]
            expected_summary=summary,
            difficulty=difficulty,  # type: ignore[arg-type]
            critical=critical,
            source="handwritten",
            notes=notes,
            added_at=STAMP,
        )
        for cid, email, category, summary, difficulty, critical, notes in DEMO
    )


async def main() -> int:
    cases = demo_cases()
    dataset = Dataset(path=OUT, cases=cases)
    prompt = PromptConfig.load("v001", root=ROOT / "prompts")

    # Baseline: everything correct.
    baseline_provider = ScriptedProvider(cases, {})
    baseline_provider.judge_score = 5  # type: ignore[attr-defined]
    baseline = await run(
        dataset,
        prompt,
        baseline_provider,
        RunConfig(run_id="run-041", git_sha="a1b2c3d4e5f6", dataset_version="v1", tier="full"),
        dataset_hash="9f2c4a17d3e8",
        judge_provider=baseline_provider,
        now=STAMP,
    )

    # Candidate: a prompt edit broke the money tie-break rule and the degenerate
    # input, and made one case non-deterministic.
    candidate_provider = ScriptedProvider(
        cases,
        {
            "tc_0002": "wrong",
            "tc_0006": "malformed",
            "tc_0004": ["correct", "wrong", "correct"],
        },
    )
    candidate_provider.judge_score = 4  # type: ignore[attr-defined]
    candidate_provider.latency_ms = 145
    candidate_provider.cost_usd = 0.0016
    candidate = await run(
        dataset,
        prompt,
        candidate_provider,
        RunConfig(run_id="run-042", git_sha="f6e5d4c3b2a1", dataset_version="v1", tier="full"),
        dataset_hash="9f2c4a17d3e8",
        judge_provider=candidate_provider,
        now=STAMP,
    )

    trend = (0.98, 0.97, 0.95, 0.94, 0.92, 0.90)
    comparison = compare(baseline, candidate, cases)
    gate = evaluate(comparison, recent_accuracy=trend)

    data = model.build(
        candidate,
        cases,
        gate=gate,
        generated_at=STAMP,
        baseline=baseline,
        comparison=comparison,
        trend=trend,
        calibration=Calibration(
            kappa=0.78, spearman=0.81, sample_count=20, scored_count=20, floor=0.60
        ),
        report_url="https://github.com/example/mrd/actions/runs/9915/artifacts/report.html",
    )

    html.write(data, OUT)

    print(f"verdict:  {gate.verdict}")
    for reason in gate.blocking:
        print(f"  BLOCK   {reason}")
    for reason in gate.warnings:
        print(f"  warn    {reason}")
    print(f"\nreport:   {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes)")
    print(f"McNemar p={comparison.mcnemar_p:.4f}  accuracy {comparison.accuracy_delta:+.1%}")
    print("\nSlack payload:")
    print(json.dumps(slack.build_payload(data), indent=2)[:600] + "\n  …")
    return gate.exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
