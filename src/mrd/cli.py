"""Eval CLI - the entry point CI calls.

    python -m mrd.cli eval --tier smoke

Exit code is the gate's: 0 for PASS or WARN, 1 for BLOCK. That is the whole
contract with CI - everything else it produces (HTML report, PR comment, Slack
message) is for humans.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import paths, sampling
from .alerts import slack
from .compare import GateReport, IncomparableRuns, Verdict, compare, evaluate
from .dataset import hashing
from .dataset.loader import DatasetValidationError, load_cases, load_holdout
from .graders.calibration import DEFAULT_KAPPA_FLOOR, calibrate
from .prompts import PromptConfig
from .providers.registry import build as build_provider
from .report import html, markdown, model
from .results import RunOutcome
from .runner import RunConfig, run
from .store import sqlite

logger = logging.getLogger("mrd")


def _defaults() -> dict[str, Path]:
    """Resolved at call time, not import time, so tests and containers can chdir."""
    return {
        "cases": paths.resolve("data/golden/emails.jsonl"),
        "holdout": paths.resolve("data/golden/judge_holdout.jsonl"),
        "lock": paths.resolve("data/golden/dataset.lock.json"),
        "prompts": paths.resolve("prompts", env_var="MRD_PROMPTS_PATH"),
        "cassettes": paths.resolve("tests/cassettes"),
        "db": paths.resolve("data/runs.sqlite"),
        "report": paths.resolve("reports/report.html"),
        "comment": paths.resolve("reports/comment.md"),
    }


def _out(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")


async def _execute(args: argparse.Namespace) -> int:
    started = datetime.now(UTC)

    # --- ground truth ------------------------------------------------------ #
    dataset = load_cases(args.cases, prompts_root=args.prompts)
    if not dataset.cases:
        _err(
            "error: the golden dataset is empty. Author cases first:\n"
            "  make dataset-new ID=tc_0001   then   make dataset-report"
        )
        return 1

    lock = hashing.verify(dataset, args.lock)
    dataset_hash = lock.sha256
    holdout = load_holdout(args.holdout, dataset=dataset) if args.holdout.exists() else ()

    prompt = (
        PromptConfig.load(args.prompt, root=args.prompts)
        if args.prompt
        else PromptConfig.latest(root=args.prompts)
    )
    cases = sampling.select(dataset, args.tier, smoke_size=args.smoke_size)
    _out(
        f"{args.tier} tier · prompt {prompt.version_id} · {len(cases)}/{len(dataset)} cases "
        f"× {args.repeats} repeats · dataset {lock.version} ({dataset_hash[:12]})"
    )

    provider = build_provider(prompt.model, tier=args.tier, cassette_dir=args.cassettes)
    judge_model = None if args.no_judge else args.judge_model
    judge_provider = (
        None
        if judge_model is None
        else build_provider(judge_model, tier=args.tier, cassette_dir=args.cassettes)
    )

    # --- is the judge trustworthy? ----------------------------------------- #
    calibration = None
    if judge_provider is not None and holdout:
        calibration = await calibrate(
            holdout,
            emails={c.id: c.input_email for c in dataset},
            references={c.id: c.expected_summary for c in dataset},
            provider=judge_provider,
            model=judge_model or "",
            floor=args.kappa_floor,
        )
        _out(f"judge calibration: {calibration.reason}")
        if not calibration.passed:
            # Deterministic grading still runs and still reports - losing the
            # judge must not cost us the signal we can trust. The gate blocks on
            # the calibration failure itself.
            judge_provider, judge_model = None, None

    run_id = args.run_id or f"{started:%Y%m%dT%H%M%SZ}-{args.git_sha[:7]}"
    outcome = await run(
        dataset,
        prompt,
        provider,
        RunConfig(
            run_id=run_id,
            git_sha=args.git_sha,
            dataset_version=lock.version,
            tier=args.tier,
            repeats=args.repeats,
            concurrency=args.concurrency,
            judge_model=judge_model,
        ),
        dataset_hash=dataset_hash,
        judge_provider=judge_provider,
        judge_kappa=calibration.kappa if calibration else None,
        cases=cases,
    )

    # --- history and comparison -------------------------------------------- #
    with sqlite.connect(args.db) as conn:
        sqlite.initialize(conn)
        sqlite.save(conn, outcome)
        baseline: RunOutcome | None = sqlite.latest_baseline(
            conn, dataset_hash=dataset_hash, exclude_run_id=run_id
        )
        trend = sqlite.recent_accuracy(conn, dataset_hash=dataset_hash)

    comparison = None
    gate = GateReport(verdict=Verdict.PASS)
    if baseline is None:
        _out("no comparable baseline yet; recording this run as the first one")
    else:
        comparison = compare(baseline, outcome, cases)
        gate = evaluate(
            comparison,
            judge_calibrated=calibration is None or calibration.passed,
            judge_confirmations=args.judge_confirmations,
            recent_accuracy=trend[:-1],
        )

    data = model.build(
        outcome,
        cases,
        gate=gate,
        generated_at=datetime.now(UTC),
        baseline=baseline,
        comparison=comparison,
        trend=trend,
        calibration=calibration,
        report_url=args.report_url,
    )

    # --- outputs ----------------------------------------------------------- #
    html.write(data, args.report)
    args.comment.parent.mkdir(parents=True, exist_ok=True)
    args.comment.write_text(markdown.render(data), encoding="utf-8")

    _out(f"\n{gate.verdict}: {data.headline}")
    for reason in gate.blocking:
        _out(f"  BLOCK  {reason}")
    for reason in gate.warnings:
        _out(f"  warn   {reason}")
    _out(f"\nreport:  {args.report}\ncomment: {args.comment}")

    if not args.no_slack:
        try:
            if slack.send(data):
                _out("slack:   sent")
        except slack.SlackError as exc:
            # Alerting is not the gate. A Slack outage must not turn a passing
            # run into a failing one, or vice versa.
            _err(f"warning: slack alert failed: {exc}")

    return gate.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrd", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("eval", help="run the eval suite and apply the gate")

    run_cmd.add_argument("--tier", choices=["unit", "smoke", "full"], default="smoke")
    run_cmd.add_argument("--prompt", default=None, help="prompt version (default: latest)")
    run_cmd.add_argument("--repeats", type=int, default=3)
    run_cmd.add_argument("--concurrency", type=int, default=8)
    run_cmd.add_argument("--smoke-size", type=int, default=sampling.DEFAULT_SMOKE_SIZE)
    run_cmd.add_argument("--judge-model", default="gpt-4o")
    run_cmd.add_argument("--no-judge", action="store_true")
    run_cmd.add_argument("--kappa-floor", type=float, default=DEFAULT_KAPPA_FLOOR)
    run_cmd.add_argument("--judge-confirmations", type=int, default=0)
    run_cmd.add_argument("--run-id", default=None)
    run_cmd.add_argument("--git-sha", default="unknown")
    run_cmd.add_argument("--report-url", default=None)
    run_cmd.add_argument("--no-slack", action="store_true")
    for name, default in _defaults().items():
        run_cmd.add_argument(f"--{name}", type=Path, default=default)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_execute(args))
    except (
        DatasetValidationError,
        hashing.LockMismatch,
        IncomparableRuns,
        FileNotFoundError,
    ) as exc:
        _err(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
