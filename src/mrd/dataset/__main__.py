"""Dataset CLI: validate, report, lock, verify, new.

python -m mrd.dataset validate
python -m mrd.dataset report
python -m mrd.dataset lock --version v1
python -m mrd.dataset verify
python -m mrd.dataset new --id tc_0007
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from .. import paths
from . import authoring, hashing, report
from .loader import DatasetValidationError, load_cases, load_holdout

DEFAULT_CASES = paths.resolve("data/golden/emails.jsonl")
DEFAULT_HOLDOUT = paths.resolve("data/golden/judge_holdout.jsonl")
DEFAULT_LOCK = paths.resolve("data/golden/dataset.lock.json")
DEFAULT_PROMPTS = paths.resolve("prompts", env_var="MRD_PROMPTS_PATH")
DEFAULT_CASES_YAML = paths.resolve("data/golden/cases.yaml")
DEFAULT_HOLDOUT_YAML = paths.resolve("data/golden/holdout.yaml")


def _out(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")


def _load(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    dataset = load_cases(args.cases, prompts_root=args.prompts)
    holdout = load_holdout(args.holdout, dataset=dataset) if args.holdout.exists() else ()
    return dataset, holdout


def cmd_validate(args: argparse.Namespace) -> int:
    dataset, holdout = _load(args)
    _out(f"OK: {len(dataset)} case(s), {len(holdout)} holdout sample(s), no errors.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    dataset, holdout = _load(args)
    result = report.build(dataset, holdout)
    _out(report.render(result))
    # Gaps are progress information, not failure: exit 0 so `make dataset-report`
    # is usable daily while the set is being written.
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    dataset, _ = _load(args)
    lock = hashing.build_lock(dataset, version=args.version, now=datetime.now(UTC))
    args.lock.write_text(lock.to_json(), encoding="utf-8")
    _out(f"Locked {lock.count} case(s) as {lock.version}\n  sha256 {lock.sha256}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    dataset, _ = _load(args)
    lock = hashing.verify(dataset, args.lock)
    _out(f"Dataset matches lock {lock.version} ({lock.count} cases).")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Compile the authoring YAML into the canonical JSONL."""
    now = datetime.now(UTC)
    result = authoring.build_cases(args.cases_yaml, args.cases, now=now)
    authoring.write_jsonl(result.cases, args.cases)
    _out(f"{args.cases.name}: {result.summary}")

    # Validate the cases - leakage included - before the holdout gets a chance to
    # raise. An incomplete holdout must not hide a leaked case.
    load_cases(args.cases, prompts_root=args.prompts)

    if args.holdout_yaml.exists():
        samples = authoring.build_holdout(args.holdout_yaml, args.holdout, now=now)
        authoring.write_jsonl(samples, args.holdout)
        _out(f"{args.holdout.name}: {len(samples)} sample(s)")

    # Re-read through the real loader, so `build` can never emit something
    # `validate` would reject.
    dataset, holdout = _load(args)
    _out(f"validated: {len(dataset)} case(s), {len(holdout)} holdout sample(s)")
    _out(report.render(report.build(dataset, holdout)))
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    """Print a blank case stanza for cases.yaml.

    YAML, not a JSONL row. Appending a row to emails.jsonl used to be the
    workflow; it is now the *generated* artifact, so anything written there is
    silently erased by the next `build`. Emitting the authoring format makes
    that mistake impossible to reach.

    Fields are left empty on purpose - `build` names each missing one.
    """
    lines = [
        "",
        "  # Written " + datetime.now(UTC).date().isoformat(),
        "  - input_email: |",
        "      ",
        '    expected_category: ""   # billing | technical | account | general',
        '    expected_summary: ""',
        '    difficulty: ""          # easy | ambiguous | adversarial',
        "    critical: false",
        '    notes: ""               # why this case exists',
    ]
    if args.id:
        lines.insert(2, f"    id: {args.id}")
    _out("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Paths live on a shared parent so they can be passed after the subcommand,
    # the order every other CLI accepts them in.
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    paths.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    paths.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    paths.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    paths.add_argument("--cases-yaml", type=Path, default=DEFAULT_CASES_YAML)
    paths.add_argument("--holdout-yaml", type=Path, default=DEFAULT_HOLDOUT_YAML)

    parser = argparse.ArgumentParser(prog="mrd.dataset", description=__doc__, parents=[paths])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="parse and validate; fail on any error", parents=[paths])
    sub.add_parser("report", help="coverage report with gaps", parents=[paths])
    sub.add_parser("verify", help="check the dataset against its lock file", parents=[paths])

    lock = sub.add_parser("lock", help="write the lock file", parents=[paths])
    lock.add_argument("--version", default="v1")

    sub.add_parser("build", help="compile cases.yaml -> emails.jsonl (idempotent)", parents=[paths])

    new = sub.add_parser("new", help="print a blank case row", parents=[paths])
    new.add_argument("--id", default=None, help="pin an id; otherwise build assigns one")

    return parser


_COMMANDS = {
    "validate": cmd_validate,
    "report": cmd_report,
    "lock": cmd_lock,
    "verify": cmd_verify,
    "build": cmd_build,
    "new": cmd_new,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (DatasetValidationError, hashing.LockMismatch, FileNotFoundError) as exc:
        _err(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
