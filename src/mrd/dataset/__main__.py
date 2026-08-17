"""Dataset CLI: validate, report, lock, verify, new.

python -m mrd.dataset validate
python -m mrd.dataset report
python -m mrd.dataset lock --version v1
python -m mrd.dataset verify
python -m mrd.dataset new --id tc_0007
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import hashing, report
from .loader import DatasetValidationError, load_cases, load_holdout

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "data" / "golden" / "emails.jsonl"
DEFAULT_HOLDOUT = ROOT / "data" / "golden" / "judge_holdout.jsonl"
DEFAULT_LOCK = ROOT / "data" / "golden" / "dataset.lock.json"
DEFAULT_PROMPTS = ROOT / "prompts"


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


def next_case_id(path: Path) -> str:
    """The lowest unused tc_NNNN id.

    Authoring eighty cases should not also mean tracking eighty ids by hand,
    and a mistyped id is a duplicate the loader will reject on the next run.
    """
    used = set()
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                try:
                    used.add(json.loads(raw).get("id"))
                except json.JSONDecodeError:
                    continue
    index = 1
    while f"tc_{index:04d}" in used:
        index += 1
    return f"tc_{index:04d}"


def cmd_new(args: argparse.Namespace) -> int:
    """Emit a blank case row for a human to fill in."""
    template = {
        "id": args.id or next_case_id(args.cases),
        "input_email": "",
        "expected_category": "billing",
        "expected_summary": "",
        "difficulty": "easy",
        "critical": False,
        "source": "handwritten",
        "notes": "",
        "added_at": datetime.now(UTC).isoformat(),
    }
    _out(json.dumps(template, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Paths live on a shared parent so they can be passed after the subcommand,
    # the order every other CLI accepts them in.
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    paths.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    paths.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    paths.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)

    parser = argparse.ArgumentParser(prog="mrd.dataset", description=__doc__, parents=[paths])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="parse and validate; fail on any error", parents=[paths])
    sub.add_parser("report", help="coverage report with gaps", parents=[paths])
    sub.add_parser("verify", help="check the dataset against its lock file", parents=[paths])

    lock = sub.add_parser("lock", help="write the lock file", parents=[paths])
    lock.add_argument("--version", default="v1")

    new = sub.add_parser("new", help="print a blank case row", parents=[paths])
    new.add_argument("--id", default=None, help="default: the next unused tc_NNNN")

    return parser


_COMMANDS = {
    "validate": cmd_validate,
    "report": cmd_report,
    "lock": cmd_lock,
    "verify": cmd_verify,
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
