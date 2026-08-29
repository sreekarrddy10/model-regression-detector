#!/usr/bin/env python3
"""Convert hand-authored case/holdout files into the repo's authoring schema.

The authored files use a slightly different vocabulary - `input` for
`input_email`, `category` for `expected_category`, a bare top-level list instead
of a `cases:` key. This maps those names without touching a single label: every
`expected_category` and `expected_summary` is copied verbatim, because those are
the ground truth and a script has no business editing them.

Usage:
    python scripts/adapt_authored_cases.py IN_CASES.yaml IN_HOLDOUT.yaml \\
        --scorer "your name" [--out-dir data/golden]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASE_RENAMES = {"input": "input_email", "category": "expected_category"}
HOLDOUT_RENAMES = {"input": "email"}
CASE_KEEP = {
    "id",
    "input_email",
    "expected_category",
    "expected_summary",
    "difficulty",
    "strata",
    "source",
    "notes",
    "added_at",
}
HOLDOUT_KEEP = {
    "id",
    "candidate_summary",
    "human_score",
    "scorer",
    "scored_at",
    "case_id",
    "email",
    "reference_summary",
    "failure_mode",
    "rationale",
}


CASE_ORDER = [
    "id",
    "input_email",
    "expected_category",
    "expected_summary",
    "difficulty",
    "strata",
    "source",
    "notes",
    "added_at",
]
HOLDOUT_ORDER = [
    "id",
    "case_id",
    "email",
    "reference_summary",
    "candidate_summary",
    "human_score",
    "scorer",
    "failure_mode",
    "rationale",
    "scored_at",
]


def _ordered(row: dict[str, Any], order: list[str]) -> dict[str, Any]:
    """Stable, human-meaningful key order - id first, prose next, tags last."""
    return {k: row[k] for k in order if k in row} | {k: v for k, v in row.items() if k not in order}


def _load(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Already in repo shape.
        for key in ("cases", "samples"):
            if key in data:
                return list(data[key] or [])
        raise SystemExit(f"{path}: expected a list, or a mapping with cases/samples")
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a top-level list")
    return data


def adapt_cases(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        row = {CASE_RENAMES.get(k, k): v for k, v in entry.items()}
        # A bare `critical: true` from the older shape folds into strata.
        strata = list(row.get("strata") or [])
        if row.pop("critical", False) and "critical" not in strata:
            strata.append("critical")
        row["strata"] = strata
        dropped = sorted(set(row) - CASE_KEEP)
        for key in dropped:
            row.pop(key)
        out.append(_ordered(row, CASE_ORDER))
    return out


def adapt_holdout(entries: list[dict[str, Any]], *, scorer: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        row = {HOLDOUT_RENAMES.get(k, k): v for k, v in entry.items()}
        row.setdefault("scorer", scorer)
        # `reference_summary` is deliberately NOT invented. Without it (and
        # without a case_id) the sample cannot be scored, and `build` will say so
        # rather than quietly calibrating against something made up.
        for key in sorted(set(row) - HOLDOUT_KEEP):
            row.pop(key)
        out.append(_ordered(row, HOLDOUT_ORDER))
    return out


class _BlockDumper(yaml.SafeDumper):
    """Emits multi-line strings as `|` block scalars.

    Without this, safe_dump quotes them and folds the line breaks into blank
    lines - the content survives a round trip but becomes unreadable, which
    defeats the entire point of authoring in YAML.
    """


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _str_representer)


def _dump(rows: list[dict[str, Any]], key: str, header: str) -> str:
    body = yaml.dump(
        {key: rows},
        Dumper=_BlockDumper,
        sort_keys=False,
        allow_unicode=True,
        width=88,
        default_flow_style=False,
    )
    return header + "\n" + body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "golden")
    args = parser.parse_args(argv)

    cases = adapt_cases(_load(args.cases))
    holdout = adapt_holdout(_load(args.holdout), scorer=args.scorer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cases.yaml").write_text(
        _dump(
            cases,
            "cases",
            "# Generated by scripts/adapt_authored_cases.py from hand-authored input.\n"
            "# Labels are copied verbatim - the adapter renames fields and nothing else.\n"
            "# Edit this file directly from here on; then: make dataset-build",
        ),
        encoding="utf-8",
    )
    (args.out_dir / "holdout.yaml").write_text(
        _dump(
            holdout,
            "samples",
            "# Generated by scripts/adapt_authored_cases.py from hand-authored input.\n"
            "# Each sample still needs either a case_id or a reference_summary before\n"
            "# calibration can run - see AUTHORING.md.",
        ),
        encoding="utf-8",
    )

    missing_ref = [
        r["id"] for r in holdout if not r.get("case_id") and not r.get("reference_summary")
    ]
    print(f"cases:   {len(cases)} -> {args.out_dir / 'cases.yaml'}")
    print(f"holdout: {len(holdout)} -> {args.out_dir / 'holdout.yaml'}")
    if missing_ref:
        print(
            f"\n{len(missing_ref)} holdout sample(s) still need a case_id or a "
            f"reference_summary:\n  " + ", ".join(missing_ref)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
