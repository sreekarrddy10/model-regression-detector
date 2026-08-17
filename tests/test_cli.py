from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mrd import sampling
from mrd.cli import main
from mrd.dataset.loader import Dataset, load_cases
from mrd.report import markdown

from .engine_fixtures import ScriptedProvider, make_case
from .test_dataset import case, write
from .test_report import ALL_PASS
from .test_report import build as build_report

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Tier sampling
# --------------------------------------------------------------------------- #


def _dataset(count: int, *, critical: int = 0) -> Dataset:
    return Dataset(
        path=Path("memory"),
        cases=tuple(
            make_case(
                i,
                critical=i < critical,
                difficulty=("easy", "ambiguous", "adversarial")[i % 3],
            )
            for i in range(count)
        ),
    )


def test_smoke_selection_is_deterministic() -> None:
    """Random sampling would report sampling churn as model change."""
    dataset = _dataset(60)
    first = sampling.stratified(dataset, 20)
    second = sampling.stratified(dataset, 20)

    assert [c.id for c in first] == [c.id for c in second]
    assert len(first) == 20


def test_smoke_selection_spans_categories_and_difficulties() -> None:
    """A sample of only easy billing cases cannot detect an account regression."""
    selected = sampling.stratified(_dataset(60), 20)

    assert len({c.expected_category for c in selected}) == 4
    assert len({c.difficulty for c in selected}) == 3


def test_every_critical_case_is_always_included() -> None:
    """The smoke tier exists to catch what matters most."""
    dataset = _dataset(60, critical=12)
    selected = sampling.stratified(dataset, 20)

    assert {c.id for c in dataset.critical} <= {c.id for c in selected}


def test_critical_cases_are_not_dropped_to_hit_the_size() -> None:
    dataset = _dataset(60, critical=25)
    selected = sampling.stratified(dataset, 20)

    assert len(selected) == 25, "size is a floor for critical cases, not a cap"


def test_smoke_selection_handles_a_small_dataset() -> None:
    assert len(sampling.stratified(_dataset(5), 20)) == 5


def test_zero_size_selects_nothing() -> None:
    assert sampling.stratified(_dataset(10), 0) == ()


@pytest.mark.parametrize("tier", ["full", "unit"])
def test_non_smoke_tiers_use_every_case(tier: str) -> None:
    dataset = _dataset(30)
    assert len(sampling.select(dataset, tier)) == 30


# --------------------------------------------------------------------------- #
# PR comment
# --------------------------------------------------------------------------- #


def test_comment_carries_the_sticky_marker() -> None:
    """CI finds and updates its own comment instead of stacking new ones."""
    body = markdown.render(build_report(ALL_PASS, baseline_flags=ALL_PASS))
    assert body.startswith(markdown.MARKER)


def test_comment_leads_with_the_verdict() -> None:
    broken = {**ALL_PASS}
    for i in range(3):
        broken[f"tc_{i:04d}"] = (False,) * 3
    body = markdown.render(build_report(broken, baseline_flags=ALL_PASS))

    assert "⛔ **BLOCK**" in body
    assert "### Blocking" in body
    assert "critical case(s) regressed" in body


def test_comment_renders_a_scorecard_and_the_statistics() -> None:
    body = markdown.render(
        build_report({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS)
    )

    assert "| Metric | Baseline | This run | Δ |" in body
    assert "McNemar exact" in body
    assert "discordant pair" in body
    assert "| `tc_0003` |" in body


def test_comment_marks_direction_of_change() -> None:
    body = markdown.render(
        build_report({**ALL_PASS, "tc_0003": (False,) * 3}, baseline_flags=ALL_PASS)
    )
    assert "🔻" in body


def test_comment_flags_an_uncalibrated_judge() -> None:
    body = markdown.render(build_report(ALL_PASS, baseline_flags=ALL_PASS))
    assert "not run" in body and "uninterpreted" in body


def test_comment_handles_a_first_run() -> None:
    body = markdown.render(build_report(ALL_PASS))
    assert "No baseline to compare against yet" in body
    assert "| Accuracy | — |" in body


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path: Path, repo_root: Path) -> list[str]:
    """A CLI invocation wired entirely at temporary paths."""
    cases = [
        case(
            id=f"tc_{i:04d}",
            input_email=f"Distinct email body number {i}.",
            expected_category="billing",
            critical=(i == 0),
        )
        for i in range(4)
    ]
    write(tmp_path / "emails.jsonl", cases)
    (tmp_path / "holdout.jsonl").write_text("", encoding="utf-8")
    return [
        "eval",
        "--tier",
        "unit",
        "--no-judge",
        "--no-slack",
        "--repeats",
        "1",
        "--cases",
        str(tmp_path / "emails.jsonl"),
        "--holdout",
        str(tmp_path / "holdout.jsonl"),
        "--lock",
        str(tmp_path / "lock.json"),
        "--prompts",
        str(repo_root / "prompts"),
        "--cassettes",
        str(tmp_path / "cassettes"),
        "--db",
        str(tmp_path / "runs.sqlite"),
        "--report",
        str(tmp_path / "report.html"),
        "--comment",
        str(tmp_path / "comment.md"),
    ]


def test_empty_dataset_says_what_to_do(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "emails.jsonl").write_text("", encoding="utf-8")

    assert main(workspace) == 1
    assert "make dataset-new" in capsys.readouterr().err


def test_missing_lock_is_a_clear_error(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(workspace) == 1
    assert "make dataset-lock" in capsys.readouterr().err


def test_drifted_dataset_refuses_to_run(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Ground truth changed since the lock: nothing downstream is comparable."""
    from mrd.dataset import hashing

    dataset = load_cases(tmp_path / "emails.jsonl")
    (tmp_path / "lock.json").write_text(
        hashing.build_lock(dataset, version="v1", now=datetime.now(UTC)).to_json()
    )
    write(tmp_path / "emails.jsonl", [case(id="tc_0000", expected_category="general")])

    assert main(workspace) == 1
    assert "not comparable" in capsys.readouterr().err


def test_invalid_dataset_reports_line_numbers(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path / "emails.jsonl", [case(notes="")])
    assert main(workspace) == 1
    assert "line 1" in capsys.readouterr().err


def test_exit_code_is_the_gate_verdict() -> None:
    """The whole contract with CI."""
    from mrd.compare import Verdict

    clean = build_report(ALL_PASS, baseline_flags=ALL_PASS)
    assert clean.gate.exit_code == 0
    assert clean.gate.verdict is Verdict.PASS

    broken = {**ALL_PASS}
    for i in range(3):
        broken[f"tc_{i:04d}"] = (False,) * 3
    assert build_report(broken, baseline_flags=ALL_PASS).gate.exit_code == 1


def test_unknown_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["nonsense"])


# --------------------------------------------------------------------------- #
# End to end, offline
# --------------------------------------------------------------------------- #


@pytest.fixture
def locked(tmp_path: Path, workspace: list[str]) -> list[str]:
    """The same workspace, with ground truth frozen so the gate can run."""
    from mrd.dataset import hashing

    dataset = load_cases(tmp_path / "emails.jsonl")
    (tmp_path / "lock.json").write_text(
        hashing.build_lock(dataset, version="v1", now=datetime.now(UTC)).to_json()
    )
    return workspace


def _scripted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, behaviours: dict[str, object]
) -> None:
    """Serve the CLI a scripted model instead of a provider."""
    dataset = load_cases(tmp_path / "emails.jsonl")
    provider = ScriptedProvider(dataset.cases, behaviours)
    monkeypatch.setattr("mrd.cli.build_provider", lambda *a, **k: provider)


def test_first_run_records_a_baseline_and_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    locked: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _scripted(monkeypatch, tmp_path, {})

    assert main(locked) == 0
    out = capsys.readouterr().out

    assert "no comparable baseline yet" in out
    assert "PASS" in out
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "comment.md").exists()
    assert (tmp_path / "runs.sqlite").exists()


def test_second_run_blocks_on_a_critical_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    locked: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole point of the system, exercised through the CLI."""
    _scripted(monkeypatch, tmp_path, {})
    assert main([*locked, "--run-id", "baseline"]) == 0
    capsys.readouterr()

    _scripted(monkeypatch, tmp_path, {"tc_0000": "wrong"})
    assert main([*locked, "--run-id", "candidate"]) == 1

    out = capsys.readouterr().out
    assert "BLOCK" in out
    assert "critical case(s) regressed: tc_0000" in out

    comment = (tmp_path / "comment.md").read_text(encoding="utf-8")
    assert comment.startswith(markdown.MARKER)
    assert "⛔ **BLOCK**" in comment
    assert "| `tc_0000` ⚠️ critical |" in comment

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "tc_0000" in report
    assert "Baseline — pass" in report
    assert "This run — fail" in report


def test_run_history_accumulates_for_the_trend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked: list[str]
) -> None:
    from mrd.store import sqlite

    _scripted(monkeypatch, tmp_path, {})
    for idx in range(3):
        main([*locked, "--run-id", f"run-{idx}"])

    with sqlite.connect(tmp_path / "runs.sqlite") as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        assert rows["n"] == 3


def test_smoke_tier_narrows_the_case_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    locked: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _scripted(monkeypatch, tmp_path, {})
    args = [a if a != "unit" else "smoke" for a in locked]

    assert main([*args, "--smoke-size", "2"]) == 0
    assert "smoke tier" in capsys.readouterr().out
