from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mrd.dataset.__main__ import main

from .test_dataset import case, write

pytestmark = pytest.mark.unit


@pytest.fixture
def workspace(tmp_path: Path, repo_root: Path) -> list[str]:
    write(tmp_path / "emails.jsonl", [case()])
    (tmp_path / "holdout.jsonl").write_text("", encoding="utf-8")
    return [
        "--cases",
        str(tmp_path / "emails.jsonl"),
        "--holdout",
        str(tmp_path / "holdout.jsonl"),
        "--lock",
        str(tmp_path / "lock.json"),
        "--prompts",
        str(repo_root / "prompts"),
    ]


def test_validate_succeeds(workspace: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*workspace, "validate"]) == 0
    assert "OK: 1 case(s)" in capsys.readouterr().out


def test_validate_fails_loudly_on_bad_data(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path / "emails.jsonl", [case(notes="")])
    assert main([*workspace, "validate"]) == 1
    assert "notes" in capsys.readouterr().err


def test_report_exits_zero_while_incomplete(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Gaps are progress information; the report must stay usable daily."""
    assert main([*workspace, "report"]) == 0
    assert "NOT READY" in capsys.readouterr().out


def test_lock_then_verify(workspace: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*workspace, "lock", "--version", "v1"]) == 0
    assert "Locked 1 case(s) as v1" in capsys.readouterr().out

    assert main([*workspace, "verify"]) == 0
    assert "matches lock v1" in capsys.readouterr().out


def test_verify_fails_after_an_edit(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    main([*workspace, "lock", "--version", "v1"])
    write(tmp_path / "emails.jsonl", [case(expected_category="general")])

    assert main([*workspace, "verify"]) == 1
    assert "not comparable" in capsys.readouterr().err


def test_new_emits_a_fillable_row(workspace: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*workspace, "new", "--id", "tc_0042"]) == 0
    row = json.loads(capsys.readouterr().out)

    assert row["id"] == "tc_0042"
    assert row["input_email"] == ""
    assert row["notes"] == ""
    datetime.fromisoformat(row["added_at"]).astimezone(UTC)


def test_missing_cases_file_exits_one(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "emails.jsonl").unlink()
    assert main([*workspace, "validate"]) == 1
    assert "error:" in capsys.readouterr().err
