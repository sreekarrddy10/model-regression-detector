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
    assert main(["validate", *workspace]) == 0
    assert "OK: 1 case(s)" in capsys.readouterr().out


def test_validate_fails_loudly_on_bad_data(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path / "emails.jsonl", [case(notes="")])
    assert main(["validate", *workspace]) == 1
    assert "notes" in capsys.readouterr().err


def test_report_exits_zero_while_incomplete(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Gaps are progress information; the report must stay usable daily."""
    assert main(["report", *workspace]) == 0
    assert "NOT READY" in capsys.readouterr().out


def test_lock_then_verify(workspace: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lock", *workspace, "--version", "v1"]) == 0
    assert "Locked 1 case(s) as v1" in capsys.readouterr().out

    assert main(["verify", *workspace]) == 0
    assert "matches lock v1" in capsys.readouterr().out


def test_verify_fails_after_an_edit(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["lock", *workspace, "--version", "v1"])
    write(tmp_path / "emails.jsonl", [case(expected_category="general")])

    assert main(["verify", *workspace]) == 1
    assert "not comparable" in capsys.readouterr().err


def test_new_emits_a_fillable_row(workspace: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["new", *workspace, "--id", "tc_0042"]) == 0
    row = json.loads(capsys.readouterr().out)

    assert row["id"] == "tc_0042"
    assert row["input_email"] == ""
    assert row["notes"] == ""
    datetime.fromisoformat(row["added_at"]).astimezone(UTC)


def test_missing_cases_file_exits_one(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "emails.jsonl").unlink()
    assert main(["validate", *workspace]) == 1
    assert "error:" in capsys.readouterr().err


def test_options_may_follow_the_subcommand(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The order every other CLI accepts them in."""
    assert main(["report", *workspace]) == 0
    assert "Golden dataset" in capsys.readouterr().out


def test_new_picks_the_next_free_id(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Eighty cases should not also mean tracking eighty ids by hand."""
    write(tmp_path / "emails.jsonl", [case(id="tc_0001"), case(id="tc_0002", input_email="other")])

    assert main(["new", *workspace]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "tc_0003"


def test_next_id_fills_gaps(
    tmp_path: Path, workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path / "emails.jsonl", [case(id="tc_0001"), case(id="tc_0003", input_email="other")])

    assert main(["new", *workspace]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "tc_0002"


def test_explicit_id_still_wins(workspace: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["new", *workspace, "--id", "tc_0099"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "tc_0099"


def test_report_suggests_the_scarcest_strata_first(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """An all-easy set produces a confident pass rate that detects nothing."""
    assert main(["report", *workspace]) == 0
    out = capsys.readouterr().out

    lines = out.splitlines()
    assert "Write next:" in lines
    after = lines[lines.index("Write next:") + 1 :]
    suggestions = [line for line in after if line.startswith("  - ")]

    assert suggestions, "the report should say which stratum to write next"
    assert "adversarial" in suggestions[0], "the scarcest, most informative stratum leads"
