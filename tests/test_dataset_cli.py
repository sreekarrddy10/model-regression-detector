from __future__ import annotations

import json
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


def test_new_emits_a_yaml_stanza_not_a_jsonl_row(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """emails.jsonl is generated. A row appended there is erased by the next build."""
    assert main(["new", *workspace]) == 0
    out = capsys.readouterr().out

    assert out.lstrip().startswith("#") or "  - input_email: |" in out
    assert "  - input_email: |" in out
    for field in ("expected_category", "expected_summary", "difficulty", "notes"):
        assert field in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_a_new_stanza_appends_to_cases_yaml_and_builds(
    tmp_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round trip the old `dataset-new` silently broke."""
    cases_yaml = tmp_path / "cases.yaml"
    cases_yaml.write_text("cases:\n", encoding="utf-8")
    (tmp_path / "holdout.yaml").write_text("samples:\n", encoding="utf-8")

    args = [
        "--cases-yaml",
        str(cases_yaml),
        "--holdout-yaml",
        str(tmp_path / "holdout.yaml"),
        "--cases",
        str(tmp_path / "emails.jsonl"),
        "--holdout",
        str(tmp_path / "holdout.jsonl"),
        "--lock",
        str(tmp_path / "lock.json"),
        "--prompts",
        str(repo_root / "prompts"),
    ]
    assert main(["new", *args]) == 0
    cases_yaml.write_text(
        cases_yaml.read_text(encoding="utf-8") + capsys.readouterr().out, encoding="utf-8"
    )

    # Parses as YAML, and build says exactly which fields still need filling.
    assert main(["build", *args]) == 1
    assert "missing or empty: input_email" in capsys.readouterr().err


def test_explicit_id_is_pinned_in_the_stanza(
    workspace: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new", *workspace, "--id", "tc_0099"]) == 0
    assert "id: tc_0099" in capsys.readouterr().out


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
