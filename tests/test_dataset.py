from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mrd.dataset import hashing, report
from mrd.dataset.loader import DatasetValidationError, load_cases, load_holdout
from mrd.dataset.schema import GoldenCase, HoldoutSample

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 16, tzinfo=UTC)


def case(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "tc_0001",
        "input_email": "You billed me twice for August.",
        "expected_category": "billing",
        "expected_summary": "Customer was billed twice in August.",
        "difficulty": "easy",
        "critical": False,
        "source": "handwritten",
        "notes": "Baseline duplicate-charge case.",
        "added_at": NOW.isoformat(),
    }
    payload.update(overrides)
    return payload


def write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_valid_case_loads() -> None:
    assert GoldenCase.model_validate(case()).id == "tc_0001"


@pytest.mark.parametrize("bad_id", ["tc_1", "TC_0001", "case_0001", "tc_00001"])
def test_case_id_format_enforced(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(case(id=bad_id))


def test_notes_are_required() -> None:
    """A case whose author cannot say why it exists is uninterpretable when it fails."""
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(case(notes=""))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(case(expected_confidence=0.9))


def test_case_is_frozen() -> None:
    parsed = GoldenCase.model_validate(case())
    with pytest.raises(ValidationError):
        parsed.expected_category = "technical"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_loads_and_sorts_by_id(tmp_path: Path) -> None:
    path = write(
        tmp_path / "e.jsonl",
        [case(id="tc_0002", input_email="b"), case(id="tc_0001", input_email="a")],
    )
    dataset = load_cases(path)
    assert [c.id for c in dataset] == ["tc_0001", "tc_0002"]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_text(json.dumps(case()) + "\n\n\n", encoding="utf-8")
    assert len(load_cases(path)) == 1


def test_all_errors_reported_at_once_with_line_numbers(tmp_path: Path) -> None:
    """A hundred-case set should surface every problem in one run, not one per run."""
    path = tmp_path / "e.jsonl"
    path.write_text(
        "\n".join(
            [
                "{not json",
                json.dumps(case(id="tc_0002", notes="")),
                json.dumps(case(id="bad_id", input_email="x")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError) as exc:
        load_cases(path)

    assert [e.line for e in exc.value.errors] == [1, 2, 3]
    assert "invalid JSON" in exc.value.errors[0].message
    assert "notes" in exc.value.errors[1].message


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "e.jsonl", [case(input_email="a"), case(input_email="b")])
    with pytest.raises(DatasetValidationError, match="duplicate id"):
        load_cases(path)


def test_duplicate_emails_rejected(tmp_path: Path) -> None:
    """Near-identical inputs inflate sample size and skew the significance test."""
    path = write(
        tmp_path / "e.jsonl",
        [case(id="tc_0001"), case(id="tc_0002", input_email="  YOU BILLED ME twice for August. ")],
    )
    with pytest.raises(DatasetValidationError, match="duplicate email"):
        load_cases(path)


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope.jsonl")


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


def test_few_shot_leakage_rejected(tmp_path: Path, repo_root: Path, prompt_v001) -> None:
    """A case the model was shown in its prompt measures recall, not capability."""
    leaked = prompt_v001.few_shot[0].email
    path = write(tmp_path / "e.jsonl", [case(input_email=leaked)])

    with pytest.raises(DatasetValidationError, match="few-shot example in prompt v001"):
        load_cases(path, prompts_root=repo_root / "prompts")


def test_leakage_check_ignores_whitespace_and_case(
    tmp_path: Path, repo_root: Path, prompt_v001
) -> None:
    mangled = "   " + prompt_v001.few_shot[1].email.upper().replace("\n", "  ") + "  "
    path = write(tmp_path / "e.jsonl", [case(input_email=mangled)])

    with pytest.raises(DatasetValidationError, match="few-shot example"):
        load_cases(path, prompts_root=repo_root / "prompts")


def test_clean_case_passes_leakage_check(tmp_path: Path, repo_root: Path) -> None:
    path = write(tmp_path / "e.jsonl", [case()])
    assert len(load_cases(path, prompts_root=repo_root / "prompts")) == 1


# --------------------------------------------------------------------------- #
# Hashing and lock
# --------------------------------------------------------------------------- #


def test_hash_is_order_independent(tmp_path: Path) -> None:
    rows = [case(id="tc_0001", input_email="a"), case(id="tc_0002", input_email="b")]
    forward = load_cases(write(tmp_path / "f.jsonl", rows))
    backward = load_cases(write(tmp_path / "b.jsonl", list(reversed(rows))))
    assert hashing.content_hash(forward) == hashing.content_hash(backward)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_category", "technical"),
        ("expected_summary", "Something else entirely."),
        ("difficulty", "adversarial"),
        ("critical", True),
        ("input_email", "different text"),
    ],
)
def test_any_label_change_changes_the_hash(tmp_path: Path, field: str, value: object) -> None:
    """Ground truth drift must invalidate the baseline, not pass silently."""
    before = load_cases(write(tmp_path / "a.jsonl", [case()]))
    after = load_cases(write(tmp_path / "b.jsonl", [case(**{field: value})]))
    assert hashing.content_hash(before) != hashing.content_hash(after)


def test_lock_round_trips(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case(critical=True)]))
    lock = hashing.build_lock(dataset, version="v1", now=NOW)

    restored = hashing.Lock.from_json(lock.to_json())
    assert restored == lock
    assert restored.count == 1
    assert restored.critical_count == 1
    assert restored.by_category["billing"] == 1


def test_verify_passes_on_unchanged_dataset(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case()]))
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(hashing.build_lock(dataset, version="v1", now=NOW).to_json())

    assert hashing.verify(dataset, lock_path).version == "v1"


def test_verify_fails_on_edited_dataset(tmp_path: Path) -> None:
    original = load_cases(write(tmp_path / "e.jsonl", [case()]))
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(hashing.build_lock(original, version="v1", now=NOW).to_json())

    edited = load_cases(write(tmp_path / "e2.jsonl", [case(expected_category="general")]))
    with pytest.raises(hashing.LockMismatch, match="not comparable"):
        hashing.verify(edited, lock_path)


def test_verify_fails_without_a_lock(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case()]))
    with pytest.raises(hashing.LockMismatch, match="make dataset-lock"):
        hashing.verify(dataset, tmp_path / "absent.json")


# --------------------------------------------------------------------------- #
# Holdout
# --------------------------------------------------------------------------- #


def holdout(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "ho_0001",
        "case_id": "tc_0001",
        "summary": "Customer was billed twice in August.",
        "human_score": 5,
        "scorer": "sreekar",
        "scored_at": NOW.isoformat(),
        "notes": "",
    }
    payload.update(overrides)
    return payload


def test_holdout_loads(tmp_path: Path) -> None:
    samples = load_holdout(write(tmp_path / "h.jsonl", [holdout()]))
    assert samples[0].human_score == 5


@pytest.mark.parametrize("score", [0, 6, -1])
def test_holdout_score_range_enforced(score: int) -> None:
    with pytest.raises(ValidationError):
        HoldoutSample.model_validate(holdout(human_score=score))


def test_holdout_case_id_must_exist(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case(id="tc_0001")]))
    path = write(tmp_path / "h.jsonl", [holdout(case_id="tc_9999")])

    with pytest.raises(DatasetValidationError, match="not in the dataset"):
        load_holdout(path, dataset=dataset)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_sparse_dataset_is_not_ready(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case()]))
    result = report.build(dataset)

    assert not result.ready
    assert any("1/80 cases written" in w for w in result.warnings)
    assert any("ambiguous" in w for w in result.warnings)


def test_flat_holdout_is_flagged(tmp_path: Path) -> None:
    """Kappa needs disagreement to measure; an all-5s holdout never tests the judge."""
    dataset = load_cases(write(tmp_path / "e.jsonl", [case()]))
    samples = tuple(
        HoldoutSample.model_validate(holdout(id=f"ho_{i:04d}", human_score=5)) for i in range(20)
    )
    result = report.build(dataset, samples)

    assert any("distinct score" in w for w in result.warnings)


def test_spread_holdout_is_not_flagged_for_variance(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case()]))
    samples = tuple(
        HoldoutSample.model_validate(holdout(id=f"ho_{i:04d}", human_score=(i % 5) + 1))
        for i in range(20)
    )
    result = report.build(dataset, samples)

    assert not any("distinct score" in w for w in result.warnings)
    assert not any("holdout:" in w for w in result.warnings)


def test_render_includes_counts_and_gaps(tmp_path: Path) -> None:
    dataset = load_cases(write(tmp_path / "e.jsonl", [case()]))
    text = report.render(report.build(dataset))

    assert "Golden dataset" in text
    assert "NOT READY" in text
    assert "by category" in text


# --------------------------------------------------------------------------- #
# Holdout error paths
# --------------------------------------------------------------------------- #


def test_holdout_reports_all_errors_with_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text(
        "\n".join(
            [
                "{not json",
                json.dumps(holdout(id="ho_0002", human_score=9)),
                json.dumps(holdout(id="ho_0003")),
                json.dumps(holdout(id="ho_0003", summary="dupe")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError) as exc:
        load_holdout(path)

    lines = [e.line for e in exc.value.errors]
    assert lines == [1, 2, 4]
    assert "invalid JSON" in exc.value.errors[0].message
    assert "human_score" in exc.value.errors[1].message
    assert "duplicate id" in exc.value.errors[2].message


def test_holdout_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(holdout()) + "\n\n", encoding="utf-8")
    assert len(load_holdout(path)) == 1


# --------------------------------------------------------------------------- #
# Report edge states
# --------------------------------------------------------------------------- #


def _full_dataset(tmp_path: Path) -> Path:
    """A dataset that clears every coverage target."""
    rows: list[dict[str, object]] = []
    categories = ["billing", "technical", "account", "general"]
    for i in range(80):
        category = categories[i % 4]
        if i < 12:
            difficulty = "adversarial"
        elif i < 28:
            difficulty = "ambiguous"
        else:
            difficulty = "easy"
        rows.append(
            case(
                id=f"tc_{i:04d}",
                input_email=f"Distinct email body number {i}.",
                expected_category=category,
                difficulty=difficulty,
                critical=i < 10,
            )
        )
    return write(tmp_path / "full.jsonl", rows)


def test_complete_dataset_reports_ready(tmp_path: Path) -> None:
    dataset = load_cases(_full_dataset(tmp_path))
    samples = tuple(
        HoldoutSample.model_validate(holdout(id=f"ho_{i:04d}", human_score=(i % 5) + 1))
        for i in range(20)
    )
    result = report.build(dataset, samples)

    assert result.ready, result.warnings
    assert "READY - all coverage targets met." in report.render(result)


def test_narrow_holdout_spread_is_flagged(tmp_path: Path) -> None:
    """Three distinct scores clustered together still calibrate weakly."""
    dataset = load_cases(_full_dataset(tmp_path))
    scores = [3] * 18 + [2, 4]
    samples = tuple(
        HoldoutSample.model_validate(holdout(id=f"ho_{i:04d}", human_score=s))
        for i, s in enumerate(scores)
    )
    result = report.build(dataset, samples)

    assert any("spread is narrow" in w for w in result.warnings)
