from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from mrd.dataset import authoring, hashing
from mrd.dataset.loader import DatasetValidationError, load_cases

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 19, tzinfo=UTC)
LATER = NOW + timedelta(days=3)

CASE = {
    "input_email": "You billed me twice for August.",
    "expected_category": "billing",
    "expected_summary": "Customer was billed twice in August.",
    "difficulty": "easy",
    "notes": "Baseline duplicate-charge case.",
}


def write_yaml(path: Path, cases: list[dict[str, object]], key: str = "cases") -> Path:
    path.write_text(yaml.safe_dump({key: cases}, sort_keys=False), encoding="utf-8")
    return path


def build(tmp_path: Path, cases: list[dict[str, object]], *, now: datetime = NOW):  # type: ignore[no-untyped-def]
    source = write_yaml(tmp_path / "cases.yaml", cases)
    return authoring.build_cases(source, tmp_path / "emails.jsonl", now=now)


# --------------------------------------------------------------------------- #
# Readability: the whole point of the YAML surface
# --------------------------------------------------------------------------- #


def test_block_scalars_preserve_real_line_breaks(tmp_path: Path) -> None:
    """The reason this surface exists: no hand-escaping of \\n in JSONL."""
    source = tmp_path / "cases.yaml"
    source.write_text(
        """
cases:
  - input_email: |
      First line.

      Third line.
    expected_category: billing
    expected_summary: Customer wrote a multi-line email.
    difficulty: easy
    notes: Multi-line authoring check.
""",
        encoding="utf-8",
    )
    result = authoring.build_cases(source, tmp_path / "emails.jsonl", now=NOW)

    assert result.cases[0].input_email.count("\n") == 3
    assert "First line." in result.cases[0].input_email


def test_compiled_jsonl_round_trips_through_the_real_loader(tmp_path: Path) -> None:
    """`build` must never emit something `validate` would reject."""
    result = build(tmp_path, [dict(CASE), {**CASE, "input_email": "Different email."}])
    path = authoring.write_jsonl(result.cases, tmp_path / "emails.jsonl")

    assert len(load_cases(path)) == 2


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def test_ids_are_assigned_in_order(tmp_path: Path) -> None:
    result = build(
        tmp_path, [dict(CASE), {**CASE, "input_email": "b"}, {**CASE, "input_email": "c"}]
    )
    assert [c.id for c in result.cases] == ["tc_0001", "tc_0002", "tc_0003"]


def test_explicit_ids_are_honoured_and_not_reused(tmp_path: Path) -> None:
    result = build(
        tmp_path,
        [
            {**CASE, "id": "tc_0050", "input_email": "a"},
            {**CASE, "input_email": "b"},
        ],
    )
    assert sorted(c.id for c in result.cases) == ["tc_0001", "tc_0050"]


def test_defaults_are_applied(tmp_path: Path) -> None:
    case = build(tmp_path, [dict(CASE)]).cases[0]
    assert case.critical is False
    assert case.strata == ()
    assert case.source == "handwritten"


def test_critical_and_source_can_be_set(tmp_path: Path) -> None:
    case = build(tmp_path, [{**CASE, "strata": ["critical"], "source": "from_failure"}]).cases[0]
    assert case.critical is True
    assert case.strata == ("critical",)
    assert case.source == "from_failure"


def test_a_bare_critical_bool_is_rejected(tmp_path: Path) -> None:
    """`critical` is a stratum tag now; the old bool would silently do nothing."""
    with pytest.raises(DatasetValidationError, match="critical"):
        build(tmp_path, [{**CASE, "critical": True}])


def test_duplicate_strata_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="duplicate strata"):
        build(tmp_path, [{**CASE, "strata": ["critical", "critical"]}])


# --------------------------------------------------------------------------- #
# Idempotence: the property the dataset lock depends on
# --------------------------------------------------------------------------- #


def test_rebuilding_does_not_change_the_content_hash(tmp_path: Path) -> None:
    """If `added_at` were re-stamped each build, a no-op rebuild would break the lock."""
    source = write_yaml(tmp_path / "cases.yaml", [dict(CASE)])
    jsonl = tmp_path / "emails.jsonl"

    first = authoring.build_cases(source, jsonl, now=NOW)
    authoring.write_jsonl(first.cases, jsonl)
    before = hashing.content_hash(load_cases(jsonl))

    second = authoring.build_cases(source, jsonl, now=LATER)
    authoring.write_jsonl(second.cases, jsonl)
    after = hashing.content_hash(load_cases(jsonl))

    assert before == after
    assert second.added == ()
    assert second.carried == ("tc_0001",)


def test_a_locked_dataset_still_verifies_after_a_rebuild(tmp_path: Path) -> None:
    source = write_yaml(tmp_path / "cases.yaml", [dict(CASE)])
    jsonl = tmp_path / "emails.jsonl"
    lock = tmp_path / "lock.json"

    authoring.write_jsonl(authoring.build_cases(source, jsonl, now=NOW).cases, jsonl)
    lock.write_text(hashing.build_lock(load_cases(jsonl), version="v1", now=NOW).to_json())

    authoring.write_jsonl(authoring.build_cases(source, jsonl, now=LATER).cases, jsonl)
    assert hashing.verify(load_cases(jsonl), lock).version == "v1"


def test_a_new_case_is_stamped_while_the_others_are_carried(tmp_path: Path) -> None:
    source = write_yaml(tmp_path / "cases.yaml", [dict(CASE)])
    jsonl = tmp_path / "emails.jsonl"
    authoring.write_jsonl(authoring.build_cases(source, jsonl, now=NOW).cases, jsonl)

    write_yaml(tmp_path / "cases.yaml", [dict(CASE), {**CASE, "input_email": "second"}])
    result = authoring.build_cases(tmp_path / "cases.yaml", jsonl, now=LATER)

    assert result.carried == ("tc_0001",)
    assert result.added == ("tc_0002",)
    stamps = {c.id: c.added_at for c in result.cases}
    assert stamps["tc_0001"] < stamps["tc_0002"]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field", ["input_email", "expected_category", "expected_summary", "difficulty", "notes"]
)
def test_missing_required_fields_are_reported_by_position(tmp_path: Path, field: str) -> None:
    payload = {k: v for k, v in CASE.items() if k != field}
    with pytest.raises(DatasetValidationError, match=field):
        build(tmp_path, [dict(CASE), payload])


def test_empty_notes_is_rejected_like_a_missing_one(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="notes"):
        build(tmp_path, [{**CASE, "notes": "   "}])


def test_bad_category_is_reported_with_the_case_id(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="tc_0001"):
        build(tmp_path, [{**CASE, "expected_category": "refunds"}])


def test_all_errors_are_reported_at_once(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError) as exc:
        build(
            tmp_path,
            [
                {k: v for k, v in CASE.items() if k != "notes"},
                {**CASE, "difficulty": "trivial", "input_email": "b"},
                {**CASE, "expected_category": "nope", "input_email": "c"},
            ],
        )
    assert len(exc.value.errors) >= 3


def test_a_missing_authoring_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="authoring file not found"):
        authoring.build_cases(tmp_path / "absent.yaml", tmp_path / "e.jsonl", now=NOW)


def test_a_yaml_file_without_the_cases_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "cases.yaml").write_text("something_else: []", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="top-level 'cases' list"):
        authoring.build_cases(tmp_path / "cases.yaml", tmp_path / "e.jsonl", now=NOW)


def test_an_empty_cases_list_builds_cleanly(tmp_path: Path) -> None:
    assert build(tmp_path, []).cases == ()


# --------------------------------------------------------------------------- #
# Holdout
# --------------------------------------------------------------------------- #


def test_holdout_ids_and_timestamps_are_assigned(tmp_path: Path) -> None:
    source = write_yaml(
        tmp_path / "holdout.yaml",
        [
            {"case_id": "tc_0001", "candidate_summary": "a", "human_score": 5, "scorer": "me"},
            {"case_id": "tc_0001", "candidate_summary": "b", "human_score": 2, "scorer": "me"},
        ],
        key="samples",
    )
    samples = authoring.build_holdout(source, tmp_path / "h.jsonl", now=NOW)

    assert [s.id for s in samples] == ["ho_0001", "ho_0002"]
    assert [s.human_score for s in samples] == [5, 2]


def test_holdout_rebuild_is_idempotent(tmp_path: Path) -> None:
    source = write_yaml(
        tmp_path / "holdout.yaml",
        [{"case_id": "tc_0001", "candidate_summary": "a", "human_score": 4, "scorer": "me"}],
        key="samples",
    )
    jsonl = tmp_path / "h.jsonl"
    authoring.write_jsonl(authoring.build_holdout(source, jsonl, now=NOW), jsonl)
    first = json.loads(jsonl.read_text().splitlines()[0])["scored_at"]

    authoring.write_jsonl(authoring.build_holdout(source, jsonl, now=LATER), jsonl)
    assert json.loads(jsonl.read_text().splitlines()[0])["scored_at"] == first


def test_out_of_range_holdout_score_is_reported(tmp_path: Path) -> None:
    source = write_yaml(
        tmp_path / "holdout.yaml",
        [{"case_id": "tc_0001", "candidate_summary": "a", "human_score": 9, "scorer": "me"}],
        key="samples",
    )
    with pytest.raises(DatasetValidationError, match="human_score"):
        authoring.build_holdout(source, tmp_path / "h.jsonl", now=NOW)
