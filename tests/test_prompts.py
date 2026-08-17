from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mrd.prompts import PromptConfig

pytestmark = pytest.mark.unit


def test_loads_v001(prompt_v001: PromptConfig) -> None:
    assert prompt_v001.version_id == "v001"
    assert prompt_v001.temperature == 0.0
    assert len(prompt_v001.few_shot) == 3
    assert prompt_v001.commit_message


def test_render_system_includes_few_shot(prompt_v001: PromptConfig) -> None:
    rendered = prompt_v001.render_system()
    assert prompt_v001.system_prompt in rendered
    assert "Examples:" in rendered
    for example in prompt_v001.few_shot:
        assert example.email in rendered
        assert f'"category": "{example.category}"' in rendered


def test_prompt_is_frozen(prompt_v001: PromptConfig) -> None:
    with pytest.raises(ValidationError):
        prompt_v001.system_prompt = "mutated"  # type: ignore[misc]


def test_nonzero_temperature_rejected() -> None:
    """Sampling makes flip detection meaningless, so it is refused at load time."""
    with pytest.raises(ValidationError, match="temperature=0.0"):
        PromptConfig(
            version_id="v002",
            created_at="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
            model="gpt-4o-mini",
            system_prompt="x",
            commit_message="x",
            temperature=0.7,
        )


def test_bad_version_id_rejected() -> None:
    with pytest.raises(ValidationError, match="vNNN"):
        PromptConfig(
            version_id="1",
            created_at="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
            model="gpt-4o-mini",
            system_prompt="x",
            commit_message="x",
        )


def test_version_id_must_match_filename(tmp_path: Path, prompt_v001: PromptConfig) -> None:
    """Run provenance is unreliable if the file and its declared version disagree."""
    payload = prompt_v001.model_dump(mode="json")
    target = tmp_path / "v009.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match filename"):
        PromptConfig.from_file(target)


def test_latest_returns_highest_version(repo_root: Path) -> None:
    latest = PromptConfig.latest(root=repo_root / "prompts")
    assert latest.version_id == "v001"


def test_prompt_root_prefers_an_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mrd import prompts

    monkeypatch.setenv(prompts.ENV_VAR, str(tmp_path / "elsewhere"))
    assert prompts.default_root() == tmp_path / "elsewhere"


def test_prompt_root_falls_back_to_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """How the container finds them: WORKDIR holds prompts/, not site-packages."""
    from mrd import prompts

    monkeypatch.delenv(prompts.ENV_VAR, raising=False)
    (tmp_path / "prompts").mkdir()
    monkeypatch.chdir(tmp_path)

    assert prompts.default_root() == tmp_path / "prompts"


def test_prompt_root_falls_back_to_the_source_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_root: Path
) -> None:
    from mrd import prompts

    monkeypatch.delenv(prompts.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    assert prompts.default_root() == repo_root / "prompts"
