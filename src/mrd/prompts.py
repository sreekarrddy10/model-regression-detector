"""Versioned prompt artifacts.

The prompt is the thing under test, so it lives outside the code as a versioned
YAML file. A run records `version_id`, and CI triggers on changes to prompts/**.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_VERSION_RE = re.compile(r"^v\d{3}$")
_PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"


class Example(BaseModel):
    """One few-shot example."""

    model_config = ConfigDict(frozen=True)

    email: str
    category: str
    summary: str


class PromptConfig(BaseModel):
    """A single immutable prompt version."""

    model_config = ConfigDict(frozen=True)

    version_id: str
    created_at: datetime
    model: str
    system_prompt: str
    commit_message: str
    temperature: float = 0.0
    max_tokens: int = Field(default=512, gt=0)
    few_shot: tuple[Example, ...] = ()

    @field_validator("version_id")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if not _VERSION_RE.match(value):
            raise ValueError(f"version_id must match vNNN, got {value!r}")
        return value

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        # Eval runs must be as deterministic as the provider allows; variance is
        # handled by repeats (N=3), not by sampling. See docs/SPEC.md 3.3.
        if value != 0.0:
            raise ValueError(
                f"eval prompts must pin temperature=0.0, got {value}. "
                "Non-zero sampling makes flip detection meaningless."
            )
        return value

    def render_system(self) -> str:
        """The full system message: instructions plus any few-shot block."""
        if not self.few_shot:
            return self.system_prompt
        blocks = [
            f"Email:\n{ex.email}\n\nExpected output:\n"
            f'{{"category": "{ex.category}", "summary": "{ex.summary}"}}'
            for ex in self.few_shot
        ]
        return self.system_prompt + "\n\nExamples:\n\n" + "\n\n---\n\n".join(blocks)

    @classmethod
    def from_file(cls, path: Path) -> PromptConfig:
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = cls.model_validate(data)
        if config.version_id != path.stem:
            raise ValueError(
                f"version_id {config.version_id!r} does not match filename {path.stem!r}; "
                "a prompt version and its file must agree or run provenance is unreliable"
            )
        return config

    @classmethod
    def load(
        cls, version_id: str, *, feature: str = "classifier", root: Path | None = None
    ) -> PromptConfig:
        base = root or _PROMPT_ROOT
        return cls.from_file(base / feature / f"{version_id}.yaml")

    @classmethod
    def latest(cls, *, feature: str = "classifier", root: Path | None = None) -> PromptConfig:
        base = (root or _PROMPT_ROOT) / feature
        versions = sorted(base.glob("v*.yaml"))
        if not versions:
            raise FileNotFoundError(f"no prompt versions found in {base}")
        return cls.from_file(versions[-1])
