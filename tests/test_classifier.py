from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mrd.feature.classifier import build_request, classify, parse
from mrd.feature.schema import CATEGORIES, Classification, strict_schema
from mrd.prompts import PromptConfig
from mrd.providers.cassette import CassetteProvider

from .conftest import run
from .fixtures import FIXTURE_EMAILS, by_key

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #


def test_schema_is_strict_dialect() -> None:
    schema = strict_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["category", "summary"]
    assert "$defs" not in schema
    assert schema["properties"]["category"]["enum"] == list(CATEGORIES)


def test_classification_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Classification.model_validate({"category": "billing", "summary": "s", "confidence": 0.9})


def test_classification_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        Classification.model_validate({"category": "refunds", "summary": "s"})


# --------------------------------------------------------------------------- #
# Parsing - never raises, always reports
# --------------------------------------------------------------------------- #


def test_parses_valid_output() -> None:
    result, error = parse('{"category": "billing", "summary": "Charged twice."}')
    assert error is None
    assert result is not None
    assert result.category == "billing"


def test_invalid_json_is_reported_not_raised() -> None:
    result, error = parse("I'm not sure how to categorise this one, sorry!")
    assert result is None
    assert error is not None
    assert "invalid JSON" in error


def test_schema_violation_is_reported_not_raised() -> None:
    result, error = parse('{"category": "refunds", "summary": "x"}')
    assert result is None
    assert error is not None
    assert "schema violation" in error


def test_empty_summary_is_a_schema_violation() -> None:
    _, error = parse('{"category": "general", "summary": ""}')
    assert error is not None and "schema violation" in error


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def test_request_carries_prompt_and_schema(prompt_v001: PromptConfig) -> None:
    request = build_request("hello", prompt_v001)
    assert request.model == prompt_v001.model
    assert request.temperature == 0.0
    assert request.user == "hello"
    assert request.json_schema == strict_schema()
    assert "Examples:" in request.system


# --------------------------------------------------------------------------- #
# End to end, offline
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture", [f for f in FIXTURE_EMAILS if not f.malformed], ids=lambda f: f.key
)
def test_classify_offline(fixture, prompt_v001: PromptConfig, replay: CassetteProvider) -> None:
    outcome = run(classify(fixture.email, prompt_v001, replay))

    assert outcome.ok
    assert outcome.parse_error is None
    assert outcome.classification is not None
    assert outcome.classification.category == fixture.category
    assert outcome.response.fingerprint == build_request(fixture.email, prompt_v001).fingerprint()


def test_classify_records_malformed_output_instead_of_crashing(
    prompt_v001: PromptConfig, replay: CassetteProvider
) -> None:
    """A bad model output is eval data, not an exception - schema validity is a gate signal."""
    fixture = by_key("malformed_output")
    outcome = run(classify(fixture.email, prompt_v001, replay))

    assert not outcome.ok
    assert outcome.classification is None
    assert outcome.parse_error is not None
    assert outcome.response.text  # the raw output is preserved for the diff report


def test_offline_tier_makes_no_network_call(
    monkeypatch: pytest.MonkeyPatch, cassette_dir: Path
) -> None:
    """N4: the unit tier must work with no keys in the environment at all."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "OPENAI_API_KEY" not in os.environ

    prompt = PromptConfig.load("v001", root=cassette_dir.parents[1] / "prompts")
    outcome = run(
        classify(by_key("billing_duplicate_charge").email, prompt, CassetteProvider(cassette_dir))
    )
    assert outcome.ok


# --------------------------------------------------------------------------- #
# Cross-provider parity - requires live keys
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")),
    reason="cross-provider parity needs both live API keys",
)
def test_same_email_classifies_identically_across_providers(prompt_v001: PromptConfig) -> None:
    from dataclasses import replace as dc_replace

    from mrd.providers.registry import resolve

    fixture = by_key("billing_duplicate_charge")
    request = build_request(fixture.email, prompt_v001)

    openai_text = run(resolve("gpt-4o-mini").complete(request)).text
    anthropic_text = run(
        resolve("claude-sonnet-5").complete(dc_replace(request, model="claude-sonnet-5"))
    ).text

    assert json.loads(openai_text)["category"] == json.loads(anthropic_text)["category"]
