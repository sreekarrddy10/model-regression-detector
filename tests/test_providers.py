"""Provider contract tests.

The point of the provider layer is that the eval engine cannot tell vendors
apart. These tests assert that property directly, using canned SDK-shaped
payloads - no network, no SDK installed, no API key.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mrd.providers import anthropic as anthropic_provider
from mrd.providers import openai as openai_provider
from mrd.providers import pricing
from mrd.providers.base import ProviderError, Request, Usage
from mrd.providers.cassette import CassetteMiss, CassetteProvider
from mrd.providers.registry import Tier, build, resolve

from .conftest import run

pytestmark = pytest.mark.unit

REQUEST = Request(
    model="gpt-4o-mini",
    system="You triage support email.",
    user="I was charged twice.",
    json_schema={"type": "object"},
)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_is_stable() -> None:
    assert REQUEST.fingerprint() == replace(REQUEST).fingerprint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "gpt-4o"),
        ("system", "You triage support email!"),
        ("user", "I was charged three times."),
        ("temperature", 0.5),
        ("max_tokens", 256),
        ("json_schema", {"type": "string"}),
    ],
)
def test_fingerprint_changes_with_every_field(field: str, value: object) -> None:
    """Any prompt edit must invalidate its cassette, or a stale replay is possible."""
    assert replace(REQUEST, **{field: value}).fingerprint() != REQUEST.fingerprint()


# --------------------------------------------------------------------------- #
# Normalization contract
# --------------------------------------------------------------------------- #


def _openai_payload(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )


def _anthropic_payload(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={"category": "billing", "summary": text})],
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )


def test_both_providers_produce_the_same_normalized_shape() -> None:
    oai = openai_provider.normalize(
        _openai_payload('{"category": "billing", "summary": "s"}'), REQUEST, latency_ms=11
    )
    ant = anthropic_provider.normalize(
        _anthropic_payload("s"), replace(REQUEST, model="claude-sonnet-5"), latency_ms=11
    )

    assert oai.usage == ant.usage == Usage(input_tokens=120, output_tokens=30)
    assert oai.latency_ms == ant.latency_ms == 11
    assert json.loads(oai.text) == {"category": "billing", "summary": "s"}
    assert json.loads(oai.text) == json.loads(ant.text)
    assert oai.provider == "openai"
    assert ant.provider == "anthropic"


def test_anthropic_falls_back_to_text_block() -> None:
    raw = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="plain prose")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    assert anthropic_provider.normalize(raw, REQUEST, 5).text == "plain prose"


def test_missing_usage_defaults_to_zero() -> None:
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
    )
    assert openai_provider.normalize(raw, REQUEST, 1).usage == Usage(0, 0)


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def test_priced_model_yields_a_cost(repo_root: Path) -> None:
    price = pricing.lookup("gpt-4o-mini", path=repo_root / "config" / "pricing.yaml")
    assert price is not None
    assert price.cost(Usage(1_000_000, 1_000_000)) == pytest.approx(0.75)


def test_unpriced_model_yields_none_not_zero(repo_root: Path) -> None:
    """An unpriced model must not masquerade as a free one."""
    assert (
        pricing.cost_for(
            "not-a-real-model", Usage(10, 10), path=repo_root / "config" / "pricing.yaml"
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("claude-sonnet-5", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
    ],
)
def test_routes_by_prefix(model: str, expected: str) -> None:
    assert resolve(model).name == expected


def test_unknown_model_is_a_clear_error() -> None:
    with pytest.raises(ProviderError, match="no provider registered"):
        resolve("llama-3-70b")


def test_unit_tier_can_never_reach_the_network(cassette_dir: Path) -> None:
    provider = build("gpt-4o-mini", tier="unit", cassette_dir=cassette_dir)
    assert isinstance(provider, CassetteProvider)
    assert provider._record_with is None  # noqa: SLF001


@pytest.mark.parametrize("tier", ["smoke", "full"])
def test_live_tiers_resolve_a_real_provider(tier: Tier, cassette_dir: Path) -> None:
    assert build("gpt-4o-mini", tier=tier, cassette_dir=cassette_dir).name == "openai"


# --------------------------------------------------------------------------- #
# Cassettes
# --------------------------------------------------------------------------- #


def test_cassette_round_trips(tmp_path: Path) -> None:
    source = openai_provider.normalize(_openai_payload('{"a": 1}'), REQUEST, latency_ms=42)
    CassetteProvider._save(
        tmp_path / f"{REQUEST.fingerprint()}.json", REQUEST, source
    )  # noqa: SLF001

    replayed = run(CassetteProvider(tmp_path).complete(REQUEST))
    assert replayed == source


def test_cassette_miss_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(CassetteMiss, match="make record"):
        run(CassetteProvider(tmp_path).complete(REQUEST))


def test_record_tier_captures_a_miss(tmp_path: Path) -> None:
    """`record` fills gaps from an upstream provider; `unit` never does."""

    class Upstream:
        name = "upstream"
        calls = 0

        async def complete(self, request: Request):  # type: ignore[no-untyped-def]
            Upstream.calls += 1
            return openai_provider.normalize(_openai_payload('{"ok": true}'), request, 7)

    provider = CassetteProvider(tmp_path, record_with=Upstream())

    first = run(provider.complete(REQUEST))
    assert Upstream.calls == 1
    assert (tmp_path / f"{REQUEST.fingerprint()}.json").exists()

    second = run(provider.complete(REQUEST))
    assert Upstream.calls == 1, "a recorded request must replay, not re-call upstream"
    assert second == first


def test_record_tier_wires_the_live_provider(cassette_dir: Path) -> None:
    provider = build("gpt-4o-mini", tier="record", cassette_dir=cassette_dir)
    assert isinstance(provider, CassetteProvider)
    assert provider._record_with is not None  # noqa: SLF001
    assert provider._record_with.name == "openai"  # noqa: SLF001


def test_edited_prompt_misses_its_old_cassette(tmp_path: Path) -> None:
    """The regression this guards: replaying a recording made for a different prompt."""
    source = openai_provider.normalize(_openai_payload("{}"), REQUEST, latency_ms=1)
    CassetteProvider._save(
        tmp_path / f"{REQUEST.fingerprint()}.json", REQUEST, source
    )  # noqa: SLF001

    edited = replace(REQUEST, system=REQUEST.system + " Be concise.")
    with pytest.raises(CassetteMiss):
        run(CassetteProvider(tmp_path).complete(edited))
