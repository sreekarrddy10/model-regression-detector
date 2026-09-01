"""Provider contract tests.

The point of the provider layer is that the eval engine cannot tell vendors
apart. These tests assert that property directly, using canned SDK-shaped
payloads - no network, no SDK installed, no API key.
"""

from __future__ import annotations

import json
import sys
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


# --------------------------------------------------------------------------- #
# One retry layer, not two
# --------------------------------------------------------------------------- #


class _RecordingSDK:
    """Captures the kwargs the provider constructs its client with."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return SimpleNamespace()


@pytest.mark.parametrize(
    "module,attr,provider_factory,env",
    [
        (
            "openai",
            "AsyncOpenAI",
            lambda: openai_provider.OpenAIProvider(api_key="k"),
            "OPENAI_API_KEY",
        ),
        (
            "anthropic",
            "AsyncAnthropic",
            lambda: anthropic_provider.AnthropicProvider(api_key="k"),
            "ANTHROPIC_API_KEY",
        ),
    ],
)
def test_sdk_retries_are_disabled(monkeypatch, module, attr, provider_factory, env) -> None:
    """The SDKs retry internally with their own backoff, and that backoff falls
    inside the latency measurement. Two retry layers turned a rate-limited call
    into a 16-second "model response", and the gate reads latency drift as a
    regression. mrd.retry must be the only retry layer."""
    sdk = _RecordingSDK()
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{attr: sdk}))
    provider_factory()._get_client()
    assert sdk.kwargs["max_retries"] == 0


# --------------------------------------------------------------------------- #
# Temperature across SDK generations
# --------------------------------------------------------------------------- #


class _FakeMessages:
    """Stands in for `client.messages`, with a controllable create() signature."""

    def __init__(self, *, takes_temperature: bool) -> None:
        self.seen: dict[str, object] = {}
        if takes_temperature:

            async def create(
                *,
                model,
                max_tokens,
                system,
                messages,
                tools=None,
                tool_choice=None,
                temperature=None,
            ):
                self.seen = {"model": model, "temperature": temperature}
                return _anthropic_payload("s")

        else:

            async def create(*, model, max_tokens, system, messages, tools=None, tool_choice=None):
                self.seen = {"model": model}
                return _anthropic_payload("s")

        self.create = create


def _anthropic_with(*, takes_temperature: bool) -> tuple[object, _FakeMessages]:
    provider = anthropic_provider.AnthropicProvider(api_key="test")
    messages = _FakeMessages(takes_temperature=takes_temperature)
    provider._client = SimpleNamespace(messages=messages)
    return provider, messages


def test_temperature_is_sent_when_the_sdk_accepts_it() -> None:
    provider, messages = _anthropic_with(takes_temperature=True)
    run(provider.complete(replace(REQUEST, model="claude-sonnet-5", temperature=0.0)))
    assert messages.seen["temperature"] == 0.0


def test_temperature_the_sdk_cannot_honour_is_refused_not_dropped() -> None:
    """anthropic 1.x removed `temperature`. Silently sampling at the provider
    default would surface as flaky classifications the gate reads as drift."""
    provider, _ = _anthropic_with(takes_temperature=False)
    with pytest.raises(ProviderError) as exc:
        run(provider.complete(replace(REQUEST, model="claude-sonnet-5", temperature=0.0)))
    message = str(exc.value)
    assert "does not accept a temperature" in message
    assert "anthropic<1.0" in message  # the error names both ways out
    assert "None" in message


def test_temperature_none_runs_on_an_sdk_without_the_parameter() -> None:
    """The documented opt-out: no determinism claim, and the call goes through."""
    provider, messages = _anthropic_with(takes_temperature=False)
    response = run(provider.complete(replace(REQUEST, model="claude-sonnet-5", temperature=None)))
    assert "temperature" not in messages.seen
    assert json.loads(response.text) == {"category": "billing", "summary": "s"}


def test_temperature_none_is_omitted_by_openai_too() -> None:
    """None must mean the same thing on both providers, or parity is a fiction."""
    captured: dict[str, object] = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _openai_payload('{"category": "billing", "summary": "s"}')

    provider = openai_provider.OpenAIProvider(api_key="test")
    completions = SimpleNamespace(create=create)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    run(provider.complete(replace(REQUEST, temperature=None)))
    assert "temperature" not in captured


def test_temperature_none_is_distinguishable_in_the_fingerprint() -> None:
    """A run that made no determinism claim must not hash like one that did."""
    assert replace(REQUEST, temperature=None).fingerprint() != REQUEST.fingerprint()


def test_accepts_temperature_reads_the_signature() -> None:
    async def with_temp(*, model, temperature=None): ...
    async def without(*, model): ...
    async def kwargs_only(**kw): ...

    assert anthropic_provider._accepts_temperature(with_temp)
    assert not anthropic_provider._accepts_temperature(without)
    assert anthropic_provider._accepts_temperature(kwargs_only)


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


# --------------------------------------------------------------------------- #
# Config discovery
# --------------------------------------------------------------------------- #


def test_pricing_path_prefers_an_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(pricing.ENV_VAR, str(tmp_path / "custom.yaml"))
    assert pricing.default_path() == tmp_path / "custom.yaml"


def test_pricing_path_falls_back_to_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """How the container finds it: WORKDIR holds config/, not site-packages."""
    monkeypatch.delenv(pricing.ENV_VAR, raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "pricing.yaml").write_text("models: {}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert pricing.default_path() == tmp_path / "config" / "pricing.yaml"


def test_pricing_path_falls_back_to_the_source_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_root: Path
) -> None:
    monkeypatch.delenv(pricing.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    assert pricing.default_path() == repo_root / "config" / "pricing.yaml"


def test_default_pricing_resolves_from_the_repo(repo_root: Path) -> None:
    """The regression the container build caught: lookup returning None silently."""
    import os

    previous = os.getcwd()
    try:
        os.chdir(repo_root)
        pricing._load.cache_clear()  # noqa: SLF001
        assert pricing.lookup("gpt-4o-mini") is not None
    finally:
        os.chdir(previous)
        pricing._load.cache_clear()  # noqa: SLF001


def test_resolver_returns_the_working_directory_candidate_when_nothing_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Error messages should name the path a user would expect to create."""
    from mrd import paths

    monkeypatch.chdir(tmp_path)
    assert paths.resolve("data/golden/nothing.jsonl") == tmp_path / "data/golden/nothing.jsonl"


def test_resolver_prefers_cwd_over_the_source_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The container mounts data beside WORKDIR; site-packages must not win."""
    from mrd import paths

    (tmp_path / "prompts").mkdir()
    monkeypatch.chdir(tmp_path)
    assert paths.resolve("prompts") == tmp_path / "prompts"
