"""Endpoint resolution, catalogs, defaults."""

import httpx
import pytest

from src.utils import provider


# --------------------------------------------------------------------------
# mode resolution
# --------------------------------------------------------------------------

def test_gateway_url_selects_omniroute(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    endpoint = provider.resolve_endpoint()
    assert endpoint.gateway == "omniroute"
    assert endpoint.base_url == "http://localhost:20128/v1"


def test_no_gateway_url_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    endpoint = provider.resolve_endpoint()
    assert endpoint.gateway == "openai"
    assert endpoint.base_url == provider.OPENAI_URL
    assert endpoint.api_key == "sk-test"


def test_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1/")
    assert provider.resolve_endpoint().base_url == "http://localhost:20128/v1"


def test_explicit_provider_override_wins(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TESTRAGIC_LLM_PROVIDER", "openai")
    assert provider.resolve_endpoint().gateway == "openai"


def test_gateway_needs_no_api_key(monkeypatch):
    """A local instance runs with REQUIRE_API_KEY=false; any string works."""
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    assert provider.resolve_endpoint().api_key == provider.GATEWAY_PLACEHOLDER_KEY
    assert provider.is_configured() is True


def test_api_key_without_gateway_is_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert provider.is_configured() is True


def test_neither_configured_is_not_configured():
    assert provider.is_configured() is False


def test_non_sk_key_is_accepted(monkeypatch):
    """The deleted `sk-` format gate must stay deleted."""
    monkeypatch.setenv("OPENAI_API_KEY", "my-gateway-token")
    assert provider.is_configured() is True


# --------------------------------------------------------------------------
# catalogs
# --------------------------------------------------------------------------

def test_chat_catalog_parses_and_orders(gateway):
    models, warning = provider.list_chat_models()
    ids = [m["id"] for m in models]

    assert warning is None
    assert ids[0].startswith("auto"), "auto variants must sort first"
    assert "gpt-4o-mini" in ids and "gemini-2.5-pro" in ids
    assert "text-embedding-3-small" not in ids, "embedding models are not chat models"


def test_chat_catalog_is_cached_then_refreshable(gateway):
    provider.list_chat_models()
    provider.list_chat_models()
    gets = [c for c in gateway.calls if c[0] == "GET" and c[1].endswith("/models")]
    assert len(gets) == 1, "second call should hit the TTL cache"

    provider.list_chat_models(refresh=True)
    gets = [c for c in gateway.calls if c[0] == "GET" and c[1].endswith("/models")]
    assert len(gets) == 2, "refresh must bust the cache"


def test_unreachable_catalog_degrades_with_a_warning(gateway):
    gateway.model_catalog_status = 503
    models, warning = provider.list_chat_models()

    assert models, "never present an empty dropdown"
    assert warning, "a static list must never be presented as if it were live"
    assert "auto" in [m["id"] for m in models]


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

def test_default_chat_model_prefers_env_then_gateway_default(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    assert provider.default_chat_model() == "auto"

    monkeypatch.setenv("OMNIROUTE_LLM_MODEL", "google/gemini-2.5-pro")
    assert provider.default_chat_model() == "google/gemini-2.5-pro"


def test_default_chat_model_in_openai_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert provider.default_chat_model() == "gpt-4o-mini"

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert provider.default_chat_model() == "gpt-4o"


def test_openai_direct_strips_the_provider_prefix(monkeypatch):
    """`provider/model` addressing is OmniRoute's; OpenAI wants the bare id."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OMNIROUTE_LLM_MODEL", "openai/gpt-4o")
    assert provider.default_chat_model() == "gpt-4o"


def test_openai_prefix_is_kept_for_the_gateway(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("OMNIROUTE_LLM_MODEL", "openai/gpt-4o")
    assert provider.default_chat_model() == "openai/gpt-4o"


def test_timeout_is_configurable_and_survives_garbage(monkeypatch):
    assert provider.timeout_seconds() == 60.0
    monkeypatch.setenv("OMNIROUTE_TIMEOUT", "5")
    assert provider.timeout_seconds() == 5.0
    monkeypatch.setenv("OMNIROUTE_TIMEOUT", "not-a-number")
    assert provider.timeout_seconds() == 60.0


# --------------------------------------------------------------------------
# the .index() regression
# --------------------------------------------------------------------------

def test_option_index_never_raises_on_a_missing_current():
    """app.py:366 used options.index(current), a guaranteed ValueError once the
    catalog became dynamic."""
    options = ["auto", "gpt-4o-mini"]
    assert provider.option_index(options, "gpt-4o-mini") == 1
    assert provider.option_index(options, "gpt-4o") == 0     # was ValueError
    assert provider.option_index(options, None) == 0
    assert provider.option_index([], "anything") == 0


# --------------------------------------------------------------------------
# the request the gateway actually receives
# --------------------------------------------------------------------------

def test_model_string_is_passed_through_untouched(gateway):
    gateway.chat_by_model["google/gemini-2.5-pro"] = gateway.chat_ok("google", "gemini-2.5-pro")
    provider.chat("hello", model="google/gemini-2.5-pro")
    assert gateway.chat_models_called == ["google/gemini-2.5-pro"]


def test_test_connection_reports_who_answered(gateway):
    gateway.default_chat = gateway.chat_ok("google", "gemini-2.5-pro", content="pong")
    ok, attribution, error = provider.test_connection()
    assert ok is True and error is None
    assert attribution.provider == "google"


def test_test_connection_reports_a_dead_gateway(gateway):
    gateway.default_chat = gateway.unreachable()
    ok, attribution, error = provider.test_connection()
    assert ok is False and attribution is None and error
