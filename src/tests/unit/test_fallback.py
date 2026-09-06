"""The outer chat fallback chain: order, bounds, and what is retryable.

Embeddings have no chain -- they are produced locally by one model, and their
width guard is covered in test_embeddings.py."""

import pytest

from src.utils import provider


@pytest.fixture
def three_rung(gateway, monkeypatch):
    """requested -> app default -> auto, all distinct, so every rung is visible."""
    monkeypatch.setenv("OMNIROUTE_LLM_MODEL", "gpt-4o-mini")
    gateway.chat_by_model["gpt-4o-mini"] = gateway.chat_ok("openai", "gpt-4o-mini")
    gateway.chat_by_model["auto"] = gateway.chat_ok("xai", "grok-4")
    return gateway


# --------------------------------------------------------------------------
# what falls through to the next rung
# --------------------------------------------------------------------------

@pytest.mark.parametrize("failure, reason", [
    (("status", 404), "404 model not found"),
    (("status", 401), "401 auth rejected"),
    (("status", 403), "403 auth rejected"),
    (("status", 429), "429 rate limited"),
    (("status", 500), "500 upstream error"),
    (("status", 503), "503 upstream error"),
    (("timeout", None), "timed out"),
    (("connect", None), "connection refused"),
])
def test_retryable_failures_advance_the_chain(three_rung, failure, reason):
    kind, code = failure
    builder = {"status": lambda: three_rung.status(code),
               "timeout": three_rung.timeout,
               "connect": three_rung.unreachable}[kind]
    three_rung.chat_by_model["google/gemini-2.5-pro"] = builder()

    text, attribution = provider.chat("hi", model="google/gemini-2.5-pro")

    assert attribution.fallback is True
    assert attribution.model == "gpt-4o-mini", "the served model is the headline"
    assert "google/gemini-2.5-pro" in attribution.note
    assert reason in attribution.note
    assert three_rung.chat_models_called == ["google/gemini-2.5-pro", "gpt-4o-mini"]


def test_a_400_does_not_trigger_fallback(three_rung):
    """Our prompt is wrong; another model burns latency to fail identically."""
    three_rung.chat_by_model["google/gemini-2.5-pro"] = three_rung.status(400)

    with pytest.raises(provider.ProviderError) as excinfo:
        provider.chat("hi", model="google/gemini-2.5-pro")

    assert three_rung.chat_models_called == ["google/gemini-2.5-pro"], "exactly one attempt"
    assert "400 bad request" in str(excinfo.value)


def test_two_step_fallback_names_every_step(three_rung):
    three_rung.chat_by_model["google/gemini-2.5-pro"] = three_rung.timeout()
    three_rung.chat_by_model["gpt-4o-mini"] = three_rung.status(429)

    _, attribution = provider.chat("hi", model="google/gemini-2.5-pro")

    assert attribution.model == "grok-4"
    assert "timed out" in attribution.note and "429" in attribution.note
    assert three_rung.chat_models_called == ["google/gemini-2.5-pro", "gpt-4o-mini", "auto"]


def test_a_fallback_never_displays_the_requested_model_as_the_producer(three_rung):
    three_rung.chat_by_model["google/gemini-2.5-pro"] = three_rung.status(404)
    _, attribution = provider.chat("hi", model="google/gemini-2.5-pro")

    assert attribution.display == "Openai — gpt-4o-mini"
    assert "gemini" in attribution.note, "the requested model appears only in the note"


# --------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------

def test_the_chain_is_bounded_at_three_attempts(three_rung):
    three_rung.default_chat = three_rung.status(500)
    three_rung.chat_by_model.clear()

    with pytest.raises(provider.ProviderError):
        provider.chat("hi", model="google/gemini-2.5-pro")

    assert len(three_rung.chat_models_called) == provider.MAX_ATTEMPTS == 3


def test_the_chain_never_revisits_a_model(gateway, monkeypatch):
    """Dedupe is the structural guarantee against a loop."""
    monkeypatch.setenv("OMNIROUTE_LLM_MODEL", "gpt-4o-mini")
    gateway.default_chat = gateway.status(500)

    with pytest.raises(provider.ProviderError):
        provider.chat("hi", model="gpt-4o-mini")   # requested == app default

    called = gateway.chat_models_called
    assert called == ["gpt-4o-mini", "auto"]
    assert len(called) == len(set(called))


def test_openai_mode_has_no_auto_rung(openai_direct):
    openai_direct.default_chat = openai_direct.status(500)

    with pytest.raises(provider.ProviderError):
        provider.chat("hi", model="gpt-4o")

    assert "auto" not in openai_direct.chat_models_called


def test_an_exhausted_chain_lists_every_attempt(three_rung):
    three_rung.chat_by_model["google/gemini-2.5-pro"] = three_rung.status(404)
    three_rung.chat_by_model["gpt-4o-mini"] = three_rung.status(401)
    three_rung.chat_by_model["auto"] = three_rung.unreachable()

    with pytest.raises(provider.ProviderError) as excinfo:
        provider.chat("hi", model="google/gemini-2.5-pro")

    message = str(excinfo.value)
    for fragment in ("gemini-2.5-pro (404", "gpt-4o-mini (401", "auto (connection refused"):
        assert fragment in message
    assert len(excinfo.value.attempts) == 3


def test_an_empty_response_is_flagged_but_not_a_provider_failure(gateway):
    gateway.default_chat = gateway.chat_ok(content="")
    text, attribution = provider.chat("hi", model="gpt-4o-mini")

    assert text == ""
    assert attribution.fallback is False
    assert "empty response" in attribution.note
