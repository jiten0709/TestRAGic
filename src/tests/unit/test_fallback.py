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


# --------------------------------------------------------------------------
# images -- the OCR path's frame captioning. Not a second call path.
# --------------------------------------------------------------------------

def capture(gateway, model="vision-1", provider_name="oc", content="a login form"):
    """Record the outgoing body so the message shape can be asserted."""
    sent = {}
    inner = gateway.chat_ok(provider_name, model, content=content)

    def handler(request, body):
        sent.update(body)
        return inner(request, body)

    gateway.chat_by_model[model] = handler
    return sent


def test_images_become_openai_content_parts(gateway):
    sent = capture(gateway)
    text, attribution = provider.chat("describe this", model="vision-1", images=[b"\xff\xd8jpeg"])

    content = sent["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert text == "a login form"
    assert attribution.model == "vision-1"


def test_a_text_prompt_still_sends_a_bare_string(gateway):
    sent = capture(gateway, model="gpt-4o-mini", provider_name="openai")
    provider.chat("describe this", model="gpt-4o-mini")
    assert sent["messages"][-1]["content"] == "describe this"


def test_a_vision_request_does_not_fall_back_to_a_text_model(three_rung):
    """The chain would descend to the text default and then to `auto`, sending an
    image to models that may not accept one -- and on the gateway this was built
    against every `auto/*` route answers 502. Failing on the named model is honest."""
    three_rung.chat_by_model["vision-1"] = three_rung.status(500)

    with pytest.raises(provider.ProviderError) as excinfo:
        provider.chat("describe", model="vision-1", images=[b"jpeg"])

    assert [model for model, _ in excinfo.value.attempts] == ["vision-1"]
    assert three_rung.chat_models_called == ["vision-1"]
