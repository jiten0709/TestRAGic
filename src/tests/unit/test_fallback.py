"""The outer fallback chain: order, bounds, what is retryable, and the
dimension guard that keeps a FAISS store from being corrupted."""

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
# embeddings: the chain is dimension-constrained
# --------------------------------------------------------------------------

def test_embeddings_succeed_and_record_attribution(gateway):
    embeddings = provider.OmniRouteEmbeddings(model="openai/text-embedding-3-small")
    vectors = embeddings.embed_documents(["one", "two"])

    assert len(vectors) == 2 and len(vectors[0]) == 1536
    assert embeddings.dimensions == 1536
    assert embeddings.last_attribution.provider == "openai"


def test_embed_query_returns_a_single_vector(gateway):
    embeddings = provider.OmniRouteEmbeddings(model="openai/text-embedding-3-small")
    assert len(embeddings.embed_query("hello")) == 1536


def test_a_width_mismatched_candidate_is_dropped_from_the_chain(gateway):
    """3-large is 3072-dim; against a 1536-dim store it is not a candidate."""
    embeddings = provider.OmniRouteEmbeddings(
        model="openai/text-embedding-3-large", expected_dimensions=1536
    )
    embeddings.embed_documents(["one"])

    assert gateway.embedding_models_called == ["openai/text-embedding-3-small"]
    assert embeddings.dimensions == 1536


def test_an_empty_dimension_chain_fails_loudly(gateway):
    """Fail rather than write vectors that would corrupt the store."""
    embeddings = provider.OmniRouteEmbeddings(
        model="openai/text-embedding-3-large", expected_dimensions=768
    )
    with pytest.raises(provider.EmbeddingDimensionMismatch) as excinfo:
        embeddings.embed_documents(["one"])

    assert gateway.embedding_models_called == [], "nothing was sent, nothing was written"
    assert "768" in str(excinfo.value)


def test_an_unexpected_returned_width_is_refused(gateway):
    """The catalog can be wrong; the vectors themselves are checked too."""
    gateway.embeddings_by_model["mystery/model"] = gateway.embeddings_ok(
        "mystery", "model", dimensions=768
    )
    embeddings = provider.OmniRouteEmbeddings(model="mystery/model", expected_dimensions=1536)

    with pytest.raises(provider.EmbeddingDimensionMismatch) as excinfo:
        embeddings.embed_documents(["one"])
    assert "768" in str(excinfo.value) and "1536" in str(excinfo.value)


def test_embedding_failures_fall_back_within_the_same_width(gateway):
    gateway.embeddings_by_model["openai/text-embedding-ada-002"] = gateway.status(429)
    embeddings = provider.OmniRouteEmbeddings(
        model="openai/text-embedding-ada-002", expected_dimensions=1536
    )
    embeddings.embed_documents(["one"])

    assert gateway.embedding_models_called == [
        "openai/text-embedding-ada-002", "openai/text-embedding-3-small",
    ]
    assert embeddings.last_attribution.fallback is True


def test_embeddings_have_no_auto_rung(gateway):
    """`auto` is documented for chat only; it is never used for embeddings."""
    gateway.default_embeddings = gateway.status(500)
    with pytest.raises(provider.ProviderError):
        provider.OmniRouteEmbeddings(model="openai/text-embedding-3-small").embed_documents(["x"])

    assert "auto" not in gateway.embedding_models_called


# --------------------------------------------------------------------------
# what the store records about itself
# --------------------------------------------------------------------------

def test_store_meta_round_trips(gateway, tmp_path):
    embeddings = provider.OmniRouteEmbeddings(model="openai/text-embedding-3-small")
    embeddings.embed_documents(["one"])
    provider.write_store_meta(tmp_path, embeddings.last_attribution, embeddings.dimensions)

    meta = provider.read_store_meta(tmp_path)
    assert meta["dimensions"] == 1536
    assert meta["model"] == "text-embedding-3-small"


def test_missing_store_meta_is_none(tmp_path):
    assert provider.read_store_meta(tmp_path) is None
    assert provider.read_store_meta(tmp_path / "nope") is None
