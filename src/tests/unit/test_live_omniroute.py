"""Live smoke test against a real OmniRoute gateway.

Skipped automatically when no gateway is reachable, so a normal `pytest` run is
unaffected. To run it, boot the gateway and point the suite at it:

    npm i -g omniroute && omniroute
    OMNIROUTE_BASE_URL=http://localhost:20128/v1 pytest -m integration src/tests/unit -o addopts=""

These are the phase-0 checks from the migration plan, as code: the whole
attribution design rests on the X-OmniRoute-* headers actually arriving.
"""

import os

import httpx
import pytest

from src.utils import provider

# Captured at import time: the autouse clean_provider fixture strips the env.
LIVE_URL = (os.getenv("OMNIROUTE_BASE_URL") or "").rstrip("/") or provider.DEFAULT_GATEWAY_URL
LIVE_KEY = os.getenv("OMNIROUTE_API_KEY") or ""


def _reachable():
    try:
        return httpx.get(f"{LIVE_URL}/models", timeout=2.0).status_code < 500
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason=f"no OmniRoute gateway at {LIVE_URL}"),
]


@pytest.fixture(autouse=True)
def live(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", LIVE_URL)
    if LIVE_KEY:
        monkeypatch.setenv("OMNIROUTE_API_KEY", LIVE_KEY)
    provider.reset_clients()


def test_the_model_catalog_is_reachable():
    models, warning = provider.list_chat_models(refresh=True)
    assert warning is None and models
    assert any(m["id"].startswith("auto") for m in models)


def test_the_embedding_catalog_is_reachable():
    models, warning = provider.list_embedding_models(refresh=True)
    assert models, "GET /v1/embeddings should return the embedding catalog"
    print(f"\nembedding models: {[m['id'] for m in models][:5]}")


def test_auto_answers_and_the_attribution_headers_arrive():
    """The finding the whole attribution design rests on."""
    text, attribution = provider.chat("Reply with the single word: pong", model="auto")

    assert text
    print(f"\nserved by: {attribution.display} "
          f"(source={attribution.source}, {attribution.latency_ms}ms, "
          f"rerouted={attribution.rerouted})")
    assert attribution.source == "headers", (
        "X-OmniRoute-Provider/-Model did not arrive; attribution degrades to the "
        "response body. Check for a reverse proxy stripping X-* headers."
    )
    assert attribution.provider and attribution.model


def test_embeddings_answer_with_the_expected_width():
    model = provider.default_embedding_model()
    embeddings = provider.OmniRouteEmbeddings(model=model)
    vectors = embeddings.embed_documents(["a short chunk", "another chunk"])

    assert len(vectors) == 2
    print(f"\nembedded by: {embeddings.last_attribution.display} "
          f"({embeddings.dimensions} dims)")
    expected = provider.model_dimensions(model)
    if expected:
        assert embeddings.dimensions == expected, "catalog width disagrees with reality"


def test_an_invalid_model_falls_back_rather_than_crashing():
    _, attribution = provider.chat("ping", model="definitely/not-a-real-model")
    assert attribution.fallback is True
    print(f"\nfell back to: {attribution.display} — {attribution.note}")
