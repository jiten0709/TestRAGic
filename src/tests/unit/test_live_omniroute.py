"""Live smoke test against a real OmniRoute gateway.

Skipped automatically when no gateway is reachable, so a normal `pytest` run is
unaffected. To run it, boot the gateway and point the suite at it:

    npm i -g omniroute && omniroute
    OMNIROUTE_BASE_URL=http://localhost:20128/v1 pytest -m integration src/tests/unit -o addopts=""

These are the phase-0 checks from the migration plan, as code: the whole
attribution design rests on the X-OmniRoute-* headers actually arriving.

Embeddings are absent here on purpose -- they no longer touch the gateway. The
local encoder is covered by test_embeddings.py (mocked) and src/tests/rag_sanity.py
(real weights, end to end).
"""

import os

import httpx
import pytest

from src.utils import provider
from src.utils.config import load_environment

# The gateway URL and key normally live in src/.env, not the shell, so read them the
# same way the app does before capturing. Without this the probe below runs keyless
# and a gateway with REQUIRE_API_KEY=true answers 401 -- which is < 500, so the tests
# would not skip, they would run unauthenticated and fail.
load_environment()

# Captured at import time: the autouse clean_provider fixture strips the env.
LIVE_URL = (os.getenv("OMNIROUTE_BASE_URL") or "").rstrip("/") or provider.DEFAULT_GATEWAY_URL
LIVE_KEY = os.getenv("OMNIROUTE_API_KEY") or ""


def _reachable():
    """Authenticated probe: a 401 means the gateway is up but we hold no usable key,
    which is a skip (nothing to smoke-test), not a failure."""
    try:
        headers = {"Authorization": f"Bearer {LIVE_KEY}"} if LIVE_KEY else {}
        return httpx.get(f"{LIVE_URL}/models", headers=headers, timeout=2.0).status_code == 200
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


def test_an_invalid_model_falls_back_rather_than_crashing():
    _, attribution = provider.chat("ping", model="definitely/not-a-real-model")
    assert attribution.fallback is True
    print(f"\nfell back to: {attribution.display} — {attribution.note}")
