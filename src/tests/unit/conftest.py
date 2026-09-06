"""Fixtures for the provider unit tests.

Everything runs through ``httpx.MockTransport``, which the `openai` SDK accepts as
its ``http_client``. That means the real request construction, the real status-code
-> exception mapping and the real header parsing are all exercised -- a hand-written
fake client would bypass exactly the code most worth testing.
"""

import json
import time

import httpx
import pytest

from src.utils import provider


# --------------------------------------------------------------------------
# response builders -- each returns handler(request, body) -> httpx.Response
# --------------------------------------------------------------------------

def chat_ok(provider_name="openai", model="gpt-4o-mini", content="[]",
            body_model=None, headers=True, latency_ms=142, request_id="req-1"):
    """A 200 chat completion, optionally without the X-OmniRoute-* headers."""
    def handler(request, body):
        hdrs = {}
        if headers:
            hdrs = {
                "X-OmniRoute-Provider": provider_name,
                "X-OmniRoute-Model": model,
                "X-OmniRoute-Latency-Ms": str(latency_ms),
                "X-OmniRoute-Request-Id": request_id,
                "X-OmniRoute-Decision": "strategy=single",
            }
        return httpx.Response(
            200,
            headers=hdrs,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body_model if body_model is not None else model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    return handler


def status(code, message="boom"):
    """A non-2xx response -- the SDK turns this into the matching exception type."""
    def handler(request, body):
        return httpx.Response(code, json={"error": {"message": message, "type": "test"}})
    return handler


def timeout():
    def handler(request, body):
        raise httpx.ReadTimeout("simulated timeout", request=request)
    return handler


def unreachable():
    def handler(request, body):
        raise httpx.ConnectError("simulated connection refused", request=request)
    return handler


# --------------------------------------------------------------------------
# the gateway double
# --------------------------------------------------------------------------

class MockGateway:
    """Routes by path + method, and for completions by the request's `model`."""

    # response builders, reachable from any test via the `gateway` fixture
    chat_ok = staticmethod(chat_ok)
    status = staticmethod(status)
    timeout = staticmethod(timeout)
    unreachable = staticmethod(unreachable)

    def __init__(self):
        self.chat_by_model = {}
        self.default_chat = chat_ok()
        self.model_catalog = [
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
            {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "text-embedding-3-small", "object": "model",
             "owned_by": "openai", "type": "embedding"},
        ]
        self.model_catalog_status = 200
        self.calls = []  # (method, path, model)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = {}
        if request.method == "POST" and request.content:
            body = json.loads(request.content)
        model = body.get("model")
        self.calls.append((request.method, path, model))

        if request.method == "GET" and path.endswith("/models"):
            if self.model_catalog_status != 200:
                return httpx.Response(self.model_catalog_status, json={"error": "nope"})
            return httpx.Response(200, json={"object": "list", "data": self.model_catalog})

        if path.endswith("/chat/completions"):
            return self.chat_by_model.get(model, self.default_chat)(request, body)

        return httpx.Response(404, json={"error": "unknown route"})

    @property
    def chat_models_called(self):
        return [m for method, p, m in self.calls
                if method == "POST" and p.endswith("/chat/completions")]



@pytest.fixture(autouse=True)
def clean_provider(monkeypatch):
    """Every test starts with no provider env and no cached clients or catalogs."""
    for var in ("OMNIROUTE_BASE_URL", "OMNIROUTE_API_KEY", "OMNIROUTE_LLM_MODEL",
                "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "OMNIROUTE_TIMEOUT",
                "TESTRAGIC_LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    provider.reset_clients()
    provider.http_client = None
    yield
    provider.reset_clients()
    provider.http_client = None


@pytest.fixture
def gateway(monkeypatch):
    """A mocked OmniRoute at localhost:20128, wired into the provider module."""
    gw = MockGateway()
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    provider.http_client = httpx.Client(transport=httpx.MockTransport(gw.handle))
    provider.reset_clients()
    yield gw
    provider.http_client.close()


@pytest.fixture
def openai_direct(monkeypatch):
    """OpenAI-direct mode (no gateway), still served by a mock transport."""
    gw = MockGateway()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    provider.http_client = httpx.Client(transport=httpx.MockTransport(gw.handle))
    provider.reset_clients()
    yield gw
    provider.http_client.close()


# --------------------------------------------------------------------------
# the local encoder double
# --------------------------------------------------------------------------

class FakeEncoder:
    """Stands in for a loaded SentenceTransformer. Records how it was called so the
    query/document asymmetry and the batch size stay observable."""

    def __init__(self, dimensions=1024):
        self.dimensions = dimensions
        self.calls = []  # (texts, kwargs)

    def get_sentence_embedding_dimension(self):
        return self.dimensions

    def encode(self, texts, **kwargs):
        import numpy as np
        self.calls.append((list(texts), kwargs))
        # distinct, unit-length rows: distinct so retrieval order is meaningful,
        # unit-length because the real encoder is asked to normalize.
        out = np.zeros((len(texts), self.dimensions), dtype="float32")
        for row, text in enumerate(texts):
            out[row][hash(text) % self.dimensions] = 1.0
        return out

    @property
    def prompt_names(self):
        return [kw.get("prompt_name") for _, kw in self.calls]


@pytest.fixture
def encoder(monkeypatch):
    """Installs a FakeEncoder as the loaded model for the default model+device."""
    from src.utils import embeddings as embeddings_mod

    fake = FakeEncoder()
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(embeddings_mod, "_loaded", {(embeddings_mod.DEFAULT_MODEL, "cpu"): fake})
    return fake
