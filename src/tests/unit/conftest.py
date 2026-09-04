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


def embeddings_ok(provider_name="openai", model="text-embedding-3-small",
                  dimensions=1536, headers=True):
    def handler(request, body):
        inputs = body.get("input") or [""]
        if isinstance(inputs, str):
            inputs = [inputs]
        hdrs = {
            "X-OmniRoute-Provider": provider_name,
            "X-OmniRoute-Model": model,
            "X-OmniRoute-Latency-Ms": "88",
            "X-OmniRoute-Request-Id": "req-emb",
        } if headers else {}
        return httpx.Response(
            200,
            headers=hdrs,
            json={
                "object": "list",
                "model": model,
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.01] * dimensions}
                    for i in range(len(inputs))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
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
    embeddings_ok = staticmethod(embeddings_ok)
    status = staticmethod(status)
    timeout = staticmethod(timeout)
    unreachable = staticmethod(unreachable)

    def __init__(self):
        self.chat_by_model = {}
        self.embeddings_by_model = {}
        self.default_chat = chat_ok()
        self.default_embeddings = embeddings_ok()
        self.model_catalog = [
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
            {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "text-embedding-3-small", "object": "model",
             "owned_by": "openai", "type": "embedding"},
        ]
        self.embedding_catalog = [
            {"id": "openai/text-embedding-3-small", "object": "model",
             "owned_by": "openai", "type": "embedding", "dimensions": 1536},
            {"id": "openai/text-embedding-3-large", "object": "model",
             "owned_by": "openai", "type": "embedding", "dimensions": 3072},
        ]
        self.model_catalog_status = 200
        self.embedding_catalog_status = 200
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

        if request.method == "GET" and path.endswith("/embeddings"):
            if self.embedding_catalog_status != 200:
                return httpx.Response(self.embedding_catalog_status, json={"error": "nope"})
            return httpx.Response(200, json={"object": "list", "data": self.embedding_catalog})

        if path.endswith("/chat/completions"):
            return self.chat_by_model.get(model, self.default_chat)(request, body)

        if path.endswith("/embeddings"):
            return self.embeddings_by_model.get(model, self.default_embeddings)(request, body)

        return httpx.Response(404, json={"error": "unknown route"})

    @property
    def chat_models_called(self):
        return [m for method, p, m in self.calls
                if method == "POST" and p.endswith("/chat/completions")]

    @property
    def embedding_models_called(self):
        return [m for method, p, m in self.calls
                if method == "POST" and p.endswith("/embeddings")]


@pytest.fixture(autouse=True)
def clean_provider(monkeypatch):
    """Every test starts with no provider env and no cached clients or catalogs."""
    for var in ("OMNIROUTE_BASE_URL", "OMNIROUTE_API_KEY", "OMNIROUTE_LLM_MODEL",
                "OMNIROUTE_EMBEDDING_MODEL", "OMNIROUTE_TIMEOUT",
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
