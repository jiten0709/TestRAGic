"""Single LLM/embedding provider abstraction for TestRAGic.

Every chat completion and every embedding in this app goes through this module.
Nothing else imports `openai`, `langchain_openai`, or any provider SDK.

Two modes, resolved by :func:`resolve_endpoint`:

* ``omniroute`` -- an OmniRoute gateway (OpenAI-compatible) at ``OMNIROUTE_BASE_URL``.
  OmniRoute is the provider abstraction; this module is only the thin client that
  points at it, passes a model string through, and reads attribution back off the
  ``X-OmniRoute-*`` response headers.
* ``openai``    -- no gateway configured: plain OpenAI, exactly as before.

Transport is the already-installed `openai` SDK pointed at ``base_url``; the two
catalog GETs use `httpx`. No new dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import openai
from langchain_core.embeddings import Embeddings

from src.utils.config import (
    get_embedding_model,
    get_gateway_api_key,
    get_gateway_base_url,
    get_llm_api_key,
    get_llm_model,
)

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = "http://localhost:20128/v1"
OPENAI_URL = "https://api.openai.com/v1"

# OmniRoute accepts any string when REQUIRE_API_KEY=false (its default); the SDK
# refuses to construct a client with an empty key, so send a placeholder.
GATEWAY_PLACEHOLDER_KEY = "omniroute-local"

MAX_ATTEMPTS = 3  # bounds the fallback chain: worst case 3 x timeout
CATALOG_TTL = 300.0

AUTO_VARIANTS = ["auto", "auto/coding", "auto/fast", "auto/cheap", "auto/smart"]
STATIC_OPENAI_CHAT = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]

# Only used when the gateway does not report dimensions itself. Vectors of
# different dimensions are not comparable -- see EmbeddingDimensionMismatch.
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Injected by tests (httpx.MockTransport). When set, both the OpenAI SDK and the
# catalog GETs run through it, so the real header-parsing path is exercised.
http_client: httpx.Client | None = None


class ProviderError(RuntimeError):
    """Raised only when the whole fallback chain is exhausted."""

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts
        detail = ", ".join(f"{m} ({why})" for m, why in attempts) or "no candidates"
        super().__init__(f"No model produced a result. Tried: {detail}")


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised instead of writing vectors that would corrupt an existing store."""


class Endpoint(NamedTuple):
    base_url: str
    api_key: str
    gateway: str  # "omniroute" | "openai"


@dataclass(frozen=True)
class Attribution:
    """Who actually answered. `source` records how confidently we know."""

    requested_model: str
    provider: str | None = None
    model: str | None = None
    display: str = ""
    fallback: bool = False
    rerouted: bool = False
    note: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    source: str = "unknown"  # "headers" | "body" | "unknown"
    gateway: str = "openai"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# endpoint / configuration
# --------------------------------------------------------------------------

def resolve_endpoint() -> Endpoint:
    """(base_url, api_key, gateway). Presence of a gateway URL selects OmniRoute."""
    forced = (os.getenv("TESTRAGIC_LLM_PROVIDER") or "").strip().lower()
    base = (get_gateway_base_url() or "").strip()

    if forced == "omniroute" or (base and forced != "openai"):
        return Endpoint(
            (base or DEFAULT_GATEWAY_URL).rstrip("/"),
            get_gateway_api_key() or GATEWAY_PLACEHOLDER_KEY,
            "omniroute",
        )
    return Endpoint(OPENAI_URL, get_llm_api_key() or "", "openai")


def is_configured() -> bool:
    """True when the app has somewhere to send a request."""
    ep = resolve_endpoint()
    return ep.gateway == "omniroute" or bool(ep.api_key)


def timeout_seconds() -> float:
    try:
        return float(os.getenv("OMNIROUTE_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def _normalize(model: str, gateway: str) -> str:
    """`provider/model` addressing is OmniRoute's; OpenAI-direct wants the bare id."""
    if gateway == "openai" and model.lower().startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def default_chat_model() -> str:
    """session_state -> $OMNIROUTE_LLM_MODEL -> 'auto' (gateway) / 'gpt-4o-mini'."""
    chosen = get_llm_model()
    gateway = resolve_endpoint().gateway
    if chosen:
        return _normalize(chosen, gateway)
    if gateway == "omniroute":
        return "auto"
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


def default_embedding_model() -> str:
    """`auto` is not verified for embeddings, so always an explicit model."""
    chosen = get_embedding_model()
    gateway = resolve_endpoint().gateway
    if chosen:
        return _normalize(chosen, gateway)
    if gateway == "omniroute":
        return "openai/text-embedding-3-small"  # 1536 dims, matches existing stores
    return "text-embedding-3-small"


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

_clients: dict[tuple, openai.OpenAI] = {}


def _client() -> openai.OpenAI:
    ep = resolve_endpoint()
    key = (ep.base_url, ep.api_key, timeout_seconds(), id(http_client))
    if key not in _clients:
        _clients[key] = openai.OpenAI(
            base_url=ep.base_url,
            api_key=ep.api_key or "missing",
            timeout=timeout_seconds(),
            max_retries=0,  # OmniRoute retries internally across its combo
            http_client=http_client,
        )
    return _clients[key]


def reset_clients() -> None:
    """Drop cached clients and catalogs (used by tests and the refresh button)."""
    _clients.clear()
    _catalogs.clear()


# --------------------------------------------------------------------------
# model catalogs
# --------------------------------------------------------------------------

_catalogs: dict[str, tuple[float, list[dict]]] = {}


def _fetch_catalog(path: str, refresh: bool) -> tuple[list[dict] | None, str | None]:
    ep = resolve_endpoint()
    url = f"{ep.base_url}{path}"
    if not refresh and url in _catalogs:
        stamped, cached = _catalogs[url]
        if time.monotonic() - stamped < CATALOG_TTL:
            return cached, None
    try:
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        client = http_client or httpx
        resp = client.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json().get("data") or []
    except Exception as exc:  # unreachable, 404, malformed -- all degrade the same
        return None, f"{type(exc).__name__}: {exc}"
    entries = [e for e in data if isinstance(e, dict) and e.get("id")]
    _catalogs[url] = (time.monotonic(), entries)
    return entries, None


def _static(ids: list[str], owner: str = "openai") -> list[dict]:
    out = []
    for i in ids:
        provider, _, name = i.partition("/")
        out.append(
            {
                "id": i,
                "owned_by": provider if name else ("omniroute" if i.startswith("auto") else owner),
                "dimensions": KNOWN_DIMENSIONS.get(name or i),
            }
        )
    return out


def list_chat_models(refresh: bool = False) -> tuple[list[dict], str | None]:
    """Chat models from ``GET /v1/models``. Returns (models, warning).

    A warning means the list is the static fallback -- never present it as live.
    """
    gateway = resolve_endpoint().gateway
    entries, warning = _fetch_catalog("/models", refresh)
    if entries is None:
        fallback = AUTO_VARIANTS + STATIC_OPENAI_CHAT if gateway == "omniroute" else STATIC_OPENAI_CHAT
        return _static(fallback), f"Model catalog unreachable ({warning}); showing a static list."

    chat = [e for e in entries if e.get("type") in (None, "", "chat", "model")]
    autos = [e for e in chat if str(e["id"]).startswith("auto")]
    rest = sorted((e for e in chat if not str(e["id"]).startswith("auto")),
                  key=lambda e: (str(e.get("owned_by") or ""), str(e["id"])))
    if gateway == "omniroute":
        have = {str(e["id"]) for e in autos}
        autos = _static([a for a in AUTO_VARIANTS if a not in have]) + autos
    models = autos + rest
    return (models, None) if models else (_static(STATIC_OPENAI_CHAT), "Catalog returned no models; showing a static list.")


def list_embedding_models(refresh: bool = False) -> tuple[list[dict], str | None]:
    """Embedding models from OmniRoute's ``GET /v1/embeddings`` catalog."""
    gateway = resolve_endpoint().gateway
    entries, warning = _fetch_catalog("/embeddings", refresh)
    if entries is None:
        ids = (
            ["openai/text-embedding-3-small", "openai/text-embedding-3-large", "openai/text-embedding-ada-002"]
            if gateway == "omniroute"
            else list(KNOWN_DIMENSIONS)
        )
        note = "Embedding catalog unavailable; showing a static list."
        return _static(ids), (note if gateway == "omniroute" else None)
    for e in entries:
        e.setdefault("dimensions", KNOWN_DIMENSIONS.get(str(e["id"]).rpartition("/")[2]))
    return sorted(entries, key=lambda e: (str(e.get("owned_by") or ""), str(e["id"]))), None


def model_dimensions(model: str) -> int | None:
    """Vector width for an embedding model, if the catalog or the static map knows."""
    entries, _ = list_embedding_models()
    for e in entries:
        if str(e["id"]) == model:
            dims = e.get("dimensions")
            if isinstance(dims, int):
                return dims
    return KNOWN_DIMENSIONS.get(model.rpartition("/")[2])


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

def pretty(provider: str | None, model: str | None) -> str:
    if provider and model:
        return f"{provider.replace('-', ' ').title()} — {model}"
    return model or provider or "unknown"


def _read_attribution(requested, headers, body_model, gateway, measured_ms,
                      fallback=False, note=None) -> Attribution:
    prov = headers.get("X-OmniRoute-Provider") or None
    served = headers.get("X-OmniRoute-Model") or None

    if prov or served:
        source = "headers"
    elif body_model:
        source, served = "body", body_model
    else:
        source = "unknown"

    if source == "unknown":
        display = f"{requested} (requested; gateway did not report)"
        rerouted = False
    else:
        display = pretty(prov, served)
        known = {v.lower() for v in (served, f"{prov}/{served}" if prov and served else None) if v}
        rerouted = requested.lower() not in known

    try:
        latency = int(headers.get("X-OmniRoute-Latency-Ms") or measured_ms)
    except (TypeError, ValueError):
        latency = measured_ms

    if rerouted and not note:
        note = f"routed by OmniRoute from '{requested}'"

    return Attribution(
        requested_model=requested, provider=prov, model=served, display=display,
        fallback=fallback, rerouted=rerouted, note=note,
        request_id=headers.get("X-OmniRoute-Request-Id") or None,
        latency_ms=latency, source=source, gateway=gateway,
    )


def summarize(pairs: "list[tuple[str, Attribution]]") -> dict:
    """Collapse per-call attributions into the persisted `metadata.attribution` block.

    `pairs` is (category, attribution). Several categories mean several independent
    calls that can land on several providers, so the summary reports the spread
    rather than pretending to one. Calls that no model served keep their reason --
    an exhausted chain must not be flattened into a generic "unknown".
    """
    pairs = [(category, a) for category, a in pairs if a]
    if not pairs:
        return unattributed("no model attribution was recorded").to_dict()

    per_category = [
        {"category": category, "requested": a.requested_model, "provider": a.provider,
         "model": a.model, "fallback": a.fallback, "rerouted": a.rerouted,
         "request_id": a.request_id, "latency_ms": a.latency_ms, "note": a.note}
        for category, a in pairs
    ]

    served_by_a_model = [a for _, a in pairs if a.requested_model]
    if not served_by_a_model:
        # Nothing here came from a model. Keep the first failure's explanation.
        summary = pairs[0][1].to_dict()
        summary.update({"per_category": per_category, "model_count": 0})
        return summary

    served = {(a.provider, a.model) for a in served_by_a_model}
    head = served_by_a_model[0]
    failed = len(pairs) - len(served_by_a_model)

    if len(served) > 1 or failed:
        n_fb = sum(1 for a in served_by_a_model if a.fallback)
        detail = ", ".join(
            part for part in (f"{n_fb} fallback" if n_fb else "",
                              f"{failed} with no model" if failed else "") if part
        )
        display = f"{len(served)} model{'s' if len(served) > 1 else ''}"
        display += f" ({detail})" if detail else ""
    else:
        display = head.display

    summary = head.to_dict()
    summary.update({
        "display": display,
        "fallback": any(a.fallback for a in served_by_a_model),
        "rerouted": any(a.rerouted for a in served_by_a_model),
        "per_category": per_category,
        "model_count": len(served),
    })
    return summary


def option_index(options: list[str], current: str | None) -> int:
    """Safe replacement for ``options.index(current)`` -- a dynamic catalog makes
    the bare call a guaranteed ValueError the moment the default drops out."""
    return options.index(current) if current in options else 0


def unattributed(reason: str) -> Attribution:
    """For output that no model produced -- mock data, exhausted chains."""
    return Attribution(requested_model="", display="no model was called", note=reason,
                       gateway=resolve_endpoint().gateway)


# --------------------------------------------------------------------------
# failure classification
# --------------------------------------------------------------------------

def _classify(exc: Exception) -> tuple[bool, str]:
    """(retryable, one-line reason). A 400 is our bug -- another model fails alike."""
    if isinstance(exc, openai.BadRequestError):
        return False, "400 bad request"
    if isinstance(exc, openai.APITimeoutError):
        return True, f"timed out after {timeout_seconds():.0f}s"
    if isinstance(exc, openai.APIConnectionError):
        return True, "connection refused"
    if isinstance(exc, openai.NotFoundError):
        return True, "404 model not found"
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return True, f"{exc.status_code} auth rejected"
    if isinstance(exc, openai.RateLimitError):
        return True, "429 rate limited"
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500, f"{exc.status_code} upstream error"
    return False, f"{type(exc).__name__}: {exc}"


def _chain(requested: str | None, candidates: list[str | None]) -> list[str]:
    """Deduped, bounded. Dedupe is the structural guarantee against a loop."""
    chain: list[str] = []
    for model in [requested, *candidates]:
        if model and model not in chain:
            chain.append(model)
    return chain[:MAX_ATTEMPTS]


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------

def chat(prompt: str, model: str | None = None, temperature: float = 0.3,
         max_tokens: int = 4000, system: str | None = None) -> tuple[str, Attribution]:
    """One chat completion, with the outer fallback chain.

    Raises ProviderError only when every rung failed.
    """
    ep = resolve_endpoint()
    requested = _normalize(model or default_chat_model(), ep.gateway)
    tail = [default_chat_model(), "auto" if ep.gateway == "omniroute" else None]
    chain = _chain(requested, tail)

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    attempts: list[tuple[str, str]] = []
    for candidate in chain:
        started = time.monotonic()
        try:
            raw = _client().chat.completions.with_raw_response.create(
                model=candidate, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as exc:
            retryable, reason = _classify(exc)
            attempts.append((candidate, reason))
            logger.warning("chat: %s failed (%s)", candidate, reason)
            if not retryable:
                break
            continue

        elapsed = int((time.monotonic() - started) * 1000)
        body = raw.parse()
        text = (body.choices[0].message.content or "") if body.choices else ""
        note = None
        if attempts:
            note = "fallback; " + "; then ".join(f"requested {m} {why}" for m, why in attempts)
        elif not text.strip():
            note = "model returned an empty response"

        return text, _read_attribution(
            requested=candidate, headers=raw.headers,
            body_model=getattr(body, "model", None), gateway=ep.gateway,
            measured_ms=elapsed, fallback=bool(attempts), note=note,
        )

    raise ProviderError(attempts)


def test_connection(model: str | None = None) -> tuple[bool, Attribution | None, str | None]:
    """5-token ping. Returns (ok, attribution, error)."""
    try:
        _, attribution = chat("ping", model=model, max_tokens=5)
        return True, attribution, None
    except Exception as exc:
        return False, None, str(exc)


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------

STORE_META = "embedding_meta.json"


def read_store_meta(store_path: str | Path) -> dict | None:
    """The model + dimensions an existing FAISS store was built with."""
    meta = Path(store_path) / STORE_META
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_store_meta(store_path: str | Path, attribution: Attribution, dimensions: int | None) -> None:
    path = Path(store_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / STORE_META).write_text(
        json.dumps(
            {"provider": attribution.provider, "model": attribution.model or attribution.requested_model,
             "requested_model": attribution.requested_model, "dimensions": dimensions,
             "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            indent=2,
        ),
        encoding="utf-8",
    )


class OmniRouteEmbeddings(Embeddings):
    """LangChain adapter so ``FAISS.from_documents`` keeps working unchanged.

    Records ``.last_attribution`` after each batch. The fallback chain here is
    dimension-constrained: a candidate whose vector width differs from the active
    store's is not a candidate, because mixing widths corrupts the store.
    """

    def __init__(self, model: str | None = None, expected_dimensions: int | None = None):
        self.model = model or default_embedding_model()
        self.expected_dimensions = expected_dimensions
        self.last_attribution: Attribution | None = None
        self.dimensions: int | None = None

    def _candidates(self) -> list[str]:
        ep = resolve_endpoint()
        requested = _normalize(self.model, ep.gateway)
        chain = _chain(requested, [default_embedding_model()])
        if self.expected_dimensions is None:
            return chain
        kept = [m for m in chain if model_dimensions(m) in (None, self.expected_dimensions)]
        if not kept:
            raise EmbeddingDimensionMismatch(
                f"No embedding model matches the existing store's {self.expected_dimensions} "
                f"dimensions (tried {', '.join(chain)}). Refusing to write mixed vectors -- "
                "clear src/data/vector_store/ or pick a matching model."
            )
        return kept

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ep = resolve_endpoint()
        attempts: list[tuple[str, str]] = []
        for candidate in self._candidates():
            started = time.monotonic()
            try:
                raw = _client().embeddings.with_raw_response.create(model=candidate, input=texts)
            except Exception as exc:
                retryable, reason = _classify(exc)
                attempts.append((candidate, reason))
                logger.warning("embeddings: %s failed (%s)", candidate, reason)
                if not retryable:
                    break
                continue

            body = raw.parse()
            vectors = [item.embedding for item in sorted(body.data, key=lambda d: d.index)]
            width = len(vectors[0]) if vectors else None
            if self.expected_dimensions and width and width != self.expected_dimensions:
                raise EmbeddingDimensionMismatch(
                    f"{candidate} returned {width}-dim vectors but the existing store is "
                    f"{self.expected_dimensions}-dim. Refusing to write mixed vectors."
                )

            self.dimensions = width
            note = "fallback; " + "; then ".join(f"requested {m} {why}" for m, why in attempts) if attempts else None
            self.last_attribution = _read_attribution(
                requested=candidate, headers=raw.headers,
                body_model=getattr(body, "model", None), gateway=ep.gateway,
                measured_ms=int((time.monotonic() - started) * 1000),
                fallback=bool(attempts), note=note,
            )
            return vectors

        raise ProviderError(attempts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
