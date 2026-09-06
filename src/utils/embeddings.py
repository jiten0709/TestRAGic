"""Local embeddings, and the metadata that keeps a FAISS store honest.

Embeddings are deliberately *not* a gateway concern. OmniRoute only serves
embeddings for providers it holds credentials for; the instance this app talks to
holds chat credentials only, so it answers ``GET /v1/embeddings`` with ``data: []``
and every ``POST /v1/embeddings`` with 400 *"No credentials for embedding
provider"*. That made RAG inert while every run still reported success.

So vectors are produced here, in-process, by a local sentence-transformers model,
and ``src/utils/provider.py`` is left addressing the gateway for chat only. The
dependency runs one way: this module imports ``Attribution`` from there, never the
reverse.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.utils.config import get_embedding_model
from src.utils.provider import Attribution

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="utils.log")

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Vector width per model, so a store's width can be checked *before* paying to load
# the weights. An unlisted model reports None, which every caller already treats as
# "cannot check" rather than "mismatch".
KNOWN_DIMENSIONS = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}

STORE_META = "embedding_meta.json"

# Streamlit re-runs the whole script on every interaction, so an uncached model is
# a multi-GB reload per click. Keyed by (model, device); the lock keeps two reruns
# from loading it twice.
_loaded: dict[tuple[str, str], object] = {}
_load_lock = threading.Lock()


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised instead of writing vectors that would corrupt an existing store."""


class NonFiniteEmbedding(RuntimeError):
    """Raised instead of indexing NaN vectors, which poison every later query."""


def default_model() -> str:
    """session_state -> $EMBEDDING_MODEL -> Qwen3-Embedding-0.6B."""
    return get_embedding_model() or DEFAULT_MODEL


def resolve_device() -> str:
    """$EMBEDDING_DEVICE, else the best accelerator actually present."""
    forced = (os.getenv("EMBEDDING_DEVICE") or "").strip().lower()
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batch_size() -> int:
    try:
        return max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
    except ValueError:
        return 16


def model_dimensions(model: str) -> int | None:
    """Vector width, from the static map or an already-loaded model. Never loads."""
    if model in KNOWN_DIMENSIONS:
        return KNOWN_DIMENSIONS[model]
    for (loaded_id, _), st_model in _loaded.items():
        if loaded_id == model:
            return st_model.get_sentence_embedding_dimension()
    return None


def _build(model_id: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_id,
        device=device,
        # Qwen3's config asks for bfloat16, which CPU kernels emulate: measured on
        # this machine, 32 texts took 2.10s in bfloat16 against 0.52s in float32,
        # and float32 was exact where bfloat16 drifted the unit norms by 3e-3.
        # 0.6B parameters is 2.4GB in float32 -- affordable anywhere this runs.
        model_kwargs={"dtype": torch.float32},
        # Qwen3 embeddings pool the LAST token. Under the default right padding,
        # every text shorter than the longest in its batch would pool a pad token
        # instead -- silently wrong vectors, not an error.
        processor_kwargs={"padding_side": "left"},
    )


def _padded_batch_is_sane(model) -> bool:
    """Encode two texts of very different lengths and check the result is finite.

    Apple's MPS attention kernel returns NaN for every *padded* row of a batch --
    at float32, float16 and bfloat16 alike, verified on torch 2.7.1 / macOS 25.2.
    Encoded one at a time the same texts are fine, so nothing upstream notices: the
    NaNs land in FAISS and every later similarity query silently returns garbage.
    One padded batch at load time is what separates a working accelerator from that.
    """
    import numpy as np

    probe = model.encode(["short", "a considerably longer sentence that forces padding"],
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
    return bool(np.isfinite(probe).all())


def load_model(model_id: str, device: str):
    key = (model_id, device)
    with _load_lock:
        if key not in _loaded:
            logger.info("loading %s on %s (first call downloads the weights)", model_id, device)
            started = time.monotonic()
            model = _build(model_id, device)
            # Only padded batches are affected, so a batch size of 1 never hits it --
            # honour that rather than advising a knob the probe would ignore.
            if device != "cpu" and batch_size() > 1 and not _padded_batch_is_sane(model):
                logger.warning(
                    "%s produces NaN for padded batches on %s; falling back to cpu. "
                    "EMBEDDING_BATCH_SIZE=1 keeps %s by avoiding padding entirely.",
                    model_id, device, device,
                )
                del model
                model = _build(model_id, "cpu")
            _loaded[key] = model
            logger.info("loaded %s in %.1fs", model_id, time.monotonic() - started)
    return _loaded[key]


class Qwen3Embeddings(Embeddings):
    """LangChain adapter so ``FAISS.from_documents`` keeps working unchanged.

    Documents are encoded in batches; a query is encoded singly and with the
    instruction prefix the model was trained to expect on queries only -- asymmetric
    by design, so prefixing documents too would degrade retrieval.

    Vectors are L2-normalized because LangChain's FAISS wrapper indexes with
    ``IndexFlatL2``: on unit vectors, L2 distance ranks identically to the cosine
    similarity Qwen3 was trained for.
    """

    def __init__(self, model: str | None = None, device: str | None = None,
                 expected_dimensions: int | None = None):
        self.model = model or default_model()
        self.device = device or resolve_device()
        self.expected_dimensions = expected_dimensions
        self.dimensions: int | None = None
        self.last_attribution: Attribution | None = None

        declared = model_dimensions(self.model)
        if expected_dimensions and declared and declared != expected_dimensions:
            raise EmbeddingDimensionMismatch(
                f"{self.model} produces {declared}-dim vectors but the existing store is "
                f"{expected_dimensions}-dim. Vectors of different widths are not comparable "
                "-- clear src/data/vector_store/ or set EMBEDDING_MODEL to a matching model."
            )

    def _encoder(self):
        encoder = load_model(self.model, self.device)
        # load_model may have fallen back (see _padded_batch_is_sane); a display
        # naming a device the model is not on is the kind of unverified claim the
        # Attribution design exists to prevent.
        self.device = str(getattr(encoder, "device", self.device))
        return encoder

    def _record(self, vectors, elapsed_ms: int) -> None:
        import numpy as np

        if not np.isfinite(vectors).all():
            # A NaN row indexes fine and then makes every similarity query involving
            # it return garbage -- exactly the silent corruption this module exists
            # to prevent. Refuse rather than write it.
            raise NonFiniteEmbedding(
                f"{self.model} on {self.device} returned non-finite vectors for "
                f"{int((~np.isfinite(vectors)).any(axis=1).sum())} of {len(vectors)} texts. "
                "Set EMBEDDING_DEVICE=cpu (or EMBEDDING_BATCH_SIZE=1) and re-ingest."
            )
        width = len(vectors[0]) if len(vectors) else None
        if self.expected_dimensions and width and width != self.expected_dimensions:
            raise EmbeddingDimensionMismatch(
                f"{self.model} returned {width}-dim vectors but the existing store is "
                f"{self.expected_dimensions}-dim. Refusing to write mixed vectors."
            )
        self.dimensions = width
        self.last_attribution = Attribution(
            requested_model=self.model, provider="local", model=self.model,
            display=f"Local — {self.model} ({self.device})",
            source="local", gateway="local", latency_ms=elapsed_ms,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        encoder = self._encoder()
        started = time.monotonic()
        vectors = encoder.encode(
            list(texts), batch_size=batch_size(), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        self._record(vectors, int((time.monotonic() - started) * 1000))
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        encoder = self._encoder()
        started = time.monotonic()
        vector = encoder.encode(
            [text], prompt_name="query", normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        self._record(vector, int((time.monotonic() - started) * 1000))
        return vector[0].tolist()


# --------------------------------------------------------------------------
# what a store records about itself
# --------------------------------------------------------------------------

def read_store_meta(store_path: str | Path) -> dict | None:
    """The model + dimensions an existing FAISS store was built with."""
    meta = Path(store_path) / STORE_META
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_store_meta(store_path: str | Path, attribution: Attribution, dimensions: int | None,
                     source: str | None = None, content_hash: str | None = None) -> None:
    """Record what built this store, and what it was built from.

    `source` is the human-readable origin (a URL or a file path) and `content_hash`
    fingerprints the chunks that were embedded, so a later run can tell whether
    re-embedding would produce anything different -- see `chunk_and_vectorize`.
    """
    path = Path(store_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / STORE_META).write_text(
        json.dumps(
            {"provider": attribution.provider,
             "model": attribution.model or attribution.requested_model,
             "requested_model": attribution.requested_model, "dimensions": dimensions,
             "source": source, "content_hash": content_hash,
             "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            indent=2,
        ),
        encoding="utf-8",
    )
