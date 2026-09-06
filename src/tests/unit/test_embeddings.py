"""The local embedding provider.

Weights are never downloaded here: the `encoder` fixture installs a FakeEncoder as
the loaded model, so the batching, the query/document asymmetry and the width guard
all run for real while the suite stays offline and fast. The genuine model is
exercised by src/tests/rag_sanity.py.
"""

import pytest

from conftest import FakeEncoder
from src.utils import embeddings as embeddings_mod
from src.utils.embeddings import (
    EmbeddingDimensionMismatch,
    NonFiniteEmbedding,
    Qwen3Embeddings,
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def test_the_default_model_is_qwen3_and_env_overrides_it(monkeypatch):
    assert embeddings_mod.default_model() == "Qwen/Qwen3-Embedding-0.6B"
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    assert embeddings_mod.default_model() == "Qwen/Qwen3-Embedding-4B"


def test_device_is_forced_by_env_before_any_hardware_probe(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    assert embeddings_mod.resolve_device() == "cpu"


def test_device_autoselects_the_best_accelerator_present(monkeypatch):
    """cuda, else mps, else cpu -- and the probe order is what decides."""
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert embeddings_mod.resolve_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert embeddings_mod.resolve_device() == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert embeddings_mod.resolve_device() == "cpu"


def test_batch_size_survives_a_garbage_env_value(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "32")
    assert embeddings_mod.batch_size() == 32
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "not-a-number")
    assert embeddings_mod.batch_size() == 16


def test_dimensions_are_known_without_loading_the_model():
    """The store's width is checked before a 1.2GB load, so it must not need one."""
    assert embeddings_mod.model_dimensions("Qwen/Qwen3-Embedding-0.6B") == 1024
    assert embeddings_mod.model_dimensions("Qwen/Qwen3-Embedding-8B") == 4096
    assert embeddings_mod.model_dimensions("some/unlisted-model") is None


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def test_documents_are_embedded_in_batches_and_attributed(encoder):
    embeddings = Qwen3Embeddings()
    vectors = embeddings.embed_documents(["one", "two", "three"])

    assert len(vectors) == 3 and len(vectors[0]) == 1024
    assert all(isinstance(value, float) for value in vectors[0]), "FAISS wants plain floats"
    assert embeddings.dimensions == 1024

    _, kwargs = encoder.calls[0]
    assert kwargs["batch_size"] == 16
    assert kwargs["normalize_embeddings"] is True, (
        "LangChain's FAISS wrapper indexes with IndexFlatL2; only unit vectors make "
        "that rank the same as the cosine similarity Qwen3 was trained for"
    )


def test_a_query_gets_the_instruction_prefix_and_a_document_does_not(encoder):
    """Qwen3 embeddings are asymmetric: the prompt belongs on queries only."""
    embeddings = Qwen3Embeddings()
    embeddings.embed_query("how do I log in")
    embeddings.embed_documents(["the login page"])

    assert encoder.prompt_names == ["query", None]


def test_embed_query_returns_one_flat_vector(encoder):
    assert len(Qwen3Embeddings().embed_query("hello")) == 1024


def test_attribution_records_the_local_model_and_device(encoder):
    embeddings = Qwen3Embeddings()
    embeddings.embed_documents(["one"])
    attribution = embeddings.last_attribution

    assert attribution.provider == "local" and attribution.gateway == "local"
    assert attribution.model == "Qwen/Qwen3-Embedding-0.6B"
    assert "cpu" in attribution.display
    assert attribution.fallback is False, "there is no chain to fall back through"


# --------------------------------------------------------------------------
# the NaN guard: Apple's MPS attention kernel returns NaN for padded rows
# --------------------------------------------------------------------------

def test_an_accelerator_that_nans_on_padded_batches_is_dropped_for_cpu(monkeypatch):
    """Verified on torch 2.7.1 / macOS 25.2: MPS returns NaN for every *padded* row
    at float32, float16 and bfloat16 alike, while solo encodes look perfect. Left
    unhandled the NaNs reach FAISS and every later query silently returns garbage."""
    import numpy as np

    built = []

    def fake_build(model_id, device):
        built.append(device)
        fake = FakeEncoder()
        if device == "mps":
            healthy = fake.encode

            def nan_on_padded_batches(texts, **kwargs):
                out = healthy(texts, **kwargs)
                if len(texts) > 1:  # solo encodes look perfect; only padding breaks
                    out[0][:] = np.nan
                return out

            fake.encode = nan_on_padded_batches
        return fake

    monkeypatch.setattr(embeddings_mod, "_loaded", {})
    monkeypatch.setattr(embeddings_mod, "_build", fake_build)

    encoder = embeddings_mod.load_model("Qwen/Qwen3-Embedding-0.6B", "mps")

    assert built == ["mps", "cpu"], "the broken accelerator must be rebuilt on cpu"
    assert np.isfinite(encoder.encode(["a", "bb"])).all(), "the kept model must be sane"


def test_a_healthy_accelerator_is_kept(monkeypatch):
    built = []

    def fake_build(model_id, device):
        built.append(device)
        return FakeEncoder()

    monkeypatch.setattr(embeddings_mod, "_loaded", {})
    monkeypatch.setattr(embeddings_mod, "_build", fake_build)
    embeddings_mod.load_model("Qwen/Qwen3-Embedding-0.6B", "mps")

    assert built == ["mps"], "a working accelerator must not be thrown away"


def test_non_finite_vectors_are_refused_rather_than_indexed(encoder, monkeypatch):
    """The probe is a one-time check; this is the guard that cannot be outrun."""
    import numpy as np

    real_encode = encoder.encode

    def nan_encode(texts, **kwargs):
        out = real_encode(texts, **kwargs)
        out[0][:] = np.nan
        return out

    monkeypatch.setattr(encoder, "encode", nan_encode)
    with pytest.raises(NonFiniteEmbedding) as excinfo:
        Qwen3Embeddings().embed_documents(["one", "two"])
    assert "1 of 2" in str(excinfo.value)


def test_the_device_reported_is_the_one_actually_used(encoder, monkeypatch):
    """A display naming a device the model is not on is exactly the unverified
    claim the Attribution design exists to prevent."""
    encoder.device = "cpu"
    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")
    monkeypatch.setattr(embeddings_mod, "_loaded", {("Qwen/Qwen3-Embedding-0.6B", "mps"): encoder})

    embeddings = Qwen3Embeddings()
    assert embeddings.device == "mps", "the requested device, before anything loads"
    embeddings.embed_documents(["one"])

    assert embeddings.device == "cpu"
    assert "cpu" in embeddings.last_attribution.display


# --------------------------------------------------------------------------
# the width guard: mixing widths corrupts a FAISS store
# --------------------------------------------------------------------------

def test_a_width_mismatch_is_refused_before_the_model_is_even_loaded(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_loaded", {})
    with pytest.raises(EmbeddingDimensionMismatch) as excinfo:
        Qwen3Embeddings(model="Qwen/Qwen3-Embedding-0.6B", expected_dimensions=1536)

    assert "1024" in str(excinfo.value) and "1536" in str(excinfo.value)


def test_an_unexpected_returned_width_is_refused(encoder):
    """The static map can be wrong or absent; the vectors themselves are checked too."""
    embeddings = Qwen3Embeddings(model="some/unlisted-model", expected_dimensions=1536)
    embeddings_mod._loaded[("some/unlisted-model", "cpu")] = encoder

    with pytest.raises(EmbeddingDimensionMismatch) as excinfo:
        embeddings.embed_documents(["one"])
    assert "1024" in str(excinfo.value) and "1536" in str(excinfo.value)


def test_a_matching_width_passes(encoder):
    embeddings = Qwen3Embeddings(expected_dimensions=1024)
    assert len(embeddings.embed_documents(["one"])[0]) == 1024


# --------------------------------------------------------------------------
# what a store records about itself
# --------------------------------------------------------------------------

def test_store_meta_round_trips(encoder, tmp_path):
    embeddings = Qwen3Embeddings()
    embeddings.embed_documents(["one"])
    embeddings_mod.write_store_meta(tmp_path, embeddings.last_attribution, embeddings.dimensions)

    meta = embeddings_mod.read_store_meta(tmp_path)
    assert meta["dimensions"] == 1024
    assert meta["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert meta["provider"] == "local"


def test_missing_store_meta_is_none(tmp_path):
    assert embeddings_mod.read_store_meta(tmp_path) is None
    assert embeddings_mod.read_store_meta(tmp_path / "nope") is None


def test_batch_size_one_keeps_the_accelerator(monkeypatch):
    """Padding is what breaks; one text per batch never pads, so the advice the
    fallback message gives has to actually be honoured."""
    built = []

    def fake_build(model_id, device):
        built.append(device)
        return FakeEncoder()

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "1")
    monkeypatch.setattr(embeddings_mod, "_loaded", {})
    monkeypatch.setattr(embeddings_mod, "_build", fake_build)
    monkeypatch.setattr(embeddings_mod, "_padded_batch_is_sane",
                        lambda m: pytest.fail("the probe must be skipped at batch size 1"))

    embeddings_mod.load_model("Qwen/Qwen3-Embedding-0.6B", "mps")
    assert built == ["mps"]
