"""One vector store per source.

Every ingestion used to `save_local` over the same `vector_store/` directory, so the
last video processed silently replaced every earlier one, and re-uploading the same
file produced a fresh `md5(name + timestamp)` id each time -- five ingestions of one
demo video left five transcripts and one clobbered store.
"""

import pytest

from src.agents.data_ingestion import (
    DEFAULT_STORE_KEY,
    DataIngestionAgent,
    chunks_fingerprint,
    store_key_for,
)
from src.utils import embeddings as embeddings_mod


# --------------------------------------------------------------------------
# key derivation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source, expected", [
    ("https://www.youtube.com/watch?v=iBTWBhODSU0", "iBTWBhODSU0"),
    ("https://youtu.be/iBTWBhODSU0", "iBTWBhODSU0"),
    ("https://www.youtube.com/embed/iBTWBhODSU0", "iBTWBhODSU0"),
    ("https://www.youtube.com/watch?v=iBTWBhODSU0&t=42s", "iBTWBhODSU0"),
    ("src/assets/videos/demo-1.mp4", "demo-1"),
    ("/tmp/My Demo (final).mov", "My-Demo-final"),
    ("", DEFAULT_STORE_KEY),
])
def test_a_source_maps_to_a_readable_key(source, expected):
    assert store_key_for(source) == expected


def test_the_same_source_always_gives_the_same_key():
    """The bug this replaces: md5(name + timestamp) changed on every upload."""
    assert store_key_for("demo-1.mp4") == store_key_for("demo-1.mp4")
    assert store_key_for("a/b/demo-1.mp4") == store_key_for("c/demo-1.mp4")


def test_different_sources_do_not_collide():
    assert store_key_for("demo-1.mp4") != store_key_for("demo-2.mp4")
    assert store_key_for("https://youtu.be/aaa") != store_key_for("https://youtu.be/bbb")


def test_a_key_is_safe_as_a_directory_name():
    for nasty in ("../../etc/passwd", "a/b/c", "name with spaces!.mp4", "..."):
        key = store_key_for(nasty)
        assert "/" not in key and ".." not in key and key


# --------------------------------------------------------------------------
# the fingerprint decides whether re-embedding is needed
# --------------------------------------------------------------------------

def chunk(i, text):
    return {"chunk_id": i, "text": text, "start_time": 0, "end_time": 1,
            "actions": [], "chunk_type": "action"}


def test_identical_chunks_fingerprint_identically():
    a = [chunk(0, "click login"), chunk(1, "enter password")]
    b = [chunk(0, "click login"), chunk(1, "enter password")]
    assert chunks_fingerprint(a) == chunks_fingerprint(b)


def test_changed_text_changes_the_fingerprint():
    a = [chunk(0, "click login")]
    assert chunks_fingerprint(a) != chunks_fingerprint([chunk(0, "click logout")])


# --------------------------------------------------------------------------
# stores live side by side and are reused
# --------------------------------------------------------------------------

@pytest.fixture
def agent(tmp_path, encoder, monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_loaded",
                        {(embeddings_mod.DEFAULT_MODEL, "cpu"): encoder})
    return DataIngestionAgent(data_dir=str(tmp_path))


def test_two_sources_do_not_overwrite_each_other(agent):
    """The whole point: ingesting B must not destroy A's vectors."""
    agent.chunk_and_vectorize([chunk(0, "video one")], store_key="demo-1",
                              source="demo-1.mp4")
    agent.chunk_and_vectorize([chunk(0, "video two")], store_key="demo-2",
                              source="demo-2.mp4")

    assert agent.store_path_for("demo-1").exists()
    assert agent.store_path_for("demo-2").exists()
    assert embeddings_mod.read_store_meta(agent.store_path_for("demo-1"))["source"] == "demo-1.mp4"
    assert embeddings_mod.read_store_meta(agent.store_path_for("demo-2"))["source"] == "demo-2.mp4"


def test_unchanged_chunks_are_not_re_embedded(agent, encoder):
    chunks = [chunk(0, "click login"), chunk(1, "enter password")]
    first = agent.chunk_and_vectorize(chunks, store_key="demo-1", source="demo-1.mp4")
    calls_after_first = len(encoder.calls)

    agent.vector_store = None  # a fresh run, as the app does
    second = agent.chunk_and_vectorize(chunks, store_key="demo-1", source="demo-1.mp4")

    assert first["reused"] is False and second["reused"] is True
    assert len(encoder.calls) == calls_after_first, "no text was embedded again"
    assert second["documents_count"] == 2


def test_changed_chunks_are_re_embedded(agent, encoder):
    agent.chunk_and_vectorize([chunk(0, "click login")], store_key="demo-1")
    calls_after_first = len(encoder.calls)

    agent.vector_store = None
    result = agent.chunk_and_vectorize([chunk(0, "click logout")], store_key="demo-1")

    assert result["reused"] is False
    assert len(encoder.calls) > calls_after_first, "changed text must be re-embedded"


def test_a_store_from_a_different_model_is_not_reused(agent, encoder, monkeypatch):
    """Same text, different embedder: the vectors are not interchangeable."""
    chunks = [chunk(0, "click login")]
    agent.chunk_and_vectorize(chunks, store_key="demo-1")

    agent.vector_store = None
    agent.embeddings.model = "Qwen/Qwen3-Embedding-4B"
    monkeypatch.setattr(embeddings_mod, "_loaded",
                        {("Qwen/Qwen3-Embedding-4B", "cpu"): encoder})

    assert agent.chunk_and_vectorize(chunks, store_key="demo-1")["reused"] is False


def test_the_retriever_follows_the_key(agent):
    agent.chunk_and_vectorize([chunk(0, "video one about logging in")], store_key="demo-1")
    agent.chunk_and_vectorize([chunk(0, "video two about billing")], store_key="demo-2")

    docs = agent.setup_retrieval_chain("demo-1").invoke("anything")
    assert [d.page_content for d in docs] == ["video one about logging in"], (
        "asking for demo-1 must not serve the store left in memory by demo-2"
    )


def test_an_unknown_key_has_no_retriever(agent):
    agent.chunk_and_vectorize([chunk(0, "video one")], store_key="demo-1")
    assert agent.setup_retrieval_chain("never-ingested") is None
