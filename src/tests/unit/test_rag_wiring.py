"""The RAG wiring, which used to fail invisibly.

Two defects are covered here, both proven against a live gateway that holds no
embedding-provider credentials (OmniRoute answers those 400):

1. ``wire_retriever`` warns when there is no retriever. Before, ``set_retriever(None)``
   was silent and the generator quietly prompted every category with
   ``transcript[:2000]`` while the run still reported success.
2. ``setup_retrieval_chain`` can actually load a store from disk. The cold-start
   ``FAISS.load_local`` call omitted ``allow_dangerous_deserialization=True``,
   which langchain-community 0.3.27 requires, so it always raised into a bare
   ``except`` and returned ``None``.
"""

import pytest

from src.agents.data_ingestion import DataIngestionAgent
from src.dashboard.app import wire_retriever


class FakeGenerator:
    def __init__(self):
        self.retriever = "unset"

    def set_retriever(self, retriever):
        self.retriever = retriever


class FakeIngestion:
    def __init__(self, retriever):
        self._retriever = retriever

    def setup_retrieval_chain(self):
        return self._retriever


# --------------------------------------------------------------------------
# 1. a missing store is reported, not swallowed
# --------------------------------------------------------------------------

def test_no_retriever_warns_and_names_the_reason(monkeypatch):
    warnings = []
    monkeypatch.setattr("src.dashboard.app.st.warning", warnings.append)

    agent = FakeGenerator()
    wire_retriever(
        agent,
        FakeIngestion(None),
        {"vector_store_info": {"success": False, "error": "400 bad request"}},
    )

    assert agent.retriever is None
    assert len(warnings) == 1, "the degradation must be visible exactly once"
    assert "RAG is off" in warnings[0]
    assert "400 bad request" in warnings[0], "the gateway's reason must survive"


def test_a_working_retriever_is_passed_through_silently(monkeypatch):
    warnings = []
    monkeypatch.setattr("src.dashboard.app.st.warning", warnings.append)

    agent = FakeGenerator()
    wire_retriever(agent, FakeIngestion("a-retriever"), {"vector_store_info": {"success": True}})

    assert agent.retriever == "a-retriever"
    assert warnings == [], "a healthy run must not nag"


def test_a_missing_store_still_warns_without_a_reason(monkeypatch):
    """The mock path builds no store and records no error; say so anyway."""
    warnings = []
    monkeypatch.setattr("src.dashboard.app.st.warning", warnings.append)

    wire_retriever(FakeGenerator(), FakeIngestion(None), {"vector_store_info": {}})

    assert len(warnings) == 1 and "RAG is off" in warnings[0]


# --------------------------------------------------------------------------
# 2. the cold-start load is opted in to deserialization
# --------------------------------------------------------------------------

def test_cold_start_load_passes_allow_dangerous_deserialization(tmp_path, monkeypatch, gateway):
    """Without the flag langchain raises, the bare except eats it, and cross-session
    store reuse is dead. Assert the flag reaches FAISS rather than the exception path."""
    store = tmp_path / "vector_store"
    store.mkdir()
    (store / "index.faiss").write_bytes(b"")

    seen = {}

    def fake_load_local(path, embeddings, **kwargs):
        seen.update(kwargs)
        return _FakeStore()

    monkeypatch.setattr("src.agents.data_ingestion.FAISS.load_local", staticmethod(fake_load_local))

    agent = DataIngestionAgent(data_dir=str(tmp_path))
    retriever = agent.setup_retrieval_chain()

    assert seen.get("allow_dangerous_deserialization") is True
    assert retriever is not None, "a loadable store must yield a retriever"


class _FakeStore:
    def as_retriever(self, **kwargs):
        return f"retriever({kwargs})"


def test_no_store_on_disk_still_returns_none(tmp_path, gateway):
    agent = DataIngestionAgent(data_dir=str(tmp_path))
    assert agent.setup_retrieval_chain() is None
