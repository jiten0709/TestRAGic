"""The RAG wiring, which used to fail invisibly.

Three defects are covered here, all of the same family: a stage degrades, every
later stage succeeds on the degraded input, and the run reports success.

1. ``wire_retriever`` warns when there is no retriever. Before, ``set_retriever(None)``
   was silent and the generator quietly prompted every category with
   ``transcript[:2000]`` while the run still reported success.
2. ``setup_retrieval_chain`` can actually load a store from disk. The cold-start
   ``FAISS.load_local`` call omitted ``allow_dangerous_deserialization=True``,
   which langchain-community 0.3.27 requires, so it always raised into a bare
   ``except`` and returned ``None``.
3. ``warn_if_synthetic_transcript`` warns when the transcript is placeholder text
   rather than the video. A silent video, a failed download or a missing ffmpeg all
   land there, and nothing else catches it -- a store *is* built, so ``wire_retriever``
   stays quiet, and the model writes confident tests about a video it never saw.
"""

import pytest

from src.agents import data_ingestion
from src.agents.data_ingestion import DataIngestionAgent
from src.dashboard.app import warn_if_synthetic_transcript, wire_retriever


class FakeGenerator:
    def __init__(self):
        self.retriever = "unset"

    def set_retriever(self, retriever):
        self.retriever = retriever


class FakeIngestion:
    def __init__(self, retriever):
        self._retriever = retriever
        self.asked_for = "unset"

    def setup_retrieval_chain(self, store_key=None):
        self.asked_for = store_key
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


def test_the_run_names_the_store_it_wants(monkeypatch):
    """Two videos have two stores; the run must not be served whichever one the
    agent happens to hold in memory."""
    monkeypatch.setattr("src.dashboard.app.st.warning", lambda *a, **k: None)

    ingestion = FakeIngestion("a-retriever")
    wire_retriever(FakeGenerator(), ingestion,
                   {"vector_store_key": "demo-2", "vector_store_info": {"success": True}})

    assert ingestion.asked_for == "demo-2"


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
    store = tmp_path / "vector_store" / "demo-1"
    store.mkdir(parents=True)
    (store / "index.faiss").write_bytes(b"")

    seen = {}

    def fake_load_local(path, embeddings, **kwargs):
        seen.update(kwargs)
        return _FakeStore()

    monkeypatch.setattr("src.agents.data_ingestion.FAISS.load_local", staticmethod(fake_load_local))

    agent = DataIngestionAgent(data_dir=str(tmp_path))
    retriever = agent.setup_retrieval_chain("demo-1")

    assert seen.get("allow_dangerous_deserialization") is True
    assert retriever is not None, "a loadable store must yield a retriever"


class _FakeStore:
    def as_retriever(self, **kwargs):
        return f"retriever({kwargs})"


def test_no_store_on_disk_still_returns_none(tmp_path, gateway):
    agent = DataIngestionAgent(data_dir=str(tmp_path))
    assert agent.setup_retrieval_chain() is None


# --------------------------------------------------------------------------
# 3. a placeholder transcript is reported, not passed off as the video
# --------------------------------------------------------------------------

@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr("src.dashboard.app.st.warning", seen.append)
    return seen


@pytest.mark.parametrize("method", ["basic_fallback", "basic_fallback_file"])
def test_a_synthetic_transcript_warns_on_both_paths(warnings, method):
    warn_if_synthetic_transcript(
        {"transcript_method": method, "transcript_reason": "Whisper: boom"}
    )

    assert len(warnings) == 1
    assert "placeholder" in warnings[0]
    assert "Whisper: boom" in warnings[0], "the real reason must survive"


def test_a_real_transcript_says_nothing(warnings):
    warn_if_synthetic_transcript({"transcript_method": "whisper_direct"})
    warn_if_synthetic_transcript({"transcript_method": "youtube_api"})
    assert warnings == [], "a healthy run must not nag"


def test_a_missing_method_is_not_treated_as_synthetic(warnings):
    """Mock data and older saved runs carry no method; do not cry wolf."""
    warn_if_synthetic_transcript({})
    assert warnings == []


def test_a_silent_video_is_named_as_the_cause(warnings):
    """The ffmpeg stderr for a video with no audio track is opaque on its own."""
    warn_if_synthetic_transcript({
        "transcript_method": "basic_fallback_file",
        "transcript_reason": "Whisper: Failed to load audio: ffmpeg ... Invalid argument",
        "transcript_cause": data_ingestion.NO_AUDIO,
    })

    assert "no audio track" in warnings[0]
    # Reaching the placeholder for a silent video now means OCR failed too -- the
    # hint has to say that, not the old "the pipeline never looks at the picture".
    assert "OCR" in warnings[0]


def test_a_missing_ffmpeg_is_named_as_the_cause(warnings):
    warn_if_synthetic_transcript({
        "transcript_method": "basic_fallback_file",
        "transcript_reason": "Whisper: [Errno 2] No such file or directory: 'ffmpeg'",
        "transcript_cause": data_ingestion.FFMPEG_MISSING,
    })

    assert "brew install ffmpeg" in warnings[0]


def test_the_cause_is_classified_on_the_untrimmed_text(warnings):
    """_brief() drops the decisive middle line, so classification must happen before
    trimming -- this is the bug that made the no-audio hint say 'install ffmpeg'."""
    from src.agents.data_ingestion import _brief, _classify_transcript_failure

    ffmpeg_stderr = (
        "Failed to load audio: ffmpeg version 9.0.1\n"
        + "\n".join(f"  configure flag {i}" for i in range(25))
        + "\nOutput file does not contain any stream"
        + "\nError opening output file -.\nError opening output files: Invalid argument"
    )

    assert _classify_transcript_failure(ffmpeg_stderr) == data_ingestion.NO_AUDIO
    assert "does not contain any stream" not in _brief(ffmpeg_stderr), (
        "the decisive line really is trimmed away -- hence the separate code"
    )
    assert _classify_transcript_failure(
        "[Errno 2] No such file or directory: 'ffmpeg'") == data_ingestion.FFMPEG_MISSING
    assert _classify_transcript_failure("some other failure") is None


def test_a_reason_is_bounded_before_it_reaches_the_ui():
    """ffmpeg prints its whole build banner and puts the real message last, so the
    naive str(exc) buries the cause under 25 lines of configure flags."""
    from src.agents.data_ingestion import _brief

    ffmpeg_style = (
        "Failed to load audio: ffmpeg version 9.0.1\n"
        + "\n".join(f"  configure flag {i}" for i in range(25))
        + "\nOutput file does not contain any stream\nError opening output file -."
    )
    brief = _brief(ffmpeg_style)

    assert len(brief) <= 240
    assert "Failed to load audio" in brief, "what failed"
    assert "does not contain any stream" in brief, "why -- the last lines carry it"
    assert "configure flag 12" not in brief

    assert _brief("plain message") == "plain message", "short reasons pass through"
    assert _brief("") == ""
