"""Reading the picture: keyframe -> text -> the transcript shape the pipeline uses.

Pure Python. `cv2`, `scenedetect` and `rapidocr` are all imported *inside*
`video_ocr`'s functions, and every one of them is stubbed here, so this file stays
in the CI gate: no browser, no network, no ONNX weights, no 2-second imports.

What is worth pinning is the arithmetic around the models, not the models:
which frames earn a 13-second vision call, what de-duplication drops, and that the
result still satisfies the six keys `chunk_and_vectorize` indexes with `[]`.
"""

import pytest

from src.agents.data_ingestion import DataIngestionAgent, _merge_segments
from src.utils import video_ocr


@pytest.fixture(autouse=True)
def no_vision_by_default(monkeypatch):
    """Vision is off unless a test names a model, exactly as in production."""
    monkeypatch.delenv("OMNIROUTE_VISION_MODEL", raising=False)
    monkeypatch.setattr(video_ocr.config, "get_vision_model", lambda: None)


def stub_frames(monkeypatch, per_frame_lines, captions=None):
    """Drive `transcript_from_video` off a script instead of a video file.

    `per_frame_lines` is one [(text, confidence), ...] list per keyframe; the frame
    objects themselves are opaque indices, since nothing under test looks at pixels.
    """
    frames = [(float(i * 10), float(i * 10 + 10), i) for i in range(len(per_frame_lines))]
    monkeypatch.setattr(video_ocr, "keyframes", lambda *a, **k: frames)
    monkeypatch.setattr(video_ocr, "ocr_frame", lambda frame: per_frame_lines[frame])
    captioned = []

    def caption(frame, model=None):
        captioned.append(frame)
        return (captions or {}).get(frame, ""), None

    monkeypatch.setattr(video_ocr, "caption_frame", caption)
    return captioned


# --------------------------------------------------------------------------
# the transcript shape -- the whole reason OCR integrates at `segments`
# --------------------------------------------------------------------------

def test_segments_survive_intelligent_chunking_with_every_required_key(monkeypatch):
    """`chunk_and_vectorize` reads six keys with [] -- a missing one is a KeyError.

    This is the contract that lets OCR reuse the entire downstream pipeline, so it
    is asserted against the real chunker rather than a restatement of the shape.
    """
    stub_frames(monkeypatch, [
        [("Welcome, please sign in!", 0.99), ("Email:", 0.99), ("LOG IN", 0.98)],
        [("Dashboard", 0.99), ("click Save", 0.97), ("Latest Orders", 0.95)],
    ])
    result = video_ocr.transcript_from_video("ignored.mp4")

    chunks = DataIngestionAgent.__new__(DataIngestionAgent)
    chunks.action_keywords = ["click", "button", "login"]
    produced = DataIngestionAgent._intelligent_chunking(chunks, result)

    assert produced
    for chunk in produced:
        for key in ("chunk_id", "text", "start_time", "end_time", "actions", "chunk_type"):
            assert key in chunk, f"{key} missing -- chunk_and_vectorize would KeyError"


def test_result_matches_the_transcript_contract(monkeypatch):
    stub_frames(monkeypatch, [[("Search", 0.9), ("Checkout", 0.9)]])
    result = video_ocr.transcript_from_video("ignored.mp4")
    assert result["success"] is True
    assert result["method"] == "ocr"
    assert result["cause"] is None
    assert "Checkout" in result["transcript"]
    segment, = result["segments"]
    assert set(segment) == {"text", "start", "end", "duration"}
    assert segment["duration"] == segment["end"] - segment["start"]


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------

def test_repeated_chrome_is_read_once(monkeypatch):
    """Every keyframe of a browser repeats the nav bar; embedding it N times would
    let it outweigh the content that actually differs between screens."""
    chrome = [("Microsoft Teams", 0.99), ("All Bookmarks", 0.99)]
    stub_frames(monkeypatch, [
        chrome + [("Sign in", 0.99)],
        chrome + [("Dashboard", 0.99)],
    ])
    result = video_ocr.transcript_from_video("ignored.mp4")
    assert result["transcript"].count("All Bookmarks") == 1
    assert "Sign in" in result["segments"][0]["text"]
    assert "Dashboard" in result["segments"][1]["text"]


def test_a_frame_showing_nothing_new_is_dropped_entirely(monkeypatch):
    lines = [("Dashboard", 0.99), ("Orders", 0.99), ("Customers", 0.99)]
    stub_frames(monkeypatch, [lines, lines, lines])
    result = video_ocr.transcript_from_video("ignored.mp4")
    assert len(result["segments"]) == 1
    assert result["ocr"]["keyframes"] == 3


def test_nothing_readable_raises_rather_than_returning_an_empty_transcript(monkeypatch):
    """An empty store that reports success is the failure mode this whole path
    exists to remove -- the caller must be free to fall through to its placeholder."""
    stub_frames(monkeypatch, [[], []])
    with pytest.raises(video_ocr.OCRUnavailable):
        video_ocr.transcript_from_video("ignored.mp4")


# --------------------------------------------------------------------------
# the vision gate -- the expensive half
# --------------------------------------------------------------------------

def test_vision_is_off_without_a_configured_model(monkeypatch):
    captioned = stub_frames(monkeypatch, [[("x", 0.9)]], captions={0: "described"})
    result = video_ocr.transcript_from_video("ignored.mp4")
    assert captioned == []
    assert result["method"] == "ocr"
    assert result["ocr"]["vision_frames"] == 0


def test_only_frames_ocr_could_not_read_are_captioned(monkeypatch):
    monkeypatch.setattr(video_ocr.config, "get_vision_model", lambda: "vision-1")
    captioned = stub_frames(
        monkeypatch,
        [
            [("Email:", 0.99), ("Password:", 0.99), ("LOG IN", 0.99)],  # text-rich
            [("+", 0.99)],                                             # icon-only
        ],
        captions={1: "A toolbar of icon-only buttons."},
    )
    result = video_ocr.transcript_from_video("ignored.mp4")
    assert captioned == [1]
    assert result["method"] == "ocr+vision"
    assert result["ocr"]["vision_frames"] == 1
    assert "icon-only buttons" in result["segments"][1]["text"]


def test_a_merely_repetitive_frame_is_not_worth_a_vision_call(monkeypatch):
    """Judged on raw yield, before de-duplication. A frame whose text is all
    duplicates reads as empty *after* dedup, and gating on that would spend 13
    seconds describing a screen already covered."""
    monkeypatch.setattr(video_ocr.config, "get_vision_model", lambda: "vision-1")
    lines = [("Dashboard", 0.99), ("Orders", 0.99), ("Customers", 0.99)]
    captioned = stub_frames(monkeypatch, [lines, lines])
    video_ocr.transcript_from_video("ignored.mp4")
    assert captioned == []


def test_vision_calls_are_hard_capped(monkeypatch):
    monkeypatch.setattr(video_ocr.config, "get_vision_model", lambda: "vision-1")
    monkeypatch.setattr(video_ocr, "VISION_MAX_FRAMES", 2)
    blank = [[("%d" % i, 0.99)] for i in range(6)]  # every frame is low-yield and distinct
    captioned = stub_frames(monkeypatch, blank)
    video_ocr.transcript_from_video("ignored.mp4")
    assert len(captioned) == 2


def test_a_failing_vision_model_degrades_to_ocr_and_says_so(monkeypatch):
    monkeypatch.setattr(video_ocr.config, "get_vision_model", lambda: "vision-1")
    stub_frames(monkeypatch, [[("Email:", 0.99), ("LOG IN", 0.99), ("Password:", 0.99)],
                              [("+", 0.99)]])
    monkeypatch.setattr(video_ocr, "caption_frame",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("502 gateway")))
    result = video_ocr.transcript_from_video("ignored.mp4")
    assert result["method"] == "ocr"          # the OCR half still landed
    assert "502 gateway" in result["ocr"]["vision_error"]


# --------------------------------------------------------------------------
# where OCR sits between the real methods and the placeholder
# --------------------------------------------------------------------------

def agent():
    return DataIngestionAgent.__new__(DataIngestionAgent)


def test_ocr_is_used_when_nothing_was_heard(monkeypatch):
    read = {"success": True, "method": "ocr", "segments": [{"text": "LOG IN", "start": 0, "end": 1}]}
    monkeypatch.setattr(DataIngestionAgent, "_ocr_transcript", lambda self, p, r: read)
    result = agent()._with_ocr(None, "v.mp4", False, [], None,
                               lambda reason, why: {"method": "basic_fallback"})
    assert result is read


def test_the_placeholder_still_wins_when_ocr_finds_nothing_either(monkeypatch):
    monkeypatch.setattr(DataIngestionAgent, "_ocr_transcript",
                        lambda self, p, r: r.append("OCR: no text") or None)
    result = agent()._with_ocr(None, "v.mp4", False, ["Whisper: no audio"], "no_audio",
                               lambda reason, why: {"method": "basic_fallback",
                                                    "reason": reason, "cause": why})
    assert result["method"] == "basic_fallback"
    assert "Whisper: no audio" in result["reason"] and "OCR: no text" in result["reason"]
    assert result["cause"] == "no_audio"


def test_speech_is_not_second_guessed_unless_ocr_is_asked_for(monkeypatch):
    monkeypatch.setattr(DataIngestionAgent, "_ocr_transcript",
                        lambda self, p, r: pytest.fail("OCR must not run"))
    spoken = {"success": True, "method": "whisper", "segments": []}
    assert agent()._with_ocr(spoken, "v.mp4", False, [], None, lambda *a: None) is spoken


def test_forcing_ocr_merges_both_streams_in_time_order(monkeypatch):
    read = {"success": True, "method": "ocr",
            "segments": [{"text": "LOG IN", "start": 5.0, "end": 6.0}],
            "ocr": {"keyframes": 1}}
    monkeypatch.setattr(DataIngestionAgent, "_ocr_transcript", lambda self, p, r: read)
    spoken = {"success": True, "method": "whisper",
              "segments": [{"text": "first", "start": 0.0, "end": 1.0},
                           {"text": "last", "start": 9.0, "end": 10.0}]}
    result = agent()._with_ocr(spoken, "v.mp4", True, [], None, lambda *a: None)

    assert result["method"] == "whisper+ocr"
    assert [s["text"] for s in result["segments"]] == ["first", "LOG IN", "last"]
    # Rebuilt, not carried over: `transcript` is the generator's no-retriever
    # fallback, so what was read has to actually be in it.
    assert "LOG IN" in result["transcript"]
    assert result["ocr"] == {"keyframes": 1}


def test_no_local_file_is_reported_rather_than_ignored():
    """A checkbox that silently does nothing is the class of failure phase 5 removed."""
    reasons = []
    assert agent()._ocr_transcript(None, reasons) is None
    assert reasons == ["OCR: no local video file to read"]


def test_merge_orders_by_start_time():
    spoken = [{"text": "b", "start": 5}, {"text": "d", "start": 30}]
    seen = [{"text": "a", "start": 1}, {"text": "c", "start": 10}]
    assert [s["text"] for s in _merge_segments(spoken, seen)] == ["a", "b", "c", "d"]
    assert [s["text"] for s in _merge_segments(None, seen)] == ["a", "c"]


def test_an_unfulfilled_ocr_request_is_reported_rather_than_dropped(monkeypatch):
    """Ticking the box on a URL whose video was never downloaded must not look
    like it worked -- there is nothing to read, and the run has to say so."""
    monkeypatch.setattr(DataIngestionAgent, "_ocr_transcript",
                        lambda self, p, r: r.append("OCR: no local video file to read") or None)
    spoken = {"success": True, "method": "youtube_api", "segments": []}
    result = agent()._with_ocr(spoken, None, True, [], None, lambda *a: None)

    assert result["method"] == "youtube_api"
    assert result["ocr"]["error"] == "OCR: no local video file to read"


def test_speech_carries_no_ocr_block_when_ocr_was_never_asked_for():
    spoken = {"success": True, "method": "whisper", "segments": []}
    assert "ocr" not in agent()._with_ocr(spoken, "v.mp4", False, [], None, lambda *a: None)
