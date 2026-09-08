"""Read a video's *picture* when its audio yields nothing.

The pipeline is otherwise transcript-only, so a silent screen recording -- the most
common way a QA engineer records a flow -- contributes nothing but the boilerplate
`_create_basic_transcript*` invents. This module is the second capture path:

    keyframes()            scene boundaries -> one frame per visual change
    ocr_frame()            on-screen text, locally, via RapidOCR (ONNX)
    caption_frame()        a vision model, for frames OCR cannot describe
    transcript_from_video() the three assembled into a transcript result

**It returns the shape `_extract_transcript` already returns** -- `segments` of
`{text, start, end, duration}` -- because that is the whole integration. Handing
`_intelligent_chunking` segments means it builds the six chunk keys
`chunk_and_vectorize` indexes with `[]`, and chunking, embedding, `store_key_for`,
`chunks_fingerprint`, store reuse and the width guards all keep working untouched.
OCR is another transcript *method*, not a second chunk stream.

Everything heavy (`cv2`, `scenedetect`, `rapidocr`) is imported inside a function:
`import rapidocr` alone costs 2.0s, and `data_ingestion` is already the app's
slowest import.

Measured on `src/assets/videos/demo-1.mp4` (3024x1898, 47.1s, no audio):

    scene detection, frame_skip=4    3.0s   (0.06x realtime; 14.3s without it)
    RapidOCR @1280px                 0.44s per frame, 27 lines, mean conf 0.955
    vision caption (oc/big-pickle)  13.1s per frame, ~570 prompt tokens

The ranking doc that motivated this quotes 50-150ms/frame for OCR and 10-20s for a
5-minute scene detection; both are optimistic by roughly 5x at screen-recording
resolutions. `frame_skip` is not optional.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.utils import config, provider
from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="utils.log")


class OCRUnavailable(RuntimeError):
    """No OCR is possible: the optional deps are missing, or the file is unreadable."""


def _env(name: str, default: str) -> str:
    """`FOO=` in a .env file means unset, not empty.

    `config.py`'s getters get this for free from their `or` chains; these are parsed
    at import, so without it an uncommented-but-blank knob raises ValueError and
    takes the whole OCR path down with it. A *garbage* value still raises, which is
    right -- that is a config error worth seeing.
    """
    return (os.getenv(name) or "").strip() or default


# ContentDetector's default threshold of 27 is tuned for film cuts. A UI walkthrough
# changes far less between frames -- measured on demo-1.mp4 the 95th-percentile
# frame delta is 7.4, and 27 finds 5 scenes across 47 seconds where 12 finds 11.
SCENE_THRESHOLD = float(_env("OCR_SCENE_THRESHOLD", "12"))
FRAME_SKIP = int(_env("OCR_FRAME_SKIP", "4"))
MAX_FRAMES = int(_env("OCR_MAX_FRAMES", "60"))
# One downscale feeds both steps: 1280px was the fastest OCR width that kept every
# useful line, and is the width the vision probe costed at ~570 prompt tokens.
FRAME_WIDTH = int(_env("OCR_FRAME_WIDTH", "1280"))
MIN_CONFIDENCE = float(_env("OCR_MIN_CONFIDENCE", "0.5"))

# Vision is the expensive half -- 13.1s and a network round-trip per frame -- so it
# is gated to frames OCR could not read, and hard-capped regardless.
VISION_MAX_FRAMES = int(_env("OCR_VISION_MAX_FRAMES", "8"))
VISION_MIN_LINES = int(_env("OCR_VISION_MIN_LINES", "3"))
VISION_MIN_CONFIDENCE = float(_env("OCR_VISION_MIN_CONFIDENCE", "0.7"))
VISION_WORKERS = int(_env("OCR_VISION_WORKERS", "4"))

VISION_PROMPT = (
    "This is one frame of a screen recording of a web application, captured for "
    "writing UI test cases. Describe what is on screen in two or three sentences: "
    "the screen or dialog it shows, and the interactive controls a tester could act "
    "on -- especially icon-only buttons with no text label. Name controls exactly as "
    "a tester would refer to them. Do not invent anything you cannot see."
)


# --------------------------------------------------------------------------
# keyframes
# --------------------------------------------------------------------------

def keyframes(video_path: str, *, threshold: float | None = None,
              frame_skip: int | None = None,
              max_frames: int | None = None) -> list[tuple[float, float, Any]]:
    """One downscaled frame per visual change: [(start_s, end_s, bgr_image)].

    Sampled at each scene's midpoint rather than its first frame -- a scene's first
    frames often catch the transition that created the boundary (a half-drawn menu,
    a fading dialog), which OCRs worse than the settled screen.

    A video with no detectable scene changes still gets sampled at a fixed interval,
    so a completely static recording is not silently reduced to nothing.
    """
    import cv2
    from scenedetect import ContentDetector, SceneManager, open_video

    threshold = SCENE_THRESHOLD if threshold is None else threshold
    frame_skip = FRAME_SKIP if frame_skip is None else frame_skip
    max_frames = MAX_FRAMES if max_frames is None else max_frames

    try:
        video = open_video(video_path)
        manager = SceneManager()
        manager.auto_downscale = True
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video, frame_skip=frame_skip, show_progress=False)
        scenes = [(a.seconds, b.seconds) for a, b in manager.get_scene_list()]
    except Exception as exc:
        raise OCRUnavailable(f"scene detection failed: {exc}") from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OCRUnavailable(f"could not open {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = (count / fps) if fps > 0 else 0.0

        if not scenes:
            # No cuts at all. Sample on a fixed grid instead of returning nothing.
            step = max(duration / max(max_frames, 1), 1.0) if duration else 1.0
            scenes = [(t, min(t + step, duration or t + step))
                      for t in _frange(0.0, duration or step, step)]

        scenes = _thin(scenes, max_frames)

        frames = []
        for start, end in scenes:
            capture.set(cv2.CAP_PROP_POS_MSEC, ((start + end) / 2.0) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append((start, end, _downscale(frame)))
    finally:
        capture.release()

    if not frames:
        raise OCRUnavailable(f"no frames could be read from {video_path}")
    logger.info("keyframes: %d scenes -> %d frames (threshold=%s skip=%s)",
                len(scenes), len(frames), threshold, frame_skip)
    return frames


def _frange(start: float, stop: float, step: float) -> list[float]:
    out, value = [], start
    while value < stop:
        out.append(value)
        value += step
    return out


def _thin(scenes: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    """Cap the frame count while keeping coverage spread across the whole video."""
    if limit <= 0 or len(scenes) <= limit:
        return scenes
    stride = len(scenes) / limit
    return [scenes[int(i * stride)] for i in range(limit)]


def _downscale(frame):
    import cv2

    height, width = frame.shape[:2]
    if width <= FRAME_WIDTH:
        return frame
    return cv2.resize(frame, (FRAME_WIDTH, int(height * FRAME_WIDTH / width)),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

_engine = None


def engine():
    """The RapidOCR engine, built once.

    Streamlit re-runs the whole script per interaction and constructing this costs
    ~0.9s, so it is cached module-level for the same reason `embeddings.py` caches
    its model. The ONNX weights ship inside the `rapidocr` wheel -- no download.
    """
    global _engine
    if _engine is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OCRUnavailable(
                "rapidocr is not installed; run `pip install rapidocr scenedetect`"
            ) from exc
        _engine = RapidOCR()
    return _engine


def ocr_frame(frame) -> list[tuple[str, float]]:
    """On-screen text as [(line, confidence)], lowest-value lines dropped.

    RapidOCR reliably emits single glyphs for icons and window chrome ("<-", "C",
    "@"), which carry no meaning for a test-case corpus but do dilute an embedding.
    A line must be at least two characters and clear `MIN_CONFIDENCE` to survive.
    """
    result = engine()(frame)
    texts, scores = result.txts or (), result.scores or ()
    lines = []
    for text, score in zip(texts, scores):
        text = str(text).strip()
        if len(text) >= 2 and float(score) >= MIN_CONFIDENCE:
            lines.append((text, float(score)))
    return lines


# --------------------------------------------------------------------------
# vision
# --------------------------------------------------------------------------

def vision_model() -> str | None:
    """The pinned vision model, or None when vision is switched off.

    There is deliberately no default. The gateway lists `auto/best-vision`,
    `auto/vision` and `auto/pro-vision`, and on the instance this was built against
    every one of them -- along with `auto/best-chat` -- answers 502
    ("Cloudflare Playground browser session failed"), while a directly-named model
    served the same image in 13.1s. Guessing a route that 502s would spend the whole
    fallback budget to produce nothing, so vision stays off until a model is named.
    """
    return config.get_vision_model() or None


def caption_frame(frame, model: str | None = None) -> tuple[str, Any]:
    """Describe one frame through the gateway. Returns ("", None) if it could not.

    Goes through `provider.chat` like every other completion -- vision is not a
    second call path -- so the attribution, the English-only system prompt and the
    error classification are all inherited.
    """
    import cv2

    model = model or vision_model()
    if not model:
        return "", None
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return "", None
    try:
        text, attribution = provider.chat(
            VISION_PROMPT, model=model, max_tokens=500, images=[buffer.tobytes()]
        )
    except Exception as exc:
        logger.warning("vision caption failed on %s: %s", model, exc)
        raise
    return text.strip(), attribution


# --------------------------------------------------------------------------
# the assembled transcript
# --------------------------------------------------------------------------

def transcript_from_video(video_path: str, use_vision: bool = True) -> dict:
    """Read the video's picture into the transcript shape the pipeline already uses.

    Raises OCRUnavailable when nothing readable came out, so the caller can fall
    through to its existing last resort rather than embedding an empty store.
    """
    frames = keyframes(video_path)
    model = vision_model() if use_vision else None

    # Pass 1 -- OCR every keyframe, and note which ones OCR could not read.
    # ponytail: every keyframe is held decoded at once -- ~3MB each at 1280px, so
    # OCR_MAX_FRAMES=60 is a ~185MB ceiling. Stream frame-by-frame if that bites.
    read: list[dict] = []
    for start, end, frame in frames:
        lines = ocr_frame(frame)
        mean = sum(score for _, score in lines) / len(lines) if lines else 0.0
        read.append({
            "start": start, "end": end, "frame": frame, "lines": lines,
            # A frame OCR could barely read is the one a vision model is for: an
            # icon-only toolbar, a canvas, a chart. Judged on the raw yield, before
            # de-duplication -- a frame that is merely *repetitive* is not worth a
            # 13-second call, it is worth skipping.
            "low_yield": len(lines) < VISION_MIN_LINES or mean < VISION_MIN_CONFIDENCE,
        })

    # Pass 2 -- caption only those, capped, and concurrently: the calls are
    # independent and I/O-bound, and 8 sequential ones would stall the UI for ~100s.
    captions: dict[int, str] = {}
    attribution = None
    vision_error = None
    candidates = [i for i, item in enumerate(read) if item["low_yield"]][:VISION_MAX_FRAMES]
    if model and candidates:
        with ThreadPoolExecutor(max_workers=VISION_WORKERS) as pool:
            futures = {pool.submit(caption_frame, read[i]["frame"], model): i
                       for i in candidates}
            for future, index in futures.items():
                try:
                    text, attr = future.result()
                except Exception as exc:
                    vision_error = vision_error or f"{type(exc).__name__}: {exc}"
                    continue
                if text:
                    captions[index] = text
                    attribution = attribution or attr

    # Assemble. A line already seen earlier is dropped: consecutive keyframes of a
    # UI repeat the whole browser chrome and navigation bar, and embedding those
    # forty labels sixty times would let them dominate every retrieval.
    seen: set[str] = set()
    segments, ocr_frames = [], 0
    for index, item in enumerate(read):
        fresh = [text for text, _ in item["lines"] if text not in seen]
        seen.update(fresh)
        caption = captions.get(index, "")
        if not fresh and not caption:
            continue  # nothing this frame shows is new
        if fresh:
            ocr_frames += 1
        body = "\n".join(part for part in ("\n".join(fresh), caption) if part)
        segments.append({
            "text": body,
            "start": float(item["start"]),
            "end": float(item["end"]),
            "duration": float(item["end"] - item["start"]),
        })

    if not segments:
        raise OCRUnavailable("no on-screen text or captions could be read")

    method = "ocr+vision" if captions else "ocr"
    logger.info("%s: %d keyframes -> %d segments (%d captioned)",
                method, len(frames), len(segments), len(captions))
    return {
        "success": True,
        "transcript": "\n\n".join(segment["text"] for segment in segments),
        "segments": segments,
        "method": method,
        "reason": None,
        "cause": None,
        "ocr": {
            "keyframes": len(frames),
            "segments": len(segments),
            "ocr_frames": ocr_frames,
            "vision_frames": len(captions),
            "vision_model": model,
            "vision_error": vision_error,
            "vision_attribution": attribution.to_dict() if attribution else None,
        },
    }
