"""End-to-end check of the OCR path. Not a pytest test -- run it directly:

    python src/tests/ocr_sanity.py

Exercises the real pipeline classes against a real silent video, which is the one
thing the unit tests deliberately cannot do: they stub every model to stay in the
offline CI gate, so nothing there proves RapidOCR can read a screen recording.

    1. keyframes  -- PySceneDetect over src/assets/videos/demo-1.mp4
    2. read       -- RapidOCR, and the vision captioner if one is configured
    3. ingest     -- the whole DataIngestionAgent path, chunked and embedded

demo-1.mp4 is a silent (no audio stream) capture of the nopCommerce admin demo, so
the controls it must name are known rather than guessed. Exits non-zero on the
first stage that fails. Writes to src/data/vector_store/demo-1/, the same store a
real upload of that file would use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.data_ingestion import DataIngestionAgent, OCR_METHODS
from src.utils import config, video_ocr

VIDEO = "src/assets/videos/demo-1.mp4"

# What is actually on screen in demo-1.mp4 -- a login form and the dashboard behind
# it. If OCR regresses, it regresses on these.
EXPECTED = ["LOG IN", "Email", "Password", "Remember me", "Dashboard"]


def main() -> int:
    config.load_environment()

    if not Path(VIDEO).exists():
        print(f"FAIL: {VIDEO} not found")
        return 1

    # ---- 1. keyframes ----------------------------------------------------
    try:
        frames = video_ocr.keyframes(VIDEO)
    except video_ocr.OCRUnavailable as exc:
        print(f"1. FAIL: {exc}")
        return 1
    print(f"1. {len(frames)} keyframes at threshold {video_ocr.SCENE_THRESHOLD} "
          f"(frame_skip {video_ocr.FRAME_SKIP})")
    if len(frames) < 2:
        print("   FAIL: a 47-second walkthrough should yield more than one keyframe")
        return 1

    # ---- 2. read ---------------------------------------------------------
    model = video_ocr.vision_model()
    print(f"2. reading{f' + captioning via {model}' if model else ' (vision off: '
                                                                 'set OMNIROUTE_VISION_MODEL)'}")
    try:
        read = video_ocr.transcript_from_video(VIDEO)
    except video_ocr.OCRUnavailable as exc:
        print(f"   FAIL: {exc}")
        return 1
    info = read["ocr"]
    print(f"   method={read['method']} segments={info['segments']} "
          f"captioned={info['vision_frames']} chars={len(read['transcript'])}")
    if info["vision_error"]:
        print(f"   WARN: vision failed: {info['vision_error']}")

    missing = [w for w in EXPECTED if w.lower() not in read["transcript"].lower()]
    if missing:
        print(f"   FAIL: OCR did not read {missing}")
        return 1
    print(f"   OK: read every expected control {EXPECTED}")

    # ---- 3. ingest -------------------------------------------------------
    agent = DataIngestionAgent()
    result = agent.process_video_file(VIDEO)
    if not result.get("success"):
        print(f"3. FAIL: {result.get('error')}")
        return 1
    method = result.get("transcript_method")
    print(f"3. ingested: method={method} chunks={result['chunks_count']} "
          f"store={result.get('vector_store_key')}")
    if method not in OCR_METHODS:
        print(f"   FAIL: a silent video must not fall through to {method!r}")
        return 1

    named = [c for c in result["chunks"] if "log in" in c["text"].lower()]
    if not named:
        print("   FAIL: no chunk names a control from the login screen")
        return 1
    print(f"   OK: {len(named)} chunk(s) name on-screen controls")

    retriever = agent.setup_retrieval_chain(result.get("vector_store_key"))
    if retriever is None:
        print("   FAIL: no retriever -- the store did not build")
        return 1
    docs = retriever.invoke("how does a user sign in to the admin area?")
    print(f"   retrieved {len(docs)} chunks; top: {docs[0].page_content[:90]!r}")

    print("\nThe picture is readable: keyframes -> OCR -> chunks -> FAISS retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
