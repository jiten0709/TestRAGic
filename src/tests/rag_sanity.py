"""End-to-end RAG check. Not a pytest test -- run it directly:

    python src/tests/rag_sanity.py

Exercises the real pipeline classes, not a parallel reimplementation:

    1. embed   -- DataIngestionAgent.chunk_and_vectorize with local Qwen3 vectors
    2. retrieve -- setup_retrieval_chain, reloaded from disk, queried by category
    3. generate -- provider.chat over the retrieved chunks, through OmniRoute

Exits non-zero on the first stage that fails, so it works as a smoke gate. It
writes to src/data/vector_store/rag-sanity/, its own keyed store, so it never
disturbs a real video's vectors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.data_ingestion import DataIngestionAgent
from src.utils import config, embeddings as embeddings_mod, provider

# Distinct topics, so a retrieval that ignores the query is visibly wrong rather
# than accidentally right.
CHUNKS = [
    "Click the Sign In button in the top right corner, then enter your email address "
    "and password into the login form and press Continue.",
    "Open the Settings menu from the sidebar and select the Billing tab to review the "
    "current subscription plan and payment method.",
    "Upload a CSV file by dragging it onto the drop zone; a progress bar appears and "
    "the row count is shown when the import finishes.",
    "To delete an account, scroll to the Danger Zone at the bottom of the profile page "
    "and confirm by typing the account name.",
]
# A phrased question, not a bare noun: measured on this corpus, "authentication"
# separated the login chunk from an unrelated one by 0.002 cosine while this query
# separates them by 0.29. The app has the same problem and solves it the same way
# -- see TestGeneratorAgent._retrieval_query.
QUERY = "how does a user sign in to the application?"


def chunk(i, text):
    return {"chunk_id": i, "text": text, "start_time": i * 10, "end_time": i * 10 + 10,
            "actions": [], "chunk_type": "action"}


def main() -> int:
    config.load_environment()

    # ---- 1. embed --------------------------------------------------------
    agent = DataIngestionAgent()
    print(f"1. embedding with {agent.embeddings.model} "
          f"(requested device {agent.embeddings.device})")
    result = agent.chunk_and_vectorize(
        [chunk(i, t) for i, t in enumerate(CHUNKS)],
        store_key="rag-sanity", source="src/tests/rag_sanity.py",
    )
    if not result.get("success"):
        print(f"   FAIL: {result.get('error')}")
        return 1
    print(f"   OK: {result['documents_count']} chunks -> {result['embedding_dimensions']} dims, "
          f"{(result.get('embedding_attribution') or {}).get('display')}"
          f"{' [reused]' if result.get('reused') else ''}")

    meta = embeddings_mod.read_store_meta(Path(result["vector_store_path"]))
    print(f"   store meta: {meta}")

    # ---- 2. retrieve -----------------------------------------------------
    # Drop the in-memory store so the disk reload path is what gets exercised;
    # that branch silently returned None for the whole life of the project.
    agent.vector_store = None
    retriever = agent.setup_retrieval_chain("rag-sanity")
    if retriever is None:
        print("   FAIL: no retriever -- the store did not reload from disk")
        return 1
    docs = retriever.invoke(QUERY)
    if not docs:
        print(f"   FAIL: retrieval returned nothing for {QUERY!r}")
        return 1
    print(f"2. retrieved {len(docs)} chunks for {QUERY!r}; best match:")
    print(f"   {docs[0].page_content[:90]}...")
    if "Sign In" not in docs[0].page_content:
        print("   WARN: the top hit is not the login chunk -- retrieval looks untargeted")

    # ---- 3. generate -----------------------------------------------------
    if not provider.is_configured():
        print("3. SKIP: no gateway or API key configured")
        return 0
    context = "\n\n".join(d.page_content for d in docs)
    try:
        text, attribution = provider.chat(
            f"Using only this context, name the UI control a user clicks to start "
            f"signing in. Answer in under ten words.\n\nContext:\n{context}",
            max_tokens=400,
        )
    except provider.ProviderError as exc:
        print(f"3. FAIL: {exc}")
        return 1
    print(f"3. generated via {attribution.display} (source={attribution.source}, "
          f"{attribution.latency_ms}ms)")
    if not text.strip():
        print(f"   FAIL: empty response -- {attribution.note}")
        return 1
    print(f"   answer: {text.strip()[:160]}")
    if "sign in" not in text.lower():
        print("   WARN: the answer does not name the control from the retrieved chunk")

    print("\nRAG is live: local Qwen3 vectors -> FAISS retrieval -> OmniRoute generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
