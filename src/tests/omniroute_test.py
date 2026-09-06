"""Smoke test for the OmniRoute gateway. Not a pytest test — run it directly:

    python src/tests/omniroute_test.py

Sends one real chat query through the provider abstraction and prints what
actually answered. Exits non-zero if the gateway/API is unreachable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config, provider

QUERY = "What is RAG? Answer in one sentence."


def main() -> int:
    config.load_environment()
    ep = provider.resolve_endpoint()
    print(f"mode={ep.gateway}  base_url={ep.base_url}")

    if not provider.is_configured():
        print("NOT CONFIGURED: set OMNIROUTE_BASE_URL or OPENAI_API_KEY")
        return 2

    try:
        text, attr = provider.chat(QUERY, max_tokens=500)
    except provider.ProviderError as exc:
        print(f"FAIL: every attempt failed -> {exc}")
        return 1

    print(f"answered by: {attr.display or provider.pretty(attr.provider, attr.model)}"
          f"  (source={attr.source}, latency={attr.latency_ms}ms)")
    if attr.fallback:
        print(f"  fallback used: {attr.note}")
    if attr.rerouted:
        print("  gateway rerouted the request")
    print(f"\nQ: {QUERY}\nA: {text.strip() or '(empty response)'}")

    return 0 if text.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
