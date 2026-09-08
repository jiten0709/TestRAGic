"""Who answered, how confidently we know it, and whether it survives to disk."""

import json

import pytest

from src.agents.test_generator import TestGeneratorAgent
from src.dashboard import app
from src.utils import provider


# --------------------------------------------------------------------------
# deriving attribution from a response
# --------------------------------------------------------------------------

def test_headers_are_authoritative(gateway):
    gateway.default_chat = gateway.chat_ok("google", "gemini-2.5-pro", latency_ms=142)
    _, attribution = provider.chat("hi", model="auto")

    assert attribution.source == "headers"
    assert (attribution.provider, attribution.model) == ("google", "gemini-2.5-pro")
    assert attribution.display == "Google — gemini-2.5-pro"
    assert attribution.latency_ms == 142
    assert attribution.request_id == "req-1"


def test_body_model_is_used_when_headers_are_stripped(gateway):
    """A reverse proxy that drops X-* must degrade, not lie."""
    gateway.default_chat = gateway.chat_ok(model="gpt-4o-mini", headers=False)
    _, attribution = provider.chat("hi", model="gpt-4o-mini")

    assert attribution.source == "body"
    assert attribution.model == "gpt-4o-mini"
    assert attribution.provider is None


def test_no_headers_and_no_body_model_is_labelled_unknown(gateway):
    gateway.default_chat = gateway.chat_ok(model="gpt-4o-mini", headers=False, body_model="")
    _, attribution = provider.chat("hi", model="gpt-4o-mini")

    assert attribution.source == "unknown"
    assert "requested; gateway did not report" in attribution.display
    assert attribution.rerouted is False, "cannot claim a reroute we cannot see"


def test_serving_a_different_model_is_a_reroute_not_a_fallback(gateway):
    gateway.default_chat = gateway.chat_ok("google", "gemini-2.5-pro")
    _, attribution = provider.chat("hi", model="auto")

    assert attribution.rerouted is True
    assert attribution.fallback is False, "the gateway rerouting is not our fallback"
    assert "routed by OmniRoute from 'auto'" in attribution.note


def test_serving_the_requested_model_is_not_a_reroute(gateway):
    gateway.default_chat = gateway.chat_ok("openai", "gpt-4o-mini")
    _, attribution = provider.chat("hi", model="gpt-4o-mini")

    assert attribution.rerouted is False
    assert attribution.note is None


def test_provider_qualified_request_is_not_a_reroute(gateway):
    gateway.chat_by_model["google/gemini-2.5-pro"] = gateway.chat_ok("google", "gemini-2.5-pro")
    _, attribution = provider.chat("hi", model="google/gemini-2.5-pro")
    assert attribution.rerouted is False


def test_mock_and_exhausted_output_carries_no_model(gateway):
    """Output no model produced must never look attributed."""
    attribution = provider.unattributed("mock data; no model was called")

    assert attribution.requested_model == ""
    assert attribution.model is None
    assert "no model was called" in attribution.display


# --------------------------------------------------------------------------
# summarising several categories
# --------------------------------------------------------------------------

def test_one_model_summarises_to_that_model(gateway):
    _, attribution = provider.chat("hi", model="gpt-4o-mini")
    summary = provider.summarize([("ui", attribution)])

    assert summary["model_count"] == 1
    assert summary["display"] == attribution.display
    assert [c["category"] for c in summary["per_category"]] == ["ui"]


def test_heterogeneous_categories_summarise_to_a_count(gateway):
    gateway.chat_by_model["a"] = gateway.chat_ok("google", "gemini-2.5-pro")
    gateway.chat_by_model["b"] = gateway.chat_ok("openai", "gpt-4o-mini")
    _, first = provider.chat("hi", model="a")
    _, second = provider.chat("hi", model="b")

    summary = provider.summarize([("ui", first), ("edge", second)])

    assert summary["model_count"] == 2
    assert summary["display"] == "2 models"
    assert len(summary["per_category"]) == 2
    assert {c["provider"] for c in summary["per_category"]} == {"google", "openai"}


def test_summary_flags_a_mixed_run_with_a_fallback(gateway):
    gateway.chat_by_model["dead"] = gateway.status(404)
    gateway.chat_by_model["gpt-4o-mini"] = gateway.chat_ok("openai", "gpt-4o-mini")
    gateway.chat_by_model["b"] = gateway.chat_ok("google", "gemini-2.5-pro")

    _, fell_back = provider.chat("hi", model="dead")
    _, direct = provider.chat("hi", model="b")

    summary = provider.summarize([("ui", fell_back), ("edge", direct)])
    assert summary["fallback"] is True
    assert "1 fallback" in summary["display"]


def test_summary_of_nothing_is_explicitly_unattributed():
    summary = provider.summarize([])
    assert summary["requested_model"] == ""
    assert summary["model"] is None


# --------------------------------------------------------------------------
# persistence: generate -> format -> save -> reload
# --------------------------------------------------------------------------

def test_attribution_round_trips_to_disk(gateway, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    gateway.default_chat = gateway.chat_ok(
        "google", "gemini-2.5-pro",
        content='[{"title": "Login works", "steps": ["open", "submit"]}]',
    )

    agent = TestGeneratorAgent(model="auto")
    cases = agent.generate_comprehensive_tests(
        {"transcript": "click sign up", "video_info": {"title": "Demo"}}, ["ui"], ["high"]
    )
    formatted = agent.format_test_cases(cases)

    saved = app.save_generated_tests_to_file(formatted, {"title": "Demo"})
    reloaded = json.loads((tmp_path / saved).read_text(encoding="utf-8"))
    attribution = reloaded["metadata"]["attribution"]

    assert reloaded["metadata"]["model"] == "auto", "metadata.model keeps meaning 'requested'"
    assert attribution["provider"] == "google"
    assert attribution["model"] == "gemini-2.5-pro"
    assert attribution["rerouted"] is True
    assert attribution["per_category"][0]["category"] == "ui"


def test_results_page_reads_attribution_back_from_a_file(gateway, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "src" / "data" / "test_cases"
    target.mkdir(parents=True)
    (target / "old.json").write_text(json.dumps(
        {"test_cases": [], "metadata": {"attribution": {"display": "Google — gemini-2.5-pro",
                                                        "requested_model": "auto"}}}
    ), encoding="utf-8")

    metadata = app.latest_test_case_metadata()
    assert metadata["attribution"]["display"] == "Google — gemini-2.5-pro"


def test_a_file_saved_before_attribution_renders_unknown(monkeypatch, tmp_path):
    """Old files have no attribution key: explicit Unknown, never a traceback."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "src" / "data" / "test_cases"
    target.mkdir(parents=True)
    (target / "legacy.json").write_text(json.dumps(
        {"test_cases": [], "metadata": {"generated_at": "2026-01-01", "model": "gpt-4o"}}
    ), encoding="utf-8")

    metadata = app.latest_test_case_metadata()
    assert "attribution" not in metadata
    app.render_attribution(metadata)   # must not raise
    app.render_attribution(None)
    app.render_attribution({})


def test_the_fallback_formatter_is_not_an_attribution_hole(gateway):
    gateway.default_chat = gateway.chat_ok("openai", "gpt-4o-mini", content="[]")
    agent = TestGeneratorAgent(model="gpt-4o-mini")
    agent.generate_comprehensive_tests({"transcript": "x", "video_info": {}}, ["ui"], ["high"])

    formatted = app.format_test_cases_fallback([{"id": "TC001"}], agent)
    assert formatted["metadata"]["attribution"]["provider"] == "openai"


def test_the_fallback_formatter_without_an_agent_is_explicit():
    formatted = app.format_test_cases_fallback([{"id": "TC001"}])
    assert formatted["metadata"]["attribution"]["requested_model"] == ""


# --------------------------------------------------------------------------
# generation labels output no model produced
# --------------------------------------------------------------------------

def test_an_exhausted_chain_produces_stubs_that_say_so(gateway):
    gateway.default_chat = gateway.status(500)
    agent = TestGeneratorAgent(model="gpt-4o-mini")
    cases = agent.generate_comprehensive_tests(
        {"transcript": "x", "video_info": {}}, ["ui"], ["high"]
    )

    assert cases and "Fallback" in cases[0]["title"]
    summary = agent.format_test_cases(cases)["metadata"]["attribution"]
    assert summary["requested_model"] == "", "stub content must not carry a model"
    assert "all models failed" in summary["note"]
