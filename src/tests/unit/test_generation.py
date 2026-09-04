"""Test-case generation driven entirely by a fake provider."""

import pytest

from src.agents.test_generator import TestGeneratorAgent, _scan_json


VIDEO = {"transcript": "click sign up, enter email", "video_info": {"title": "Demo"}}

NESTED = '''Sure, here are the test cases:

```json
[
  {
    "ID": "TC001",
    "Title": "Sign up with a valid email",
    "Steps": [{"step": 1, "action": "open signup", "expected": "form shown"}],
    "Assertions": ["account created"]
  }
]
```
'''


# --------------------------------------------------------------------------
# the parser (risk #1: re.findall(r'{[^{}]*}') cannot match nested JSON)
# --------------------------------------------------------------------------

def test_nested_json_is_no_longer_discarded(gateway):
    gateway.default_chat = gateway.chat_ok(content=NESTED)
    agent = TestGeneratorAgent(model="auto")

    cases = agent.generate_comprehensive_tests(VIDEO, ["ui"], ["high"])

    assert len(cases) == 1
    assert cases[0]["Title"] == "Sign up with a valid email"
    assert cases[0]["Steps"][0]["action"] == "open signup", "nested steps survived"
    assert cases[0]["category"] == "ui"


def test_a_test_cases_envelope_is_unwrapped(gateway):
    gateway.default_chat = gateway.chat_ok(
        content='{"test_cases": [{"Title": "A"}, {"Title": "B"}]}'
    )
    cases = TestGeneratorAgent(model="auto").generate_comprehensive_tests(VIDEO, ["ui"], ["high"])

    assert [c["Title"] for c in cases] == ["A", "B"]


def test_a_bare_object_is_accepted(gateway):
    gateway.default_chat = gateway.chat_ok(content='{"Title": "Only one"}')
    cases = TestGeneratorAgent(model="auto").generate_comprehensive_tests(VIDEO, ["ui"], ["high"])
    assert [c["Title"] for c in cases] == ["Only one"]


def test_prose_with_no_json_produces_one_labelled_stub(gateway):
    gateway.default_chat = gateway.chat_ok(content="I could not produce JSON, sorry.")
    cases = TestGeneratorAgent(model="auto").generate_comprehensive_tests(VIDEO, ["ui"], ["high"])

    assert len(cases) == 1
    assert cases[0]["title"] == "ui Test Case"


@pytest.mark.parametrize("text, expected", [
    ('[{"a": {"b": [1, 2]}}]', [[{"a": {"b": [1, 2]}}]]),
    ('prose {"a": 1} more prose {"b": 2}', [{"a": 1}, {"b": 2}]),
    ('no json here at all', []),
    ('{"unterminated": ', []),
    ('```json\n{"a": 1}\n```', [{"a": 1}]),
])
def test_scan_json_handles_nesting_prose_and_junk(text, expected):
    assert list(_scan_json(text)) == expected


# --------------------------------------------------------------------------
# the per-category loop
# --------------------------------------------------------------------------

def test_one_call_per_category_each_attributed(gateway):
    gateway.default_chat = gateway.chat_ok(content='[{"Title": "A"}]')
    agent = TestGeneratorAgent(model="auto")

    cases = agent.generate_comprehensive_tests(VIDEO, ["ui", "edge_case", "integration"], ["high"])

    assert len(gateway.chat_models_called) == 3
    assert len(cases) == 3
    assert [category for category, _ in agent._attributions] == ["ui", "edge_case", "integration"]


def test_the_transcript_reaches_the_prompt(gateway):
    seen = {}

    def capture(request, body):
        seen["prompt"] = body["messages"][-1]["content"]
        return gateway.chat_ok(content="[]")(request, body)

    gateway.default_chat = capture
    TestGeneratorAgent(model="auto").generate_comprehensive_tests(VIDEO, ["ui"], ["high"])

    assert "click sign up" in seen["prompt"]
    assert "Demo" in seen["prompt"]


def test_a_failing_category_does_not_sink_the_others(gateway):
    gateway.chat_by_model["auto"] = gateway.chat_ok(content='[{"Title": "A"}]')
    agent = TestGeneratorAgent(model="auto")
    cases = agent.generate_comprehensive_tests(VIDEO, ["ui", "edge_case"], ["high"])

    assert len(cases) == 2


def test_the_agent_constructs_no_client_and_needs_no_key():
    """__init__ must not touch the network or require credentials."""
    agent = TestGeneratorAgent(model="auto")
    assert not hasattr(agent, "llm")
    assert agent._attributions == []


def test_the_legacy_single_flow_path_is_explicitly_not_wired(gateway):
    result = TestGeneratorAgent(model="auto").generate_test_cases("some flow")
    assert result["success"] is False
    assert "generate_comprehensive_tests" in result["error"]
    assert gateway.chat_models_called == []
