"""Output language.

The gateway's `auto` route reaches models that answer in their own language:
an English 8.4k-char YouTube transcript produced a fully Chinese test suite
(iBTWBhODSU0, 2026-09-06). provider.chat now carries an English-only system
message by default, so every caller is covered, present and future.
"""

from src.agents.test_generator import TestGeneratorAgent
from src.utils import provider


VIDEO = {"transcript": "click sign up, enter email", "video_info": {"title": "Demo"}}


def _systems(gateway, body):
    return [m for m in body["messages"] if m["role"] == "system"]


def test_chat_sends_an_english_only_system_message(gateway):
    sent = []
    gateway.default_chat = lambda request, body: (
        sent.append(body) or gateway.chat_ok()(request, body)
    )

    provider.chat("generate test cases")

    system = _systems(gateway, sent[0])
    assert system, "no system message: models answer in their own language"
    assert "English" in system[0]["content"]


def test_the_user_prompt_stays_last(gateway):
    sent = []
    gateway.default_chat = lambda request, body: (
        sent.append(body) or gateway.chat_ok()(request, body)
    )

    provider.chat("generate test cases")

    assert sent[0]["messages"][-1] == {"role": "user", "content": "generate test cases"}


def test_an_explicit_system_message_wins(gateway):
    sent = []
    gateway.default_chat = lambda request, body: (
        sent.append(body) or gateway.chat_ok()(request, body)
    )

    provider.chat("hi", system="be terse")

    assert [m["content"] for m in _systems(gateway, sent[0])] == ["be terse"]


def test_system_none_opts_out(gateway):
    sent = []
    gateway.default_chat = lambda request, body: (
        sent.append(body) or gateway.chat_ok()(request, body)
    )

    provider.chat("hi", system=None)

    assert _systems(gateway, sent[0]) == []


def test_generation_inherits_the_english_default(gateway):
    """The path that produced the Chinese suite -- not just provider.chat directly."""
    sent = []
    gateway.default_chat = lambda request, body: (
        sent.append(body) or gateway.chat_ok(content='[{"Title": "A"}]')(request, body)
    )

    TestGeneratorAgent(model="auto").generate_comprehensive_tests(VIDEO, ["ui"], ["high"])

    assert "English" in _systems(gateway, sent[0])[0]["content"]
