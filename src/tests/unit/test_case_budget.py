"""The suite budget: a total cap, split across categories, then across priorities."""

import pytest

from src.agents.test_generator import (
    MAX_TEST_CASES,
    PRIORITY_WEIGHTS,
    TestGeneratorAgent,
    allocate,
    nearest_priority,
)


VIDEO = {"transcript": "click sign up, enter email", "video_info": {"title": "Demo"}}


def cases_json(n, priority="high"):
    """A model response carrying `n` well-formed cases."""
    return "[" + ", ".join(
        f'{{"title": "T{i}", "priority": "{priority}", "steps": ["do it"]}}'
        for i in range(n)
    ) + "]"


# --------------------------------------------------------------------------
# allocate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("total,buckets,weights", [
    (20, ["a", "b", "c"], None),
    (5, ["a", "b", "c"], None),
    (2, ["a", "b", "c", "d"], None),          # fewer cases than buckets
    (1, ["a"], None),
    (12, ["critical", "high", "low"], PRIORITY_WEIGHTS),
    (7, ["critical", "medium"], PRIORITY_WEIGHTS),
])
def test_the_parts_always_sum_to_the_total(total, buckets, weights):
    split = allocate(total, buckets, weights)
    assert sum(split.values()) == total
    assert set(split) == set(buckets)
    assert all(n >= 0 for n in split.values())


def test_an_even_split_is_within_one_of_flat():
    split = allocate(5, ["a", "b", "c"])
    assert sorted(split.values()) == [1, 2, 2]


@pytest.mark.parametrize("priorities,expected", [
    (["critical", "high"], {"critical": 2, "high": 2}),
    (["critical", "high", "low"], {"critical": 2, "high": 1, "low": 1}),
    (["medium", "low"], {"medium": 3, "low": 1}),
])
def test_the_weighted_split_matches_the_agreed_table(priorities, expected):
    assert allocate(4, priorities, PRIORITY_WEIGHTS) == expected


def test_nothing_is_allocated_from_nothing():
    assert allocate(0, ["a", "b"]) == {"a": 0, "b": 0}
    assert allocate(5, []) == {}


# --------------------------------------------------------------------------
# nearest_priority
# --------------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("critical", "critical"),                 # already allowed, untouched
    ("medium", "high"),                       # nearest by rank
    ("low", "high"),
    ("wildly-invented", "high"),              # unknown -> treated as medium -> high
])
def test_a_priority_outside_the_selection_snaps_into_it(given, expected):
    assert nearest_priority(given, ["critical", "high"]) == expected


def test_a_tie_goes_to_the_more_severe_level():
    # "high" sits one rank from both "critical" and "medium".
    assert nearest_priority("high", ["critical", "medium"]) == "critical"


# --------------------------------------------------------------------------
# the generator honours the budget
# --------------------------------------------------------------------------

def test_the_suite_never_exceeds_the_budget(gateway):
    gateway.default_chat = gateway.chat_ok(content=cases_json(6))
    agent = TestGeneratorAgent(model="auto")

    cases = agent.generate_comprehensive_tests(
        VIDEO, ["ui", "edge", "a11y"], ["high"], max_cases=7)

    assert len(cases) == 7, "the model offered 18; 7 were asked for"
    per_category = {c: sum(1 for x in cases if x["category"] == c)
                    for c in ("ui", "edge", "a11y")}
    assert sorted(per_category.values()) == [2, 2, 3]


def test_the_budget_is_capped_at_twenty(gateway):
    gateway.default_chat = gateway.chat_ok(content=cases_json(30))
    cases = TestGeneratorAgent(model="auto").generate_comprehensive_tests(
        VIDEO, ["ui"], ["high"], max_cases=500)
    assert len(cases) == MAX_TEST_CASES


def test_a_category_with_no_budget_costs_no_call(gateway):
    gateway.default_chat = gateway.chat_ok(content=cases_json(3))
    cases = TestGeneratorAgent(model="auto").generate_comprehensive_tests(
        VIDEO, ["ui", "edge", "a11y"], ["high"], max_cases=2)

    assert len(cases) == 2
    assert len(gateway.chat_models_called) == 2, "the third category was skipped, not sliced"


# --------------------------------------------------------------------------
# the generator honours the selected priorities (it used to ignore them)
# --------------------------------------------------------------------------

def test_every_case_lands_on_a_selected_priority(gateway):
    gateway.default_chat = gateway.chat_ok(
        content='[{"title": "A", "priority": "low"}, {"title": "B"}]')

    cases = TestGeneratorAgent(model="auto").generate_comprehensive_tests(
        VIDEO, ["ui"], ["Critical", "High"], max_cases=4)

    assert {c["priority"] for c in cases} <= {"critical", "high"}


def test_the_prompt_states_the_count_and_the_mix(gateway):
    seen = []

    def capture(request, body):
        seen.append(body["messages"][-1]["content"])
        return gateway.chat_ok(content=cases_json(4))(request, body)

    gateway.default_chat = capture
    TestGeneratorAgent(model="auto").generate_comprehensive_tests(
        VIDEO, ["ui"], ["Critical", "High"], max_cases=4, distribution="weighted")

    assert "Generate exactly 4 test case(s)." in seen[0]
    assert "Use only these priority values: critical, high." in seen[0]
    assert "2 x critical, 2 x high" in seen[0]
