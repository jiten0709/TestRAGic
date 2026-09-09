"""The canonical schema, the step converter, and the JUnit reader.

All three are pure functions over dicts and text: no browser, no network, no
gateway. The end-to-end run against a real browser lives in
`src/tests/generated/test_execution_pipeline.py`.
"""

import ast
import textwrap

import pytest

from src.agents.test_executor import TestExecutorAgent
from src.agents.test_generator import normalise_case, normalise_cases
from src.utils.playwright_converter import PlaywrightConverter


# --------------------------------------------------------------------------
# 1.1 -- one shape out, whatever shape went in
# --------------------------------------------------------------------------

# The three variants observed in this repo: the Title-Case the functional template
# used to ask for, the lowercase the module constants specify, and the
# `test_case_*` keys a gateway `auto` model returned live on 2026-09-06.
TITLE_CASE = {
    "ID": "TC001", "Title": "Sign in", "Description": "Signs a user in",
    "Category": "ui", "Priority": "Critical",
    "Steps": ["Open the login page", "Click the Sign in button"],
    "Expected Results": "The dashboard appears",
    "Assertions": ["Dashboard is visible"],
}
LOWER_CASE = {
    "id": "TC001", "title": "Sign in", "description": "Signs a user in",
    "category": "ui", "priority": "critical",
    "steps": ["Open the login page", "Click the Sign in button"],
    "expected_result": "The dashboard appears",
    "assertions": ["Dashboard is visible"],
}
PREFIXED = {
    "test_case_id": "TC001", "test_case_title": "Sign in",
    "test_description": "Signs a user in", "test_category": "ui",
    "test_priority": "critical",
    "test_steps": [
        {"step": 1, "action": "Open the login page", "expected": "The login form shows"},
        {"step": 2, "action": "Click the Sign in button", "expected": "The dashboard appears"},
    ],
    "assertions": ["Dashboard is visible"],
}


@pytest.mark.parametrize("variant", [TITLE_CASE, LOWER_CASE, PREFIXED],
                         ids=["title-case", "lowercase", "test_case_-prefixed"])
def test_every_observed_variant_normalises_to_one_shape(variant):
    case = normalise_case(variant, category="ui")

    assert case["id"] == "TC001"
    assert case["title"] == "Sign in"
    assert case["description"] == "Signs a user in"
    assert case["category"] == "ui"
    assert case["priority"] == "critical"
    assert case["steps"] == ["Open the login page", "Click the Sign in button"]
    assert case["expected_result"] == "The dashboard appears"
    assert case["assertions"] == ["Dashboard is visible"]


def test_normalisation_is_idempotent():
    once = normalise_case(PREFIXED, category="ui")
    assert normalise_case(once, category="ui") == once


def test_unknown_keys_survive():
    """Templates ask for extras (wcag_guideline, risk_level); losing them narrows output."""
    case = normalise_case({"title": "A", "wcag_guideline": "1.4.3"}, category="a11y")
    assert case["wcag_guideline"] == "1.4.3"


def test_ids_are_unique_and_sequential_across_a_suite():
    cases = normalise_cases([{"title": "A", "id": "TC001"},
                             {"title": "B", "id": "TC001"},
                             {"title": "C"}])
    assert [c["id"] for c in cases] == ["TC001", "TC002", "TC003"]


def test_non_dicts_are_dropped_rather_than_crashing():
    assert normalise_cases(["not a case", None, {"title": "A"}]) == normalise_cases([{"title": "A"}])


def test_a_missing_title_is_labelled_not_blank():
    assert normalise_case({"steps": ["x"]})["title"] == "Untitled test case"


# --------------------------------------------------------------------------
# 1.3 -- conversion produces runnable code, and says so when it cannot
# --------------------------------------------------------------------------

@pytest.fixture
def converter(tmp_path):
    return PlaywrightConverter(output_dir=tmp_path)


@pytest.mark.parametrize("step, expected", [
    ("Navigate to https://example.com/login", "page.goto('https://example.com/login')"),
    ('Click the "Sign in" button', "page.get_by_role('button', name='Sign in').click()"),
    ("Click the Submit button", "page.get_by_role('button', name='Submit').click()"),
    ('Enter "a@b.com" into the Email field', "page.get_by_label('Email').fill('a@b.com')"),
    ("Check the Terms checkbox", "page.get_by_role('checkbox', name='Terms').check()"),
    ("Press Enter", "page.keyboard.press('Enter')"),
    ("Wait 2 seconds", "page.wait_for_timeout(2000)"),
])
def test_recognised_steps_convert_to_playwright(converter, step, expected):
    assert converter.convert_step(step) == expected


@pytest.mark.parametrize("step", [
    "Frobnicate the widget",              # no action word
    "Click",                              # action, but no element named
])
def test_an_unconvertible_step_skips_loudly(converter, step):
    """A bare `# TODO` comment is a green test that did nothing -- the worst outcome."""
    line = converter.convert_step(step)
    assert line.startswith("pytest.skip("), line
    assert step in line, "the skip message must name the step that failed to convert"


def test_no_todo_placeholders_survive_in_generated_code(converter):
    case = {"id": "TC001", "title": "Mixed", "category": "ui",
            "steps": ["Frobnicate", 'Click the "Go" button'],
            "expected_result": "something vague happens", "assertions": []}
    code = converter.convert_test_cases([case])["converted_tests"]["ui"]
    assert "TODO" not in code


def test_generated_modules_are_valid_python(converter):
    case = {"id": "TC001", "title": "Log in: 'quoted' & odd/chars", "category": "core flows",
            "priority": "high",
            "steps": ['Navigate to /login', 'Enter "a@b.com" into the Email field',
                      'Click the "Sign in" button', "Frobnicate"],
            "expected_result": 'The "Dashboard" heading is visible',
            "assertions": ['The "Dashboard" heading is visible']}

    result = converter.generate_complete_test_suite([case], "suite", "http://localhost:8000")

    assert result["success"] and result["test_files"]
    source = open(result["test_files"][0]).read()
    ast.parse(source)  # a quote in a title used to break the module outright
    assert "def test_" in source
    assert result["case_index"], "the suite must map generated functions back to cases"
    assert list(result["case_index"].values())[0]["id"] == "TC001"


def test_an_unconvertible_check_does_not_abort_the_checks_that_did_convert(converter):
    """The killer: one unparseable sentence used to skip the whole flow at step 5."""
    case = {"id": "TC001", "title": "Login", "category": "ui",
            "steps": ['Navigate to https://example.com/login',
                      'Enter "admin" in the Password field',
                      'Verify redirection to the dashboard'],   # nothing to convert
            "assertions": ['The "Dashboard" heading is visible']}
    code = converter.convert_test_cases([case])["converted_tests"]["ui"]

    assert "pytest.skip" not in code, "a converted assertion ran: the case has a verdict"
    assert code.index("get_by_label('Password')") < code.index("expect(")
    assert "NOT CHECKED" in code, "the gap still has to be visible in the file"


def test_a_case_where_nothing_could_be_checked_still_skips(converter):
    """A flow that verifies nothing must never report green."""
    case = {"id": "TC002", "title": "Blind", "category": "ui",
            "steps": ['Navigate to https://example.com/'],
            "assertions": ["Everything looks fine"]}
    code = converter.convert_test_cases([case])["converted_tests"]["ui"]
    assert "pytest.skip" in code


def test_an_unlisted_verb_does_not_leak_into_the_element_name(converter):
    """`get_by_label('Leave Password')` matched nothing and burned the 30s timeout."""
    assert converter.convert_step("Leave the Password field empty.") == \
        "page.get_by_label('Password').clear()"


def test_a_host_without_a_scheme_is_still_a_host(converter):
    assert converter.convert_step("Navigate directly to admin-demo.nopcommerce.com/admin/") == \
        "page.goto('https://admin-demo.nopcommerce.com/admin/')"


def test_an_assertion_about_the_url_checks_the_url(converter):
    """get_by_text of a URL matches nothing; to_have_url also waits for the nav."""
    line = converter._generate_assertion("The URL contains '/admin/'.")
    assert line.startswith("expect(page).to_have_url(")


def test_a_case_with_no_steps_skips_rather_than_passing_empty(converter):
    code = converter.convert_test_cases([{"title": "Empty", "category": "ui", "steps": []}])
    assert "pytest.skip" in code["converted_tests"]["ui"]


# --------------------------------------------------------------------------
# 1.2 -- the JUnit reader behind every number on the Results page
# --------------------------------------------------------------------------

JUNIT = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <testsuites><testsuite name="pytest" tests="3">
      <testcase classname="t.TestUi" name="test_a" time="1.25"/>
      <testcase classname="t.TestUi" name="test_b" time="0.50">
        <failure message="Locator not found&#10;last line here">trace</failure>
      </testcase>
      <testcase classname="t.TestUi" name="test_c" time="0.01">
        <skipped message="unconverted step: Frobnicate"/>
      </testcase>
    </testsuite></testsuites>
""")


def test_junit_is_read_back_per_test(tmp_path):
    report = tmp_path / "r.xml"
    report.write_text(JUNIT)

    cases = TestExecutorAgent.parse_junit(report)

    assert [c["status"] for c in cases] == ["passed", "failed", "skipped"]
    assert cases[0]["duration"] == 1.25
    assert cases[1]["error"] == "last line here"


def test_a_missing_report_reads_as_no_tests_not_as_success(tmp_path):
    assert TestExecutorAgent.parse_junit(tmp_path / "absent.xml") == []


def test_a_run_that_collected_nothing_is_a_failure(tmp_path):
    """Exit code 5 with an empty report must never render as a green run."""
    agent = TestExecutorAgent(results_dir=tmp_path / "r", artifacts_dir=tmp_path / "a")
    result = agent.execute_tests([str(tmp_path / "does_not_exist.py")])
    assert result["status"] == "failed"
    assert result["passed"] == 0
