"""Generated test cases really run in a browser, and really pass and fail.

This is the check behind objective 1.2: a canonical test case is converted by
`PlaywrightConverter`, executed by `TestExecutorAgent` (which shells out to
pytest), and the outcome comes back from the browser rather than from
`random.choice`. It drives the committed fixture page, so it needs no network and
no server.

It spawns a pytest subprocess, so it is marked `integration` and takes a few
seconds. `pytest -m "not integration"` skips it.
"""

from pathlib import Path

import pytest

from src.agents.test_executor import TestExecutorAgent
from src.utils.playwright_converter import PlaywrightConverter

pytestmark = pytest.mark.integration

PASSING_CASE = {
    "id": "TC001", "title": "Sign in reaches the dashboard", "category": "core flows",
    "priority": "critical",
    "steps": ['Enter "qa@example.com" into the Email field',
              'Click the "Sign in" button'],
    "expected_result": 'The "Dashboard" heading is visible',
    "assertions": ['The "Dashboard" heading is visible'],
}
FAILING_CASE = {
    "id": "TC002", "title": "A control that does not exist", "category": "core flows",
    "priority": "low",
    "steps": ['Click the "Delete everything" button'],
    "expected_result": "", "assertions": [],
}
UNCONVERTIBLE_CASE = {
    "id": "TC003", "title": "A step no parser can read", "category": "core flows",
    "priority": "low",
    "steps": ["Frobnicate the doohickey"],
    "expected_result": "", "assertions": [],
}


@pytest.fixture(scope="module")
def execution(tmp_path_factory, local_app_url):
    """Convert three cases and run them once; the tests below read the outcome."""
    workdir = tmp_path_factory.mktemp("execution")
    converter = PlaywrightConverter(output_dir=workdir / "suite")
    suite = converter.generate_complete_test_suite(
        [PASSING_CASE, FAILING_CASE, UNCONVERTIBLE_CASE], "pipeline", local_app_url)
    assert suite["success"], suite

    executor = TestExecutorAgent(results_dir=workdir / "results",
                                 artifacts_dir=workdir / "artifacts",
                                 timeout=300)
    # A missing control should fail fast rather than burn the default 30s timeout.
    source = Path(suite["test_files"][0])
    source.write_text(source.read_text().replace("TIMEOUT = 30000", "TIMEOUT = 3000"))

    result = executor.execute_tests(suite["test_files"], browser="chromium",
                                    headless=True, base_url=local_app_url,
                                    capture=())
    return result, suite["case_index"]


def test_the_run_completed_against_a_real_browser(execution):
    result, _ = execution
    assert result["status"] == "completed", result.get("error")
    assert result["simulated"] is False
    assert result["total_tests"] == 3


def test_a_convertible_case_passes_and_a_broken_one_fails(execution):
    result, case_index = execution
    by_case = {case_index[c["name"]]["id"]: c["status"] for c in result["test_results"]}

    assert by_case["TC001"] == "passed", "the fixture page really signs in"
    assert by_case["TC002"] == "failed", "a control that does not exist must fail"


def test_an_unconvertible_step_is_reported_as_skipped_not_passed(execution):
    result, case_index = execution
    case = next(c for c in result["test_results"] if case_index[c["name"]]["id"] == "TC003")

    assert case["status"] == "skipped"
    assert "Frobnicate" in case["error"], "the skip names the step that could not convert"
