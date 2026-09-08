import os
from pathlib import Path

import pytest

# The local page that stands in for "the application under test". Committed so the
# browser tests need no network and no running server -- test_search_functionality
# used to drive google.com, whose consent page meant the search box never appeared
# and whose networkidle wait never settled.
FIXTURE_APP = Path(__file__).parent / "src" / "tests" / "fixtures" / "demo_app.html"


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for all tests.

    Video recording is deliberately not forced here: pytest-playwright's --video
    flag owns that decision, and hardcoding record_video_dir overrode the CLI for
    every run, CI included.
    """
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Configure browser launch arguments.

    `headless` is not set here on purpose. It used to be pinned to False, which
    overrode --headed/CI and made a headless run impossible; pytest-playwright
    defaults to headless and --headed turns it off.
    """
    return {
        **browser_type_launch_args,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security",
        ],
    }


# Session-scoped to match the pytest-base-url fixture this overrides: the plugin's
# autouse session-scoped _verify_url requests it, and a function-scoped override
# raises ScopeMismatch during collection for every test in the repo.
@pytest.fixture(scope="session")
def base_url():
    """Base URL for the application under test."""
    return os.getenv("BASE_URL", FIXTURE_APP.as_uri())


@pytest.fixture(scope="session")
def local_app_url():
    """file:// URL of the committed demo page. Offline, deterministic."""
    return FIXTURE_APP.as_uri()


@pytest.fixture
def test_timeout():
    """Default timeout for tests"""
    return 30000  # 30 seconds


@pytest.fixture
def test_data_dir():
    """Directory containing test data files"""
    return "src/data/test_data"


@pytest.fixture
def screenshots_dir():
    """Directory for storing screenshots"""
    return "test-results/screenshots"
