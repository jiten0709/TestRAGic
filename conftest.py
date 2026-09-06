import pytest
import os
from playwright.sync_api import Browser, BrowserContext, Page, Playwright

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for all tests"""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
        "record_video_dir": "test-results/videos/",
        "record_video_size": {"width": 1280, "height": 720}
    }

@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Configure browser launch arguments"""
    return {
        **browser_type_launch_args,
        "headless": False,  # Set to True for CI
        "slow_mo": 100 if not os.getenv("CI") else 0,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor"
        ]
    }

# Session-scoped to match the pytest-base-url fixture this overrides: the plugin's
# autouse session-scoped _verify_url requests it, and a function-scoped override
# raises ScopeMismatch during collection for every test in the repo.
@pytest.fixture(scope="session")
def base_url():
    """Base URL for the application under test"""
    return os.getenv("BASE_URL", "http://localhost:3000")

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