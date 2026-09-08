"""Smoke tests for the Playwright setup itself.

They drive the committed fixture page (`src/tests/fixtures/demo_app.html`) rather
than google.com: the old version filled `textarea[name="q"]` on google, which the
consent interstitial hides, so it failed with a 30s timeout on every machine.
"""

import pytest
from playwright.sync_api import Page, expect


def test_basic_navigation(page: Page, local_app_url):
    """Basic test to verify Playwright setup"""
    page.goto(local_app_url)
    assert page.title() == "Demo App"


def test_search_functionality(page: Page, local_app_url):
    """A form round-trip: fill, submit, assert on the result."""
    page.goto(local_app_url)
    page.get_by_label("Email").fill("qa@example.com")
    page.get_by_role("button", name="Sign in").click()
    expect(page.get_by_role("heading", name="Dashboard")).to_be_visible()


@pytest.mark.slow
def test_multiple_tabs(page: Page, local_app_url):
    """Test opening a second tab in the same context"""
    page.goto(local_app_url)
    second = page.context.new_page()
    second.goto(local_app_url)
    assert "Demo App" in second.title()
    second.close()
