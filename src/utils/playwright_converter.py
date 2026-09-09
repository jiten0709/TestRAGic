"""Turn canonical test cases into runnable Playwright/pytest suites.

Steps arrive as plain English (see `test_generator.normalise_case`), so conversion
is a heuristic: a step names an action, an element and sometimes a value, and this
module maps that onto Playwright's role-first locator API.

The one rule that matters: **a step that cannot be converted must fail loudly.**
This used to emit `# TODO: Define selector` as a bare comment, which pytest reads
as a passing test that does nothing -- the worst possible outcome for a QA tool.
Unconvertible steps now emit `pytest.skip(...)`, which shows up as a skip in the
report and in the dashboard. An unconvertible *check* is deferred to the end of the
test instead: the flow and the case's real assertions still run, and the test still
finishes on a skip, so nothing unchecked is ever reported as a pass.
"""

import re
from typing import Dict, List
from pathlib import Path

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="utils.log")

DEFAULT_TIMEOUT = 30000  # 30 seconds

# Where generated suites land. Deliberately NOT src/tests/generated/: that path is
# on pytest.ini's testpaths next to the committed test_sample.py, so generated
# suites written there are collected by every CI run of the repo's own tests.
GENERATED_TESTS_DIR = Path("src/data/generated_tests")

# action -> (recogniser, how to emit it). "page" actions take no locator.
# Ordered: back/forward are tried before navigate, or "navigate back" would goto.
PAGE_ACTIONS = {
    'back': (r'go back|navigate back', 'go_back'),
    'forward': (r'go forward|navigate forward', 'go_forward'),
    'reload': (r'reload|refresh', 'reload'),
    'navigate': (r'navigate|go to|visit|open|browse to', 'goto'),
    'wait': (r'wait for|wait|pause', 'wait_for_timeout'),
}

# Locator actions, longest-phrase-first so "double click" beats "click".
LOCATOR_ACTIONS = [
    ('scroll', r'scroll to|scroll into view|scroll', 'scroll_into_view_if_needed'),
    ('double_click', r'double[- ]click', 'dblclick'),
    ('right_click', r'right[- ]click|context click', 'click'),
    ('uncheck', r'uncheck|untick', 'uncheck'),
    ('check', r'\bcheck\b|\btick\b', 'check'),
    ('select', r'select option|choose|pick|select', 'select_option'),
    ('upload', r'upload|attach file|select file', 'set_input_files'),
    ('fill', r'\btype\b|\benter\b|\binput\b|\bfill\b', 'fill'),
    ('clear', r'\bclear\b|\bempty\b', 'clear'),
    ('hover', r'hover|mouse over', 'hover'),
    ('focus', r'focus on|focus', 'focus'),
    ('press', r'press key|press the|hit', 'press'),
    ('click', r'click on|click|tap|\bpress\b', 'click'),
]

# Element words -> ARIA role. get_by_role is the selector Playwright recommends and
# the only one that survives a CSS refactor of the app under test.
ROLE_WORDS = {
    'button': 'button', 'link': 'link', 'checkbox': 'checkbox', 'check box': 'checkbox',
    'radio button': 'radio', 'radio': 'radio', 'tab': 'tab', 'heading': 'heading',
    'header': 'heading', 'title': 'heading', 'menu item': 'menuitem', 'menu': 'menu',
    'dialog': 'dialog', 'modal': 'dialog', 'popup': 'dialog', 'alert': 'alert',
    'dropdown': 'combobox', 'combobox': 'combobox', 'select': 'combobox',
    'textbox': 'textbox', 'text box': 'textbox', 'field': 'textbox',
    'input': 'textbox', 'textarea': 'textbox', 'search box': 'searchbox',
    'table': 'table', 'list': 'list', 'image': 'img',
}
# Longest first: "radio button" must win over "button".
_ROLE_WORDS_ORDERED = sorted(ROLE_WORDS, key=len, reverse=True)

_QUOTED = re.compile(r'["“‘\']([^"”’\']{1,80})["”’\']')
# Stripped when reading an element's name out of a step: articles, prepositions and
# the verbs the step itself is built from. "Check the Terms checkbox" names "Terms".
_ARTICLES = {
    'the', 'a', 'an', 'on', 'to', 'into', 'in', 'at', 'and', 'then', 'its', 'of', 'for',
    'click', 'clicks', 'press', 'tap', 'check', 'uncheck', 'select', 'choose', 'pick',
    'enter', 'type', 'input', 'fill', 'hover', 'focus', 'upload', 'attach', 'clear',
    'verify', 'ensure', 'confirm', 'see', 'is', 'are', 'should', 'be', 'displayed',
    'visible', 'shown', 'user', 'page', 'scroll', 'locate', 'leave', 'that',
}

# Marker on a skip that stands for a *check* rather than an action. An unchecked
# assertion must not abort the flow (see _convert_single_test); an unconvertible
# action must, because everything after it runs on a screen that never happened.
UNCONVERTED_CHECK = "unconverted assertion: "

# Keys a bare "press X" step can mean. Anything else is treated as an element name.
KEY_NAMES = {'enter', 'tab', 'escape', 'esc', 'space', 'backspace', 'delete', 'end',
             'home', 'pageup', 'pagedown', 'arrowup', 'arrowdown', 'arrowleft',
             'arrowright', 'up', 'down', 'left', 'right'}


def _as_key(step):
    """'Press Enter' -> 'Enter'. Returns "" when the step names an element instead."""
    match = re.search(r'\b(?:press|hit)\s+(?:the\s+)?["\']?([\w+]+)["\']?\s*(?:key)?\s*[.!]?$',
                      step, re.IGNORECASE)
    if not match:
        return ""
    name = match.group(1)
    if name.lower() in KEY_NAMES:
        return {'esc': 'Escape', 'up': 'ArrowUp', 'down': 'ArrowDown',
                'left': 'ArrowLeft', 'right': 'ArrowRight'}.get(name.lower(), name.title())
    return name.title() if len(name) == 1 else ""


class PlaywrightConverter:
    def __init__(self, output_dir: Path = None):
        self.generated_tests_dir = Path(output_dir or GENERATED_TESTS_DIR)
        self.generated_tests_dir.mkdir(parents=True, exist_ok=True)
        # generated pytest function name -> the test case it came from, so an
        # execution report can be read in the vocabulary of the test cases.
        self.case_index = {}
        logger.info("Playwright Converter initialized -> %s", self.generated_tests_dir)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def convert_test_cases(self, test_cases, base_url: str = "") -> Dict:
        """Convert canonical test cases to Playwright code, one module per category."""
        try:
            grouped = self._group_by_category(test_cases)
            converted = {
                category: self._convert_category_tests(tests, category, base_url)
                for category, tests in grouped.items() if tests
            }
            return {
                "success": True,
                "converted_tests": converted,
                "total_categories": len(converted),
            }
        except Exception as e:
            logger.error("Error converting test cases: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    def generate_complete_test_suite(self, test_cases, suite_name: str, base_url: str = "") -> Dict:
        """Write one runnable pytest module per category. Returns the file paths."""
        converted = self.convert_test_cases(test_cases, base_url)
        if not converted["success"]:
            return converted

        saved_files = []
        for category, test_code in converted["converted_tests"].items():
            path = self.save_test_file(test_code, f"test_{self._sanitize(suite_name)}_{self._sanitize(category)}")
            if path:
                saved_files.append(path)

        return {"success": True, "test_files": saved_files, "total_files": len(saved_files),
                "case_index": self.case_index}

    def save_test_file(self, test_code: str, filename: str) -> str:
        try:
            file_path = self.generated_tests_dir / f"{filename}.py"
            file_path.write_text(test_code, encoding="utf-8")
            logger.info("Test file saved: %s", file_path)
            return str(file_path)
        except OSError as e:
            logger.error("Error saving test file: %s", e)
            return ""

    # ------------------------------------------------------------------
    # code generation
    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_category(test_cases) -> Dict[str, List[Dict]]:
        """Accept either {category: [case]} or the flat canonical list app.py holds."""
        if isinstance(test_cases, dict):
            if isinstance(test_cases.get("test_cases"), list):
                test_cases = test_cases["test_cases"]
            else:
                return {k: v for k, v in test_cases.items() if isinstance(v, list)}
        grouped = {}
        for case in test_cases or []:
            grouped.setdefault(str(case.get("category") or "general"), []).append(case)
        return grouped

    def _convert_category_tests(self, tests: List[Dict], category: str, base_url: str) -> str:
        seen = set()
        methods = []
        for i, test in enumerate(tests, 1):
            name = self._sanitize(test.get("title") or f"{category}_{i}") or f"case_{i}"
            while name in seen:                      # two cases may share a title
                name = f"{name}_{i}"
            seen.add(name)
            code, unchecked = self._convert_single_test(test, name, base_url)
            self.case_index[f"test_{name}"] = {
                "id": test.get("id", ""), "title": test.get("title", ""),
                "priority": test.get("priority", ""), "category": category,
                "unchecked": unchecked,
            }
            methods.append(code)

        # Module-level functions, not a Test* class: this repo's pytest.ini sets
        # `python_classes =` (empty) so that TestGeneratorAgent / TestExecutorAgent
        # are not collected as test classes, which also means a generated Test*
        # class collects zero tests and the run silently reports nothing to do.
        return self._module_header(base_url) + "\n\n" + "\n".join(methods)

    def _convert_single_test(self, test: Dict, method_name: str, base_url: str) -> str:
        title = test.get("title", "Untitled Test")
        steps = test.get("steps") or []
        expected = test.get("expected_result", "")

        body = [
            f"def test_{method_name}(page: Page):",
            '    """',
            f"    {self._comment(title)}",
            f"    Expected: {self._comment(expected) or 'not stated'}",
            '    """',
        ]
        if not steps:
            body.append(f"    pytest.skip({'test case has no steps: ' + title!r})")
            return "\n".join(body) + "\n", []

        lines = [(step, self.convert_step(str(step), base_url)) for step in steps]

        # A test case describes a flow, not a host: "click Sign in" assumes the app
        # is already open. Without this the first locator runs against about:blank
        # and every generated test fails for the same uninteresting reason.
        if not any(code.startswith("page.goto(") for _, code in lines):
            body.append("    page.goto(BASE_URL)")

        items = [(f"Step {n}", step, code) for n, (step, code) in enumerate(lines, 1)]
        items += [("Verify", source, code) for source, code in self._assertions_for(test)]

        # A check that could not be converted does not skip where it stands: that
        # threw away the flow and every real assertion after it over one sentence
        # nothing could parse. It is recorded, and the test still skips when it
        # leaves *nothing* verified -- a flow that checks nothing is a green lie.
        unchecked, checked = [], False
        for label, source, code in items:
            if UNCONVERTED_CHECK in code and code.startswith("pytest.skip("):
                unchecked.append(str(source))
                body.append(f"    # {label} (NOT CHECKED): {self._comment(source)}")
                continue
            checked = checked or code.startswith("expect(")
            body.append(f"    # {label}: {self._comment(source)}")
            body.append(f"    {code}")

        if unchecked and not checked:
            body.append(f"    pytest.skip({UNCONVERTED_CHECK + '; '.join(unchecked)!r})")
        return "\n".join(body) + "\n", unchecked

    def convert_step(self, step: str, base_url: str = "") -> str:
        """One English step -> one line of Playwright, or a visible skip."""
        lowered = step.lower().strip()

        # A step phrased as a check is an assertion, not an action -- models write
        # plenty of them, and converting one to a click would be nonsense.
        if re.match(r'(verify|ensure|confirm|assert|validate|observe|locate|check that|make sure|see that)\b', lowered):
            return self._generate_assertion(step)

        for action, (pattern, method) in PAGE_ACTIONS.items():
            if re.search(pattern, lowered):
                if action == 'navigate':
                    return f"page.goto({self._extract_url(step, base_url)!r})"
                if action == 'wait':
                    return f"page.wait_for_timeout({self._extract_timeout(step)})"
                return f"page.{method}()"

        action, method = self._detect_action(lowered)
        if not action:
            return f"pytest.skip({'unconverted step (no action recognised): ' + step!r})"

        if action == 'press':
            key = self._extract_value(step, action) or "Enter"
            locator = self._locator_for(step, action)
            return (f"{locator}.press({key!r})" if locator else f"page.keyboard.press({key!r})")

        locator = self._locator_for(step, action)
        if not locator:
            key = _as_key(step)
            if key:
                return f"page.keyboard.press({key!r})"
            return f"pytest.skip({'unconverted step (no element identified): ' + step!r})"

        value = self._extract_value(step, action)
        if action in ('fill', 'select', 'upload'):
            if not value:
                return f"pytest.skip({'unconverted step (no value found): ' + step!r})"
            return f"{locator}.{method}({value!r})"
        if action == 'right_click':
            return f"{locator}.click(button='right')"
        return f"{locator}.{method}()"

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_action(lowered: str):
        for action, pattern, method in LOCATOR_ACTIONS:
            if re.search(pattern, lowered):
                return action, method
        return "", ""

    def _locator_for(self, step: str, action: str = "") -> str:
        """A Playwright locator expression, preferring role/label over CSS guesses.

        Returns "" when the step names no element -- the caller turns that into a
        skip. Guessing (the old behaviour was `text="<last long word>"`) produced
        selectors that matched nothing and tests that looked real.
        """
        found = re.search(r'#([A-Za-z][\w-]*)', step)
        if found:
            return f"page.locator({'#' + found.group(1)!r})"

        lowered = step.lower()
        for word in _ROLE_WORDS_ORDERED:
            if re.search(rf'\b{re.escape(word)}s?\b', lowered):
                role = ROLE_WORDS[word]
                name = self._accessible_name(step, word, action)
                if role in ('textbox', 'searchbox', 'combobox') and name:
                    # A form control is identified by its label far more reliably
                    # than by its role name.
                    return f"page.get_by_label({name!r})"
                return (f"page.get_by_role({role!r}, name={name!r})" if name
                        else f"page.get_by_role({role!r})")

        quoted = _QUOTED.search(step)
        if quoted and action != 'fill':
            return f"page.get_by_text({quoted.group(1).strip()!r})"

        found = re.search(r'(?:^|\s)\.([a-zA-Z][\w-]*-[\w-]+)\b', step)
        if found:
            return f"page.locator({'.' + found.group(1)!r})"
        return ""

    @staticmethod
    def _accessible_name(step: str, role_word: str, action: str = "") -> str:
        """The element's visible name: quoted if given, else the words before it."""
        quoted = _QUOTED.search(step)
        # For fill/select the quoted text is the value being typed, not the label.
        if quoted and action not in ('fill', 'select', 'upload'):
            return quoted.group(1).strip()

        before = re.search(rf'((?:[\w\'-]+\s+){{0,3}})\b{re.escape(role_word)}s?\b',
                           step, re.IGNORECASE)
        if before:
            words = before.group(1).split()
            # A step reads "<verb> the <Name> <role>": keep only what follows the
            # last article, so a verb this list has never heard of ("Leave the
            # Password field empty") cannot leak into the element's name.
            cut = max((i for i, w in enumerate(words) if w.lower() in _ARTICLES), default=-1)
            kept = [w for w in words[cut + 1:] if w.lower() not in _ARTICLES]
            kept = kept or [w for w in words if w.lower() not in _ARTICLES]
            if kept:
                return " ".join(kept).strip("\"'.,:")
        return ""

    @staticmethod
    def _extract_value(step: str, action: str) -> str:
        if action == 'press':
            key = re.search(r'press(?:\s+the)?\s+["\']?([\w+ ]+?)["\']?\s*(?:key)?\s*$',
                            step, re.IGNORECASE)
            return key.group(1).strip().title().replace(" ", "+") if key else ""

        quoted = _QUOTED.search(step)
        if quoted:
            return quoted.group(1).strip()

        verbs = {'fill': r'(?:type|enter|input|fill(?:\s+in)?)',
                 'select': r'(?:select|choose|pick)',
                 'upload': r'(?:upload|attach)'}.get(action)
        if verbs:
            match = re.search(rf'{verbs}\s+(.+?)(?:\s+(?:in|into|on|to|from)\b|$)',
                              step, re.IGNORECASE)
            if match:
                return match.group(1).strip(" .\"'")
        return ""

    @staticmethod
    def _extract_timeout(step: str) -> int:
        match = re.search(r'(\d+)\s*(ms|millisecond|second|sec)', step, re.IGNORECASE)
        if not match:
            return 1000
        value = int(match.group(1))
        return value if match.group(2).lower().startswith('m') else value * 1000

    @staticmethod
    def _extract_url(step: str, base_url: str) -> str:
        match = re.search(r'https?://[^\s"\'<>]+', step)
        if match:
            return match.group(0).rstrip('.,')
        host = re.search(r'\b((?:[\w-]+\.)+[a-z]{2,}\b(?:/[\w\-./]*)?)', step, re.IGNORECASE)
        if host:
            return "https://" + host.group(1).rstrip('.,')
        path = re.search(r'\s(/[\w\-/]*)', step)
        if path and base_url:
            return base_url.rstrip('/') + path.group(1)
        return base_url or "about:blank"

    def _assertions_for(self, test: Dict) -> List[tuple]:
        """(source, code) from the case's own `assertions`, else its expected_result."""
        sources = [a for a in (test.get("assertions") or []) if str(a).strip()]
        if not sources and test.get("expected_result"):
            sources = [test["expected_result"]]
        return [(source, self._generate_assertion(str(source))) for source in sources]

    def _generate_assertion(self, expected: str) -> str:
        """An `expect(...)` call, or a skip that names what could not be checked."""
        quoted = _QUOTED.search(expected)
        lowered = expected.lower()

        url = re.search(r'https?://[^\s"\']+', expected)
        if url:
            return f"expect(page).to_have_url({url.group(0).rstrip('.,')!r})"

        if quoted and re.search(r'\burls?\b|\bredirect', lowered):
            # get_by_text of a URL matches nothing; to_have_url retries while the
            # navigation the previous step kicked off is still in flight.
            return f"expect(page).to_have_url(re.compile({re.escape(quoted.group(1).strip())!r}))"

        if quoted:
            text = quoted.group(1).strip()
            if any(w in lowered for w in ('hidden', 'not visible', 'disappear', 'no longer')):
                return f"expect(page.get_by_text({text!r})).to_be_hidden()"
            return f"expect(page.get_by_text({text!r})).to_be_visible()"

        locator = self._locator_for(expected)
        if locator:
            if any(w in lowered for w in ('hidden', 'not visible', 'disappear', 'no longer')):
                return f"expect({locator}).to_be_hidden()"
            if 'enabled' in lowered:
                return f"expect({locator}).to_be_enabled()"
            if 'disabled' in lowered:
                return f"expect({locator}).to_be_disabled()"
            return f"expect({locator}).to_be_visible()"

        return f"pytest.skip({UNCONVERTED_CHECK + expected!r})"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _module_header(base_url: str) -> str:
        return (
            '"""Generated Playwright suite -- regenerate rather than edit."""\n'
            "\n"
            "import re\n"
            "\n"
            "import pytest\n"
            "from playwright.sync_api import Page, expect\n"
            "\n"
            f"BASE_URL = {base_url or 'about:blank'!r}\n"
            f"TIMEOUT = {DEFAULT_TIMEOUT}\n"
            "\n"
            "\n"
            "@pytest.fixture(autouse=True)\n"
            "def _timeout(page: Page):\n"
            "    page.set_default_timeout(TIMEOUT)\n"
        )

    @staticmethod
    def _comment(text) -> str:
        """Flatten to one line: a newline inside a comment or docstring breaks the file."""
        return re.sub(r'\s+', ' ', str(text or "")).replace('"""', "'''").strip()

    @staticmethod
    def _sanitize(title: str) -> str:
        """Any title -> a valid, lowercase Python identifier fragment."""
        sanitized = re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', str(title))).strip('_').lower()
        if sanitized and sanitized[0].isdigit():
            sanitized = f"case_{sanitized}"
        return sanitized or "unnamed_test"

    # kept for callers that named it the old way
    _sanitize_method_name = _sanitize
    _convert_step_to_playwright = convert_step
