"""
This complete implementation provides:

Comprehensive test generation - Functional, edge cases, accessibility, and performance tests
Context-aware generation - Uses RAG retrieval for video-based context
Multiple output formats - JSON and Markdown reports
Robust error handling - Fallback parsing for malformed responses
Template-based prompts - Structured prompts for consistent outputs
Integration ready - Works with data ingestion and Playwright converter
Detailed logging - Comprehensive error tracking and info logging
"""

import datetime
import json
import re

from src.utils import provider

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="agents.log")

def _scan_json(text):
    r"""Yield every JSON array/object embedded in `text`, nesting included.

    json.JSONDecoder().raw_decode understands nested structures; the regex it
    replaces (r'\{[^{}]*\}') matched only flat objects, so any test case with a
    nested "steps" list of objects was silently dropped on the floor.
    """
    decoder = json.JSONDecoder()
    index, length = 0, len(text)
    while index < length:
        if text[index] in "[{":
            try:
                value, end = decoder.raw_decode(text, index)
            except ValueError:
                index += 1
                continue
            yield value
            index = end
        else:
            index += 1



# --------------------------------------------------------------------------
# The canonical test-case schema
# --------------------------------------------------------------------------
# One shape, lowercase, for every consumer: app.py's display, PlaywrightConverter,
# TestExecutorAgent, the saved JSON. Three variants were observed in the wild --
# the Title-Case the functional template used to ask for, the lowercase the
# module constants specify, and `test_case_title` / `test_steps` returned live by
# a gateway `auto` model on 2026-09-06 -- and every consumer grew its own
# `.get('ID', .get('id', ...))` cascade, each missing a different variant.
# Normalising once here is what lets those cascades go away.
CANONICAL_KEYS = ("id", "title", "description", "category", "priority",
                  "steps", "expected_result", "assertions")

# Aliases are matched after stripping case and separators, so "Expected Results",
# "expected_results" and "expectedResults" are one entry.
_ALIASES = {
    "id": ("id", "test_case_id", "testcase_id", "tc_id", "case_id", "identifier"),
    "title": ("title", "test_case_title", "test_title", "name", "test_name", "summary"),
    "description": ("description", "test_description", "desc", "objective", "purpose"),
    "category": ("category", "test_category", "suite", "type", "test_type"),
    "priority": ("priority", "test_priority", "severity", "importance"),
    "steps": ("steps", "test_steps", "step", "actions", "procedure", "test_procedure"),
    "expected_result": ("expected_result", "expected_results", "expected",
                        "expected_outcome", "expected_behavior", "expected_behaviour"),
    "assertions": ("assertions", "assertion", "validations", "verifications",
                   "verification_points", "checks", "acceptance_criteria"),
}

# Step-level keys, for the `[{"step": 1, "action": ..., "expected": ...}]` shape.
_STEP_ACTION_KEYS = ("action", "step", "description", "instruction", "step_description", "text")
_STEP_EXPECTED_KEYS = ("expected", "expected_result", "expected_results", "expected_outcome")

PRIORITIES = ("critical", "high", "medium", "low")

# A whole suite is capped here, not per category: the cost of a run is one
# provider call per category times whatever the model felt like returning, and
# before this there was no ceiling at all -- only the prompt string "3-5".
MAX_TEST_CASES = 20
PRIORITY_WEIGHTS = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# A transcript at or under this many characters goes into every prompt whole, and
# retrieval is skipped. Retrieval only helps on a corpus that does not fit, and
# measured on the two videos in src/data/transcripts/ neither does: k=5 hands the
# model ~1,266 chars of a 5,534-char transcript -- less than the 2,000-char head it
# is supposed to improve on, and under a quarter of a source that is ~1.4k tokens
# whole. 12000 chars is roughly 3k tokens. Above it, RAG works as before.
FULL_CONTEXT_CHARS = 12000


def allocate(total, buckets, weights=None):
    """Split `total` across `buckets`; the parts sum to exactly `total`.

    Even when `weights` is None, largest-remainder otherwise. Ties on the
    remainder go to the bucket holding the least, so a selected priority is
    never starved to zero while a neighbour takes two.
    """
    if not buckets or total <= 0:
        return {b: 0 for b in buckets}
    w = [(weights or {}).get(b, 1) for b in buckets]
    share = [total * x / sum(w) for x in w]
    out = [int(s) for s in share]
    order = sorted(range(len(buckets)),
                   key=lambda i: (share[i] - out[i], -out[i]), reverse=True)
    for i in order[:total - sum(out)]:
        out[i] += 1
    return dict(zip(buckets, out))


def nearest_priority(priority, allowed):
    """Snap a model's priority onto the set the user selected.

    The UI has always collected priority levels and the generator has always
    ignored them, so the Results page filtered on levels nobody asked for.
    Nothing is dropped -- an out-of-set case moves to the closest level by rank,
    ties going to the more severe one.
    """
    if priority in allowed:
        return priority
    rank = list(PRIORITIES)
    i = rank.index(priority) if priority in rank else rank.index("medium")
    return min(allowed, key=lambda p: (abs(rank.index(p) - i), rank.index(p)))


def _flat(name):
    """Key identity: case- and separator-insensitive. 'Expected Results' -> 'expectedresults'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_ALIAS_LOOKUP = {_flat(alias): canonical
                 for canonical, aliases in _ALIASES.items()
                 for alias in aliases}


def _as_text_list(value):
    """Anything the model called a list of sentences -> a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.splitlines()
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for item in value:
        text = str(item).strip().strip("-*• \t")
        if text:
            out.append(text)
    return out


def _normalise_steps(value):
    """-> (steps as plain strings, any expected-result text carried on a step).

    PlaywrightConverter and the executor both read steps as sentences; a
    `{"step": 1, "action": ..., "expected": ...}` object has to collapse to one.
    The `expected` text is not thrown away: it becomes the case's
    expected_result when the case itself did not state one.
    """
    if isinstance(value, str):
        value = value.splitlines()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ([] if value is None else _as_text_list(value)), None

    steps, expected = [], None
    for item in value:
        if isinstance(item, dict):
            flat = {_flat(k): v for k, v in item.items()}
            text = next((flat[k] for k in map(_flat, _STEP_ACTION_KEYS)
                         if isinstance(flat.get(k), str) and flat[k].strip()), None)
            if text is None:
                text = " ".join(str(v) for v in item.values()
                                if isinstance(v, (str, int, float)))
            found = next((flat[k] for k in map(_flat, _STEP_EXPECTED_KEYS)
                          if isinstance(flat.get(k), str) and flat[k].strip()), None)
            expected = found or expected
            steps.extend(_as_text_list(text))
        else:
            steps.extend(_as_text_list(item))
    return steps, expected


def normalise_case(case, category=None, index=None):
    """One model-shaped test case -> the canonical schema. Idempotent.

    Unrecognised keys are kept as-is: templates ask for extras like
    `wcag_guideline` or `risk_level` and losing them would silently narrow the
    output. Returns None for anything that is not a dict.
    """
    if not isinstance(case, dict):
        return None

    out, extras = {}, {}
    for key, value in case.items():
        canonical = _ALIAS_LOOKUP.get(_flat(key))
        if canonical is None:
            extras[key] = value
        elif canonical not in out or out[canonical] in (None, "", [], {}):
            out[canonical] = value

    steps, step_expected = _normalise_steps(out.get("steps"))

    normalised = dict(extras)
    normalised["id"] = str(out.get("id") or "").strip() or (
        f"TC{index:03d}" if index else "")
    normalised["title"] = str(out.get("title") or "").strip() or "Untitled test case"
    normalised["description"] = str(out.get("description") or "").strip()
    # The caller's category wins: it is the category actually requested, whereas
    # the model's is whatever it decided to echo back.
    normalised["category"] = str(category or out.get("category") or "").strip()
    priority = str(out.get("priority") or "").strip().lower()
    normalised["priority"] = priority if priority in PRIORITIES else (priority or "medium")
    normalised["steps"] = steps
    normalised["expected_result"] = str(
        out.get("expected_result") or step_expected or "").strip()
    normalised["assertions"] = _as_text_list(out.get("assertions"))
    return normalised


def normalise_cases(cases, category=None):
    """A list of model-shaped cases -> canonical cases with unique, sequential ids."""
    if isinstance(cases, dict):
        cases = cases.get("test_cases", [])
    normalised, seen = [], set()
    for case in cases or []:
        position = len(normalised) + 1   # number the cases kept, not the ones seen
        canonical = normalise_case(case, category=category, index=position)
        if canonical is None:
            continue
        if not canonical["id"] or canonical["id"] in seen:
            canonical["id"] = f"TC{position:03d}"
        seen.add(canonical["id"])
        normalised.append(canonical)
    return normalised


# The one JSON shape every category prompt asks for. Models still drift -- that is
# what normalise_case() is for -- but asking for the canonical keys directly means
# the common case needs no repair.
JSON_SCHEMA_INSTRUCTION = """Return ONLY a JSON array. Each element must use exactly these keys:
[
  {
    "id": "TC001",
    "title": "Clear, descriptive test title",
    "description": "What this test validates",
    "category": "the category you were asked for",
    "priority": "critical|high|medium|low",
    "steps": ["One imperative sentence per step, naming the element acted on"],
    "expected_result": "What must be true when the steps have run",
    "assertions": ["One verifiable statement per assertion"]
  }
]
Use lowercase keys exactly as shown. "steps" and "assertions" are arrays of plain strings, not objects.

Every step must be one thing a person does to the UI -- click, type, select, upload,
navigate -- naming the element and quoting any literal value: Enter "admin" in the
Password field. Steps are converted to Playwright automatically, so:
- No narrative or precondition steps ("Start from the dashboard after login"): write
  out the actions that get there, every time, from the first page.
- No verification steps ("Verify the dashboard loads"): checks belong in "assertions".
- Write URLs in full, with the scheme: https://example.com/admin/.
- Quote only text you actually saw in the material above; do not invent error wording.
- Never write a placeholder value. "Enter a valid password" is typed into the field
  literally, character for character; write Enter "admin123" instead.
- Every assertion must quote the exact on-screen text, URL fragment or element name
  you expect to see. An assertion that quotes nothing cannot be checked and is
  discarded, leaving the case with no verdict.
- Every case starts from the first page and performs its own navigation and login.
  Nothing is carried over from another test case.
- A control's starting state is a step, not an assumption. A form can arrive pre-filled,
  pre-checked or pre-selected, so if the case needs a field empty, write "Clear the
  <name> field" as its own step before submitting. A case whose title promises a state
  its steps never create tests the opposite of what it claims.

BAD -- vague; compiles to a test that cannot fail:
  step:      "Enter a valid password in the Password field"
  step:      "Start from the admin dashboard after successful login"
  assertion: "An error message is visible next to the password field"
GOOD -- every value and every expected string is a literal you saw:
  step:      "Enter \\"admin123\\" in the Password field"
  step:      "Navigate to https://example.com/login"
  assertion: "The text \\"Login was unsuccessful\\" is visible"

BAD -- the title promises a state the steps never create, so the form's own default
value is submitted and the test exercises a *successful* login:
  title: "Login attempt with empty password field"
  steps: ["Enter \\"admin@example.com\\" in the Email field",
          "Click the \\"LOG IN\\" button"]
GOOD -- the state is established before it is relied on:
  title: "Login attempt with empty password field"
  steps: ["Enter \\"admin@example.com\\" in the Email field",
          "Clear the Password field",
          "Click the \\"LOG IN\\" button"]
"""


# --------------------------------------------------------------------------
# Category templates
# --------------------------------------------------------------------------
# One template per category the UI actually offers. Each is a persona line plus a
# numbered focus list, and **the numbering is load-bearing**: _retrieval_query
# parses `^\s*\d+\. (.+)$` out of the template to build the vector-store query,
# so these lines are both the model's instructions and the retrieval query text.
#
# They carry no JSON block on purpose. JSON_SCHEMA_INSTRUCTION is appended to every
# prompt separately; the schema these constants used to declare (`"priority":
# "High|Medium|Low"`, `test_data`, `preconditions`) contradicted it.

FUNCTIONAL_TEMPLATE = """You are a senior QA engineer specialising in end-to-end frontend automation.
Generate functional test cases for the user flow shown in the material above.

Focus on:
1. The primary user journey demonstrated, from the first screen to the last
2. Form entry, submission, and the confirmation that follows
3. Navigation between the screens that actually appear in the material
4. Data shown back to the user after an action succeeds
5. What the screen shows when a required field is missing or wrong"""

EDGE_CASE_TEMPLATE = """You are a senior QA engineer who specialises in breaking software.
Generate edge case and negative-path test cases for the user flow shown above.

Focus on:
1. Invalid, empty and malformed input in the fields that appear on screen
2. Boundary values: maximum length, zero, negative numbers, special characters
3. Wrong credentials, expired sessions, and direct access to a protected screen
4. Interrupting a flow midway: browser back, reload, double submit
5. The exact message the user is shown when an action is refused

Every case must name the specific field or control it abuses.
Do not restate the happy path -- that is another category's job."""

ACCESSIBILITY_TEMPLATE = """You are an accessibility specialist testing against WCAG 2.1 level AA.
Generate accessibility test cases for the screens shown above.

Focus on:
1. Reaching and operating every control with the keyboard alone, in tab order
2. Visible focus indication as focus moves between controls
3. Form fields having a programmatically associated label
4. Heading levels, landmarks, and the reading order of the page
5. Text alternatives for images, icons, and controls that show no text

Name the specific control or region on screen, never "the page".
Add a "wcag_guideline" key naming the success criterion (for example "1.4.3")."""

PERFORMANCE_TEMPLATE = """You are a performance engineer.
Generate performance test cases for the flows shown above.

Focus on:
1. Time for the first screen of the flow to become interactive
2. Time between submitting a form and the next screen appearing
3. Rendering a list or table that holds many rows
4. Repeating the flow back to back without a full reload
5. Behaviour on a slow or throttled network connection

Add a "metric" key and a "threshold" key carrying a number and a unit."""

CROSS_BROWSER_TEMPLATE = """You are a QA engineer who tests the same flow across rendering engines.
Generate cross-browser test cases for the flows shown above.

Focus on:
1. The same journey completing identically on Chromium, Firefox and WebKit
2. Form controls that engines render differently: date, file, and select inputs
3. Layout of the screens as the viewport is resized
4. Fonts, icons and images loading on every engine
5. Copy, paste and keyboard shortcuts inside text fields"""

MOBILE_TEMPLATE = """You are a QA engineer testing on mobile viewports.
Generate mobile test cases for the flows shown above.

Focus on:
1. The flow completing on a narrow viewport without horizontal scrolling
2. Tap targets being large enough to hit and not overlapping
3. Menus and navigation that collapse behind a toggle
4. The on-screen keyboard covering the field being typed into
5. Portrait and landscape orientation

Name the specific control or screen from the material above."""

UI_TEMPLATE = """You are a QA engineer specialising in user interface verification.
Generate UI test cases for the screens shown above.

Focus on:
1. Every control that appears on screen being present and in the right state
2. Labels, headings and button text reading exactly as shown in the material
3. Enabled, disabled and error states of the form controls
4. What appears and disappears as the user moves through the flow
5. Lists, tables and panels rendering the data they are given"""

INTEGRATION_TEMPLATE = """You are a QA engineer specialising in integration testing.
Generate integration test cases for the flows shown above.

Focus on:
1. Data entered on one screen appearing correctly on a later screen
2. A saved record surviving a reload or a fresh navigation
3. Search, filter and list views reflecting a change made elsewhere
4. Actions that depend on a prior action having completed
5. The state the user is left in after leaving and re-entering the flow"""


# Keyed by _flat(), the same case- and separator-insensitive identity the schema
# aliases use, so "Edge Cases", "edge_case" and "edge cases" are one entry.
#
# This map replaced a dict keyed `functional|ui|integration|edge_case` that was
# looked up with `category.lower().replace(' ', '_')`. Every label the UI offers
# ("Core User Flows", "Edge Cases", "Accessibility", ...) missed it and silently
# fell through to the functional template -- so every category was prompted as
# functional, AND, because _retrieval_query derives its query from the template,
# every category retrieved on the same functional topics. One lookup bug disabled
# per-category prompting and per-category retrieval at once.
CATEGORY_TEMPLATES = {
    "coreuserflows": FUNCTIONAL_TEMPLATE,
    "functional": FUNCTIONAL_TEMPLATE,
    "edgecases": EDGE_CASE_TEMPLATE,
    "edgecase": EDGE_CASE_TEMPLATE,        # singular: what the old dict was keyed on
    "accessibility": ACCESSIBILITY_TEMPLATE,
    "performance": PERFORMANCE_TEMPLATE,
    "crossbrowser": CROSS_BROWSER_TEMPLATE,
    "mobile": MOBILE_TEMPLATE,
    "ui": UI_TEMPLATE,
    "integration": INTEGRATION_TEMPLATE,
}


def template_for(category):
    """The prompt template for a category label, however it is spelled."""
    return CATEGORY_TEMPLATES.get(_flat(category), FUNCTIONAL_TEMPLATE)


class TestGeneratorAgent:
    def __init__(self, model=None):
        """Initialize Test Generator Agent"""
        # Default through the provider, not a hardcoded OpenAI id: in omniroute
        # mode that id is a model the gateway may hold no credentials for.
        self.model = model or provider.default_chat_model()

        # No client is constructed here: every call goes through src/utils/provider.py,
        # which owns the endpoint, the fallback chain and the attribution headers.
        self.retriever = None
        self._attributions = []  # (category, Attribution), one per generated category

        logger.info(f"TestGeneratorAgent initialized with model: {self.model}")

    def set_retriever(self, retriever):
        """Set the retriever from data ingestion agent"""
        self.retriever = retriever
        logger.info("Retriever set for context-aware test generation")

    @staticmethod
    def _retrieval_query(category, template=None):
        """What to actually ask the vector store for.

        A bare category label is a poor query: measured against a 4-chunk corpus,
        "authentication" separated the login chunk from an unrelated one by 0.002
        cosine, while a phrased query separated them by 0.29. Qwen3 embeds queries
        under a "retrieve passages that answer this" instruction, so it needs a
        question's worth of content, not a heading.

        Each template already lists five concrete topics for its category; those
        numbered lines are the description the label is missing.
        """
        focus = re.findall(r"^\s*\d+\.\s*(.+?)\s*$", template or "", re.M)
        return f"{category}: {', '.join(focus)}" if focus else category

    def _context_for(self, category, transcript, template=None):
        """Source material for one category's prompt.

        Three tiers. A transcript that fits (FULL_CONTEXT_CHARS) is passed whole --
        no subset of a 5k-char source beats the source. Above that, retrieval queries
        the vector store per category, which is what makes the chosen embedding model
        affect the output at all. If that fails or returns nothing, the last resort is
        `transcript[:2000]` -- the same opening ~2000 *characters* for every category,
        so accessibility tests get written from the opening ninety seconds.
        """
        if transcript and len(transcript) <= FULL_CONTEXT_CHARS:
            # The whole source beats any subset of it, and costs fewer tokens than
            # the retrieval round-trip saves. See FULL_CONTEXT_CHARS.
            return transcript

        if self.retriever:
            try:
                docs = self.retriever.invoke(self._retrieval_query(category, template))
                chunks = "\n\n".join(d.page_content for d in docs)
                if chunks.strip():
                    return chunks
                logger.warning("Retrieval returned nothing for %s; using transcript head", category)
            except Exception as e:
                logger.warning("Retrieval failed for %s, using transcript head: %s", category, e)
        return f"{transcript[:2000]}..."
    
    def generate_comprehensive_tests(self, video_content, categories, priorities,
                                     max_cases=MAX_TEST_CASES, distribution="even"):
        """Generate comprehensive test cases based on video content.

        `max_cases` is a budget for the whole suite, split evenly across the
        categories; each category's share is then split across `priorities`,
        evenly or by PRIORITY_WEIGHTS when `distribution == "weighted"`. The
        prompt asks for that mix and the parse enforces it -- a model asked for
        four cases routinely returns seven.
        """
        try:
            # Extract relevant information from video content
            transcript = video_content.get('transcript', '')
            video_info = video_content.get('video_info', {})
            
            all_test_cases = []
            self._attributions = []

            # Derived from PRIORITIES, so the UI's Title-Case "Critical" lands
            # lowercase and in severity order without the caller normalising it.
            allowed = [p for p in PRIORITIES
                       if p in {str(x).strip().lower() for x in (priorities or [])}]
            allowed = allowed or list(PRIORITIES)
            budget = allocate(min(max_cases, MAX_TEST_CASES), list(categories))
            weights = PRIORITY_WEIGHTS if distribution == "weighted" else None
            logger.info("case budget %s across priorities %s (%s)",
                        budget, allowed, distribution)

            for category in categories:
                quota = budget[category]
                if quota <= 0:
                    # Fewer cases than categories: don't pay for a call whose
                    # every case would be sliced off again.
                    logger.info("skipping %s: no budget left", category)
                    continue

                # Generate test cases for each category
                template = template_for(category)
                context = self._context_for(category, transcript, template)
                mix = allocate(quota, allowed, weights)

                prompt = f"""
                Based on the following video content, generate test cases for {category}:

                Video Title: {video_info.get('title', 'Unknown')}
                Video Transcript: {context}

                {template}

                Generate exactly {quota} test case(s).
                Use only these priority values: {", ".join(allowed)}.
                Aim for this mix: {", ".join(f"{n} x {p}" for p, n in mix.items() if n)}.
                {JSON_SCHEMA_INSTRUCTION}
                """
                
                try:
                    text, attribution = provider.chat(prompt, model=self.model)
                    self._attributions.append((category, attribution))
                    # Parse and add test cases. The count and the priority set
                    # are requests in the prompt and contracts here.
                    test_cases = self._parse_llm_response(text, category)[:quota]
                    for case in test_cases:
                        case["priority"] = nearest_priority(case["priority"], allowed)
                    all_test_cases.extend(test_cases)
                except Exception as e:
                    logger.warning(f"Failed to generate {category} tests: {e}")
                    # No model produced this: the stub must say so, or it is
                    # indistinguishable from generated output.
                    self._attributions.append(
                        (category, provider.unattributed(f"all models failed: {e}"))
                    )
                    stub = self._create_fallback_test_case(category)
                    stub["priority"] = nearest_priority(stub["priority"], allowed)
                    all_test_cases.append(stub)
            
            return all_test_cases
            
        except Exception as e:
            logger.error(f"Error in generate_comprehensive_tests: {e}")
            return self._create_fallback_test_cases()[:max_cases]

    def _parse_llm_response(self, response_text, category):
        """Parse LLM response into test cases"""
        # Simple parsing - you can make this more sophisticated
        test_cases = []
        
        try:
            # Scan for JSON values with the decoder itself. The previous
            # regex r'\{[^{}]*\}' could not match nested objects, so any test case
            # with a nested "steps" object was silently discarded.
            for value in _scan_json(response_text):
                if isinstance(value, dict) and isinstance(value.get('test_cases'), list):
                    value = value['test_cases']
                for case in (value if isinstance(value, list) else [value]):
                    # Whatever key schema the model chose, one shape comes out.
                    canonical = normalise_case(case, category=category,
                                               index=len(test_cases) + 1)
                    if canonical:
                        test_cases.append(canonical)

            if not test_cases:
                # Fallback: create basic test case from text
                test_cases.append(normalise_case({
                    'id': 'TC001',
                    'title': f'{category} Test Case',
                    'description': response_text[:200] + '...',
                    'priority': 'medium',
                    'steps': ['Execute test'],
                    'expected_result': 'Test passes',
                    'assertions': ['Basic validation'],
                }, category=category))
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            test_cases.append(self._create_fallback_test_case(category))
        
        return test_cases

    def _create_fallback_test_case(self, category):
        """Create a fallback test case when LLM fails"""
        return normalise_case({
            'id': f'TC_FALLBACK_{category}',
            'title': f'{category} Test Case (Fallback)',
            'description': f'Basic {category} test case generated as fallback',
            'priority': 'medium',
            'steps': ['Navigate to application', 'Perform basic interaction'],
            'expected_result': 'The application loads and responds',
            'assertions': ['Application is functional', 'No errors occur'],
        }, category=category)

    def _create_fallback_test_cases(self):
        """Create basic fallback test cases"""
        return [
            self._create_fallback_test_case('Core User Flows'),
            self._create_fallback_test_case('Edge Cases'),
            self._create_fallback_test_case('UI Validation')
        ]

    def format_test_cases(self, test_cases):
        """Format test cases for display.

        Normalisation runs here as well as in the parser: fallback cases and cases
        handed in by a caller never went through the parser, and ids are only
        unique once the whole suite is in one list.
        """
        cases = normalise_cases(test_cases)
        return {
            'test_cases': cases,
            'metadata': {
                'generated_at': datetime.datetime.now().isoformat(),
                'total_cases': len(cases),
                'generator': 'TestGeneratorAgent',
                # 'model' keeps its existing meaning (what we asked for); who actually
                # answered lives under 'attribution'.
                'model': self.model,
                'attribution': provider.summarize(self._attributions)
            }
        }