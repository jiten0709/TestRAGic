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
- Quote only text you actually saw in the material above; do not invent error wording."""


# Template Constants
FUNCTIONAL_TEMPLATE = """You are an expert QA engineer specializing in frontend test automation. 
Generate comprehensive functional test cases for the given user flow.

Return ONLY a valid JSON array of test cases with this exact structure:
[
  {
    "title": "Clear, descriptive test title",
    "description": "Detailed test description",
    "steps": ["Step 1", "Step 2", "Step 3"],
    "expected_result": "Expected outcome",
    "priority": "High|Medium|Low",
    "test_data": "Required test data",
    "preconditions": "Setup requirements"
  }
]

Focus on:
- Happy path scenarios
- Critical user journeys
- Form validations
- Navigation flows
- Data submission and retrieval
- User interactions (clicks, inputs, selections)

Make tests specific, actionable, and directly related to the user flow."""

EDGE_CASE_TEMPLATE = """You are an expert QA engineer specializing in edge case testing.
Generate edge cases and boundary condition tests for the given user flow.

Return ONLY a valid JSON array of edge cases with this exact structure:
[
  {
    "title": "Edge case title",
    "description": "What edge condition this tests",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "Expected behavior",
    "priority": "High|Medium|Low",
    "edge_condition": "Specific boundary/edge condition",
    "risk_level": "High|Medium|Low"
  }
]

Focus on:
- Boundary value testing (min/max inputs)
- Invalid data scenarios
- Network failures and timeouts
- Browser compatibility issues
- Concurrent user actions
- Data corruption scenarios
- System resource limitations
- Security edge cases"""

ACCESSIBILITY_TEMPLATE = """You are an accessibility testing expert following WCAG 2.1 guidelines.
Generate accessibility test cases for the given user flow.

Return ONLY a valid JSON array of accessibility tests with this exact structure:
[
  {
    "title": "Accessibility test title",
    "description": "Accessibility requirement being tested",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "Accessible behavior expected",
    "priority": "High|Medium|Low",
    "wcag_guideline": "Relevant WCAG guideline",
    "assistive_technology": "Screen reader|Keyboard navigation|Voice control"
  }
]

Focus on:
- Keyboard navigation
- Screen reader compatibility
- Color contrast and visual accessibility
- Focus management
- Alternative text for images
- Form label associations
- Semantic HTML structure
- ARIA attributes"""

PERFORMANCE_TEMPLATE = """You are a performance testing expert.
Generate performance test scenarios for the given user flow.

Return ONLY a valid JSON array of performance tests with this exact structure:
[
  {
    "title": "Performance test title",
    "description": "Performance aspect being tested",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "Performance criteria",
    "priority": "High|Medium|Low",
    "metric": "Load time|Response time|Memory usage",
    "threshold": "Performance threshold (e.g., < 2 seconds)"
  }
]

Focus on:
- Page load times
- API response times
- Memory consumption
- CPU usage
- Network efficiency
- Large dataset handling
- Concurrent user load
- Mobile performance"""

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
        
        # Test case templates
        self.templates = {
            "functional": self._get_functional_template(),
            "ui": self._get_ui_template(),
            "integration": self._get_integration_template(),
            "edge_case": self._get_edge_case_template()
        }
        
        logger.info(f"TestGeneratorAgent initialized with model: {self.model}")

    def _get_functional_template(self):
        """Get functional test case template"""
        return """
        Generate functional test cases for the given application based on the video content.

        Focus on:
        1. Core user workflows shown in the video
        2. UI interactions and validations
        3. Data input/output scenarios
        4. Navigation flows
        5. Error handling scenarios
        
        Generate practical, executable test cases that can be automated with Playwright.
        """

    def _get_ui_template(self):
        """Get UI test case template"""
        return """
        Generate UI-focused test cases based on the video content.
        
        Focus on:
        1. Element visibility and positioning
        2. Interactive elements (buttons, forms, links)
        3. Responsive design aspects
        4. Visual validation
        5. Cross-browser compatibility
        
        Generate test cases that verify the user interface works correctly.
        """

    def _get_integration_template(self):
        """Get integration test case template"""
        return """
        Generate integration test cases based on the video content.
        
        Focus on:
        1. API interactions shown in the video
        2. Data flow between components
        3. Third-party service integrations
        4. Database operations
        5. System interactions
        
        Generate test cases that verify different system components work together.
        """

    def _get_edge_case_template(self):
        """Get edge case test template"""
        return """
        Generate edge case and negative test scenarios.
        
        Focus on:
        1. Invalid input handling
        2. Boundary conditions
        3. Error scenarios
        4. Network failures
        5. Security edge cases
        
        Generate test cases that verify system robustness.
        """
    
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

        Without a retriever every category gets `transcript[:2000]` -- the same
        first ~2000 *characters* of the video, so accessibility tests are written
        from whatever was said in the opening ninety seconds. Retrieval queries the
        vector store instead, which is also what makes the chosen embedding model
        affect the output at all.
        """
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
                template = self.templates.get(category.lower().replace(' ', '_'), self.templates['functional'])
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