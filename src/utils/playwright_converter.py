"""
This complete implementation provides:

Natural Language Parsing - Converts test steps to Playwright actions
Intelligent Selector Generation - Creates appropriate selectors based on UI elements
Action Mapping - Maps common actions to Playwright methods
Error Handling - Includes try-catch blocks and failure artifacts
Test Suite Generation - Creates complete test files with proper structure
Assertion Generation - Creates assertions based on expected results
Test Runner - Generates pytest runner scripts
File Management - Saves tests to organized directory structure
"""

import re
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="utils.log")

# Global constants
DEFAULT_TIMEOUT = 30000  # 30 seconds
DEFAULT_WAIT_TIMEOUT = 5000  # 5 seconds
SCREENSHOT_ON_FAILURE = True
VIDEO_RECORDING = True
TRACE_ON_FAILURE = True

# Action mapping patterns
ACTION_PATTERNS = {
    # Navigation patterns
    'navigate': [r'navigate to|go to|visit|open', 'goto'],
    'reload': [r'reload|refresh', 'reload'],
    'back': [r'go back|navigate back', 'goBack'],
    'forward': [r'go forward|navigate forward', 'goForward'],
    
    # Click patterns
    'click': [r'click on|click|press|tap', 'click'],
    'double_click': [r'double click|double-click', 'dblclick'],
    'right_click': [r'right click|right-click|context click', 'click', {'button': 'right'}],
    
    # Input patterns
    'fill': [r'type|enter|input|fill', 'fill'],
    'clear': [r'clear|empty', 'clear'],
    'upload': [r'upload|select file', 'setInputFiles'],
    
    # Selection patterns
    'select': [r'select option|choose|pick', 'selectOption'],
    'check': [r'check|tick', 'check'],
    'uncheck': [r'uncheck|untick', 'uncheck'],
    
    # Hover and focus
    'hover': [r'hover|mouse over', 'hover'],
    'focus': [r'focus on|focus', 'focus'],
    
    # Keyboard actions
    'press': [r'press key|press|hit', 'press'],
    
    # Wait actions
    'wait': [r'wait for|wait|pause', 'waitForTimeout'],
    'wait_for_element': [r'wait for element|wait until visible', 'waitForSelector'],
    'wait_for_page': [r'wait for page|wait for load', 'waitForLoadState']
}

# Selector patterns for UI elements
SELECTOR_PATTERNS = {
    'button': [r'button', 'button', 'input[type="button"]', 'input[type="submit"]', '[role="button"]'],
    'link': [r'link', 'a', '[role="link"]'],
    'input': [r'input|field|textbox', 'input', 'textarea', '[contenteditable]'],
    'dropdown': [r'dropdown|select|combobox', 'select', '[role="combobox"]', '[role="listbox"]'],
    'checkbox': [r'checkbox', 'input[type="checkbox"]', '[role="checkbox"]'],
    'radio': [r'radio button|radio', 'input[type="radio"]', '[role="radio"]'],
    'menu': [r'menu', '[role="menu"]', '.menu', '#menu'],
    'modal': [r'modal|dialog|popup', '[role="dialog"]', '.modal', '.popup'],
    'form': [r'form', 'form'],
    'table': [r'table', 'table', '[role="table"]'],
    'header': [r'header|heading', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', '[role="heading"]'],
    'navigation': [r'navigation|nav', 'nav', '[role="navigation"]']
}

class PlaywrightConverter:
    def __init__(self):
        """Initialize Playwright Converter"""
        self.generated_tests_dir = Path("src/tests/generated")
        self.generated_tests_dir.mkdir(parents=True, exist_ok=True)
        
        # Test file templates
        self.test_template = self._get_test_template()
        self.imports_template = self._get_imports_template()
        
        logger.info("Playwright Converter initialized")
    
    def convert_test_cases(self, test_cases: Dict, base_url: str = "") -> Dict:
        """Convert test cases to Playwright scripts"""
        try:
            converted_tests = {}
            
            for category, tests in test_cases.items():
                if tests:
                    playwright_code = self._convert_category_tests(tests, category, base_url)
                    converted_tests[category] = playwright_code
            
            return {
                "success": True,
                "converted_tests": converted_tests,
                "total_categories": len(converted_tests)
            }
            
        except Exception as e:
            logger.error(f"Error converting test cases: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _convert_category_tests(self, tests: List[Dict], category: str, base_url: str) -> str:
        """Convert a category of tests to Playwright code"""
        test_methods = []
        
        for i, test in enumerate(tests, 1):
            method_name = self._sanitize_method_name(test.get("title", f"test_{category}_{i}"))
            test_method = self._convert_single_test(test, method_name, base_url)
            test_methods.append(test_method)
        
        # Combine all test methods
        full_test_code = self.imports_template + "\n\n"
        full_test_code += f"class Test{category.title().replace('_', '')}:\n"
        full_test_code += "\n".join(test_methods)
        
        return full_test_code
    
    def _convert_single_test(self, test: Dict, method_name: str, base_url: str) -> str:
        """Convert a single test case to Playwright method"""
        title = test.get("title", "Untitled Test")
        description = test.get("description", "")
        steps = test.get("steps", [])
        expected_result = test.get("expected_result", "")
        
        # Generate test method
        test_code = f"""
    def test_{method_name}(self, page):
        \"\"\"
        Test: {title}
        Description: {description}
        Expected Result: {expected_result}
        \"\"\"
        try:
"""
        
        # Convert steps to Playwright actions
        for step_num, step in enumerate(steps, 1):
            playwright_action = self._convert_step_to_playwright(step, base_url)
            test_code += f"            # Step {step_num}: {step}\n"
            test_code += f"            {playwright_action}\n"
            test_code += f"            page.wait_for_timeout(500)  # Brief pause between steps\n\n"
        
        # Add assertion if expected result is provided
        if expected_result:
            assertion = self._generate_assertion(expected_result)
            test_code += f"            # Verify expected result\n"
            test_code += f"            {assertion}\n\n"
        
        # Add error handling
        test_code += """        except Exception as e:
            # Capture screenshot on failure
            page.screenshot(path=f"src/tests/artifacts/screenshots/{test_name}_failure.png")
            raise e
"""
        
        return test_code
    
    def _convert_step_to_playwright(self, step: str, base_url: str) -> str:
        """Convert a natural language step to Playwright action"""
        step_lower = step.lower().strip()
        
        # Extract URL for navigation
        if any(pattern in step_lower for pattern in ['navigate', 'go to', 'visit', 'open']):
            url = self._extract_url(step, base_url)
            return f"page.goto('{url}')"
        
        # Extract and convert actions
        action, selector, value, options = self._parse_step_components(step)
        
        if not action:
            return f"# TODO: Parse step manually - {step}"
        
        # Generate Playwright action
        playwright_action = self._generate_playwright_action(action, selector, value, options)
        
        return playwright_action
    
    def _parse_step_components(self, step: str) -> Tuple[str, str, str, Dict]:
        """Parse step into action, selector, value, and options"""
        step_lower = step.lower()
        
        # Find action
        action = self._detect_action(step_lower)
        
        # Extract selector
        selector = self._extract_selector(step)
        
        # Extract value (for input actions)
        value = self._extract_value(step, action)
        
        # Extract additional options
        options = self._extract_options(step, action)
        
        return action, selector, value, options
    
    def _detect_action(self, step: str) -> str:
        """Detect the action type from step description"""
        for action, patterns in ACTION_PATTERNS.items():
            for pattern in patterns[0].split('|'):
                if re.search(rf'\b{pattern}\b', step):
                    return action
        
        return ""
    
    def _extract_selector(self, step: str) -> str:
        """Extract or generate selector from step description"""
        step_lower = step.lower()
        
        # Look for specific text in quotes
        quoted_text = re.search(r'"([^"]+)"', step)
        if quoted_text:
            text = quoted_text.group(1)
            return f'text="{text}"'
        
        # Look for ID patterns
        id_match = re.search(r'#(\w+)', step)
        if id_match:
            return f"#{id_match.group(1)}"
        
        # Look for class patterns
        class_match = re.search(r'\.(\w+)', step)
        if class_match:
            return f".{class_match.group(1)}"
        
        # Look for UI element types and generate appropriate selectors
        for element_type, patterns in SELECTOR_PATTERNS.items():
            if any(re.search(rf'\b{pattern}\b', step_lower) for pattern in patterns[0].split('|')):
                # Return the most specific selector for this element type
                selectors = patterns[1:]
                return selectors[0] if selectors else f'[role="{element_type}"]'
        
        # Extract text that might be clickable
        words = step.split()
        for i, word in enumerate(words):
            if word.lower() in ['button', 'link', 'menu', 'field']:
                if i > 0:
                    # Try to get the preceding word as identifier
                    identifier = words[i-1].strip('"\'.,!?')
                    return f'text="{identifier}"'
        
        # Fallback: try to extract any meaningful text
        meaningful_words = [w for w in step.split() if len(w) > 3 and w.isalpha()]
        if meaningful_words:
            return f'text="{meaningful_words[-1]}"'
        
        return "# TODO: Define selector"
    
    def _extract_value(self, step: str, action: str) -> str:
        """Extract input value from step"""
        if action not in ['fill', 'select', 'upload']:
            return ""
        
        # Look for quoted values
        quoted_value = re.search(r'"([^"]+)"', step)
        if quoted_value:
            return quoted_value.group(1)
        
        # Look for specific patterns
        if action == 'fill':
            # Look for "type X" or "enter X"
            type_match = re.search(r'(?:type|enter|input)\s+["\']?([^"\']+)["\']?', step, re.IGNORECASE)
            if type_match:
                return type_match.group(1).strip()
        
        elif action == 'select':
            # Look for "select X" or "choose X"
            select_match = re.search(r'(?:select|choose)\s+["\']?([^"\']+)["\']?', step, re.IGNORECASE)
            if select_match:
                return select_match.group(1).strip()
        
        return ""
    
    def _extract_options(self, step: str, action: str) -> Dict:
        """Extract additional options from step"""
        options = {}
        
        # Extract timeout if specified
        timeout_match = re.search(r'(\d+)\s*(?:seconds?|ms|milliseconds?)', step)
        if timeout_match:
            time_value = int(timeout_match.group(1))
            unit = timeout_match.group(0).lower()
            if 'ms' in unit or 'millisecond' in unit:
                options['timeout'] = time_value
            else:
                options['timeout'] = time_value * 1000
        
        # Extract keyboard modifiers
        if 'ctrl' in step.lower() or 'cmd' in step.lower():
            options['modifiers'] = ['Meta'] if 'cmd' in step.lower() else ['Control']
        
        # Extract click options
        if action == 'click':
            if 'right' in step.lower():
                options['button'] = 'right'
            elif 'middle' in step.lower():
                options['button'] = 'middle'
        
        return options
    
    def _generate_playwright_action(self, action: str, selector: str, value: str, options: Dict) -> str:
        """Generate Playwright action code"""
        if action not in ACTION_PATTERNS:
            return f"# TODO: Implement action '{action}' for selector '{selector}'"
        
        playwright_method = ACTION_PATTERNS[action][1]
        
        # Build action call
        if action == 'fill' and value:
            action_call = f"page.{playwright_method}('{selector}', '{value}'"
        elif action == 'select' and value:
            action_call = f"page.{playwright_method}('{selector}', '{value}'"
        elif action == 'upload' and value:
            action_call = f"page.{playwright_method}('{selector}', '{value}'"
        elif action == 'press' and value:
            action_call = f"page.{playwright_method}('{value}'"
        elif action == 'wait' and options.get('timeout'):
            action_call = f"page.{playwright_method}({options['timeout']}"
        elif action == 'wait_for_element':
            timeout = options.get('timeout', DEFAULT_TIMEOUT)
            action_call = f"page.{playwright_method}('{selector}', timeout={timeout}"
        else:
            action_call = f"page.{playwright_method}('{selector}'"
        
        # Add options
        if options and action not in ['wait', 'press']:
            for key, val in options.items():
                if key != 'timeout' or action in ['wait_for_element']:
                    if isinstance(val, str):
                        action_call += f", {key}='{val}'"
                    else:
                        action_call += f", {key}={val}"
        
        action_call += ")"
        
        return action_call
    
    def _extract_url(self, step: str, base_url: str) -> str:
        """Extract URL from navigation step"""
        # Look for full URLs
        url_match = re.search(r'https?://[^\s]+', step)
        if url_match:
            return url_match.group(0)
        
        # Look for relative paths
        path_match = re.search(r'/[^\s]*', step)
        if path_match and base_url:
            return base_url.rstrip('/') + path_match.group(0)
        
        # Look for quoted URLs or paths
        quoted_match = re.search(r'"([^"]+)"', step)
        if quoted_match:
            url = quoted_match.group(1)
            if url.startswith('http'):
                return url
            elif base_url:
                return base_url.rstrip('/') + '/' + url.lstrip('/')
        
        # Default to base URL
        return base_url or "about:blank"
    
    def _generate_assertion(self, expected_result: str) -> str:
        """Generate assertion based on expected result"""
        expected_lower = expected_result.lower()
        
        # Check for visibility assertions
        if 'visible' in expected_lower or 'displayed' in expected_lower:
            return "assert page.is_visible('selector')  # TODO: Define selector"
        
        # Check for text assertions
        if 'text' in expected_lower or 'message' in expected_lower:
            return f"assert '{expected_result}' in page.text_content('body')"
        
        # Check for URL assertions
        if 'url' in expected_lower or 'page' in expected_lower:
            return "assert 'expected_url' in page.url  # TODO: Define expected URL"
        
        # Check for success indicators
        if 'success' in expected_lower or 'complete' in expected_lower:
            return "assert page.is_visible('.success')  # TODO: Define success indicator"
        
        # Default assertion
        return f"# TODO: Implement assertion for: {expected_result}"
    
    def _sanitize_method_name(self, title: str) -> str:
        """Sanitize test title to valid Python method name"""
        # Remove special characters and replace with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', title)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_').lower()
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'test_' + sanitized
        
        return sanitized or 'unnamed_test'
    
    def save_test_file(self, test_code: str, filename: str) -> str:
        """Save generated test code to file"""
        try:
            file_path = self.generated_tests_dir / f"{filename}.py"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            logger.info(f"Test file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving test file: {str(e)}")
            return ""
    
    def generate_complete_test_suite(self, test_cases: Dict, suite_name: str, base_url: str = "") -> Dict:
        """Generate complete test suite with multiple categories"""
        try:
            converted_result = self.convert_test_cases(test_cases, base_url)
            
            if not converted_result["success"]:
                return converted_result
            
            # Save individual test files
            saved_files = []
            for category, test_code in converted_result["converted_tests"].items():
                filename = f"{suite_name}_{category}"
                file_path = self.save_test_file(test_code, filename)
                if file_path:
                    saved_files.append(file_path)
            
            # Generate test suite runner
            runner_code = self._generate_test_runner(saved_files, suite_name)
            runner_path = self.save_test_file(runner_code, f"{suite_name}_runner")
            
            return {
                "success": True,
                "test_files": saved_files,
                "runner_file": runner_path,
                "total_files": len(saved_files) + 1
            }
            
        except Exception as e:
            logger.error(f"Error generating test suite: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_runner(self, test_files: List[str], suite_name: str) -> str:
        """Generate test runner script"""
        runner_code = f'''
        """
        Test Runner for {suite_name} Test Suite
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        import pytest
        import sys
        from pathlib import Path

        # Add src to path for imports
        sys.path.append(str(Path(__file__).parent.parent.parent))

        def run_test_suite():
            """Run the complete test suite"""
            test_files = {test_files}
            
            # Run tests with pytest
            pytest.main([
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                "--html=src/tests/results/report.html",  # HTML report
                "--self-contained-html",  # Self-contained HTML
                *test_files
            ])

        if __name__ == "__main__":
            run_test_suite()
        '''
        return runner_code
    
    def _get_imports_template(self) -> str:
        """Get standard imports for Playwright tests"""
        return '''
            """
            Generated Playwright Test Cases
            Auto-generated by QA Agent - Do not edit manually
            """

            import pytest
            from playwright.sync_api import Page, Browser, BrowserContext
            import os
            from pathlib import Path

            # Test configuration
            BASE_URL = os.getenv("BASE_URL", "http://localhost:3000")
            TIMEOUT = 30000  # 30 seconds

            @pytest.fixture(scope="function")
            def context(browser: Browser):
                """Create a new browser context for each test"""
                context = browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    record_video_dir="src/tests/artifacts/videos/",
                    record_video_size={"width": 1280, "height": 720}
                )
                yield context
                context.close()

            @pytest.fixture(scope="function")
            def page(context: BrowserContext):
                """Create a new page for each test"""
                page = context.new_page()
                page.set_default_timeout(TIMEOUT)
                yield page
                page.close()
            '''
    
    def _get_test_template(self) -> str:
        """Get basic test method template"""
        return '''
            def test_{method_name}(self, page):
                """
                {test_description}
                """
                try:
                    {test_steps}
                except Exception as e:
                    # Capture artifacts on failure
                    page.screenshot(path=f"src/tests/artifacts/screenshots/{method_name}_failure.png")
                    raise e
        '''