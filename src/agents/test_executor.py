"""Run generated Playwright suites for real and report what actually happened.

The suites this executes are ordinary pytest modules written by
`PlaywrightConverter`, so the right way to run them is pytest itself -- in a
subprocess, with the repo's own `addopts` neutralised (they force three browsers
and `--headed`). Results come back as JUnit XML, which pytest writes natively and
`xml.etree` reads, so no reporting plugin is involved.

This replaces an in-process interpreter that re-parsed English steps at run time
with its own hardcoded selectors (`#login-button`, `test@example.com`). One step
parser is enough, and it lives in PlaywrightConverter.
"""

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="agents.log")

# One output root. The dashboard reads src/data/; the executor used to write
# src/tests/, so its results were invisible to the Results page.
RESULTS_DIR = Path("src/data/test_results")
ARTIFACTS_DIR = Path("src/data/test_artifacts")

DEFAULT_RUN_TIMEOUT = 900  # seconds for the whole pytest process


class TestExecutorAgent:
    def __init__(self, results_dir=None, artifacts_dir=None, timeout=DEFAULT_RUN_TIMEOUT):
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self.timeout = timeout
        for directory in (self.results_dir, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def execute_tests(self, test_files: List[str], browser: str = "chromium",
                      headless: bool = True, base_url: str = "",
                      capture: tuple = ("screenshots", "traces")) -> Dict:
        """Run the given pytest modules in one browser. Never raises."""
        execution_id = f"exec_{int(time.time())}_{browser}"
        started = time.time()
        results = {
            "execution_id": execution_id,
            "start_time": datetime.now().isoformat(),
            "browser": browser,
            "base_url": base_url,
            "simulated": False,          # this ran a real browser; see app.py's labelling
            "total_tests": 0, "passed": 0, "failed": 0, "skipped": 0,
            "test_results": [],
            "duration": 0.0,
        }

        existing = [f for f in test_files if Path(f).exists()]
        if not existing:
            results.update(status="failed", error="no test files to run",
                           end_time=datetime.now().isoformat())
            logger.error("execution %s aborted: no test files exist", execution_id)
            return results

        report_xml = self.artifacts_dir / f"{execution_id}.xml"
        command = self._pytest_command(existing, browser, headless, report_xml, capture)
        logger.info("executing %s: %s", execution_id, " ".join(command))

        try:
            completed = subprocess.run(
                command, cwd=Path.cwd(), capture_output=True, text=True,
                timeout=self.timeout, env=self._env(base_url),
            )
            results["stdout_tail"] = completed.stdout[-4000:]
            results["exit_code"] = completed.returncode
        except subprocess.TimeoutExpired:
            results.update(status="failed", error=f"pytest exceeded {self.timeout}s",
                           duration=time.time() - started,
                           end_time=datetime.now().isoformat())
            logger.error("execution %s timed out after %ss", execution_id, self.timeout)
            return results

        results["test_results"] = self.parse_junit(report_xml)
        for case in results["test_results"]:
            results["total_tests"] += 1
            results[case["status"]] = results.get(case["status"], 0) + 1

        results["duration"] = time.time() - started
        results["end_time"] = datetime.now().isoformat()
        # Exit code 5 is "no tests collected" -- a real failure of the conversion
        # step, not a passing run, and it must not read as success.
        results["status"] = "completed" if results["total_tests"] else "failed"
        if not results["total_tests"]:
            results["error"] = (f"pytest collected no tests (exit {results.get('exit_code')}). "
                                f"{(results.get('stdout_tail') or '')[-500:]}")

        logger.info("execution %s: %d passed, %d failed, %d skipped in %.1fs",
                    execution_id, results["passed"], results["failed"],
                    results["skipped"], results["duration"])
        return results

    # ------------------------------------------------------------------
    def _pytest_command(self, test_files, browser, headless, report_xml, capture) -> List[str]:
        command = [
            sys.executable, "-m", "pytest",
            # The repo's own addopts pin three browsers and --headed; a generated
            # suite must be run on exactly the browser that was asked for.
            "-o", "addopts=",
            "-p", "no:cacheprovider",
            "--browser", browser,
            f"--junitxml={report_xml}",
            f"--output={self.artifacts_dir}",
            "-q",
        ]
        if not headless:
            command.append("--headed")
        command.append("--screenshot=" + ("only-on-failure" if "screenshots" in capture else "off"))
        command.append("--video=" + ("retain-on-failure" if "videos" in capture else "off"))
        command.append("--tracing=" + ("retain-on-failure" if "traces" in capture else "off"))
        return command + list(test_files)

    @staticmethod
    def _env(base_url: str) -> Dict:
        import os
        env = dict(os.environ)
        if base_url:
            # Read by conftest.py's base_url fixture and by generated modules.
            env["BASE_URL"] = base_url
        return env

    @staticmethod
    def parse_junit(report_xml: Path) -> List[Dict]:
        """JUnit XML -> one dict per test. Empty list when pytest wrote nothing."""
        if not Path(report_xml).exists():
            return []
        try:
            root = ET.parse(report_xml).getroot()
        except ET.ParseError as e:
            logger.error("could not parse %s: %s", report_xml, e)
            return []

        cases = []
        for node in root.iter("testcase"):
            status, message = "passed", ""
            for outcome, label in (("failure", "failed"), ("error", "failed"), ("skipped", "skipped")):
                child = node.find(outcome)
                if child is not None:
                    status = label
                    message = (child.get("message") or child.text or "").strip()
                    break
            # pytest-playwright parametrizes on the browser, so the name arrives as
            # `test_sign_in[chromium]`; the converter's case index is keyed on the
            # bare function name.
            raw_name = node.get("name", "")
            cases.append({
                "name": raw_name.split("[", 1)[0],
                "raw_name": raw_name,
                "classname": node.get("classname", ""),
                "status": status,
                "duration": float(node.get("time") or 0.0),
                "error": message.splitlines()[-1][:500] if message else "",
            })
        return cases
