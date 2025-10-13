# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Periodic JUnit XML Reporter for pytest.

This module provides a lightweight periodic JUnit XML reporter that uses incremental
append strategy for ultra-fast periodic saves, ideal for very large test suites.
"""

import datetime
import os
import platform
import signal
import time


class PeriodicJUnitXML:
    """
    Periodic JUnit XML reporter using incremental append.

    This version is efficient by appending test results incrementally
    instead of rewriting the entire file each time. Best for very large test suites.

    Trade-offs:
    + Extremely fast saves (constant time)
    + Minimal memory usage
    - More complex XML handling
    - Final cleanup needed to consolidate statistics

    Usage:
        Add to conftest.py:

        def pytest_configure(config):
            from utils.periodic_junit import PeriodicJUnitXML

            reporter = PeriodicJUnitXML(
                xmlpath='results/junit_report.xml',
                interval=1800,  # Save every 30 minutes
                batch_size=10   # Or save every 10 tests
            )
            reporter.pytest_configure(config)
            config.pluginmanager.register(reporter, 'periodic_junit')
    """

    def __init__(
            self,
            xmlpath: str,
            interval: int = 1800,
            batch_size: int = 10,
            logger=None,  # Optional logger (print_info, print_warning functions)
    ):
        """
        Initialize periodic reporter.

        Args:
            xmlpath: Path to the output XML file
            interval: Time interval in seconds between saves (default: 1800 = 30 min)
            batch_size: Number of tests before triggering a save (default: 10)
            logger: Optional dictionary with 'info' and 'warning' functions for logging
        """
        self.xmlpath = os.path.abspath(xmlpath)
        self.time_interval = interval
        self.batch_size = batch_size
        self.logger = logger or {}

        self.completed_tests = 0
        self.last_save_time = time.time()
        self.suite_start_time = time.time()

        # Initialize statistics and tracking structures
        self.__init_stats()

    def _log_info(self, message):
        """Log info message."""
        if 'info' in self.logger:
            self.logger['info'](message)
        else:
            print(f"INFO: {message}")

    def _log_warning(self, message):
        """Log warning message."""
        if 'warning' in self.logger:
            self.logger['warning'](message)
        else:
            print(f"WARNING: {message}")

    @staticmethod
    def _sanitize_xml_text(text):
        """
        Sanitize text for XML by removing illegal characters.

        XML 1.0 only allows:
        - #x9 (TAB)
        - #xA (LF)
        - #xD (CR)
        - #x20-#xD7FF
        - #xE000-#xFFFD
        - #x10000-#x10FFFF

        This removes:
        - ANSI escape sequences (terminal colors)
        - Other control characters
        """
        if not text:
            return text

        import re

        # Remove ANSI escape sequences (e.g., \x1b[31m for colors)
        # Pattern: ESC [ ... m (most common)
        # Also handles: ESC ] ... BEL/ESC\ (less common)
        ansi_escape_pattern = re.compile(
            r'\x1b'  # ESC character
            r'(?:'  # Start non-capturing group
            r'\[[0-?]*[ -/]*[@-~]'  # CSI sequences: ESC [ ... letter
            r'|].*?(?:\x07|\x1b\\)'  # OSC sequences: ESC ] ... BEL/ESC\
            r'|[()][AB012]'  # Character set selection
            r')')
        text = ansi_escape_pattern.sub('', text)

        # Remove other control characters except allowed ones
        # Keep: tab(0x09), newline(0x0A), carriage return(0x0D)
        def is_valid_xml_char(c):
            codepoint = ord(c)
            return (codepoint == 0x09 or codepoint == 0x0A or codepoint == 0x0D
                    or (0x20 <= codepoint <= 0xD7FF)
                    or (0xE000 <= codepoint <= 0xFFFD)
                    or (0x10000 <= codepoint <= 0x10FFFF))

        # Filter out invalid characters
        text = ''.join(c if is_valid_xml_char(c) else '' for c in text)

        return text

    @staticmethod
    def _format_classname(classname):
        """
        Format classname for JUnit XML report.

        Removes .py extension and converts path separators to dots
        to create a proper package/class hierarchy.

        Examples:
            'tests/integration/test_foo.py' -> 'tests.integration.test_foo'
            'H100.examples.test_eagle' -> 'H100.examples.test_eagle'
            'path/to/test_bar.py::TestClass' -> 'path.to.test_bar.TestClass'
        """
        if not classname:
            return classname

        # Remove .py extension
        if classname.endswith('.py'):
            classname = classname[:-3]

        # Replace path separators with dots
        classname = classname.replace('/', '.').replace('\\', '.')

        # Remove leading/trailing dots
        classname = classname.strip('.')

        return classname

    @staticmethod
    def _extract_error_message(report):
        """
        Extract a concise error message from a test report.

        This extracts the most relevant error information for the 'message' attribute
        of error/failure elements in JUnit XML.

        Returns a string suitable for the message attribute.
        """
        # Try to get exception info first
        if hasattr(report, 'longrepr') and report.longrepr:
            try:
                # For ExceptionInfo objects
                if hasattr(report.longrepr, 'reprcrash'):
                    crash = report.longrepr.reprcrash
                    if crash:
                        # Get the crash message (usually "file:line: ExceptionType: message")
                        return str(crash.message) if hasattr(
                            crash, 'message') else str(crash)

                # For string representations
                longrepr_str = str(report.longrepr)

                # Try to extract the last line which usually contains the key error
                lines = longrepr_str.strip().split('\n')

                # Look for exception type and message (usually in the last few lines)
                for line in reversed(lines[-10:]):  # Check last 10 lines
                    line = line.strip()
                    # Common patterns: "ExceptionType: message" or "E   AssertionError: ..."
                    if line and (': ' in line or line.startswith('E ')):
                        # Remove leading 'E ' from pytest output
                        if line.startswith('E '):
                            line = line[2:].strip()

                        # Limit length to avoid too long messages
                        if len(line) > 200:
                            line = line[:197] + '...'

                        return line

                # Fallback: use first non-empty line
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('_'):
                        if len(line) > 200:
                            line = line[:197] + '...'
                        return line

            except Exception:
                pass

        # Fallback to longreprtext if available
        if hasattr(report, 'longreprtext') and report.longreprtext:
            lines = report.longreprtext.strip().split('\n')
            for line in reversed(lines[-5:]):
                line = line.strip()
                if line:
                    if len(line) > 200:
                        line = line[:197] + '...'
                    return line

        # Last resort: generic message
        if report.failed:
            return f"{report.when} failed"
        elif report.skipped:
            return "skipped"
        else:
            return "test issue"

    def __init_stats(self):
        """Initialize test statistics tracking."""
        # Statistics tracking
        self.stats = {
            'tests': 0,
            'errors': 0,
            'failures': 0,
            'skipped': 0,
            'passed': 0,
            'time': 0.0,
        }

        # Pending test cases buffer
        self.pending_cases = []

        # Store test reports by nodeid to accumulate across phases (setup, call, teardown)
        self.test_reports = {}

        # File handle for incremental writes
        self.xml_file = None
        self.header_written = False

    def pytest_configure(self, config):
        """Configure and initialize output file."""
        os.makedirs(os.path.dirname(self.xmlpath), exist_ok=True)

        # Open file for incremental writing
        self.xml_file = open(self.xmlpath, 'w', encoding='utf-8')

        # Write XML header and opening tags
        self.xml_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
        self.xml_file.write('<testsuites>\n')
        self.xml_file.write('  <testsuite name="pytest">\n')
        self.xml_file.flush()
        self.header_written = True

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        self._log_info(f"PeriodicJUnitXML: Initialized at {self.xmlpath}")

    def pytest_runtest_logreport(self, report):
        """Collect test results and save periodically."""
        nodeid = report.nodeid

        # Accumulate reports for all phases (setup, call, teardown)
        if nodeid not in self.test_reports:
            self.test_reports[nodeid] = {
                'setup': None,
                'call': None,
                'teardown': None,
                'duration': 0.0,
                'outcome': 'passed',
            }

        # Store the report for this phase
        self.test_reports[nodeid][report.when] = report
        self.test_reports[nodeid]['duration'] += getattr(
            report, 'duration', 0.0)

        # Update outcome (failed takes precedence over skipped)
        if report.failed:
            self.test_reports[nodeid]['outcome'] = 'failed'
        elif report.skipped and self.test_reports[nodeid]['outcome'] != 'failed':
            self.test_reports[nodeid]['outcome'] = 'skipped'

        # Only process on teardown phase (when test is fully complete)
        if report.when != 'teardown':
            return

        # Extract test case information from all accumulated reports
        test_case = self._create_testcase_xml_from_reports(
            nodeid, self.test_reports[nodeid])
        self.pending_cases.append(test_case)

        # Update statistics
        self.stats['tests'] += 1
        self.stats['time'] += self.test_reports[nodeid]['duration']

        outcome = self.test_reports[nodeid]['outcome']
        if outcome == 'failed':
            self.stats['failures'] += 1
        elif outcome == 'skipped':
            self.stats['skipped'] += 1
        else:
            self.stats['passed'] += 1

        # Clean up stored reports for this test
        del self.test_reports[nodeid]

        self.completed_tests += 1
        current_time = time.time()

        # Check if we should flush pending cases
        if (self.completed_tests >= self.batch_size
                or current_time - self.last_save_time >= self.time_interval):
            self._flush_pending_cases()
            self.completed_tests = 0
            self.last_save_time = current_time

    def _create_testcase_xml_from_reports(self, nodeid, test_info):
        """Create XML string for a single test case from accumulated reports."""
        import xml.etree.ElementTree as ET

        # Get the primary report (prefer call, then teardown, then setup)
        primary_report = (test_info.get('call') or test_info.get('teardown')
                          or test_info.get('setup'))

        if not primary_report:
            return None

        testcase = ET.Element(
            'testcase',
            classname=self._format_classname(primary_report.location[0]),
            name=primary_report.location[2],
            time=f"{test_info['duration']:.3f}",
        )

        # Add file and line attributes if available
        if hasattr(primary_report, 'location') and len(
                primary_report.location) > 1:
            testcase.set('file', primary_report.location[0])
            if primary_report.location[1] is not None:
                testcase.set('line', str(primary_report.location[1]))

        # Add failure/error/skip information
        outcome = test_info['outcome']
        failed_report = None

        # Check each phase for failures
        for phase in ['setup', 'call', 'teardown']:
            report = test_info.get(phase)
            if report and report.failed:
                failed_report = report

                # Extract meaningful error message
                error_msg = self._extract_error_message(report)
                error_msg = self._sanitize_xml_text(error_msg)

                if phase == 'setup':
                    failure = ET.SubElement(testcase,
                                            'error',
                                            message=error_msg)
                else:
                    failure = ET.SubElement(testcase,
                                            'failure',
                                            message=error_msg)

                # Add full traceback as element text
                if hasattr(report, 'longreprtext'):
                    failure.text = self._sanitize_xml_text(report.longreprtext)
                break

        # Check for skipped
        if not failed_report and outcome == 'skipped':
            for phase in ['setup', 'call', 'teardown']:
                report = test_info.get(phase)
                if report and report.skipped:
                    # Extract skip reason
                    skip_msg = self._extract_error_message(report)
                    skip_msg = self._sanitize_xml_text(
                        skip_msg) if skip_msg else 'skipped'

                    skip = ET.SubElement(testcase, 'skipped', message=skip_msg)
                    if hasattr(report, 'longreprtext'):
                        skip.text = self._sanitize_xml_text(report.longreprtext)
                    break

        # Collect captured output from ALL phases (setup, call, teardown)
        # This is the key part for --junit_logging functionality
        stdout_content = []
        stderr_content = []

        for phase in ['setup', 'call', 'teardown']:
            report = test_info.get(phase)
            if not report:
                continue

            # Method 1: Check for sections attribute (pytest's captured output)
            if hasattr(report, 'sections') and report.sections:
                for section_name, section_content in report.sections:
                    if section_content:
                        if 'stdout' in section_name.lower(
                        ) or 'Captured stdout' in section_name:
                            stdout_content.append(
                                f'--- {phase} {section_name} ---\n{section_content}'
                            )
                        elif 'stderr' in section_name.lower(
                        ) or 'Captured stderr' in section_name:
                            stderr_content.append(
                                f'--- {phase} {section_name} ---\n{section_content}'
                            )
                        elif 'log' in section_name.lower(
                        ) or 'Captured log' in section_name:
                            # Logs typically go to stdout
                            stdout_content.append(
                                f'--- {phase} {section_name} ---\n{section_content}'
                            )

            # Method 2: Fallback to direct capture attributes
            if hasattr(report, 'capstdout') and report.capstdout:
                stdout_content.append(
                    f'--- {phase} stdout ---\n{report.capstdout}')
            if hasattr(report, 'capstderr') and report.capstderr:
                stderr_content.append(
                    f'--- {phase} stderr ---\n{report.capstderr}')
            if hasattr(report, 'caplog') and report.caplog:
                stdout_content.append(
                    f'--- {phase} Captured Log ---\n{report.caplog}')

        # Add system-out element if we have stdout content
        if stdout_content:
            system_out = ET.SubElement(testcase, 'system-out')
            # Sanitize the combined stdout to remove ANSI codes and invalid XML chars
            system_out.text = self._sanitize_xml_text(
                '\n\n'.join(stdout_content))

        # Add system-err element if we have stderr content
        if stderr_content:
            system_err = ET.SubElement(testcase, 'system-err')
            # Sanitize the combined stderr to remove ANSI codes and invalid XML chars
            system_err.text = self._sanitize_xml_text(
                '\n\n'.join(stderr_content))

        return ET.tostring(testcase, encoding='unicode')

    def _flush_pending_cases(self):
        """Write pending test cases to file."""
        if not self.pending_cases or not self.xml_file:
            return

        try:
            for case_xml in self.pending_cases:
                # Indent for pretty printing
                lines = case_xml.split('\n')
                for line in lines:
                    if line.strip():
                        self.xml_file.write(f'    {line}\n')

            self.xml_file.flush()
            os.fsync(self.xml_file.fileno())  # Force write to disk

            self._log_info(
                f"Flushed {len(self.pending_cases)} test cases to report")
            self.pending_cases.clear()

        except Exception as e:
            self._log_warning(f"Failed to flush test cases: {e}")

    def pytest_sessionfinish(self):
        """Finalize report with closing tags and statistics."""
        try:
            # Flush any remaining test cases
            self._flush_pending_cases()

            if self.xml_file:
                # Write closing tags
                self.xml_file.write('  </testsuite>\n')
                self.xml_file.write('</testsuites>\n')
                self.xml_file.close()
                self.xml_file = None

                # Update testsuite attributes using post-processing
                self._update_suite_attributes()

                self._log_info(f"\nPeriodicJUnitXML: Session finished")
                self._log_info(f"  Tests: {self.stats['tests']}")
                self._log_info(f"  Failures: {self.stats['failures']}")
                self._log_info(f"  Errors: {self.stats['errors']}")
                self._log_info(f"  Skipped: {self.stats['skipped']}")
                self._log_info(f"  Total time: {self.stats['time']:.3f}s")

        except Exception as e:
            self._log_warning(f"Error finalizing report: {e}")

    def _update_suite_attributes(self):
        """Update testsuite element with final statistics."""
        import xml.etree.ElementTree as ET

        try:
            # Parse the file we just wrote
            tree = ET.parse(self.xmlpath)
            root = tree.getroot()
            suite = root.find('testsuite')

            if suite is not None:
                # Update attributes
                suite.set('tests', str(self.stats['tests']))
                suite.set('failures', str(self.stats['failures']))
                suite.set('errors', str(self.stats['errors']))
                suite.set('skipped', str(self.stats['skipped']))
                suite.set('time', f"{self.stats['time']:.3f}")
                suite.set(
                    'timestamp',
                    datetime.datetime.fromtimestamp(
                        self.suite_start_time).strftime("%Y-%m-%dT%H:%M:%S"))
                suite.set('hostname', platform.node())

                # Write back
                tree.write(self.xmlpath, encoding='utf-8', xml_declaration=True)

        except Exception as e:
            self._log_warning(f"Failed to update suite attributes: {e}")

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown on interruption."""

        def signal_handler(signum, frame):
            """Handle interrupt signals by saving current progress before exit."""
            signal_name = signal.Signals(signum).name
            self._log_warning(
                f"\n\nReceived {signal_name} signal. Saving test results before exit..."
            )

            try:
                # Flush any pending test cases
                if self.pending_cases:
                    self._log_info(
                        f"Flushing {len(self.pending_cases)} pending test cases..."
                    )
                    self._flush_pending_cases()

                # Close the file properly
                if self.xml_file:
                    self.xml_file.write('  </testsuite>\n')
                    self.xml_file.write('</testsuites>\n')
                    self.xml_file.close()
                    self.xml_file = None

                    # Update statistics
                    self._update_suite_attributes()

                    self._log_info(
                        f"✅ Successfully saved {self.stats['tests']} test results to {self.xmlpath}"
                    )
                else:
                    self._log_warning("XML file already closed")

            except Exception as e:
                self._log_warning(
                    f"Failed to save results on interruption: {e}")

            # Re-raise the signal to continue normal termination
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        # Register handlers for common interrupt signals
        try:
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
            self._log_info(
                "Registered signal handlers for graceful shutdown (SIGINT, SIGTERM)"
            )
        except (AttributeError, ValueError) as e:
            # Signal handling might not be available on all platforms
            self._log_warning(
                f"Could not register signal handlers (not supported on this platform): {e}"
            )
