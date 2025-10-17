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

This module provides a lightweight periodic JUnit XML reporter that leverages
pytest's built-in junitxml plugin for simplified test result handling.
"""

import os
import platform
import signal
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

try:
    from _pytest.config import Config
    from _pytest.junitxml import LogXML
    from _pytest.reports import TestReport
except ImportError:
    # Fallback for different pytest versions
    Config = None  # type: ignore
    TestReport = None  # type: ignore
    LogXML = None  # type: ignore


class PeriodicJUnitXML:
    """
    Periodic JUnit XML reporter using lightweight collection and batch processing.

    This reporter uses a two-phase approach for maximum performance:
    1. Collection phase (during test execution): Quickly collect TestReport objects
    2. Processing phase (when generating reports): Batch process through pytest's LogXML

    This approach provides:
    - Fast test execution (minimal overhead during test runs)
    - Standard JUnit XML output (using pytest's LogXML for compatibility)
    - Periodic saves to prevent data loss
    - Graceful shutdown on interruption (SIGINT/SIGTERM)

    Usage:
        Add to conftest.py:

        def pytest_configure(config):
            from utils.periodic_junit import PeriodicJUnitXML

            reporter = PeriodicJUnitXML(
                xmlpath='results/junit_report.xml',
                interval=18000,    # Save every 5 hours
                batch_size=10,     # Or save every 10 tests
            )
            reporter.pytest_configure(config)
            config.pluginmanager.register(reporter, 'periodic_junit')
    """

    def __init__(
            self,
            xmlpath: str,
            interval: int = 18000,  # Default 5 hours
            batch_size: int = 10,
            logger=None,  # Optional logger (info, warning functions)
    ):
        """
        Initialize periodic reporter.

        Uses lightweight collection mode: test reports are collected quickly during execution
        and processed in batch only when generating reports (much faster).

        Args:
            xmlpath: Path to the output XML file
            interval: Time interval in seconds between saves (default: 18000 = 5 hours)
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

        # Store raw reports for batch processing
        self.pending_reports = [
        ]  # Collected reports, processed only when generating XML

        # LogXML will be created only when generating reports
        self.logxml: Optional[LogXML] = None
        self.config = None

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

    def pytest_configure(self, config: Config):
        """Configure and initialize the reporter."""
        # Store config for later use
        self.config = config

        # Ensure required attributes exist on config
        if not hasattr(config, 'option'):
            config.option = type('Namespace', (), {})()
        if not hasattr(config.option, 'xmlpath'):
            config.option.xmlpath = self.xmlpath
        if not hasattr(config.option, 'junit_logging'):
            config.option.junit_logging = 'out-err'

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        self._log_info(f"PeriodicJUnitXML: Initialized at {self.xmlpath} "
                       "(lightweight mode - fast collection, batch processing)")

    def _init_logxml(self):
        """Initialize or re-initialize the LogXML plugin."""
        if self.config is None:
            return

        # Initialize the LogXML plugin (pytest's built-in junitxml reporter)
        self.logxml = LogXML(
            self.xmlpath,
            None,  # prefix
            'pytest',  # suite_name
            'out-err',  # logging - capture stdout and stderr only
        )
        self.logxml.config = self.config
        self.logxml.stats = dict.fromkeys(
            ['error', 'passed', 'failure', 'skipped'], 0)
        self.logxml.node_reporters = {}  # type: ignore
        self.logxml.node_reporters_ordered = []  # type: ignore
        self.logxml.global_properties = []

    def pytest_runtest_logreport(self, report: TestReport):
        """Handle test reports and trigger periodic saving."""
        # Collect the report for later batch processing (fast)
        self.pending_reports.append(report)

        # Only increment counter and check for save on teardown phase
        if report.when == "teardown":
            self.completed_tests += 1
            current_time = time.time()

            # Flush if batch threshold reached OR time interval elapsed
            should_flush_by_time = (current_time -
                                    self.last_save_time) >= self.time_interval
            should_flush_by_batch = self.completed_tests >= self.batch_size

            if should_flush_by_batch or should_flush_by_time:
                if should_flush_by_batch:
                    self._log_info(
                        f"Completed {self.completed_tests} cases in the last "
                        f"{current_time - self.last_save_time:.0f} seconds")
                # Reset counters before generating
                self.completed_tests = 0
                self.last_save_time = current_time
                try:
                    self._generate_report()
                except Exception as e:
                    self._log_warning(f"Error generating periodic report: {e}")

    def pytest_sessionfinish(self):
        """Generate final report at session end."""
        try:
            self._generate_report(is_final=True)
        except Exception as e:
            self._log_warning(f"Error generating final report: {e}")

    def _process_pending_reports(self):
        """Process all pending reports through LogXML (lightweight mode only)."""
        if not self.pending_reports:
            return

        report_count = len(self.pending_reports)
        self._log_info(
            f"Processing {report_count} pending reports through LogXML...")

        # Initialize LogXML if not already done
        if self.logxml is None:
            self._init_logxml()

        # Process all collected reports
        start_time = time.time()
        for report in self.pending_reports:
            if self.logxml:
                self.logxml.pytest_runtest_logreport(report)

        # Clear processed reports
        self.pending_reports.clear()

        elapsed = (time.time() - start_time) * 1000
        self._log_info(
            f"Processed {report_count} reports in {elapsed:.1f}ms ({elapsed/report_count:.2f}ms per report)"
        )

    def _generate_report(self, is_final=False):
        """
        Generate XML report with executed tests.

        Always writes to the same output file (self.xmlpath) for simplicity.

        Args:
            is_final: Kept for compatibility but not used (always writes to same file)
        """
        # Process pending reports (batch processing for performance)
        if self.pending_reports:
            self._process_pending_reports()

        if self.logxml is None or not self.logxml.node_reporters_ordered:
            return

        temp_file = None
        try:
            # Always use the same output file
            output_file = self.xmlpath

            # Create directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Write to temporary file first for atomic operation
            temp_file = output_file + '.tmp'
            with open(temp_file, "w", encoding="utf-8") as logfile:
                suite_stop_time = time.time()
                suite_time_delta = suite_stop_time - self.suite_start_time

                stats = self.logxml.stats
                numtests = sum(stats.values()) - getattr(
                    self.logxml, 'cnt_double_fail_tests', 0)

                # Build XML structure
                logfile.write('<?xml version="1.0" encoding="utf-8"?>')
                suite_node = ET.Element(
                    "testsuite",
                    name="pytest",
                    errors=str(stats["error"]),
                    failures=str(stats["failure"]),
                    skipped=str(stats["skipped"]),
                    tests=str(numtests),
                    time=f"{suite_time_delta:.3f}",
                    timestamp=datetime.fromtimestamp(
                        self.suite_start_time).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    hostname=platform.node(),
                )

                # Add all test cases using pytest's node reporters
                for node_reporter in self.logxml.node_reporters_ordered:
                    suite_node.append(node_reporter.to_xml())

                testsuites = ET.Element("testsuites")
                testsuites.append(suite_node)
                logfile.write(ET.tostring(testsuites, encoding="unicode"))

            # Atomic rename to final location
            os.replace(temp_file, output_file)

            self._log_info(
                f"{'Final report' if is_final else 'Periodic report'} generated with {numtests} tests: {output_file}"
            )

        except Exception as e:
            self._log_warning(f"Error in report generation: {e}")
            # Clean up temporary file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            raise

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown on interruption."""

        def signal_handler(signum, frame):
            """Handle interrupt signals by saving current progress before exit."""
            signal_name = signal.Signals(signum).name
            self._log_warning(
                f"\n\nReceived {signal_name} signal. Saving test results before exit..."
            )

            try:
                # Process any pending reports first
                if self.pending_reports:
                    self._log_info(
                        f"Processing {len(self.pending_reports)} pending test reports before exit..."
                    )
                    self._process_pending_reports()

                # Save current progress with all completed tests
                if self.logxml and self.logxml.node_reporters_ordered:
                    self._log_info(
                        f"Saving {len(self.logxml.node_reporters_ordered)} test results..."
                    )
                    self._generate_report(is_final=True)
                    self._log_info(
                        f"✅ Successfully saved test results to {self.xmlpath}")
                else:
                    self._log_warning("No test results to save")

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
