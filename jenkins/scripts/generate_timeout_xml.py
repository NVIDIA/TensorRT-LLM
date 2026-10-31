# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import sys
from html import escape

from pytest_timeout_evidence import load_timeout_evidence

_FALLBACK_TEST_TIME_SECONDS = 1.0


def parse_xml_classname_name_file_from_testname(testname, stage_name):
    """Parse XML attributes from a test name.

    Args:
        testname: Test identifier, may be prefixed with stage_name and can have
        different formats (e.g., "unittest/...", "file.py::class::test")
        stage_name: Name of the test stage, used for classname construction

    Returns:
        Tuple of (classname, name, file) where:
        - classname: Fully qualified class name for the test
        - name: Test method or case name
        - file: Source file containing the test
    """
    classname, name, file = "", "", ""

    # Remove stage_name prefix if present
    if testname.startswith(stage_name + "/"):
        testname = testname[len(stage_name) + 1 :]

    # Get file name
    if testname.startswith("unittest/"):
        file = "test_unittests.py"
    else:
        file = testname.split("::")[0]

    # Get test name
    if testname.startswith("unittest/"):
        name = "test_unittests_v2[" + testname + "]"
    else:
        name = testname.split("::")[-1]

    # Get class name
    if testname.startswith("unittest/"):
        classname = stage_name + ".test_unittests"
    elif len(testname.split("::")) == 3:
        classname = (
            stage_name
            + "."
            + testname.split("::")[0].replace(".py", "").replace("/", ".")
            + "."
            + testname.split("::")[1]
        )
    else:
        classname = stage_name + "." + testname.split("::")[0].replace(".py", "").replace("/", ".")
        if testname.startswith("accuracy/"):
            classname = ""

    return classname, name, file


def generate_timeout_xml(
    stage_name,
    testList,
    outputFilePath,
    test_timeouts=None,
):
    """Generate JUnit XML report for timed-out tests.

    Args:
        stage_name: Name of the test stage
        testList: List of test names that timed out
        outputFilePath: Path where the XML report will be written
        test_timeouts: Effective pytest timeout in seconds for each nodeid
    """
    test_timeouts = test_timeouts or {}
    confirmed_timeout_tests = set(test_timeouts)
    num_tests = len(testList)
    timeout_count = sum(test in confirmed_timeout_tests for test in testList)
    terminated_unexpectedly_count = num_tests - timeout_count
    total_time = sum(test_timeouts.get(test, _FALLBACK_TEST_TIME_SECONDS) for test in testList)
    # Escape stage_name for XML safety
    stage_name_escaped = escape(stage_name, quote=True)
    xmlContent = (
        f'<?xml version="1.0" encoding="UTF-8"?><testsuites>\n'
        f'        <testsuite name="{stage_name_escaped}" errors="{num_tests}" '
        f'failures="0" skipped="0" tests="{num_tests}" '
        f'unfinished_test="{num_tests}" timeout="{timeout_count}" '
        f'terminated_unexpectedly="{terminated_unexpectedly_count}" '
        f'time="{total_time:.2f}">\n'
    )

    for test in testList:
        classname, name, file = parse_xml_classname_name_file_from_testname(test, stage_name)
        duration = test_timeouts.get(test, _FALLBACK_TEST_TIME_SECONDS)
        # Escape all XML attribute values
        classname_escaped = escape(classname, quote=True)
        name_escaped = escape(name, quote=True)
        file_escaped = escape(file, quote=True)
        if test in confirmed_timeout_tests:
            error_block = '<error message="pytest_timeout">Pytest timeout.</error>'
        else:
            error_block = (
                '<error message="terminated_unexpectedly">Test terminated unexpectedly.</error>'
            )
        xmlContent += (
            f'<testcase classname="{classname_escaped}" name="{name_escaped}" '
            f'file="{file_escaped}" time="{duration:.2f}">\n'
            f"        {error_block}\n"
            f"</testcase>\n"
        )
    xmlContent += "</testsuite>\n</testsuites>\n"

    with open(outputFilePath, "w", encoding="utf-8") as f:
        f.write(xmlContent)


def main():
    """Parse arguments and generate timeout test XML report.

    Reads a list of timed-out tests from a file and generates a JUnit-compatible
    XML report marking each test with an error status.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-name", required=True, help="Stage name")
    parser.add_argument("--test-file-path", required=True, help="Test list file path")
    parser.add_argument("--output-file", required=True, help="Output file path")
    parser.add_argument(
        "--timeout-evidence-file",
        default=None,
        help="Optional file containing structured pytest-timeout records",
    )
    args = parser.parse_args(sys.argv[1:])
    stageName = args.stage_name
    testFilePath = args.test_file_path
    outputFilePath = args.output_file

    full_path = os.path.join(stageName, testFilePath)
    if not os.path.exists(full_path):
        print(f"No {full_path} found, skipping timeout XML generation")
        try:
            os.remove(outputFilePath)
        except FileNotFoundError:
            pass
        return

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            timeoutTests = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    except IOError as e:
        print(f"Error reading {full_path}: {e}")
        return

    if not timeoutTests:
        print(f"No timeout tests found in {full_path}, skipping timeout XML generation")
        try:
            os.remove(outputFilePath)
        except FileNotFoundError:
            pass
        return

    test_timeouts = load_timeout_evidence(args.timeout_evidence_file, timeoutTests)
    print(
        f"Timeout classification summary for {stageName}: {len(timeoutTests)} unfinished, "
        f"{len(test_timeouts)} pytest_timeout, "
        f"{len(timeoutTests) - len(test_timeouts)} terminated_unexpectedly"
    )
    generate_timeout_xml(
        stageName,
        timeoutTests,
        outputFilePath,
        test_timeouts,
    )


if __name__ == "__main__":
    main()
