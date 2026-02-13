# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        if testname.startswith("accuracy/") or (
            testname.startswith("examples/") and "[" not in testname
        ):
            classname = ""

    return classname, name, file


def generate_timeout_xml(stage_name, testList, outputFilePath):
    """Generate JUnit XML report for timed-out tests.

    Args:
        stage_name: Name of the test stage
        testList: List of test names that timed out
        outputFilePath: Path where the XML report will be written
    """
    num_tests = len(testList)
    # Escape stage_name for XML safety
    stage_name_escaped = escape(stage_name, quote=True)
    xmlContent = (
        f'<?xml version="1.0" encoding="UTF-8"?><testsuites>\n'
        f'        <testsuite name="{stage_name_escaped}" errors="{num_tests}" '
        f'failures="0" skipped="0" tests="{num_tests}" time="1.00">'
    )

    for test in testList:
        classname, name, file = parse_xml_classname_name_file_from_testname(test, stage_name)
        # Escape all XML attribute values
        classname_escaped = escape(classname, quote=True)
        name_escaped = escape(name, quote=True)
        file_escaped = escape(file, quote=True)
        xmlContent += (
            f'<testcase classname="{classname_escaped}" name="{name_escaped}" '
            f'file="{file_escaped}" time="1.0">\n'
            f'        <error message="Test terminated unexpectedly">'
            f" Test terminated unexpectedly\n"
            f"        </error></testcase>"
        )
    xmlContent += "</testsuite></testsuites>"

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
    args = parser.parse_args(sys.argv[1:])
    stageName = args.stage_name
    testFilePath = args.test_file_path
    outputFilePath = args.output_file

    full_path = os.path.join(stageName, testFilePath)
    if not os.path.exists(full_path):
        print(f"No {full_path} found, skipping timeout XML generation")
        return

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            timeoutTests = [line.strip() for line in f if line.strip()]
    except IOError as e:
        print(f"Error reading {full_path}: {e}")
        return

    if len(timeoutTests) == 0:
        print(f"No timeout tests found in {full_path}, skipping timeout XML generation")
        return

    generate_timeout_xml(stageName, timeoutTests, outputFilePath)


if __name__ == "__main__":
    main()
