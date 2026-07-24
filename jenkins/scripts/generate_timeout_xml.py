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

    # A unittest record is normally the wrapper case (e.g.
    # "unittest/_torch/thop/parallel"), but if it carries the specific inner test
    # that was running (".../test_x.py::test_name[...]"), surface that file+test
    # instead of flattening the timeout to the whole wrapper case.
    is_unittest_wrapper = testname.startswith("unittest/") and ".py::" not in testname

    if is_unittest_wrapper:
        file = "test_unittests.py"
        name = "test_unittests_v2[" + testname + "]"
        classname = stage_name + ".test_unittests"
    else:
        # Split the structural node id (file[::class]::test) from the
        # parametrization suffix ("[...]") before splitting on "::", so a "::"
        # inside a param id does not corrupt the file/class/name parsing.
        structural, bracket, params = testname.partition("[")
        parts = structural.split("::")
        file = parts[0]
        name = parts[-1] + bracket + params
        module = parts[0].replace(".py", "").replace("/", ".")
        if len(parts) == 3:
            classname = stage_name + "." + module + "." + parts[1]
        elif testname.startswith("accuracy/"):
            classname = ""
        else:
            classname = stage_name + "." + module

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
            timeoutTests = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    except IOError as e:
        print(f"Error reading {full_path}: {e}")
        return

    if len(timeoutTests) == 0:
        print(f"No timeout tests found in {full_path}, skipping timeout XML generation")
        return

    generate_timeout_xml(stageName, timeoutTests, outputFilePath)


if __name__ == "__main__":
    main()
