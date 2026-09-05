# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import os
import re
import sys
from html import escape

# ---------------------------------------------------------------------------
# XML sanitisation helpers
# ---------------------------------------------------------------------------

# ANSI escape sequences (e.g. colour codes from pytest output).
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Characters that are illegal in XML 1.0 (excluding the accepted whitespace
# codepoints \t, \n, \r).
_XML_ILLEGAL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f"
    r"\ud800-\udfff￾￿]"
)

_RANK_TIMEOUT_DATA_FILE_RE = re.compile(r"timeout_data_step(\d+)_rank(\d+)\.jsonl$")
_FALLBACK_TEST_TIME_SECONDS = 1.0


def sanitize_for_xml(text):
    """Remove ANSI escapes and XML-illegal characters, then XML-escape the result."""
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _XML_ILLEGAL_RE.sub("", text)
    return escape(text, quote=False)


def format_timeout_system_out(nodeid, snippet):
    """Return a human-readable timeout section for a JUnit ``system-out`` field."""
    return (
        "\n"
        "==============================================================================\n"
        "PYTEST TIMEOUT\n"
        f"Test: {nodeid}\n"
        "------------------------------------------------------------------------------\n"
        "pytest-timeout thread stack dump:\n"
        "------------------------------------------------------------------------------\n"
        f"{snippet.rstrip()}\n"
        "==============================================================================\n"
        "END PYTEST TIMEOUT\n"
        "==============================================================================\n"
    )


def timeout_data_sort_key(path):
    """Return a deterministic numeric sort key for timeout-data files."""
    path = os.fspath(path)
    match = _RANK_TIMEOUT_DATA_FILE_RE.search(os.path.basename(path))
    if match:
        return (0, int(match.group(1)), int(match.group(2)), path)
    return (1, 0, 0, path)


def load_timeout_data(paths, expected_nodeids=None):
    """Load timeout snippets and configured durations from NDJSON files.

    Args:
        paths: Paths to rank-specific or per-invocation timeout-data files.
        expected_nodeids: Optional nodeids that remain unfinished. Records for
            other tests are ignored before their snippets are retained.

    Returns:
        A tuple of ``({nodeid: snippet}, {nodeid: timeout})``. For rank-specific
        files, the first timeout record by numeric step and rank order wins.
        For a non-rank local file, later records overwrite earlier ones.
    """
    if not paths:
        return {}, {}
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    timeout_map = {}
    test_timeouts = {}
    for path in sorted(map(os.fspath, paths), key=timeout_data_sort_key):
        is_rank_file = _RANK_TIMEOUT_DATA_FILE_RE.search(os.path.basename(path)) is not None
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                        nodeid = rec.get("nodeid") if isinstance(rec, dict) else None
                        if not isinstance(nodeid, str):
                            raise ValueError("expected nodeid string")
                        if expected_nodeids is not None and nodeid not in expected_nodeids:
                            continue
                        record_type = rec.get("type")
                        if record_type == "timeout_config":
                            timeout = rec.get("timeout")
                            if timeout is not None and not isinstance(timeout, (int, float)):
                                raise ValueError("expected timeout to be numeric or null")
                            if timeout is not None:
                                test_timeouts.setdefault(nodeid, float(timeout))
                        elif record_type == "timeout" or "snippet" in rec:
                            snippet = rec.get("snippet")
                            if not isinstance(snippet, str):
                                raise ValueError("expected snippet string")
                            if is_rank_file:
                                timeout_map.setdefault(nodeid, snippet)
                            else:
                                timeout_map[nodeid] = snippet
                    except (json.JSONDecodeError, ValueError) as exc:
                        print(
                            f"WARNING: generate_timeout_xml: skipping corrupt line "
                            f"{lineno} in {path}: {exc}",
                            file=sys.stderr,
                        )
        except OSError as exc:
            print(
                f"WARNING: generate_timeout_xml: cannot read {path}: {exc}",
                file=sys.stderr,
            )
    return timeout_map, test_timeouts


def test_duration(test, timeout_map, test_timeouts):
    """Return the best available JUnit duration for an interrupted test."""
    if test in timeout_map:
        timeout = test_timeouts.get(test)
        if timeout is not None and timeout > 0:
            return timeout
    return _FALLBACK_TEST_TIME_SECONDS


# ---------------------------------------------------------------------------


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
    timeout_map=None,
    test_timeouts=None,
):
    """Generate JUnit XML report for timed-out tests.

    Args:
        stage_name: Name of the test stage.
        testList: List of test nodeids that did not complete (raw lines from
            ``unfinished_test.txt``, including the stage prefix).
        outputFilePath: Path where the XML report will be written.
        timeout_map: Optional mapping of ``{nodeid: snippet}`` produced by
            ``load_timeout_data()``.  A nodeid present in this map is classified
            as ``pytest_timeout`` and its snippet is embedded in
            ``<system-out>``. All other nodeids are classified as
            ``terminated_unexpectedly``. When *None* or empty every test falls
            back to ``terminated_unexpectedly``.
        test_timeouts: Optional mapping containing the effective pytest timeout
            recorded for each nodeid.
    """
    if timeout_map is None:
        timeout_map = {}
    if test_timeouts is None:
        test_timeouts = {}
    num_tests = len(testList)
    durations = {test: test_duration(test, timeout_map, test_timeouts) for test in testList}
    timeout_count = sum(test in timeout_map for test in testList)
    terminated_unexpectedly_count = num_tests - timeout_count
    # Escape stage_name for XML safety
    stage_name_escaped = escape(stage_name, quote=True)
    xmlContent = (
        f'<?xml version="1.0" encoding="UTF-8"?><testsuites>\n'
        f'        <testsuite name="{stage_name_escaped}" errors="{num_tests}" '
        f'tests="{num_tests}" '
        f'unfinished_test="{num_tests}" timeout="{timeout_count}" '
        f'terminated_unexpectedly="{terminated_unexpectedly_count}" '
        f'time="{sum(durations.values()):.2f}">\n'
    )

    for test in testList:
        classname, name, file = parse_xml_classname_name_file_from_testname(test, stage_name)
        # Escape all XML attribute values
        classname_escaped = escape(classname, quote=True)
        name_escaped = escape(name, quote=True)
        file_escaped = escape(file, quote=True)

        if test in timeout_map:
            # Confirmed pytest-timeout kill: embed the captured log snippet.
            system_out = format_timeout_system_out(test, timeout_map[test])
            snippet_escaped = sanitize_for_xml(system_out)
            error_block = (
                f'        <error message="pytest_timeout">Pytest timeout.</error>\n'
                f"        <system-out>{snippet_escaped}</system-out>\n"
            )
        else:
            # Unexpected termination without a confirmed pytest-timeout banner.
            error_block = (
                '        <error message="terminated_unexpectedly">'
                "Test terminated unexpectedly.</error>"
            )

        xmlContent += (
            f'<testcase classname="{classname_escaped}" name="{name_escaped}" '
            f'file="{file_escaped}" time="{durations[test]:.2f}">\n'
            f"{error_block}</testcase>\n"
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
    parser.add_argument(
        "--test-file-path",
        action="append",
        required=True,
        help="Test list file path; may be passed for the original run and reruns",
    )
    parser.add_argument("--output-file", required=True, help="Output file path")
    parser.add_argument(
        "--timeout-data-file",
        action="append",
        default=[],
        help="Optional timeout-data file; may be passed once per Slurm rank",
    )
    args = parser.parse_args(sys.argv[1:])
    stageName = args.stage_name
    outputFilePath = args.output_file

    timeoutTests = []
    for testFilePath in args.test_file_path:
        full_path = (
            testFilePath if os.path.isabs(testFilePath) else os.path.join(stageName, testFilePath)
        )
        if not os.path.exists(full_path):
            print(f"No {full_path} found, skipping it while generating timeout XML")
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                timeoutTests.extend(line.strip() for line in f if line.strip())
        except OSError as exc:
            print(f"Error reading {full_path}: {exc}")

    timeoutTests = list(dict.fromkeys(timeoutTests))

    if len(timeoutTests) == 0:
        print(f"No timeout tests found for {stageName}, skipping timeout XML generation")
        return

    timeout_map, test_timeouts = load_timeout_data(args.timeout_data_file, set(timeoutTests))
    classified_count = sum(test in timeout_map for test in timeoutTests)
    print(
        f"Timeout classification summary for {stageName}: {len(timeoutTests)} unfinished, "
        f"{classified_count} pytest_timeout, "
        f"{len(timeoutTests) - classified_count} terminated_unexpectedly"
    )
    generate_timeout_xml(
        stageName,
        timeoutTests,
        outputFilePath,
        timeout_map,
        test_timeouts,
    )


if __name__ == "__main__":
    main()
