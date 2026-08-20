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


def load_timeout_map(paths, expected_nodeids=None):
    """Load NDJSON files and return a ``{nodeid: snippet}`` mapping.

    Args:
        paths: Paths to rank-specific or per-invocation timeout-data files.
        expected_nodeids: Optional nodeids that remain unfinished. Records for
            other tests are ignored before their snippets are retained.

    Returns:
        Timeout records keyed by nodeid. For rank-specific files, the first
        record by numeric step and rank order wins deterministically. For a
        non-rank local file, later records overwrite earlier ones so the most
        recent rerun supplies the displayed snippet. Unreadable files and
        corrupt lines are reported and skipped.
    """
    if not paths:
        return {}
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    timeout_map = {}
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
                        if isinstance(rec, dict) and rec.get("type") in {"start", "end"}:
                            continue
                        if (
                            not isinstance(rec, dict)
                            or not isinstance(rec.get("nodeid"), str)
                            or not isinstance(rec.get("snippet"), str)
                        ):
                            raise ValueError(
                                f'expected {{"nodeid": str, "snippet": str}}, '
                                f"got {type(rec).__name__} with keys "
                                f"{list(rec.keys()) if isinstance(rec, dict) else 'N/A'}"
                            )
                        nodeid = rec["nodeid"]
                        if expected_nodeids is not None and nodeid not in expected_nodeids:
                            continue
                        if is_rank_file:
                            timeout_map.setdefault(nodeid, rec["snippet"])
                        else:
                            timeout_map[nodeid] = rec["snippet"]
                    except (json.JSONDecodeError, KeyError, ValueError) as exc:
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
    return timeout_map


def load_unfinished_test_data(paths, expected_nodeids=None):
    """Load ``{nodeid: {start_time, end_time, timeout}}`` records.

    The first start/end pair wins, preserving the invocation in which a test
    actually terminated when a later invocation is run in the same stage.
    Invalid records are skipped rather than making timeout-report generation
    fail.
    """
    if not paths:
        return {}
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    test_data = {}
    for path in sorted(map(os.fspath, paths)):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                        if isinstance(record, dict) and (
                            record.get("type") == "timeout" or "snippet" in record
                        ):
                            continue
                        nodeid = record.get("nodeid") if isinstance(record, dict) else None
                        start_time = record.get("start_time") if isinstance(record, dict) else None
                        end_time = record.get("end_time") if isinstance(record, dict) else None
                        timeout = record.get("timeout") if isinstance(record, dict) else None
                        if not isinstance(nodeid, str):
                            raise ValueError("expected nodeid string")
                        if start_time is not None and not isinstance(start_time, (int, float)):
                            raise ValueError("expected start_time to be numeric or null")
                        if end_time is not None and not isinstance(end_time, (int, float)):
                            raise ValueError("expected end_time to be numeric or null")
                        if timeout is not None and not isinstance(timeout, (int, float)):
                            raise ValueError("expected timeout to be numeric or null")
                        if start_time is None and end_time is None:
                            raise ValueError("expected numeric start_time or end_time")
                        if expected_nodeids is None or nodeid in expected_nodeids:
                            data = test_data.setdefault(nodeid, {})
                            if start_time is not None:
                                data.setdefault("start_time", float(start_time))
                            if end_time is not None:
                                data.setdefault("end_time", float(end_time))
                            if timeout is not None:
                                data.setdefault("timeout", float(timeout))
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
    return test_data


def test_duration(test, timeout_map, unfinished_test_data):
    """Return the best available JUnit duration for an interrupted test."""
    data = unfinished_test_data.get(test, {})
    if test in timeout_map:
        timeout = data.get("timeout")
        if timeout is not None and timeout > 0:
            return timeout
        return _FALLBACK_TEST_TIME_SECONDS

    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if start_time is not None and end_time is not None:
        return max(0.0, end_time - start_time)
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
    unfinished_test_data=None,
    now=None,
):
    """Generate JUnit XML report for timed-out tests.

    Args:
        stage_name: Name of the test stage.
        testList: List of test nodeids that did not complete (raw lines from
            ``unfinished_test.txt``, including the stage prefix).
        outputFilePath: Path where the XML report will be written.
        timeout_map: Optional mapping of ``{nodeid: snippet}`` produced by
            ``load_timeout_map()``.  A nodeid present in this map is classified
            as ``pytest_timeout`` and its snippet is embedded in
            ``<system-out>``.  All other nodeids are classified as ``unknown``.
            When *None* or empty every test falls back to ``unknown``.
        unfinished_test_data: Optional mapping containing the test start time,
            end time, and effective pytest timeout recorded during execution.
        now: Retained for backward compatibility with callers that inject a
            deterministic clock; timings come from the recorded metadata.
    """
    if timeout_map is None:
        timeout_map = {}
    if unfinished_test_data is None:
        unfinished_test_data = {}
    num_tests = len(testList)
    durations = {test: test_duration(test, timeout_map, unfinished_test_data) for test in testList}
    timeout_count = sum(test in timeout_map for test in testList)
    unknown_count = num_tests - timeout_count
    # Escape stage_name for XML safety
    stage_name_escaped = escape(stage_name, quote=True)
    xmlContent = (
        f'<?xml version="1.0" encoding="UTF-8"?><testsuites>\n'
        f'        <testsuite name="{stage_name_escaped}" errors="{num_tests}" '
        f'tests="{num_tests}" '
        f'unfinished_test="{num_tests}" timeout="{timeout_count}" unknown="{unknown_count}" '
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
            # Unexpected termination (OOM, node crash, etc.) or unknown cause.
            error_block = '        <error message="unknown">Test terminated unexpectedly.</error>'

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

    timeout_map = load_timeout_map(args.timeout_data_file, set(timeoutTests))
    unfinished_test_data = load_unfinished_test_data(args.timeout_data_file, set(timeoutTests))
    classified_count = sum(test in timeout_map for test in timeoutTests)
    print(
        f"Timeout classification summary for {stageName}: {len(timeoutTests)} unfinished, "
        f"{classified_count} pytest_timeout, {len(timeoutTests) - classified_count} unknown"
    )
    generate_timeout_xml(
        stageName,
        timeoutTests,
        outputFilePath,
        timeout_map,
        unfinished_test_data,
    )


if __name__ == "__main__":
    main()
