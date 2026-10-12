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

"""CPU-only regression tests for CI attempt history and final-result merging."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts"]
from test_rerun import generate_rerun_report, merge_junit_xmls

pytestmark = pytest.mark.cpu_only


def _write_results(path: Path, outcomes: list[str]) -> None:
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="CPU",
        tests=str(len(outcomes)),
        failures=str(outcomes.count("failure")),
        errors=str(outcomes.count("error")),
        skipped="0",
    )
    for attempt, outcome in enumerate(outcomes):
        case = ET.SubElement(
            suite,
            "testcase",
            classname="test_example",
            name="test_retry[param]",
            time=str(attempt + 1),
        )
        ET.SubElement(case, "system-out").text = f"attempt {attempt}"
        ET.SubElement(case, "system-err")
        if outcome != "passed":
            ET.SubElement(case, outcome, message=f"attempt {attempt} failed")
    ET.ElementTree(root).write(path, encoding="unicode")


@pytest.mark.parametrize("final_outcome", ["passed", "failure", "error"])
@pytest.mark.parametrize("layout", ["single_file", "retry_file", "across_files"])
def test_final_report_keeps_last_attempt(tmp_path: Path, final_outcome: str, layout: str) -> None:
    initial = tmp_path / "results.xml"
    retry = tmp_path / "retry.xml"
    if layout == "single_file":
        _write_results(initial, ["failure", "failure", final_outcome])
        inputs = [str(initial)]
    else:
        _write_results(initial, ["failure", "failure"])
        outcomes = ["failure", final_outcome] if layout == "retry_file" else [final_outcome]
        _write_results(retry, outcomes)
        inputs = [str(initial), str(retry)]
    expected = ET.parse(inputs[-1]).findall(".//testcase")[-1]

    merge_junit_xmls(str(initial), inputs, deduplicate=True)

    suite = ET.parse(initial).getroot().find("testsuite")
    assert suite is not None
    cases = suite.findall("testcase")
    assert len(cases) == 1
    assert ET.tostring(cases[0]) == ET.tostring(expected)
    assert suite.get("tests") == "1"
    assert suite.get("failures") == str(int(final_outcome == "failure"))
    assert suite.get("errors") == str(int(final_outcome == "error"))
    assert suite.get("skipped") == "0"


@pytest.mark.parametrize("final_outcome", ["passed", "failure"])
def test_rerun_history_preserves_failed_attempts(tmp_path: Path, final_outcome: str) -> None:
    initial = tmp_path / "results.xml"
    retry = tmp_path / "retry.xml"
    history = tmp_path / "rerun_results.xml"
    _write_results(initial, ["failure"])
    _write_results(retry, ["failure", final_outcome])

    generate_rerun_report(str(history), [str(initial), str(retry)])

    suite = ET.parse(history).getroot().find("testsuite")
    assert suite is not None
    cases = suite.findall("testcase")
    assert len(cases) == 3
    assert all(case.find("failure") is not None for case in cases[:2])
    assert (cases[-1].find("failure") is not None) == (final_outcome == "failure")
    assert cases[0].get("isrerun") is None
    assert all(case.get("isrerun") == "true" for case in cases[1:])
    assert suite.get("tests") == "3"
    assert suite.get("failures") == ("3" if final_outcome == "failure" else "2")
    assert history.with_suffix(".html").is_file()
