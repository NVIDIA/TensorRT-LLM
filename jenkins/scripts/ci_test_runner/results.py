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
"""Count, summarize and merge the stage's JUnit XMLs."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import test_rerun


def count_xml_results(xml_path):
    """Return (passed, failed) test case counts from a pytest JUnit XML.

    Skipped cases count as neither. A missing or unreadable XML yields (0, 0).
    """
    try:
        cases = ET.parse(xml_path).getroot().findall(".//testcase")
    except (OSError, ET.ParseError):
        return 0, 0
    failed = sum(
        1 for tc in cases if tc.find("failure") is not None or tc.find("error") is not None
    )
    skipped = sum(1 for tc in cases if tc.find("skipped") is not None)
    return len(cases) - failed - skipped, failed


def print_xml_failure_summary(xml_path, tail_lines=None):
    """Parse a pytest result XML and print failure/error messages.

    Called after a fast isolation-test failure to surface the root cause
    without requiring manual inspection of the SLURM job log.

    Args:
        xml_path: Path to the pytest JUnit XML result file.
        tail_lines: Optional list of captured pytest output lines.  Printed
            when no XML is found (collection error) so the error is visible
            in the [FAST FAILURE] banner even without a result file.
    """
    if not os.path.exists(xml_path):
        print("  [No result XML — pytest exited before generating results]")
        if tail_lines:
            print("  [Last pytest output (collection error likely above):]")
            for line in tail_lines:
                print(f"    {line}")
        return
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"  [Could not parse result XML: {exc}]")
        return

    cases = root.findall(".//testcase")
    if not cases:
        print("  [Result XML contains no test cases — 0 tests were collected]")
        return

    printed = 0
    for tc in cases:
        for tag in ("failure", "error"):
            node = tc.find(tag)
            if node is None:
                continue
            name = f"{tc.get('classname', '')}.{tc.get('name', '')}"
            msg = (node.get("message") or "").strip()
            detail = (node.text or "").strip()
            print(f"  [{tag.upper()}] {name}")
            if msg:
                print(f"    message : {msg[:600]}")
            if detail:
                excerpt = "\n    ".join(detail.splitlines()[-30:])
                print(f"    detail  :\n    {excerpt}")
            printed += 1
    if printed == 0:
        print("  [No failure/error nodes found in result XML]")


def create_empty_results_xml(output_dir, stage_name):
    """Create an empty results.xml for stages with no regular tests."""
    os.makedirs(output_dir, exist_ok=True)
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<testsuites>\n"
        f'<testsuite name="{stage_name}" errors="0" failures="0" skipped="0" tests="0" time="0.0">\n'
        "</testsuite>\n"
        "</testsuites>\n"
    )
    Path(output_dir, "results.xml").write_text(content)


def _name_testsuites_after_stage(xml_files, stage_name):
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except (OSError, ET.ParseError) as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")
            continue
        modified = False
        for ts in tree.iter("testsuite"):
            if ts.get("name") == "pytest":
                ts.set("name", stage_name)
                modified = True
        if modified:
            try:
                tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            except OSError as e:
                print(f"Warning: Failed to write {xml_file}: {e}")


def merge_results(output_dir, stage_name, outcomes):
    """Write the rerun report and merge every XML into the stage's results.xml.

    Args:
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
        outcomes: execution.RunOutcome of every regular and isolated run, in run order.
    """
    xml_files = [f for o in outcomes for f in (o.result_xml, *o.rerun_xmls) if os.path.exists(f)]
    _name_testsuites_after_stage(xml_files, stage_name)

    # The rerun report covers only runs that were rerun: their first XML, then the reruns.
    rerun_outcomes = [o for o in outcomes if o.rerun_xmls]
    report_inputs = [o.result_xml for o in rerun_outcomes if os.path.exists(o.result_xml)]
    report_inputs += [f for o in rerun_outcomes for f in o.rerun_xmls]
    if report_inputs:
        print(f"Generating rerun report with input files: {report_inputs}")
        test_rerun.generate_rerun_report(
            os.path.join(output_dir, "rerun_results.xml"), report_inputs
        )

    if xml_files:
        merged_output = os.path.join(output_dir, "results.xml")
        print(f"Merging {len(xml_files)} XML files into {merged_output}")
        test_rerun.merge_junit_xmls(merged_output, xml_files, deduplicate=True)

    # Isolated results are now in results.xml; drop them so junit does not count them twice.
    for o in outcomes:
        if "results_isolated_" in o.result_xml:
            try:
                os.remove(o.result_xml)
            except OSError:
                pass

    print("Result merging completed")
