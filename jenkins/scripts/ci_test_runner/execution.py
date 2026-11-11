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
"""Run pytest for regular and isolated tests, and rerun the eligible failures."""

import collections
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import test_rerun

from .commands import build_isolated_command, build_rerun_command
from .console import format_duration, print_banner, print_phase_summary
from .results import count_xml_results, print_xml_failure_summary

# Isolation tests that finish in under this many seconds almost certainly
# failed before the test body ran (server not ready, import error, etc.).
_FAST_FAILURE_THRESHOLD_SECS = 120

# Output kept from the end of a failed pytest session, and the size cap of
# failure_evidence.log, which Jenkins matches against the infra-failure signatures.
_OUTPUT_TAIL_BYTES = 2 * 1024 * 1024
FAILURE_EVIDENCE_LOG = "failure_evidence.log"


@dataclass
class RunOutcome:
    """The XMLs and verdict of one pytest run and its reruns."""

    # XML of the first run; it may be missing if pytest died before writing it.
    result_xml: str
    # XMLs written by the reruns, in rerun order.
    rerun_xmls: list = field(default_factory=list)
    # Some test still failed (or crashed) when rerun.
    failed_after_rerun: bool = False
    # Some test failed but the stage's rerun policy did not rerun it (rerun_0.txt),
    # e.g. it did not finish, or ran too long with no matching failure signature.
    not_rerun: bool = False
    # Output tails of the sessions behind the verdict: the reruns that failed
    # (failed_after_rerun) and the first run (not_rerun).
    failed_rerun_logs: list = field(default_factory=list)
    first_run_log: str = None


def _count_lines(path):
    return len([line for line in Path(path).read_text().splitlines() if line.strip()])


def run_pytest(pytest_cmd, working_dir, tail_lines=0, log_path=None):
    """Execute a pytest command and return the exit code and duration.

    The output is streamed through to stdout, and its last ``_OUTPUT_TAIL_BYTES``
    are kept in memory.

    Args:
        pytest_cmd: The full pytest command string.
        working_dir: Working directory for pytest execution.
        tail_lines: When > 0, return the last N output lines as the third tuple
            element.  Use for isolation tests so collection errors that
            disappear between the session header and the summary line are
            captured and can be re-printed in the [FAST FAILURE] banner.
        log_path: When set and pytest fails, the kept output is written there.

    Returns:
        Tuple of (exit_code, duration_seconds, captured_tail: list[str]).
        captured_tail is an empty list when tail_lines == 0.
    """
    print(f"Running pytest: {pytest_cmd}")
    sys.stdout.flush()
    t0 = time.monotonic()

    buf = collections.deque()
    buf_bytes = 0
    # PYTHONUNBUFFERED=1 prevents pytest from block-buffering stdout when
    # it detects a pipe (non-tty), which would otherwise delay or lose
    # collection-error output before the process exits.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        pytest_cmd,
        shell=True,
        cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.append(line)
        buf_bytes += len(line)
        while buf_bytes > _OUTPUT_TAIL_BYTES and len(buf) > 1:
            buf_bytes -= len(buf.popleft())
    proc.wait()
    rc = proc.returncode
    elapsed = time.monotonic() - t0
    print(f"Pytest finished with exit code {rc} ({format_duration(elapsed)})")

    if log_path and rc != 0:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        Path(log_path).write_text("".join(buf))
    tail = [line.rstrip("\n") for line in list(buf)[-tail_lines:]] if tail_lines > 0 else []
    return rc, elapsed, tail


def write_failure_evidence(output_dir, log_paths):
    """Write the tail of the given session logs to failure_evidence.log.

    Jenkins matches the infra-failure signatures against this file instead of the
    run_tests.py step log, which also carries the sessions that passed and the
    signature list itself.
    """
    text = "".join(Path(p).read_text() for p in log_paths if p and os.path.exists(p))
    Path(output_dir, FAILURE_EVIDENCE_LOG).write_text(text[-_OUTPUT_TAIL_BYTES:])


def check_and_rerun(
    result_xml,
    base_cmd,
    working_dir,
    output_dir,
    rerun_tag,
    fail_signatures,
    test_list_file,
    first_run_log=None,
    rerun_policy=None,
):
    """Classify the failures of a run and rerun the eligible ones.

    Args:
        result_xml: Path to the results XML to analyze.
        base_cmd: The pytest command of the run being checked.
        working_dir: Working directory for pytest execution.
        output_dir: Base output directory for the stage.
        rerun_tag: Tag for this rerun (e.g., "regular", "isolated_0").
        fail_signatures: List of failure signature strings.
        test_list_file: Path to the test list file that drove this pytest run
            (used to detect tests that never ran).
        first_run_log: Output tail of the run being checked.
        rerun_policy: The stage's rerun policy from jenkins/config/rerun_policy.yml;
            None applies the policy's default.

    Returns:
        RunOutcome for the run that wrote ``result_xml``.
    """
    outcome = RunOutcome(result_xml, first_run_log=first_run_log)
    if not os.path.exists(result_xml):
        # No results were ever flushed (e.g. killed before any test finished);
        # test_rerun falls back to unfinished_test.txt / the test list.
        print(f"No {result_xml} file found; falling back to unfinished/not-run classification")

    rerun_dir = os.path.join(output_dir, "rerun", rerun_tag)
    os.makedirs(rerun_dir, exist_ok=True)

    # test_rerun splits the failures into rerun_0.txt (not eligible), rerun_1.txt
    # (rerun once) and rerun_2.txt (rerun twice), writing only non-empty lists.
    print(f"Generating rerun test lists for {rerun_tag}...")
    test_rerun.generate_rerun_tests_list(
        rerun_dir,
        result_xml,
        fail_signatures,
        test_list_file,
        os.path.join(output_dir, "unfinished_test.txt"),
        policy=rerun_policy,
    )
    for times in [1, 2]:
        rerun_file = os.path.join(rerun_dir, f"rerun_{times}.txt")
        if os.path.exists(rerun_file):
            print(f"Found {_count_lines(rerun_file)} {rerun_tag} tests to rerun {times} time(s)")
    if os.path.exists(os.path.join(rerun_dir, "rerun_0.txt")):
        outcome.not_rerun = True
        print(
            f"Some {rerun_tag} tests failed and were not eligible for rerun under this "
            "stage's rerun policy (the rerun list log above says why)."
        )

    for times in [1, 2]:
        rerun_list = os.path.join(rerun_dir, f"rerun_{times}.txt")
        if not os.path.exists(rerun_list):
            print(f"No failed {rerun_tag} tests need to be rerun {times} time(s)")
            continue

        print(f"Rerun test list ({times}):")
        print(Path(rerun_list).read_text())

        xml_file = os.path.join(rerun_dir, f"rerun_results_{times}.xml")
        csv_file = os.path.join(rerun_dir, f"rerun_report_{times}.csv")
        rerun_cmd = build_rerun_command(base_cmd, rerun_list, xml_file, csv_file, times - 1)
        rerun_log = os.path.join(rerun_dir, f"rerun_output_{times}.log")
        rc, _, _ = run_pytest(rerun_cmd, working_dir, log_path=rerun_log)

        if os.path.exists(xml_file):
            outcome.rerun_xmls.append(xml_file)
        if rc != 0:
            outcome.failed_after_rerun = True
            outcome.failed_rerun_logs.append(rerun_log)
            if not os.path.exists(xml_file):
                # Stop rerunning, but keep the XMLs of the earlier attempts so
                # they are merged into the stage results.
                print(f"The {rerun_tag} tests crashed during rerun attempt (no XML produced).")
                break
            print(f"The {rerun_tag} tests still failed after rerun attempt.")

    return outcome


def run_regular_tests(
    pytest_cmd,
    regular_list,
    working_dir,
    output_dir,
    stage_name,
    fail_signatures,
    rerun_policy=None,
):
    """Run the regular tests in one pytest session and rerun eligible failures.

    Args:
        pytest_cmd: The full pytest command string (already includes --test-list).
        regular_list: Path to the regular test list file.
        working_dir: Working directory for pytest execution.
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
        fail_signatures: List of failure signature strings.
        rerun_policy: The stage's rerun policy; None applies the policy's default.

    Returns:
        RunOutcome of the regular run.
    """
    print_banner(f"REGULAR TESTS — {_count_lines(regular_list)} test(s) for stage [{stage_name}]")

    result_xml = os.path.join(output_dir, "results.xml")
    first_run_log = os.path.join(output_dir, "rerun", "regular", "first_run_output.log")
    rc, elapsed, _ = run_pytest(pytest_cmd, working_dir, log_path=first_run_log)
    passed, failed = count_xml_results(result_xml)
    print_phase_summary("REGULAR TESTS", passed, failed, elapsed, phase_failed=rc != 0)
    if rc == 0:
        return RunOutcome(result_xml)

    print_banner(f"RERUN — checking failed regular tests for stage [{stage_name}]", char="-")
    return check_and_rerun(
        result_xml,
        pytest_cmd,
        working_dir,
        output_dir,
        "regular",
        fail_signatures,
        regular_list,
        first_run_log,
        rerun_policy=rerun_policy,
    )


def run_isolated_tests(
    pytest_cmd,
    isolate_list,
    working_dir,
    output_dir,
    stage_name,
    fail_signatures,
    rerun_policy=None,
):
    """Run each isolated test in its own pytest session and rerun it on failure.

    Args:
        pytest_cmd: The base pytest command string.
        isolate_list: Path to the isolate test list file.
        working_dir: Working directory for pytest execution.
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
        fail_signatures: List of failure signature strings.
        rerun_policy: The stage's rerun policy; None applies the policy's default.
            Each isolated test is checked against it on its own.

    Returns:
        List of RunOutcome, one per isolated test.
    """
    isolate_tests = [
        line.strip() for line in Path(isolate_list).read_text().splitlines() if line.strip()
    ]
    print_banner(f"ISOLATION TESTS — {len(isolate_tests)} test(s) for stage [{stage_name}]")

    outcomes = []
    iso_passed = 0
    iso_failed = 0
    t0 = time.monotonic()

    for i, test_name in enumerate(isolate_tests):
        print(f"\n--- [ISOLATION {i + 1}/{len(isolate_tests)}] {test_name} ---")

        single_test_file = os.path.join(output_dir, f"isolated_{i}.txt")
        Path(single_test_file).write_text(test_name + "\n")
        xml_path = os.path.join(output_dir, f"results_isolated_{i}.xml")
        csv_path = os.path.join(output_dir, f"report_isolated_{i}.csv")

        isolated_cmd = build_isolated_command(
            pytest_cmd, single_test_file, xml_path, csv_path, stage_name
        )
        first_run_log = os.path.join(output_dir, "rerun", f"isolated_{i}", "first_run_output.log")
        rc, elapsed, pytest_tail = run_pytest(
            isolated_cmd, working_dir, tail_lines=200, log_path=first_run_log
        )

        if rc == 0:
            outcome = RunOutcome(xml_path)
        else:
            if elapsed < _FAST_FAILURE_THRESHOLD_SECS:
                print(
                    f"\n  [FAST FAILURE] Isolation test {i + 1} exited in "
                    f"{format_duration(elapsed)} (< {_FAST_FAILURE_THRESHOLD_SECS}s). "
                    f"Likely failed before the test body ran (server not ready, "
                    f"collection error, etc.)."
                )
                print_xml_failure_summary(xml_path, tail_lines=pytest_tail)
            print_banner(
                f"RERUN — isolated test {i + 1}/{len(isolate_tests)}: {test_name}",
                char="-",
            )
            outcome = check_and_rerun(
                xml_path,
                isolated_cmd,
                working_dir,
                output_dir,
                f"isolated_{i}",
                fail_signatures,
                single_test_file,
                first_run_log,
                rerun_policy=rerun_policy,
            )
            if outcome.failed_after_rerun:
                print(f"Isolated test {i + 1} ({test_name}) failed after rerun attempt")
            elif outcome.not_rerun:
                print(
                    f"Isolated test {i + 1} ({test_name}) failed and was not eligible "
                    "for rerun under this stage's rerun policy"
                )

        outcomes.append(outcome)
        if outcome.failed_after_rerun or outcome.not_rerun:
            iso_failed += 1
        else:
            iso_passed += 1

        try:
            os.remove(single_test_file)
        except OSError:
            pass

    print_phase_summary("ISOLATION TESTS", iso_passed, iso_failed, time.monotonic() - t0)
    return outcomes
