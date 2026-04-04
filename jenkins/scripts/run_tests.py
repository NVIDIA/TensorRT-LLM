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
"""Unified test runner for TensorRT-LLM CI.

Handles regular tests, isolated tests, rerun logic, and result merging
in a single script that can be invoked from both Groovy (K8s pod) and
SLURM (slurm_run.sh) execution paths.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Allow importing test_rerun from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_rerun


# ---------------------------------------------------------------------------
# render_test_list: replaces Groovy processShardTestList()
# ---------------------------------------------------------------------------
def render_test_list(test_db_list, working_dir, splits, group, perf_mode):
    """Parse test-db list, split into shards, separate regular and isolated tests.

    Args:
        test_db_list: Path to the test-db rendered test list file.
        working_dir: Working directory for pytest (tests/integration/defs).
        splits: Total number of shards.
        group: This shard's group number (1-based).
        perf_mode: If True, skip pytest collection and use all tests as regular.

    Returns:
        Tuple of (regular_list_path, isolate_list_path, regular_count, isolate_count).
    """
    original_lines = Path(test_db_list).read_text().splitlines()

    cleaned_lines = []
    isolation_tests = set()

    for line in original_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "ISOLATION" in stripped:
            cleaned = stripped
            if "ISOLATION," in cleaned:
                cleaned = cleaned.replace("ISOLATION,", "").strip()
            elif ",ISOLATION" in cleaned:
                cleaned = cleaned.replace(",ISOLATION", "").strip()
            else:
                cleaned = cleaned.replace(" ISOLATION", "").strip()
            isolation_tests.add(cleaned)
            cleaned_lines.append(cleaned)
        else:
            cleaned_lines.append(stripped)

    # Write cleaned test list (without ISOLATION markers)
    cleaned_file = test_db_list.replace(".txt", "_cleaned.txt")
    Path(cleaned_file).write_text("\n".join(cleaned_lines) + "\n" if cleaned_lines else "")
    print(f"Created cleaned testDBList: {cleaned_file} with {len(cleaned_lines)} lines")
    print(f"Original testDBList contains {len(isolation_tests)} tests with ISOLATION markers")

    shard_tests = []

    if perf_mode:
        print("Performance mode enabled - skipping pytest collection, using all tests as regular")
    else:
        # Use pytest --collect-only to determine which tests belong to this shard
        # Clear MPI/SLURM env vars to prevent MPI_Init during collection
        # (same prefixes as slurm_run.sh and trtllm-llmapi-launch)
        mpi_prefixes = ("PMI", "PMIX", "MPI", "OMPI", "SLURM", "UCX")
        env_vars = {
            k: v for k, v in os.environ.items() if not any(k.startswith(p) for p in mpi_prefixes)
        }
        collect_output_dir = os.path.join(os.path.dirname(test_db_list), "collect_output")
        os.makedirs(collect_output_dir, exist_ok=True)
        collect_cmd = (
            f"pytest --collect-only --splitting-algorithm least_duration "
            f"--test-list={cleaned_file} --quiet "
            f"--splits {splits} --group {group} "
            f"--output-dir={collect_output_dir}"
        )
        print(f"Running: {collect_cmd}")
        result = subprocess.run(
            collect_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            env=env_vars,
        )

        output = result.stdout.strip()
        print("<<<START_PYTEST_OUTPUT>>>")
        print(output)
        print("<<<END_PYTEST_OUTPUT>>>")

        if result.stderr:
            print(f"pytest --collect-only stderr:\n{result.stderr}")

        if result.returncode != 0 and not output:
            print(f"Error: pytest --collect-only failed with exit code {result.returncode}")
            print(f"stderr: {result.stderr}")
            sys.exit(1)

        # Parse output: collect test IDs after "Running N items in this shard" line
        found_running_line = False
        for line in output.split("\n"):
            if re.search(r"Running \d+ items in this shard", line) or re.search(
                r"\[pytest-split\] Running group", line
            ):
                found_running_line = True
                continue
            if found_running_line and "======================" in line:
                found_running_line = False
                continue
            if found_running_line and "::" in line:
                shard_tests.append(line.strip())

        print(f"Filtering complete. shard_tests size: {len(shard_tests)}")

    # Split into regular and isolate
    regular_tests = []
    isolate_tests = []

    if perf_mode:
        regular_tests = [t for t in cleaned_lines if t.strip()]
    else:
        for test in shard_tests:
            trimmed = test.strip()
            if not trimmed:
                continue
            # Handle test_unittests.py::test_unittests_v2[xxxx] pattern
            if trimmed.startswith("test_unittests.py::test_unittests_v2[") and trimmed.endswith(
                "]"
            ):
                start = trimmed.index("[") + 1
                end = trimmed.rindex("]")
                trimmed = trimmed[start:end]

            # Check if this test is in the isolation set
            isolation_match = next((t for t in isolation_tests if trimmed in t), None)
            if isolation_match:
                isolate_tests.append(isolation_match)
            else:
                cleaned_match = next((t for t in cleaned_lines if trimmed in t), None)
                if cleaned_match:
                    regular_tests.append(cleaned_match)

    # Write regular and isolate list files
    regular_file = test_db_list.replace(".txt", "_regular.txt")
    isolate_file = test_db_list.replace(".txt", "_isolate.txt")

    Path(regular_file).write_text("\n".join(regular_tests) + "\n" if regular_tests else "")
    Path(isolate_file).write_text("\n".join(isolate_tests) + "\n" if isolate_tests else "")

    print(f"Created {regular_file} with {len(regular_tests)} regular tests")
    print(f"Created {isolate_file} with {len(isolate_tests)} isolate tests")

    return regular_file, isolate_file, len(regular_tests), len(isolate_tests)


# ---------------------------------------------------------------------------
# Core pytest execution
# ---------------------------------------------------------------------------
def run_pytest(pytest_cmd, working_dir):
    """Execute a pytest command and return the exit code.

    Args:
        pytest_cmd: The full pytest command string.
        working_dir: Working directory for pytest execution.

    Returns:
        Process exit code.
    """
    print(f"Running pytest: {pytest_cmd}")
    result = subprocess.run(pytest_cmd, shell=True, cwd=working_dir)
    print(f"Pytest finished with exit code {result.returncode}")
    return result.returncode


def build_rerun_command(base_cmd, test_list, xml_path, csv_path, reruns):
    """Build a rerun pytest command by stripping split/cov args and replacing test-list/output args.

    Args:
        base_cmd: The original pytest command string.
        test_list: Path to the rerun test list file.
        xml_path: Path for the rerun results XML.
        csv_path: Path for the rerun results CSV.
        reruns: Number of reruns (passed to --reruns flag).

    Returns:
        Modified pytest command string.
    """
    # Remove args that shouldn't be in rerun commands
    no_need_patterns = ["--splitting-algorithm", "--splits", "--group", "--cov"]
    need_to_change_patterns = ["--test-list", "--csv", "--periodic-junit-xmlpath"]

    parts = []
    tokens = base_cmd.split()
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        # Check if this token should be removed
        if any(token.startswith(p) for p in no_need_patterns):
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                skip_next = True
            continue
        # Check if this token should be replaced
        if any(token.startswith(p) for p in need_to_change_patterns):
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                skip_next = True
            continue
        parts.append(token)

    cmd = " ".join(parts)
    cmd += f" --test-list={test_list}"
    cmd += f" --csv={csv_path}"
    cmd += f" --periodic-junit-xmlpath {xml_path}"
    if reruns > 0:
        cmd += f" --reruns {reruns}"
    return cmd


def build_isolated_command(base_cmd, test_list, xml_path, csv_path):
    """Build an isolated test command by replacing test-list/output/prefix args.

    Args:
        base_cmd: The original pytest command string.
        test_list: Path to the single-test list file.
        xml_path: Path for the isolated results XML.
        csv_path: Path for the isolated results CSV.

    Returns:
        Modified pytest command string.
    """
    need_to_change_patterns = ["--test-list", "--test-prefix", "--csv", "--periodic-junit-xmlpath"]

    parts = []
    tokens = base_cmd.split()
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if any(token.startswith(p) for p in need_to_change_patterns):
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                skip_next = True
            continue
        parts.append(token)

    cmd = " ".join(parts)
    cmd += f" --test-list={test_list}"
    cmd += f" --csv={csv_path}"
    cmd += f" --periodic-junit-xmlpath {xml_path}"
    cmd += " --cov-append"
    return cmd


# ---------------------------------------------------------------------------
# Rerun logic: replaces Groovy rerunFailedTests()
# ---------------------------------------------------------------------------
def check_and_rerun(
    result_xml, base_cmd, working_dir, output_dir, rerun_tag, fail_signatures, max_rerun_tests
):
    """Analyze test failures and rerun eligible tests.

    Args:
        result_xml: Path to the results XML to analyze.
        base_cmd: The original pytest command string.
        working_dir: Working directory for pytest execution.
        output_dir: Base output directory for the stage.
        rerun_tag: Tag for this rerun (e.g., "regular", "isolated_0").
        fail_signatures: List of failure signature strings.
        max_rerun_tests: Max number of failed tests that can trigger rerun.

    Returns:
        Tuple of (is_rerun_failed: bool, rerun_xml_files: list of paths to rerun result XMLs).
    """
    if not os.path.exists(result_xml):
        print(f"No {result_xml} file found, skip the rerun step")
        return True, []

    rerun_dir = os.path.join(output_dir, "rerun", rerun_tag)
    os.makedirs(rerun_dir, exist_ok=True)

    # Step 1: Generate rerun test lists
    print(f"Generating rerun test lists for {rerun_tag}...")
    test_rerun.generate_rerun_tests_list(rerun_dir, result_xml, fail_signatures)

    # Step 2: If rerun_0.txt exists, some tests can't be rerun
    rerun_0_file = os.path.join(rerun_dir, "rerun_0.txt")
    if os.path.exists(rerun_0_file):
        print(f"Contents of {rerun_0_file}:")
        print(Path(rerun_0_file).read_text())
        print("There are some failed tests that cannot be rerun, skip the rerun step.")
        return True, []

    # Step 3: Count total failed tests
    valid_count = 0
    for times in [1, 2]:
        rerun_file = os.path.join(rerun_dir, f"rerun_{times}.txt")
        if os.path.exists(rerun_file):
            lines = [line for line in Path(rerun_file).read_text().splitlines() if line.strip()]
            count = len(lines)
            print(f"Found {count} {rerun_tag} tests to rerun {times} time(s)")
            valid_count += count

    if valid_count > max_rerun_tests:
        print(
            f"There are more than {max_rerun_tests} failed {rerun_tag} tests, skip the rerun step."
        )
        return True, []
    elif valid_count == 0:
        print(f"No failed {rerun_tag} tests need to be rerun.")
        return True, []

    # Step 4: Execute reruns
    is_rerun_failed = False
    rerun_xml_files = []

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
        rc = run_pytest(rerun_cmd, working_dir)

        if os.path.exists(xml_file):
            rerun_xml_files.append(xml_file)

        if rc != 0:
            if not os.path.exists(xml_file):
                print(f"The {rerun_tag} tests crashed during rerun attempt (no XML produced).")
                raise RuntimeError(f"Rerun crashed for {rerun_tag}, no XML produced")
            print(f"The {rerun_tag} tests still failed after rerun attempt.")
            is_rerun_failed = True

    print(f"is_rerun_failed for {rerun_tag}: {is_rerun_failed}")
    return is_rerun_failed, rerun_xml_files


# ---------------------------------------------------------------------------
# Regular test execution
# ---------------------------------------------------------------------------
def run_regular_tests(
    pytest_cmd, regular_list, working_dir, output_dir, stage_name, fail_signatures, max_rerun_tests
):
    """Run regular tests and rerun failures if applicable.

    Args:
        pytest_cmd: The full pytest command string (already includes --test-list).
        regular_list: Path to the regular test list file.
        working_dir: Working directory for pytest execution.
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
        fail_signatures: List of failure signature strings.
        max_rerun_tests: Max number of failed tests that can trigger rerun.

    Returns:
        Tuple of (rerun_failed: bool, all_xml_files: list of result XML paths).
    """
    result_xml = os.path.join(output_dir, "results.xml")
    all_xml_files = [result_xml]

    rc = run_pytest(pytest_cmd, working_dir)

    if rc != 0:
        is_rerun_failed, rerun_xmls = check_and_rerun(
            result_xml,
            pytest_cmd,
            working_dir,
            output_dir,
            "regular",
            fail_signatures,
            max_rerun_tests,
        )
        all_xml_files.extend(rerun_xmls)
        return is_rerun_failed, all_xml_files

    return False, all_xml_files


# ---------------------------------------------------------------------------
# Isolated test execution: replaces Groovy runIsolatedTests()
# ---------------------------------------------------------------------------
def run_isolated_tests(
    pytest_cmd, isolate_list, working_dir, output_dir, stage_name, fail_signatures, max_rerun_tests
):
    """Run isolated tests one by one, rerunning each on failure.

    Args:
        pytest_cmd: The base pytest command string.
        isolate_list: Path to the isolate test list file.
        working_dir: Working directory for pytest execution.
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
        fail_signatures: List of failure signature strings.
        max_rerun_tests: Max number of failed tests that can trigger rerun.

    Returns:
        Tuple of (rerun_failed: bool, all_xml_files: list of result XML paths).
    """
    isolate_tests = [
        line.strip() for line in Path(isolate_list).read_text().splitlines() if line.strip()
    ]
    rerun_failed = False
    all_xml_files = []

    for i, test_name in enumerate(isolate_tests):
        print(f"\n=== Isolated test {i}: {test_name} ===")

        # Create a temporary file for this single test
        single_test_file = os.path.join(output_dir, f"isolated_{i}.txt")
        Path(single_test_file).write_text(test_name + "\n")

        xml_path = os.path.join(output_dir, f"results_isolated_{i}.xml")
        csv_path = os.path.join(output_dir, f"report_isolated_{i}.csv")

        isolated_cmd = build_isolated_command(pytest_cmd, single_test_file, xml_path, csv_path)
        rc = run_pytest(isolated_cmd, working_dir)
        all_xml_files.append(xml_path)

        if rc != 0:
            try:
                is_rerun_failed, rerun_xmls = check_and_rerun(
                    xml_path,
                    isolated_cmd,
                    working_dir,
                    output_dir,
                    f"isolated_{i}",
                    fail_signatures,
                    max_rerun_tests,
                )
                all_xml_files.extend(rerun_xmls)
                if is_rerun_failed:
                    print(f"Isolated test {i} ({test_name}) failed after rerun attempt")
                    rerun_failed = True
            except RuntimeError as e:
                print(f"Isolated test {i} ({test_name}) rerun crashed: {e}")
                rerun_failed = True

        # Clean up temporary file
        try:
            os.remove(single_test_file)
        except OSError:
            pass

    if rerun_failed:
        print("One or more isolated tests failed after rerun attempts")

    return rerun_failed, all_xml_files


# ---------------------------------------------------------------------------
# Result merging: replaces Groovy generateRerunReport()
# ---------------------------------------------------------------------------
def merge_results(output_dir, stage_name, all_xml_files):
    """Merge all result XMLs and generate rerun report.

    Args:
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
        all_xml_files: List of all result XML file paths.
    """
    # Fix testsuite names in all XMLs
    for xml_file in all_xml_files:
        if os.path.exists(xml_file):
            try:
                content = Path(xml_file).read_text()
                content = content.replace(
                    'testsuite name="pytest"', f'testsuite name="{stage_name}"'
                )
                Path(xml_file).write_text(content)
            except Exception as e:
                print(f"Warning: Failed to fix testsuite name in {xml_file}: {e}")

    # Separate original results and rerun results
    rerun_result_files = []
    for xml_file in all_xml_files:
        if not os.path.exists(xml_file):
            continue
        # Rerun results live under rerun/ directory
        if "/rerun/" in xml_file and "rerun_results_" in xml_file:
            rerun_result_files.append(xml_file)

    # Also add original results that have corresponding reruns to the rerun report
    rerun_base_dir = os.path.join(output_dir, "rerun")
    original_results_with_reruns = []

    # Check regular reruns
    regular_rerun_dir = os.path.join(rerun_base_dir, "regular")
    if os.path.isdir(regular_rerun_dir):
        has_regular_reruns = any(
            f.startswith("rerun_results_") and f.endswith(".xml")
            for f in os.listdir(regular_rerun_dir)
        )
        if has_regular_reruns:
            results_xml = os.path.join(output_dir, "results.xml")
            if os.path.exists(results_xml):
                original_results_with_reruns.append(results_xml)

    # Check isolated reruns
    if os.path.isdir(rerun_base_dir):
        for d in os.listdir(rerun_base_dir):
            if d.startswith("isolated_") and os.path.isdir(os.path.join(rerun_base_dir, d)):
                iso_dir = os.path.join(rerun_base_dir, d)
                has_iso_reruns = any(
                    f.startswith("rerun_results_") and f.endswith(".xml")
                    for f in os.listdir(iso_dir)
                )
                if has_iso_reruns:
                    iso_num = d.replace("isolated_", "")
                    iso_result = os.path.join(output_dir, f"results_isolated_{iso_num}.xml")
                    if os.path.exists(iso_result):
                        original_results_with_reruns.append(iso_result)

    # Generate rerun report if any reruns occurred
    rerun_report_inputs = original_results_with_reruns + rerun_result_files
    if rerun_report_inputs:
        print(f"Generating rerun report with input files: {rerun_report_inputs}")
        rerun_report_xml = os.path.join(output_dir, "rerun_results.xml")
        test_rerun.generate_rerun_report(rerun_report_xml, rerun_report_inputs)

    # Merge all XMLs into a single results.xml for junit
    existing_xml_files = [f for f in all_xml_files if os.path.exists(f)]
    if existing_xml_files:
        merged_output = os.path.join(output_dir, "results.xml")
        print(f"Merging {len(existing_xml_files)} XML files into {merged_output}")
        test_rerun.merge_junit_xmls(merged_output, existing_xml_files, deduplicate=True)

    # Remove isolation results since they are merged into results.xml
    for f in all_xml_files:
        if "results_isolated_" in f and os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print("Result merging completed")


# ---------------------------------------------------------------------------
# Create empty results XML
# ---------------------------------------------------------------------------
def create_empty_results_xml(output_dir, stage_name):
    """Create an empty results.xml for stages with no tests.

    Args:
        output_dir: Output directory for the stage.
        stage_name: Name of the test stage.
    """
    os.makedirs(output_dir, exist_ok=True)
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<testsuites>\n"
        f'<testsuite name="{stage_name}" errors="0" failures="0" skipped="0" tests="0" time="0.0">\n'
        "</testsuite>\n"
        "</testsuites>\n"
    )
    Path(os.path.join(output_dir, "results.xml")).write_text(content)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified TensorRT-LLM test runner")

    # Render mode arguments
    parser.add_argument(
        "--render",
        action="store_true",
        help="Run render_test_list to split test-db list into regular/isolate shards",
    )
    parser.add_argument("--test-db-list", help="Path to the test-db rendered test list file")
    parser.add_argument("--splits", type=int, default=1, help="Total number of shards")
    parser.add_argument("--group", type=int, default=1, help="This shard group number (1-based)")
    parser.add_argument(
        "--perf-mode", action="store_true", help="Performance mode: skip collection, run all tests"
    )

    # Run mode arguments
    parser.add_argument("--pytest-base-cmd", help="Base pytest command string")
    parser.add_argument(
        "--regular-test-list", help="Path to regular test list (if not using --render)"
    )
    parser.add_argument(
        "--isolate-test-list", help="Path to isolate test list (if not using --render)"
    )
    parser.add_argument("--stage-name", required=True, help="Name of the test stage")
    parser.add_argument("--output-dir", required=True, help="Output directory for the stage")
    parser.add_argument("--working-dir", help="Working directory for pytest (defaults to cwd)")
    parser.add_argument(
        "--fail-signatures",
        default="",
        help="Comma-separated list of failure signatures for rerun eligibility",
    )
    parser.add_argument(
        "--max-rerun-tests", type=int, default=5, help="Max failed tests to trigger rerun"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    working_dir = args.working_dir or os.getcwd()
    output_dir = args.output_dir
    stage_name = args.stage_name
    fail_signatures = [s for s in args.fail_signatures.split(",") if s]
    max_rerun_tests = args.max_rerun_tests

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Render test list if requested
    regular_list = args.regular_test_list
    isolate_list = args.isolate_test_list
    regular_count = 0
    isolate_count = 0

    if args.render:
        if not args.test_db_list:
            print("Error: --test-db-list is required when using --render")
            sys.exit(1)
        regular_list, isolate_list, regular_count, isolate_count = render_test_list(
            args.test_db_list, working_dir, args.splits, args.group, args.perf_mode
        )
    else:
        if regular_list and os.path.exists(regular_list):
            regular_count = len(
                [line for line in Path(regular_list).read_text().splitlines() if line.strip()]
            )
        if isolate_list and os.path.exists(isolate_list):
            isolate_count = len(
                [line for line in Path(isolate_list).read_text().splitlines() if line.strip()]
            )

    if not args.pytest_base_cmd:
        # Render-only mode: just output the lists and exit
        if args.render:
            print(f"regular_list={regular_list}")
            print(f"isolate_list={isolate_list}")
            print(f"regular_count={regular_count}")
            print(f"isolate_count={isolate_count}")
            sys.exit(0)
        else:
            print("Error: --pytest-base-cmd is required for test execution")
            sys.exit(1)

    pytest_base_cmd = args.pytest_base_cmd

    # Step 2: Run regular tests
    all_xml_files = []
    rerun_failed = False

    if regular_count > 0:
        # Add --test-list to the base command for regular tests
        regular_cmd = f"{pytest_base_cmd} --test-list={regular_list}"
        failed, xml_files = run_regular_tests(
            regular_cmd,
            regular_list,
            working_dir,
            output_dir,
            stage_name,
            fail_signatures,
            max_rerun_tests,
        )
        rerun_failed = rerun_failed or failed
        all_xml_files.extend(xml_files)
    else:
        print(f"No regular tests to run for stage {stage_name}")
        create_empty_results_xml(output_dir, stage_name)
        all_xml_files.append(os.path.join(output_dir, "results.xml"))

    # Step 3: Run isolated tests
    if isolate_count > 0:
        print(f"\n{'=' * 60}")
        print(f"Running {isolate_count} isolated tests")
        print(f"{'=' * 60}")
        failed, xml_files = run_isolated_tests(
            pytest_base_cmd,
            isolate_list,
            working_dir,
            output_dir,
            stage_name,
            fail_signatures,
            max_rerun_tests,
        )
        rerun_failed = rerun_failed or failed
        all_xml_files.extend(xml_files)
    else:
        print(f"No isolated tests to run for stage {stage_name}")

    # Step 4: Check that at least some tests were executed
    if regular_count == 0 and isolate_count == 0:
        print(f"Error: No tests were executed for stage {stage_name}")
        sys.exit(1)

    # Step 5: Merge all results
    merge_results(output_dir, stage_name, all_xml_files)

    # Step 6: Exit with appropriate code
    if rerun_failed:
        print("Some tests still failed after rerun attempts, please check the test report.")
        sys.exit(1)

    print("All tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
