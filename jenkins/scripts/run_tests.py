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
"""Run a CI stage's tests.

Renders the shard, runs the regular and isolated tests, reruns eligible
failures, and merges the results. Invoked from L0_Test.groovy (K8s pods and
SLURM agents) and slurm_run.sh (single-node sbatch); the work is done by the
ci_test_runner package.
"""

import argparse
import os
import shlex
import sys
import time
import traceback

# ci_test_runner and test_rerun live next to this script; import them from here
# even when Python does not put the script's directory on sys.path (-P).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ci_test_runner import SetupError  # noqa: E402
from ci_test_runner.console import format_duration, print_banner  # noqa: E402
from ci_test_runner.execution import (  # noqa: E402
    RunOutcome,
    run_isolated_tests,
    run_regular_tests,
    write_failure_evidence,
)
from ci_test_runner.render import render_test_list  # noqa: E402
from ci_test_runner.results import create_empty_results_xml, merge_results  # noqa: E402
from test_rerun import load_rerun_policy  # noqa: E402

# Exit codes. Callers treat EXIT_TESTS_FAILED as a deterministic test failure
# (no infra retry); EXIT_SETUP_FAILED and EXIT_INTERNAL_ERROR mean the run did
# not complete. Keep in sync with RUN_TESTS_EXIT_* in jenkins/L0_Test.groovy.
EXIT_TESTS_FAILED = 1
EXIT_SETUP_FAILED = 2
EXIT_INTERNAL_ERROR = 3
# Failures that were never rerun: unfinished (crashed/timed out) tests, or tests
# the stage's rerun policy (jenkins/config/rerun_policy.yml) does not rerun.
EXIT_TESTS_NOT_RERUN = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified TensorRT-LLM test runner")
    parser.add_argument(
        "--test-db-list", required=True, help="Path to the test-db rendered test list file"
    )
    parser.add_argument("--splits", type=int, default=1, help="Total number of shards")
    parser.add_argument("--group", type=int, default=1, help="This shard group number (1-based)")
    parser.add_argument("--pytest-base-cmd", required=True, help="Base pytest command string")
    parser.add_argument(
        "--pytest-target",
        action="append",
        default=[],
        help="Positional pytest node ID added to collection and every pytest run (repeatable)",
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
        "--durations-path",
        default=None,
        help="Path to per-cluster .test_durations file for least_duration splitting. "
        "Overrides the default .test_durations when set.",
    )
    return parser.parse_args(argv)


def run(args):
    """Run the stage's tests and return the process exit code."""
    working_dir = args.working_dir or os.getcwd()
    output_dir = args.output_dir
    stage_name = args.stage_name
    fail_signatures = [s for s in args.fail_signatures.split(",") if s]
    os.makedirs(output_dir, exist_ok=True)
    # Load it before any test runs, so an invalid policy file fails the stage early.
    rerun_policy, rerun_rule = load_rerun_policy(stage_name)
    rule_source = f"stage rule '{rerun_rule}'" if rerun_rule else "default"
    print(f"Rerun policy for {stage_name} ({rule_source}): {rerun_policy}")

    regular_list, isolate_list, regular_count, isolate_count = render_test_list(
        args.test_db_list,
        working_dir,
        args.splits,
        args.group,
        args.durations_path,
        args.pytest_target,
    )
    if regular_count == 0 and isolate_count == 0:
        create_empty_results_xml(output_dir, stage_name)
        raise SetupError(f"No tests were executed for stage {stage_name}")

    pytest_base_cmd = args.pytest_base_cmd
    if args.pytest_target:
        pytest_base_cmd = f"{pytest_base_cmd} {shlex.join(args.pytest_target)}"
    total_t0 = time.monotonic()
    print_banner(
        f"TEST EXECUTION START — stage [{stage_name}]  "
        f"({regular_count} regular, {isolate_count} isolation)"
    )

    outcomes = []
    if regular_count > 0:
        outcomes.append(
            run_regular_tests(
                f"{pytest_base_cmd} --test-list={regular_list}",
                regular_list,
                working_dir,
                output_dir,
                stage_name,
                fail_signatures,
                rerun_policy=rerun_policy,
            )
        )
    else:
        print(f"No regular tests to run for stage {stage_name}")
        create_empty_results_xml(output_dir, stage_name)
        outcomes.append(RunOutcome(os.path.join(output_dir, "results.xml")))

    if isolate_count > 0:
        outcomes.extend(
            run_isolated_tests(
                pytest_base_cmd,
                isolate_list,
                working_dir,
                output_dir,
                stage_name,
                fail_signatures,
                rerun_policy=rerun_policy,
            )
        )
    else:
        print(f"No isolated tests to run for stage {stage_name}")

    print_banner("RESULT MERGING", char="-")
    merge_results(output_dir, stage_name, outcomes)

    print_banner(f"TEST EXECUTION COMPLETE — stage [{stage_name}]")
    print(f"  Regular tests : {regular_count}")
    print(f"  Isolation tests : {isolate_count}")
    print(f"  Total time      : {format_duration(time.monotonic() - total_t0)}")

    if any(o.failed_after_rerun for o in outcomes):
        write_failure_evidence(output_dir, [log for o in outcomes for log in o.failed_rerun_logs])
        print("  Result          : FAILED")
        print("\nSome tests still failed after rerun attempts, please check the test report.")
        return EXIT_TESTS_FAILED
    if any(o.not_rerun for o in outcomes):
        write_failure_evidence(output_dir, [o.first_run_log for o in outcomes if o.not_rerun])
        print("  Result          : FAILED")
        print(
            "\nSome tests failed and were not eligible for rerun under this stage's rerun "
            "policy (the rerun list log says why), please check the test report."
        )
        return EXIT_TESTS_NOT_RERUN
    print("  Result          : PASSED")
    print("\nAll tests passed.")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except SetupError as e:
        print(f"Error: {e}")
        return EXIT_SETUP_FAILED
    except Exception:
        # An uncaught exception would exit with 1, which callers read as EXIT_TESTS_FAILED.
        traceback.print_exc()
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    # Line-buffer stdout so banners interleave correctly with pytest's output.
    sys.stdout.reconfigure(line_buffering=True)
    sys.exit(main())
