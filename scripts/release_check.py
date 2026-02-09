#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
import subprocess as sp
import sys
import time


def run_cmd(cmd):
    """Run a command and return output, raising error by default if command fails"""

    print(f"Running command: {cmd}")
    result = sp.run(cmd,
                    shell=True,
                    stdout=sp.PIPE,
                    text=True,
                    stderr=sp.STDOUT)
    if result.returncode != 0:
        print(f"Failing command output:\n{result.stdout}")
        sys.exit(1)
    return result


def run_precommit_with_timing():
    """Run pre-commit with timing information for each hook"""

    print("Running pre-commit checks with performance monitoring...")
    print("=" * 80)

    cmd = "pre-commit run -a --show-diff-on-failure --verbose"

    # Track hook execution times
    # Since hooks run sequentially, we can estimate each hook's duration
    # by tracking when each hook result appears
    hook_timings = []  # List of (hook_name, start_time, end_time, status)
    last_hook_end_time = None
    total_start_time = time.time()

    # Pattern to match hook result lines like "isort....................................................................Passed"
    # or "isort....................................................................Failed"
    hook_result_pattern = re.compile(r'^([^\.]+)\.+(\w+)$')

    # Use Popen to capture real-time output
    process = sp.Popen(cmd,
                       shell=True,
                       stdout=sp.PIPE,
                       stderr=sp.STDOUT,
                       text=True,
                       bufsize=1,
                       universal_newlines=True)

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        line_stripped = line.strip()

        # Check if this is a hook result line (e.g., "isort........Passed")
        match = hook_result_pattern.match(line_stripped)
        if match:
            hook_name = match.group(1).strip()
            status = match.group(2)
            hook_end_time = time.time()

            # Estimate start time: use last hook's end time, or total start time for first hook
            if last_hook_end_time is None:
                hook_start_time = total_start_time
            else:
                hook_start_time = last_hook_end_time

            hook_timings.append(
                (hook_name, hook_start_time, hook_end_time, status))
            last_hook_end_time = hook_end_time

            print(line, end='')  # Print the original line
        else:
            # Print other lines normally
            print(line, end='')

    # Wait for process to complete
    returncode = process.wait()
    total_time = time.time() - total_start_time

    # Calculate and print timing summary
    print("\n" + "=" * 80)
    print("PRE-COMMIT PERFORMANCE SUMMARY")
    print("=" * 80)
    print(
        f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )

    # Calculate durations and sort by duration (descending) to identify slowest hooks
    hook_durations = []
    for hook_name, start_time, end_time, status in hook_timings:
        duration = end_time - start_time
        hook_durations.append((hook_name, duration, status))

    # Sort by duration (longest first)
    hook_durations.sort(key=lambda x: x[1], reverse=True)

    print(
        f"\nHook execution timing (sorted by duration, {len(hook_durations)} hooks total):"
    )
    print(f"{'Hook Name':<50} {'Duration (seconds)':<25} {'Status':<15}")
    print("-" * 90)
    for hook_name, duration, status in hook_durations:
        duration_str = f"{duration:.2f} ({duration/60:.2f} min)"
        print(f"{hook_name:<50} {duration_str:<25} {status:<15}")

    # Show top 5 slowest hooks
    if len(hook_durations) > 0:
        print(f"\nTop 5 slowest hooks:")
        for i, (hook_name, duration,
                status) in enumerate(hook_durations[:5], 1):
            print(
                f"  {i}. {hook_name}: {duration:.2f}s ({duration/60:.2f} min)")

    print("=" * 80)

    if returncode != 0:
        print(f"\nPre-commit checks failed with return code {returncode}")
        # Print full output for debugging
        print("\nFull output:")
        print(''.join(output_lines))
        sys.exit(1)

    # Create a result-like object for compatibility
    class Result:

        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    return Result(returncode, ''.join(output_lines))


def handle_check_failure(error_msg):
    """Helper function to handle check failures with consistent messaging"""

    print(f"\nError: {error_msg}")
    print(
        "Please refer to our coding style guidelines at: https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md#coding-style to fix this issue"
    )
    sys.exit(1)


def main():
    # Install pre-commit and bandit from requirements-dev.txt
    with open("requirements-dev.txt") as f:
        reqs = f.readlines()
        pre_commit_req = next(line for line in reqs if "pre-commit" in line)
        bandit_req = next(line for line in reqs if "bandit" in line)

    run_cmd(f"pip3 install {pre_commit_req.strip()}")
    run_cmd(f"pip3 install {bandit_req.strip()}")

    # Install pre-commit hooks
    run_cmd("pre-commit install")

    # Run pre-commit on all files with performance monitoring
    try:
        run_precommit_with_timing()
    except SystemExit:
        handle_check_failure("pre-commit checks failed")

    # Run bandit security checks
    bandit_output = run_cmd(
        "bandit --configfile scripts/bandit.yaml -r tensorrt_llm").stdout
    print(f"Bandit output:\n{bandit_output}")

    # Check bandit results
    if "Total lines skipped (#nosec): 0" not in bandit_output:
        handle_check_failure("Found #nosec annotations in code")

    if "Issue:" in bandit_output:
        handle_check_failure("Bandit found security issues")

    print("pre-commit and bandit checks passed")


if __name__ == "__main__":
    main()
