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

import subprocess as sp
import sys


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

    # Run pre-commit on all files
    try:
        run_cmd("pre-commit run -a --show-diff-on-failure")
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
