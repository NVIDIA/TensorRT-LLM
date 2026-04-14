#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify a CUDA Graph implementation produces correct results.

Standalone script -- only Python stdlib required (torch needed at runtime
for GPU verification, but not for --mock mode).
Outputs structured JSON to stdout.

Runs the original and CUDA-graphed workload scripts as subprocesses,
parses their structured ``KEY:VALUE`` output lines, and compares numeric
values within tolerance.

Usage:
    python verify_workload.py --original orig.py --modified mod.py
    python verify_workload.py --mock
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def _run_script(
    script_path: str,
    timeout: int = 60,
) -> tuple[dict | None, str | None]:
    """Run a Python script and return parsed KEY:VALUE output.

    Args:
        script_path: Absolute path to the workload script.
        timeout: Execution timeout in seconds.

    Returns:
        Tuple of (parsed_output_dict, error_message).
        On success error_message is None; on failure parsed_output is None.
    """
    if not os.path.exists(script_path):
        return None, f"File not found: {script_path}"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(script_path)),
        )
    except subprocess.TimeoutExpired:
        return None, f"Script timed out after {timeout}s"

    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else "unknown error"
        return None, f"Exit code {result.returncode}: {stderr}"

    output: dict = {}
    for line in result.stdout.strip().split("\n"):
        match = re.match(r"^([A-Z][A-Z0-9_]+):(.+)$", line)
        if match:
            key, value = match.group(1), match.group(2).strip()
            try:
                output[key] = float(value)
            except ValueError:
                output[key] = value

    if not output:
        return None, f"No KEY:VALUE lines in output: {result.stdout[:300]}"

    return output, None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _compare(
    original: dict,
    modified: dict,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> tuple[list[str], float]:
    """Compare parsed outputs within tolerance.

    Returns:
        Tuple of (mismatch_messages, max_abs_diff).
    """
    mismatches: list[str] = []
    max_abs = 0.0

    for key, orig_val in original.items():
        if key not in modified:
            mismatches.append(f"Missing key in modified output: {key}")
            continue
        mod_val = modified[key]

        if isinstance(orig_val, float) and isinstance(mod_val, float):
            if math.isnan(orig_val) or math.isnan(mod_val):
                if not (math.isnan(orig_val) and math.isnan(mod_val)):
                    mismatches.append(f"{key}: NaN mismatch")
                continue
            diff = abs(orig_val - mod_val)
            max_abs = max(max_abs, diff)
            if diff > atol + rtol * abs(orig_val):
                mismatches.append(f"{key}: {orig_val:.6f} vs {mod_val:.6f} (diff={diff:.2e})")
        elif str(orig_val) != str(mod_val):
            mismatches.append(f"{key}: {orig_val!r} vs {mod_val!r}")

    return mismatches, max_abs


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data() -> dict:
    """Return realistic mock verification data for testing."""
    return {
        "correct": True,
        "max_abs_diff": 0.0,
        "details": "Mock mode -- no GPU required",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Verify CUDA-graphed workload produces correct results."
    )
    parser.add_argument(
        "--original",
        help="Path to original workload script.",
    )
    parser.add_argument(
        "--modified",
        help="Path to modified workload script.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance (default: 1e-4).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance (default: 1e-4).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-script execution timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing (no GPU required).",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data()
    elif args.original and args.modified:
        orig_out, err = _run_script(args.original, args.timeout)
        if err:
            data = {"correct": False, "max_abs_diff": float("inf"), "details": err}
            json.dump(data, sys.stdout, indent=2)
            print()
            sys.exit(1)

        mod_out, err = _run_script(args.modified, args.timeout)
        if err:
            data = {"correct": False, "max_abs_diff": float("inf"), "details": err}
            json.dump(data, sys.stdout, indent=2)
            print()
            sys.exit(1)

        mismatches, max_abs = _compare(orig_out, mod_out, rtol=args.rtol, atol=args.atol)
        correct = not mismatches
        data = {
            "correct": correct,
            "max_abs_diff": max_abs,
            "details": "All outputs match" if correct else "; ".join(mismatches),
        }
    else:
        parser.error("Either --mock or both --original and --modified required.")

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
