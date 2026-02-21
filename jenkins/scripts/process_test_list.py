#!/usr/bin/env python3
"""Process test list to extract ISOLATION markers and split into regular/isolated tests.

Usage:
    python3 process_test_list.py \
        --llm-src <path> \
        --test-list <path> \
        --split-id <int> \
        --splits <int> \
        [--perf-mode]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Tuple


def extract_isolation_markers(test_db_list: Path) -> Tuple[List[str], List[str]]:
    """Extract tests with ISOLATION markers and return cleaned versions."""
    regular_tests = []
    isolation_tests = []

    with open(test_db_list, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "ISOLATION" in line:
                # Remove ISOLATION marker and nearby commas
                cleaned = re.sub(r"ISOLATION\s*,?|,\s*ISOLATION", "", line).strip()
                isolation_tests.append(cleaned)
                regular_tests.append(cleaned)
            else:
                regular_tests.append(line)

    return regular_tests, isolation_tests


def collect_shard_tests(
    llm_src: Path, cleaned_test_list: Path, split_id: int, splits: int, perf_mode: bool
) -> List[str]:
    """Collect tests for current shard using pytest collection."""
    if perf_mode:
        # Skip pytest collection in perf mode
        with open(cleaned_test_list, "r") as f:
            return [line.strip() for line in f if line.strip()]

    try:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--collect-only",
            "--splitting-algorithm",
            "least_duration",
            f"--test-list={cleaned_test_list}",
            "--quiet",
            "--splits",
            str(splits),
            "--group",
            str(split_id),
        ]

        result = subprocess.run(
            cmd,
            cwd=llm_src / "tests" / "integration" / "defs",
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Pytest collection failed: {result.stderr}")

        # Parse output to extract test names
        shard_tests = []
        found_running = False
        for line in result.stdout.split("\n"):
            if "Running" in line and "items in this shard" in line:
                found_running = True
                continue
            if found_running and "==" in line:
                break
            if found_running and "::" in line:
                shard_tests.append(line.strip())

        return shard_tests

    except Exception as e:
        print(f"Error during pytest collection: {e}", file=sys.stderr)
        raise


def split_shard_tests(
    shard_tests: List[str], regular_tests: List[str], isolation_tests: List[str]
) -> Tuple[List[str], List[str]]:
    """Split shard tests into regular and isolated categories."""
    shard_regular = []
    shard_isolated = []

    for test in shard_tests:
        test_trimmed = test.strip()

        # Handle test_unittests.py pattern
        if "test_unittests.py::test_unittests_v2[" in test_trimmed:
            match = re.search(r"\[(.*?)\]", test_trimmed)
            if match:
                test_trimmed = match.group(1)

        # Check if this test is in isolation list
        is_isolated = any(test_trimmed in iso_test for iso_test in isolation_tests)

        if is_isolated:
            # Find the full isolation test line
            matching = [t for t in isolation_tests if test_trimmed in t]
            if matching:
                shard_isolated.append(matching[0])
        else:
            # Find the full regular test line
            matching = [t for t in regular_tests if test_trimmed in t]
            if matching:
                shard_regular.append(matching[0])

    return shard_regular, shard_isolated


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-src", required=True, help="Path to TensorRT-LLM source")
    parser.add_argument("--test-list", required=True, help="Path to test list file")
    parser.add_argument("--split-id", type=int, default=1, help="Current split number")
    parser.add_argument("--splits", type=int, default=1, help="Total number of splits")
    parser.add_argument("--perf-mode", action="store_true", help="Skip pytest collection")

    args = parser.parse_args()

    llm_src = Path(args.llm_src)
    test_list = Path(args.test_list)

    if not test_list.exists():
        print(f"Error: Test list file not found: {test_list}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing test list: {test_list}")
    print(f"Split: {args.split_id} / {args.splits}")
    print(f"Performance mode: {args.perf_mode}")

    # Step 1: Extract ISOLATION markers
    print("Extracting ISOLATION markers...")
    regular_tests, isolation_tests = extract_isolation_markers(test_list)

    cleaned_list = test_list.parent / f"{test_list.stem}_cleaned.txt"
    with open(cleaned_list, "w") as f:
        f.write("\n".join(regular_tests))

    print(f"Created cleaned test list: {cleaned_list}")
    print(f"Found {len(isolation_tests)} tests with ISOLATION markers")

    # Step 2: Collect shard tests
    print("Collecting shard tests via pytest...")
    shard_tests = collect_shard_tests(
        llm_src, cleaned_list, args.split_id, args.splits, args.perf_mode
    )

    # Step 3: Split into regular and isolated
    print("Splitting shard tests into regular and isolated...")
    shard_regular, shard_isolated = split_shard_tests(shard_tests, regular_tests, isolation_tests)

    # Step 4: Write output files
    regular_list = test_list.parent / f"{test_list.stem}_regular.txt"
    isolate_list = test_list.parent / f"{test_list.stem}_isolate.txt"

    with open(regular_list, "w") as f:
        f.write("\n".join(shard_regular) if shard_regular else "")

    with open(isolate_list, "w") as f:
        f.write("\n".join(shard_isolated) if shard_isolated else "")

    print(f"Created {regular_list} with {len(shard_regular)} regular tests")
    print(f"Created {isolate_list} with {len(shard_isolated)} isolated tests")

    # Output results in key=value format for Groovy parsing
    result: dict[str, Any] = {
        "REGULAR_TEST_LIST": str(regular_list),
        "ISOLATE_TEST_LIST": str(isolate_list),
        "REGULAR_COUNT": len(shard_regular),
        "ISOLATE_COUNT": len(shard_isolated),
    }

    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
