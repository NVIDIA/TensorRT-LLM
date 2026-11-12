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
"""Split the test-db list into this shard's regular and isolated tests."""

import os
import re
import shlex
import subprocess
from pathlib import Path

from . import SetupError

COLLECT_TIMEOUT_SECONDS = 600

# Same markers, in the same order, as kVALID_TEST_LIST_MARKERS in
# tests/integration/defs/test_list_parser.py.
_TEST_LIST_MARKERS = ("XFAIL", "SKIP", "UNSTABLE", "TIMEOUT")

# Env prefixes cleared for collection so it does not call MPI_Init (same
# prefixes as slurm_run.sh and trtllm-llmapi-launch).
_MPI_ENV_PREFIXES = ("PMI", "PMIX", "MPI", "OMPI", "SLURM", "UCX")


def _test_list_name(line):
    """Return the test name of a test list line, without its trailing marker."""
    for marker in _TEST_LIST_MARKERS:
        name, sep, _ = line.partition(f" {marker}")
        if sep:
            return name.strip()
    return line.strip()


def _find_test_list_line(test_name, lines):
    """Return the line in ``lines`` that ``test_name`` was collected from.

    An exact name match wins. A substring match is only a fallback, because a
    test name can be a substring of an unrelated line, e.g. a unittest file
    that another line excludes with ``--ignore=<file>``.
    """
    name = _test_list_name(test_name)
    exact = next((line for line in lines if _test_list_name(line) == name), None)
    if exact is not None:
        return exact
    return next((line for line in lines if test_name in line), None)


def _strip_isolation_marker(line):
    if "ISOLATION," in line:
        return line.replace("ISOLATION,", "").strip()
    if ",ISOLATION" in line:
        return line.replace(",ISOLATION", "").strip()
    return line.replace(" ISOLATION", "").strip()


def _collect_shard_tests(
    cleaned_file, working_dir, splits, group, durations_path, pytest_targets, output_dir
):
    """Return the test ids that pytest-split assigns to this shard."""
    env = {
        k: v for k, v in os.environ.items() if not any(k.startswith(p) for p in _MPI_ENV_PREFIXES)
    }
    collect_output_dir = os.path.join(output_dir, "collect_output")
    os.makedirs(collect_output_dir, exist_ok=True)
    durations_arg = f" --durations-path={durations_path}" if durations_path else ""
    targets_arg = f" {shlex.join(pytest_targets)}" if pytest_targets else ""
    collect_cmd = (
        f"pytest --collect-only --splitting-algorithm least_duration "
        f"--test-list={cleaned_file} --quiet "
        f"--splits {splits} --group {group} "
        f"--output-dir={collect_output_dir}"
        f"{durations_arg}{targets_arg}"
    )
    print(f"Running: {collect_cmd}")
    try:
        result = subprocess.run(
            collect_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            env=env,
            timeout=COLLECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        for name, stream in (("stdout", e.stdout), ("stderr", e.stderr)):
            partial = stream.strip() if isinstance(stream, str) else ""
            if partial:
                print(f"partial {name}:\n{partial}")
        raise SetupError(
            f"pytest --collect-only timed out after {COLLECT_TIMEOUT_SECONDS}s "
            "(possible import-time deadlock)"
        ) from e

    output = result.stdout.strip()
    print("<<<START_PYTEST_OUTPUT>>>")
    print(output)
    print("<<<END_PYTEST_OUTPUT>>>")
    if result.stderr:
        print(f"pytest --collect-only stderr:\n{result.stderr}")
    if result.returncode != 0:
        raise SetupError(f"pytest --collect-only failed with exit code {result.returncode}")

    # The shard's test ids follow the pytest-split "Running ..." line.
    shard_tests = []
    in_shard = False
    for line in output.split("\n"):
        if re.search(r"Running \d+ items in this shard", line) or re.search(
            r"\[pytest-split\] Running group", line
        ):
            in_shard = True
            continue
        if in_shard and "======================" in line:
            in_shard = False
            continue
        if in_shard and "::" in line:
            shard_tests.append(line.strip())
    print(f"Filtering complete. shard_tests size: {len(shard_tests)}")
    return shard_tests


def render_test_list(
    test_db_list, working_dir, splits, group, durations_path=None, pytest_targets=None
):
    """Split the test-db list into this shard's regular and isolated test lists.

    Args:
        test_db_list: Path to the test-db rendered test list file.
        working_dir: Working directory for pytest (tests/integration/defs).
        splits: Total number of shards.
        group: This shard's group number (1-based).
        durations_path: Optional path to per-cluster .test_durations file for
            least_duration splitting. Falls back to the default .test_durations
            when None.
        pytest_targets: Optional positional pytest node IDs. --test-list filters
            after collection, so these keep collection from importing unrelated
            test modules.

    Returns:
        Tuple of (regular_list_path, isolate_list_path, regular_count, isolate_count).

    Raises:
        SetupError: pytest collection failed or timed out.
    """
    cleaned_lines = []
    isolation_tests = []
    for line in Path(test_db_list).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "ISOLATION" in stripped:
            stripped = _strip_isolation_marker(stripped)
            isolation_tests.append(stripped)
        cleaned_lines.append(stripped)

    # Collection reads the list without ISOLATION markers.
    test_db_path = Path(test_db_list)
    cleaned_file = str(test_db_path.with_stem(test_db_path.stem + "_cleaned"))
    Path(cleaned_file).write_text("\n".join(cleaned_lines) + "\n" if cleaned_lines else "")
    print(f"Created cleaned testDBList: {cleaned_file} with {len(cleaned_lines)} lines")
    print(f"Original testDBList contains {len(isolation_tests)} tests with ISOLATION markers")

    regular_tests = []
    isolate_tests = []
    shard_tests = _collect_shard_tests(
        cleaned_file,
        working_dir,
        splits,
        group,
        durations_path,
        pytest_targets,
        os.path.dirname(test_db_list),
    )
    for test in shard_tests:
        trimmed = test.strip()
        if not trimmed:
            continue
        # test_unittests_v2[<test list line>] carries the line as its parameter.
        if trimmed.startswith("test_unittests.py::test_unittests_v2[") and trimmed.endswith("]"):
            trimmed = trimmed[trimmed.index("[") + 1 : trimmed.rindex("]")]

        # Every isolation test is also in cleaned_lines, so resolve the line
        # first and then check whether it needs isolation.
        line = _find_test_list_line(trimmed, cleaned_lines)
        if line is None:
            continue
        if line in isolation_tests:
            isolate_tests.append(line)
        else:
            regular_tests.append(line)

    regular_file = str(test_db_path.with_stem(test_db_path.stem + "_regular"))
    isolate_file = str(test_db_path.with_stem(test_db_path.stem + "_isolate"))
    Path(regular_file).write_text("\n".join(regular_tests) + "\n" if regular_tests else "")
    Path(isolate_file).write_text("\n".join(isolate_tests) + "\n" if isolate_tests else "")
    print(f"Created {regular_file} with {len(regular_tests)} regular tests")
    print(f"Created {isolate_file} with {len(isolate_tests)} isolate tests")

    return regular_file, isolate_file, len(regular_tests), len(isolate_tests)
