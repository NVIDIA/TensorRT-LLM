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
"""The one command that runs every serve CPU test, as the CPU CI stage runs them.

    LLM_MODELS_ROOT=... PYTHONPATH=tests python3 -m test_common.serve.cpu_tests [--list] [pytest args...]

Reads the CPU test-db list, keeps the entries whose files are serve CPU tests,
and runs one pytest process per kept entry from ``tests/`` with
``tests/unittest`` on ``PYTHONPATH`` — one process per entry, as the CPU stage
does — so process-global state does not leak between entries. Extra arguments go
to every process. Exits with the first non-zero pytest exit code, or 0.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

__all__ = ["Entry", "select_entries"]

REPO_ROOT = Path(__file__).resolve().parents[3]
CPU_LIST = REPO_ROOT / "tests" / "integration" / "test_lists" / "test-db" / "l0_cpu.yml"

_MODULE_MARKERS = (
    "tensorrt_llm.serve",
    "tensorrt_llm.commands.serve",
    "from tensorrt_llm.commands import serve",
    "test_common.serve",
)


@dataclass(frozen=True)
class Entry:
    """One kept test-db entry with the targets it runs."""

    entry: str  # the test-db line, e.g. "unittest/serve"
    targets: Tuple[str, ...]  # pytest targets relative to tests/


def _iter_test_entries(data) -> List[str]:
    """Every string test entry of a test-db document, in file order."""
    entries: List[str] = []

    def _walk(node) -> None:
        if isinstance(node, str):
            entries.append(node)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value)

    _walk(data)
    return entries


def _is_serve_cpu_test(path: Path) -> bool:
    """Whether one test file is a serve CPU test.

    The file must contain the literal ``pytest.mark.cpu_only`` (the CPU stage
    collects nothing from it otherwise) and either live under
    ``tests/unittest/serve/`` or mention a serve module or ``test_common.serve``
    in its text. The path condition catches files that name no module, such as
    checks over the test tree itself.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    if "pytest.mark.cpu_only" not in text:
        return False
    serve_dir = REPO_ROOT / "tests" / "unittest" / "serve"
    if serve_dir in path.parents:
        return True
    return any(marker in text for marker in _MODULE_MARKERS)


def select_entries(cpu_list: Path = CPU_LIST) -> Tuple[List[Entry], List[str]]:
    """(kept entries with targets, notes about skipped entries)."""
    data = yaml.safe_load(cpu_list.read_text())
    tests_root = REPO_ROOT / "tests"
    kept: List[Entry] = []
    notes: List[str] = []
    for raw_entry in _iter_test_entries(data):
        entry = raw_entry.strip()
        if not entry.startswith("unittest/"):
            continue
        target_part = entry.split("::")[0].split(" ")[0]
        if entry != target_part and "::" not in entry:
            # Entries with -k, -m, TIMEOUT or ISOLATION run with options this
            # runner does not reproduce; run them through the CI stage instead.
            notes.append(f"skipped (has options): {entry}")
            continue
        if "::" in entry and " " in entry:
            notes.append(f"skipped (has options): {entry}")
            continue
        target = entry.split("::")[0].rstrip("/")
        path = tests_root / target
        if "::" in entry:
            # A node-id entry runs only that node, never the whole file.
            if path.is_file() and _is_serve_cpu_test(path):
                kept.append(Entry(entry=raw_entry, targets=(entry,)))
            continue
        if path.is_file():
            if _is_serve_cpu_test(path):
                kept.append(Entry(entry=raw_entry, targets=(target,)))
            continue
        if path.is_dir():
            selected = tuple(
                str(candidate.relative_to(tests_root))
                for candidate in sorted(path.rglob("test_*.py"))
                if _is_serve_cpu_test(candidate)
            )
            if selected:
                kept.append(Entry(entry=raw_entry, targets=selected))
    return kept, notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m test_common.serve.cpu_tests",
        description="Run every serve CPU test, one pytest process per test-db entry.",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the kept entries and their targets"
    )
    args, pytest_args = parser.parse_known_args(argv)

    kept, notes = select_entries()
    if args.list:
        for note in notes:
            print(note)
        for entry in kept:
            print(entry.entry)
            for target in entry.targets:
                print(f"  {target}")
        return 0

    if not os.environ.get("LLM_MODELS_ROOT"):
        print(
            "LLM_MODELS_ROOT is not set. Some selected tests read model files through "
            "autouse fixtures and would error in setup; set LLM_MODELS_ROOT and rerun.",
            file=sys.stderr,
        )
        return 2

    env = dict(os.environ)
    unittest_dir = str(REPO_ROOT / "tests" / "unittest")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (unittest_dir, env.get("PYTHONPATH", "")) if part
    )
    first_failure = 0
    for note in notes:
        print(note)
    for entry in kept:
        command = [sys.executable, "-m", "pytest", "-m", "cpu_only", *entry.targets, *pytest_args]
        print(f"[{entry.entry}] {' '.join(command)}")
        result = subprocess.run(command, cwd=REPO_ROOT / "tests", env=env)
        if result.returncode != 0 and first_failure == 0:
            first_failure = result.returncode
    return first_failure


if __name__ == "__main__":
    sys.exit(main())
