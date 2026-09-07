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

import ast
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_ROOT = _REPO_ROOT / "tests"


def _subprocess_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    module_names: set[str] = set()
    run_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    module_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for alias in node.names:
                if alias.name == "run":
                    run_names.add(alias.asname or alias.name)

    return module_names, run_names


def _is_subprocess_run(call: ast.Call, module_names: set[str], run_names: set[str]) -> bool:
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "run"
        and isinstance(function.value, ast.Name)
        and function.value.id in module_names
    ) or (isinstance(function, ast.Name) and function.id in run_names)


def _candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "grep", "-l", "subprocess", "--", "tests/**/*.py"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [_REPO_ROOT / path for path in result.stdout.splitlines()]


def _missing_timeout_lines(source: str, filename: str = "<unknown>") -> list[int]:
    tree = ast.parse(source, filename=filename)
    module_names, run_names = _subprocess_aliases(tree)

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_subprocess_run(node, module_names, run_names)
        and not any(keyword.arg == "timeout" for keyword in node.keywords)
    ]


@pytest.mark.parametrize(
    "source",
    (
        "import subprocess\nsubprocess.run(command)",
        "import subprocess as sp\nsp.run(command)",
        "from subprocess import run\nrun(command)",
        "from subprocess import run as execute\nexecute(command)",
        "import subprocess\nsubprocess.run(command, **kwargs)",
    ),
)
def test_finds_subprocess_run_without_timeout(source: str) -> None:
    assert _missing_timeout_lines(source) == [2]


@pytest.mark.parametrize(
    "source",
    (
        "import subprocess\nsubprocess.run(command, timeout=60)",
        "client.run(command)",
    ),
)
def test_ignores_safe_or_unrelated_run_calls(source: str) -> None:
    assert _missing_timeout_lines(source) == []


def test_subprocess_run_calls_have_timeouts() -> None:
    """Every subprocess.run call in the test tree must have a timeout."""
    missing_timeouts: list[str] = []

    for path in _candidate_paths():
        lines = _missing_timeout_lines(path.read_text(encoding="utf-8"), filename=str(path))
        missing_timeouts.extend(f"{path.relative_to(_TESTS_ROOT)}:{line}" for line in lines)

    assert not missing_timeouts, "subprocess.run calls without timeout:\n" + "\n".join(
        missing_timeouts
    )
