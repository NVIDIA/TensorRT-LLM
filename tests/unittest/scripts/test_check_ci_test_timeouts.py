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
"""CPU-only CLI regressions for the standalone CI timeout check."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/check_ci_test_timeouts.py"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(SCRIPT_PATH, tmp_path / "scripts/check_ci_test_timeouts.py")
    return tmp_path


def run_check(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo / "scripts/check_ci_test_timeouts.py")],
        cwd=repo.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("suffix", [".txt", ".yml", ".yaml"])
def test_rejects_explicit_defaults(repo: Path, suffix: str) -> None:
    ci_dir = repo / "tests/integration/test_lists/test-db/nested"
    ci_dir.mkdir(parents=True)
    path = ci_dir / f"l0_sample{suffix}"
    path.write_text(
        "# TIMEOUT (60)\n"
        "  - test_a.py::test_a TIMEOUT (60)\n"
        "  - test_a.py::test_b TIMEOUT(60) ISOLATION\n"
        "  - test_a.py::test_c TIMEOUT ( 060 ) # comment\n",
        encoding="utf-8",
    )
    result = run_check(repo)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        f"{path}:{line}: Remove TIMEOUT (60); 60 minutes is the default." for line in (2, 3, 4)
    ]


def test_allows_other_timeouts_and_ignores_qa_and_comments(repo: Path) -> None:
    lists_dir = repo / "tests/integration/test_lists"
    ci_dir = lists_dir / "test-db"
    ci_dir.mkdir(parents=True)
    (ci_dir / "l0_sample.yml").write_text(
        "# TIMEOUT (60)\n"
        "  - test_a.py::test_default ISOLATION # TIMEOUT (60)\n"
        + "".join(
            f"  - test_a.py::test_{minutes} TIMEOUT ({minutes})\n"
            for minutes in (30, 90, 120, 180, 600)
        ),
        encoding="utf-8",
    )
    (ci_dir / "README.md").write_text("Example: TIMEOUT (60)\n", encoding="utf-8")
    (lists_dir / "qa").mkdir()
    (lists_dir / "qa/sample.txt").write_text("test_a.py::test_qa TIMEOUT (60)\n", encoding="utf-8")
    result = run_check(repo)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_missing_ci_directory_fails(repo: Path) -> None:
    result = run_check(repo)
    assert result.returncode == 1
    assert "Missing CI test-list directory:" in result.stderr
