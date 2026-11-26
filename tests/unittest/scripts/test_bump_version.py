# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for scripts/bump_version.py on a copy of the files that it changes."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for path in [
        "scripts/bump_version.py",
        "scripts/generate_trtllm_serve_schemas.py",
        "tensorrt_llm/version.py",
        "examples/constraints.txt",
        "README.md",
    ]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, tmp_path / path)
    shutil.copytree(ROOT / "tensorrt_llm/schemas", tmp_path / "tensorrt_llm/schemas")
    return tmp_path


def _bump_version(repo: Path, *args: str) -> int:
    command = [sys.executable, str(repo / "scripts/bump_version.py"), *args]
    return subprocess.run(command, capture_output=True, timeout=60).returncode


def test_bump_version(repo: Path):
    version_file = repo / "tensorrt_llm/version.py"
    version_text = version_file.read_text()
    assert _bump_version(repo, ".dev2026110700") == 2
    assert _bump_version(repo, "9.9.9rc1", "--from-version", "0.0.1") == 3
    assert _bump_version(repo, "9.9.9rc1", "--dry-run") == 0
    assert version_file.read_text() == version_text
    assert _bump_version(repo, "9.9.9rc1") == 0
    assert '__version__ = "9.9.9rc1"' in version_file.read_text()
    assert _bump_version(repo, "--check") == 0
    assert _bump_version(repo, "9.9.9rc1", "--from-version", "0.0.1") == 4


def test_check_rejects_a_record_that_does_not_match(repo: Path):
    (repo / "examples/constraints.txt").write_text("tensorrt_llm==0.0.1\n")
    assert _bump_version(repo, "--check") == 1


def test_bump_skips_files_that_a_docker_wheel_build_does_not_copy(repo: Path):
    shutil.rmtree(repo / "examples")
    assert _bump_version(repo, "9.9.9rc1") == 0
