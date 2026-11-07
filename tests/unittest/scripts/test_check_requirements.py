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
"""CPU-only checks for scripts/check_requirements.py."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]


def _check(tmp_path: Path, text: str) -> tuple[int, list[str]]:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(text)
    command = [sys.executable, str(ROOT / "scripts/check_requirements.py"), str(requirements)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    reported = [
        line.split(":")[1]
        for line in result.stdout.splitlines()
        if line.startswith(f"{requirements}:")
    ]
    return result.returncode, reported


def test_check_requirements_accepts_what_setup_py_reads(tmp_path: Path):
    assert _check(
        tmp_path,
        "# Comment\n"
        "\n"
        "-i https://pypi.org/simple\n"
        "--extra-index-url https://download.pytorch.org/whl/cu132\n"
        "-r requirements-dev.txt\n"
        "-c constraints.txt\n"
        "torch>=2.10.0,<2.13.0  # https://pytorch.org\n",
    ) == (0, [])


def test_check_requirements_reports_lines_that_setup_py_would_break(tmp_path: Path):
    assert _check(
        tmp_path,
        "torch>=2.10.0,<2.13.0\n"
        "etcd3 @ git+https://github.com/kragniz/python-etcd3.git@e58a899\n"
        "git+https://github.com/flashinfer-ai/flashinfer.git@e3853dd#egg=flashinfer-python\n"
        "-ihttps://pypi.org/simple\n"
        "--extra-index-url=https://pypi.org/simple\n",
    ) == (1, ["2", "3", "4", "5"])
