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
"""Tests for the public telemetry architecture allowlist."""

import subprocess
import sys
from pathlib import Path

import pytest

from tensorrt_llm.usage.architecture_allowlist import PUBLIC_HF_ARCHITECTURES

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_allowlist_is_synchronized_with_public_model_catalogs() -> None:
    """The checked-in allowlist must be regenerated when public catalogs change."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(_REPO_ROOT / "scripts" / "update_telemetry_architecture_allowlist.py"),
            "--check",
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_allowlist_entries_are_well_formed() -> None:
    """Runtime entries are valid schema-bounded identifiers."""
    assert PUBLIC_HF_ARCHITECTURES
    assert all(
        name.isidentifier() and name.strip() == name and len(name) <= 256
        for name in PUBLIC_HF_ARCHITECTURES
    )
