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
"""Live VisualGen-prefix check of scripts/check_serve_ci_coverage.py.

Kept in its own file under a `visual_gen/` path with its own l0_cpu.yml entry:
CBTS narrows VisualGen-only PRs to test-db entries whose text contains
`visual_gen/`, so the check must live under such a path to run on those PRs.
The synthetic tests and the other live checks are in
tests/unittest/scripts/test_serve_ci_coverage.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "check_serve_ci_coverage.py"
DATA_FILE = REPO_ROOT / "scripts" / "serve_ci_coverage.yml"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_serve_ci_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_passes_visual_gen_check(mod):
    data = mod.load_data(DATA_FILE)
    failures = mod.check_visual_gen(REPO_ROOT, data)
    assert not failures, "\n\n".join(failures)
