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
"""Unit tests for the BOLT OSS engine helpers (scripts/bolt).

Covers the pure logic most likely to regress silently:
- manifest.select_workloads: the manifest records the workloads ACTUALLY
  profiled (explicit list) rather than the full suite declaration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BOLT_DIR = REPO_ROOT / "scripts" / "bolt"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BOLT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return _load("manifest")


# --------------------------- manifest.select_workloads ---------------------------
def test_select_workloads_explicit_overrides_suite(manifest, tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text("workloads:\n  - name: from_suite\n")
    # The explicit list (what actually ran) must win over the suite declaration.
    assert manifest.select_workloads("a,b,c", suite) == ["a", "b", "c"]


def test_select_workloads_strips_and_drops_empty(manifest):
    assert manifest.select_workloads(" a , ,b ,", None) == ["a", "b"]


def test_select_workloads_no_arg_no_suite(manifest):
    assert manifest.select_workloads(None, None) == []


def test_select_workloads_falls_back_to_enabled_suite_entries(manifest, tmp_path):
    pytest.importorskip("yaml")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "workloads:\n" "  - name: w_enabled\n" "  - name: w_disabled\n" "    enabled: false\n"
    )
    assert manifest.select_workloads(None, suite) == ["w_enabled"]
