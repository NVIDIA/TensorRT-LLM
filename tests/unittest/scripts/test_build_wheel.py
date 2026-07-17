#!/usr/bin/env python3
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

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_wheel.py"
MSA_INTERFACE = Path("python/fmha_sm100/cute/interface.py")

_SPEC = importlib.util.spec_from_file_location("build_wheel", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BUILD_WHEEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILD_WHEEL)
stage_msa_package = _BUILD_WHEEL.stage_msa_package


def test_stage_msa_package_applies_patch_without_modifying_submodule(tmp_path):
    source_interface = REPO_ROOT / "3rdparty" / "MSA" / MSA_INTERFACE
    if not source_interface.is_file():
        pytest.skip("3rdparty/MSA is not initialized")

    source_before = source_interface.read_bytes()
    staged_package = stage_msa_package(REPO_ROOT, tmp_path)
    staged_interface = staged_package / "cute" / "interface.py"

    assert b"def _prepare_paged_hnd_input" not in source_before
    assert "def _prepare_paged_hnd_input" in staged_interface.read_text()
    assert source_interface.read_bytes() == source_before


def test_stage_msa_package_requires_initialized_submodule(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="initialize 3rdparty/MSA"):
        stage_msa_package(project_dir, tmp_path / "build")
