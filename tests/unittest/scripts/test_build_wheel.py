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
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_wheel.py"
MSA_INTERFACE = Path("python/fmha_sm100/cute/interface.py")
MSA_PATCH = Path("3rdparty/patches/msa_strided_paged_kv.patch")

_SPEC = importlib.util.spec_from_file_location("build_wheel", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BUILD_WHEEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILD_WHEEL)
apply_msa_patch = _BUILD_WHEEL.apply_msa_patch


def _stage_project(tmp_path: Path) -> Path:
    """Copy the MSA submodule and patch into a throwaway project tree.

    The real submodule tree is left untouched; the copy drops .git so the apply
    runs against a plain working tree, as it does after a fresh checkout.
    """
    source_msa = REPO_ROOT / "3rdparty" / "MSA"
    if not (source_msa / MSA_INTERFACE).is_file():
        pytest.skip("3rdparty/MSA is not initialized")

    project_dir = tmp_path / "project"
    (project_dir / "3rdparty" / "patches").mkdir(parents=True)
    shutil.copytree(
        source_msa, project_dir / "3rdparty" / "MSA", ignore=shutil.ignore_patterns(".git")
    )
    shutil.copy(REPO_ROOT / MSA_PATCH, project_dir / MSA_PATCH)
    return project_dir


def test_apply_msa_patch_is_idempotent_in_place(tmp_path):
    project_dir = _stage_project(tmp_path)
    patched_interface = project_dir / "3rdparty" / "MSA" / MSA_INTERFACE

    apply_msa_patch(project_dir)
    assert "def _prepare_paged_hnd_input" in patched_interface.read_text()

    # A second call must short-circuit via the reverse-check guard rather than
    # raise, leaving the patched content in place.
    apply_msa_patch(project_dir)
    assert "def _prepare_paged_hnd_input" in patched_interface.read_text()


def test_apply_msa_patch_requires_initialized_submodule(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="initialize 3rdparty/MSA"):
        apply_msa_patch(project_dir)
