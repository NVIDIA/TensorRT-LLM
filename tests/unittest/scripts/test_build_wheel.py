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
MSA_KERNEL = Path("python/fmha_sm100/csrc/include/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp")
MSA_JIT = Path("python/fmha_sm100/jit.py")
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


def _assert_msa_patch_applied(project_dir: Path) -> None:
    msa_dir = project_dir / "3rdparty" / "MSA"
    assert "def _prepare_paged_hnd_input" in (msa_dir / MSA_INTERFACE).read_text()
    assert "cudaGridDependencySynchronize();" in (msa_dir / MSA_KERNEL).read_text()
    patched_jit = msa_dir / MSA_JIT
    source = patched_jit.read_text()
    assert "def _compute_csrc_fingerprint" in source
    compile(source, str(patched_jit), "exec")


def test_apply_msa_patch_is_idempotent_in_place(tmp_path):
    project_dir = _stage_project(tmp_path)

    apply_msa_patch(project_dir)
    _assert_msa_patch_applied(project_dir)

    # A second call must short-circuit via the reverse-check guard rather than
    # raise, leaving the patched content in place.
    apply_msa_patch(project_dir)
    _assert_msa_patch_applied(project_dir)


def test_apply_msa_patch_with_dangling_submodule_gitlink(tmp_path):
    """Patching must work in a copied tree whose gitlink points nowhere."""
    project_dir = _stage_project(tmp_path)
    (project_dir / "3rdparty" / "MSA" / ".git").write_text(
        "gitdir: ../../.git/modules/3rdparty/MSA\n"
    )

    apply_msa_patch(project_dir)
    _assert_msa_patch_applied(project_dir)

    apply_msa_patch(project_dir)
    _assert_msa_patch_applied(project_dir)


def test_patched_msa_jit_cache_is_source_qualified(tmp_path, monkeypatch):
    project_dir = _stage_project(tmp_path)
    patched_msa = project_dir / "3rdparty" / "MSA"
    patched_jit = patched_msa / MSA_JIT
    apply_msa_patch(project_dir)

    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(tmp_path / "cache"))
    spec = importlib.util.spec_from_file_location("patched_msa_jit", patched_jit)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first_cache = module._compute_cache_base()
    source = patched_msa / "python/fmha_sm100/csrc/gmem_bounds_check.h"
    source.write_text(source.read_text() + "\n")
    second_cache = module._compute_cache_base()

    assert first_cache.parent == tmp_path / "cache"
    assert first_cache.name.startswith("source-")
    assert first_cache != second_cache


def test_apply_msa_patch_reports_conflict(tmp_path):
    """A patch that does not apply must fail, not pass as an applied one."""
    project_dir = _stage_project(tmp_path)
    (project_dir / "3rdparty" / "MSA" / MSA_INTERFACE).write_text("unrelated\n")

    with pytest.raises(RuntimeError, match="Cannot apply"):
        apply_msa_patch(project_dir)


def test_apply_msa_patch_requires_initialized_submodule(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="initialize 3rdparty/MSA"):
        apply_msa_patch(project_dir)
