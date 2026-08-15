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
"""Equivalence tests for build_wheel.py's sync_tree copy-back.

sync_tree replaced an rmtree+copytree copy-back of build artifacts into the
wheel. These tests pin the on-disk result of sync_tree to that of the old
path across cold populate and incremental re-sync, so the optimization can
never silently drop, stale, or corrupt a file relative to a clean recopy.

Small synthetic trees only (no build required), so this runs in well under a
second on a CPU node.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_wheel.py"


@pytest.fixture(scope="module")
def sync_tree():
    spec = importlib.util.spec_from_file_location("build_wheel", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sync_tree


def old_copy(src, dst, exclude=()):
    """Replicate the pre-sync_tree copy-back.

    Wipe dst, then copytree dereferencing symlinks. ``exclude`` mirrors the
    old shutil.ignore_patterns call.
    """
    src = Path(src).resolve()
    dst = Path(dst)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(*exclude) if exclude else None
    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


def snapshot(root):
    """Map relpath -> (kind, content-or-None, perm-bits).

    A surviving symlink is recorded as its own kind so that a deref mismatch
    shows up as a diff.
    """
    root = Path(root)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if p.is_symlink():
                out[rel] = ("dirsymlink", None, None)
            else:
                out[rel] = ("dir", None, stat.S_IMODE(p.stat().st_mode))
        for name in filenames:
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if p.is_symlink():
                out[rel] = ("filesymlink", None, None)
            else:
                out[rel] = ("file", p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
    return out


def assert_trees_equal(new_dir, old_dir):
    new_snap, old_snap = snapshot(new_dir), snapshot(old_dir)
    assert new_snap == old_snap


def build_tree(base):
    """Build a small tree covering where the two copy paths could differ.

    Nested dirs, an empty dir, an executable file, a symlink to a file, a
    symlink to a dir, and __pycache__/*.pyc that the stage path excludes.
    """
    base = Path(base)
    (base / "pkg/sub").mkdir(parents=True)
    (base / "pkg/__init__.py").write_bytes(b"x = 1\n")
    (base / "pkg/sub/mod.py").write_bytes(b"def f():\n    return 42\n")
    (base / "pkg/data.bin").write_bytes(bytes(range(256)))
    run_sh = base / "pkg/run.sh"
    run_sh.write_bytes(b"#!/bin/sh\necho hi\n")
    run_sh.chmod(0o755)
    (base / "pkg/emptydir").mkdir()
    (base / "pkg/link_to_init.py").symlink_to(base / "pkg/__init__.py")
    (base / "pkg/link_to_sub").symlink_to(base / "pkg/sub")
    (base / "pkg/__pycache__").mkdir()
    (base / "pkg/__pycache__/mod.cpython-312.pyc").write_bytes(b"\x00\x01")
    (base / "pkg/stale.pyc").write_bytes(b"\x00")


def test_cold_populate_matches_copytree(sync_tree, tmp_path):
    """Cold sync (streamed tar populate) == old clean copytree."""
    src = tmp_path / "src"
    src.mkdir()
    build_tree(src)
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    sync_tree(src, new_dir)
    old_copy(src, old_dir)
    assert_trees_equal(new_dir, old_dir)


def test_exclude_matches_ignore_patterns(sync_tree, tmp_path):
    """sync_tree(exclude=...) drops the same entries as ignore_patterns."""
    src = tmp_path / "src"
    src.mkdir()
    build_tree(src)
    exclude = ("__pycache__", "*.pyc")
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    sync_tree(src, new_dir, exclude=exclude)
    old_copy(src, old_dir, exclude=exclude)
    assert_trees_equal(new_dir, old_dir)
    # exclusion actually happened
    assert not (new_dir / "pkg/__pycache__").exists()
    assert not (new_dir / "pkg/stale.pyc").exists()


def test_incremental_converges_to_fresh_copy(sync_tree, tmp_path):
    """Incremental re-sync after mutation must match a fresh full recopy.

    Covers changed/added/deleted files and file<->dir type swaps.
    """
    src = tmp_path / "src"
    src.mkdir()
    build_tree(src)
    new_dir = tmp_path / "new"
    sync_tree(src, new_dir)  # initial populate

    # mutate the source
    (src / "pkg/sub/mod.py").write_bytes(b"def f():\n    return 99\n")  # change
    (src / "pkg/data.bin").unlink()  # delete
    (src / "pkg/added.txt").write_bytes(b"new\n")  # add
    (src / "pkg/run.sh").unlink()  # file -> dir
    (src / "pkg/run.sh").mkdir()
    (src / "pkg/run.sh/inner.txt").write_bytes(b"now a dir\n")
    (src / "pkg/emptydir").rmdir()  # dir -> file
    (src / "pkg/emptydir").write_bytes(b"now a file\n")

    sync_tree(src, new_dir)  # incremental re-sync onto existing dst
    old_dir = tmp_path / "old"
    old_copy(src, old_dir)  # ground truth
    assert_trees_equal(new_dir, old_dir)


def test_warm_noop_is_stable(sync_tree, tmp_path):
    """Re-syncing an unchanged source must not corrupt the destination."""
    src = tmp_path / "src"
    src.mkdir()
    build_tree(src)
    new_dir = tmp_path / "new"
    sync_tree(src, new_dir)
    sync_tree(src, new_dir)  # second sync, no source change
    old_dir = tmp_path / "old"
    old_copy(src, old_dir)
    assert_trees_equal(new_dir, old_dir)


def test_same_src_dst_is_noop(sync_tree, tmp_path):
    """sync_tree onto itself must not wipe the tree.

    Guards the case where source and destination resolve to the same path.
    """
    src = tmp_path / "src"
    src.mkdir()
    build_tree(src)
    before = snapshot(src)
    sync_tree(src, src)
    assert snapshot(src) == before
