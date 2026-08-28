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
- apply_bolt.profile_for: ELF-basename -> profile-file mapping (.yaml preferred,
  .fdata fallback, multi-dot names like the python bindings, empty/missing).
- apply_bolt.repack_wheel: member permission bits survive the unzip/rezip round
  trip, which zipfile does NOT give us for free.
"""

from __future__ import annotations

import importlib.util
import stat
import zipfile
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


@pytest.fixture(scope="module")
def apply_bolt():
    return _load("apply_bolt")


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
    suite.write_text("workloads:\n  - name: w_enabled\n  - name: w_disabled\n    enabled: false\n")
    assert manifest.select_workloads(None, suite) == ["w_enabled"]


# ------------------------------ apply_bolt.profile_for ---------------------------
def test_profile_for_prefers_yaml_over_fdata(apply_bolt, tmp_path):
    (tmp_path / "libtensorrt_llm.yaml").write_text("x")
    (tmp_path / "libtensorrt_llm.fdata").write_text("y")
    got = apply_bolt.profile_for("libtensorrt_llm.so", tmp_path)
    assert got is not None and got.name == "libtensorrt_llm.yaml"


def test_profile_for_falls_back_to_fdata(apply_bolt, tmp_path):
    (tmp_path / "libth_common.fdata").write_text("y")
    got = apply_bolt.profile_for("libth_common.so", tmp_path)
    assert got is not None and got.name == "libth_common.fdata"


def test_profile_for_strips_only_trailing_so(apply_bolt, tmp_path):
    # Python bindings carry dots in the stem; only the final `.so` is stripped.
    stem = "bindings.cpython-312-aarch64-linux-gnu"
    (tmp_path / f"{stem}.yaml").write_text("x")
    got = apply_bolt.profile_for(f"{stem}.so", tmp_path)
    assert got is not None and got.name == f"{stem}.yaml"


def test_profile_for_missing_returns_none(apply_bolt, tmp_path):
    assert apply_bolt.profile_for("no_such_lib.so", tmp_path) is None


def test_profile_for_ignores_empty_profile(apply_bolt, tmp_path):
    (tmp_path / "lib.yaml").write_text("")  # zero-size is treated as absent
    assert apply_bolt.profile_for("lib.so", tmp_path) is None


# ------------------------------ apply_bolt.repack_wheel --------------------------
def _mode_of(zip_path: Path, member: str) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.getinfo(member).external_attr >> 16


def _build_wheel(path: Path, members: dict) -> dict:
    """Write a zip whose members carry explicit unix modes; return their ZipInfos."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, (data, mode) in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    with zipfile.ZipFile(path) as zf:
        return {i.filename: i for i in zf.infolist()}


def test_repack_wheel_preserves_member_modes(apply_bolt, tmp_path):
    src = tmp_path / "src.whl"
    infos = _build_wheel(
        src,
        {
            "pkg/runner.sh": ("#!/bin/sh\n", 0o755),
            "pkg/lib.so": ("\x7fELF-ish", 0o644),
        },
    )
    # Mimic process_wheel: extract (which drops modes), mutate, then repack.
    work = tmp_path / "work"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)
    (work / "pkg" / "lib.so").write_text("bolted payload")

    out = tmp_path / "out.whl"
    apply_bolt.repack_wheel(work, out, infos)

    # The executable keeps its exec bit; the untouched member keeps its mode.
    assert _mode_of(out, "pkg/runner.sh") & stat.S_IXUSR
    assert stat.S_IMODE(_mode_of(out, "pkg/runner.sh")) == 0o755
    assert stat.S_IMODE(_mode_of(out, "pkg/lib.so")) == 0o644
    with zipfile.ZipFile(out) as zf:
        assert zf.read("pkg/lib.so") == b"bolted payload"


def test_repack_wheel_regression_plain_write_loses_exec_bit(apply_bolt, tmp_path):
    """Guards the reason repack_wheel exists: zf.write() would drop the mode."""
    src = tmp_path / "src.whl"
    _build_wheel(src, {"pkg/runner.sh": ("#!/bin/sh\n", 0o755)})
    work = tmp_path / "work"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)
    naive = tmp_path / "naive.whl"
    with zipfile.ZipFile(naive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(work / "pkg" / "runner.sh", "pkg/runner.sh")
    assert not _mode_of(naive, "pkg/runner.sh") & stat.S_IXUSR


def test_repack_wheel_falls_back_for_new_members(apply_bolt, tmp_path):
    src = tmp_path / "src.whl"
    infos = _build_wheel(src, {"pkg/lib.so": ("x", 0o644)})
    work = tmp_path / "work"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)
    # A member created after extraction has no original ZipInfo to honor.
    (work / "pkg" / "GENERATED").write_text("new")
    out = tmp_path / "out.whl"
    apply_bolt.repack_wheel(work, out, infos)
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["pkg/GENERATED", "pkg/lib.so"]
