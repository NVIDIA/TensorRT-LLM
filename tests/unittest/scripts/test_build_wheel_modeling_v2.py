# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check of how build_wheel.py installs the modeling_v2 family libraries."""

import importlib.util
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def _family(build_dir: Path, family: str, archs: str, library: bool) -> None:
    family_dir = build_dir / family
    family_dir.mkdir(parents=True)
    (family_dir / "cuda_architectures.txt").write_text(archs)
    if library:
        (family_dir / f"libth_modeling_v2_{family}.so").write_bytes(b"\x7fELF")


def test_install_modeling_v2_libs(tmp_path: Path) -> None:
    """A family library built for some architecture is installed.

    One this build did not produce is removed.
    """
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", script)
    build_wheel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_wheel)

    build_dir, lib_dir = tmp_path / "build", tmp_path / "libs"
    lib_dir.mkdir()
    _family(build_dir, "gpt_oss", "100f-real", library=True)
    _family(build_dir, "glm_moe_dsa", "", library=True)  # left by an earlier build
    (lib_dir / "libth_common.so").write_bytes(b"stock")
    (lib_dir / "libth_modeling_v2_glm_moe_dsa.so").write_bytes(b"earlier build")
    (lib_dir / "libth_modeling_v2_removed.so").symlink_to(lib_dir / "gone.so")

    build_wheel.install_modeling_v2_libs(build_dir, lib_dir, shutil.copy)
    assert sorted(p.name for p in lib_dir.iterdir()) == [
        "libth_common.so",
        "libth_modeling_v2_gpt_oss.so",
    ]

    _family(build_dir, "qwen4_exp", "103a-real", library=False)
    with pytest.raises(AssertionError, match="Exactly one modeling_v2 library"):
        build_wheel.install_modeling_v2_libs(build_dir, lib_dir, shutil.copy)
