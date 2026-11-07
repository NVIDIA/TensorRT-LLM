# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check of how build_wheel.py installs the VisualGen model libraries."""

import importlib.util
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def test_install_visual_gen_libs(tmp_path: Path) -> None:
    """The libraries of the models in models.txt are installed; one installed for any other model is removed."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", script)
    build_wheel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_wheel)

    build_dir, lib_dir = tmp_path / "build", tmp_path / "libs"
    for model in ("minimax_h3", "removed"):  # "removed" is left by an earlier configure
        (build_dir / model).mkdir(parents=True)
        (build_dir / model / f"libth_visual_gen_{model}.so").write_bytes(b"\x7fELF")
    (build_dir / "models.txt").write_text("minimax_h3")
    lib_dir.mkdir()
    (lib_dir / "libth_common.so").write_bytes(b"stock")
    (lib_dir / "libth_visual_gen_removed.so").write_bytes(b"earlier build")

    build_wheel.install_visual_gen_libs(build_dir, lib_dir, shutil.copy)
    assert sorted(p.name for p in lib_dir.iterdir()) == [
        "libth_common.so",
        "libth_visual_gen_minimax_h3.so",
    ]
