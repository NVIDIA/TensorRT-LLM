# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of how build_wheel.py installs the modeling_v2 family libraries."""

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def build_wheel() -> ModuleType:
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family(build_dir: Path, directory: str, archs: str, libraries=()) -> None:
    family_dir = build_dir / directory
    family_dir.mkdir(parents=True)
    (family_dir / "cuda_architectures.txt").write_text(archs)
    for library in libraries:
        (family_dir / library).write_bytes(b"\x7fELF")


@pytest.fixture
def dirs(tmp_path: Path):
    build_dir = tmp_path / "build" / "tensorrt_llm" / "modeling_v2"
    lib_dir = tmp_path / "tensorrt_llm" / "libs"
    build_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)
    return build_dir, lib_dir


def test_installs_only_families_with_architectures(build_wheel, dirs) -> None:
    build_dir, lib_dir = dirs
    _family(build_dir, "fixture", "103a-real", ["libth_modeling_v2_fixture.so"])
    _family(build_dir, "glmMoeDsa", "")
    (lib_dir / "libth_common.so").write_bytes(b"stock")

    build_wheel.install_modeling_v2_libs(build_dir, lib_dir, shutil.copy)

    assert sorted(p.name for p in lib_dir.iterdir()) == [
        "libth_common.so",
        "libth_modeling_v2_fixture.so",
    ]


def test_removes_a_library_this_build_did_not_produce(build_wheel, dirs) -> None:
    build_dir, lib_dir = dirs
    _family(build_dir, "fixture", "", ["libth_modeling_v2_fixture.so"])
    (lib_dir / "libth_modeling_v2_fixture.so").write_bytes(b"earlier build")
    (lib_dir / "libth_modeling_v2_removed_family.so").symlink_to(lib_dir / "gone.so")

    build_wheel.install_modeling_v2_libs(build_dir, lib_dir, shutil.copy)

    assert list(lib_dir.iterdir()) == []


def test_a_family_with_architectures_must_have_one_library(build_wheel, dirs) -> None:
    build_dir, lib_dir = dirs
    _family(build_dir, "fixture", "103a-real")

    with pytest.raises(AssertionError, match="Exactly one modeling_v2 library"):
        build_wheel.install_modeling_v2_libs(build_dir, lib_dir, shutil.copy)
