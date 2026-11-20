# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a variant compiles or loads. Every path is relative to the package (csrc/ or python/tp4c1/), never absolute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class NvccModule:
    """CUDA C++ compiled by nvcc. `sources`/`include_dirs` are relpaths under csrc/. kind=torch_ext builds a torch
    extension (pybind or TORCH_LIBRARY); kind=cubin builds a cubin loaded with cuModuleLoad."""

    family: str
    variant: str
    name: str  # module role inside the variant, e.g. "front", "ext"
    sources: tuple[str, ...]
    include_dirs: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()  # nvcc flags, -gencode included explicitly
    cflags: tuple[str, ...] = ()  # host compiler flags (torch_ext)
    defines: tuple[tuple[str, str], ...] = ()
    ldflags: tuple[str, ...] = ()
    kind: Literal["torch_ext", "cubin"] = "torch_ext"


@dataclass(frozen=True)
class CuteModule:
    """A CuTe DSL kernel package: every .py under `package_dir` (relpath under python/tp4c1/) is keyed."""

    family: str
    variant: str
    name: str
    package_dir: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Asset:
    """A prebuilt file shipped as package data (trtllm-gen cubins, engine ports): relpath under csrc/."""

    family: str
    variant: str
    relpath: str
    sha256: str  # pinned at import; a mismatch is an error


__all__ = ["NvccModule", "CuteModule", "Asset"]
