# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BuildCtx: what make() gets on `blk.llk`. The only route from a factory to compilation, arenas and package data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Loaded:
    key: str  # full sha256 content key
    name: str  # unique module name llk_<family>_<variant>_<key12>[_<role>]
    path: str  # the built .so / cubin / asset file
    module: Any = None  # the imported python module (torch_ext) or None


@dataclass(frozen=True)
class CommBuf:
    """A carve-out of the comm arena: this rank's view plus every peer's IPC-mapped pointer of the same offset."""

    local: Any  # torch uint8 tensor
    peer_ptrs: tuple[int, ...]  # indexed by TP rank (own rank included)
    offset: int
    nbytes: int


@runtime_checkable
class BuildCtx(Protocol):
    preset_id: str
    arch: str  # "sm_107a"
    tp: int
    rank: int
    conc: int
    t: int
    layer: int  # -1 for non-layer pieces (EAGLE components)
    deadline_ns: int  # runtime deadline for every Spin
    build_deadline_ns: int  # 10x, eager build/warm calls only
    tp_group: Any

    def load(self, mod) -> Loaded: ...  # NvccModule | CuteModule | Asset
    def csrc_dir(self, family: str, variant: str) -> str: ...  # installed csrc/<family>/<variant>
    def data_path(
        self, family: str, variant: str, relpath: str
    ) -> str: ...  # installed package data under csrc
    def local(
        self, name: str, nbytes: int, align: int = 256
    ) -> Any: ...  # torch uint8 view of the local arena
    def comm(self, name: str, nbytes: int, align: int = 256) -> CommBuf: ...
    def counters(self, name: str, nbytes: int) -> Any: ...  # zeroed once at load, persistent
    def debug(
        self, name: str, default: str | None = None
    ) -> str | None: ...  # None unless a debug variant


__all__ = ["BuildCtx", "Loaded", "CommBuf"]
