# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BuildContext: the implementation of api.BuildCtx handed to make() as blk.llk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..api.ctx import CommBuf, Loaded
from ..toolchain import paths
from .arena import Arena
from .debug import Debug


@dataclass
class Runtime:
    """Per-process state shared by every BuildContext of one plan."""

    preset_id: str
    arch: str
    tp: int
    rank: int
    conc: int
    deadline_ns: int
    jit: Any  # toolchain.jit.Jit, or a fake with .load(mod) in CPU tests
    local: Arena
    comm: Any  # CommArena | LoopbackComm
    counters: Arena
    tp_group: Any = None
    debug: Debug = field(default_factory=Debug)
    loaded: dict = field(default_factory=dict)  # key -> Loaded


@dataclass
class BuildContext:
    rt: Runtime
    t: int
    layer: int
    variant_id: str
    variant_debug: bool = False

    preset_id = property(lambda s: s.rt.preset_id)
    arch = property(lambda s: s.rt.arch)
    tp = property(lambda s: s.rt.tp)
    rank = property(lambda s: s.rt.rank)
    conc = property(lambda s: s.rt.conc)
    deadline_ns = property(lambda s: s.rt.deadline_ns)
    build_deadline_ns = property(lambda s: 10 * s.rt.deadline_ns)
    tp_group = property(lambda s: s.rt.tp_group)

    def _scope(self, name: str) -> str:
        return f"{self.variant_id}/L{self.layer}/T{self.t}/{name}"

    def load(self, mod) -> Loaded:
        fam_var = f"{mod.family}.{mod.variant}"
        if fam_var.split(".")[0] != self.variant_id.split(".")[0]:
            raise PermissionError(f"{self.variant_id} loads a module of another family ({fam_var})")
        ld = self.rt.jit.load(mod)
        self.rt.loaded[ld.key] = ld
        return ld

    def csrc_dir(self, family: str, variant: str) -> str:
        return paths.csrc_dir(family, variant)

    def data_path(self, family: str, variant: str, relpath: str) -> str:
        return paths.data_path(family, variant, relpath)

    def local(self, name: str, nbytes: int, align: int = 256):
        return self.rt.local.carve(self._scope(name), nbytes, align)

    def comm(self, name: str, nbytes: int, align: int = 256) -> CommBuf:
        return self.rt.comm.carve(self._scope(name), nbytes, align)

    def counters(self, name: str, nbytes: int):
        return self.rt.counters.carve(self._scope(name), nbytes, 16)

    def debug(self, name: str, default: str | None = None) -> str | None:
        return self.rt.debug.get(name, default, variant_is_debug=self.variant_debug)
