# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Variant: the one declaration a family writes per landed variant (python/tp4c1/families/<family>/<mod>.py)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .contracts import FAMILIES, STAGES
from .params import Params
from .pdl import Footprint, PdlSpec

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class Variant:
    family: str
    name: str  # kebab or snake, unique within the family; append-only
    stages: tuple[str, ...]  # Slot stages it covers (api.contracts.STAGES keys)
    params: type  # a @params Params subclass; defaults = landed values
    make: Callable[[Any, Params], Any]  # make(blk, p) -> Stage
    pdl: PdlSpec | Callable[[Params], PdlSpec]  # static declaration (CI checks it without a GPU)
    footprint: Footprint | Callable[[Params, int, int], Footprint] = (
        Footprint()
    )  # (p, t, tp) -> Footprint
    modules: tuple = ()  # every NvccModule / CuteModule / Asset make() may load (AOT builds them all)
    tp: tuple[int, ...] = (2,)
    arch: tuple[str, ...] = ("sm_107a",)
    debug: bool = False  # debug variants may call blk.llk.debug(); never in a production preset
    lossless: bool = True
    kernels: tuple = ()  # static ((sub, PdlSpec | p -> PdlSpec), ...) for a multi-kernel Stage

    def kernels_for(self, p: Params) -> tuple:
        return tuple((s, k(p) if callable(k) else k) for s, k in self.kernels)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Variant.family {self.family!r} not in {FAMILIES}")
        if not _NAME.match(self.name):
            raise ValueError(f"Variant.name {self.name!r} must match {_NAME.pattern}")
        if not self.stages or any(s not in STAGES for s in self.stages):
            raise ValueError(
                f"{self.id}: stages {self.stages} must be non-empty api.contracts.STAGES keys"
            )
        if not (isinstance(self.params, type) and issubclass(self.params, Params)):
            raise TypeError(f"{self.id}: params must be a Params subclass")

    @property
    def id(self) -> str:
        return f"{self.family}.{self.name}"

    def pdl_for(self, p: Params) -> PdlSpec:
        return self.pdl(p) if callable(self.pdl) else self.pdl

    def footprint_for(self, p: Params, t: int, tp: int) -> Footprint:
        return self.footprint(p, t, tp) if callable(self.footprint) else self.footprint


__all__ = ["Variant"]
