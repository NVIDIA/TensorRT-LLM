# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage: what a factory returns. Calling it issues captured launches only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .pdl import Footprint, PdlSpec


@runtime_checkable
class Stage(Protocol):
    name: str  # "<family>.<variant>[/<sub>]"
    keys: tuple[str, ...]  # content keys of the compiled modules it launches
    stages: tuple[
        str, ...
    ]  # Slot API stage names it covers (e.g. ("boundary_in", "qkv", "producer"))
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    pdl: PdlSpec
    footprint: Footprint
    # optional: kernels: tuple[tuple[str, PdlSpec], ...]

    def __call__(self) -> None: ...


@dataclass(frozen=True)
class StageFn:
    """The standard Stage: wraps a no-arg launch closure built in make()."""

    name: str
    fn: Callable[[], None] = field(repr=False, compare=False)
    stages: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    pdl: PdlSpec
    footprint: Footprint = Footprint()
    keys: tuple[str, ...] = ()
    kernels: tuple = ()  # optional ((sub, PdlSpec), ...) for a multi-kernel Stage; checked as name/sub

    def __call__(self) -> None:
        self.fn()


__all__ = ["Stage", "StageFn"]
