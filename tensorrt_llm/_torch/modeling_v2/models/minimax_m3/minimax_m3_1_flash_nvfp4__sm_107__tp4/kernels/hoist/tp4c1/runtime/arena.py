# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arenas: one allocation each, sized from the preset footprint before vLLM profiles memory; stages carve named,
aligned views at build. Running out is an error at build, never a late OOM."""

from __future__ import annotations

from dataclasses import dataclass, field


class ArenaExhausted(MemoryError):
    pass


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class Arena:
    name: str
    nbytes: int
    device: str = "cuda"
    storage: object = None  # torch uint8 tensor, allocated by allocate()
    used: int = 0
    carves: dict = field(default_factory=dict)  # name -> (offset, nbytes)

    def allocate(self):
        import torch

        if self.storage is None:
            self.storage = torch.empty(max(self.nbytes, 1), dtype=torch.uint8, device=self.device)
        return self

    def carve(self, name: str, nbytes: int, align: int = 256):
        if nbytes < 0 or align <= 0 or align & (align - 1):
            raise ValueError(f"carve {name}: nbytes={nbytes} align={align}")
        if name in self.carves:
            off, n = self.carves[name]
            if n != nbytes:
                raise ValueError(
                    f"{self.name} carve {name!r} re-requested with {nbytes} B (had {n} B)"
                )
            return self._view(off, n)
        off = _align(self.used, align)
        if off + nbytes > self.nbytes:
            raise ArenaExhausted(
                f"{self.name} arena slack: {name!r} needs {nbytes} B at {off}, arena is "
                f"{self.nbytes} B ({self.used} B used); raise the preset footprint"
            )
        self.carves[name] = (off, nbytes)
        self.used = off + nbytes
        return self._view(off, nbytes)

    def offset(self, name: str) -> int:
        return self.carves[name][0]

    def _view(self, off: int, n: int):
        if self.storage is None:
            return None  # plan-only (CPU checks)
        return self.storage.narrow(0, off, n)

    def zero(self):
        if self.storage is not None:
            self.storage.zero_()

    def report(self) -> dict:
        return {
            "name": self.name,
            "nbytes": self.nbytes,
            "used": self.used,
            "carves": len(self.carves),
        }
