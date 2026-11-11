# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host memory with real addresses for driver tests, guarded by canaries on both sides."""

from __future__ import annotations

import ctypes
from typing import Protocol, runtime_checkable

from tensorrt_llm._torch.disaggregation.blob.store import Span

CANARY = bytes.fromhex("deadbeef") * 4
GUARD = len(CANARY)


@runtime_checkable
class Memory(Protocol):
    """What the conformance harness needs of a piece of memory a driver may read and write."""

    @property
    def address(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    def read(self, offset: int = 0, nbytes: int | None = None) -> bytes: ...

    def write(self, data: bytes, offset: int = 0) -> None: ...

    def canaries_intact(self) -> bool: ...


class MemoryWindow:
    """``nbytes`` of memory at a given address, with a canary written before and after it.

    The caller owns ``GUARD`` bytes on either side of the window. ``owner`` is whatever keeps the
    backing memory alive; the window holds a reference to it, so memory a driver may still touch
    stays allocated for as long as the window is retained.
    """

    def __init__(self, address: int, nbytes: int, fill: int = 0, *, owner: object = None) -> None:
        if nbytes <= 0:
            raise ValueError("a memory window needs a positive size")
        self._owner = owner
        self.address = address
        self.nbytes = nbytes
        ctypes.memmove(address - GUARD, CANARY, GUARD)
        ctypes.memmove(address + nbytes, CANARY, GUARD)
        ctypes.memset(address, fill, nbytes)

    def span(self, offset: int = 0, nbytes: int | None = None) -> Span:
        return Span(self.address + offset, self.nbytes - offset if nbytes is None else nbytes)

    def read(self, offset: int = 0, nbytes: int | None = None) -> bytes:
        return ctypes.string_at(
            self.address + offset, self.nbytes - offset if nbytes is None else nbytes
        )

    def write(self, data: bytes, offset: int = 0) -> None:
        if offset < 0 or offset + len(data) > self.nbytes:
            raise ValueError("write outside the window")
        ctypes.memmove(self.address + offset, data, len(data))

    def canaries_intact(self) -> bool:
        before = ctypes.string_at(self.address - GUARD, GUARD)
        after = ctypes.string_at(self.address + self.nbytes, GUARD)
        return before == CANARY and after == CANARY


class HostBuffer(MemoryWindow):
    """A freshly allocated window that owns its backing storage for as long as it lives."""

    def __init__(self, nbytes: int, fill: int = 0) -> None:
        if nbytes <= 0:
            raise ValueError("a host buffer needs a positive size")
        self._raw = ctypes.create_string_buffer(GUARD * 2 + nbytes)
        super().__init__(ctypes.addressof(self._raw) + GUARD, nbytes, fill)
