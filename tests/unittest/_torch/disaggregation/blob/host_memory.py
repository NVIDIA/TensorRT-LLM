# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host buffers with real addresses for driver tests, guarded by canaries on both sides."""

from __future__ import annotations

import ctypes

from tensorrt_llm._torch.disaggregation.blob.store import Span

CANARY = bytes.fromhex("deadbeef") * 4


class HostBuffer:
    """``nbytes`` of host memory a driver may read and write, with a canary before and after.

    The buffer keeps its own reference to the backing storage, so the address stays valid for as
    long as the object lives.
    """

    def __init__(self, nbytes: int, fill: int = 0) -> None:
        if nbytes <= 0:
            raise ValueError("a host buffer needs a positive size")
        guard = len(CANARY)
        self._raw = ctypes.create_string_buffer(guard * 2 + nbytes)
        base = ctypes.addressof(self._raw)
        ctypes.memmove(base, CANARY, guard)
        ctypes.memmove(base + guard + nbytes, CANARY, guard)
        self.address = base + guard
        self.nbytes = nbytes
        ctypes.memset(self.address, fill, nbytes)

    def span(self, offset: int = 0, nbytes: int | None = None) -> Span:
        return Span(self.address + offset, self.nbytes - offset if nbytes is None else nbytes)

    def read(self, offset: int = 0, nbytes: int | None = None) -> bytes:
        return ctypes.string_at(
            self.address + offset, self.nbytes - offset if nbytes is None else nbytes
        )

    def write(self, data: bytes, offset: int = 0) -> None:
        if offset < 0 or offset + len(data) > self.nbytes:
            raise ValueError("write outside the buffer")
        ctypes.memmove(self.address + offset, data, len(data))

    def canaries_intact(self) -> bool:
        guard = len(CANARY)
        before = ctypes.string_at(self.address - guard, guard)
        after = ctypes.string_at(self.address + self.nbytes, guard)
        return before == CANARY and after == CANARY
