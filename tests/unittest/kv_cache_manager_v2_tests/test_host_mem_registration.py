# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure unit tests for how HostMem splits CUDA page-locking into chunks."""

import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.cpu_only


if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2._utils import HostMem
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import HostMem


class _FakeHostMem:
    """Stand-in exposing only what _iterate_chunks reads, so no memory is mapped or pinned."""

    _CHUNK_SIZE = HostMem._CHUNK_SIZE

    def __init__(self, address: int, size: int) -> None:
        self._address = address
        self._size = size


def _chunks(size: int, address: int = 0x1000) -> list[tuple[int, int]]:
    """Enumerate the ranges HostMem would page-lock, without allocating anything."""
    return list(HostMem._iterate_chunks(_FakeHostMem(address, size)))


class TestHostMemChunking(unittest.TestCase):
    def test_pinning_is_always_chunked(self) -> None:
        # A pool much larger than the chunk size must never be page-locked in a single
        # cuMemHostRegister call: that holds driver-global locks for the whole operation and
        # stalls the CUDA/NVML calls of every other process on the node.
        size = 5 * HostMem._CHUNK_SIZE
        chunks = _chunks(size)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(length <= HostMem._CHUNK_SIZE for _, length in chunks))

    def test_chunks_tile_the_range_exactly(self) -> None:
        # Registration and unregistration walk the same boundaries, so the chunks must be
        # contiguous, non-overlapping and cover the range exactly -- including a partial tail.
        address = 0x1000
        size = 2 * HostMem._CHUNK_SIZE + 4096
        chunks = _chunks(size, address)
        self.assertEqual(chunks[0][0], address)
        self.assertEqual(sum(length for _, length in chunks), size)
        for (addr, length), (next_addr, _) in zip(chunks, chunks[1:]):
            self.assertEqual(addr + length, next_addr)
        self.assertEqual(chunks[-1][1], 4096)

    def test_range_smaller_than_a_chunk_is_a_single_chunk(self) -> None:
        self.assertEqual(_chunks(4096, 0x1000), [(0x1000, 4096)])

    def test_empty_range_yields_no_chunks(self) -> None:
        self.assertEqual(_chunks(0), [])


if __name__ == "__main__":
    unittest.main()
