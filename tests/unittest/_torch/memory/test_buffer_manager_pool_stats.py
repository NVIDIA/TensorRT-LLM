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
"""``BufferManager`` memory pool statistics.

These counters describe TensorRT-LLM's own per-device pool
(``createPrimaryDevicePool`` in ``cudaMemPool.cpp``), not the CUDA default
pool and not PyTorch's allocator. The scope is the easiest thing to get
wrong about them, so it is asserted here rather than left to a comment:
a PyTorch allocation must leave them untouched.

``memory_pool_free`` is deliberately *not* checked against the other two.
It is defined as ``reserved - used`` (``bufferManager.cpp:267``), so
``reserved == used + free`` restates that definition and would hold against
any data at all.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device and the built bindings"
)

MB = 1 << 20


@pytest.fixture
def manager():
    from tensorrt_llm.bindings.internal.runtime import BufferManager

    # The constructor takes the raw stream handle; a CudaStream object raises
    # std::bad_cast.
    return BufferManager(torch.cuda.current_stream().cuda_stream, False)


def _read(manager):
    return (
        manager.memory_pool_reserved,
        manager.memory_pool_used,
        manager.memory_pool_free,
    )


class TestReachableAndTyped:
    def test_all_three_are_integers(self, manager):
        values = _read(manager)
        assert all(isinstance(v, int) for v in values), (
            f"expected ints, got {[type(v).__name__ for v in values]}"
        )

    def test_non_negative(self, manager):
        assert all(v >= 0 for v in _read(manager))


class TestScopeIsTrtllmsOwnPool:
    """The counters' blind spot, asserted so it cannot be forgotten."""

    def test_a_torch_allocation_does_not_move_them(self, manager):
        before = _read(manager)
        buf = torch.empty(256 * MB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        try:
            assert _read(manager) == before, (
                "PyTorch allocates from its own pool; if these counters moved "
                "they would overlap torch_reserved and could not be summed "
                "with it"
            )
        finally:
            del buf


class TestFreeIsDerivedNotMeasured:
    def test_the_identity_holds_but_proves_nothing(self, manager):
        reserved, used, free = _read(manager)
        # True by construction: memoryPoolFree() returns reserved - used.
        # Asserted only to pin that definition, never as a conservation check.
        assert reserved == used + free

    def test_free_tracks_used_exactly(self, manager):
        reserved, used, free = _read(manager)
        assert free == reserved - used
