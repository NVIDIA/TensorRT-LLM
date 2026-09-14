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

"""``VirtMem`` must outlive the Python handle to the allocator it borrows.

``VirtMem`` holds ``PooledPhysMemAllocator&`` in C++ and dereferences it whenever it
unmaps physical memory. The binding's ``nb::keep_alive`` is what ties the allocator's
Python lifetime to the ``VirtMem``, and the native disaggregated bounce buffer depends
on it: ``bounce/buffer.py`` builds the allocator as a local and lets it go out of scope
while keeping only the ``VirtMem``. Without that tie the allocator is freed at the end
of that constructor and every later unmap reads freed memory.

Only ``test_virt_mem_holds_a_reference_to_its_allocator`` guards the contract. Dropping
``keep_alive`` from the binding and rerunning this file leaves the end-to-end case below
passing, because the freed allocator reads back intact in a release build; the reference
count is what actually changes. Keep any new case here anchored on something observable
rather than on an unmap that happens to succeed.
"""

import gc
import sys

import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2._introspection import PooledPhysMemAllocator, VirtMem

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_CHUNK = 2 << 20


@pytest.fixture(autouse=True)
def _cuda_context() -> None:
    # PooledPhysMemAllocator drives the driver API directly, so it needs a current
    # primary context; torch only creates one lazily on first use.
    torch.cuda.init()
    torch.zeros(1, device="cuda")


def _make_virt_mem_from_temporary_allocator() -> VirtMem:
    """Build a VirtMem exactly as bounce/buffer.py does: allocator is a local."""
    allocator = PooledPhysMemAllocator(_CHUNK)
    return VirtMem(_CHUNK, allocator, init_num_phys_mem=1)


def test_virt_mem_outlives_its_allocator_handle() -> None:
    vm = _make_virt_mem_from_temporary_allocator()
    # The only Python reference to the allocator went out of scope on return, so a
    # missing keep_alive leaves the C++ reference dangling from here on.
    gc.collect()

    assert vm.address != 0
    # destroy() unmaps the backing chunk, which is the path that dereferences the
    # allocator. It must still be alive.
    vm.destroy()


def test_virt_mem_holds_a_reference_to_its_allocator() -> None:
    """The retention is asserted directly, not inferred from a surviving unmap.

    Dropping ``keep_alive`` leaves freed memory that a release build usually reads back
    without faulting, so a test that only called ``destroy()`` could keep passing while
    the contract is broken. The reference count is exact and fails the moment it goes.
    """
    unheld = PooledPhysMemAllocator(_CHUNK)
    baseline = sys.getrefcount(unheld)

    allocator = PooledPhysMemAllocator(_CHUNK)
    vm = VirtMem(_CHUNK, allocator, init_num_phys_mem=1)
    assert sys.getrefcount(allocator) == baseline + 1

    del allocator
    gc.collect()
    assert vm.address != 0
    vm.destroy()
