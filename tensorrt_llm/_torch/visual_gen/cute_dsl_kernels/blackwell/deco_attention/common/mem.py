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
"""Memory access on the raw global and shared-memory addresses the kernels compute.

* gptr, sptr: typed pointers for the loads, stores, atomics and barrier operations of the CuTe DSL (cute.arch).
* load_invariant: for data that stays constant during the launch, so the compiler may merge or hoist the load.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int64, Uint8
from cutlass.cutlass_dsl import T

from .dsl import LoadCacheModifier, cp_async_shared_global


def gptr(dtype, addr, align=None):
    """Return a pointer to dtype elements at the global address addr, aligned to align bytes (default: one element)."""
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def sptr(dtype, addr, align=None):
    """Return a pointer to dtype elements at the shared address addr, aligned to align bytes (default: one element)."""
    return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align)


def load_vector(ptr, n, cop=None):
    """Return the n elements at ptr as a list, read by one vector load; cop: its cache modifier ("cg": cached at the
    global level and not in L1, for data another CTA writes)."""
    vector = cutlass.Vector(
        cute.arch.load(ptr, T.vector(n, ptr.dtype.mlir_type), cop=cop), dtype=ptr.dtype
    )
    return [vector[i] for i in range(n)]


def store_vector(ptr, values, cop=None):
    """Write the values at ptr by one vector store, as elements of ptr's type; cop: cache modifier ("cs": streaming)."""
    cute.arch.store(ptr, cutlass.Vector.from_elements(tuple(values), ptr.dtype), cop=cop)


def load_invariant(dtype, addr, n=None):
    """Return the dtype element (with n: n elements as a vector) at the global address addr of data no thread writes."""
    ptr = cutlass.inttoptr(addr, cute.AddressSpace.gmem, dtype)
    if n is None:
        return ptr.load(is_invariant=True)
    return ptr.load(alignment=n * dtype.width // 8, count=n, is_invariant=True)


def cp_async16(dst, src, nbytes=None):
    """Copy 16 bytes from the global address src to the shared address dst by cp.async, cached at the global level and
    not in L1; nbytes: the source size in bytes, the bytes beyond it are zero-filled (default: all 16 are copied)."""
    cp_async_shared_global(
        sptr(Uint8, dst), gptr(Uint8, src), 16, LoadCacheModifier.CG, cp_size=nbytes
    )


@cute.jit
def mbar_wait(addr, parity):
    """Wait until the phase of parity `parity` of the mbarrier at the shared address addr has completed: a loop on
    mbarrier.try_wait, without a limit on the tries."""
    ptr = sptr(Int64, addr)
    done = cutlass.Boolean(False)
    while not done:
        done = cute.arch.mbarrier_try_wait(ptr, parity)
