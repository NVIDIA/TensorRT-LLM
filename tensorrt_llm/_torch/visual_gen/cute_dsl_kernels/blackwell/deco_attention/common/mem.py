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
"""Typed pointers to the raw global and shared-memory addresses the kernels compute, for the loads, stores, atomics
and barrier operations of the CuTe DSL (cute.arch), which take a pointer; the vector load as a list and the vector
store of a sequence; the load of data that stays constant during the launch, which the compiler may merge or
hoist; and, on raw addresses, the asynchronous copy of 16 bytes and the mbarrier wait."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int64, Uint8
from cutlass.cutlass_dsl import T

from .dsl import LoadCacheModifier, cp_async_shared_global


def gptr(dtype, addr, align=None):
    """Pointer to dtype elements at the global address addr, aligned to align bytes (default: one element)."""
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def sptr(dtype, addr, align=None):
    """Pointer to dtype elements at the shared-memory address addr, aligned to align bytes (default: one element)."""
    return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align)


def load_vector(ptr, n):
    """The n elements at ptr, read by one vector load, as a list."""
    vector = cutlass.Vector(cute.arch.load(ptr, T.vector(n, ptr.dtype.mlir_type)), dtype=ptr.dtype)
    return [vector[i] for i in range(n)]


def store_vector(ptr, values, cop=None):
    """The values, as elements of the type of ptr, written at ptr by one vector store; cop: its cache modifier ("cs":
    a streaming store)."""
    cute.arch.store(ptr, cutlass.Vector.from_elements(tuple(values), ptr.dtype), cop=cop)


def load_invariant(dtype, addr, n=None):
    """The dtype element (with n: the n elements, as a vector) at the global address addr of data no thread writes."""
    ptr = cutlass.inttoptr(addr, cute.AddressSpace.gmem, dtype)
    if n is None:
        return ptr.load(is_invariant=True)
    return ptr.load(alignment=n * dtype.width // 8, count=n, is_invariant=True)


def cp_async16(dst, src, nbytes=None):
    """cp.async of 16 bytes from the global address src to the shared address dst, cached at the global level and
    not in L1; nbytes: the source size in bytes, the bytes beyond it are zero-filled (default: all 16 are copied)."""
    cp_async_shared_global(
        sptr(Uint8, dst), gptr(Uint8, src), 16, LoadCacheModifier.CG, cp_size=nbytes
    )


@cute.jit
def mbar_wait(addr, parity):
    """Waits until the phase of parity `parity` of the mbarrier at the shared address addr has completed: a loop on
    mbarrier.try_wait, without a limit on the tries."""
    ptr = sptr(Int64, addr)
    done = cutlass.Boolean(False)
    while not done:
        done = cute.arch.mbarrier_try_wait(ptr, parity)
