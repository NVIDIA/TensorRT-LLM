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
"""Tensor memory and tensor core operations in CuTe DSL operations, the experimental primitives of common/dsl.py.

tcgen05.ld and tcgen05.st of register values with their waits, tcgen05.mma and tcgen05.commit from an elected lane;
the tcgen05 operations that are inline PTX are in tcgen05/ptx.py.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32, Int32, Int64

from ..common.dsl import (
    CTAGroup,
    ReductionKind,
    Tcgen05LdStShape,
    Tcgen05MMAKind,
    Tcgen05Wait,
    make_tmem_ptr,
    tcgen05_ld,
    tcgen05_ld_red,
    tcgen05_mma,
    tcgen05_st,
    tcgen05_wait,
)
from ..common.mem import sptr


def ld_tmem_f32(addr, n):
    """Return n f32 by tcgen05.ld in the 32x32b shape: n columns of the thread's TMEM lane at the TMEM address addr."""
    values = tcgen05_ld(Tcgen05LdStShape.SHAPE_32X32B, make_tmem_ptr(addr, Float32), num=n)
    return [values[i] for i in range(n)]


def st_tmem_f32(addr, values):
    """Store a list of f32 by tcgen05.st in the 32x32b shape: that many columns of the thread's TMEM lane at addr."""
    if len(values) == 1:
        data = Float32(values[0]).bitcast(Int32)
    else:
        data = cutlass.Vector.from_elements(tuple(values), Float32)
    tcgen05_st(Tcgen05LdStShape.SHAPE_32X32B, make_tmem_ptr(addr, Float32), data)


def ld_tmem_f32_x32_wait(addr):
    """Return 32 f32 by tcgen05.ld in the 32x32b shape and the wait for the load: the values are ready on return."""
    values = tcgen05_ld(Tcgen05LdStShape.SHAPE_32X32B, make_tmem_ptr(addr, Float32), num=32)
    tcgen05_wait(Tcgen05Wait.LOAD)
    return [values[i] for i in range(32)]


def st_tmem_16x32bx2_f32_x32_wait(addr, values):
    """Store 32 f32 by tcgen05.st in the 16x32bx2 shape (second column base 32 columns on) and wait for the store."""
    data = cutlass.Vector.from_elements(tuple(values), Float32)
    tcgen05_st(Tcgen05LdStShape.SHAPE_16X32BX2, make_tmem_ptr(addr, Float32), data, offset=32)
    tcgen05_wait(Tcgen05Wait.STORE)


def ld_tmem_16x32bx2_f32_x32_absmax_wait(addr, ld_red: bool):
    """Return 32 f32 from tensor memory and the max |x| of each half of 16: (values, maximum, maximum).

    Two tcgen05.ld in the 16x32bx2 shape of 16 columns each, 16 columns apart, and one wait for both. The maxima come
    from the loads with ld_red (tcgen05.ld.red with max.abs), else from a tree of maxima of |x| after the wait.
    """
    shape = Tcgen05LdStShape.SHAPE_16X32BX2
    halves, maxima = [], []
    if ld_red:
        for half in range(2):
            data, maximum = tcgen05_ld_red(
                shape,
                make_tmem_ptr(addr + 16 * half, Float32),
                ReductionKind.MAX,
                num=16,
                abs=True,
                offset=32,
            )
            data = cutlass.Vector(data, dtype=Float32)
            halves.append([data[i] for i in range(16)])
            maxima.append(Float32(maximum))
        tcgen05_wait(Tcgen05Wait.LOAD)
    else:
        for half in range(2):
            data = tcgen05_ld(shape, make_tmem_ptr(addr + 16 * half, Float32), num=16, offset=32)
            halves.append([data[i] for i in range(16)])
        tcgen05_wait(Tcgen05Wait.LOAD)
        for values in halves:
            tree = [cute.math.abs(value) for value in values]
            while len(tree) > 1:
                half = len(tree) // 2
                tree = [cute.arch.fmax(tree[i], tree[i + half]) for i in range(half)]
            maxima.append(tree[0])
    return halves[0] + halves[1], maxima[0], maxima[1]


def tc_mma(d_addr, a, b_desc, idesc, acc, kind: str, a_tmem: bool):
    """Issue tcgen05.mma (cta_group 1, kind "f16" or "tf32") from an elected lane: D += A B if acc != 0, else D = A B.
    a: the TMEM address of A with a_tmem, its shared-memory descriptor without."""
    with cute.arch.elect_one():
        tcgen05_mma(
            Tcgen05MMAKind(kind),
            CTAGroup.CTA_1,
            make_tmem_ptr(d_addr, Float32),
            make_tmem_ptr(a, Float32) if a_tmem else Int64(a),
            Int64(b_desc),
            Int32(idesc),
            Int32(acc) != 0,
        )


def tc_commit(addr):
    """Issue tcgen05.commit from an elected lane on the mbarrier at the shared address addr: an arrival on the
    mbarrier once the tcgen05 operations that thread issued before have completed."""
    with cute.arch.elect_one():
        tcgen05.commit(sptr(Int64, addr))
