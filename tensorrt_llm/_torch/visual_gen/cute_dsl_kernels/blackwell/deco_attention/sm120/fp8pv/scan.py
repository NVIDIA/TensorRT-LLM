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
"""The scan launch of the sm_120 FP8-PV attention: one flag per (batch, head, key tile) for a non-finite value in V
or v_mean. The attention skips the PV and the means MMAs of key ranges whose P is zero, which is exact only for
finite V: 0 * NaN must stay NaN. A head with a flag set takes every MMA (head_v_nan). The flags are rebuilt by every
call."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64

from ...common.dsl import barrier_cta_sync
from ...common.mem import gptr, load_invariant, load_vector, sptr
from ..ptx.fp8pv import vote_any
from ..valid_len import valid_len
from .layout import KT, NTHR

SCAN_THREADS = 128


@cute.kernel
def v_tile_nan_kernel(
    v_ptr: cute.Pointer,
    vm_ptr: cute.Pointer,
    flags_ptr: cute.Pointer,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    lens_ptr,
):
    """Scan every V byte and v_mean element once, with four warps per (batch, head, key tile): one flag per tile, set
    for a NaN in V (E4M3FN: exactly the bytes 0x7f and 0xff) or a non-finite v_mean (BF16 exponent all ones).
    lens_ptr: None, or the valid prefix lengths: a tile without a valid key is not read and its flag is zero."""
    tx, _, _ = cute.arch.thread_idx()
    kt, hh, bb = cute.arch.block_idx()
    tidx, tile, head, batch = Int32(tx), Int32(kt), Int32(hh), Int32(bb)
    lane = tidx & 31
    warp = tidx >> 5
    nt = (S + KT - 1) // KT
    bad = Int32(0)
    vb = v_ptr.toint()
    live = cutlass.const_expr(lens_ptr is None) or tile * KT < valid_len(lens_ptr.toint(), batch, S)
    if live:
        for it in cutlass.range_constexpr(8):
            segment = tidx + it * SCAN_THREADS
            row = tile * KT + (segment >> 3)
            col = (segment & 7) * 16
            if row < S:
                vals = load_invariant(
                    Int32, vb + Int64((batch * S + row) * H + head) * 128 + col, 4
                )
                for w in cutlass.range_constexpr(4):
                    # Every masked byte is <= 127, so the add has no byte carries.
                    bad = bad | (((vals[w] & 0x7F7F7F7F) + 0x01010101) & Int32(-2139062144))
        # the tile's 8 v_mean rows (2 KB): one 16-byte chunk per thread; a BF16 half with exponent bits 0x7F80 is
        # non-finite, and its masked value plus 0x80 sets bit 15 (no carry between the halves)
        mrow = (batch * H + head) * (8 * nt) + tile * 8 + (tidx >> 4)
        mvals = load_invariant(Int32, vm_ptr.toint() + Int64(mrow) * 256 + (tidx & 15) * 16, 4)
        for w in cutlass.range_constexpr(4):
            bad = bad | (((mvals[w] & 0x7F807F80) + 0x00800080) & Int32(-2147450880))
    bad = vote_any(bad)
    sp = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
    sa = Int32(sp.toint())
    if lane == 0:
        cute.arch.store(sptr(Int32, sa + warp * 4), bad)
    barrier_cta_sync(Int32(0), thread_count=SCAN_THREADS)
    if tidx == 0:
        vals = load_vector(sptr(Int32, sa, 16), 4)
        bad = vals[0] | vals[1] | vals[2] | vals[3]
        cute.arch.store(
            gptr(Int32, flags_ptr.toint() + Int64((batch * H + head) * nt + tile) * 4), bad
        )


@cute.jit
def head_v_nan(vflags, tidx, nT: cutlass.Constexpr):
    """This thread's OR of the head's per-tile flags (combined across the CTA by a barrier reduction)."""
    bad = Int32(0)
    for part in cutlass.range_constexpr((nT + NTHR - 1) // NTHR):
        tile = tidx + part * NTHR
        if tile < nT:
            bad = bad | load_invariant(Int32, vflags + Int64(tile) * 4)
    return bad
