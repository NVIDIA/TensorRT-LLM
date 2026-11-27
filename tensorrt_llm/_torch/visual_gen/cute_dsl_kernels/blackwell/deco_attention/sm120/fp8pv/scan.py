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
"""The non-finite flags of the sm_120 FP8-PV attention: one int32 per (batch, head, key tile), set for a NaN in V or a
non-finite v_mean of the tile by the V pass that writes them (the preparation zeroes the flags first) and read by the
attention."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64, Uint32

from ...common.mem import load_invariant
from ..ptx.fp8pv import st_global_cs_b32_pred
from .layout import NTHR


def e4m3_nan(w):
    """Return a nonzero word when one of the four E4M3 bytes of w is a NaN (E4M3FN: exactly the bytes 0x7f, 0xff)."""
    # Every masked byte is <= 127, so the add has no byte carries.
    return ((w & Uint32(0x7F7F7F7F)) + Uint32(0x01010101)) & Uint32(0x80808080)


def bf16_nonfinite(w):
    """Return a nonzero word when one of the two BF16 halves of w is non-finite (exponent bits 0x7F80 all set: the
    masked half plus 0x80 sets its bit 15, without a carry between the halves)."""
    return ((w & Uint32(0x7F807F80)) + Uint32(0x00800080)) & Uint32(0x80008000)


def flag_tile(flags_p, bh, tile, T, bad):
    """Set the flag of key tile `tile` of (batch, head) bh (T tiles each) when the word bad is nonzero."""
    st_global_cs_b32_pred(flags_p + Int64(bh * T + tile) * 4, Uint32(1), Int32(bad != Uint32(0)))


@cute.jit
def head_v_nan(vflags, tidx, nT):
    """Return the OR of the flags of the head that this thread reads; a head with a flag set takes every MMA."""
    bad = Int32(0)
    for tile in cutlass.range(tidx, nT, NTHR, unroll=1):
        bad = bad | load_invariant(Int32, vflags + Int64(tile) * 4)
    return bad
