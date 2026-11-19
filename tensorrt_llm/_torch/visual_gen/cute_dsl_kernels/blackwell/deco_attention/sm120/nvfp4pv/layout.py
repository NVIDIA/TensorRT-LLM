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
"""Tile images of the sm_120 attention kernel: the operand layout its shared-memory stages take, and the re-layout
launch that writes it.

The preparation stores K' per key, V' per channel and their scales and means in their own tensors. The attention
kernel reads a 128-key tile of all of them with one or two TMA bulk copies, in the order the warp-level MMAs take
their B fragments (ldmatrix.x4 over 16-byte rows, scale words per lane). So a first launch (relayout_kernel, one CTA
per (batch, head, key tile)) packs every tile into one image of STAGE bytes:

- K' codes, the B operand of the QK MMA (m16n8k64: n = keys, k = channels): one 16-byte row per key and 32-channel
  chunk, the rows of an array in n-tile order. Key r = 64 s + 32 h + 8 t + 2 i + e of the tile is column 2 t + e of
  n-tile 8 s + 4 h + i.
- K' scales: the three 512-byte atoms of k_sf as stored (the four UE4M3 bytes of key r and 64-channel step a at
  512 a + 16 (r % 32) + 4 (r // 32)).
- V' codes, the B operand of the PV MMA (n = channels, k = keys): one 16-byte row per channel and 32-key chunk.
- V' scales: per MMA lane (channel 32 u + 8 t + g at lane 4 g + t) one 128-bit vector of the scales of two adjacent
  32-channel groups.
- v_mean, the B operand of the means MMA (BF16 m16n8k16: n = channels, k = the sixteen 16-key blocks of a tile pair):
  only in the image of the pair's odd tile, or of an unpaired last tile (its second half zero).
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64, Uint32

from ...common.mem import gptr, load_invariant, store_vector
from ..ptx.nvfp4pv import ld_global_nc_b32_v4_pred

# tile image layout (bytes)
SK_OFF = 0  # K' codes: 6 arrays (16-byte channel chunk u) of 128 rows x 16 B, rows in n-tile order (8 j + c)
SKSF_OFF = 12288  # K' scale atoms as stored: 3 x 512 B
SV_OFF = 13824  # V' codes: 4 arrays (16-byte key chunk u) of 128 channels x 16 B
SVSF_OFF = 22016  # V' scales: two 512-byte groups of lane-contiguous 128-bit vectors
SVM_OFF = 23040  # v_mean of a tile pair (odd tiles and an unpaired last tile only): 16 rows (k = 16-key blocks of
#                  the pair) x 256 B, 16-byte chunk jc of row r stored at jc ^ (r % 8)
IMG_NOVM = 23040  # bytes of a tile image without the v_mean rows (even tiles with a partner)
STAGE = 27136  # global packed-image stride


@cute.kernel
def relayout_kernel(
    k_ptr: cute.Pointer,
    ksf_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    vsf_ptr: cute.Pointer,
    vm_ptr: cute.Pointer,
    img_ptr: cute.Pointer,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
):
    """The image of one key tile of one (batch, head): CTA = (batch, head, tile), 256 threads, every thread a few
    16-byte chunks (K' rows of keys past S are zero)."""
    nT = (S + 127) // 128
    SP2 = 64 * nT
    SP16 = 8 * nT
    tidx, _, _ = cute.arch.thread_idx()
    tile_id, _, _ = cute.arch.block_idx()
    kt = tile_id % nT
    bh = tile_id // nT
    hh = bh % H
    bb = bh // H
    img = img_ptr.toint() + Int64(bh * nT + kt) * STAGE
    # K': key r of the tile -> row pos(r) = 8 j + c of the n-tile order, 16-byte chunk u -> array u
    for i in cutlass.range_constexpr(3):
        q = tidx + 256 * i
        r = q // 6
        u = q - r * 6
        rp = r & 31
        pos = 8 * (8 * (r >> 6) + 4 * ((r >> 5) & 1) + ((rp & 7) >> 1)) + 2 * (rp >> 3) + (rp & 1)
        key = kt * 128 + r
        w = ld_global_nc_b32_v4_pred(
            k_ptr.toint() + Int64((bb * S + key) * H + hh) * 128 + 16 * u, Int32(key < S)
        )
        store_vector(gptr(Int32, img + SK_OFF + u * 2048 + pos * 16, 16), w)
    if tidx < 96:
        w = ld_global_nc_b32_v4_pred(
            ksf_ptr.toint() + Int64(bh * nT + kt) * 1536 + 16 * tidx, Int32(1)
        )
        store_vector(gptr(Int32, img + SKSF_OFF + 16 * tidx, 16), w)
    for i in cutlass.range_constexpr(2):
        q = tidx + 256 * i
        ch = q >> 2
        u = q & 3
        w = ld_global_nc_b32_v4_pred(
            v_ptr.toint() + Int64(bh * 128 + ch) * SP2 + kt * 64 + 16 * u, Int32(1)
        )
        store_vector(gptr(Int32, img + SV_OFF + u * 2048 + ch * 16, 16), w)
    if tidx < 128:
        w2 = load_invariant(Int32, vsf_ptr.toint() + Int64(bh * 128 + tidx) * SP16 + kt * 8, 2)
        # Two adjacent 32-channel groups form one 128-bit vector per MMA lane.
        u = tidx >> 5
        r = tidx & 31
        lane_perm = 4 * (r & 7) + (r >> 3)
        store_vector(
            gptr(Uint32, img + SVSF_OFF + 512 * (u >> 1) + 16 * lane_perm + 8 * (u & 1), 8), w2
        )
    # v_mean rows of the pair (kt - 1, kt) for an odd tile, (kt, zeros) for an unpaired last tile
    if ((kt & 1) == 1) | (kt == nT - 1):
        r = tidx >> 4
        jc = tidx & 15
        src_t = kt - 1 + (r >> 3)
        ok = Int32(1)
        if (kt & 1) == 0:
            src_t = kt
            ok = Int32(r < 8)
        w = ld_global_nc_b32_v4_pred(
            vm_ptr.toint() + (Int64(bh * SP16 + src_t * 8 + (r & 7)) * 128 + 8 * jc) * 2, ok
        )
        store_vector(gptr(Int32, img + SVM_OFF + r * 256 + (jc ^ (r & 7)) * 16, 16), w)
