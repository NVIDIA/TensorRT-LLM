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
"""Tile images of the sm_120 attention kernel: the operands of a key tile in the layout of its shared-memory stages.

The attention kernel reads a 128-key tile of K', V' and their scales and means with one or two TMA bulk copies, in
the order the warp-level MMAs take their B fragments (ldmatrix.x4 over 16-byte rows, scale words per lane).

The preparation writes every tile of a (batch, head) into one image of STAGE bytes (the Q'/K' pass and the encode
launch its K' part, the V pass the rest):
    * K' codes, the B operand of the QK MMA (m16n8k64: n = keys, k = channels): one 16-byte row per key and 32-channel
      chunk, the rows of an array in n-tile order. Key r = 64 s + 32 h + 8 t + 2 i + e of the tile is column 2 t + e
      of n-tile 8 s + 4 h + i; the rows of the keys past S are zero.
    * K' scales: the three 512-byte atoms of the tile (the four UE4M3 bytes of key r and 64-channel step a at
      512 a + 16 (r % 32) + 4 (r // 32)).
    * V' codes, the B operand of the PV MMA (n = channels, k = keys): one 16-byte row per channel and 32-key chunk.
    * V' scales: per MMA lane (channel 32 u + 8 t + g at lane 4 g + t) one 128-bit vector of the scales of two
      adjacent 32-channel groups.
    * v_mean, the B operand of the means MMA (BF16 m16n8k16: n = channels, k = the sixteen 16-key blocks of a tile
      pair): only in the image of the pair's odd tile, or of an unpaired last tile (its second half zero).
With valid prefix lengths the tiles of a batch element end with the one that holds its last valid key.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Uint32

from ...common.mem import gptr, store_vector
from ...constants import HEAD_DIM, QK_CHANNELS, QK_SCALE_ATOMS

# tile image layout (bytes)
SK_OFF = 0  # K' codes: 6 arrays (16-byte channel chunk u) of 128 rows x 16 B, rows in n-tile order (8 j + c)
SKSF_OFF = SK_OFF + (QK_CHANNELS // 32) * 128 * 16  # K' scale atoms: 3 x 512 B
SV_OFF = (
    SKSF_OFF + QK_SCALE_ATOMS * 512
)  # V' codes: 4 arrays (16-byte key chunk u) of 128 channels x 16 B
SVSF_OFF = (
    SV_OFF + 4 * HEAD_DIM * 16
)  # V' scales: two 512-byte groups of lane-contiguous 128-bit vectors
SVM_OFF = (
    SVSF_OFF + 2 * 512
)  # v_mean of a tile pair (odd tiles and an unpaired last tile only): 16 rows (k = 16-key
#                               blocks of the pair) x 256 B, 16-byte chunk jc of row r stored at jc ^ (r % 8)
IMG_NOVM = SVM_OFF  # bytes of a tile image without the v_mean rows (even tiles with a partner)
STAGE = SVM_OFF + 16 * 2 * HEAD_DIM  # global packed-image stride


def key_tiles(s):
    return (s + 127) >> 7


def v_scale_off(ch):
    """Offset in a tile image of the 8 scale bytes of channel ch: half 8 (u & 1) of the vector of lane 4 g + t in
    group u >> 1, ch = 32 u + 8 t + g."""
    u = ch >> 5
    r = ch & 31
    return SVSF_OFF + 512 * (u >> 1) + 16 * (4 * (r & 7) + (r >> 3)) + 8 * (u & 1)


def mean_off(row, chunk):
    """Offset in a tile image of 16-byte chunk `chunk` of the v_mean row `row` of its tile pair."""
    return SVM_OFF + row * 256 + ((chunk ^ (row & 7)) << 4)


@cute.jit
def store_means(img, tt, last, off, words):
    """Store a thread's words of a v_mean row of tile tt (img: its image; off: their offset as a row of an even tile,
    mean_off) with the rows of its pair: in the image of the odd tile, or in its own for the unpaired last tile `last`,
    with zeros in the row of the missing tile."""
    odd = tt & 1
    pair = Int32(cutlass.select_((odd == 0) & (tt < last), Int32(STAGE), Int32(0)))
    dst = img + (off + odd * (8 * 256) + pair)
    store_vector(gptr(Uint32, dst, 4 * len(words)), words, cop="cs")
    if (odd == 0) & (tt == last):
        zeros = [Uint32(0)] * len(words)
        store_vector(gptr(Uint32, dst + 8 * 256, 4 * len(words)), zeros, cop="cs")
