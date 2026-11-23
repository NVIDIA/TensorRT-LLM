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
"""The FP8-PV kernel of the sm120 family: NVFP4 QK with residual compensation, FP8 PV with V smoothing per 128-key tile.

Files (kernel.py describes the attention):
    prep.py      host chain of the preparation (the passes of ../prep and this kernel's V pass)
    v_pass.py    E4M3 V minus its means of every 128 keys: the V scan and V pass of the basis launch, or its own launch
    launch.py    host entry of the attention: launch plan, scratch, compile cache
    layout.py    shared-memory stages, key order and per-lane fragment addresses
    kernel.py    the attention kernel: stage ring and the CTA bodies of whole units and of key-range pieces
    mma.py       the MMA phases: QK, transposed PV and means, zero-P skipping
    softmax.py   the P pass: I-ExpCast landing into E4M3, lazy rescale (T = 4 binades), row and block sums
    epilogue.py  output rows, partials of the split tail and their merge launch
    scan.py      the scan of V and v_mean for non-finite values
    small.py     the kernel for fewer than 128 keys
"""

import torch

from ...constants import HEAD_DIM, QK_ROW_BYTES, QK_SCALE_ATOMS
from . import launch, prep


def layout(b, s, h, nsm):
    """Return the buffers of one call in their order in the workspace, name -> (dtype, shape), with T key tiles.

    The operands, which the preparation writes and the attention reads, then the scratch of both:
        q4, k4                  the QK channels as E2M1 in the first bytes of a token row
        q_sf, k_sf              UE4M3 per token and 16 channels, in the scale atoms of the QK MMA
        v8                      E4M3 V minus its means of every 128 keys
        v_mean                  BF16, one row per 16 keys (a tile's mean in all 8 of its rows), divided by v8_scale
        score_scale, v8_scale   one float32 each
    """
    tiles = (s + 127) // 128
    return {
        "q4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "k4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "q_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "k_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "v8": (torch.float8_e4m3fn, (b, s, h, HEAD_DIM)),
        "v_mean": (torch.bfloat16, (b, h, tiles * 8, HEAD_DIM)),
        "score_scale": (torch.float32, (1,)),
        "v8_scale": (torch.float32, (1,)),
        **prep.scratch(b, s, h, nsm),
        **launch.scratch(b, s, h, nsm),
    }


def prepare(q, k, v, rows, buffers, lens=None, map_rows=False):
    """Run the preparation of this kernel (prep.run) on the buffers of a call, which may hold anything on entry."""
    # only the two counters of the basis launch must be zero when it starts
    buffers["done"].zero_()
    prep.run(q, k, v, rows, buffers, lens, map_rows)


def attention(buffers, o, lens=None):
    """Run the attention of this kernel: BF16 o from the buffers that prepare has filled, with the lengths it took."""
    launch.run(buffers, o, lens)
