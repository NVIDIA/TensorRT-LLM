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
    kernel.py    the attention kernel: stage ring and the CTA body of a whole unit or a key-range piece
    mma.py       the MMA phases: QK, transposed PV and means, zero-P skipping
    softmax.py   the P pass: I-ExpCast landing into E4M3, lazy rescale (threshold of T binades), row and block sums
    epilogue.py  output rows, partials of the split tail and their merge launch
    scan.py      the per-tile flags of a non-finite V or v_mean
The kernel takes sequences of at least one key tile (128 tokens).
"""

import torch

from ...constants import HEAD_DIM, QK_ROW_BYTES, QK_SCALE_ATOMS, key_tiles
from ...inputs import Compensation, Transforms
from ...runtime import Layout
from . import launch, prep


def layout(b: int, s: int, h: int, nsm: int, rank: tuple[int, int], bases: str) -> Layout:
    """Return the buffers of one call in their order in the workspace, name -> (dtype, shape), with T key tiles.
    The operands, which the preparation writes and the attention reads, then the scratch of both (that of the
    preparation follows the rank (r_q, r_k) of the compensation terms and the bases kind):
        q4, k4                  the QK channels as E2M1 in the first bytes of a token row
        q_sf, k_sf              UE4M3 per token and 16 channels, in the scale atoms of the QK MMA
        v8                      E4M3 V minus its means of every 128 keys
        v_mean                  BF16, one row per 16 keys (a tile's mean in all 8 of its rows), divided by v8_scale
        score_scale, v8_scale   one float32 each
        v_flags                 one int32 per key tile: a NaN in v8 or a non-finite v_mean
        transform               the records of the transformations of Q and K (prep.transform_record), behind
                                the scratch
    """
    tiles = key_tiles(s)
    return {
        "q4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "k4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "q_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "k_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "v8": (torch.float8_e4m3fn, (b, s, h, HEAD_DIM)),
        "v_mean": (torch.bfloat16, (b, h, tiles * 8, HEAD_DIM)),
        "score_scale": (torch.float32, (1,)),
        "v8_scale": (torch.float32, (1,)),
        "v_flags": (torch.int32, (b, h, tiles)),
        **prep.scratch(b, s, h, nsm, rank, bases),
        **launch.scratch(b, s, h, nsm),
        **prep.transform_record(b, h, rank),
    }


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None,
    comp: Compensation,
    scale_rule: str,
    transforms: Transforms,
) -> None:
    """Run the preparation of this kernel (prep.run) on the buffers of a call, which may hold anything on entry."""
    # the four words of the basis launch must be zero when it starts; the V pass sets the flags of its tiles
    buffers["done"].zero_()
    buffers["v_flags"].zero_()
    prep.run(q, k, v, buffers, lens, comp, scale_rule, transforms)


def attention(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    lens: torch.Tensor | None,
    rescale_threshold: float,
) -> None:
    """Run the attention of this kernel: BF16 o from the buffers that prepare has filled, with the lengths it took
    and the rescale threshold of the softmax, one of constants.RESCALE_THRESHOLDS["fp8"]."""
    launch.run(buffers, o, lens, rescale_threshold)
