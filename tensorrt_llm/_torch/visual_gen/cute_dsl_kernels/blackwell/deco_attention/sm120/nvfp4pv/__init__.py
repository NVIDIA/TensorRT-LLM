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
"""The NVFP4-PV kernel of the sm120 family: NVFP4 QK with residual compensation and NVFP4 PV, V' = V minus its
16-key means in V's own channels.

  prep.py      host chain of the preparation (the passes of ../prep and this kernel's V pass)
  v_pass.py    V' = V minus its 16-key means in NVFP4: the V scan and V pass of the basis launch, and its own launch
  launch.py    host entry of the attention: launch plan, scratch, compile cache
  layout.py    the re-layout of K', V' and their scales into per-tile images
  kernel.py    the attention kernel
  softmax.py   the P pass: anchors, block scales and the landing of the E2M1 elements"""

import torch

from ...constants import HEAD_DIM, QK_ROW_BYTES, QK_SCALE_ATOMS
from . import launch, prep


def layout(b, s, h, nsm):
    """The buffers of one call for inputs [B, S, H, 128] on a device of nsm SMs, in their order in the workspace:
    name -> (dtype, shape), T key tiles. First the operands, which the preparation writes and the attention reads.
    q4, k4: the QK channels as E2M1 in the first bytes of a token row; q_sf, k_sf: UE4M3 per token and 16 channels, in
    the scale atoms of the QK MMA; v4: E2M1 per channel along the keys and v4_sf: UE4M3 per channel and 16 keys, zero
    past S; v_mean: BF16, one row per 16 keys, divided by v4_scale; score_scale, v4_scale: one float32 each. Then the
    scratch of the preparation (prep.scratch) and of the attention (launch.scratch)."""
    tiles = (s + 127) // 128
    return {
        "q4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "k4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "q_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "k_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "v4": (torch.uint8, (b, h, HEAD_DIM, tiles * 64)),
        "v4_sf": (torch.uint8, (b, h, HEAD_DIM, tiles * 8)),
        "v_mean": (torch.bfloat16, (b, h, tiles * 8, HEAD_DIM)),
        "score_scale": (torch.float32, (1,)),
        "v4_scale": (torch.float32, (1,)),
        **prep.scratch(b, s, h, nsm),
        **launch.scratch(b, s, h, nsm),
    }


def prepare(q, k, v, rows, buffers):
    """The preparation of this kernel on the buffers of a call (layout): the operands from BF16 q, k, v and the
    sampled rows. Every buffer may hold anything on entry. Only the two counters of the basis launch and the arrival
    counters of the attention's split units must be zero when their launches start: they are zeroed here."""
    buffers["done"].zero_()
    launch.split_counters(buffers["split"]).zero_()
    prep.run(q, k, v, rows, buffers)


def attention(buffers, o):
    """The attention of this kernel: BF16 o from the buffers of a call that prepare has filled."""
    launch.run(buffers, o)
