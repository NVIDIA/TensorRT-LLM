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
"""The FP8-PV kernel of the tcgen05 family: NVFP4 QK with residual compensation and FP8 PV with V smoothing.

  prep.py      host chain of the preparation (the family's shared passes of ../prep and this kernel's V pass)
  v_pass.py    E4M3 V minus its means of every 128 keys
  launch.py    host entry of the attention: launch plan, scratch of the split launch, compile cache
  kernel.py    the kernel object: configuration, host-side setup of a launch, the CTA body
  load.py, mma.py, softmax.py, epilogue.py   the warp roles
  sched.py     work distribution inside a launch
  tuning.py    schedule constants per target

The attention exponentiates the scores straight into E4M3 P (I-ExpCast) with a lazy rescale threshold of T = 4
binades, takes the row sum of P from the PV MMA and adds the V means back through a BF16 MMA over the block sums of P,
once per eight key tiles."""

import torch

from ...constants import HEAD_DIM, QK_ROW_BYTES, qk_scale_atoms
from . import launch, prep


def layout(b, s, h, rank, nsm):
    """The buffers of one call for inputs [B, S, H, 128] at compensation rank `rank` on a device of nsm SMs, in their
    order in the workspace: name -> (dtype, shape). First the operands, which the preparation writes and the
    attention reads. q4, k4: rows of packed E2M1, the QK channels in the first bytes of a token row (the bytes after
    them are not written and no MMA reads them); q_sf, k_sf: their UE4M3 scale factors, one byte per token and 16
    channels in the scale atoms of the QK MMA (an atom per 64 channels); v8: E4M3 V minus its means of every 128
    keys; v_mean: one row of BF16 means per key tile; score_scale, v8_scale: one float32 each. Then the scratch of
    the preparation (prep.scratch) and of the split launch (launch.scratch)."""
    tiles = (s + 127) // 128
    atoms = qk_scale_atoms(rank)
    return {
        "q4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "k4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "q_sf": (torch.uint8, (b, h, tiles, atoms, 512)),
        "k_sf": (torch.uint8, (b, h, tiles, atoms, 512)),
        "v8": (torch.float8_e4m3fn, (b, s, h, HEAD_DIM)),
        "v_mean": (torch.bfloat16, (b, h, tiles, HEAD_DIM)),
        "score_scale": (torch.float32, (1,)),
        "v8_scale": (torch.float32, (1,)),
        **prep.scratch(b, s, h, rank),
        **launch.scratch(b, s, h, nsm),
    }


def prepare(q, k, v, rows, buffers, rank, lens=None, map_rows=False):
    """The preparation of this kernel on the buffers of a call (layout): the operands from BF16 q, k, v and the
    sampled rows. Every buffer may hold anything on entry. Only the ready flags of a split launch must be zero when
    the attention starts: they are zeroed here. lens: None, or the int32 valid prefix lengths [B], with map_rows for
    the default draw (prep.run)."""
    flags = buffers.get("split_flags")
    if flags is not None:
        flags.zero_()
    prep.run(q, k, v, rows, buffers, rank, lens, map_rows)


def attention(buffers, o, rank, lens=None):
    """The attention of this kernel: BF16 o from the buffers of a call that prepare has filled. lens: the valid prefix
    lengths that prepare took (None: every batch element is full)."""
    launch.run(buffers, o, rank, lens)
