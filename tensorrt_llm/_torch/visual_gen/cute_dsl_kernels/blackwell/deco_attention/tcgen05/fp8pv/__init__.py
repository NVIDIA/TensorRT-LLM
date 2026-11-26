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

Files:
    prep.py      host chain of the preparation (the family's shared passes of ../prep and this kernel's V pass)
    v_pass.py    E4M3 V minus its means of every 128 keys
    launch.py    host entry of the attention: launch plan, scratch of the split launch, compile cache
    kernel.py    the kernel object: configuration, host-side setup of a launch, the CTA body
    load.py, mma.py, softmax.py, epilogue.py   the warp roles
    sched.py     work distribution inside a launch
    tuning.py    schedule constants per target
Attention:
    * The scores are exponentiated straight into E4M3 P (I-ExpCast), with a lazy rescale threshold of T binades, one
      of constants.RESCALE_THRESHOLDS["fp8"].
    * The row sum of P comes from the PV MMA; a BF16 MMA over the block sums of P adds the V means back, once per
      eight key tiles.
"""

import functools

import torch

from ...constants import HEAD_DIM, QK_ROW_BYTES, key_tiles, qk_channels, qk_scale_atoms
from ...inputs import Compensation
from ...runtime import Layout
from . import launch, prep


@functools.cache
def layout(b: int, s: int, h: int, rank: tuple[int, int], bases: str, nsm: int) -> Layout:
    """Return the buffers of one call in workspace order, name -> (dtype, shape): the operands, then the scratch.
    Built once per shape, rank, bases and SM count; read only."""
    tiles = key_tiles(s)
    atoms = qk_scale_atoms(qk_channels(rank))
    return {
        # operands, which the preparation writes and the attention reads. q4, k4: rows of packed E2M1, the QK channels
        # in the first bytes of a token row (the bytes after them are not written and no MMA reads them)
        "q4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "k4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        # UE4M3 scale factors: one byte per token and 16 channels, in the QK MMA's scale atoms (one per 64 channels)
        "q_sf": (torch.uint8, (b, h, tiles, atoms, 512)),
        "k_sf": (torch.uint8, (b, h, tiles, atoms, 512)),
        # E4M3 V minus its means of every 128 keys; one row of BF16 means per key tile
        "v8": (torch.float8_e4m3fn, (b, s, h, HEAD_DIM)),
        "v_mean": (torch.bfloat16, (b, h, tiles, HEAD_DIM)),
        "score_scale": (torch.float32, (1,)),
        "v8_scale": (torch.float32, (1,)),
        # scratch of the preparation and of the split launch
        **prep.scratch(b, s, h, rank, bases),
        **launch.scratch(b, s, h, nsm),
    }


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None,
    comp: Compensation,
) -> None:
    """Run the preparation of this kernel on the buffers of a call (layout); the arguments as in prep.run."""
    # every buffer may hold anything on entry; only the arrival counters of a split launch must be zero when the
    # attention starts: they are zeroed here
    flags = buffers.get("split_flags")
    if flags is not None:
        flags.zero_()
    prep.run(q, k, v, buffers, lens, comp)


def attention(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    rank: tuple[int, int],
    lens: torch.Tensor | None,
    rescale_threshold: float,
) -> None:
    """Run the attention of this kernel into BF16 o on the buffers that prepare filled; lens: those prepare took."""
    launch.run(buffers, o, qk_channels(rank), lens, rescale_threshold)


basis = prep.basis
