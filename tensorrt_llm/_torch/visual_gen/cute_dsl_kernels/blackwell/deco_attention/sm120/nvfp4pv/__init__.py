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
"""NVFP4-PV kernel of the sm120 family: NVFP4 QK with residual compensation, NVFP4 PV of V' in V's own channels.

Files:
    prep.py      host chain of the preparation (the passes of ../prep and this kernel's V pass)
    v_pass.py    V' = V minus its 16-key means in NVFP4: the V scan and V pass of the basis launch, or its own launch
    launch.py    host entry of the attention: launch plan, scratch, compile cache
    layout.py    the per-tile images of K', V' and their scales and means, which the preparation writes
    kernel.py    the attention kernel
    softmax.py   the P pass: anchors, block scales and the landing of the E2M1 elements
"""

import torch

from ...constants import QK_ROW_BYTES, QK_SCALE_ATOMS
from . import launch, prep
from .layout import STAGE, key_tiles


def layout(b: int, s: int, h: int, nsm: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Return the buffers of one call in workspace order, name -> (dtype, shape): the operands, then the scratch.

    q4          the QK channels of Q' as E2M1 in the first bytes of a token row
    q_sf        UE4M3 per token and 16 channels, in the scale atoms of the QK MMA
    img         per key tile the image of K', V', their scales and v_mean (layout.py); zero past S
    score_scale, v4_scale: one float32 each
    """
    tiles = key_tiles(s)
    return {
        "q4": (torch.uint8, (b, s, h, QK_ROW_BYTES)),
        "q_sf": (torch.uint8, (b, h, tiles, QK_SCALE_ATOMS, 512)),
        "img": (torch.uint8, (b * h * tiles * STAGE,)),
        "score_scale": (torch.float32, (1,)),
        "v4_scale": (torch.float32, (1,)),
        **prep.scratch(b, s, h, nsm),
        **launch.scratch(b, s, h, nsm),
    }


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None = None,
    map_rows: bool = False,
) -> None:
    """Run the preparation on the buffers of a call (layout): the operands from BF16 q, k, v and the sampled rows.
    Every buffer may hold anything on entry; the counters of the basis launch and of the attention's split units, which
    must be zero when their launches start, are zeroed here. lens, map_rows: as in prep.run."""
    buffers["done"].zero_()
    launch.split_counters(buffers["split"]).zero_()
    prep.run(q, k, v, rows, buffers, lens, map_rows)


def attention(
    buffers: dict[str, torch.Tensor], o: torch.Tensor, lens: torch.Tensor | None = None
) -> None:
    """Run the attention: BF16 o from the buffers of a call that prepare has filled; lens: those of prepare, or None."""
    launch.run(buffers, o, lens)
