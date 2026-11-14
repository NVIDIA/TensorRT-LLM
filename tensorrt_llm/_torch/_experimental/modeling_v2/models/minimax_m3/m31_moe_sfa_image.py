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
"""The m3_moe engines' SFA_BULK FC1 scale image (family level: every MiniMax-M3.1 target's m3_moe engine imports it).

The routed experts' FC1 (w3_w1) block scales are stored in the TRTLLM-Gen swizzled layout, read as int32 words
[g, q, kb, lane, cm] (g = a 16-atom raw group, q / kb = the stage's MMA k-blocks, lane = the copying lane, cm = the
unit's c % 4). A ring stage's scale atoms are word [kb][lane][q] for one (g, cm): 32 lanes x 16 strided 4-byte reads in
that layout. The image stores them as [g, cm, kb, lane, q], so a stage's 2 KB of atoms is one contiguous block and one
cp.async.bulk (the kernels' SFA_BULK). Ported from the low-latency reference fused engine host (host.py
:935-937).

The image is an extra copy of the FC1 scales (the raw layout stays: the TRTLLM-Gen MoE of wide / generic steps reads
it). One image per scale tensor, shared by every engine state (token count) that serves the layer.
"""

from typing import Dict, Tuple

import torch

_IMAGES: Dict[Tuple[int, int, int, str], torch.Tensor] = {}
_WORDS_PER_GROUP = 4 * 4 * 32 * 4  # [q, kb, lane, cm] int32 words of one 16-atom raw group


def fc1_scale_image(w3_w1_weight_scale: torch.Tensor) -> torch.Tensor:
    """The SFA_BULK image (int32 words, flat) of a layer's FC1 block scales; built once per tensor."""
    scale = w3_w1_weight_scale.view(torch.uint8)
    if not scale.is_contiguous() or scale.numel() % (4 * _WORDS_PER_GROUP) != 0:
        raise ValueError(
            f"m3_moe SFA_BULK: the FC1 scales must be contiguous with a multiple of {4 * _WORDS_PER_GROUP} bytes, "
            f"got {tuple(scale.shape)}"
        )
    key = (scale.data_ptr(), scale.numel(), scale.device.index or 0, str(scale.device.type))
    img = _IMAGES.get(key)
    if img is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_moe SFA_BULK: build the FC1 scale image before CUDA-graph capture"
            )
        words = scale.reshape(-1).view(torch.int32).view(-1, 4, 4, 32, 4)  # [g, q, kb, lane, cm]
        img = words.permute(0, 4, 2, 3, 1).contiguous().view(-1)  # [g, cm, kb, lane, q]
        _IMAGES[key] = img
    return img
