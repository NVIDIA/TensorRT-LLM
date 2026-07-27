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
"""MXFP8 (OCP microscaling FP8) reference quant/dequant helpers.

Weight format: e4m3 elements + one UE8M0 (uint8 biased exponent) scale per
`block_size` (=32) contiguous elements along the K (last) dim.
These are the slow PyTorch reference paths used as the numeric oracle for the
CUTLASS kernels; they are NOT the production path.
"""

import torch

UE8M0_BIAS = 127


def dequant_mxfp8_weight(
    w_e4m3: torch.Tensor, scale_ue8m0: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """[O, K] e4m3 + [O, K//block] uint8 -> [O, K] float32."""
    assert w_e4m3.shape[-1] % block_size == 0
    o, k = w_e4m3.shape
    w = w_e4m3.float().view(o, k // block_size, block_size)
    exp = scale_ue8m0.to(torch.int32) - UE8M0_BIAS  # [O, K//block]
    scale = torch.exp2(exp.float()).unsqueeze(-1)  # [O, K//block, 1]
    return (w * scale).view(o, k)


def quant_bf16_to_mxfp8(w: torch.Tensor, block_size: int = 32):
    """[O, K] bf16/fp32 -> (e4m3 [O,K], ue8m0 uint8 [O, K//block]).

    Reference path: per block, pick the power-of-two scale so the block max
    maps near e4m3 max (448). Used to synthesize test inputs and as a fallback
    on-the-fly quantizer; producers ship pre-quantized scales in the ckpt.
    """
    assert w.shape[-1] % block_size == 0
    o, k = w.shape
    wb = w.float().view(o, k // block_size, block_size)
    amax = wb.abs().amax(dim=-1).clamp_min(1e-12)  # [O, K//block]
    E4M3_MAX = 448.0
    # exponent e such that amax / 2^e <= E4M3_MAX, i.e. 2^e >= amax / E4M3_MAX
    exp = torch.ceil(torch.log2(amax / E4M3_MAX))
    scale_ue8m0 = (exp + UE8M0_BIAS).clamp(0, 255).to(torch.uint8)
    scale = torch.exp2(exp).unsqueeze(-1)
    w_e4m3 = (wb / scale).to(torch.float8_e4m3fn).view(o, k)
    return w_e4m3, scale_ue8m0.view(o, k // block_size)
