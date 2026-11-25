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
"""Per-tensor static FP8 (E4M3) quantization of a bf16 activation.

Bit-identical to ``static_quantize_e4m3_per_tensor``: both multiply by the correctly rounded
reciprocal of the scale and round to nearest even, saturating at +-448.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

_BLOCK_SIZE = 16384
_NUM_WARPS = 8


@triton.jit
def _fp8_static_quant_kernel(x_ptr, scale_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    inv_scale = tldevice.rcp_rn(tl.load(scale_ptr).to(tl.float32))
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offsets, (x * inv_scale).to(y_ptr.dtype.element_ty), mask=mask)


def fp8_static_quant(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``input`` to float8_e4m3fn with the static per-tensor ``scale``.

    Args:
        input: Contiguous bf16 tensor of any shape.
        scale: One-element fp32 tensor; the output is ``fp8(input / scale)``.

    Returns:
        A float8_e4m3fn tensor with the shape of ``input``.
    """
    assert input.dtype == torch.bfloat16 and input.is_contiguous()
    assert scale.dtype == torch.float32 and scale.numel() == 1
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    numel = input.numel()
    _fp8_static_quant_kernel[(triton.cdiv(numel, _BLOCK_SIZE),)](
        input, scale, output, numel, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS
    )
    return output
