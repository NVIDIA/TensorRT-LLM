# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused RMSNorm + NVFP4 quantize (flashinfer CuTe-DSL ``rmsnorm_fp4quant``).

Replaces the unfused pair in front of an NVFP4 GEMM whose input is the
RMSNorm of a bf16 activation (e.g. a pre-feedforward norm feeding a
gate_up projection):

    trtllm::flashinfer_rmsnorm (bf16 out, full [M, H] HBM round-trip)
    -> trtllm::fp4_quantize (reads it back, emits FP4 + swizzled scales)

with flashinfer's CuTe-DSL ``rmsnorm_fp4quant`` (SM100+), which reads the
input once and emits the packed E2M1 payload plus the 128x4-swizzled E4M3
block scales directly - eliminating the intermediate bf16 activation
write+read.

TRT-LLM's static ``input_scale`` (448*6/amax convention) is exactly
flashinfer's ``global_scale``: both the unfused ``trtllm::fp4_quantize`` and
the fused kernel store ``e4m3(gs * blockAmax / 6)`` as the block scale, and
``is_sf_swizzled_layout=True`` emits the same 128x4 layout as
``get_sf_out_offset_128x4`` with the identical padded size
(``ceil(M/128) * 512 * numKTiles``) - so the outputs are consumable as
``Fp4QuantizedTensor(fp4, sf)`` wherever the unfused op's would be.

Numerics: the fused kernel quantizes the fp32 norm result directly, without
the intermediate bf16 round the unfused chain performs when materializing
the normed tensor. Outputs are therefore near- but not byte-identical to the
unfused pair (~1.4% of payload nibbles differ by one code step at Gemma4
serving shapes); the quantization error against the fp32 norm is statistically
identical (see tests/unittest/_torch/modules/fused_ops/test_rmsnorm_fp4_quant.py).

Callers own enablement and keep the unfused pair as the fallback for
configurations this kernel does not support (non-NVFP4 consumer, LoRA,
torch.compile, flashinfer's CuTe-DSL kernels unavailable, ...); see the
pre-feedforward norm + gate_up quantize fusion in modeling_gemma4.py.
"""

from typing import Tuple

import torch

from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE

if IS_FLASHINFER_AVAILABLE:
    try:
        # None (rather than an ImportError) when flashinfer's optional
        # CuTe-DSL dependency (nvidia-cutlass-dsl) is not importable.
        from flashinfer.norm import rmsnorm_fp4quant as _rmsnorm_fp4quant
    except ImportError:  # flashinfer version without the CuTe-DSL kernels
        _rmsnorm_fp4quant = None
else:
    _rmsnorm_fp4quant = None


def rmsnorm_fp4_quant_available() -> bool:
    """Whether the flashinfer CuTe-DSL fused norm+quant kernel is usable."""
    return _rmsnorm_fp4quant is not None


def rmsnorm_fp4_quant(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + NVFP4 block-scale quantize.

    Args:
        x: [M, H] bf16 input (a row stride larger than the width is
            allowed; the innermost dim must be contiguous).
        norm_weight: [H] bf16 RMSNorm weight (plain multiplier convention,
            ``use_gemma=False``).
        eps: RMSNorm epsilon.
        global_scale: [1] fp32 tensor - the consumer Linear's static
            ``input_scale`` (448*6/amax convention).

    Returns:
        (fp4, sf): the packed E2M1 payload [M, H//2] (uint8, element 2j in
        the low nibble of byte j) and the E4M3 block scales (uint8, 1D,
        swizzled 128x4 layout padded to 128 rows) - layout-compatible with
        ``trtllm::fp4_quantize``'s outputs and consumable as
        ``Fp4QuantizedTensor(fp4, sf)``.
    """
    assert rmsnorm_fp4_quant_available()
    assert x.dim() == 2 and x.stride(-1) == 1
    assert x.dtype == torch.bfloat16
    assert global_scale.dtype == torch.float32
    m, h = x.shape
    assert h % 16 == 0 and h >= 64, "hidden size must be a multiple of 16 and >= 64"
    assert norm_weight.shape == (h,)

    if m == 0:
        fp4 = torch.empty((0, h // 2), dtype=torch.uint8, device=x.device)
        sf = torch.empty((0,), dtype=torch.uint8, device=x.device)
        return fp4, sf

    fp4, sf = _rmsnorm_fp4quant(
        x,
        norm_weight,
        global_scale=global_scale.reshape(1),
        eps=eps,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=True,
    )
    return fp4.view(torch.uint8), sf.view(torch.uint8)
