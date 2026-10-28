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
"""Fused (add +) RMSNorm + NVFP4 quantize using flashinfer CuTe-DSL.

The residual-less wrapper uses ``rmsnorm_fp4quant`` and emits swizzled scales
for dense NVFP4 GEMMs. The add wrapper uses ``add_rmsnorm_fp4quant`` at a
GPT-OSS layer boundary and emits row-major scales for TRTLLMGen MoE. When
supported by FlashInfer, it writes directly into the physically padded expert
input layout; older versions use a separate padding launch. The BF16 router
input retains its logical width, and the router GEMM is unchanged.

TRT-LLM's static ``input_scale`` (448*6/amax convention) is exactly
flashinfer's ``global_scale``: both the unfused ``trtllm::fp4_quantize`` and
the fused kernel store ``e4m3(gs * blockAmax / 6)`` as the block scale.
For the residual-less wrapper, ``is_sf_swizzled_layout=True`` emits the same
128x4 layout as ``get_sf_out_offset_128x4`` with the identical padded size
(``ceil(M/128) * 512 * numKTiles``) - so the outputs are consumable as
``Fp4QuantizedTensor(fp4, sf)`` wherever the unfused op's would be.

Residual-less numerics: the kernel quantizes the fp32 norm result directly, without
the intermediate bf16 round the unfused chain performs when materializing
the normed tensor. Outputs are therefore near- but not byte-identical to the
unfused pair (~1.4% of payload nibbles differ by one code step at Gemma4
serving shapes); the quantization error against the fp32 norm is statistically
identical (see tests/unittest/_torch/modules/fused_ops/test_rmsnorm_fp4_quant.py).

Callers own enablement and keep the unfused pair as the fallback for
configurations this kernel does not support (non-NVFP4 consumer, LoRA,
flashinfer's CuTe-DSL kernels unavailable, ...); see the
pre-feedforward fusion in modeling_gemma4.py and the GPT-OSS MLP boundary in
modeling_gpt_oss.py.
"""

from inspect import signature
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
    try:
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant as _add_rmsnorm_fp4quant
    except ImportError:  # flashinfer version without the add variant
        _add_rmsnorm_fp4quant = None
else:
    _rmsnorm_fp4quant = None
    _add_rmsnorm_fp4quant = None


# Detect the optional API once, not during model execution or graph capture.
_ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING = (
    _add_rmsnorm_fp4quant is not None
    and "output_hidden_size" in signature(_add_rmsnorm_fp4quant).parameters
)


def rmsnorm_fp4_quant_available() -> bool:
    """Whether the flashinfer CuTe-DSL fused norm+quant kernel is usable."""
    return _rmsnorm_fp4quant is not None


def add_rmsnorm_fp4_quant_available() -> bool:
    """Whether flashinfer's fused add+norm+quant kernel is usable."""
    return _add_rmsnorm_fp4quant is not None


@torch.library.custom_op(
    "trtllm::add_rmsnorm_fp4_quant_out", mutates_args=("residual", "fp4", "sf", "y_out")
)
def _add_rmsnorm_fp4_quant_out(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    global_scale: torch.Tensor,
    fp4: torch.Tensor,
    sf: torch.Tensor,
    y_out: torch.Tensor,
    eps: float,
    direct_padding: bool,
) -> None:
    """Opaque compiler boundary with explicit outputs and residual mutation.

    Output allocation stays in the graph so TRT-LLM's re-inplace pass can
    remove functionalization copies without changing the kernel arithmetic.
    """
    padding_kwargs = {"output_hidden_size": fp4.shape[1] * 2} if direct_padding else {}
    _add_rmsnorm_fp4quant(
        x,
        residual,
        norm_weight,
        global_scale=global_scale.reshape(1),
        eps=eps,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=False,
        y_fp4=fp4.view(torch.float4_e2m1fn_x2),
        block_scale=sf.view(torch.float8_e4m3fn),
        y_out=y_out,
        **padding_kwargs,
    )


@_add_rmsnorm_fp4_quant_out.register_fake
def _add_rmsnorm_fp4_quant_out_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    global_scale: torch.Tensor,
    fp4: torch.Tensor,
    sf: torch.Tensor,
    y_out: torch.Tensor,
    eps: float,
    direct_padding: bool,
) -> None:
    # All shapes/dtypes come from the graph's allocations; no output aliases.
    return None


def add_rmsnorm_fp4_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    global_scale: torch.Tensor,
    padded_hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual add, RMSNorm and NVFP4 quantization for a MoE edge.

    Flashinfer updates ``residual`` in place and writes the BF16 normalized
    value to ``y_out`` for the router. With direct-padding support, it writes
    the row-major FP4 payload and E4M3 scales at the physical expert width,
    including zero tails, while normalizing only the logical input width.
    Older FlashInfer versions use the compact output and a custom padding op.

    Args:
        x: BF16 activation with shape ``[num_tokens, hidden_size]``.
        residual: BF16 residual with the same shape; updated in place.
        norm_weight: BF16 RMSNorm weight with shape ``[hidden_size]``.
        eps: RMSNorm epsilon.
        global_scale: FP32 static NVFP4 input scale with shape ``[1]``.
        padded_hidden_size: Physical hidden size expected by the MoE GEMM.

    Returns:
        The padded packed FP4 activation and row-major E4M3 block scales as
        uint8 storage views, and the BF16 post-RMSNorm activation.
    """
    assert add_rmsnorm_fp4_quant_available()
    assert x.dim() == 2 and x.is_contiguous()
    assert residual.shape == x.shape and residual.is_contiguous()
    assert x.dtype == torch.bfloat16 and residual.dtype == x.dtype
    assert norm_weight.shape == (x.shape[1],) and norm_weight.dtype == x.dtype
    assert global_scale.dtype == torch.float32
    hidden_size = x.shape[1]
    assert hidden_size % 16 == 0
    assert padded_hidden_size >= hidden_size and padded_hidden_size % 16 == 0

    direct_padding = _ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING and padded_hidden_size != hidden_size
    if torch.compiler.is_compiling():
        output_width = padded_hidden_size if direct_padding else hidden_size
        fp4 = torch.empty((x.shape[0], output_width // 2), device=x.device, dtype=torch.uint8)
        sf = torch.empty((x.shape[0], output_width // 16), device=x.device, dtype=torch.uint8)
        y_out = torch.empty_like(x)
        _add_rmsnorm_fp4_quant_out(
            x, residual, norm_weight, global_scale, fp4, sf, y_out, eps, direct_padding
        )
        if padded_hidden_size != hidden_size and not direct_padding:
            fp4, sf = torch.ops.trtllm.pad_nvfp4_activation(fp4, sf, padded_hidden_size)
        return fp4, sf, y_out

    padding_kwargs = {"output_hidden_size": padded_hidden_size} if direct_padding else {}
    y_out = torch.empty_like(x)
    fp4, sf = _add_rmsnorm_fp4quant(
        x,
        residual,
        norm_weight,
        global_scale=global_scale.reshape(1),
        eps=eps,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=False,
        y_out=y_out,
        **padding_kwargs,
    )
    if padded_hidden_size != hidden_size and not direct_padding:
        fp4, sf = torch.ops.trtllm.pad_nvfp4_activation(fp4, sf, padded_hidden_size)
    # FlashInfer returns float4/float8 tensors, but the MoE consumes packed
    # uint8 storage. Reinterpret the bytes without a cast, copy, or GPU launch.
    return fp4.view(torch.uint8), sf.view(torch.uint8), y_out


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
