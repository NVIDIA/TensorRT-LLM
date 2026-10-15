# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-backed conversion of dense NVFP4 KV rows with linear block scales."""

import torch

from tensorrt_llm.quantization.utils.fp4_utils import NVFP4_SF_VEC_SIZE

_VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)


def quantize_nvfp4_kv(
    values: torch.Tensor, dequant_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode rows as E2M1 pairs and one E4M3 scale per 16 values.

    ``dequant_scale`` is the per-tensor multiplier used when reading the cache;
    the existing quantizer takes its reciprocal. Scales are linear, not GEMM
    swizzled. Each K/V head is a separate row.
    """
    packed, scales = torch.ops.trtllm.fp4_quantize(
        values.contiguous(), dequant_scale.reciprocal(), NVFP4_SF_VEC_SIZE, False, False
    )
    return packed, scales.view(*values.shape[:-1], values.shape[-1] // NVFP4_SF_VEC_SIZE)


def dequantize_nvfp4_kv(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dequant_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode directly to the requested precision, without an intermediate FP8 cast."""
    if dtype not in _VALUE_DTYPES:
        raise ValueError("NVFP4 KV conversion supports FP16, BF16 and FP8 output")
    width = packed.shape[-1] * 2
    if width % NVFP4_SF_VEC_SIZE or scales.shape != (
        *packed.shape[:-1],
        width // NVFP4_SF_VEC_SIZE,
    ):
        raise ValueError("NVFP4 KV data and linear block-scale shapes disagree")
    data = packed.contiguous().view(torch.uint8).view(-1, width // 2)
    scale_rows = scales.contiguous().view(torch.uint8).view(-1, width // NVFP4_SF_VEC_SIZE)
    if not data.shape[0]:
        return torch.empty(*packed.shape[:-1], width, device=packed.device, dtype=dtype)
    if dtype == torch.float8_e4m3fn:
        # The direct gather also supports ordinary rows with residual_dim=0.
        # It rewrites its index argument, so use a disposable compact index.
        indices = torch.arange(data.shape[0], device=data.device, dtype=torch.int32)[:, None]
        output = torch.empty((data.shape[0], 1, width), device=data.device, dtype=dtype)
        torch.ops.trtllm.nvfp4_mla_kv_cache_gather_direct(
            data, scale_rows.view(dtype), indices, output, dequant_scale, 0, data.shape[0]
        )
    else:
        # This existing 2D kernel accepts linear NVFP4 rows independently of
        # the weight/expert layout of its original caller.
        from tensorrt_llm._torch.moe.fused_moe.triton_dequant_nvfp4 import dequant_nvfp4_2d_triton

        output = dequant_nvfp4_2d_triton(
            data, scale_rows, dequant_scale, target_dtype=dtype, sf_vec_size=NVFP4_SF_VEC_SIZE
        )
    return output.view(*packed.shape[:-1], width)


def encode_high_precision_kv(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Store the uncompressed copy; FP8 uses the native saturating conversion.

    The high pool stores unscaled FP8 values. Its scale is independent of the
    NVFP4 history scale, and values never pass through the NVFP4 representation.
    """
    if dtype == torch.float8_e4m3fn and values.dtype != dtype:
        unit_scale = torch.ones(1, dtype=torch.float32, device=values.device)
        values, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            values.contiguous(), unit_scale
        )
        return values
    return values.to(dtype)
