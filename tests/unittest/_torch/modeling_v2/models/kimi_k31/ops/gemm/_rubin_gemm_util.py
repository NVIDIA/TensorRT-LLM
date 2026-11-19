# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the tests of the family's ``rubin_*`` GEMM entries."""

import zlib

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
)

__extra_import_path__ = ["../.."]
from _k31_utils import on_sm

requires_sm107 = on_sm(107)


def generator(why: str) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(why.encode()) % 2**31)


def activation(rows: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """BF16 ``[rows, k]`` with a column of outliers every 7: the 32-value scales differ by block."""
    x = torch.randn(rows, k, generator=gen, device="cuda")
    x[:, ::7] *= 30.0
    return x.to(torch.bfloat16)


def mxfp8_weight(rows: int, k: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 values and their flat R128c4 scales of a ``[rows, k]`` weight of row norm ~1."""
    w = (torch.randn(rows, k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)
    values, scales = quantize_mxfp8_reference(w)
    return values, mxfp8_scales_to_r128c4(scales)


def mxfp8_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """A checkpoint's K32 scales ``[N, K / 32]`` -> the flat R128c4 buffer ``rubin_mxfp8_linear``
    takes.

    The bytes of ``mxfp8_scales_to_r128c4`` and of the engine's ``block_scale_interleave``, with
    zero padding. Plain torch: it runs on CPU, CUDA and meta tensors.
    """
    assert scales.dim() == 2 and scales.dtype == torch.uint8, (
        "a checkpoint's [N, K / 32] uint8 scale bytes; flat scales are relaid already"
    )
    return mxfp8_scales_to_r128c4(scales)
