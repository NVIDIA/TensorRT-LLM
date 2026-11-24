# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the tests of the family's ``rubin_*`` GEMM entries."""

import importlib.util
import zlib
from collections.abc import Sequence

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    N_MULTIPLE,
    mxfp8_weight_scales_shape,
)

_KERNELS = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm."

requires_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="certified on sm_107",
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def needs_kernel(name: str):
    """Skip when the family kernel ``kernels/gemm/<name>.py`` is missing."""
    return pytest.mark.skipif(not _has(_KERNELS + name), reason=f"needs kernels/gemm/{name}.py")


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


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


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


def stack_mxfp8_weights(
    parts: Sequence[tuple[torch.Tensor, torch.Tensor]], n_multiple: int = N_MULTIPLE
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows of several MXFP8 weights as one ``(values [N, K], relaid scales)`` pair.

    ``parts`` are ``(values [n_i, K] E4M3, relaid scales)`` pairs in output order, the scales as
    ``mxfp8_weight_scales`` gives them. Every part but the last fills whole 128-row scale tiles,
    and R128c4 stores the tiles one after another, so the stacked scales are the parts' scales
    one after another, and part ``i`` is a view at its rows (its tiles) in both tensors. A scale
    never spans two rows, so each output column is its own projection's. Zero rows pad N up to
    ``n_multiple``; they sit in the last part's zero-padded scale tile, so their output columns
    are zero.
    """
    assert parts and 128 % n_multiple == 0, "one part or more; n_multiple divides 128"
    k = parts[0][0].shape[1]
    rows = 0
    for i, (values, scales) in enumerate(parts):
        n = values.shape[0]
        assert values.dtype == torch.float8_e4m3fn and values.shape[1] == k, "E4M3 [n_i, K] rows"
        relaid = mxfp8_weight_scales_shape(n, k)
        assert scales.dtype == torch.uint8 and tuple(scales.shape) == relaid, (
            "relaid scales: the flat uint8 buffer of mxfp8_weight_scales"
        )
        assert n % 128 == 0 or i == len(parts) - 1, "only the last part may end inside a tile"
        rows += n
    value_rows = [values.view(torch.uint8) for values, _ in parts]
    if _pad_up(rows, n_multiple) > rows:
        value_rows.append(value_rows[-1].new_zeros(_pad_up(rows, n_multiple) - rows, k))
    stacked = torch.cat(value_rows).view(torch.float8_e4m3fn)
    return stacked, torch.cat([scales for _, scales in parts])
