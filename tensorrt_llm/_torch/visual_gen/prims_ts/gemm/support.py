# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Problem checks shared by the one-shot and prepared dense GEMM paths."""

from __future__ import annotations

from typing import Optional

import torch

_FP8 = torch.float8_e4m3fn
_SUPPORTED_ARCHES = (100, 103, 107)


def nvfp4_128x4_numel(rows: int, k: int) -> int:
    """Element count of a padded 128x4 E4M3 block-scale buffer."""
    sf_columns = ((k // 16 + 3) // 4) * 4
    return ((rows + 127) // 128) * 128 * sf_columns


def _validate_nvfp4_128x4_scale(
    scale: torch.Tensor, rows: int, k: int, name: str
) -> None:
    expected = nvfp4_128x4_numel(rows, k)
    if (
        scale.ndim != 1
        or scale.dtype not in (_FP8, torch.uint8)
        or not scale.is_contiguous()
        or scale.numel() != expected
    ):
        raise ValueError(
            f"{name} must be a contiguous 1D 128x4 E4M3 scale buffer with {expected} elements"
        )


def _validate_fp32_scale(
    scale: torch.Tensor, shape: tuple[int, ...], name: str
) -> None:
    if (
        scale.dtype != torch.float32
        or tuple(scale.shape) != shape
        or not scale.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous float32 with shape {shape}")


def validate_dense_gemm(
    *,
    arch: int,
    operand_format: str,
    epilogue: str,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    m: Optional[int] = None,
    a_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    a_global_scale: Optional[float] = None,
    weight_global_scale: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    q_norm: Optional[torch.Tensor] = None,
    k_norm: Optional[torch.Tensor] = None,
    cos_sin: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    is_neox: Optional[bool] = None,
    qkv_scale: Optional[torch.Tensor] = None,
) -> None:
    """Refuse a problem this kernel cannot run, before any compilation.

    ``N`` is not required to be a multiple of 512. That width is only the
    default cluster (cluster N 2 times tile N 256). The grid rounds up to
    the tile and the epilogue predicates a short tail. ``K`` stays aligned
    to one 128-byte TMA box: 128 elements for FP8 and 256 for NVFP4.
    """
    if arch not in _SUPPORTED_ARCHES:
        raise RuntimeError(
            f"PrimsTS dense GEMM supports SM100, SM103, and SM107, got SM{arch}"
        )
    if epilogue not in ("linear", "swiglu", "qkv_qknorm_rope"):
        raise ValueError(f"unsupported epilogue {epilogue!r}")
    if operand_format == "fp8_e4m3":
        if k % 128:
            raise ValueError(
                f"FP8 GEMM requires K divisible by 128, got M={m}, N={n}, K={k}"
            )
    elif operand_format == "nvfp4_e2m1":
        if k % 256:
            raise ValueError(
                f"NVFP4 GEMM requires logical K divisible by 256, got M={m}, N={n}, K={k}"
            )
    else:
        raise ValueError(f"unsupported operand format {operand_format!r}")

    if epilogue == "swiglu" and n % 2:
        raise ValueError("SwiGLU requires even N")
    if (
        operand_format == "nvfp4_e2m1"
        and out_dtype == torch.uint8
        and epilogue != "swiglu"
    ):
        raise ValueError("NVFP4 output is implemented only for fp4_linear_swiglu")
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError("bias must be BF16 with shape [N]")

    if operand_format == "fp8_e4m3":
        if weight_scale is None or (m is not None and a_scale is None):
            raise ValueError("FP8 GEMM requires per-token and per-channel FP32 scales")
        if weight_scale is not None:
            _validate_fp32_scale(weight_scale, (n,), "weight_scale")
        if m is not None and a_scale is not None:
            _validate_fp32_scale(a_scale, (m,), "a_scale")
    else:
        if weight_scale is None:
            raise ValueError("NVFP4 GEMM requires a weight block scale")
        _validate_nvfp4_128x4_scale(weight_scale, n, k, "weight_block_scale")
        if m is not None:
            if a_scale is None:
                raise ValueError("NVFP4 GEMM requires an activation block scale")
            _validate_nvfp4_128x4_scale(a_scale, m, k, "a_block_scale")
        if not isinstance(a_global_scale, (float, int)) or not isinstance(
            weight_global_scale, (float, int)
        ):
            raise TypeError(
                "NVFP4 global scales must be host floats to remain CUDA-graph capture safe"
            )

    if qkv_scale is not None:
        if (
            qkv_scale.dtype != torch.float32
            or tuple(qkv_scale.shape) != (3,)
            or not qkv_scale.is_contiguous()
        ):
            raise ValueError(
                "qkv_scale must be a contiguous float32 tensor with shape [3]"
            )
        if n % 3 or (n // 3) % 32:
            raise ValueError("qkv_scale requires N/3 to be divisible by 32")

    if epilogue != "qkv_qknorm_rope":
        return
    if head_dim != 128 or is_neox:
        raise ValueError("the QKV kernel supports only head_dim=128 and is_neox=False")
    if operand_format == "nvfp4_e2m1" and out_dtype != torch.bfloat16:
        raise ValueError("the QKV epilogue supports BF16 output only")
    if n % (3 * head_dim):
        raise ValueError("QKV N must be divisible by 3 * head_dim")
    for tensor, name in ((q_norm, "q_norm"), (k_norm, "k_norm")):
        if (
            tensor is None
            or tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != (head_dim,)
        ):
            raise ValueError(f"{name} must be BF16 with shape [{head_dim}]")
    if m is None:
        return
    for tensor, name, dtype, shape in (
        (cos_sin, "cos_sin", torch.float32, (m, head_dim)),
        (positions, "positions", torch.int64, (m,)),
    ):
        if tensor is None or tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
