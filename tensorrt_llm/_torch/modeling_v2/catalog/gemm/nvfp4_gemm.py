# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 x NVFP4 dense GEMM in nn.Linear layout via the trtllm unified nvfp4_gemm op."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def _pad_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _swizzled_scale_numel(rows: int, k: int) -> int:
    """Bytes a 128x4 swizzled block-scale buffer holds for `rows` x `k`.

    The swizzle addresses the padded rectangle, not the real one, so the
    buffer is this size whatever `rows` and `k` are -- see nvfp4_gemm.md.
    """
    return _pad_up(rows, 128) * _pad_up(k // 16, 4)


def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    output_buffer_kind: int = 0,
    allowed_backends: str = "cutlass,cublaslt,cuda_core",
    group: list[int] | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `alpha * (act @ weight.T) (+ bias)` over block-scaled NVFP4 operands, one op call."""
    # Every guard is pure metadata, and every guarded violation was observed on
    # this machine to be read wrongly *without raising* by at least one backend
    # the default `allowed_backends` string can select: the cutlass path checks
    # contiguity and raises, cublaslt does not, and none of the three default
    # backends rejects an oversized `alpha`.
    assert act_fp4.is_contiguous() and weight.is_contiguous(), (
        "act_fp4 [M, K/2] and weight [N, K/2] must be contiguous; the cublaslt "
        "backend ignores strides and returns wrong results"
    )
    assert act_sf.is_contiguous() and weight_scale.is_contiguous(), (
        "act_sf and weight_scale must be contiguous; the cublaslt backend "
        "ignores strides and returns wrong results"
    )
    assert alpha.numel() == 1, (
        "alpha must hold exactly one element; extra elements are silently "
        "ignored (this build has no per-token alpha)"
    )
    # Contiguity above says nothing about length, and a short scale buffer is
    # read past its end rather than rejected: the kernel indexes the *padded*
    # rectangle. Both operands are checked because M and N pad independently.
    #
    # Only for 2-D operands, and only for *under*-length. Anything else about
    # the shapes -- wrong rank, zero rows, K disagreement -- the op rejects
    # itself and loudly, and pre-empting that here would swap its RuntimeError
    # for an AssertionError that says less. Over-length is not the hazard
    # either: the kernel never reads past what the swizzle addresses, and the
    # contract already says the padding bytes may hold anything.
    if act_fp4.dim() == 2 and weight.dim() == 2:
        k = act_fp4.shape[1] * 2  # act_fp4 packs two 4-bit values per byte
        floor_act_sf = _swizzled_scale_numel(act_fp4.shape[0], k)
        floor_weight_scale = _swizzled_scale_numel(weight.shape[0], k)
        assert act_sf.numel() >= floor_act_sf, (
            f"act_sf must hold at least pad_up(M,128) * pad_up(K/16,4) = "
            f"{floor_act_sf} bytes for M={act_fp4.shape[0]}, K={k}; got "
            f"{act_sf.numel()} -- the kernel would read past its end"
        )
        assert weight_scale.numel() >= floor_weight_scale, (
            f"weight_scale must hold at least pad_up(N,128) * pad_up(K/16,4) = "
            f"{floor_weight_scale} bytes for N={weight.shape[0]}, K={k}; got "
            f"{weight_scale.numel()} -- the kernel would read past its end"
        )
    return torch.ops.trtllm.nvfp4_gemm(
        act_fp4,
        weight,
        act_sf,
        weight_scale,
        alpha,
        output_dtype,
        output_buffer_kind,
        allowed_backends,
        group,
        bias,
    )
