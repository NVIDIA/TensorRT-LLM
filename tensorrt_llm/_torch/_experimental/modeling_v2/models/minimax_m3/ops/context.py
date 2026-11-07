# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MiniMax-M3 context policy, fused-QKV runner ownership and router weight packing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.kernels.rubin.m3_qkv_fusion.kernel import M3QkvFusedRunner

NORM_MXFP8 = os.environ.get("TRTLLM_CTX_NORM_MXFP8", "1") == "1"
NORM_MXFP8_MIN_ROWS = 8192
NORM_FP4 = os.environ.get("TRTLLM_CTX_NORM_FP4", "1") == "1"
QKV_FUSED = os.environ.get("TRTLLM_CTX_QKV_FUSED", "1") == "1"
QKV_FUSED_MIN_ROWS = 1024
P5A_COMBINE_MXFP8 = os.environ.get("TRTLLM_CTX_P5A_COMBINE_MXFP8", "1") == "1"
ROUTER_TF32_FIXED = os.environ.get("TRTLLM_CTX_ROUTER_TF32_FIXED", "1") == "1"
ROUTER_BF16_SPLIT_TERMS = 3

_CTX_QKV_FUSED_RUNNERS: dict[tuple[int, int, int], M3QkvFusedRunner | None] = {}
_CTX_QKV_FUSED_YIELD_LOGGED = False


def norm_op_has_fp4() -> bool:
    """Whether the installed fused norm op takes fp4_global_scale (resolved once)."""
    has = getattr(norm_op_has_fp4, "_cached", None)
    if has is None:
        op = getattr(torch.ops.trtllm, "minimax_m3_ctx_residual_rmsnorm_mxfp8", None)
        has = op is not None and "fp4_global_scale" in str(op.default._schema)
        norm_op_has_fp4._cached = has
    return has


def split_router_weight_bf16(
    weight: torch.Tensor, terms: int = ROUTER_BF16_SPLIT_TERMS
) -> torch.Tensor:
    """Split the FP32 router weight into BF16 terms laid side by side along K.

    ``weight`` [num_experts, hidden] FP32 becomes [num_experts, terms * hidden] BF16 with
    ``weight ~= sum_t result[:, t * hidden:(t + 1) * hidden]``; each term is the BF16
    rounding of what the previous terms left over (three terms carry 24 mantissa bits). A
    BF16 activation ``y`` then gets FP32-grade logits from one tensor-core GEMM with FP32
    accumulation, ``cat([y] * terms, -1) @ result.t()``, since BF16 x BF16 products are
    exact in FP32. This is the multi-term emulation of CUTLASS's 3xTF32 GEMMs and of
    cuBLAS's BF16x9 FP32 mode, with only the weight split because the activation is
    already BF16.
    """
    if weight.dim() != 2 or weight.dtype != torch.float32:
        raise ValueError("router weight must be a 2-D FP32 tensor")
    pieces = []
    remainder = weight
    for _ in range(terms):
        piece = remainder.to(torch.bfloat16)
        pieces.append(piece)
        remainder = remainder - piece.float()
    return torch.cat(pieces, dim=1).contiguous()


def qkv_fused_takes_rows(rows: int) -> bool:
    """Whether a step of rows QKV rows is large enough for the fused QKV GEMM + producer (eager and breakable alike)."""
    return rows > QKV_FUSED_MIN_ROWS


def autotuner_tuning() -> bool:
    """Whether the AutoTuner warmup pass is running. The fused QKV producer stays out of it, so the stock MXFP8 QKV
    GEMM profiles its shape there: the steps the producer does not take (captured buckets, < 1024 rows) otherwise run
    that GEMM on the CUTLASS runner's untuned default tactic."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    global _CTX_QKV_FUSED_YIELD_LOGGED
    tuning = AutoTuner.get().is_tuning_mode
    if tuning and not _CTX_QKV_FUSED_YIELD_LOGGED:
        _CTX_QKV_FUSED_YIELD_LOGGED = True
        logger.info("[CTX R3] fused QKV yields to AutoTuner tuning pass; plain QKV GEMM tuned")
    return tuning


def qkv_fused_runner(
    num_heads_q: int, num_heads_kv: int, num_heads_index: int
) -> M3QkvFusedRunner | None:
    """The compiled fused QKV runner for one per-rank head geometry (compiled on first use, outside capture)."""
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        return None
    key = (num_heads_q, num_heads_kv, num_heads_index)
    if key not in _CTX_QKV_FUSED_RUNNERS:
        try:
            from tensorrt_llm._torch.kernels.rubin.m3_qkv_fusion.kernel import M3QkvFusedRunner

            runner = M3QkvFusedRunner(
                num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, num_heads_index=num_heads_index
            )
        except Exception as exc:  # noqa: BLE001 — initialization must leave the stock path available
            logger.warning(
                f"MiniMax-M3 fused QKV initialization failed for heads {key}: {exc}; using stock QKV"
            )
            runner = None
        _CTX_QKV_FUSED_RUNNERS[key] = runner
    return _CTX_QKV_FUSED_RUNNERS[key]
