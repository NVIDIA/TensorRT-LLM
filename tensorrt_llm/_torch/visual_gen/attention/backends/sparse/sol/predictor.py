# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stage VisualGen SOL predictor: exact-block routes and K/V proxy summaries from Q/K/V."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal

import torch

from . import kernels as _kernels  # noqa: F401  (registers trtllm::visual_gen_sol_predictor)

BLOCK_SIZE = 64
HEAD_DIM = 128
ThreshType = Literal["diag", "exact"]


@dataclass(frozen=True)
class SolPredictorOutputs:
    """Predictor tensors consumed by block-sparse attention."""

    exact_block_bits: torch.Tensor
    k_summary: torch.Tensor
    v_summary: torch.Tensor


def support_reason(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str | None:
    """Return why compact two-stage SOL cannot serve these tensors, or ``None``."""

    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return "q, k, and v must be torch tensors"
    if q.ndim != 4:
        return f"q must use compact BSHD layout; got rank {q.ndim}"
    if k.shape != q.shape or v.shape != q.shape:
        return "SOL predictor requires uniform self-attention q/k/v shapes"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "SOL predictor requires matching BF16 q/k/v"
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        return "SOL predictor requires CUDA q/k/v"
    if k.device != q.device or v.device != q.device:
        return "SOL predictor requires q/k/v on one CUDA device"
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return "SOL predictor requires contiguous BSHD q/k/v"
    if q.shape[-1] != HEAD_DIM:
        return f"SOL predictor requires head_dim={HEAD_DIM}; got {q.shape[-1]}"
    if q.shape[0] <= 0 or q.shape[1] <= 0 or q.shape[2] <= 0:
        return "SOL predictor requires positive B, S, and H"
    return None


def _runtime_scalar(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite Python real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def predict(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: object,
    sm_scale: object,
    thresh_type: ThreshType = "diag",
) -> SolPredictorOutputs:
    """Predict SOL routes and proxy summaries for compact BF16 self-attention tensors.

    One graph-visible operator pools ``q``, ``k`` and ``v`` per ``BLOCK_SIZE`` tokens, derives the
    routing threshold of every query block from the key block statistics (``thresh_type`` selects
    the diagonal or the full key covariance), and packs the exact-block decisions into
    ``exact_block_bits``. The outputs are fresh tensors: inside CUDA Graph capture they come from the
    graph pool and stay valid for replay, and under torch.compile the operator is opaque.
    """

    reason = support_reason(q, k, v)
    if reason is not None:
        raise ValueError(reason)
    if thresh_type not in ("diag", "exact"):
        raise ValueError(f"thresh_type must be 'diag' or 'exact'; got {thresh_type!r}")
    exact_block_bits, k_summary, v_summary = torch.ops.trtllm.visual_gen_sol_predictor(
        q,
        k,
        v,
        BLOCK_SIZE,
        _runtime_scalar(tau, "tau"),
        _runtime_scalar(sm_scale, "sm_scale", positive=True),
        thresh_type,
    )
    return SolPredictorOutputs(
        exact_block_bits=exact_block_bits, k_summary=k_summary, v_summary=v_summary
    )


__all__ = ["BLOCK_SIZE", "HEAD_DIM", "SolPredictorOutputs", "predict", "support_reason"]
