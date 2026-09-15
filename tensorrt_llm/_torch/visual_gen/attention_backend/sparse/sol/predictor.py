# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-owned runtime for the two-stage VisualGen SOL predictor."""

from __future__ import annotations

import numbers
import struct
from dataclasses import dataclass

import torch

from . import kernels as _kernels

BLOCK_SIZE = 64
HEAD_DIM = 128


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _float32_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite Python real")
    try:
        result = struct.unpack("=f", struct.pack("=f", float(value)))[0]
    except (OverflowError, TypeError, ValueError, struct.error) as error:
        raise ValueError(f"{name} must be representable as float32") from error
    if not -float("inf") < result < float("inf"):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_runtime_scalars(*, tau: object, sm_scale: object) -> tuple[float, float]:
    """Validate and round the two dynamic selector scalars to binary32."""

    effective_tau = _float32_scalar(tau, "tau")
    effective_sm_scale = _float32_scalar(sm_scale, "sm_scale")
    if effective_sm_scale <= 0.0:
        raise ValueError("sm_scale must be positive")
    return effective_tau, effective_sm_scale


@dataclass(frozen=True)
class SolPredictorGeometry:
    """Static shape specialization for compact BF16 self-MHA."""

    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    num_q_blocks: int
    num_kv_blocks: int
    exact_words: int
    tail_tokens: int

    @classmethod
    def create(
        cls,
        *,
        batch_size: object,
        seq_len: object,
        num_heads: object,
        head_dim: object = HEAD_DIM,
    ) -> "SolPredictorGeometry":
        batch = _positive_int(batch_size, "batch_size")
        tokens = _positive_int(seq_len, "seq_len")
        heads = _positive_int(num_heads, "num_heads")
        dim = _positive_int(head_dim, "head_dim")
        if dim != HEAD_DIM:
            raise ValueError(f"SOL predictor only supports head_dim={HEAD_DIM}; got {dim}")
        blocks = _kernels.num_blocks(tokens, BLOCK_SIZE)
        tail = tokens - (blocks - 1) * BLOCK_SIZE
        return cls(
            batch_size=batch,
            seq_len=tokens,
            num_heads=heads,
            head_dim=dim,
            num_q_blocks=blocks,
            num_kv_blocks=blocks,
            exact_words=_kernels.num_words(blocks),
            tail_tokens=tail,
        )

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.seq_len, self.num_heads, self.head_dim)

    @property
    def summary_shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.num_kv_blocks, self.num_heads, self.head_dim)

    @property
    def stats_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.num_heads, self.head_dim)

    @property
    def exact_block_bits_shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.num_heads, self.num_q_blocks, self.exact_words)


@dataclass(frozen=True)
class SolPredictorPlanKey:
    """Cache key containing only static kernel specialization state."""

    geometry: SolPredictorGeometry
    device_index: int
    dtype: torch.dtype


@dataclass(frozen=True)
class SolPredictorOutputs:
    """Live predictor tensors consumed by block-sparse attention."""

    exact_block_bits: torch.Tensor
    k_summary: torch.Tensor
    v_summary: torch.Tensor


@dataclass(frozen=True)
class SolPredictorPlan:
    """One published shape specialization and its stable live storage."""

    key: SolPredictorPlanKey
    outputs: SolPredictorOutputs
    k_mean: torch.Tensor
    k_var_diag: torch.Tensor
    q_centroid: torch.Tensor


class SOLSparsePredictor:
    """Cache of shape-specialized, graph-stable SOL predictor plans."""

    def __init__(self) -> None:
        self._plans: dict[SolPredictorPlanKey, SolPredictorPlan] = {}

    @property
    def num_plans(self) -> int:
        return len(self._plans)

    @staticmethod
    def support_reason(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str | None:
        """Return why compact two-stage SOL cannot serve these tensors."""

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

    @classmethod
    def _key_from_inputs(
        cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> SolPredictorPlanKey:
        reason = cls.support_reason(q, k, v)
        if reason is not None:
            raise ValueError(reason)
        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        geometry = SolPredictorGeometry.create(
            batch_size=q.shape[0],
            seq_len=q.shape[1],
            num_heads=q.shape[2],
            head_dim=q.shape[3],
        )
        return SolPredictorPlanKey(
            geometry=geometry,
            device_index=device_index,
            dtype=q.dtype,
        )

    @staticmethod
    def _launch(
        plan: SolPredictorPlan,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tau: float,
        sm_scale: float,
    ) -> None:
        torch.ops.trtllm.visual_gen_sol_predictor(
            q,
            k,
            v,
            plan.outputs.exact_block_bits,
            plan.outputs.k_summary,
            plan.outputs.v_summary,
            plan.k_mean,
            plan.k_var_diag,
            plan.q_centroid,
            BLOCK_SIZE,
            tau,
            sm_scale,
        )

    @torch.compiler.disable
    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> SolPredictorPlan:
        """Allocate one geometry and warm its kernels outside compiled or captured regions.

        The host-only boundary preserves per-instance plan ownership and keeps
        kernel compilation and allocation out of Dynamo and CUDA Graph capture.
        It requires the VisualGen default ``torch.compile(fullgraph=False)``.
        """

        key = self._key_from_inputs(q, k, v)
        existing = self._plans.get(key)
        if existing is not None:
            return existing
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SOL predictor plan must be prepared before CUDA graph capture")

        geometry = key.geometry
        with torch.cuda.device(key.device_index):
            k_summary = torch.empty(geometry.summary_shape, dtype=key.dtype, device=q.device)
            v_summary = torch.empty_like(k_summary)
            exact_block_bits = torch.empty(
                geometry.exact_block_bits_shape, dtype=torch.uint32, device=q.device
            )
            k_mean = torch.empty(geometry.stats_shape, dtype=torch.float32, device=q.device)
            k_var_diag = torch.empty_like(k_mean)
            q_centroid = torch.empty(geometry.summary_shape, dtype=torch.float32, device=q.device)
        plan = SolPredictorPlan(
            key=key,
            outputs=SolPredictorOutputs(
                exact_block_bits=exact_block_bits,
                k_summary=k_summary,
                v_summary=v_summary,
            ),
            k_mean=k_mean,
            k_var_diag=k_var_diag,
            q_centroid=q_centroid,
        )
        # The warm-up launch compiles every kernel specialization of this geometry.
        self._launch(plan, q, k, v, tau=0.0, sm_scale=1.0)
        self._plans[key] = plan
        return plan

    def predict(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tau: object,
        sm_scale: object,
    ) -> SolPredictorOutputs:
        """Update and return graph-stable SOL routes and proxy summaries."""

        effective_tau, effective_sm_scale = _normalize_runtime_scalars(tau=tau, sm_scale=sm_scale)
        plan = self.prepare(q, k, v)
        self._launch(plan, q, k, v, tau=effective_tau, sm_scale=effective_sm_scale)
        return plan.outputs


__all__ = [
    "SOLSparsePredictor",
    "SolPredictorGeometry",
    "SolPredictorOutputs",
    "SolPredictorPlan",
    "SolPredictorPlanKey",
]
