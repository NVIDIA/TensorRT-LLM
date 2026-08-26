# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp Hyper-Connection residual module (contract C1).

Qwen3.8-Flash-Next does **not** use the standard
``input_layernorm`` / ``post_attention_layernorm`` + single residual decoder
contract. Instead it carries ``hc_count`` (4) parallel residual streams — the
hidden state is a ``hc_count * hidden_size`` (= 10240) bundle between layers —
and each block wraps its compute in a Hyper-Connection:

* ``mix``   (10240 -> 2560): a low-rank, silu/sigmoid-gated weighted average of
  the 4 (grouped-RMSNorm-normalized) streams, feeding the block a single
  ``hidden_size`` tensor.
* ``combine`` (2560 -> 10240): injects the block output back into the 4 streams
  with a learned per-stream sigmoid gate.
* the model's final ``hyper_connection_mixer.mix`` (10240 -> 2560) is the last
  norm before ``lm_head`` (there is **no** separate final RMSNorm) and is built
  with ``use_combine=False``.

This is a TRT-LLM reimplementation of the sglang reference
``GatedResidualSimple`` / ``GroupedGemmaRMSNorm`` (source of truth:
``sglang/srt/layers/hyperconnection.py``), matching the reference **non-fused
fallback** math exactly. Its grouped Gemma RMSNorm reuses TRT-LLM's existing
Mamba Triton kernel with FP32 ``1 + delta_weight`` semantics. The checkpoint
stores the Qwen4-Exp variant with ``hc_per_branch_norm=True`` (a per-element
``[10240]`` grouped-RMSNorm weight, grouped by ``hidden_size``).

Parameter names (``hc_norm.weight``, ``input_mix_weight_down.weight``,
``input_mix_weight_up.weight``, ``block_inject_weight.weight``) mirror the
checkpoint exactly so the model weight loader maps them by direct copy. All
three HC projections operate on the full per-token bundle and are **replicated**
across tensor-parallel ranks (the reference does not shard them; the block's TP
all-reduce happens between ``mix`` and ``combine`` on the block output), so plain
``nn.Linear`` / a native norm are used rather than TP-sharded TRT-LLM ``Linear``.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .mamba.layernorm_gated import RMSNorm as TritonRMSNorm

__all__ = ["GroupedRMSNorm", "Qwen4ExpHyperConnection"]


class GroupedRMSNorm(TritonRMSNorm):
    """Gemma-style grouped RMS norm using TRT-LLM's grouped Triton kernel.

    A native implementation remains available for CPU/model-construction tests.
    This matches sglang's ``GroupedGemmaRMSNorm``.

    Normalizes over ``group_size`` slices of the last dim independently (RMS,
    fp32 accumulation) and scales by ``(1.0 + weight)`` (Gemma convention: the
    stored weight is a delta around 1). When ``group_size is None`` the whole
    last dim is one group (a plain Gemma RMSNorm).

    For Qwen4-Exp ``hc_per_branch_norm=True``: ``normalized_shape=10240``,
    ``group_size=2560`` — each of the 4 streams is RMS-normalized over its 2560
    dims and scaled by its own slice of the ``[10240]`` weight.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        if group_size is not None and normalized_shape % group_size != 0:
            raise ValueError(
                f"normalized_shape ({normalized_shape}) must be divisible by "
                f"group_size ({group_size})"
            )
        super().__init__(
            normalized_shape,
            eps=eps,
            group_size=group_size,
            dtype=dtype,
            device=device,
            weight_is_delta=True,
        )
        # Gemma init: the weight is a delta around 1, so a fresh (unloaded)
        # module is the identity scale. Constructing a new tensor is required
        # here because TRT-LLM meta initialization rejects in-place ``zero_``.
        self.weight = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return super().forward(x)
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size == self.hidden_size:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (x_grouped * torch.rsqrt(variance + self.variance_epsilon)).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


# The residual state threaded from ``mix`` to ``combine``: the original bundle
# and its grouped-RMSNorm-normalized form (both are needed by ``combine``).
HCResidual = Tuple[torch.Tensor, torch.Tensor]


class Qwen4ExpHyperConnection(nn.Module):
    """Qwen4-Exp Hyper-Connection block (sglang ``GatedResidualSimple``).

    Args:
        hc_count: number of parallel residual streams (config ``hc_count``, 4).
        hidden_size: per-stream hidden size (config ``hidden_size``, 2560).
        hc_lowrank: low-rank width of the mix down/up projection
            (config ``hc_lowrank``, 320).
        rms_norm_eps: epsilon for the grouped RMS norm (config ``rms_norm_eps``).
        hc_per_branch_norm: when True (the Qwen4-Exp checkpoint layout) the norm
            weight is ``[hc_count*hidden_size]`` grouped by ``hidden_size``;
            when False it is a shared ``[hidden_size]`` weight.
        dtype: parameter dtype (checkpoint is bf16).
        use_mix / use_combine: which sub-projections to build. The per-layer
            attn/mlp mixers use both; the model's final ``hyper_connection_mixer``
            uses ``use_combine=False`` (mix-only "last norm").
    """

    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        hc_lowrank: int,
        rms_norm_eps: float = 1e-6,
        hc_per_branch_norm: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        use_mix: bool = True,
        use_combine: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not (use_mix or use_combine):
            raise ValueError("use_mix or use_combine must be set")
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hc_lowrank = hc_lowrank
        self.hc_per_branch_norm = hc_per_branch_norm
        self.params_dtype = dtype
        self.use_mix = use_mix
        self.use_combine = use_combine

        norm_dim = hidden_size * hc_count if hc_per_branch_norm else hidden_size
        norm_group_size = hidden_size if hc_per_branch_norm else None
        self.hc_norm = GroupedRMSNorm(
            norm_dim, eps=rms_norm_eps, group_size=norm_group_size, dtype=dtype, device=device
        )

        hc_dim = hc_count * hidden_size
        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                hc_dim, hc_lowrank, bias=False, dtype=dtype, device=device
            )
            self.input_mix_weight_up = nn.Linear(
                hc_lowrank, hc_dim, bias=False, dtype=dtype, device=device
            )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                hc_dim, hc_count, bias=False, dtype=dtype, device=device
            )

    @classmethod
    def from_config(
        cls,
        config,
        use_mix: bool = True,
        use_combine: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> "Qwen4ExpHyperConnection":
        """Build from a ``Qwen4ExpTextConfig``-like object (reads ``hc_count``,
        ``hidden_size``, ``hc_lowrank``, ``rms_norm_eps``). Qwen4-Exp always uses
        ``hc_per_branch_norm=True``."""
        return cls(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            hc_per_branch_norm=True,
            dtype=dtype,
            use_mix=use_mix,
            use_combine=use_combine,
            device=device,
        )

    def _normed_bundle(self, hyper_input: torch.Tensor) -> torch.Tensor:
        """Grouped-RMSNorm the ``[..., hc_count*hidden_size]`` bundle (per-stream)."""
        if self.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        # Shared-weight variant: norm each stream over hidden_size with the same
        # [hidden_size] weight, then re-flatten.
        return self.hc_norm(hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))).flatten(
            -2
        )

    def mix(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, HCResidual]:
        """10240 -> 2560. Returns ``(mixed_input, residual_state)`` where
        ``residual_state = (hyper_input, hyper_input_normed)`` is threaded to
        ``combine``."""
        assert self.use_mix, "mix() called on a combine-only Hyper-Connection"
        hc, hs = self.hc_count, self.hidden_size
        assert hyper_input.shape[-1] == hc * hs, (
            f"hyper_input last dim {hyper_input.shape[-1]} != hc_count*hidden ({hc * hs})"
        )
        # Empty batch (idle rank / CUDA-graph padding): return an empty mix and a
        # trivial residual state, matching the reference guard.
        if hyper_input.shape[0] == 0:
            mixed = hyper_input.new_empty((*hyper_input.shape[:-1], hs), dtype=self.params_dtype)
            return mixed, (hyper_input, hyper_input)

        normed = self._normed_bundle(hyper_input)
        # low-rank gate: silu(down(normed)/hc) -> up -> sigmoid, then a
        # per-(stream,dim) weighted average of the normed streams.
        gate = F.silu(self.input_mix_weight_down(normed) / hc)
        gate = self.input_mix_weight_up(gate)
        gate = torch.sigmoid(gate).unflatten(-1, (hc, hs))
        mixed = (gate * normed.unflatten(-1, (hc, hs))).mean(dim=-2)
        return mixed.to(self.params_dtype), (hyper_input, normed)

    def combine(self, block_output: torch.Tensor, residual: HCResidual) -> torch.Tensor:
        """2560 -> 10240. Injects ``block_output`` into the 4 residual streams
        with a learned per-stream sigmoid gate."""
        assert self.use_combine, "combine() on a mix-only Hyper-Connection"
        hc, hs = self.hc_count, self.hidden_size
        hyper_input, normed = residual
        assert hyper_input.shape[-1] == hc * hs
        assert block_output.shape[-1] == hs
        # Empty batch: pass the (untouched) bundle through, matching the reference.
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype)

        streams = hyper_input.unflatten(-1, (hc, hs))
        inject_gate = 2.0 * torch.sigmoid(self.block_inject_weight(normed) / hc)
        injection = block_output.unsqueeze(-2) * inject_gate.unsqueeze(-1)
        return (streams + injection).flatten(-2).to(self.params_dtype)

    def extra_repr(self) -> str:
        return (
            f"hc_count={self.hc_count}, hidden_size={self.hidden_size}, "
            f"hc_lowrank={self.hc_lowrank}, "
            f"hc_per_branch_norm={self.hc_per_branch_norm}, "
            f"use_mix={self.use_mix}, use_combine={self.use_combine}"
        )
