# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLA-only Kimi KDA parity reference.

This is a structural mirror of the HF reference ``KimiDeltaAttention`` in
``modeling_kimi.py``. Same parameter names, same layer shapes, same short
convolution + FLA gating + FusedRMSNormGated output-gate stack. The
delta-rule inner loop calls FLA directly so optimized production kernels
are always compared against an independent implementation. Production code
must use ``tensorrt_llm._torch.modules.kimi_kda.KimiKDALinearAttention``.

Cache ownership
---------------
KDA carries three short-convolution states (``conv_state_{q,k,v}``, HF
layout ``[B, D, W]`` bf16) and one delta-rule recurrent state
(``recurrent_state``, layout ``[B, HV, V, K]`` fp32, matching the optimized
kernel's transposed convention). These match the hybrid-cache ownership
pattern used by the mamba modules. The reference consumes and returns state
tensors directly so module-level tests can prove state roundtrip without
runtime cache-manager plumbing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from torch import nn

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention


def get_production_prefill_kernel_path(attention: KimiKDALinearAttention) -> str:
    """Return the selected production prefill path for kernel-routing tests."""
    return attention._dispatch.prefill_kernel_path


def get_production_decode_kernel_path(attention: KimiKDALinearAttention) -> str:
    """Return the selected production decode path for kernel-routing tests."""
    return attention._dispatch.decode_kernel_path


def _meta_safe_cast_dtype(module, dtype):
    """``module.to(dtype=dtype)`` that also works under ``MetaInitMode``.

    ``Module.to`` dispatches ``aten._to_copy``, which MetaInitMode rejects
    (it would silently fall back to full CPU construction of the model —
    ~70 GB of host RAM per rank for Kimi K3). Under meta init the values
    are garbage anyway, so a dtype-only re-allocation via ``empty_like``
    (an allowed init op) is equivalent; off meta this matches ``.to``.
    """
    import torch as _torch

    def _cast(t):
        if not t.is_floating_point():
            return t
        if t.is_meta:
            return _torch.empty_like(t, dtype=dtype)
        return t.to(dtype=dtype)

    module._apply(_cast)


class _MetaSafeFusedRMSNormGated(FusedRMSNormGated):
    """FusedRMSNormGated whose init survives the model loader's MetaInitMode.

    ``FusedRMSNormGated.reset_parameters`` uses ``nn.init.ones_`` (a plain
    ``fill_``), which MetaInitMode rejects on meta tensors and which would
    force the whole model construction to fall back to eager CPU init.
    ``uniform_(1, 1)`` produces identical values and is on MetaInitMode's
    random-init allowlist.
    """

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            with torch.no_grad():
                self.weight.uniform_(1.0, 1.0)


@dataclass
class KimiKDATestCachedState:
    """Per-layer KDA cache tensors in HF layout.

    ``conv_state_*`` — shape ``[B, D, W]`` bf16 where ``W`` is
    ``short_conv_kernel_size`` and the newest processed token sits at
    position ``W-1``. ``D`` is ``num_heads * head_dim`` for q/k and
    ``num_heads * head_dim`` for v (K3 uses HV == H).

    ``recurrent_state`` — shape ``[B, HV, V, K]`` fp32. This is the
    transposed layout the optimized KDA kernels expect, and it is the same
    layout HF stores when running with ``transpose_state_layout=True``.
    ``None`` fields are treated as zero.
    """

    conv_state_q: Optional[torch.Tensor]
    conv_state_k: Optional[torch.Tensor]
    conv_state_v: Optional[torch.Tensor]
    recurrent_state: Optional[torch.Tensor]


class KimiKDAReference(nn.Module):
    """Standalone FLA Kimi K3 linear-attention parity reference.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    head_dim : int
    conv_kernel_size : int
    use_full_rank_gate : bool
    gate_lower_bound : Optional[float]
    rms_norm_eps : float
    dtype : Optional[torch.dtype]
    layer_idx : int
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        use_full_rank_gate: bool,
        gate_lower_bound: Optional[float],
        rms_norm_eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_k_dim = head_dim
        self.num_k_heads = num_heads
        self.conv_size = conv_kernel_size
        self.use_full_rank_gate = use_full_rank_gate
        self.gate_lower_bound = gate_lower_bound
        self.rms_norm_eps = rms_norm_eps
        self.layer_idx = layer_idx

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=conv_kernel_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=conv_kernel_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=conv_kernel_size,
            activation="silu",
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32).uniform_(
                math.log(1e-3), math.log(1e-1)
            )
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        if use_full_rank_gate:
            self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
            self.g_b_proj = nn.Linear(head_dim, projection_size, bias=False)

        self.o_norm = _MetaSafeFusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

        if dtype is not None:
            _meta_safe_cast_dtype(self, dtype)

    # ------------------------------------------------------------------
    # Prefill entry (Goal 2.1 pass path).
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prefill forward matching HF ``KimiDeltaAttention.forward`` in chunk mode.

        Parameters
        ----------
        hidden_states : ``(B, T, hidden_size)`` for equal-length prefill or
            ``(1, sum(seq_lens), hidden_size)`` when ``cu_seqlens`` is given.
        cu_seqlens : optional cumulative sequence lengths for varlen inputs.

        Returns
        -------
        ``(B, T, hidden_size)`` output tensor (equal-length case) or
        ``(1, sum(seq_lens), hidden_size)`` (varlen case).
        """
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device=hidden_states.device, dtype=torch.long)

        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)

        q, _ = self.q_conv1d(
            x=q_proj_states,
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        k, _ = self.k_conv1d(
            x=k_proj_states,
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        v, _ = self.v_conv1d(
            x=v_proj_states,
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        lower_bound = self.gate_lower_bound
        safe_gate = lower_bound is not None
        scale = self.head_k_dim**-0.5

        o, _final_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=scale,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

        if self.use_full_rank_gate:
            g_out = self.g_proj(hidden_states)
        else:
            g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g_out)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[KimiKDATestCachedState] = None,
    ) -> Tuple[torch.Tensor, KimiKDATestCachedState]:
        """Run the FLA T=1 cached-decode reference.

        ``hidden_states`` shape ``(B, 1, hidden_size)``. Cache is
        ``KimiKDATestCachedState`` in HF layout; ``None`` fields become zero
        tensors.
        """
        _, q_len, _ = hidden_states.shape
        assert q_len == 1, f"KimiKDAReference.forward_decode expects T=1, got T={q_len}"
        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)

        conv_q_in = cache.conv_state_q if cache is not None else None
        conv_k_in = cache.conv_state_k if cache is not None else None
        conv_v_in = cache.conv_state_v if cache is not None else None
        recurrent_in = cache.recurrent_state if cache is not None else None

        q, new_conv_q = self.q_conv1d(x=q_proj_states, cache=conv_q_in, output_final_state=True)
        k, new_conv_k = self.k_conv1d(x=k_proj_states, cache=conv_k_in, output_final_state=True)
        v, new_conv_v = self.v_conv1d(x=v_proj_states, cache=conv_v_in, output_final_state=True)

        g_hidden = self.f_b_proj(self.f_a_proj(hidden_states))
        if self.use_full_rank_gate:
            onorm_g_hidden = self.g_proj(hidden_states)
        else:
            onorm_g_hidden = self.g_b_proj(self.g_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float()

        g = rearrange(g_hidden, "... (h d) -> ... h d", d=self.head_dim)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        o, new_recurrent = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_in,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            state_v_first=True,
        )

        onorm_g = rearrange(onorm_g_hidden, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, onorm_g)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        new_cache = KimiKDATestCachedState(
            conv_state_q=new_conv_q,
            conv_state_k=new_conv_k,
            conv_state_v=new_conv_v,
            recurrent_state=new_recurrent,
        )
        return o, new_cache
