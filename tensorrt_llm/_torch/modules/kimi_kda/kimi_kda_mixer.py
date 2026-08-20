# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiKDALinearAttention — Kimi K3 linear-attention module for the PyTorch backend.

Structural mirror of the HF reference ``KimiDeltaAttention`` in
``modeling_kimi.py``. Same parameter names, same layer shapes, same short
convolution + FLA gating + FusedRMSNormGated output-gate stack. The
delta-rule inner loop is routed through :mod:`_kda_kernels`, which selects
the optimized sm_100 CuTe/Triton chunked prefill and fused CUDA decode
kernels on Blackwell and falls back to the FLA references elsewhere.

Cache ownership
---------------
KDA carries three short-convolution states (``conv_state_{q,k,v}``, HF
layout ``[B, D, W]`` bf16) and one delta-rule recurrent state
(``recurrent_state``, layout ``[B, HV, V, K]`` fp32, matching the optimized
kernel's transposed convention). These match the hybrid-cache ownership
pattern used by the mamba modules; the runtime cache-manager plumbing
(``AttentionMetadata`` split, cache indices, spec/verify path) is deferred
to the model-assembly wiring goal. This module exposes parity entry points
that consume and return the state tensors directly so module-level tests
can prove state roundtrip without the runtime plumbing.

Kernel mutations for negative controls
--------------------------------------
Two invariants have their own construction switches so parity tests can
prove they are actually being enforced:

* ``gate_lower_bound_override`` — replace the ``linear_attn_config``
  gate lower bound at forward time. A value that disagrees with the HF
  reference must fail parity.
* ``wrong_state_layout`` — permute the recurrent state's V/K axes before
  and after the decode kernel call so read/write hit mislabeled slots.
  Because K == V the shape check still passes but the numerics break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import fused_recurrent_kda
from torch import nn

from ._kda_kernels import KDAKernelDispatch, is_kda_optimized_supported


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
class KimiKDACachedState:
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


class KimiKDAKernelPath:
    """Enum-like string tags for the selected KDA kernel path."""

    OPTIMIZED = "optimized"
    FLA = "fla"


def _hf_conv_to_kernel_conv(
    hf_cache: Optional[torch.Tensor],
    b: int,
    d: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """HF ``[B, D, W]`` conv cache -> optimized kernel's ``[B, D, W-1]``.

    HF stores W positions with the newest processed token last. The
    optimized decode kernel's ``cs_*`` argument stores the ``W-1``
    historical positions before the incoming token; drop the oldest column.
    """
    if hf_cache is None:
        return torch.zeros(b, d, w - 1, device=device, dtype=dtype)
    return hf_cache[:, :, 1:].contiguous()


def _roll_hf_conv(
    prev_hf: Optional[torch.Tensor],
    x_new_col: torch.Tensor,
    b: int,
    d: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Roll an HF-layout conv cache by one token.

    HF ``ShortConvolution.step`` does
    ``cache.copy_(cache.roll(shifts=-1, dims=-1)); cache[:, :, -1] = x``.
    We implement the same semantics via ``torch.cat`` so the update is
    independent of the kernel's internal cs handling.
    """
    if prev_hf is None:
        prev = torch.zeros(b, d, w, device=device, dtype=dtype)
    else:
        prev = prev_hf.to(dtype=dtype)
    return torch.cat([prev[:, :, 1:], x_new_col.to(dtype)], dim=-1).contiguous()


class KimiKDALinearAttention(nn.Module):
    """Kimi K3 linear-attention module — in-tree production version.

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
    use_optimized_prefill : bool
        Enable the optimized prefill path when supported.
    use_optimized_decode : bool
        Enable the optimized decode path when supported.
    gate_lower_bound_override : Optional[float]
        Override the ``linear_attn_config`` gate lower bound. Test knob for
        the "wrong gate lower bound" mutation control.
    wrong_state_layout : bool
        Swap the V and K axes of the recurrent state around the decode
        kernel call. Test knob for the "wrong state layout" mutation
        control on the decode path.
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
        use_optimized_prefill: bool = True,
        use_optimized_decode: bool = True,
        gate_lower_bound_override: Optional[float] = None,
        wrong_state_layout: bool = False,
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
        self.gate_lower_bound_override = gate_lower_bound_override
        self.wrong_state_layout = wrong_state_layout

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
        # dt_bias must be initialized: torch.empty heap garbage can contain
        # NaN bit patterns, which poison both the optimized and FLA gates in
        # randomly-constructed modules (parity tests) — TRTLLM-15204. This
        # matches FLA's KDA init (inverse-softplus of dt ~ LogUniform[1e-3,
        # 1e-1], fla/layers/kda.py) in its small-dt regime where
        # softplus(x) ≈ exp(x), expressed as a single uniform_ so it stays on
        # MetaInitMode's random-init allowlist (exp/expm1/clamp would raise
        # MetaInitException and force the full-CPU-init fallback). Checkpoint
        # loading overwrites the value either way.
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

        # Installed together by the FP8 weight loader (fused [q | k | v | g]
        # decode GEMM). Declared here so the decode path never sees a
        # half-installed pair.
        self.qkvg_proj: Optional[nn.Module] = None
        self.qkvg_split_sizes: Optional[list[int]] = None

        if dtype is not None:
            _meta_safe_cast_dtype(self, dtype)

        # The optimized decode/verify kernels are specialized for the Kimi
        # K3 shape (K == V == 128). Reduced-dim test configurations must
        # fall back to FLA instead of hard-failing inside the kernels.
        kernel_shape_ok = self.head_k_dim == 128 and self.head_dim == 128
        self._dispatch = KDAKernelDispatch(
            use_optimized_prefill=use_optimized_prefill,
            use_optimized_decode=use_optimized_decode and kernel_shape_ok,
            use_optimized_verify=kernel_shape_ok,
        )

    # ------------------------------------------------------------------
    # Introspection helpers used by the test smoke.
    # ------------------------------------------------------------------

    @property
    def prefill_kernel_path(self) -> str:
        return self._dispatch.prefill_kernel_path

    @property
    def decode_kernel_path(self) -> str:
        return self._dispatch.decode_kernel_path

    @property
    def verify_kernel_path(self) -> str:
        return self._dispatch.verify_kernel_path

    @property
    def sm_100_optimized_supported(self) -> bool:
        return is_kda_optimized_supported()

    def prefill_kernel_source(self) -> str:
        return self._dispatch.get_prefill_source()

    def decode_kernel_source(self) -> str:
        return self._dispatch.get_decode_source()

    def prefill_chunk_kda(self, **kwargs):
        """Kernel-level chunked prefill via the dispatch.

        Used by the executor runtime (``KimiKDARuntime``), which owns the
        projections, convs, and cache pools itself and only needs the
        delta-rule inner loop. States are exchanged in the V-first
        ``[N, H, V, K]`` pool layout on both dispatch paths — see
        ``KDAKernelDispatch.prefill_chunk_kda``.
        """
        return self._dispatch.prefill_chunk_kda(**kwargs)

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

        lower_bound = (
            self.gate_lower_bound_override
            if self.gate_lower_bound_override is not None
            else self.gate_lower_bound
        )
        safe_gate = lower_bound is not None
        scale = self.head_k_dim**-0.5

        o, _final_state = self._dispatch.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=scale,
            initial_state=None,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
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

    # ------------------------------------------------------------------
    # Decode entry (Goal 2.1 pass path).
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[KimiKDACachedState] = None,
        ssm_state_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KimiKDACachedState]:
        """T=1 cached-decode forward. Returns ``(o, new_cache)``.

        ``hidden_states`` shape ``(B, 1, hidden_size)``. Cache is
        ``KimiKDACachedState`` in HF layout; ``None`` fields become zero
        tensors.
        """
        b, q_len, _ = hidden_states.shape
        assert q_len == 1, f"KimiKDALinearAttention.forward_decode expects T=1, got T={q_len}"

        if self._dispatch.decode_kernel_path == KimiKDAKernelPath.OPTIMIZED:
            return self._decode_via_optimized(hidden_states, cache, b, ssm_state_indices)
        if ssm_state_indices is not None:
            raise ValueError("ssm_state_indices requires the optimized KDA decode kernel")
        return self._decode_via_fla(hidden_states, cache, b)

    # ------------------------------------------------------------------
    # Internals — optimized decode dispatch.
    # ------------------------------------------------------------------

    def _decode_via_optimized(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[KimiKDACachedState],
        b: int,
        ssm_state_indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, KimiKDACachedState]:
        dev = hidden_states.device
        H = self.num_heads
        HV = self.num_heads
        K_dim = self.head_dim
        V_dim = self.head_dim
        W = self.conv_size

        projection_size = H * K_dim
        projection_v_size = HV * V_dim

        # q/k/v and the full-rank output gate all read this same normed hidden.
        # When their weights are read at FP8 block-scale (Blackwell decode), the
        # loader fuses them into one ``qkvg_proj`` GEMM: one activation quant and
        # one GEMM launch replace four, which is what the launch-bound
        # generation step needs. The split is output-identical to the per
        # projection GEMMs (same activation, same weight slices). The forget
        # gate (f_a/f_b), beta and low-rank output gate stay BF16, so they keep
        # their own calls.
        fused_qkvg = self.qkvg_proj if self.qkvg_split_sizes is not None else None
        if fused_qkvg is not None:
            parts = fused_qkvg(hidden_states).split(self.qkvg_split_sizes, dim=-1)
            q_proj_states, k_proj_states, v_proj_states = parts[0], parts[1], parts[2]
            onorm_g_hidden = (
                parts[3] if self.use_full_rank_gate else self.g_b_proj(self.g_a_proj(hidden_states))
            )
        else:
            q_proj_states = self.q_proj(hidden_states)
            k_proj_states = self.k_proj(hidden_states)
            v_proj_states = self.v_proj(hidden_states)
            onorm_g_hidden = (
                self.g_proj(hidden_states)
                if self.use_full_rank_gate
                else self.g_b_proj(self.g_a_proj(hidden_states))
            )

        g_hidden = self.f_b_proj(self.f_a_proj(hidden_states))

        beta_hidden = self.b_proj(hidden_states).float()

        def _kernel_input(proj: torch.Tensor, h: int, d: int) -> torch.Tensor:
            x = rearrange(proj, "b t (h d) -> t b h d", h=h, d=d)
            return x.to(dtype=torch.bfloat16).contiguous()

        x_q_full = _kernel_input(q_proj_states, H, K_dim)
        x_k_full = _kernel_input(k_proj_states, H, K_dim)
        x_v_full = _kernel_input(v_proj_states, HV, V_dim)
        g_full = _kernel_input(g_hidden, H, K_dim)
        onorm_g_full = _kernel_input(onorm_g_hidden, HV, V_dim)
        beta_full = rearrange(beta_hidden, "b t h -> t b h").to(torch.bfloat16).contiguous()

        w_q_t_full = (
            self.q_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        w_k_t_full = (
            self.k_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        w_v_t_full = (
            self.v_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )

        if cache is not None and cache.conv_state_q is not None:
            hf_cs_q_pre = cache.conv_state_q.to(torch.bfloat16)
        else:
            hf_cs_q_pre = torch.zeros(b, projection_size, W, device=dev, dtype=torch.bfloat16)
        if cache is not None and cache.conv_state_k is not None:
            hf_cs_k_pre = cache.conv_state_k.to(torch.bfloat16)
        else:
            hf_cs_k_pre = torch.zeros(b, projection_size, W, device=dev, dtype=torch.bfloat16)
        if cache is not None and cache.conv_state_v is not None:
            hf_cs_v_pre = cache.conv_state_v.to(torch.bfloat16)
        else:
            hf_cs_v_pre = torch.zeros(b, projection_v_size, W, device=dev, dtype=torch.bfloat16)

        cs_q_full = _hf_conv_to_kernel_conv(hf_cs_q_pre, b, projection_size, W, dev, torch.bfloat16)
        cs_k_full = _hf_conv_to_kernel_conv(hf_cs_k_pre, b, projection_size, W, dev, torch.bfloat16)
        cs_v_full = _hf_conv_to_kernel_conv(
            hf_cs_v_pre, b, projection_v_size, W, dev, torch.bfloat16
        )

        x_q_col = q_proj_states.transpose(1, 2).to(torch.bfloat16)
        x_k_col = k_proj_states.transpose(1, 2).to(torch.bfloat16)
        x_v_col = v_proj_states.transpose(1, 2).to(torch.bfloat16)
        new_hf_cs_q = _roll_hf_conv(
            hf_cs_q_pre, x_q_col, b, projection_size, W, dev, torch.bfloat16
        )
        new_hf_cs_k = _roll_hf_conv(
            hf_cs_k_pre, x_k_col, b, projection_size, W, dev, torch.bfloat16
        )
        new_hf_cs_v = _roll_hf_conv(
            hf_cs_v_pre, x_v_col, b, projection_v_size, W, dev, torch.bfloat16
        )

        if cache is not None and cache.recurrent_state is not None:
            if ssm_state_indices is not None:
                if self.wrong_state_layout:
                    raise ValueError("ssm_state_indices is incompatible with wrong_state_layout")
                state_full = cache.recurrent_state
            else:
                state_full = cache.recurrent_state.to(dtype=torch.float32).contiguous()
        else:
            if ssm_state_indices is not None:
                raise ValueError("ssm_state_indices requires a recurrent state pool")
            state_full = torch.zeros(b, HV, V_dim, K_dim, device=dev, dtype=torch.float32)

        # The decode op requires fp32 A_log/dt_bias even in a bf16-cast module.
        A_log_full = self.A_log.detach().float().contiguous()
        dt_bias_full = self.dt_bias.detach().float().contiguous()
        onorm_weight_full = self.o_norm.weight.detach().to(torch.float32).contiguous()
        lower_bound = (
            self.gate_lower_bound_override
            if self.gate_lower_bound_override is not None
            else self.gate_lower_bound
        )

        kernel_state = (
            state_full.transpose(-1, -2).contiguous() if self.wrong_state_layout else state_full
        )
        o_bfhvk = self._dispatch.decode_kda(
            x_q=x_q_full,
            x_k=x_k_full,
            x_v=x_v_full,
            w_q_t=w_q_t_full,
            w_k_t=w_k_t_full,
            w_v_t=w_v_t_full,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=cs_q_full,
            cs_k=cs_k_full,
            cs_v=cs_v_full,
            A_log=A_log_full,
            g=g_full,
            dt_bias=dt_bias_full,
            beta=beta_full,
            state=kernel_state,
            onorm_g=onorm_g_full,
            onorm_weight=onorm_weight_full,
            out=None,
            ssm_state_indices=ssm_state_indices,
            cu_seqlens=None,
            scale=K_dim**-0.5,
            onorm_eps=self.o_norm.eps,
            lower_bound=lower_bound,
            use_beta_sigmoid_in_kernel=True,
            verbose=False,
            update_conv_cache=False,
        )
        state_full = (
            kernel_state.transpose(-1, -2).contiguous() if self.wrong_state_layout else kernel_state
        )

        o_flat = rearrange(o_bfhvk, "b t h d -> b t (h d)")
        o = self.o_proj(o_flat)

        new_cache = KimiKDACachedState(
            conv_state_q=new_hf_cs_q,
            conv_state_k=new_hf_cs_k,
            conv_state_v=new_hf_cs_v,
            recurrent_state=state_full,
        )
        return o, new_cache

    # ------------------------------------------------------------------
    # Internals — FLA fallback decode (non-sm_100 path).
    # ------------------------------------------------------------------

    def _decode_via_fla(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[KimiKDACachedState],
        b: int,
    ) -> Tuple[torch.Tensor, KimiKDACachedState]:
        """FLA ``fused_recurrent_kda`` decode path — used when sm_100 is unavailable.

        Matches HF ``KimiDeltaAttention`` in ``fused_recurrent`` mode: uses
        the ``ShortConvolution.step`` semantics and dispatches the delta
        update to ``fla.ops.kda.fused_recurrent_kda``.
        """
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

        lower_bound = (
            self.gate_lower_bound_override
            if self.gate_lower_bound_override is not None
            else self.gate_lower_bound
        )

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
            lower_bound=lower_bound,
            state_v_first=True,
        )

        onorm_g = rearrange(onorm_g_hidden, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, onorm_g)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        new_cache = KimiKDACachedState(
            conv_state_q=new_conv_q,
            conv_state_k=new_conv_k,
            conv_state_v=new_conv_v,
            recurrent_state=new_recurrent,
        )
        return o, new_cache

    # ------------------------------------------------------------------
    # Weight helper for random-weight parity tests.
    # ------------------------------------------------------------------

    def copy_weights_from(self, source: nn.Module) -> "dict[str, Tuple[Tuple[int, ...], str]]":
        """Copy every named parameter/buffer from ``source`` into ``self``.

        Because ``KimiKDALinearAttention`` mirrors the HF reference's
        parameter names 1:1, the mapping is identity: every source name is
        assigned to the identically named target. Shape mismatches raise
        loudly. Returns a ``{name: (shape, dtype)}`` provenance dict.
        """
        src: dict[str, torch.Tensor] = {}
        for name, p in source.named_parameters(recurse=True):
            src[name] = p.data
        for name, buf in source.named_buffers(recurse=True):
            src[name] = buf

        dst: dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters(recurse=True):
            dst[name] = p.data
        for name, buf in self.named_buffers(recurse=True):
            dst[name] = buf

        missing_on_dst = sorted(set(src) - set(dst))
        missing_on_src = sorted(set(dst) - set(src))
        if missing_on_dst:
            raise KeyError(
                f"copy_weights_from: source params missing on target: {missing_on_dst[:5]}"
            )
        if missing_on_src:
            raise KeyError(
                f"copy_weights_from: target params missing on source: {missing_on_src[:5]}"
            )

        provenance: "dict[str, Tuple[Tuple[int, ...], str]]" = {}
        for name, srct in src.items():
            dstt = dst[name]
            if srct.shape != dstt.shape:
                raise ValueError(
                    f"shape mismatch for {name}: source {tuple(srct.shape)} "
                    f"vs target {tuple(dstt.shape)}"
                )
            dstt.copy_(srct.to(dtype=dstt.dtype, device=dstt.device))
            provenance[name] = (tuple(srct.shape), str(srct.dtype))
        return provenance
