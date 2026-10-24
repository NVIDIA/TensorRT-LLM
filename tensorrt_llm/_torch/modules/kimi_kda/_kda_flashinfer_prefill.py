# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FlashInfer KDA prefill for the Kimi K3 KDA mixer.

``TLLM_KDA_FLASHINFER_PREFILL=1`` routes KDA context (prefill) batches through the
FlashInfer BT16 / Small-BH chunk kernels and the CuTe DSL fused token-major prefill
convolution (``trtllm::kda_prefill_flashinfer`` and ``trtllm::kda_prefill_conv``,
``custom_ops/cute_dsl_kimi_k3_flashinfer_ops.py``). Unset or ``0`` (the default) leaves
the in-tree prefill untouched: the mixer runs its own path, the metadata adds nothing
and no FlashInfer KDA module is imported.

Decode and multi-token verification are shared and unchanged. The FlashInfer prefill
reads and writes the same value-first ``[slots, H, V, K]`` recurrent pool and the same
``[slots, 3D, W - 1]`` convolution pool as the in-tree prefill and decode.
"""

from __future__ import annotations

import importlib
import math
import os
from types import ModuleType
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

from ..mamba.causal_conv1d import causal_conv1d_fn
from ..mamba.recurrent_state_cache import reset_recurrent_state_rows
from ._kda_flashinfer_schedule import FLASHINFER_PREFILL_ENV, flashinfer_prefill_requested
from ._kda_kernels import get_kda_sm_version, is_kda_optimized_supported

if TYPE_CHECKING:
    from .kimi_kda_mixer import KimiKDALinearAttention

_OPS_MODULE = "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops"


def load_flashinfer_prefill_ops() -> ModuleType:
    """Import the FlashInfer KDA prefill ops module (registers its two ops)."""
    return importlib.import_module(_OPS_MODULE)


def flashinfer_prefill_unavailable_reason(mixer: KimiKDALinearAttention) -> Optional[str]:
    """``None`` when ``mixer`` can run the FlashInfer prefill, else the reason it cannot."""
    if os.environ.get("TLLM_KDA_ENABLE_OPT_PREFILL", "1") != "1":
        return "TLLM_KDA_ENABLE_OPT_PREFILL disables the optimized prefill"
    if mixer.head_k_dim != 128 or mixer.head_dim != 128:
        return f"head dims {mixer.head_k_dim}/{mixer.head_dim} are not 128"
    if mixer.conv_size != 4:
        return f"short convolution width {mixer.conv_size} is not 4"
    bound = mixer.gate_lower_bound
    if bound is None or not math.isfinite(bound) or bound >= 0:
        return f"gate lower bound {bound!r} is not finite and negative"
    if not is_kda_optimized_supported():
        return f"SM{get_kda_sm_version()} is not SM100, SM103 or SM107"
    from ...cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return "the CuTe DSL is not available"
    try:
        load_flashinfer_prefill_ops()
    except ImportError as exc:
        return f"the FlashInfer KDA prefill ops do not import ({exc})"
    return None


def select_flashinfer_prefill(mixer: KimiKDALinearAttention) -> bool:
    """Decide the prefill path of one mixer; log the engagement line once per process.

    Returns ``False`` without importing anything when the flag is unset or ``0``. When the
    flag is ``1`` but the mixer cannot run the FlashInfer prefill, the in-tree prefill
    stays selected and a warning names the reason.
    """
    if not flashinfer_prefill_requested():
        return False
    reason = flashinfer_prefill_unavailable_reason(mixer)
    if reason is not None:
        logger.warning_once(
            f"{FLASHINFER_PREFILL_ENV}=1 ignored: {reason}; Kimi K3 KDA prefill stays on the "
            "in-tree path.",
            key="kimi_k3_kda_flashinfer_prefill_unavailable",
        )
        return False
    logger.info_once(
        "Kimi K3 KDA prefill: FlashInfer BT16/Small-BH chunk kernels with the fused "
        f"token-major convolution ({FLASHINFER_PREFILL_ENV}=1)",
        key="kimi_k3_kda_flashinfer_prefill",
    )
    return True


def kda_layout_qkv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
    *,
    normalize_qk: bool = True,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portable layout conversion for the unfused convolution fallback.

    ``packed`` is the channel-major ``[3 * H * D, T]`` convolution output; the result is
    ``[1, T, H, D]`` Q/K/V. The fused path produces this layout directly in the
    convolution kernel and normalizes Q/K within the recurrent kernel.
    """
    if packed.ndim != 2 or packed.shape[0] != 3 * num_heads * head_dim:
        raise ValueError("Expected packed [3 * H * D, T] convolution output")
    qkv = tuple(
        t.t().reshape(1, packed.shape[1], num_heads, head_dim).contiguous()
        for t in packed.chunk(3, dim=0)
    )
    if normalize_qk:
        q, k, v = qkv

        def normalize(t):
            f = t.float()
            return (f * torch.rsqrt((f * f).sum(-1, keepdim=True) + l2_norm_eps)).to(t.dtype)

        qkv = (normalize(q), normalize(k), v)
    if out is not None:
        if len(out) != 3 or any(
            dst.shape != src.shape
            or dst.dtype != src.dtype
            or dst.device != src.device
            or not dst.is_contiguous()
            for dst, src in zip(out, qkv)
        ):
            raise ValueError("Invalid KDA layout output buffers")
        for dst, src in zip(out, qkv):
            dst.copy_(src)
        return out
    return qkv


def can_use_indexed_flashinfer_prefill(
    *,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    num_sequences: int,
    num_tokens: int,
) -> bool:
    """Return whether the FlashInfer prefill can update this V-first pool directly."""
    if num_tokens == 0:
        return False
    if (
        cu_seqlens is None
        or cu_seqlens.dtype != torch.int64
        or cu_seqlens.shape != (num_sequences + 1,)
        or not cu_seqlens.is_contiguous()
    ):
        return False
    if state_pool.dtype not in (torch.bfloat16, torch.float32) or state_pool.ndim != 4:
        return False
    _, H, V, K = state_pool.shape
    if state_pool.stride()[1:] != (V * K, K, 1):
        return False
    if (
        (V, K) != (128, 128)
        or state_pool.stride(0) < H * V * K
        or state_pool.stride(0) * state_pool.element_size() % 16
    ):
        return False
    if state_pool.data_ptr() % 16:
        return False
    if (
        state_indices.ndim != 1
        or state_indices.shape[0] != num_sequences
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
    ):
        return False
    if (
        has_initial_states.ndim != 1
        or has_initial_states.shape[0] != num_sequences
        or has_initial_states.dtype != torch.bool
        or not has_initial_states.is_contiguous()
    ):
        return False
    if state_pool.device != state_indices.device or state_pool.device != has_initial_states.device:
        return False
    return True


def _project_conv_input(
    mixer: KimiKDALinearAttention, x: torch.Tensor, x2d: torch.Tensor
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Token-major ``[T, 3D or 4D]`` q|k|v(|g) for the fused conv, else the packed ``[3D, T]``."""
    if not mixer._kda_fi_fuse_conv:
        return mixer._project_packed_conv_input(x, x2d)
    d = mixer.proj_size
    if mixer._qkvg_proj_weight is not None:
        qkvg = torch.nn.functional.linear(x, mixer._qkvg_proj_weight)
    elif mixer.qkvg_proj is not None:
        qkvg = mixer.qkvg_proj(x)
    else:
        qkvg = torch.cat((mixer.q_proj(x), mixer.k_proj(x), mixer.v_proj(x)), dim=-1)
    onorm_g = qkvg[..., 3 * d : 4 * d] if qkvg.shape[-1] == 4 * d else None
    return qkvg.squeeze(0), onorm_g


def _convolve(
    mixer: KimiKDALinearAttention,
    conv_input: torch.Tensor,
    slot_indices: torch.Tensor,
    has_init: torch.Tensor,
    conv_pool: torch.Tensor,
    mamba_metadata,
    num_prefills: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convolve, update the live convolution cache and return raw ``[1, T, H, D]`` Q/K/V."""
    assert mixer._packed_conv_weight is not None
    query_start_loc = mamba_metadata.query_start_loc[: num_prefills + 1]
    if mixer._kda_fi_fuse_conv:
        return torch.ops.trtllm.kda_prefill_conv(
            conv_input,
            mixer._packed_conv_weight,
            conv_pool,
            query_start_loc,
            slot_indices,
            has_init,
            getattr(mamba_metadata, "kda_max_seqlen", conv_input.shape[0]),
        )
    packed = causal_conv1d_fn(
        conv_input,
        mixer._packed_conv_weight,
        query_start_loc=query_start_loc,
        cache_indices=mamba_metadata.state_indices[:num_prefills],
        has_initial_state=has_init,
        conv_states=conv_pool,
        activation="silu",
    )
    return kda_layout_qkv(packed, mixer.num_heads, mixer.head_dim, normalize_qk=False)


def forward_prefill_flashinfer(
    mixer: KimiKDALinearAttention,
    x2d: torch.Tensor,
    cu_seqlens: torch.Tensor,
    mamba_metadata,
    num_prefills: int,
    conv_pool: torch.Tensor,
    ssm_pool: torch.Tensor,
    slot_indices: torch.Tensor,
    layer_cache=None,
) -> torch.Tensor:
    """``KimiKDALinearAttention.forward_prefill`` on the FlashInfer prefill.

    Same contract as the in-tree path: packed context tokens in, the gated o_proj output
    out; the selected rows of the recurrent and convolution pools hold each request's
    state after its last token.
    """
    from einops import rearrange

    if x2d.shape[0] == 0:
        return x2d.new_empty((0, mixer.hidden_size))
    x = x2d.unsqueeze(0)  # [1, T, hidden]
    # The metadata owns stable int32 slot indices; never convert per layer.
    slot_indices = mamba_metadata.state_indices[:num_prefills]
    has_init = mamba_metadata.has_initial_states[:num_prefills]

    conv_input, onorm_g = _project_conv_input(mixer, x, x2d)
    if not can_use_indexed_flashinfer_prefill(
        state_pool=ssm_pool,
        state_indices=slot_indices,
        has_initial_states=has_init,
        cu_seqlens=cu_seqlens,
        num_sequences=num_prefills,
        num_tokens=x2d.shape[0],
    ):
        raise ValueError("Unsupported FlashInfer KDA state layout or prepared metadata")
    # Fresh requests start from zero state; the recurrence then updates the rows in place.
    reset_recurrent_state_rows(ssm_pool, slot_indices, has_init)
    q, k, v = _convolve(
        mixer, conv_input, slot_indices, has_init, conv_pool, mamba_metadata, num_prefills
    )

    if mixer._bfa_proj_weight is not None:
        bfa = torch.nn.functional.linear(x, mixer._bfa_proj_weight)
        f_a = bfa[..., : mixer.head_dim]
        beta = bfa[..., mixer.head_dim : mixer.head_dim + mixer.num_heads]
        g = mixer.f_b_proj(f_a)
    else:
        g = mixer.f_b_proj(mixer.f_a_proj(x))
        beta = mixer.b_proj(x)
    g = rearrange(g, "... (h d) -> ... h d", d=mixer.head_dim)

    a_log = mixer._A_log_f32 if mixer._A_log_f32 is not None else mixer.A_log
    dt_bias = mixer._dt_bias_f32 if mixer._dt_bias_f32 is not None else mixer.dt_bias
    o = torch.ops.trtllm.kda_prefill_flashinfer(
        q,
        k,
        v,
        g,
        beta.to(torch.bfloat16),
        ssm_pool,
        slot_indices,
        cu_seqlens,
        a_log.float().contiguous(),
        dt_bias.float().contiguous(),
        mixer.head_k_dim**-0.5,
        mixer.gate_lower_bound,
        mixer._kda_fi_mode,
        getattr(mamba_metadata, "kda_max_seqlen", 0),
        getattr(mamba_metadata, "kda_seq_order", None),
        getattr(mamba_metadata, "kda_cu_chunks", None),
    )
    # Fused-verify replay caches: seed the committed conv window so the first verify
    # round convolves the correct history.
    mixer._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
    return mixer._output_gate_and_proj(x, o, onorm_g)
