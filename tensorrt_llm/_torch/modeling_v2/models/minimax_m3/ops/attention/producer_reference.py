# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared unchanged producer references and test-only input validation."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch


class ProducerTarget(Protocol):
    HEAD_DIM: int
    # Multi-KV geometry fields are read only by the TP2 reference variant.
    HEADS_PACKED: int
    HEADS_Q: int
    HEADS_KV: int
    HEADS_INDEX: int
    FP4_BLOCK: int
    E2M1_VALUES: tuple[float, ...]
    _norm_rope: Callable[..., torch.Tensor]
    PAGE: int


def _norm_rope(
    state: ProducerTarget,
    /,
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
):
    """Gemma RMSNorm (``x * rsqrt(mean(x^2) + eps) * (1 + w)``) then NeoX RoPE on the first 64 dims, in fp32.
    x [T, 128] fp32; cos / sin [T, 32]."""
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.float())
    lo, hi, rest = (y[:, :32], y[:, 32:64], y[:, 64:])
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin, rest], dim=-1)


def decode_nvfp4_token(
    state: ProducerTarget, /, data: torch.Tensor, scales: torch.Tensor, quant_scale: float
) -> torch.Tensor:
    """fp32 [128] of one token's packed E2M1 row (64 bytes, low nibble first) and its 8 E4M3 block scales."""
    table = torch.tensor(state.E2M1_VALUES, dtype=torch.float32, device=data.device)
    nib = torch.stack([data & 15, data >> 4], dim=-1).reshape(-1).long()
    vals = table[nib & 7] * torch.where(nib >= 8, -1.0, 1.0)
    s = scales.view(torch.float8_e4m3fn).float() / quant_scale
    return vals * s.repeat_interleave(state.FP4_BLOCK)


def scale_index(state: ProducerTarget, /, within_page: int, plane: int) -> torch.Tensor:
    """Positions of a token's 8 block scales in its page's [128 * 8] scale plane: row-major for K, the 4-token
    swizzle for V."""
    cols = torch.arange(8)
    if plane == 0:
        return within_page * 8 + cols
    return within_page // 4 * 32 + cols * 4 + within_page % 4


def producer_reference_single_kv(
    self: Any,
    state: ProducerTarget,
    /,
    packed: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    heads_q: int,
    pdl: bool = True,
    early: bool = True,
    prewait_step_inputs: bool = False,
) -> dict[str, torch.Tensor]:
    """fp32 values before their quantization: "q" [T, heads_q, 128], "index_q" [T, 128], and per token "k", "v",
    "index_k" [T, 128] (the values written to the token's slot). The caches, ``pdl`` and ``early`` do not change
    them."""
    t = packed.shape[0]
    x = packed.float().view(t, heads_q + 4, state.HEAD_DIM)
    rope = rotary_cos_sin[position_ids[:t].long()]
    cos, sin = (rope[:, 0], rope[:, 1])
    q = torch.stack(
        [state._norm_rope(x[:, h], q_weight, cos, sin, eps) for h in range(heads_q)], dim=1
    )
    return {
        "q": q,
        "k": state._norm_rope(x[:, heads_q], k_weight, cos, sin, eps),
        "v": x[:, heads_q + 1],
        "index_q": state._norm_rope(x[:, heads_q + 2], index_q_weight, cos, sin, eps),
        "index_k": state._norm_rope(x[:, heads_q + 3], index_k_weight, cos, sin, eps),
    }


def producer_reference_multi_kv(
    self: Any,
    state: ProducerTarget,
    /,
    packed: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    heads_q: int,
    pdl: bool = True,
    early: bool = True,
    prewait_step_inputs: bool = False,
) -> dict[str, torch.Tensor]:
    """fp32 values before their quantization: "q" [T, 32, 128], "index_q" [T, 2, 128], per token and KV head "k"
    and "v" [T, 2, 128], and "index_k" [T, 128] (the values written to the token's slot). The caches, ``pdl``,
    ``early`` and ``prewait_step_inputs`` do not change them."""
    t = packed.shape[0]
    x = packed.float().view(t, state.HEADS_PACKED, state.HEAD_DIM)
    rope = rotary_cos_sin[position_ids[:t].long()]
    cos, sin = (rope[:, 0], rope[:, 1])

    def heads(first, count, weight):
        return torch.stack(
            [state._norm_rope(x[:, h], weight, cos, sin, eps) for h in range(first, first + count)],
            dim=1,
        )

    k0 = state.HEADS_Q
    v0 = k0 + state.HEADS_KV
    iq0 = v0 + state.HEADS_KV
    return {
        "q": heads(0, state.HEADS_Q, q_weight),
        "k": heads(k0, state.HEADS_KV, k_weight),
        "v": x[:, v0 : v0 + state.HEADS_KV],
        "index_q": heads(iq0, state.HEADS_INDEX, index_q_weight),
        "index_k": state._norm_rope(x[:, iq0 + state.HEADS_INDEX], index_k_weight, cos, sin, eps),
    }


def validate_producer(
    self: Any,
    state: ProducerTarget,
    /,
    packed: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    heads_q: int,
    pdl: bool = True,
    early: bool = True,
    prewait_step_inputs: bool = False,
) -> None:
    t = packed.shape[0]
    pos = position_ids[:t]
    assert bool(((pos >= 0) & (pos < rotary_cos_sin.shape[0])).all()), (
        "position_ids must index rotary_cos_sin"
    )
    slots = out_cache_loc[:t]
    live = slots[(slots >= 0) & (slots < kv_data.shape[0] * state.PAGE)]
    assert live.unique().numel() == live.numel(), "out_cache_loc must not repeat a slot"
