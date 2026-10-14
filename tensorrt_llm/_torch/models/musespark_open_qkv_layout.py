# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4/TP8 plain-BF16 QKV row layout with separate native normalization."""

import torch


@torch.no_grad()
def prepare_weight_layout(attention):
    """Called after native QKV loading/sharding, before derived weight state."""
    from ..modules.linear import UnquantizedLinearMethod

    linear = attention.qkv_proj
    tp = linear.mapping.tp_size
    q_width = attention.num_heads * attention.head_dim
    kv_width = attention.num_key_value_heads * attention.head_dim
    if not (
        type(linear.quant_method) is UnquantizedLinearMethod
        and tp in (4, 8)
        and linear.weight.dtype == torch.bfloat16
        and linear.weight.shape == (2 * (q_width + kv_width), 8192)
        and linear.weight.is_contiguous()
        and linear.bias is None
        and attention.num_heads == 128 // tp
        and attention.num_key_value_heads == 16 // tp
        and attention.head_dim == 64
        and not attention.fuse_qk_norm_rope
        and not attention._fuse_qk_norm_rope_gate
        and not attention.q_norm.use_gemma
        and not attention.k_norm.use_gemma
    ):
        raise ValueError("QKV weight layout requires plain BF16 TP4/TP8 MuseSpark attention")
    device = linear.weight.device
    interleaved = torch.arange(2 * q_width, device=device).reshape(
        attention.num_heads, 2 * attention.head_dim
    )
    permutation = torch.cat(
        (
            interleaved[:, : attention.head_dim].reshape(-1),
            torch.arange(2 * q_width, 2 * (q_width + kv_width), device=device),
            interleaved[:, attention.head_dim :].reshape(-1),
        )
    )
    inverse = torch.argsort(permutation)
    linear.weight.copy_(linear.weight.index_select(0, permutation))
    return inverse


def preprocess_one_row(attention, projection, position_ids):
    """Consume owned GEMM output; preserve native calls and BF16 rounding."""
    from flashinfer.norm import rmsnorm

    from ..flashinfer_utils import get_env_enable_pdl
    from ..modules.multi_stream_utils import maybe_execute_in_parallel

    q_width, kv_width = attention.q_size, attention.kv_size
    q, k = projection[:, :q_width], projection[:, q_width : q_width + kv_width]

    def normalize_q():
        return rmsnorm(
            q.view(attention.num_heads, attention.head_dim),
            attention.q_norm.weight,
            attention.q_norm.variance_epsilon,
            out=q.view(attention.num_heads, attention.head_dim),
            enable_pdl=get_env_enable_pdl(),
        )

    def normalize_k():
        return rmsnorm(
            k.view(attention.num_key_value_heads, attention.head_dim),
            attention.k_norm.weight,
            attention.k_norm.variance_epsilon,
            out=k.view(attention.num_key_value_heads, attention.head_dim),
            enable_pdl=get_env_enable_pdl(),
        )

    maybe_execute_in_parallel(
        normalize_q,
        normalize_k,
        attention.ln_events[0],
        attention.ln_events[1],
        attention.aux_stream,
        disable_on_compile=True,
    )
    torch.mul(q, attention.scale_query_by, out=q)
    if not attention.skip_rope and position_ids is not None:
        attention.rotary_emb(position_ids, [q, k])
    qkv_width = q_width + 2 * kv_width
    return projection[:, :qkv_width], None, None, projection[:, qkv_width:]
