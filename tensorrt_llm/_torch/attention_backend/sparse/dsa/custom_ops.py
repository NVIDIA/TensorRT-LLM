# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSA custom ops for piecewise CUDA graph capture.

These ops split the DSA MLA forward into two phases:
- Op 1 (mla_dsa_proj): token-wise projections, CUDA-graph-capturable
- Op 2 (mla_dsa_attn_inplace): batch-dependent attention, NOT captured

They are registered as torch.library custom ops so that
piecewise_optimizer can split the graph at the boundary.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ....modules.attention import extract_extra_attrs


@torch.library.custom_op("trtllm::mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and indexer.pre_indexer_proj.
    Does NOT update the indexer k cache — that happens in Op 2.

    Returns [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
    weights].  Under torch compile MLA._should_use_short_mha returns False
    so the result is always length 8 for straight-line control flow.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")

    if mla_layer.is_lite:
        compressed_kv, k_pe = mla_layer.kv_a_proj_with_mqa(hidden_states).split(
            [mla_layer.kv_lora_rank, mla_layer.qk_rope_head_dim], -1
        )
        compressed_kv = mla_layer.kv_a_layernorm(compressed_kv)
        qr = hidden_states
    else:
        from ....modules.multi_stream_utils import maybe_execute_in_parallel

        qr, compressed_kv, k_pe = mla_layer.kv_a_proj_with_mqa(hidden_states).split(
            [mla_layer.q_lora_rank, mla_layer.kv_lora_rank, mla_layer.qk_rope_head_dim], -1
        )
        qr, compressed_kv = maybe_execute_in_parallel(
            lambda: mla_layer.q_a_layernorm(qr),
            lambda: mla_layer.kv_a_layernorm(compressed_kv),
            mla_layer.ln_events[0],
            mla_layer.ln_events[1],
            mla_layer.aux_stream,
        )

    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
    q = mla_layer.q_b_proj(qr)

    # Run indexer pre-projection (CUDA-graph-safe: pure token-wise compute)
    q_fp8, k_fp8, k_scale, weights = mla_layer.mqa.indexer.pre_indexer_proj(
        qr, hidden_states, position_ids
    )

    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]


@mla_dsa_proj.register_fake
def _mla_dsa_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    # Under torch compile always return 8 tensors for straight-line control flow.
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    num_tokens = hidden_states.shape[0]
    indexer = mla_layer.mqa.indexer
    q = hidden_states.new_empty([num_tokens, mla_layer.num_heads_tp * mla_layer.qk_head_dim])
    compressed_kv = hidden_states.new_empty([num_tokens, mla_layer.kv_lora_rank])
    k_pe = hidden_states.new_empty([num_tokens, mla_layer.qk_rope_head_dim])
    latent_cache = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank + mla_layer.qk_rope_head_dim]
    )
    q_fp8 = hidden_states.new_empty(
        [num_tokens, indexer.n_heads, indexer.head_dim], dtype=torch.float8_e4m3fn
    )
    k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim], dtype=torch.float8_e4m3fn)
    k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.float32)
    weights = hidden_states.new_empty([num_tokens, indexer.n_heads], dtype=torch.float32)
    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]


@torch.library.custom_op("trtllm::mla_dsa_attn_inplace", mutates_args=("output",))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights].
    Runs k-cache scatter, then dispatches context/generation attention.
    This op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")

    num_tokens = metadata.num_tokens
    num_ctx_tokens = metadata.num_ctx_tokens

    # Slice to actual num_tokens (Op 1 operates on full padded tensor)
    q = q[:num_tokens, ...]
    compressed_kv = compressed_kv[:num_tokens, ...]
    k_pe = k_pe[:num_tokens, ...]
    latent_cache = latent_cache[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    q_fp8, k_fp8, k_scale, weights = indexer_intermediates
    q_fp8 = q_fp8[:num_tokens, ...]
    k_fp8 = k_fp8[:num_tokens, ...]
    k_scale = k_scale[:num_tokens, ...]
    weights = weights[:num_tokens, ...]

    # K-cache scatter (batch-dependent, not graph-capturable)
    mla_layer.mqa.indexer._update_k_cache(k_fp8, k_scale, metadata)

    intermediates = {
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "k_scale": k_scale,
        "weights": weights,
    }

    # Context phase
    if metadata.num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        if mla_layer.apply_rotary_emb and position_ids is not None:
            k_pe_ctx = mla_layer.apply_rope(q_ctx, k_pe_ctx, position_ids)
        ctx_kwargs = {k: v[:num_ctx_tokens] for k, v in intermediates.items()}
        mla_layer.forward_context(
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            position_ids,
            metadata,
            output[:num_ctx_tokens, :],
            latent_cache_ctx,
            **ctx_kwargs,
        )

    # Generation phase
    if metadata.num_generations > 0:
        q_gen = q[num_ctx_tokens:, ...]
        compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
        k_pe_gen = k_pe[num_ctx_tokens:, ...]
        latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
        if mla_layer.apply_rotary_emb and position_ids is not None:
            k_pe_gen = mla_layer.apply_rope(q_gen, k_pe_gen, position_ids)
        gen_kwargs = {k: v[num_ctx_tokens:num_tokens] for k, v in intermediates.items()}
        mla_layer.forward_generation(
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            metadata,
            output[num_ctx_tokens:num_tokens, :],
            position_ids=position_ids,
            latent_cache=latent_cache_gen,
            **gen_kwargs,
        )


@mla_dsa_attn_inplace.register_fake
def _mla_dsa_attn_inplace_fake(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    pass  # in-place mutation of output, no return value
