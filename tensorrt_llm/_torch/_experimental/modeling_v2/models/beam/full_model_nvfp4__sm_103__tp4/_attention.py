# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention sublayer of the Beam decoder block components, one request step.

Shared by the dense and sparse block components. It holds no parameters, cache or
configuration: each block passes its own tensors and geometry, its layer's position
encoding and window, and the prepared attention metadata of the call. The computation,
before the block's post-attention norm::

    q, k, v, g = h @ q_proj^T, h @ k_proj^T, h @ v_proj^T, h @ gate_proj^T
    q, k = rmsnorm(q, q_norm), rmsnorm(k, k_norm)          per head, rounded to bf16
    sliding layer: q, k = rope(q, k)                        fp32 NeoX rotation, rounded
    full layer:    q = q * exp(0.1 * log1p((p + 1) / 2048)) fp32 scale, rounded; k as is
    a = causal attention(q, k, v), scale 1/sqrt(128), 6 q heads per kv head,
        window 512 on a sliding layer, the whole context on a full one
    a = a * sigmoid(g + gate_bias)                          per head, bf16 steps
    out = a @ o_proj^T

A full-attention layer has no position embedding: neither q nor k is rotated, and its
only position dependence is the per-token query scale above.

A step of one request: a context call (``num_contexts == 1``) appends its T new tokens
after the ``past`` tokens the cache already holds for the request -- none for a fresh
context, a reused or earlier-computed prefix otherwise -- and a generation call
(``num_contexts == 0``) appends exactly one token. Every position is absolute: the new
tokens sit at ``past .. past + T - 1``, where ``past + T`` is the request's KV length in
the metadata, and the positions the block receives must be exactly those. A context call
over a cached prefix must use paged-context FMHA, the path that reads the prefix's keys and
values from the pool; without it the context kernel would attend to the new rows only.
The keys and values the op appends are the transformed k (normalized, rotated on a sliding
layer) and the projected v of the new tokens, written to the request's own pages.

A step of several requests (``check_batch_metadata``) is the native runtime's batch: its
context requests first, then its generation requests, each with its own ``past``, KV length
and pages, their new tokens concatenated in that order. Every operation but the attention
op is per token; the op reads each request's own metadata row.

Logical layer and cache slot. A block's ``layer_idx`` is its depth in the model: it fixes
the block's semantics (its residual coefficients, its attention and feed-forward kind).
The KV cache it appends to is a ``KVCacheManagerV2`` that holds that layer, so its storage
slot is the manager's local index for it -- ``layer_offsets[layer_idx]``: 0 for the one
layer a ``layer_mask`` selects, the layer's own position in a manager holding several.
The attention op and the manager's per-layer arrays take that local index; manager APIs
such as ``get_buffers`` take the logical one and translate it. The slot is resolved from
each call's metadata, never stored, and a manager that does not hold the block's logical
layer is rejected before any kernel runs.

Host synchronization: the one-request position check (``check_step``) compares the
positions with ``arange(past, past + T)`` on the device and reads the answer back, once per
call. The metadata checks read host tensors only.
"""

from __future__ import annotations

from typing import Mapping

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_neox_rope import (
    beam_neox_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_nope_query_scale import (  # noqa: E501
    beam_nope_query_scale,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_output_gate import (
    beam_output_gate,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    STANDARD,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.bindings import DataType

#: thop_attention arguments that do not change with the step on these blocks: packed q|k|v,
#: append-and-attend, causal, bf16 pool, no quantization, and any position transform
#: applied before the call -- the rope_* values are inert placeholders the op requires,
#: not this model's RoPE.
CALL_CONSTANTS = dict(
    k=None,
    v=None,
    output_sf=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=STANDARD,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)


def local_layer_index(md: TrtllmAttentionMetadata, layer_idx: int) -> int:
    """The cache slot of logical layer ``layer_idx`` in the call's cache manager.

    Raises ValueError unless the manager holds ``layer_idx`` at an integer local index
    within its local layers. Reads the manager's host-side layer map only.
    """
    mgr = md.kv_cache_manager
    offsets = getattr(mgr, "layer_offsets", None)
    if not isinstance(offsets, Mapping) or layer_idx not in offsets:
        held = sorted(offsets) if isinstance(offsets, Mapping) else None
        raise ValueError(
            f"the KV cache manager does not hold logical layer {layer_idx}; it holds {held}. "
            f"Select the layer with a layer_mask rather than reusing another layer's cache"
        )
    num_local = getattr(mgr, "num_local_layers", None)
    local = offsets[layer_idx]
    if (
        type(num_local) is not int
        or type(local) is not int
        or len(offsets) != num_local
        or not 0 <= local < num_local
    ):
        raise ValueError(
            f"the KV cache manager maps logical layer {layer_idx} to slot {local!r} of "
            f"{num_local!r} local layers holding {len(offsets)} logical layers"
        )
    return local


def check_step(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    md: TrtllmAttentionMetadata,
    *,
    layer_idx: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    max_tokens: int,
) -> None:
    """Raise ValueError on any call input a block would otherwise compute wrongly.

    Accepted: ``hidden_states`` contiguous ``[1, T, hidden_size]`` bf16 on ``device``,
    1 <= T <= ``max_tokens``; integer ``position_ids`` ``[1, T]`` equal to
    ``arange(past, past + T)`` on that device; a prepared ``TrtllmAttentionMetadata``
    holding one step of one request on a BF16 ``KVCacheManagerV2`` that holds logical
    layer ``layer_idx`` with ``num_kv_heads`` kv heads of ``head_dim`` (``check_metadata``
    states the steps and returns ``past``).
    """
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be a torch.bfloat16 tensor")
    if (
        hidden_states.dim() != 3
        or hidden_states.shape[0] != 1
        or hidden_states.shape[2] != hidden_size
    ):
        raise ValueError(
            f"hidden_states must be [1, T, {hidden_size}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.device != device or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous on the block's device")
    num_tokens = hidden_states.shape[1]
    if not 1 <= num_tokens <= max_tokens:
        raise ValueError(f"supported token counts per call are 1..{max_tokens}, got {num_tokens}")
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.is_floating_point()
        or tuple(position_ids.shape) != (1, num_tokens)
        or position_ids.device != hidden_states.device
    ):
        raise ValueError(
            f"position_ids must be an integer [1, {num_tokens}] tensor on the block's device"
        )
    past = check_metadata(
        md, num_tokens, layer_idx=layer_idx, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    expected = torch.arange(
        past, past + num_tokens, dtype=position_ids.dtype, device=position_ids.device
    )
    if not torch.equal(position_ids[0], expected):
        raise ValueError(
            f"the call appends tokens {past}..{past + num_tokens - 1} after {past} cached "
            "tokens; its positions must be exactly those"
        )


def check_metadata(
    md: TrtllmAttentionMetadata,
    num_tokens: int,
    *,
    layer_idx: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Raise ValueError unless ``md`` is one prepared step of one request; return ``past``.

    ``check_batch_metadata`` restricted to a batch of exactly one request: a context step of
    ``num_tokens`` new tokens after ``past`` cached ones, or a generation step of one token.
    ``past`` is the absolute position of the step's first new token. Reads host-side state
    only.
    """
    spans = check_batch_metadata(
        md, num_tokens, layer_idx=layer_idx, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    if len(spans) != 1:
        raise ValueError(f"one request per call is supported; metadata has {len(spans)}")
    return spans[0][0]


def check_batch_metadata(
    md: TrtllmAttentionMetadata,
    num_tokens: int,
    *,
    layer_idx: int,
    num_kv_heads: int,
    head_dim: int,
) -> list:
    """Raise ValueError unless ``md`` is one prepared step of its requests; return their spans.

    The native runtime orders a step's ``num_seqs`` requests with its ``num_contexts``
    context requests first, and their new tokens fill the call's ``num_tokens`` rows in that
    order. Accepted on a BF16 ``KVCacheManagerV2`` holding logical layer ``layer_idx`` with
    ``num_kv_heads`` kv heads of ``head_dim``, beam width 1, no cross attention or
    speculative decoding, where each request ``i`` with KV length ``L_i`` (at most the
    manager's ``max_seq_len``) is

    * a context request of ``n_i >= 1`` new tokens: context length ``n_i``,
      ``past_i = L_i - n_i >= 0`` tokens already cached, and paged-context FMHA whenever
      ``past_i > 0``; or
    * a generation request: one new token after ``past_i = L_i - 1 >= 0`` cached tokens
      (the native runtime's warmup issues generation steps over a cache that holds only the
      new token).

    Returns ``[(past_i, n_i), ...]`` in row order: request ``i``'s new tokens sit at the
    absolute positions ``past_i .. past_i + n_i - 1``. Reads host-side state only.
    """
    if not isinstance(md, TrtllmAttentionMetadata):
        raise ValueError(f"attn_metadata must be TrtllmAttentionMetadata, got {type(md).__name__}")
    mgr = md.kv_cache_manager
    if mgr is None:
        raise ValueError("attn_metadata needs a KV cache manager for the paged append")
    if md.is_cross or md.is_spec_decoding_enabled or md.beam_width != 1:
        raise ValueError("cross attention, speculative decoding and beams are not supported")
    num_seqs, num_contexts = md.num_seqs, md.num_contexts
    if num_seqs < 1 or not 0 <= num_contexts <= num_seqs:
        raise ValueError(
            f"the metadata holds {num_seqs} requests of which {num_contexts} are contexts"
        )
    if md.kv_cache_params is None or md.kv_cache_params.num_extra_kv_tokens:
        raise ValueError("the metadata must use the KV cache without extra KV tokens")
    if num_tokens > md.max_num_tokens:
        raise ValueError(
            f"the call has {num_tokens} tokens; the metadata holds {md.max_num_tokens} per call"
        )
    new_tokens = [int(n) for n in md.seq_lens[:num_seqs]]
    kv_lens = [int(n) for n in md.kv_lens_runtime[:num_seqs]]
    request_types = [int(t) for t in md.host_request_types_runtime[:num_seqs]]
    context_lens = [int(n) for n in md.prompt_lens_cpu_runtime[:num_contexts]]
    if sum(new_tokens) != num_tokens or md.num_ctx_tokens != sum(new_tokens[:num_contexts]):
        raise ValueError(
            f"the requests' new tokens {new_tokens} ({md.num_ctx_tokens} in contexts) do not "
            f"fill the call's {num_tokens} tokens"
        )
    spans = []
    for i, (n, kv_len, request_type) in enumerate(zip(new_tokens, kv_lens, request_types)):
        past = kv_len - n
        if i < num_contexts:
            if n < 1 or context_lens[i] != n or request_type != 0:
                raise ValueError(
                    f"context request {i} of {n} new tokens needs context length {n}; "
                    f"metadata has context length {context_lens[i]} and request type "
                    f"{request_type}"
                )
            if past < 0:
                raise ValueError(
                    f"request {i}'s KV length {kv_len} is shorter than its {n} new tokens"
                )
            if past and not md.use_paged_context_fmha:
                raise ValueError(
                    f"a context step after {past} cached tokens must read them from the pool: "
                    "paged-context FMHA is required"
                )
        else:
            if n != 1 or request_type != 1:
                raise ValueError(
                    f"generation request {i} appends exactly one token, got {n} (request type "
                    f"{request_type})"
                )
            if past < 0:
                raise ValueError(
                    f"request {i}'s KV length {kv_len} is shorter than its generation token"
                )
        spans.append((past, n))
    if max(kv_lens) > mgr.max_seq_len or md.max_seq_len != mgr.max_seq_len:
        raise ValueError(
            f"a request reaches {max(kv_lens)} tokens; the cache holds {mgr.max_seq_len} per "
            f"sequence and the metadata states {md.max_seq_len}"
        )
    local = local_layer_index(md, layer_idx)
    heads = getattr(mgr, "num_kv_heads_per_layer", None) or []
    local_heads = heads[local] if len(heads) == mgr.num_local_layers else None
    if mgr.dtype != DataType.BF16 or local_heads != num_kv_heads or mgr.head_dim != head_dim:
        raise ValueError(
            f"the KV cache must be BF16 with {num_kv_heads} kv heads of {head_dim}; "
            f"got {mgr.dtype}, {heads}, head_dim {mgr.head_dim}"
        )
    return spans


def step_args(md: TrtllmAttentionMetadata, layer_idx: int) -> dict:
    """Project the prepared metadata onto thop_attention's explicit batch state.

    Includes ``local_layer_idx``, the cache slot of logical layer ``layer_idx`` resolved
    from this call's manager.
    """
    return dict(
        local_layer_idx=local_layer_index(md, layer_idx),
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        workspace=md.effective_workspace,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        use_paged_context_fmha=md.use_paged_context_fmha,
        beam_width=md.effective_beam_width,
        cache_indirection=md.cache_indirection,
        block_ids_per_seq=md.block_ids_per_seq,
        max_context_q_len_override=md.max_context_q_len_override,
        is_cross=md.is_cross,
        is_spec_decoding_enabled=md.is_spec_decoding_enabled,
        use_spec_decoding=md.use_spec_decoding,
        is_spec_dec_tree=md.is_spec_dec_tree,
        spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
        spec_decoding_position_offsets=md.spec_decoding_position_offsets,
        spec_decoding_packed_mask=md.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
        spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
        spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
        num_sparse_topk=md.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=md.flash_mla_num_splits,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
    )


def attention(
    h: torch.Tensor,
    positions: torch.Tensor,
    step: dict,
    *,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    gate_proj: torch.Tensor,
    o_proj: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    gate_bias: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    rope_theta: float | None,
    window: int,
) -> torch.Tensor:
    """Normalized input [T, hidden] -> attention output [T, hidden], before post_attention_layernorm.

    ``step`` is ``step_args`` of the call's metadata; the projections are ``[out, in]``.
    ``rope_theta`` is the sliding layer's RoPE base; ``None`` is a full-attention layer
    without position embedding, whose q takes the per-token query scale instead. ``window``
    is the attention window the op applies: the sliding window, or the metadata's
    ``max_seq_len`` for a full layer.
    """
    num_tokens = h.shape[0]
    heads, kv_heads, d = num_heads, num_kv_heads, head_dim
    q = cublas_mm(h, q_proj.t())
    k = cublas_mm(h, k_proj.t())
    v = cublas_mm(h, v_proj.t())
    gate = cublas_mm(h, gate_proj.t())
    q = flashinfer_rmsnorm(q.view(num_tokens * heads, d), q_norm, eps)
    k = flashinfer_rmsnorm(k.view(num_tokens * kv_heads, d), k_norm, eps)
    q, k = q.view(num_tokens, heads, d), k.view(num_tokens, kv_heads, d)
    if rope_theta is None:
        q = beam_nope_query_scale(q, positions)
    else:
        q, k = beam_neox_rope(q, k, positions, rope_theta, 1.0)
    qkv = torch.cat((q.view(num_tokens, heads * d), k.view(num_tokens, kv_heads * d), v), dim=-1)
    core = torch.empty(num_tokens, heads * d, dtype=h.dtype, device=h.device)
    core = thop_attention(
        q=qkv,
        output=core,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=d,
        attention_window_size=window,
        **step,
        **CALL_CONSTANTS,
    )
    gated = beam_output_gate(
        core.view(num_tokens, heads, d), gate.view(num_tokens, heads, d), gate_bias
    )
    return cublas_mm(gated.view(num_tokens, heads * d), o_proj.t())
