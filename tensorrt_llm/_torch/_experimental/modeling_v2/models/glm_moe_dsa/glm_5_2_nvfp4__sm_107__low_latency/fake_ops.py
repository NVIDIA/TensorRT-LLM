# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for this target's C++ Torch operators."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


@torch.library.register_fake(f"{_NS}::glm_lowm_bf16_moe")
def _glm_lowm_bf16_moe(
    x, router_logits, e_score_correction_bias, w3_w1, w2, top_k, routed_scaling_factor
):
    return x.new_empty((x.shape[0], w2.shape[1]))


@torch.library.register_fake(f"{_NS}::glm_verify_head_accept")
def _glm_verify_head_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    accepted_tokens: Optional[torch.Tensor],
    force_pool: Optional[torch.Tensor],
    force_counter: Optional[torch.Tensor],
    force_mode: int,
    force_base: int,
    force_max: int,
    force_frac: float,
    counter_stride: int,
    slot_stride: int,
    pool_size: int,
):
    return draft_tokens.new_empty((num_contexts + draft_tokens.shape[0],))


@torch.library.register_fake(f"{_NS}::glm_draft_sample_glue")
def _glm_draft_sample_glue(
    logits,
    exchange_buffer,
    exchange_flags,
    group_size,
    draft_tokens,
    draft_step,
    embed_table,
    hidden,
    enorm_weight,
    hnorm_weight,
    eps,
    col_lo,
    col_hi,
    glue_out,
    positions_in,
    positions_out,
    gather_ids,
    seq_lens,
    fill_seq_lens,
    kv_lens,
    kv_update,
    rewind_accepted,
    draft_len,
    block_offsets,
    dsa_layout,
    req_idx_per_token,
    slot_mapping_fp8,
    slot_mapping_scale,
    gen_kv_indptr,
    gen_cached_token_indptr,
    accepted_tokens,
    batch_indices,
    num_accepted,
    next_new_tokens,
):
    return None


@torch.library.register_fake(f"{_NS}::glm_draft_ehproj_table")
def _glm_draft_ehproj_table(
    logits,
    exchange_buffer,
    exchange_flags,
    group_size,
    draft_tokens,
    draft_step,
    hidden_half,
    token_table,
    norm_weight,
    eps,
    normed_out,
    residual_out,
    positions_in,
    positions_out,
    gather_ids,
    seq_lens,
    fill_seq_lens,
    kv_lens,
    kv_update,
    rewind_accepted,
    draft_len,
    block_offsets,
    dsa_layout,
    req_idx_per_token,
    slot_mapping_fp8,
    slot_mapping_scale,
    gen_kv_indptr,
    gen_cached_token_indptr,
):
    return None


@torch.library.register_fake(f"{_NS}::glm_fused_embedding")
def _glm_fused_embedding(input: torch.Tensor, weight: torch.Tensor):
    return weight.new_empty((*input.shape, weight.shape[1]))


@torch.library.register_fake(f"{_NS}::glm_dense_mlp_small_m")
def _glm_dense_mlp_small_m(x, gate_up_weight, down_weight, gate_up_tiling=-1, down_tiling=-1):
    return x.new_empty((x.shape[0], down_weight.shape[0]))


@torch.library.register_fake(f"{_NS}::glm_fused_a_gemm")
def _glm_fused_a_gemm(hidden_states, weight, l2_prefetch):
    return hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]))


@torch.library.register_fake(f"{_NS}::glm_router_gemm")
def _glm_router_gemm(hidden_states, weight):
    return hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]), dtype=torch.float32)


@torch.library.register_fake(f"{_NS}::glm_mnnvl_hop")
def _glm_mnnvl_hop(input, residual, gamma, epsilon, comm_buffer, buffer_flags, shard_published):
    return [torch.empty_like(residual), torch.empty_like(residual)]


@torch.library.register_fake(f"{_NS}::glm_mnnvl_moe_finalize_hop")
def _glm_mnnvl_moe_finalize_hop(
    gemm2_output,
    expert_weights,
    expanded_idx_to_permuted_idx,
    addend,
    residual,
    gamma,
    epsilon,
    comm_buffer,
    buffer_flags,
):
    return [torch.empty_like(residual), torch.empty_like(residual)]


@torch.library.register_fake(f"{_NS}::glm_mnnvl_allgather")
def _glm_mnnvl_allgather(input, comm_buffer, buffer_flags, group_size):
    return input.new_empty((input.shape[0], input.shape[1] * group_size), dtype=torch.float32)


@torch.library.register_fake(f"{_NS}::glm_sparse_mla_flash_decode")
def _glm_sparse_mla_flash_decode(
    q,
    kv_pool,
    indices,
    seq_lens_kv,
    mla_bmm1_scale,
    mla_bmm2_scale,
    v_b_proj,
    output,
    seq_len_q,
    top_k,
    block_table=None,
    tokens_per_block=0,
    page_index_scale=0,
    layer_offset=0,
    gather_before_wait=False,
):
    return None


@torch.library.register_fake(f"{_NS}::glm_sparse_mla_flash_decode_latent")
def _glm_sparse_mla_flash_decode_latent(
    q,
    kv_pool,
    indices,
    seq_lens_kv,
    mla_bmm1_scale,
    mla_bmm2_scale,
    output,
    seq_len_q,
    top_k,
    block_table=None,
    tokens_per_block=0,
    page_index_scale=0,
    layer_offset=0,
    gather_before_wait=False,
    trigger_after_wait=False,
):
    return None


@torch.library.register_fake(f"{_NS}::glm_intra_node_reduce")
def _glm_intra_node_reduce(
    input,
    residual,
    gamma,
    epsilon,
    table,
    flags,
    layout,
    rank,
    nranks,
    buffer_bytes,
    trigger_completion_at_end,
):
    return torch.empty_like(residual), torch.empty_like(residual)


@torch.library.register_fake(f"{_NS}::glm_intra_node_moe_finalize")
def _glm_intra_node_moe_finalize(
    fc2,
    weights,
    indices,
    shared,
    residual,
    gamma,
    epsilon,
    table,
    flags,
    layout,
    rank,
    nranks,
    buffer_bytes,
):
    return torch.empty_like(residual), torch.empty_like(residual)


@torch.library.register_fake(f"{_NS}::glm_pre_moe_fp4_linear")
def _glm_pre_moe_fp4_linear(
    input, residual, gamma, epsilon, scale, table, flags, layout, rank, nranks, buffer_bytes
):
    rows, hidden = residual.shape
    return (
        torch.empty_like(residual),
        torch.empty_like(residual),
        residual.new_empty((rows, hidden // 2), dtype=torch.uint8),
        residual.new_empty((rows, hidden // 16), dtype=torch.uint8),
    )


@torch.library.register_fake(f"{_NS}::glm_verify_head_sample_tokens")
def _glm_verify_head_sample_tokens(
    logits, temperatures, top_ks, top_ps, min_ps, seed, offset, row_counters, workspace
):
    return logits.new_empty((logits.shape[0],), dtype=torch.int32)


@torch.library.register_fake(f"{_NS}::glm_mla_kv_half_append")
def _glm_mla_kv_half_append(
    latent,
    kv_norm_weight,
    kv_norm_eps,
    rotary_cos_sin,
    sequence_length,
    num_contexts,
    seq_len_q,
    kv_cache_block_offsets,
    host_kv_cache_pool_pointers,
    host_kv_cache_pool_mapping,
    layer_idx,
    tokens_per_block,
    attention_window_size,
    mla_bmm1_scale,
    mla_bmm2_scale,
    softmax_scale,
    trigger_at_entry=False,
):
    return None


@torch.library.register_fake(f"{_NS}::glm_dsa_hist_topk")
def _glm_dsa_hist_topk(logits, kv_lens, out, comm_buffer, state, next_n, group_size):
    return None


@torch.library.register_fake(f"{_NS}::glm_dsa_hist_shard_meta")
def _glm_dsa_hist_shard_meta(
    kv_lens, block_table, shard_ctx, shard_block_table, group_size, rank, tokens_per_block
):
    return None


def _host_query(qualname: str):
    """Setup queries have no tensors and must run outside tracing."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{qualname} is a setup-time query; call it outside tracing")

    return torch.library.register_fake(qualname)(fake)


_host_query(f"{_NS}::glm_dense_mlp_small_m_supported")
_host_query(f"{_NS}::glm_intra_node_reduce_supported")
_host_query(f"{_NS}::glm_mnnvl_hop_supported")


@torch.library.register_fake(f"{_NS}::glm_megakernel_indexer_tmap")
def _glm_megakernel_indexer_tmap(pool: torch.Tensor, rows: int):
    return torch.empty((128,), dtype=torch.uint8, device="cpu")


@torch.library.register_fake(f"{_NS}::glm_megakernel_weight_table")
def _glm_megakernel_weight_table(params: torch.Tensor, offset: int):
    return None


@torch.library.register_fake(f"{_NS}::glm_megakernel")
def _glm_megakernel(params, x, residual, topk, block_table, kv_lens, state, kv_cache, pdl):
    return torch.empty_like(x), torch.empty_like(residual)
