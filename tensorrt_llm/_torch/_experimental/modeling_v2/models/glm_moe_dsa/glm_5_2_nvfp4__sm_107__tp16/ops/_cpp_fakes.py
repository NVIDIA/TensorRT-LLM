# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for this target's C++ Torch operators."""

from typing import Optional

import torch


@torch.library.register_fake("trtllm::glm_lowm_bf16_moe")
def _glm_lowm_bf16_moe(
    x, router_logits, e_score_correction_bias, w3_w1, w2, top_k, routed_scaling_factor
):
    return x.new_empty((x.shape[0], w2.shape[1]))


@torch.library.register_fake("trtllm::glm_verify_head_accept")
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


@torch.library.register_fake("trtllm::glm_draft_sample_glue")
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


@torch.library.register_fake("trtllm::glm_draft_ehproj_table")
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


@torch.library.register_fake("trtllm::glm_fused_embedding")
def _glm_fused_embedding(input: torch.Tensor, weight: torch.Tensor):
    return weight.new_empty((*input.shape, weight.shape[1]))


@torch.library.register_fake("trtllm::glm_dense_mlp_small_m")
def _glm_dense_mlp_small_m(x, gate_up_weight, down_weight, gate_up_tiling=-1, down_tiling=-1):
    return x.new_empty((x.shape[0], down_weight.shape[0]))


@torch.library.register_fake("trtllm::glm_fused_a_gemm")
def _glm_fused_a_gemm(hidden_states, weight, l2_prefetch):
    return hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]))


@torch.library.register_fake("trtllm::glm_router_gemm")
def _glm_router_gemm(hidden_states, weight):
    return hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]), dtype=torch.float32)


@torch.library.register_fake("trtllm::glm_mnnvl_hop")
def _glm_mnnvl_hop(input, residual, gamma, epsilon, comm_buffer, buffer_flags, shard_published):
    return [torch.empty_like(residual), torch.empty_like(residual)]


@torch.library.register_fake("trtllm::glm_mnnvl_moe_finalize_hop")
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


@torch.library.register_fake("trtllm::glm_mnnvl_allgather")
def _glm_mnnvl_allgather(input, comm_buffer, buffer_flags, group_size):
    return input.new_empty((input.shape[0], input.shape[1] * group_size), dtype=torch.float32)
