# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the batch arguments shared by every attention layer in this forward."""

from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


def generation_width(num_rows: int, num_requests: int) -> int:
    """Number of consecutive candidates per generation request in a linear draft."""
    if not num_requests and not num_rows:
        return 1
    if num_requests <= 0 or num_rows < num_requests or num_rows % num_requests:
        raise ValueError(f"Cannot split {num_rows} generation rows across {num_requests} requests")
    return num_rows // num_requests


def attention_step_args(md: TrtllmAttentionMetadata, *, n_rows: int, decode: bool = False) -> dict:
    """Build thop_attention arguments, skipping context counts for pure decode."""
    num_contexts = 0 if decode else md.num_contexts
    num_ctx_tokens = 0 if decode else md.num_ctx_tokens
    return dict(
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
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        predicted_tokens_per_seq=generation_width(
            n_rows - num_ctx_tokens, md.num_seqs - num_contexts
        ),
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
        spec_decoding_position_offsets=get_spec_decoding_position_offsets(md),
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
