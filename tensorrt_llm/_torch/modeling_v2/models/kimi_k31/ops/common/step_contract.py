# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine attention-metadata surface the Kimi K3.1 targets read, and the opt-in check of it.

The targets read their step state off the engine's attention metadata (``AttentionMetadata``,
``TrtllmAttentionMetadata`` and the DSA ``DSAtrtllmAttentionMetadata``) in the MLA, DSA indexer, KDA and drafter
code, not through one step-args projection. These are the names they read. Most are private trtllm surface, so a
rename upstream would otherwise surface only mid-forward.

``TRTLLM_MODELING_V2_VALIDATE`` makes each target's prefill and decode paths call :func:`check_step_contract` on
their first forward. It checks :data:`STEP_FIELDS`, the names the metadata class declares or sets when it is
built. :data:`STEP_FIELDS_SET_LATER` are set only by the metadata's own methods (``prepare``, the KV-length
updates, the DSA indexer buffers) and only on the steps that need them, so a first forward may not have them yet.
The sm_107 targets also read the names of :data:`SM107_STEP_FIELDS` and :data:`SM107_STEP_FIELDS_SET_LATER`, which
the check leaves out.
"""

#: Engine attention-metadata names the targets read that the metadata class declares or sets when it is built.
STEP_FIELDS = frozenset(
    """
    _cached_block_table_ctx _cached_block_table_gen _cached_pool_view _cached_req_idx_ctx
    _cached_req_idx_gen _cached_tokens_per_block _ensure_pool_view_cached _helix_spec_tokens_valid _seq_lens
    _seq_lens_cuda _tokens_per_block beam_width block_ids_per_seq cache_indirection cu_kv_seqlens
    cu_q_seqlens cuda_graph_buffers dsl_atom dsl_expand_factor dsl_kernel_atoms dsl_kernel_next_n
    effective_beam_width effective_workspace enable_context_mla_with_cached_kv enable_flash_mla
    flash_mla_num_splits flash_mla_tile_scheduler_metadata force_prepare_spec_dec_tree_mask
    gen_indexer_kv_lens_cuda_runtime get_empty get_indexer_kv_lens get_indexer_max_seq_len
    helix_is_inactive_rank helix_local_slots helix_position_offsets host_kv_cache_pool_mapping
    host_kv_cache_pool_pointers indexer_head_dim indexer_max_chunk_size indexer_prefill_chunks
    indexer_prefill_q_split indexer_quant_block_size is_cross is_cuda_graph is_spec_dec_tree
    is_spec_decoding_enabled kv_cache_block_offsets kv_cache_manager kv_cache_params mamba_metadata mapping
    max_context_length max_context_q_len_override max_draft_tokens max_num_requests max_num_sequences
    max_num_tokens max_seq_len max_total_draft_tokens num_contexts num_ctx_kv_tokens num_ctx_tokens
    num_generations num_seqs num_sms num_sparse_topk num_tokens padded_num_tokens radix_aux_indices
    radix_aux_logits request_ids runtime_features seq_lens shared_topk_indices skip_indexer_for_ctx_reqs
    skip_indexer_for_gen_reqs sparse_metadata_params spec_bl_tree_first_sparse_mask_offset_kv
    spec_decoding_bl_tree_mask spec_decoding_bl_tree_mask_offset spec_decoding_generation_lengths
    spec_decoding_packed_mask tokens_per_block trtllm_gen_jit_warmup
    update_blackwell_first_sparse_mask_offset update_for_spec_dec update_indexer_kv_lens_2d
    use_paged_context_fmha use_spec_decoding workspace_reclaimable
    """.split()
)
#: Engine attention-metadata names the targets read that the metadata's methods set on the steps that need them.
STEP_FIELDS_SET_LATER = frozenset(
    """
    block_table_expanded ctx_cached_token_indptr ctx_kv_indptr cu_seqlen_ke cu_seqlen_ks
    host_ctx_cached_token_indptr host_ctx_kv_indptr host_indexer_k_cache_block_offsets
    host_request_types_runtime host_slot_mapping_fp8 host_slot_mapping_scale host_total_kv_lens
    indexer_k_cache_block_offsets kv_lens_cuda kv_lens_cuda_2d kv_lens_cuda_runtime kv_lens_expanded_cuda
    kv_lens_runtime max_ctx_seq_len prompt_lens_cpu_runtime prompt_lens_cuda_runtime
    scheduler_metadata_buffer scheduler_metadata_buffer_expanded scheduler_metadata_buffer_full_next_n
    slot_mapping_fp8 slot_mapping_fp8_fullkv slot_mapping_scale slot_mapping_scale_fullkv
    topk_indices_buffer
    """.split()
)
#: Engine attention-metadata names only the sm_107 (NVFP4) targets read, declared or set when it is built. The
#: first-forward check reads :data:`STEP_FIELDS` alone, which every target's metadata carries.
SM107_STEP_FIELDS = frozenset(
    """
    _draft_sizing_cap _indexer_compress_ratio all_rank_num_tokens compress_ratios enable_helix
    enable_ragged_verification gen_token_repeat_list gen_token_stride helix_kv_bounds is_ragged_verify
    num_ctx_mla_kv_tokens seq_lens_cuda token_major_gen_view use_cute_dsl_paged_mqa_logits
    use_expanded_buffers_for_mtp use_fp8_ds_mla use_self_sampling_topk
    """.split()
)
#: Engine attention-metadata names only the sm_107 targets read that the metadata's methods set later.
SM107_STEP_FIELDS_SET_LATER = frozenset(
    """
    block_table gen_cached_token_indptr gen_kv_indptr nvfp4_mla_compact_indices nvfp4_mla_fp8_scratch
    req_idx_per_token token_positions_cuda
    """.split()
)


def check_step_contract(attn_metadata) -> None:
    """Fail fast if ``attn_metadata`` lacks a name in :data:`STEP_FIELDS`.

    A name passes when the metadata object holds it or a class of its MRO declares it (an attribute, a property or
    an annotated field). Nothing is evaluated, so the check reads no device state.
    """
    present = set(vars(attn_metadata))
    for cls in type(attn_metadata).__mro__:
        present.update(vars(cls))
        present.update(vars(cls).get("__annotations__", {}))
    missing = sorted(STEP_FIELDS - present)
    if missing:
        raise AssertionError(
            f"attention metadata surface drifted: {type(attn_metadata).__name__} has no {missing}"
        )
