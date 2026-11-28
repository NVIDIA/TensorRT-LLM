# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace


class _ContextSparseParams(SimpleNamespace):
    sparse_kv_indices = None
    sparse_kv_offsets = None
    sparse_attn_indices = None
    sparse_attn_offsets = None
    sparse_attn_indices_block_size = 0
    sparse_attn_kv_lens = None
    aux_kv_cache_pool_ptr = None
    max_num_ctas_per_seq_kv = None
    threshold_scale_factor_decode = 0.0


class _ContextForwardArgs(SimpleNamespace):
    output_sf = None
    variable_window_token_starts = None
    variable_window_token_ends = None
    out_scale = None
    out_scale_sf = None
    latent_cache = None
    q_pe = None
    mrope_rotary_cos_sin = None
    mrope_position_deltas = None
    softmax_stats_tensor = None
    softmax_stats_is_lse = False
    attention_sinks = None
    cu_q_seqlens = None
    cu_kv_seqlens = None
    fmha_scheduler_counter = None
    locality_domain_id = -1
    mla_bmm1_scale = None
    mla_bmm2_scale = None
    quant_q_buffer = None
    cross_kv = None
    relative_attention_bias = None
    quant_scale_qkv = None
    dsv4_inv_rope_cos_sin_cache = None
    enable_dsv4_epilogue_fusion = False
    kv_norm_weight = None
    kv_norm_eps = 1e-06
    attention_input_type = 0
    attention_mask_data = None
    relative_attention_max_distance = 0
    chunked_prefill_buffer_batch_size = 1
    skip_mla_rope_generation = False
    sage_attn_smooth_v = False
    qk_fp8_per_tensor = False
    v_fp8_per_tensor = False
    vc_attention = False
    update_kv_cache = True
    output_gate = None
    output_gate_applied = False
    gated_out_scale = None
    output_fp8 = None
    sparse_attn_phase = None
    sparse_backend_args = None
    mask_type = 0  # Native padding mask includes every token in the uncached sequence.


class _ContextParams(SimpleNamespace):
    query_input = None
    seq_offset = 0
    token_offset = 0
    total_num_blocks = 0
    use_spec_decoding = False
    spec_decoding_generation_lengths = None
    spec_decoding_position_offsets = None
    spec_decoding_packed_mask = None
    spec_decoding_bl_tree_mask_offset = None
    spec_decoding_bl_tree_mask = None
    spec_bl_tree_first_sparse_mask_offset_kv = None
    is_cross = False
    num_gen_tokens = 0


class _ContextRope(SimpleNamespace):
    dim = 0
    theta = 10000.0
    scale_type = 0
    scale = 1.0
    short_m_scale = 1.0
    long_m_scale = 1.0
    max_positions = 1024
    original_max_positions = 1024
