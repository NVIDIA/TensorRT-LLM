from typing import List, Optional

import torch


def attention(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    host_total_kv_lens: torch.Tensor,
    context_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    cache_indirection: Optional[torch.Tensor],
    kv_scale_orig_quant: Optional[torch.Tensor],
    kv_scale_quant_orig: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    rotary_inv_freq: Optional[torch.Tensor],
    rotary_cos_sin: Optional[torch.Tensor],
    latent_cache: Optional[torch.Tensor],
    q_pe: Optional[torch.Tensor],
    block_ids_per_seq: Optional[torch.Tensor],
    attention_sinks: Optional[torch.Tensor],
    is_fused_qkv: bool,
    update_kv_cache: bool,
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: Optional[int],
    max_num_requests: int,
    max_context_length: int,
    attention_window_size: int,
    sink_token_length: int,
    beam_width: int,
    mask_type: int,
    quant_mode: int,
    q_scaling: float,
    position_embedding_type: int,
    rotary_embedding_dim: int,
    rotary_embedding_base: float,
    rotary_embedding_scale_type: int,
    rotary_embedding_scales: List[float],
    rotary_embedding_max_position_info: List[int],
    use_paged_context_fmha: bool,
    attention_input_type: Optional[int],
    is_mla_enable: bool,
    chunked_prefill_buffer_batch_size: Optional[int],
    q_lora_rank: Optional[int],
    kv_lora_rank: Optional[int],
    qk_nope_head_dim: Optional[int],
    qk_rope_head_dim: Optional[int],
    v_head_dim: Optional[int],
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    mla_tensor_params: List[Optional[torch.Tensor]],
    attention_chunk_size: Optional[int],
    softmax_stats_tensor: Optional[torch.Tensor],
    spec_decoding_bool_params: List[bool],
    spec_decoding_tensor_params: List[Optional[torch.Tensor]],
    sparse_kv_indices: Optional[torch.Tensor],
    sparse_kv_offsets: Optional[torch.Tensor],
    sparse_attn_indices: Optional[torch.Tensor],
    sparse_attn_offsets: Optional[torch.Tensor],
    sparse_attn_indices_block_size: int,
    sparse_mla_topk: Optional[int],
    skip_softmax_threshold_scale_factor_prefill: Optional[float],
    skip_softmax_threshold_scale_factor_decode: Optional[float],
    skip_softmax_stat: Optional[torch.Tensor],
    cu_q_seqlens: Optional[torch.Tensor],
    cu_kv_seqlens: Optional[torch.Tensor],
    fmha_scheduler_counter: Optional[torch.Tensor],
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
) -> None:
    """
    Drop-in replacement for thop.attention().

    This function implements attention computation for both context (prefill) and generation (decode) phases,
    with support for paged KV cache, various attention patterns, MLA (Multi-head Latent Attention),
    and speculative decoding.

    The implementation merges the logic from C++ attentionOp.cpp::attention() and Runner::run()
    into a single Python function, eliminating the need for a separate Runner class.

    Args:
        q: Query tensor [num_tokens, hidden_dim] or fused QKV tensor
        k: Key tensor (optional, None if using fused QKV)
        v: Value tensor (optional, None if using fused QKV)
        output: Output tensor [num_tokens, hidden_dim]
        output_sf: Output scale factor tensor for NVFP4 output (optional)
        workspace: Workspace tensor for intermediate computations
        sequence_length: Current sequence lengths per request
        host_past_key_value_lengths: Past KV lengths on host
        host_total_kv_lens: Total KV lengths [ctx_total, gen_total]
        context_lengths: Context lengths per request
        host_context_lengths: Context lengths on host
        host_request_types: Request types (CONTEXT=0, GENERATION=1) on host
        kv_cache_block_offsets: Block offsets for paged KV cache
        host_kv_cache_block_offsets: Block offsets on host
        host_kv_cache_pool_pointers: Pool pointers for KV cache
        host_kv_cache_pool_mapping: Pool mapping [layer_idx] -> [pool_idx, layer_in_pool]
        cache_indirection: Cache indirection for beam search
        kv_scale_orig_quant: KV cache quantization scale (orig->quant)
        kv_scale_quant_orig: KV cache dequantization scale (quant->orig)
        out_scale: Output quantization scale
        rotary_inv_freq: RoPE inverse frequencies
        rotary_cos_sin: Precomputed RoPE cos/sin cache
        latent_cache: MLA latent cache tensor
        q_pe: Query position embedding for MLA
        block_ids_per_seq: Block IDs per sequence for flash MLA
        attention_sinks: Attention sink values per head
        is_fused_qkv: Whether Q/K/V are fused into single tensor
        update_kv_cache: Whether to update KV cache (must be True)
        predicted_tokens_per_seq: Number of predicted tokens per sequence
        layer_idx: Current layer index
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA/MQA)
        head_size: Size of each attention head
        tokens_per_block: Tokens per KV cache block
        max_num_requests: Maximum number of requests
        max_context_length: Maximum context length
        attention_window_size: Attention window size
        sink_token_length: Number of sink tokens for StreamingLLM
        beam_width: Beam width for beam search
        mask_type: Attention mask type (AttentionMaskType)
        quant_mode: Quantization mode flags
        q_scaling: Query scaling factor
        position_embedding_type: Position embedding type (PositionEmbeddingType)
        rotary_embedding_dim: RoPE embedding dimension
        rotary_embedding_base: RoPE base frequency
        rotary_embedding_scale_type: RoPE scaling type
        rotary_embedding_scales: RoPE scaling factors [scale, short_m_scale, long_m_scale]
        rotary_embedding_max_position_info: [max_positions, original_max_positions]
        use_paged_context_fmha: Whether to use paged context FMHA
        attention_input_type: Input type (Mixed=0, ContextOnly=1, GenerationOnly=2)
        is_mla_enable: Whether MLA is enabled
        chunked_prefill_buffer_batch_size: Batch size for chunked prefill buffer
        q_lora_rank: MLA Q LoRA rank
        kv_lora_rank: MLA KV LoRA rank
        qk_nope_head_dim: MLA QK nope head dimension
        qk_rope_head_dim: MLA QK rope head dimension
        v_head_dim: MLA V head dimension
        mrope_rotary_cos_sin: Multi-rope rotary cos/sin
        mrope_position_deltas: Multi-rope position deltas
        mla_tensor_params: MLA tensor parameters [helix_position_offsets, helix_is_inactive_rank]
        attention_chunk_size: Chunked attention size
        softmax_stats_tensor: Softmax statistics output tensor
        spec_decoding_bool_params: [is_spec_decoding_enabled, use_spec_decoding, is_spec_dec_tree]
        spec_decoding_tensor_params: Speculative decoding tensors
        sparse_kv_indices: Sparse KV indices
        sparse_kv_offsets: Sparse KV offsets
        sparse_attn_indices: Sparse attention indices
        sparse_attn_offsets: Sparse attention offsets
        sparse_attn_indices_block_size: Block size for sparse attention indices
        sparse_mla_topk: Top-K for sparse MLA
        skip_softmax_threshold_scale_factor_prefill: Skip softmax threshold (prefill)
        skip_softmax_threshold_scale_factor_decode: Skip softmax threshold (decode)
        skip_softmax_stat: Skip softmax statistics tensor
        cu_q_seqlens: Cumulative Q sequence lengths
        cu_kv_seqlens: Cumulative KV sequence lengths
        fmha_scheduler_counter: FMHA scheduler counter
        mla_bmm1_scale: MLA BMM1 scale
        mla_bmm2_scale: MLA BMM2 scale
        quant_q_buffer: Quantized Q buffer for FP8 MLA
    """
    raise NotImplementedError("TRTLLM-Gen attention backend is not implemented yet.")
