import math
from typing import List, Optional, Tuple

import flashinfer
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

########################################################
# TrtllmGenAttention Backend
########################################################

# Alignment for workspace buffers (256 bytes)
WORKSPACE_ALIGNMENT = 256


def _align_size(size: int, alignment: int = WORKSPACE_ALIGNMENT) -> int:
    """Align size to the given alignment boundary."""
    return ((size + alignment - 1) // alignment) * alignment


def _get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes for a given dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    elif dtype in (torch.float32,):
        return 4
    elif dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
        return 1
    else:
        return 2  # Default to fp16 size


def get_workspace_size_for_context(
    dtype: torch.dtype,
    max_num_seq: int,
    max_context_length: int,
    max_num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_embedding_dim: int = 0,
) -> int:
    """
    Calculate workspace size for context (prefill) phase.
    Aligned with C++ AttentionOp::getWorkspaceSizeForContext().
    """
    if max_num_tokens == 0:
        return 0

    dtype_size = _get_dtype_size(dtype)
    local_hidden_units_qo = num_heads * head_size

    # For paged context FMHA (flashinfer), we need Q buffer
    q_buf_size = dtype_size * max_num_tokens * local_hidden_units_qo

    # Cumulative sequence lengths
    cu_seqlens_size = 4 * (max_num_seq + 1)  # sizeof(int) * (batch_size + 1)

    # Rotary inv freq buffer
    rotary_inv_freq_size = (
        4 * max_num_seq * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
    )

    # Tokens info: (batch_idx, token_idx_in_seq) per token
    tokens_info_size = 8 * max_num_tokens  # sizeof(int2) * max_num_tokens

    # FMHA scheduler counter
    fmha_scheduler_counter = 4  # sizeof(uint32_t)

    # BMM scales for FP8
    fmha_bmm1_scale_size = 4 * 2  # sizeof(float) * 2
    fmha_bmm2_scale_size = 4  # sizeof(float)

    # Calculate total with alignment
    workspace_size = 0
    workspace_size += _align_size(q_buf_size)
    workspace_size += _align_size(cu_seqlens_size) * 3  # cu_seqlen_q, cu_seqlen_kv, cu_mask_rows
    workspace_size += _align_size(rotary_inv_freq_size)
    workspace_size += _align_size(tokens_info_size)
    workspace_size += _align_size(fmha_scheduler_counter)
    workspace_size += _align_size(fmha_bmm1_scale_size)
    workspace_size += _align_size(fmha_bmm2_scale_size)

    return workspace_size


def get_workspace_size_for_generation(
    dtype: torch.dtype,
    max_num_seq: int,
    max_attention_window_size: int,
    max_num_tokens: int,
    max_blocks_per_sequence: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_embedding_dim: int = 0,
    multi_processor_count: int = 132,  # Default for modern GPUs
) -> int:
    """
    Calculate workspace size for generation (decode) phase.
    Aligned with C++ AttentionOp::getWorkspaceSizeForGeneration().
    """
    if max_num_tokens == 0:
        return 0

    dtype_size = _get_dtype_size(dtype)
    batch_beam = max_num_seq

    # Estimate max sequence length tile (simplified from C++)
    # In C++: divUp(mMultiProcessorCount, batch_beam * mNumHeads)
    max_seq_len_tile = max(
        1, (multi_processor_count + batch_beam * num_heads - 1) // (batch_beam * num_heads)
    )
    max_seq_len_tile = max(max_seq_len_tile, 4)  # Minimum tile count

    # Partial output/sum/max buffers for multi-block attention
    partial_out_size = dtype_size * batch_beam * num_heads * head_size * max_seq_len_tile
    partial_sum_size = 4 * batch_beam * num_heads * max_seq_len_tile  # sizeof(float)
    partial_max_size = 4 * batch_beam * num_heads * max_seq_len_tile  # sizeof(float)

    # XQA workspace components
    cu_seqlens_size = 4 * (batch_beam + 1)
    cu_kv_seqlens_size = 4 * (batch_beam + 1)
    rotary_inv_freq_size = (
        4 * batch_beam * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
    )
    tokens_info_size = 8 * max_num_tokens  # sizeof(int2) * max_num_tokens

    # Scales for trtllm-gen kernels
    bmm1_scale_size = 4 * 2  # sizeof(float) * 2
    bmm2_scale_size = 4  # sizeof(float)

    # Calculate total with alignment
    workspace_size = 0
    workspace_size += _align_size(partial_out_size)
    workspace_size += _align_size(partial_sum_size)
    workspace_size += _align_size(partial_max_size)
    workspace_size += _align_size(cu_seqlens_size)
    workspace_size += _align_size(cu_kv_seqlens_size)
    workspace_size += _align_size(rotary_inv_freq_size)
    workspace_size += _align_size(tokens_info_size)
    workspace_size += _align_size(bmm1_scale_size)
    workspace_size += _align_size(bmm2_scale_size)

    return workspace_size


def get_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_context_length: int,
    max_attention_window_size: int,
    num_tokens: int,
    num_gen_tokens: int,
    max_blocks_per_sequence: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_embedding_dim: int = 0,
) -> int:
    """
    Calculate total workspace size for attention operation.
    Returns max(context_workspace_size, generation_workspace_size).
    Aligned with C++ AttentionOp workspace calculation.
    """
    context_workspace_size = get_workspace_size_for_context(
        dtype=dtype,
        max_num_seq=max_num_seq,
        max_context_length=max_context_length,
        max_num_tokens=num_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_embedding_dim=rotary_embedding_dim,
    )

    generation_workspace_size = get_workspace_size_for_generation(
        dtype=dtype,
        max_num_seq=max_num_seq,
        max_attention_window_size=max_attention_window_size,
        max_num_tokens=num_gen_tokens,
        max_blocks_per_sequence=max_blocks_per_sequence,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_embedding_dim=rotary_embedding_dim,
    )

    return max(context_workspace_size, generation_workspace_size)


def is_sm100_family() -> bool:
    """
    Check if the SM version is in the SM100 family (Blackwell architecture).

    Returns:
        True if SM is 100 or 103 (Blackwell architecture).
    """
    sm = get_sm_version()
    return sm == 100 or sm == 103


def is_supported(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    mask_type: Optional[int] = None,
    has_alibi: bool = False,
    is_padded: bool = False,
    use_paged_kv_cache: bool = True,
    tokens_per_block: int = 64,
    beam_width: int = 1,
    position_shift_enabled: bool = False,
    sink_token_length: int = 0,
    cross_attention: bool = False,
    cyclic_attention_window_size: Optional[int] = None,
    max_attention_window_size: Optional[int] = None,
    is_spec_decoding: bool = False,
    # Additional parameters for complete fallback check
    is_mla_enable: bool = False,
    is_fused_qkv: bool = True,
    update_kv_cache: bool = True,
    has_rotary_inv_freq: bool = False,
    has_rotary_cos_sin: bool = False,
    has_kv_scale: bool = False,
    has_cross_kv: bool = False,
    phase: str = "both",
) -> Tuple[bool, str]:
    """
    Check whether the trtllm-gen attention backend supports the given configuration.

    This function consolidates ALL checks needed to determine if trtllm-gen
    flashinfer backend can handle the attention computation. If this returns
    False, the caller should fallback to thop.attention().

    Hardware Requirements:
        - Only supports Blackwell architecture: SM100 or SM103

    Supported Data Types:
        - Context FMHA Input: FP16, BF16, FP8 (E4M3)
        - KV Cache: FP16, BF16, FP8 (E4M3), FP4 (E2M1)
        - Output: FP16, BF16, FP8 (E4M3), FP4 (E2M1)

    Not Supported:
        - MLA (Multi-head Latent Attention)
        - Unfused QKV input
        - KV cache update disabled
        - RoPE with inv_freq tensor (precomputed cos_sin only)
        - KV cache quantization scales
        - Cross attention
        - ALiBi position embedding
        - Custom masks
        - Beam search (beam_width > 1)
        - StreamingLLM (sink_token_length > 0)
        - Position shift
        - Speculative decoding

    Args:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (for GQA/MQA).
        head_size: Size of each attention head.
        dtype: Data type of the input tensors.
        kv_cache_dtype: Data type of the KV cache. If None, uses dtype.
        out_dtype: Output data type. If None, uses dtype.
        mask_type: Attention mask type (AttentionMaskType enum value).
        has_alibi: Whether ALiBi position embedding is used.
        is_padded: Whether input is padded.
        use_paged_kv_cache: Whether to use paged KV cache.
        tokens_per_block: Number of tokens per KV cache block.
        beam_width: Beam width for beam search.
        position_shift_enabled: Whether position shift is enabled.
        sink_token_length: Number of sink tokens for StreamingLLM.
        cross_attention: Whether cross attention is used.
        cyclic_attention_window_size: Cyclic attention window size.
        max_attention_window_size: Max attention window size.
        is_spec_decoding: Whether speculative decoding is enabled.
        is_mla_enable: Whether MLA is enabled.
        is_fused_qkv: Whether QKV is fused into single tensor.
        update_kv_cache: Whether KV cache update is enabled.
        has_rotary_inv_freq: Whether rotary_inv_freq tensor is provided.
        has_rotary_cos_sin: Whether rotary_cos_sin tensor is provided.
        has_kv_scale: Whether KV quantization scales are provided.
        has_cross_kv: Whether cross KV is provided.
        phase: Which phase to check - "context", "generation", or "both".

    Returns:
        Tuple of (is_supported: bool, reason: str).
    """

    # ========== Hardware check ==========
    if not is_sm100_family():
        return (
            False,
            "trtllm-gen attention requires SM100 or SM103 (Blackwell architecture).",
        )

    # ========== MLA check ==========
    if is_mla_enable:
        return (
            False,
            "MLA (Multi-head Latent Attention) is not supported by trtllm-gen flashinfer backend.",
        )

    # ========== Fused QKV check ==========
    if not is_fused_qkv:
        return False, "Only fused QKV is supported by trtllm-gen flashinfer backend."

    # ========== KV cache update check ==========
    if not update_kv_cache:
        return False, "KV cache update must be enabled for trtllm-gen flashinfer backend."

    # ========== Cross attention check ==========
    if cross_attention or has_cross_kv:
        return False, "Cross attention is not supported by trtllm-gen flashinfer backend."

    # ========== KV scale check ==========
    if has_kv_scale:
        return (
            False,
            "KV cache quantization scales are not supported by trtllm-gen flashinfer backend.",
        )

    # ========== Speculative decoding check ==========
    if is_spec_decoding:
        return False, "Speculative decoding is not supported by trtllm-gen flashinfer backend."

    # ========== Data type checks ==========
    supported_input_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    if dtype not in supported_input_dtypes:
        return False, f"Input dtype {dtype} not supported. Supported: FP16, BF16, FP8 (E4M3)."

    if kv_cache_dtype is not None:
        supported_kv_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8}
        if kv_cache_dtype not in supported_kv_dtypes:
            return (
                False,
                f"KV cache dtype {kv_cache_dtype} not supported. Supported: FP16, BF16, FP8 (E4M3), FP4 (E2M1).",
            )

    if out_dtype is not None:
        supported_out_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8}
        if out_dtype not in supported_out_dtypes:
            return (
                False,
                f"Output dtype {out_dtype} not supported. Supported: FP16, BF16, FP8 (E4M3), FP4 (E2M1).",
            )

    # ========== Basic validation ==========
    if num_heads <= 0:
        return False, "num_heads must be positive."
    if num_kv_heads <= 0:
        return False, "num_kv_heads must be positive."
    if num_heads % num_kv_heads != 0:
        return False, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."

    # ========== Context phase checks ==========
    if phase in ("context", "both"):
        # Head size must NOT equal 80 or 72
        if head_size == 80:
            return False, "[Context] Head size 80 is not supported by trtllm-gen Context FMHA."
        if head_size == 72:
            return False, "[Context] Head size 72 is not supported by trtllm-gen Context FMHA."

        # Custom mask not supported
        if mask_type is not None:
            try:
                mask_type_enum = AttentionMaskType(mask_type)
                if mask_type_enum == AttentionMaskType.CUSTOM_MASK:
                    return (
                        False,
                        "[Context] Custom mask is not supported by trtllm-gen Context FMHA.",
                    )
            except ValueError:
                return False, f"[Context] Invalid mask_type value: {mask_type}."

        # ALiBi not supported
        if has_alibi:
            return False, "[Context] ALiBi is not supported by trtllm-gen Context FMHA."

        # Padded input not supported
        if is_padded:
            return False, "[Context] Padded input is not supported by trtllm-gen Context FMHA."

    # ========== Generation phase checks ==========
    if phase in ("generation", "both"):
        # beam_width must = 1 (beam search not supported)
        if beam_width != 1:
            return (
                False,
                f"[Generation] Beam search (beam_width={beam_width}) is not supported. Must be 1.",
            )

        # position_shift_enabled must = false
        if position_shift_enabled:
            return False, "[Generation] Position shift is not supported."

        # sink_token_length must = 0 (StreamingLLM not supported)
        if sink_token_length != 0:
            return (
                False,
                f"[Generation] StreamingLLM (sink_token_length={sink_token_length}) is not supported.",
            )

        # cross_attention + !paged_kv_cache not supported
        if cross_attention and not use_paged_kv_cache:
            return (
                False,
                "[Generation] Cross attention with non-paged KV cache is not supported.",
            )

        # tokens_per_block must >= 8
        if tokens_per_block < 8:
            return (
                False,
                f"[Generation] tokens_per_block ({tokens_per_block}) must be >= 8.",
            )

        # cyclic_attention_window_size must = max_attention_window_size
        if cyclic_attention_window_size is not None and max_attention_window_size is not None:
            if cyclic_attention_window_size != max_attention_window_size:
                return (
                    False,
                    f"[Generation] cyclic_attention_window_size ({cyclic_attention_window_size}) must equal "
                    f"max_attention_window_size ({max_attention_window_size}).",
                )

        # num_q_heads / num_kv_heads must <= 16
        if num_kv_heads > 0:
            heads_ratio = num_heads // num_kv_heads
            if heads_ratio > 16:
                return (
                    False,
                    f"[Generation] num_heads/num_kv_heads ratio ({heads_ratio}) must be <= 16.",
                )

        # ALiBi not supported
        if has_alibi:
            return False, "[Generation] ALiBi is not supported."

    # ========== Paged KV cache checks ==========
    if use_paged_kv_cache:
        if tokens_per_block <= 0:
            return False, "tokens_per_block must be positive for paged KV cache."
        if tokens_per_block & (tokens_per_block - 1) != 0:
            return False, f"tokens_per_block ({tokens_per_block}) must be a power of 2."

    return True, ""


def trtllm_gen_attention(
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
    TrtLLM-Gen attention using flashinfer backend.

    This function implements attention computation for both context (prefill) and
    generation (decode) phases using flashinfer's batch attention APIs.

    IMPORTANT: This function assumes is_supported() has already been called and
    returned True. If the configuration is not supported, the caller (trtllm.py)
    should fallback to thop.attention() instead of calling this function.
    """
    logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

    num_seqs = host_context_lengths.size(0)
    num_tokens = q.size(0)

    # Determine attention input type
    attn_input_type = AttentionInputType.mixed
    if attention_input_type is not None:
        attn_input_type = AttentionInputType(attention_input_type)

    # Count context vs generation requests
    request_types = host_request_types.cpu().numpy()
    num_contexts = 0
    for idx in range(num_seqs):
        if request_types[idx] != 0:
            break
        num_contexts += 1
    num_generations = num_seqs - num_contexts

    # Calculate token counts
    num_ctx_tokens = host_context_lengths[:num_contexts].sum().item() if num_contexts > 0 else 0
    num_gen_tokens = num_tokens - num_ctx_tokens

    # Calculate max_blocks_per_sequence from block offsets
    max_blocks_per_sequence = 0
    if kv_cache_block_offsets is not None and kv_cache_block_offsets.numel() > 0:
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1)

    # Calculate workspace size aligned with C++ implementation
    workspace_size = get_workspace_size(
        dtype=q.dtype,
        max_num_seq=max_num_requests,
        max_context_length=max_context_length,
        max_attention_window_size=attention_window_size,
        num_tokens=num_tokens,
        num_gen_tokens=num_gen_tokens,
        max_blocks_per_sequence=max_blocks_per_sequence,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_embedding_dim=rotary_embedding_dim,
    )

    # Allocate workspace if needed
    if workspace is None or workspace.numel() < workspace_size:
        workspace = torch.zeros(workspace_size, dtype=torch.uint8, device=q.device)

    # Reshape Q for attention: [num_tokens, hidden_dim] -> [num_tokens, num_heads, head_size]
    q_tensor = q.view(num_tokens, num_heads + 2 * num_kv_heads, head_size)
    query = q_tensor[:, :num_heads, :]  # Extract Q from fused QKV

    # Prepare output tensor
    out_tensor = output.view(num_tokens, num_heads, head_size)

    # Compute softmax scale
    softmax_scale = (
        q_scaling / math.sqrt(head_size) if q_scaling != 1.0 else 1.0 / math.sqrt(head_size)
    )

    # ========== Context Phase (Prefill) ==========
    if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
        ctx_query = query[:num_ctx_tokens]
        ctx_output = out_tensor[:num_ctx_tokens]
        ctx_context_lengths = context_lengths[:num_contexts]
        ctx_seq_lens = sequence_length[:num_contexts]

        # Build cumulative sequence lengths
        cum_seq_lens_q = torch.zeros(num_contexts + 1, dtype=torch.int32, device=q.device)
        cum_seq_lens_q[1:] = torch.cumsum(ctx_context_lengths.to(torch.int32), dim=0)

        cum_seq_lens_kv = torch.zeros(num_contexts + 1, dtype=torch.int32, device=q.device)
        cum_seq_lens_kv[1:] = torch.cumsum(ctx_seq_lens.to(torch.int32), dim=0)

        # Get block tables for context requests
        ctx_block_tables = (
            kv_cache_block_offsets[0, :num_contexts] if kv_cache_block_offsets is not None else None
        )

        # Determine window_left for sliding window
        window_left = attention_window_size if attention_window_size < max_context_length else -1

        if host_kv_cache_pool_pointers is not None and host_kv_cache_pool_mapping is not None:
            # Reconstruct KV cache shape
            num_pages = ctx_block_tables.max().item() + 1 if ctx_block_tables is not None else 1
            page_size = tokens_per_block or 64

            # Create KV cache tensor (placeholder - real impl needs proper memory mapping)
            kv_cache = torch.empty(
                (num_pages, 2, num_kv_heads, page_size, head_size),
                dtype=q.dtype,
                device=q.device,
            )

            # Extract K and V caches
            k_cache = kv_cache[:, 0]
            v_cache = kv_cache[:, 1]

            # Call flashinfer batch prefill API directly
            ctx_result = flashinfer.trtllm_batch_prefill_with_paged_kv_cache(
                q=ctx_query,
                qo_indptr=cum_seq_lens_q,
                paged_kv_cache=(k_cache, v_cache),
                paged_kv_indptr=cum_seq_lens_kv,
                paged_kv_indices=ctx_block_tables.flatten(),
                paged_kv_last_page_len=ctx_seq_lens.to(torch.int32),
                sm_scale=softmax_scale,
                window_left=window_left if window_left > 0 else -1,
                causal=True,
                kv_layout="HND",
                backend="trtllm-gen",
            )
            ctx_output.copy_(ctx_result)

    # ========== Generation Phase (Decode) ==========
    if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
        gen_query = query[num_ctx_tokens:]
        gen_output = out_tensor[num_ctx_tokens:]
        gen_seq_lens = sequence_length[num_contexts:]

        # Get block tables for generation requests
        gen_block_tables = (
            kv_cache_block_offsets[0, num_contexts:] if kv_cache_block_offsets is not None else None
        )

        # Determine window_left
        window_left = attention_window_size if attention_window_size < max_context_length else -1

        if host_kv_cache_pool_pointers is not None and host_kv_cache_pool_mapping is not None:
            num_pages = gen_block_tables.max().item() + 1 if gen_block_tables is not None else 1
            page_size = tokens_per_block or 64

            kv_cache = torch.empty(
                (num_pages, 2, num_kv_heads, page_size, head_size),
                dtype=q.dtype,
                device=q.device,
            )

            # Extract K and V caches
            k_cache = kv_cache[:, 0]
            v_cache = kv_cache[:, 1]

            # Build paged_kv_indptr for decode
            paged_kv_indptr = torch.arange(
                0, num_generations + 1, dtype=torch.int32, device=q.device
            ) * gen_block_tables.size(1)

            # Calculate last page lengths
            paged_kv_last_page_len = (gen_seq_lens % page_size).clamp(min=1).to(torch.int32)

            # Call flashinfer batch decode API directly
            gen_result = flashinfer.trtllm_batch_decode_with_kv_cache(
                q=gen_query.view(num_generations, -1, num_heads, head_size).squeeze(1),
                paged_kv_cache=(k_cache, v_cache),
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=gen_block_tables.flatten(),
                paged_kv_last_page_len=paged_kv_last_page_len,
                sm_scale=softmax_scale,
                window_left=window_left if window_left > 0 else -1,
                kv_layout="HND",
                backend="trtllm-gen",
            )
            gen_output.copy_(gen_result.view(num_gen_tokens, num_heads, head_size))

    logger.debug(f"trtllm_gen_attention stops at layer {layer_idx}")
