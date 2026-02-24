"""
TrtLLM-Gen Attention Backend

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a drop-in replacement for thop.attention() with support for trtllm-gen
kernel only (Blackwell architecture: SM100/SM103). Enabled via TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1.

Architecture:
    - QKV preprocessing & RoPE: C++ kernels via torch.ops.trtllm.qkv_preprocessing,
      same as thop.attention. Writes K/V to paged KV cache via pool pointers.
    - Attention: flashinfer trtllm-gen FMHA kernels, reading KV cache through
      kv_cache_manager.get_buffers() (flashinfer tensor format).

Entry points:
    is_supported()           - Check if trtllm-gen can handle the given config.
    trtllm_gen_attention()   - Main attention function (called from TrtllmAttention.run).

Example:
    # Check if configuration is supported
    supported, reason = is_supported(num_heads=32, num_kv_heads=8, ...)
    if supported:
        trtllm_gen_attention(q, k, v, output, ...)
    else:
        Fallback to thop.attention()
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import (
    get_size_in_bytes,
    get_sm_version,
    is_sm_100f,
    torch_dtype_to_binding,
)
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

# Default KV layout for flashinfer
# HND = [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]
DEFAULT_KV_LAYOUT = "HND"


class TrtllmGenSupportChecker:
    """
    Validates if a configuration is supported by trtllm-gen backend.

    Implements all checks from the original C++ AttentionOp to determine
    if trtllm-gen kernel can handle the attention computation.
    """

    # Supported data types
    SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_KV_CACHE_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    # Supported Q:KV:O dtype combinations for trtllm-gen kernels
    # Format: (q_dtype, kv_dtype, o_dtype)
    # Context phase supported combinations
    SUPPORTED_DTYPE_COMBOS_CONTEXT = {
        # e4m3:e4m3:e4m3
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn),
        # e4m3:e4m3:e2m1 (FP4 output not directly representable, skip)
        # fp16:fp16:fp16
        (torch.float16, torch.float16, torch.float16),
        # bf16:bf16:bf16
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        # e4m3:e4m3:fp16
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float16),
        # e4m3:e4m3:bf16
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16),
    }

    # Generation phase supported combinations (includes context + additional)
    SUPPORTED_DTYPE_COMBOS_GENERATION = {
        # All context combinations
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float16, torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16),
        # Additional generation-only combinations
        # bf16:e4m3:bf16
        (torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16),
        # fp16:e4m3:fp16
        (torch.float16, torch.float8_e4m3fn, torch.float16),
    }

    # Unsupported head sizes for context FMHA
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80}

    # Maximum heads ratio for generation
    MAX_HEADS_RATIO_GENERATION = 16

    # Minimum tokens per block, tokens_per_block < 8 is not supported by TRTLLM-GEN kernels.
    MIN_TOKENS_PER_BLOCK = 8

    # Supported tokens_per_block values for trtllm-gen kernels
    SUPPORTED_TOKENS_PER_BLOCK = {16, 32, 64}

    @classmethod
    def is_supported(
        cls,
        q_dtype: torch.dtype,
        kv_cache_dtype: torch.dtype,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        phase: str = "both",
        out_dtype: Optional[torch.dtype] = None,
        mask_type: int = 1,
        beam_width: int = 1,
        sink_token_length: int = 0,
        tokens_per_block: int = 64,
        use_paged_kv_cache: bool = True,
        is_mla_enable: bool = False,
        is_fused_qkv: bool = True,
        update_kv_cache: bool = True,
        cross_attention: bool = False,
        is_spec_decoding: bool = False,
        has_alibi: bool = False,
        is_padded: bool = False,
        position_shift_enabled: bool = False,
        quant_config: Optional[QuantConfig] = None,
    ) -> Tuple[bool, str]:
        sm = get_sm_version()
        if not is_sm_100f(sm):
            return (False, f"trtllm-gen requires SM100 or SM103 (Blackwell). Current: SM{sm}.")

        if is_mla_enable:
            return False, "MLA is not supported by trtllm-gen backend."
        if not is_fused_qkv:
            return False, "Only fused QKV is supported by trtllm-gen backend."
        if not update_kv_cache:
            return False, "KKV cache update cannot be disabled now."
        if cross_attention:
            return False, "Cross attention is not supported by trtllm-gen backend."
        if is_spec_decoding:
            return False, "Speculative decoding is not supported by trtllm-gen backend."

        has_fp4_kv = (
            quant_config.layer_quant_mode.has_fp4_kv_cache()
            if quant_config is not None
            else kv_cache_dtype == torch.uint8
        )
        if has_fp4_kv:
            return False, "NVFP4 KV cache is not supported by flashinfer trtllm-gen kernels."
        if q_dtype not in cls.SUPPORTED_INPUT_DTYPES:
            return False, f"Input dtype {q_dtype} not supported. Supported: FP16, BF16, FP8 (E4M3)."
        if kv_cache_dtype not in cls.SUPPORTED_KV_CACHE_DTYPES:
            return (
                False,
                f"KV cache dtype {kv_cache_dtype} not supported. Supported: FP16, BF16, FP8.",
            )
        if out_dtype is not None and out_dtype not in cls.SUPPORTED_OUT_DTYPES:
            return False, f"Output dtype {out_dtype} not supported. Supported: FP16, BF16, FP8."

        assert num_heads > 0, "num_heads must be positive."
        assert num_kv_heads > 0, "num_kv_heads must be positive."
        if num_heads % num_kv_heads != 0:
            return (
                False,
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}).",
            )

        o_dtype = out_dtype if out_dtype is not None else q_dtype
        heads_ratio = num_heads // num_kv_heads if num_kv_heads > 0 else 1

        if phase in ("context", "both"):
            if head_size in cls.UNSUPPORTED_HEAD_SIZES_CONTEXT:
                return False, f"[Context] Head size {head_size} is not supported."
            try:
                if AttentionMaskType(mask_type) == AttentionMaskType.custom_mask:
                    return False, "[Context] Custom mask is not supported."
            except ValueError:
                return False, f"[Context] Invalid mask_type: {mask_type}."
            if has_alibi:
                return False, "[Context] ALiBi is not supported."
            if is_padded:
                return False, "[Context] Padded input is not supported."
            if (q_dtype, kv_cache_dtype, o_dtype) not in cls.SUPPORTED_DTYPE_COMBOS_CONTEXT:
                return False, (
                    f"[Context] Unsupported dtype combination: Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )

        if phase in ("generation", "both"):
            if beam_width != 1:
                return (
                    False,
                    f"[Generation] Beam search (beam_width={beam_width}) is not supported. Must be 1.",
                )
            if position_shift_enabled:
                return False, "[Generation] Position shift is not supported."
            if sink_token_length != 0:
                return (
                    False,
                    f"[Generation] StreamingLLM (sink_token_length={sink_token_length}) is not supported.",
                )
            if tokens_per_block < cls.MIN_TOKENS_PER_BLOCK:
                return (
                    False,
                    f"[Generation] tokens_per_block ({tokens_per_block}) must be >= {cls.MIN_TOKENS_PER_BLOCK}.",
                )
            if heads_ratio > cls.MAX_HEADS_RATIO_GENERATION:
                max_ratio = cls.MAX_HEADS_RATIO_GENERATION
                return (
                    False,
                    f"[Generation] num_heads/num_kv_heads ratio "
                    f"({heads_ratio}) must be <= {max_ratio}.",
                )
            if has_alibi:
                return False, "[Generation] ALiBi is not supported."
            if (q_dtype, kv_cache_dtype, o_dtype) not in cls.SUPPORTED_DTYPE_COMBOS_GENERATION:
                return False, (
                    f"[Generation] Unsupported dtype combination: Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )

        if use_paged_kv_cache:
            if tokens_per_block <= 0:
                return False, "tokens_per_block must be positive."
            if tokens_per_block & (tokens_per_block - 1) != 0:
                return False, f"tokens_per_block ({tokens_per_block}) must be power of 2."
            if tokens_per_block not in cls.SUPPORTED_TOKENS_PER_BLOCK:
                supported = sorted(cls.SUPPORTED_TOKENS_PER_BLOCK)
                return (
                    False,
                    f"tokens_per_block ({tokens_per_block}) is not supported "
                    f"by trtllm-gen kernels. Supported: {supported}.",
                )

        return True, ""


@dataclass
class ContextWorkspaceBuffers:
    """
    Workspace buffers for context phase.
    """

    # Trtllm-gen workspace
    trtllm_gen_workspace: Optional[torch.Tensor] = None

    # Attention mask (only for unfused MHA)
    attention_mask: Optional[torch.Tensor] = None

    # Cumulative sequence lengths
    cu_q_seqlens: Optional[torch.Tensor] = None
    cu_kv_seqlens: Optional[torch.Tensor] = None
    cu_mask_rows: Optional[torch.Tensor] = None

    # Rotary embedding inverse frequencies
    rotary_inv_freq_buf: Optional[torch.Tensor] = None

    # Q/K/V buffers
    q_buf: Optional[torch.Tensor] = None
    k_buf: Optional[torch.Tensor] = None
    v_buf: Optional[torch.Tensor] = None

    # Token info: (batch_idx, token_idx_in_seq) per token
    tokens_info: Optional[torch.Tensor] = None

    # FMHA scheduler
    fmha_tile_counter: Optional[torch.Tensor] = None

    # BMM scales for FP8
    fmha_bmm1_scale: Optional[torch.Tensor] = None
    fmha_bmm2_scale: Optional[torch.Tensor] = None


@dataclass
class GenerationWorkspaceBuffers:
    """
    Workspace buffers for generation phase.
    """

    # Trtllm-gen workspace
    trtllm_gen_workspace: Optional[torch.Tensor] = None

    # Multi-block attention partial buffers
    partial_out: Optional[torch.Tensor] = None
    partial_sum: Optional[torch.Tensor] = None
    partial_max: Optional[torch.Tensor] = None

    # Shift K cache (for position shift)
    shift_k_cache: Optional[torch.Tensor] = None

    # Multi-block metadata
    max_num_seq_len_tiles: int = 0
    enable_multi_block: bool = False

    # Cumulative sequence lengths
    cu_seqlens: Optional[torch.Tensor] = None
    cu_kv_seqlens: Optional[torch.Tensor] = None

    # Rotary embedding inverse frequencies
    rotary_inv_freq: Optional[torch.Tensor] = None

    # Token info
    tokens_info: Optional[torch.Tensor] = None

    # Query buffer (FLAT output buffer for qkv_preprocessing)
    q_buf: Optional[torch.Tensor] = None

    # BMM scales
    bmm1_scale: Optional[torch.Tensor] = None
    bmm2_scale: Optional[torch.Tensor] = None

    # Sparse attention cache
    sparse_attn_cache: Optional[torch.Tensor] = None


class WorkspaceManager:
    """
    Manages workspace allocation for attention computation.
    """

    # Alignment for workspace buffers (256 bytes) (same as C++ kCudaMemAlign constant).
    ALIGNMENT = 256

    # Flashinfer's API requires 128MB workspace size for trtllm-gen backend,
    # but we use 32MB for now.
    #
    # 128MB is extremely conservative: it originates as FlashInfer's universal
    # recommendation, designed to cover all backends (FA2/FA3 split-k paths have
    # different memory patterns), all architectures, and all configurations. For a
    # trtllm-gen-only scenario, 32MB is more than enough to cover any generation +
    # context configuration.
    TRTLLM_GEN_WORKSPACE_SIZE = 32 * 1024 * 1024

    @staticmethod
    def _align_size(size: int) -> int:
        """Align size to kernel alignment requirement."""
        alignment = WorkspaceManager.ALIGNMENT
        if (size % alignment) != 0:
            size += alignment - (size % alignment)
        return size

    @classmethod
    def _calculate_total_workspace_size(cls, workspaces: List[int], alignment: int = None) -> int:
        """
        Calculate total workspace size with alignment.

        Args:
            workspaces: List of workspace sizes.
            alignment: Alignment boundary (default: ALIGNMENT).

        Returns:
            Total aligned workspace size.
        """
        if alignment is None:
            alignment = cls.ALIGNMENT
        total = 0
        for ws in workspaces:
            total += ws
            if ws % alignment:
                total += alignment - (ws % alignment)
        return total

    @classmethod
    def get_context_workspace_size(
        cls,
        dtype: torch.dtype,
        max_num_seq: int,
        max_num_tokens: int,
        num_heads: int,
        head_size: int,
        rotary_embedding_dim: int = 0,
        num_kv_heads: int = None,
        input_seq_length: int = 0,
        cross_kv_length: int = 0,
        enable_context_fmha: bool = True,
        is_cross_attention: bool = False,
        separate_q_kv_input: bool = True,
        fp8_context_fmha: bool = False,
        fp8_context_mla: bool = False,
        is_mla_enabled: bool = False,
        use_sparse_mla: bool = False,
        mla_qk_rope_head_dim: int = 0,
        mla_qk_nope_head_dim: int = 0,
        mla_v_head_dim: int = 0,
        mla_kv_lora_rank: int = 0,
        chunk_prefill_buffer_batch_size: int = 1,
    ) -> int:
        """
        Calculate workspace size for context phase.

        Args:
            dtype: Data type for attention computation.
            max_num_seq: Maximum number of sequences (batch_size).
            max_num_tokens: Maximum number of tokens.
            num_heads: Number of query attention heads.
            head_size: Size of each attention head.
            rotary_embedding_dim: Rotary embedding dimension (0 if not used).
            num_kv_heads: Number of KV heads (defaults to num_heads if None).
            input_seq_length: Maximum input sequence length.
            cross_kv_length: Cross attention KV length (for encoder-decoder).
            enable_context_fmha: Whether context FMHA is enabled.
            is_cross_attention: Whether this is cross attention.
            separate_q_kv_input: Whether Q and KV inputs are separate (paged FMHA).
            fp8_context_fmha: Whether FP8 context FMHA is used.
            fp8_context_mla: Whether FP8 context MLA is used.
            is_mla_enabled: Whether Multi-head Latent Attention is enabled.
            use_sparse_mla: Whether sparse MLA (absorption mode) is used.
            mla_qk_rope_head_dim: MLA QK RoPE head dimension.
            mla_qk_nope_head_dim: MLA QK non-positional head dimension.
            mla_v_head_dim: MLA value head dimension.
            mla_kv_lora_rank: MLA KV LoRA rank.
            chunk_prefill_buffer_batch_size: Batch size for chunked prefill buffer.

        Returns:
            Workspace size in bytes.
        """
        if max_num_tokens == 0:
            return 0

        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Convert torch dtype to binding dtype for get_size_in_bytes
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)

        local_hidden_units_qo = num_heads * head_size
        local_hidden_units_kv = num_kv_heads * head_size

        batch_size = max_num_seq
        kv_seq_length = cross_kv_length if is_cross_attention else input_seq_length

        # Attention mask size (only for unfused MHA)
        attention_mask_size = (
            0 if enable_context_fmha else dtype_size * max_num_tokens * kv_seq_length
        )

        # Cumulative sequence lengths: sizeof(int) * (batch_size + 1)
        cu_seqlens_size = 4 * (batch_size + 1)

        # Rotary inv freq buffer: sizeof(float) * batch_size * rotary_dim / 2
        rotary_inv_freq_size = (
            4 * batch_size * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )

        # Q buffer size calculation
        q_buf_2_size = 0
        if not enable_context_fmha:
            # Unfused MHA
            q_buf_2_size = dtype_size * batch_size * input_seq_length * local_hidden_units_qo
        elif separate_q_kv_input:
            # Paged context FMHA
            q_buf_2_size = (
                (1 if fp8_context_fmha else dtype_size) * max_num_tokens * local_hidden_units_qo
            )

        # K, V buffers (only for unfused MHA)
        k_buf_2_size = (
            0
            if enable_context_fmha
            else dtype_size * batch_size * kv_seq_length * local_hidden_units_kv
        )
        v_buf_2_size = (
            0
            if enable_context_fmha
            else dtype_size * batch_size * kv_seq_length * local_hidden_units_kv
        )

        # Tokens info: (batch_idx, token_idx_in_seq) per token - sizeof(int2) = 8
        tokens_info_size = 8 * max_num_tokens

        # FMHA scheduler counter
        fmha_scheduler_counter = 4 if enable_context_fmha else 0  # sizeof(uint32_t)

        # BMM scales for FP8
        fmha_bmm1_scale_size = 4 * 2 if fp8_context_fmha else 0  # sizeof(float) * 2
        fmha_bmm2_scale_size = 4 if fp8_context_fmha else 0  # sizeof(float)

        # Build workspace array
        workspaces = [
            cls.TRTLLM_GEN_WORKSPACE_SIZE,  # 0
            attention_mask_size,  # 1
            cu_seqlens_size,  # 2: cu_seqlen_q
            cu_seqlens_size,  # 3: cu_seqlen_kv
            cu_seqlens_size,  # 4: cu_mask_rows
            rotary_inv_freq_size,  # 5
            q_buf_2_size,  # 6
            k_buf_2_size,  # 7
            v_buf_2_size,  # 8
            tokens_info_size,  # 9
            fmha_scheduler_counter,  # 10
            fmha_bmm1_scale_size,  # 11
            fmha_bmm2_scale_size,  # 12
        ]

        return cls._calculate_total_workspace_size(workspaces)

    @classmethod
    def get_generation_workspace_size(
        cls,
        dtype: torch.dtype,
        max_num_seq: int,
        max_num_tokens: int,
        num_heads: int,
        head_size: int,
        multi_processor_count: int,
        num_kv_heads: int = None,
        max_attention_window_size: int = 0,
        position_shift_enabled: bool = False,
        is_cross_attention: bool = False,
    ) -> int:
        """
        Calculate workspace size for generation (decode) phase.

        Args:
            dtype: Data type for attention computation.
            max_num_seq: Maximum number of sequences (batch_beam).
            max_num_tokens: Maximum number of tokens.
            num_heads: Number of query attention heads.
            head_size: Size of each attention head.
            multi_processor_count: Number of GPU SMs.
            num_kv_heads: Number of KV heads (defaults to num_heads).
            max_attention_window_size: Maximum attention window size.
            position_shift_enabled: Whether position shift is enabled.
            is_cross_attention: Whether this is cross attention.

        Returns:
            Workspace size in bytes.
        """
        if max_num_tokens == 0:
            return 0

        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Convert torch dtype to binding dtype
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)
        batch_beam = max_num_seq

        min_seq_len_tile = max(
            1, (max_attention_window_size + 1023) // 1024 if max_attention_window_size > 0 else 1
        )

        # Calculate max sequence length tile
        max_seq_len_tile = max(
            min_seq_len_tile,
            max(
                1, (multi_processor_count + batch_beam * num_heads - 1) // (batch_beam * num_heads)
            ),
        )
        max_seq_len_tile = max(max_seq_len_tile, 4)

        # Partial output/sum/max buffers for multi-block attention
        partial_out_size = dtype_size * batch_beam * num_heads * head_size * max_seq_len_tile
        partial_sum_size = 4 * batch_beam * num_heads * max_seq_len_tile
        partial_max_size = 4 * batch_beam * num_heads * max_seq_len_tile

        # Shift K cache size (for position shift)
        shift_k_cache_size = 0
        if position_shift_enabled and not is_cross_attention:
            shift_k_cache_size = (
                dtype_size * batch_beam * num_heads * head_size * max_attention_window_size
            )

        generation_workspaces = [
            cls.TRTLLM_GEN_WORKSPACE_SIZE,  # 0
            partial_out_size,
            partial_sum_size,
            partial_max_size,
            shift_k_cache_size,
        ]

        return cls._calculate_total_workspace_size(generation_workspaces)

    @classmethod
    def get_workspace_size(
        cls,
        dtype: torch.dtype,
        num_tokens: int,
        num_gen_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_num_requests: int,
        max_context_length: int,
        attention_window_size: int,
        rotary_embedding_dim: int = 0,
        position_shift_enabled: bool = False,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Calculate total workspace size.

        Returns max(context_workspace, generation_workspace).
        """
        if device is None:
            device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device=device)
        multi_processor_count = device_props.multi_processor_count

        if num_kv_heads <= 0:
            num_kv_heads = num_heads

        context_size = cls.get_context_workspace_size(
            dtype=dtype,
            max_num_seq=max_num_requests,
            max_num_tokens=num_tokens,
            num_heads=num_heads,
            head_size=head_size,
            rotary_embedding_dim=rotary_embedding_dim,
            num_kv_heads=num_kv_heads,
            input_seq_length=max_context_length,
            enable_context_fmha=True,
            separate_q_kv_input=True,
        )

        effective_window = (
            attention_window_size if attention_window_size > 0 else max_context_length
        )

        generation_size = cls.get_generation_workspace_size(
            dtype=dtype,
            max_num_seq=max_num_requests,
            max_num_tokens=num_gen_tokens,
            num_heads=num_heads,
            head_size=head_size,
            multi_processor_count=multi_processor_count,
            num_kv_heads=num_kv_heads,
            max_attention_window_size=effective_window,
            position_shift_enabled=position_shift_enabled,
        )

        min_workspace_size = cls.TRTLLM_GEN_WORKSPACE_SIZE
        return max(context_size, generation_size, min_workspace_size)

    @classmethod
    def _next_workspace_ptr(
        cls, base_offset: int, size: int, alignment: int = None
    ) -> Tuple[int, int]:
        """
        Calculate next workspace pointer offset with alignment.

        Args:
            base_offset: Current offset in workspace buffer.
            size: Size of the buffer to allocate.
            alignment: Alignment boundary.

        Returns:
            Tuple of (buffer_offset, new_base_offset).
        """
        if alignment is None:
            alignment = cls.ALIGNMENT
        curr_offset = base_offset
        next_offset = curr_offset + ((size + alignment - 1) // alignment) * alignment
        buffer_offset = curr_offset if size > 0 else -1  # -1 indicates nullptr
        return buffer_offset, next_offset

    @classmethod
    def _get_view(
        cls,
        workspace: torch.Tensor,
        buf_offset: int,
        size: int,
        view_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Get a view of the workspace buffer.

        Args:
            workspace: Workspace tensor.
            buf_offset: Buffer offset.
            size: Buffer size.
            view_dtype: View data type.

        Returns:
            View of the workspace buffer.
        """
        if workspace is None or buf_offset < 0 or size == 0:
            return None

        workspace_byte = workspace.view(torch.uint8)
        # Validate workspace bounds
        end_offset = buf_offset + size
        if end_offset > workspace_byte.numel():
            raise RuntimeError(
                f"[trtllm-gen] Split workspace buffer overflow! "
                f"Trying to access [{buf_offset}:{end_offset}] "
                f"but workspace size is only {workspace_byte.numel()} bytes. "
                f"Buffer size needed: {size} bytes"
            )
        return workspace_byte[buf_offset : buf_offset + size].view(view_dtype)

    @classmethod
    def split_context_workspace(
        cls,
        workspace: torch.Tensor,
        dtype: torch.dtype,
        batch_size: int,
        num_tokens: int,
        num_heads: int,
        head_size: int,
        rotary_embedding_dim: int = 0,
        num_kv_heads: int = None,
        input_seq_length: int = 0,
        cross_kv_length: int = 0,
        enable_context_fmha: bool = True,
        is_cross_attention: bool = False,
        separate_q_kv_input: bool = True,
        fp8_context_fmha: bool = False,
        fp8_context_mla: bool = False,
    ) -> ContextWorkspaceBuffers:
        """
        Split workspace buffer into sub-buffers for context phase.

        Args:
            workspace: Workspace tensor (contiguous byte buffer).
            dtype: Data type for attention computation.
            batch_size: Batch size.
            num_tokens: Number of tokens.
            num_heads: Number of query attention heads.
            head_size: Size of each attention head.
            rotary_embedding_dim: Rotary embedding dimension.
            num_kv_heads: Number of KV heads.
            input_seq_length: Input sequence length.
            cross_kv_length: Cross attention KV length.
            enable_context_fmha: Whether context FMHA is enabled.
            is_cross_attention: Whether this is cross attention.
            separate_q_kv_input: Whether Q and KV inputs are separate.
            fp8_context_fmha: Whether FP8 context FMHA is used.
            fp8_context_mla: Whether FP8 context MLA is used.

        Returns:
            Dictionary containing sub-buffer views and metadata:
            {
                'attention_mask': Tensor or None,
                'cu_q_seqlens': Tensor,
                'cu_kv_seqlens': Tensor,
                'cu_mask_rows': Tensor,
                'rotary_inv_freq_buf': Tensor or None,
                'q_buf': Tensor or None,
                'k_buf': Tensor or None,
                'v_buf': Tensor or None,
                'tokens_info': Tensor,
                'fmha_tile_counter': Tensor or None,
                'fmha_bmm1_scale': Tensor or None,
                'fmha_bmm2_scale': Tensor or None,
            }
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)

        local_hidden_units_qo = num_heads * head_size
        local_hidden_units_kv = num_kv_heads * head_size
        kv_seq_length = cross_kv_length if is_cross_attention else input_seq_length

        attention_mask_size = (
            0 if enable_context_fmha else dtype_size * batch_size * input_seq_length * kv_seq_length
        )
        cu_seqlens_size = 4 * (batch_size + 1)
        rotary_inv_freq_size = (
            4 * batch_size * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )

        # Q buffer size
        q_buf_2_size = 0
        if not enable_context_fmha:
            q_buf_2_size = dtype_size * batch_size * input_seq_length * local_hidden_units_qo
        elif separate_q_kv_input:
            q_buf_2_size = (
                (1 if fp8_context_fmha else dtype_size) * num_tokens * local_hidden_units_qo
            )

        k_buf_2_size = (
            0
            if enable_context_fmha
            else dtype_size * batch_size * kv_seq_length * local_hidden_units_kv
        )
        v_buf_2_size = (
            0
            if enable_context_fmha
            else dtype_size * batch_size * kv_seq_length * local_hidden_units_kv
        )

        tokens_info_size = 8 * num_tokens  # sizeof(int2)
        fmha_scheduler_counter = 4 if enable_context_fmha else 0
        fmha_bmm1_scale_size = 4 * 2 if (fp8_context_fmha or fp8_context_mla) else 0
        fmha_bmm2_scale_size = 4 if (fp8_context_fmha or fp8_context_mla) else 0

        offset = 0
        trtllm_gen_workspace, offset = cls._next_workspace_ptr(
            offset, cls.TRTLLM_GEN_WORKSPACE_SIZE
        )
        attention_mask_offset, offset = cls._next_workspace_ptr(offset, attention_mask_size)
        cu_q_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        cu_kv_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        cu_mask_rows_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        rotary_inv_freq_offset, offset = cls._next_workspace_ptr(offset, rotary_inv_freq_size)
        q_buf_offset, offset = cls._next_workspace_ptr(offset, q_buf_2_size)
        k_buf_offset, offset = cls._next_workspace_ptr(offset, k_buf_2_size)
        v_buf_offset, offset = cls._next_workspace_ptr(offset, v_buf_2_size)
        tokens_info_offset, offset = cls._next_workspace_ptr(offset, tokens_info_size)
        fmha_tile_counter_offset, offset = cls._next_workspace_ptr(offset, fmha_scheduler_counter)
        fmha_bmm1_scale_offset, offset = cls._next_workspace_ptr(offset, fmha_bmm1_scale_size)
        fmha_bmm2_scale_offset, offset = cls._next_workspace_ptr(offset, fmha_bmm2_scale_size)

        result = {
            "trtllm_gen_workspace": cls._get_view(
                workspace, trtllm_gen_workspace, cls.TRTLLM_GEN_WORKSPACE_SIZE, torch.uint8
            ),
            "attention_mask": cls._get_view(
                workspace, attention_mask_offset, attention_mask_size, dtype
            ),
            "cu_q_seqlens": cls._get_view(
                workspace, cu_q_seqlens_offset, cu_seqlens_size, torch.int32
            ),
            "cu_kv_seqlens": cls._get_view(
                workspace, cu_kv_seqlens_offset, cu_seqlens_size, torch.int32
            ),
            "cu_mask_rows": cls._get_view(
                workspace, cu_mask_rows_offset, cu_seqlens_size, torch.int32
            ),
            "rotary_inv_freq_buf": cls._get_view(
                workspace, rotary_inv_freq_offset, rotary_inv_freq_size, torch.float32
            ),
            "q_buf": cls._get_view(workspace, q_buf_offset, q_buf_2_size, dtype)
            if not fp8_context_fmha
            else cls._get_view(workspace, q_buf_offset, q_buf_2_size, torch.uint8),
            "k_buf": cls._get_view(workspace, k_buf_offset, k_buf_2_size, dtype),
            "v_buf": cls._get_view(workspace, v_buf_offset, v_buf_2_size, dtype),
            "tokens_info": cls._get_view(
                workspace, tokens_info_offset, tokens_info_size, torch.int32
            ),  # int2 as 2 x int32
            "fmha_tile_counter": cls._get_view(
                workspace, fmha_tile_counter_offset, fmha_scheduler_counter, torch.uint32
            ),
            "fmha_bmm1_scale": cls._get_view(
                workspace, fmha_bmm1_scale_offset, fmha_bmm1_scale_size, torch.float32
            ),
            "fmha_bmm2_scale": cls._get_view(
                workspace, fmha_bmm2_scale_offset, fmha_bmm2_scale_size, torch.float32
            ),
        }

        return ContextWorkspaceBuffers(**result)

    @classmethod
    def split_generation_workspace(
        cls,
        workspace: torch.Tensor,
        dtype: torch.dtype,
        batch_beam: int,
        num_tokens: int,
        num_heads: int,
        head_size: int,
        multi_processor_count: int,
        rotary_embedding_dim: int = 0,
        num_kv_heads: int = None,
        max_attention_window_size: int = 0,
        max_blocks_per_sequence: int = 0,
        max_past_kv_length: int = 0,
        cyclic_attention_window_size: int = 0,
        position_shift_enabled: bool = False,
        is_cross_attention: bool = False,
        multi_block_mode: bool = True,
        use_sparse_attention: bool = False,
    ) -> GenerationWorkspaceBuffers:
        """
        Split workspace buffer into sub-buffers for generation phase.

        Args:
            workspace: Workspace tensor (contiguous byte buffer).
            dtype: Data type for attention computation.
            batch_beam: Batch size * beam width.
            num_tokens: Number of tokens.
            num_heads: Number of query attention heads.
            head_size: Size of each attention head.
            multi_processor_count: Number of GPU SMs.
            rotary_embedding_dim: Rotary embedding dimension.
            num_kv_heads: Number of KV heads.
            max_attention_window_size: Maximum attention window size.
            max_blocks_per_sequence: Maximum blocks per sequence.
            max_past_kv_length: Maximum past KV length.
            cyclic_attention_window_size: Cyclic attention window size.
            position_shift_enabled: Whether position shift is enabled.
            is_cross_attention: Whether this is cross attention.
            multi_block_mode: Whether multi-block mode is enabled.
            use_sparse_attention: Whether sparse attention is used.

        Returns:
            GenerationWorkspaceBuffers containing sub-buffer views:
            {
                'partial_out': Tensor or None,
                'partial_sum': Tensor or None,
                'partial_max': Tensor or None,
                'shift_k_cache': Tensor or None,
                'cu_seqlens': Tensor or None,
                'cu_kv_seqlens': Tensor or None,
                'rotary_inv_freq': Tensor or None,
                'tokens_info': Tensor or None,
                'bmm1_scale': Tensor or None,
                'bmm2_scale': Tensor or None,
                'sparse_attn_cache': Tensor or None,
            }
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Get dtype size
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)

        max_timesteps = (
            min(max_past_kv_length, cyclic_attention_window_size)
            if cyclic_attention_window_size > 0
            else max_past_kv_length
        )
        estimated_min_multi_block_count = max(
            1, (max_timesteps + 1023) // 1024 if max_timesteps > 0 else 1
        )

        max_num_seq_len_tiles = max(
            max(
                1, (multi_processor_count + batch_beam * num_heads - 1) // (batch_beam * num_heads)
            ),
            estimated_min_multi_block_count,
        )
        max_num_seq_len_tiles = max(max_num_seq_len_tiles, 4)

        enable_multi_block = (
            multi_block_mode and max_num_seq_len_tiles > 1
        ) or estimated_min_multi_block_count > 1

        partial_out_size = (
            dtype_size * batch_beam * num_heads * head_size * max_num_seq_len_tiles
            if enable_multi_block
            else 0
        )
        partial_sum_size = (
            4 * batch_beam * num_heads * max_num_seq_len_tiles if enable_multi_block else 0
        )
        partial_max_size = (
            4 * batch_beam * num_heads * max_num_seq_len_tiles if enable_multi_block else 0
        )
        shift_k_cache_size = (
            0
            if (not position_shift_enabled or is_cross_attention)
            else dtype_size * batch_beam * num_heads * head_size * max_attention_window_size
        )

        cu_seqlens_size = 4 * (batch_beam + 1)
        cu_kv_seqlens_size = 4 * (batch_beam + 1)
        rotary_inv_freq_size = (
            4 * batch_beam * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )
        tokens_info_size = 8 * num_tokens
        q_buf_size = dtype_size * num_tokens * num_heads * head_size
        bmm1_scale_size = 4 * 2
        bmm2_scale_size = 4
        sparse_attn_cache_size = (
            4 * (batch_beam + batch_beam * 2 * max_blocks_per_sequence) * num_kv_heads
            if use_sparse_attention
            else 0
        )

        # Workspace pointer allocation
        offset = 0

        trtllm_gen_workspace, offset = cls._next_workspace_ptr(
            offset, cls.TRTLLM_GEN_WORKSPACE_SIZE
        )

        # Multi-block attention buffers
        partial_out_offset, offset = cls._next_workspace_ptr(offset, partial_out_size)
        partial_sum_offset, offset = cls._next_workspace_ptr(offset, partial_sum_size)
        partial_max_offset, offset = cls._next_workspace_ptr(offset, partial_max_size)
        shift_k_cache_offset, offset = cls._next_workspace_ptr(offset, shift_k_cache_size)

        cu_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        cu_kv_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_kv_seqlens_size)
        rotary_inv_freq_offset, offset = cls._next_workspace_ptr(offset, rotary_inv_freq_size)
        tokens_info_offset, offset = cls._next_workspace_ptr(offset, tokens_info_size)
        q_buf_offset, offset = cls._next_workspace_ptr(offset, q_buf_size)
        bmm1_scale_offset, offset = cls._next_workspace_ptr(offset, bmm1_scale_size)
        bmm2_scale_offset, offset = cls._next_workspace_ptr(offset, bmm2_scale_size)
        sparse_attn_cache_offset, offset = cls._next_workspace_ptr(offset, sparse_attn_cache_size)

        result = {
            "trtllm_gen_workspace": cls._get_view(
                workspace, trtllm_gen_workspace, cls.TRTLLM_GEN_WORKSPACE_SIZE, torch.uint8
            ),
            # Multi-block attention buffers
            "partial_out": cls._get_view(workspace, partial_out_offset, partial_out_size, dtype),
            "partial_sum": cls._get_view(
                workspace, partial_sum_offset, partial_sum_size, torch.float32
            ),
            "partial_max": cls._get_view(
                workspace, partial_max_offset, partial_max_size, torch.float32
            ),
            "shift_k_cache": cls._get_view(
                workspace, shift_k_cache_offset, shift_k_cache_size, dtype
            ),
            # Multi-block metadata
            "max_num_seq_len_tiles": max_num_seq_len_tiles,
            "enable_multi_block": enable_multi_block,
            "cu_seqlens": cls._get_view(workspace, cu_seqlens_offset, cu_seqlens_size, torch.int32),
            "cu_kv_seqlens": cls._get_view(
                workspace, cu_kv_seqlens_offset, cu_kv_seqlens_size, torch.int32
            ),
            "rotary_inv_freq": cls._get_view(
                workspace, rotary_inv_freq_offset, rotary_inv_freq_size, torch.float32
            ),
            "tokens_info": cls._get_view(
                workspace, tokens_info_offset, tokens_info_size, torch.int32
            ),
            "q_buf": cls._get_view(workspace, q_buf_offset, q_buf_size, dtype),
            "bmm1_scale": cls._get_view(
                workspace, bmm1_scale_offset, bmm1_scale_size, torch.float32
            ),
            "bmm2_scale": cls._get_view(
                workspace, bmm2_scale_offset, bmm2_scale_size, torch.float32
            ),
            "sparse_attn_cache": cls._get_view(
                workspace, sparse_attn_cache_offset, sparse_attn_cache_size, torch.int32
            ),
        }

        return GenerationWorkspaceBuffers(**result)


@dataclass
class EnqueueParams:
    attention_input: Optional[torch.Tensor] = None
    qkv_input: Optional[torch.Tensor] = None
    context_buf: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    host_context_lengths: Optional[torch.Tensor] = None
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0
    max_cyclic_attention_window_size: int = 0
    sink_token_length: int = 0
    num_tokens: int = 0
    total_kv_len: int = 0
    max_blocks_per_sequence: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_size: int = 0
    rotary_embedding_dim: int = 0
    rotary_embedding_base: float = 0.0
    rotary_embedding_scale_type: int = 0
    rotary_embedding_scale: float = 1.0
    rotary_embedding_max_positions: int = 0
    kv_cache_block_offsets: Optional[torch.Tensor] = None
    host_kv_cache_pool_pointers: Optional[torch.Tensor] = None
    host_kv_cache_pool_mapping: Optional[torch.Tensor] = None
    block_ids_per_seq: Optional[torch.Tensor] = None
    pool_index: int = 0
    seq_offset: int = 0
    request_ids: Optional[List] = None
    tokens_per_block: int = 64
    mask_type: int = 1
    kv_cache_quant_mode: int = 0
    position_embedding_type: int = 0
    layer_idx: int = 0
    kv_scale_orig_quant: Optional[torch.Tensor] = None
    kv_scale_quant_orig: Optional[torch.Tensor] = None
    attention_output_orig_quant: Optional[torch.Tensor] = None
    bmm1_scale: float = 1.0
    bmm2_scale: float = 1.0


@dataclass
class EnqueueContextParams(EnqueueParams):
    batch_size: int = 0
    max_context_length: int = 0
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None


@dataclass
class EnqueueGenerationParams(EnqueueParams):
    beam_width: int = 1
    num_requests: int = 0
    host_past_key_value_lengths: Optional[torch.Tensor] = None
    cache_indir: Optional[torch.Tensor] = None


class FlashInferTrtllmGenAttention:
    """
    An attention backend using pure trtllm-gen kernels from flashinfer.
    """

    def __init__(
        self,
        kv_cache_manager: Optional[KVCacheManager] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        self._checker = TrtllmGenSupportChecker()
        self._layout = DEFAULT_KV_LAYOUT
        self._kv_cache_manager = kv_cache_manager
        self._quant_config = quant_config

    @property
    def layout(self) -> str:
        """KV cache layout."""
        return self._layout

    def is_supported(self, phase: str = "both", **kwargs) -> Tuple[bool, str]:
        if not IS_FLASHINFER_AVAILABLE:
            return False, "flashinfer package is not installed."
        return self._checker.is_supported(phase=phase, **kwargs)

    def run_context(self, params: EnqueueContextParams):
        block_tables = (
            params.block_ids_per_seq[params.seq_offset : params.seq_offset + params.batch_size]
            if params.block_ids_per_seq is not None
            else None
        )
        window_left = (
            params.cyclic_attention_window_size
            if 0 < params.cyclic_attention_window_size < params.max_context_length
            else -1
        )

        ctx_ws = WorkspaceManager.split_context_workspace(
            workspace=params.workspace,
            dtype=params.attention_input.dtype,
            batch_size=params.batch_size,
            num_tokens=params.num_tokens,
            num_heads=params.num_heads,
            head_size=params.head_size,
            rotary_embedding_dim=params.rotary_embedding_dim,
            num_kv_heads=params.num_kv_heads,
            input_seq_length=params.input_seq_length,
            enable_context_fmha=True,
            separate_q_kv_input=True,
            fp8_context_fmha=False,
        )

        torch.ops.trtllm.build_decoder_info(
            seq_q_offsets=ctx_ws.cu_q_seqlens,
            seq_kv_offsets=ctx_ws.cu_kv_seqlens,
            padding_offsets=None,
            tokens_info=ctx_ws.tokens_info,
            encoder_padding_offsets=None,
            packed_mask_row_offsets=ctx_ws.cu_mask_rows,
            seq_cp_partial_offsets=None,
            attention_mask=ctx_ws.attention_mask,
            seq_q_lengths=params.context_lengths,
            seq_kv_lengths=params.sequence_lengths,
            cp_size=1,
            fmha_tile_counter=ctx_ws.fmha_tile_counter,
            dequant_scale_qkv=params.kv_scale_quant_orig,
            separate_qkv_scales=False,
            quant_scale_o=params.attention_output_orig_quant,
            fmha_host_bmm1_scale=params.bmm1_scale,
            fmha_bmm1_scale=ctx_ws.fmha_bmm1_scale,
            fmha_bmm2_scale=ctx_ws.fmha_bmm2_scale,
            batch_size=params.batch_size,
            max_q_seq_length=params.input_seq_length,
            max_encoder_q_seq_length=0,
            attention_window_size=params.cyclic_attention_window_size,
            sink_token_length=params.sink_token_length,
            num_tokens=params.num_tokens,
            remove_padding=True,
            attention_mask_type=params.mask_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_scaling_type=params.rotary_embedding_scale_type,
            rotary_embedding_inv_freq=None,
            rotary_embedding_inv_freq_cache=None,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
        )

        has_kv_cache_quant = params.kv_cache_quant_mode != 0
        preprocessing_params = dict(
            qkv_input=params.qkv_input,
            cross_kv_input=None,
            quantized_qkv_output=None,
            q_output=ctx_ws.q_buf,
            kv_cache_block_offsets=params.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
            qkv_bias=None,
            tokens_info=ctx_ws.tokens_info,
            seq_lens=params.context_lengths,
            cache_seq_lens=params.sequence_lengths,
            encoder_seq_lens=None,
            cu_seq_lens=ctx_ws.cu_q_seqlens,
            cu_kv_seq_lens=ctx_ws.cu_kv_seqlens,
            rotary_embedding_inv_freq=None,
            rotary_coef_cache_buffer=None,
            mrope_rotary_cos_sin=None,
            qkv_scale_orig_quant=params.kv_scale_orig_quant,
            spec_decoding_position_offsets=None,
            logn_scaling=None,
            sparse_kv_offsets=None,
            sparse_kv_indices=None,
            batch_size=params.batch_size,
            max_input_seq_len=params.input_seq_length,
            max_kv_seq_len=params.max_past_kv_length,
            cyclic_kv_cache_len=params.cyclic_attention_window_size,
            sink_token_len=params.sink_token_length,
            token_num=params.num_tokens,
            remove_padding=True,
            cross_attention=False,
            head_num=params.num_heads,
            kv_head_num=params.num_kv_heads,
            qheads_per_kv_head=params.num_heads // params.num_kv_heads,
            size_per_head=params.head_size,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_scale_type=params.rotary_embedding_scale_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
            position_embedding_type=params.position_embedding_type,
            position_shift_enabled=False,
            cache_type=0,
            separate_q_kv_output=True,
            quantized_fp8_output=False,
            generation_phase=False,
            rotary_vision_start=0,
            rotary_vision_length=0,
            is_last_chunk=True,
            qkv_scale_quant_orig=params.kv_scale_quant_orig if has_kv_cache_quant else None,
            o_scale_orig_quant=params.attention_output_orig_quant,
            fmha_bmm1_scale=ctx_ws.fmha_bmm1_scale,
            fmha_bmm2_scale=ctx_ws.fmha_bmm2_scale,
            fmha_host_bmm1_scale=params.bmm1_scale,
            fmha_tile_counter=ctx_ws.fmha_tile_counter,
            mrope_position_deltas=None,
            layer_idx=params.layer_idx,
            tokens_per_block=params.tokens_per_block,
            max_attention_window_size=params.max_attention_window_size,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
        )

        torch.ops.trtllm.qkv_preprocessing(**preprocessing_params)

        q_processed = ctx_ws.q_buf.view(params.num_tokens, params.num_heads, params.head_size)

        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q_processed,
            kv_cache=self._kv_cache_manager.get_buffers(
                params.layer_idx, kv_layout=DEFAULT_KV_LAYOUT
            ),
            workspace_buffer=ctx_ws.trtllm_gen_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_q_len=params.input_seq_length,
            max_kv_len=params.max_past_kv_length,
            bmm1_scale=params.bmm1_scale,
            bmm2_scale=params.bmm2_scale,
            batch_size=params.batch_size,
            cum_seq_lens_q=ctx_ws.cu_q_seqlens,
            cum_seq_lens_kv=ctx_ws.cu_kv_seqlens,
            window_left=window_left,
            out=params.context_buf,
            kv_layout=self._layout,
        )

        torch.ops.trtllm.kv_cache_postprocessing(**preprocessing_params)

    def run_generation(self, params: EnqueueGenerationParams):
        block_tables = (
            params.block_ids_per_seq[
                params.seq_offset : params.seq_offset + params.num_requests * params.beam_width
            ]
            if params.block_ids_per_seq is not None
            else None
        )
        window_left = (
            params.cyclic_attention_window_size
            if 0 < params.cyclic_attention_window_size < params.max_past_kv_length
            else -1
        )

        gen_ws = WorkspaceManager.split_generation_workspace(
            workspace=params.workspace,
            dtype=params.attention_input.dtype,
            batch_beam=params.num_requests * params.beam_width,
            num_tokens=params.num_tokens,
            num_heads=params.num_heads,
            head_size=params.head_size,
            multi_processor_count=torch.cuda.get_device_properties(
                params.attention_input.device
            ).multi_processor_count,
            num_kv_heads=params.num_kv_heads,
            max_attention_window_size=params.max_attention_window_size,
        )

        # For standard generation (single token, no dynamic rotary, no bmm scales),
        # build_decoder_info is skipped — GEN_PHASE kernel uses
        # batch_idx = global_token_idx directly.
        need_build_decoder_info = (
            params.input_seq_length > 1 and (params.num_requests * params.beam_width) > 1
        ) or params.rotary_embedding_scale_type != 0

        if need_build_decoder_info:
            torch.ops.trtllm.build_decoder_info(
                seq_q_offsets=gen_ws.cu_seqlens,
                seq_kv_offsets=gen_ws.cu_kv_seqlens,
                seq_q_lengths=params.context_lengths,
                seq_kv_lengths=params.sequence_lengths,
                batch_size=params.num_requests * params.beam_width,
                max_q_seq_length=params.input_seq_length,
                num_tokens=params.num_tokens,
                remove_padding=True,
                rotary_embedding_scale=params.rotary_embedding_scale,
                rotary_embedding_base=params.rotary_embedding_base,
                rotary_embedding_dim=params.rotary_embedding_dim,
                rotary_scaling_type=params.rotary_embedding_scale_type,
                rotary_embedding_inv_freq=None,
                rotary_embedding_inv_freq_cache=None,
                rotary_embedding_max_positions=params.rotary_embedding_max_positions,
                padding_offsets=None,
                tokens_info=None,
                encoder_padding_offsets=None,
                packed_mask_row_offsets=None,
                seq_cp_partial_offsets=None,
                attention_mask=None,
                cp_size=1,
                fmha_tile_counter=None,
                dequant_scale_qkv=None,
                separate_qkv_scales=False,
                quant_scale_o=None,
                fmha_host_bmm1_scale=0.0,
                fmha_bmm1_scale=None,
                fmha_bmm2_scale=None,
                max_encoder_q_seq_length=0,
                attention_window_size=0,
                sink_token_length=0,
                attention_mask_type=0,
            )

        has_kv_cache_quant = params.kv_cache_quant_mode != 0
        preprocessing_params = dict(
            qkv_input=params.qkv_input,
            q_output=gen_ws.q_buf,
            kv_cache_block_offsets=params.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
            qkv_bias=None,
            fmha_bmm1_scale=gen_ws.bmm1_scale,
            fmha_bmm2_scale=gen_ws.bmm2_scale,
            qkv_scale_quant_orig=params.kv_scale_quant_orig if has_kv_cache_quant else None,
            o_scale_orig_quant=params.attention_output_orig_quant,
            logn_scaling=None,
            seq_lens=params.context_lengths,
            cache_seq_lens=params.sequence_lengths,
            cu_seq_lens=gen_ws.cu_seqlens,
            cu_kv_seq_lens=gen_ws.cu_kv_seqlens,
            rotary_embedding_inv_freq=None,
            rotary_coef_cache_buffer=None,
            qkv_scale_orig_quant=params.kv_scale_orig_quant,
            spec_decoding_position_offsets=None,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            batch_size=params.num_requests * params.beam_width,
            max_input_seq_len=params.input_seq_length,
            max_kv_seq_len=params.max_past_kv_length,
            cyclic_kv_cache_len=params.cyclic_attention_window_size,
            sink_token_len=params.sink_token_length,
            token_num=params.num_tokens,
            remove_padding=True,
            cross_attention=False,
            head_num=params.num_heads,
            kv_head_num=params.num_kv_heads,
            qheads_per_kv_head=params.num_heads // params.num_kv_heads,
            size_per_head=params.head_size,
            fmha_host_bmm1_scale=params.bmm1_scale,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_scale_type=params.rotary_embedding_scale_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
            position_embedding_type=params.position_embedding_type,
            position_shift_enabled=False,
            cache_type=0,
            separate_q_kv_output=True,
            quantized_fp8_output=False,
            generation_phase=True,
            rotary_vision_start=0,
            rotary_vision_length=0,
            cross_kv_input=None,
            quantized_qkv_output=None,
            encoder_seq_lens=None,
            fmha_tile_counter=None,
            tokens_info=gen_ws.tokens_info if need_build_decoder_info else None,
            sparse_kv_offsets=None,
            sparse_kv_indices=None,
            is_last_chunk=True,
            layer_idx=params.layer_idx,
            tokens_per_block=params.tokens_per_block,
            max_attention_window_size=params.max_attention_window_size,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
        )

        torch.ops.trtllm.qkv_preprocessing(**preprocessing_params)

        q_processed = gen_ws.q_buf.view(params.num_tokens, params.num_heads, params.head_size)

        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q_processed,
            kv_cache=self._kv_cache_manager.get_buffers(params.layer_idx, kv_layout=self._layout),
            workspace_buffer=gen_ws.trtllm_gen_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_seq_len=params.max_past_kv_length,
            out=params.context_buf,
            bmm1_scale=params.bmm1_scale,
            bmm2_scale=params.bmm2_scale,
            window_left=window_left,
            kv_layout=self._layout,
        )


def _parse_request_types(host_request_types: torch.Tensor) -> Tuple[int, int]:
    """
    Parse request types to count context and generation requests.

    Args:
        host_request_types: Request types tensor (0=context, 1=generation).
        num_seqs: Total number of sequences.

    Returns:
        Tuple of (num_contexts, num_generations).
    """

    num_generations = host_request_types.sum().item()
    num_contexts = host_request_types.size(0) - num_generations
    return num_contexts, num_generations


def is_supported(
    q: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
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
    is_spec_decoding: bool = False,
    is_mla_enable: bool = False,
    is_fused_qkv: bool = True,
    update_kv_cache: bool = True,
    has_cross_kv: bool = False,
    quant_config: Optional[QuantConfig] = None,
    kv_cache_manager: Optional[KVCacheManager] = None,
    phase: str = "both",
) -> Tuple[bool, str]:
    """
    Check if trtllm-gen backend supports the given configuration.

    This is the compatibility function that wraps TrtllmGenSupportChecker.

    Args:
        q: Query tensor.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads.
        head_size: Size of each attention head.
        out_dtype: Output data type.
        mask_type: Attention mask type.
        has_alibi: Whether ALiBi is used.
        is_padded: Whether input is padded.
        use_paged_kv_cache: Whether paged KV cache is used.
        tokens_per_block: Tokens per KV cache block.
        beam_width: Beam search width.
        position_shift_enabled: Whether position shift is enabled.
        sink_token_length: Sink token length for StreamingLLM.
        cross_attention: Whether cross attention is used.
        is_spec_decoding: Whether speculative decoding is enabled.
        is_mla_enable: Whether MLA is enabled.
        is_fused_qkv: Whether QKV is fused.
        update_kv_cache: Whether KV cache update is enabled.
        has_cross_kv: Whether cross KV is provided.
        quant_config: Quantization configuration (QuantConfig). If provided,
                     will automatically determine kv_cache_dtype based on
                     has_fp8_kv_cache() or has_fp4_kv_cache().
        kv_cache_manager: KV cache manager.
        phase: Phase to check ("context", "generation", or "both").

    Returns:
        Tuple of (is_supported, reason_if_not_supported).
    """
    kv_cache_dtype = q.dtype
    if kv_cache_manager is not None:
        kv_cache_dtype = kv_cache_manager.get_buffers(0, kv_layout=DEFAULT_KV_LAYOUT).dtype

    return FlashInferTrtllmGenAttention(
        kv_cache_manager=kv_cache_manager, quant_config=quant_config
    ).is_supported(
        phase=phase,
        q_dtype=q.dtype,
        kv_cache_dtype=kv_cache_dtype,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        out_dtype=out_dtype,
        mask_type=mask_type if mask_type is not None else 1,
        beam_width=beam_width,
        sink_token_length=sink_token_length,
        tokens_per_block=tokens_per_block,
        use_paged_kv_cache=use_paged_kv_cache,
        is_mla_enable=is_mla_enable,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=update_kv_cache,
        cross_attention=cross_attention or has_cross_kv,
        is_spec_decoding=is_spec_decoding,
        has_alibi=has_alibi,
        is_padded=is_padded,
        position_shift_enabled=position_shift_enabled,
        quant_config=quant_config,
    )


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
    quant_config: Optional[QuantConfig],
    kv_cache_manager: Optional[KVCacheManager],
    request_ids: Optional[List[int]] = None,
) -> None:
    """
    TrtLLM-Gen attention using flashinfer backend.

    This function is a drop-in replacement for thop.attention() but only
    supports trtllm-gen kernel (Blackwell architecture).

    It uses flashinfer's batch attention APIs:
    - flashinfer.prefill.trtllm_batch_context_with_kv_cache for context phase
    - flashinfer.decode.trtllm_batch_decode_with_kv_cache for generation phase

    Args:
        q: Query tensor [num_tokens, hidden_dim].
        k: Key tensor (None if fused QKV).
        v: Value tensor (None if fused QKV).
        output: Output tensor [num_tokens, num_heads * head_size].
        output_sf: Output scale factor for FP4 output (optional).
        workspace: Workspace tensor for attention computation.
        sequence_length: KV sequence lengths per request [batch_size].
        host_past_key_value_lengths: Past KV lengths on host [batch_size].
        host_total_kv_lens: Total KV lengths on host.
        context_lengths: Context lengths per request [batch_size].
        host_context_lengths: Context lengths on host [batch_size].
        host_request_types: Request types on host (0=context, 1=generation) [batch_size].
        kv_cache_block_offsets: Block offsets for paged KV cache [num_pools, batch, 2, max_blocks].
        host_kv_cache_pool_pointers: KV cache pool pointers on host.
        host_kv_cache_pool_mapping: KV cache pool mapping on host [num_layers, num_pools].
        kv_cache: Actual KV cache tensor from kv_cache_manager [num_blocks, 2, num_kv_heads,
                  tokens_per_block, head_size].
        cache_indirection: Cache indirection tensor for beam search.
        kv_scale_orig_quant: KV cache quantization scales (original to quantized).
        kv_scale_quant_orig: KV cache dequantization scales (quantized to original).
        out_scale: Output scaling factor for quantized output.
        rotary_inv_freq: Rotary embedding inverse frequencies.
        rotary_cos_sin: Precomputed rotary cosine/sine values.
        latent_cache: Latent cache for MLA (Multi-head Latent Attention).
        q_pe: Query positional encoding for MLA.
        block_ids_per_seq: Block IDs per sequence for sparse attention.
        attention_sinks: Attention sink tokens for StreamingLLM.
        is_fused_qkv: Whether Q, K, V are fused in the query tensor.
        update_kv_cache: Whether to update KV cache with new K, V values.
        predicted_tokens_per_seq: Number of predicted tokens per sequence (speculative decoding).
        layer_idx: Current transformer layer index.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value attention heads (for GQA/MQA).
        head_size: Size of each attention head.
        tokens_per_block: Number of tokens per KV cache block (page size).
        max_num_requests: Maximum number of requests in a batch.
        max_context_length: Maximum context/sequence length supported.
        attention_window_size: Sliding window attention size (0 for full attention).
        sink_token_length: Number of sink tokens for StreamingLLM.
        beam_width: Beam search width (1 for greedy decoding).
        mask_type: Attention mask type (0=padding, 1=causal, 2=bidirectional, 3=custom).
        quant_mode: Quantization mode flags.
        q_scaling: Query scaling factor for attention scores.
        position_embedding_type: Type of position embedding (0=learned, 1=rope, etc.).
        rotary_embedding_dim: Dimension for rotary embeddings.
        rotary_embedding_base: Base value for rotary embedding frequencies.
        rotary_embedding_scale_type: Scaling type for rotary embeddings.
        rotary_embedding_scales: Scaling factors for rotary embeddings.
        rotary_embedding_max_position_info: Maximum position info for rotary embeddings.
        use_paged_context_fmha: Whether to use paged attention for context phase.
        attention_input_type: Input type (0=context_only, 1=generation_only, 2=mixed).
        is_mla_enable: Whether Multi-head Latent Attention is enabled.
        chunked_prefill_buffer_batch_size: Batch size for chunked prefill buffer.
        q_lora_rank: LoRA rank for query projection (MLA).
        kv_lora_rank: LoRA rank for key-value projection (MLA).
        qk_nope_head_dim: Non-positional head dimension for QK (MLA).
        qk_rope_head_dim: Rotary positional head dimension for QK (MLA).
        v_head_dim: Value head dimension (MLA).
        mrope_rotary_cos_sin: Multi-dimensional rotary cosine/sine values.
        mrope_position_deltas: Position deltas for multi-dimensional rotary.
        mla_tensor_params: Additional tensor parameters for MLA.
        attention_chunk_size: Chunk size for chunked attention computation.
        softmax_stats_tensor: Tensor for storing softmax statistics.
        spec_decoding_bool_params: Boolean parameters for speculative decoding.
        spec_decoding_tensor_params: Tensor parameters for speculative decoding.
        sparse_kv_indices: Indices for sparse KV cache access.
        sparse_kv_offsets: Offsets for sparse KV cache access.
        sparse_attn_indices: Indices for sparse attention patterns.
        sparse_attn_offsets: Offsets for sparse attention patterns.
        sparse_attn_indices_block_size: Block size for sparse attention indices.
        sparse_mla_topk: Top-K value for sparse MLA attention.
        skip_softmax_threshold_scale_factor_prefill: Scale factor for skip softmax threshold (prefill).
        skip_softmax_threshold_scale_factor_decode: Scale factor for skip softmax threshold (decode).
        skip_softmax_stat: Statistics for skip softmax optimization.
        cu_q_seqlens: Cumulative query sequence lengths [batch_size + 1].
        cu_kv_seqlens: Cumulative KV sequence lengths [batch_size + 1].
        fmha_scheduler_counter: Counter for FMHA scheduler.
        mla_bmm1_scale: BMM1 scale for MLA attention.
        mla_bmm2_scale: BMM2 scale for MLA attention.
        quant_q_buffer: Buffer for quantized query tensor.
        quant_config: Quantization configuration (QuantConfig).
        kv_cache_manager: KV cache manager (KVCacheManager).
        request_ids: Request IDs for beam search.

    Returns:
        None. Results are written to the output tensor in-place.
    """
    logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

    kv_cache = (
        kv_cache_manager.get_buffers(layer_idx, kv_layout=DEFAULT_KV_LAYOUT)
        if kv_cache_manager is not None
        else None
    )

    backend = FlashInferTrtllmGenAttention(
        kv_cache_manager=kv_cache_manager, quant_config=quant_config
    )

    num_tokens = q.size(0)

    attn_input_type = AttentionInputType.mixed
    if attention_input_type is not None:
        attn_input_type = AttentionInputType(attention_input_type)

    num_contexts, num_generations = _parse_request_types(host_request_types)

    num_ctx_tokens = int(host_context_lengths[:num_contexts].sum()) if num_contexts > 0 else 0
    num_gen_tokens = num_tokens - num_ctx_tokens

    # Prepare Workspace
    required_workspace_size = WorkspaceManager.get_workspace_size(
        dtype=q.dtype,
        num_tokens=num_tokens,
        num_gen_tokens=num_gen_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        max_num_requests=max_num_requests,
        max_context_length=max_context_length,
        attention_window_size=attention_window_size,
        rotary_embedding_dim=rotary_embedding_dim,
        device=q.device,
    )

    # Check if we need to create/resize workspace
    current_workspace_size = (
        workspace.numel() * workspace.element_size() if workspace is not None else 0
    )

    if current_workspace_size < required_workspace_size:
        workspace.resize_(required_workspace_size)

    # Reshape Tensors
    # Input q shape: [num_tokens, (num_heads + 2*num_kv_heads) * head_size] for fused QKV
    # Need: [num_tokens, num_heads, head_size]
    if is_fused_qkv:
        q_tensor = q.view(num_tokens, num_heads + 2 * num_kv_heads, head_size)
        query = q_tensor[:, :num_heads, :].contiguous()
    else:
        query = q.view(num_tokens, num_heads, head_size)

    out_tensor = output.view(num_tokens, num_heads, head_size)

    has_kv_cache = (
        kv_cache_block_offsets is not None
        and host_kv_cache_pool_pointers is not None
        and host_kv_cache_pool_mapping is not None
        and kv_cache is not None
    )

    max_attn_window_size = (
        attention_window_size
        if beam_width == 1
        else (cache_indirection.size(2) if cache_indirection is not None else attention_window_size)
    )
    cyclic_attn_window_size = attention_window_size

    pool_index = int(host_kv_cache_pool_mapping[layer_idx, 0]) if has_kv_cache else 0
    max_blocks_per_sequence = int(kv_cache_block_offsets.size(-1)) if has_kv_cache else 0

    ctx_total_kv_len = int(host_total_kv_lens[0])
    gen_total_kv_len = int(host_total_kv_lens[1])

    is_gen_only = attn_input_type == AttentionInputType.generation_only

    common_params = dict(
        workspace=workspace,
        host_context_lengths=host_context_lengths,
        max_attention_window_size=max_attn_window_size,
        cyclic_attention_window_size=cyclic_attn_window_size,
        max_cyclic_attention_window_size=cyclic_attn_window_size,
        sink_token_length=sink_token_length,
        max_blocks_per_sequence=max_blocks_per_sequence,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_embedding_dim=rotary_embedding_dim,
        rotary_embedding_base=rotary_embedding_base,
        rotary_embedding_scale_type=rotary_embedding_scale_type,
        rotary_embedding_scale=rotary_embedding_scales[0] if rotary_embedding_scales else 1.0,
        rotary_embedding_max_positions=rotary_embedding_max_position_info[0]
        if rotary_embedding_max_position_info
        else 0,
        kv_cache_block_offsets=kv_cache_block_offsets,
        host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
        pool_index=pool_index,
        tokens_per_block=tokens_per_block if tokens_per_block is not None else 64,
        mask_type=mask_type,
        kv_cache_quant_mode=quant_mode,
        position_embedding_type=position_embedding_type,
        layer_idx=layer_idx,
        kv_scale_orig_quant=kv_scale_orig_quant,
        kv_scale_quant_orig=kv_scale_quant_orig,
        attention_output_orig_quant=out_scale,
        bmm1_scale=1.0 / (math.sqrt(head_size) * q_scaling),
        bmm2_scale=1.0,
    )

    # Context Phase
    if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
        seq_offset = 0
        token_offset = 0
        num_seqs = num_contexts

        max_context_q_len = int(host_context_lengths[seq_offset : seq_offset + num_seqs].max())
        max_past_kv_len = int(host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max())

        ctx_request_ids = request_ids[:num_seqs] if request_ids is not None else None
        ctx_params = EnqueueContextParams(
            **common_params,
            attention_input=query[token_offset : token_offset + num_ctx_tokens],
            qkv_input=q[token_offset : token_offset + num_ctx_tokens],
            block_ids_per_seq=block_ids_per_seq,
            context_buf=out_tensor[token_offset : token_offset + num_ctx_tokens],
            sequence_lengths=sequence_length[seq_offset:],
            context_lengths=context_lengths[seq_offset:],
            max_past_kv_length=max_past_kv_len,
            num_tokens=num_ctx_tokens,
            total_kv_len=ctx_total_kv_len,
            seq_offset=seq_offset,
            request_ids=ctx_request_ids,
            input_seq_length=max_context_q_len,
            batch_size=num_seqs,
            max_context_length=max_context_length,
        )
        backend.run_context(ctx_params)

    # Generation Phase
    if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
        seq_offset = num_contexts
        token_offset = 0 if is_gen_only else num_ctx_tokens
        num_seqs = num_generations

        max_past_kv_len = int(host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max())
        input_seq_length = num_gen_tokens // num_seqs if num_seqs > 0 else 1

        gen_request_ids = (
            request_ids[seq_offset : seq_offset + num_seqs] if request_ids is not None else None
        )
        gen_params = EnqueueGenerationParams(
            **common_params,
            attention_input=query[token_offset : token_offset + num_gen_tokens],
            qkv_input=q[token_offset : token_offset + num_gen_tokens],
            block_ids_per_seq=block_ids_per_seq,
            context_buf=out_tensor[token_offset : token_offset + num_gen_tokens],
            sequence_lengths=sequence_length[seq_offset:],
            context_lengths=context_lengths[seq_offset:],
            max_past_kv_length=max_past_kv_len,
            num_tokens=num_gen_tokens,
            total_kv_len=gen_total_kv_len,
            seq_offset=seq_offset,
            request_ids=gen_request_ids,
            input_seq_length=input_seq_length,
            beam_width=beam_width,
            num_requests=num_seqs // beam_width,
            host_past_key_value_lengths=host_past_key_value_lengths,
            cache_indir=cache_indirection,
        )
        backend.run_generation(gen_params)

    logger.debug(f"trtllm_gen_attention stops at layer {layer_idx}")
