"""
TrtLLM-Gen Attention Backend

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a drop-in replacement for thop.attention() with support for trtllm-gen
kernel only (Blackwell architecture: SM100/SM103). Enabled via TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1.

Architecture:
    - QKV preprocessing & RoPE: C++ kernels via tensorrt_llm.bindings.internal.thop,
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
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantMode

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
    SUPPORTED_KV_CACHE_DTYPES = {DataType.HALF, DataType.BF16, DataType.FP8}
    SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    # Supported Q:KV:O dtype combinations for trtllm-gen kernels
    # Format: (q_dtype: torch.dtype, kv_dtype: DataType, o_dtype: torch.dtype)
    # Context phase supported combinations
    SUPPORTED_DTYPE_COMBOS_CONTEXT = {
        # e4m3:e4m3:e4m3
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        # fp16:fp16:fp16
        (torch.float16, DataType.HALF, torch.float16),
        # bf16:bf16:bf16
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        # e4m3:e4m3:fp16
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        # e4m3:e4m3:bf16
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
    }

    # Generation phase supported combinations (includes context + additional)
    SUPPORTED_DTYPE_COMBOS_GENERATION = {
        # All context combinations
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        # Additional generation-only combinations
        # bf16:e4m3:bf16
        (torch.bfloat16, DataType.FP8, torch.bfloat16),
        # fp16:e4m3:fp16
        (torch.float16, DataType.FP8, torch.float16),
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
        kv_cache_dtype: DataType,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        phase: str = "both",
        out_dtype: Optional[torch.dtype] = None,
        mask_type: int = 1,
        beam_width: int = 1,
        sink_token_length: int = 0,
        tokens_per_block: Optional[int] = 64,
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
        sparse_kv_indices: Optional[torch.Tensor] = None,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
        skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if tokens_per_block is None:
            tokens_per_block = 0

        sm = get_sm_version()
        if not is_sm_100f(sm):
            return (False, f"trtllm-gen requires SM100 or SM103 (Blackwell). Current: SM{sm}.")

        if (
            skip_softmax_threshold_scale_factor_prefill is not None
            or skip_softmax_threshold_scale_factor_decode is not None
        ):
            return (
                False,
                "Skip-softmax attention is not supported by trtllm-gen backend.",
            )

        has_sparse_kv = sparse_kv_indices is not None and sparse_kv_indices.numel() > 0
        has_sparse_attn = sparse_attn_indices is not None and sparse_attn_indices.numel() > 0
        if has_sparse_kv or has_sparse_attn:
            return False, "Sparse attention is not supported by trtllm-gen backend."
        if not update_kv_cache:
            return False, "KV cache update cannot be disabled now."
        if cross_attention:
            return False, "Cross attention is not supported by trtllm-gen backend."

        has_fp4_kv = (
            quant_config.layer_quant_mode.has_fp4_kv_cache()
            if quant_config is not None
            else kv_cache_dtype == DataType.NVFP4
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


@dataclass(slots=True)
class ContextWorkspaceBuffers:
    """
    Workspace buffers for context phase.
    """

    # Trtllm-gen workspace
    trtllm_gen_workspace: Optional[torch.Tensor] = None

    # Cumulative sequence lengths
    cu_q_seqlens: Optional[torch.Tensor] = None
    cu_kv_seqlens: Optional[torch.Tensor] = None
    cu_mask_rows: Optional[torch.Tensor] = None

    # Rotary embedding inverse frequencies
    rotary_inv_freq_buf: Optional[torch.Tensor] = None

    # Q buffer (FMHA output buffer for qkv_preprocessing)
    q_buf: Optional[torch.Tensor] = None

    # Token info: (batch_idx, token_idx_in_seq) per token
    tokens_info: Optional[torch.Tensor] = None

    # FMHA scheduler
    fmha_tile_counter: Optional[torch.Tensor] = None

    # BMM scales for FP8
    fmha_bmm1_scale: Optional[torch.Tensor] = None
    fmha_bmm2_scale: Optional[torch.Tensor] = None


@dataclass(slots=True)
class GenerationWorkspaceBuffers:
    """
    Workspace buffers for generation phase.
    """

    # Flashinfer / trtllm-gen FMHA workspace (multi-block handled internally)
    trtllm_gen_workspace: Optional[torch.Tensor] = None

    # Cumulative sequence lengths
    cu_seqlens: Optional[torch.Tensor] = None
    cu_kv_seqlens: Optional[torch.Tensor] = None

    # Rotary embedding inverse frequencies
    rotary_inv_freq: Optional[torch.Tensor] = None

    # Token info
    tokens_info: Optional[torch.Tensor] = None

    # Query buffer (output of qkv_preprocessing, input to FMHA)
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

    # C/CUDA type sizes in bytes, mirroring sizeof() semantics.
    SIZEOF_INT32 = 4
    SIZEOF_FLOAT32 = 4
    SIZEOF_INT2 = 8  # CUDA int2: two int32_t packed
    SIZEOF_FP8 = 1

    # Flashinfer recommends 128MB workspace, but 32MB is sufficient for
    # trtllm-gen FMHA kernels.
    # https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L1378
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
        separate_q_kv_input: bool = True,
        fp8_context_fmha: bool = False,
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
            separate_q_kv_input: Whether Q and KV inputs are separate (paged FMHA).
            fp8_context_fmha: Whether FP8 context FMHA is used.

        Returns:
            Workspace size in bytes.
        """
        if max_num_tokens == 0:
            return 0

        # Convert torch dtype to binding dtype for get_size_in_bytes
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)

        local_hidden_units_qo = num_heads * head_size
        batch_size = max_num_seq

        cu_seqlens_size = cls.SIZEOF_INT32 * (batch_size + 1)
        rotary_inv_freq_size = (
            cls.SIZEOF_FLOAT32 * batch_size * rotary_embedding_dim // 2
            if rotary_embedding_dim > 0
            else 0
        )

        q_buf_2_size = 0
        if separate_q_kv_input:
            q_buf_2_size = (
                (cls.SIZEOF_FP8 if fp8_context_fmha else dtype_size)
                * max_num_tokens
                * local_hidden_units_qo
            )

        tokens_info_size = cls.SIZEOF_INT2 * max_num_tokens
        fmha_scheduler_counter = cls.SIZEOF_INT32
        fmha_bmm1_scale_size = cls.SIZEOF_FLOAT32 * 2 if fp8_context_fmha else 0
        fmha_bmm2_scale_size = cls.SIZEOF_FLOAT32 if fp8_context_fmha else 0

        # Build workspace array
        workspaces = [
            cls.TRTLLM_GEN_WORKSPACE_SIZE,
            cu_seqlens_size,  # cu_seqlen_q
            cu_seqlens_size,  # cu_seqlen_kv
            cu_seqlens_size,  # cu_mask_rows
            rotary_inv_freq_size,
            q_buf_2_size,
            tokens_info_size,
            fmha_scheduler_counter,
            fmha_bmm1_scale_size,
            fmha_bmm2_scale_size,
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
        num_kv_heads: int = None,
        rotary_embedding_dim: int = 0,
    ) -> int:
        """
        Calculate workspace size for generation (decode) phase.

        Args:
            dtype: Data type for attention computation.
            max_num_seq: Maximum number of sequences (batch_beam).
            max_num_tokens: Maximum number of tokens.
            num_heads: Number of query attention heads.
            head_size: Size of each attention head.
            num_kv_heads: Number of KV heads (defaults to num_heads).
            rotary_embedding_dim: Rotary embedding dimension.

        Returns:
            Workspace size in bytes.
        """
        if max_num_tokens == 0:
            return 0

        if num_kv_heads is None:
            num_kv_heads = num_heads

        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)
        batch_beam = max_num_seq

        cu_seqlens_size = cls.SIZEOF_INT32 * (batch_beam + 1)
        cu_kv_seqlens_size = cls.SIZEOF_INT32 * (batch_beam + 1)
        rotary_inv_freq_size = (
            cls.SIZEOF_FLOAT32 * batch_beam * rotary_embedding_dim // 2
            if rotary_embedding_dim > 0
            else 0
        )
        tokens_info_size = cls.SIZEOF_INT2 * max_num_tokens
        q_buf_size = dtype_size * max_num_tokens * num_heads * head_size
        bmm1_scale_size = cls.SIZEOF_FLOAT32 * 2
        bmm2_scale_size = cls.SIZEOF_FLOAT32

        generation_workspaces = [
            cls.TRTLLM_GEN_WORKSPACE_SIZE,
            cu_seqlens_size,
            cu_kv_seqlens_size,
            rotary_inv_freq_size,
            tokens_info_size,
            q_buf_size,
            bmm1_scale_size,
            bmm2_scale_size,
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
        rotary_embedding_dim: int = 0,
    ) -> int:
        """
        Calculate total workspace size.

        Returns max(context_workspace, generation_workspace).
        """
        if num_kv_heads <= 0:
            num_kv_heads = num_heads

        context_size = cls.get_context_workspace_size(
            dtype=dtype,
            max_num_seq=max_num_requests,
            max_num_tokens=num_tokens,
            num_heads=num_heads,
            head_size=head_size,
            rotary_embedding_dim=rotary_embedding_dim,
            separate_q_kv_input=True,
        )

        generation_size = cls.get_generation_workspace_size(
            dtype=dtype,
            max_num_seq=max_num_requests,
            max_num_tokens=num_gen_tokens,
            num_heads=num_heads,
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            rotary_embedding_dim=rotary_embedding_dim,
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
            separate_q_kv_input: Whether Q and KV inputs are separate.
            fp8_context_fmha: Whether FP8 context FMHA is used.
            fp8_context_mla: Whether FP8 context MLA is used.

        Returns:
            Dictionary containing sub-buffer views and metadata:
            {
                'cu_q_seqlens': Tensor,
                'cu_kv_seqlens': Tensor,
                'cu_mask_rows': Tensor,
                'rotary_inv_freq_buf': Tensor or None,
                'q_buf': Tensor or None,
                'tokens_info': Tensor,
                'fmha_tile_counter': Tensor or None,
                'fmha_bmm1_scale': Tensor or None,
                'fmha_bmm2_scale': Tensor or None,
            }
        """
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)

        local_hidden_units_qo = num_heads * head_size
        cu_seqlens_size = cls.SIZEOF_INT32 * (batch_size + 1)
        rotary_inv_freq_size = (
            cls.SIZEOF_FLOAT32 * batch_size * rotary_embedding_dim // 2
            if rotary_embedding_dim > 0
            else 0
        )

        # Q buffer size (paged context FMHA)
        q_buf_2_size = 0
        if separate_q_kv_input:
            q_buf_2_size = (
                (cls.SIZEOF_FP8 if fp8_context_fmha else dtype_size)
                * num_tokens
                * local_hidden_units_qo
            )

        tokens_info_size = cls.SIZEOF_INT2 * num_tokens
        fmha_scheduler_counter = cls.SIZEOF_INT32
        fmha_bmm1_scale_size = (
            cls.SIZEOF_FLOAT32 * 2 if (fp8_context_fmha or fp8_context_mla) else 0
        )
        fmha_bmm2_scale_size = cls.SIZEOF_FLOAT32 if (fp8_context_fmha or fp8_context_mla) else 0

        offset = 0
        trtllm_gen_workspace, offset = cls._next_workspace_ptr(
            offset, cls.TRTLLM_GEN_WORKSPACE_SIZE
        )
        cu_q_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        cu_kv_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        cu_mask_rows_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        rotary_inv_freq_offset, offset = cls._next_workspace_ptr(offset, rotary_inv_freq_size)
        q_buf_offset, offset = cls._next_workspace_ptr(offset, q_buf_2_size)
        tokens_info_offset, offset = cls._next_workspace_ptr(offset, tokens_info_size)
        fmha_tile_counter_offset, offset = cls._next_workspace_ptr(offset, fmha_scheduler_counter)
        fmha_bmm1_scale_offset, offset = cls._next_workspace_ptr(offset, fmha_bmm1_scale_size)
        fmha_bmm2_scale_offset, offset = cls._next_workspace_ptr(offset, fmha_bmm2_scale_size)

        q_buf_dtype = torch.uint8 if fp8_context_fmha else dtype
        return ContextWorkspaceBuffers(
            trtllm_gen_workspace=cls._get_view(
                workspace, trtllm_gen_workspace, cls.TRTLLM_GEN_WORKSPACE_SIZE, torch.uint8
            ),
            cu_q_seqlens=cls._get_view(
                workspace, cu_q_seqlens_offset, cu_seqlens_size, torch.int32
            ),
            cu_kv_seqlens=cls._get_view(
                workspace, cu_kv_seqlens_offset, cu_seqlens_size, torch.int32
            ),
            cu_mask_rows=cls._get_view(
                workspace, cu_mask_rows_offset, cu_seqlens_size, torch.int32
            ),
            rotary_inv_freq_buf=cls._get_view(
                workspace, rotary_inv_freq_offset, rotary_inv_freq_size, torch.float32
            ),
            q_buf=cls._get_view(workspace, q_buf_offset, q_buf_2_size, q_buf_dtype),
            tokens_info=cls._get_view(workspace, tokens_info_offset, tokens_info_size, torch.int32),
            fmha_tile_counter=cls._get_view(
                workspace, fmha_tile_counter_offset, fmha_scheduler_counter, torch.uint32
            ),
            fmha_bmm1_scale=cls._get_view(
                workspace, fmha_bmm1_scale_offset, fmha_bmm1_scale_size, torch.float32
            ),
            fmha_bmm2_scale=cls._get_view(
                workspace, fmha_bmm2_scale_offset, fmha_bmm2_scale_size, torch.float32
            ),
        )

    @classmethod
    def split_generation_workspace(
        cls,
        workspace: torch.Tensor,
        dtype: torch.dtype,
        batch_beam: int,
        num_tokens: int,
        num_heads: int,
        head_size: int,
        rotary_embedding_dim: int = 0,
        num_kv_heads: int = None,
        max_blocks_per_sequence: int = 0,
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
            rotary_embedding_dim: Rotary embedding dimension.
            num_kv_heads: Number of KV heads.
            max_blocks_per_sequence: Maximum blocks per sequence.
            use_sparse_attention: Whether sparse attention is used.

        Returns:
            GenerationWorkspaceBuffers containing sub-buffer views.
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)

        cu_seqlens_size = cls.SIZEOF_INT32 * (batch_beam + 1)
        cu_kv_seqlens_size = cls.SIZEOF_INT32 * (batch_beam + 1)
        rotary_inv_freq_size = (
            cls.SIZEOF_FLOAT32 * batch_beam * rotary_embedding_dim // 2
            if rotary_embedding_dim > 0
            else 0
        )
        tokens_info_size = cls.SIZEOF_INT2 * num_tokens
        q_buf_size = dtype_size * num_tokens * num_heads * head_size
        bmm1_scale_size = cls.SIZEOF_FLOAT32 * 2
        bmm2_scale_size = cls.SIZEOF_FLOAT32
        sparse_attn_cache_size = (
            4 * (batch_beam + batch_beam * 2 * max_blocks_per_sequence) * num_kv_heads
            if use_sparse_attention
            else 0
        )

        offset = 0
        trtllm_gen_workspace, offset = cls._next_workspace_ptr(
            offset, cls.TRTLLM_GEN_WORKSPACE_SIZE
        )
        cu_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_seqlens_size)
        cu_kv_seqlens_offset, offset = cls._next_workspace_ptr(offset, cu_kv_seqlens_size)
        rotary_inv_freq_offset, offset = cls._next_workspace_ptr(offset, rotary_inv_freq_size)
        tokens_info_offset, offset = cls._next_workspace_ptr(offset, tokens_info_size)
        q_buf_offset, offset = cls._next_workspace_ptr(offset, q_buf_size)
        bmm1_scale_offset, offset = cls._next_workspace_ptr(offset, bmm1_scale_size)
        bmm2_scale_offset, offset = cls._next_workspace_ptr(offset, bmm2_scale_size)
        sparse_attn_cache_offset, offset = cls._next_workspace_ptr(offset, sparse_attn_cache_size)

        return GenerationWorkspaceBuffers(
            trtllm_gen_workspace=cls._get_view(
                workspace, trtllm_gen_workspace, cls.TRTLLM_GEN_WORKSPACE_SIZE, torch.uint8
            ),
            cu_seqlens=cls._get_view(workspace, cu_seqlens_offset, cu_seqlens_size, torch.int32),
            cu_kv_seqlens=cls._get_view(
                workspace, cu_kv_seqlens_offset, cu_kv_seqlens_size, torch.int32
            ),
            rotary_inv_freq=cls._get_view(
                workspace, rotary_inv_freq_offset, rotary_inv_freq_size, torch.float32
            ),
            tokens_info=cls._get_view(workspace, tokens_info_offset, tokens_info_size, torch.int32),
            q_buf=cls._get_view(workspace, q_buf_offset, q_buf_size, dtype),
            bmm1_scale=cls._get_view(workspace, bmm1_scale_offset, bmm1_scale_size, torch.float32),
            bmm2_scale=cls._get_view(workspace, bmm2_scale_offset, bmm2_scale_size, torch.float32),
            sparse_attn_cache=cls._get_view(
                workspace, sparse_attn_cache_offset, sparse_attn_cache_size, torch.int32
            ),
        )


@dataclass
class EnqueueParams:
    attention_input: Optional[torch.Tensor] = None
    qkv_input: Optional[torch.Tensor] = None
    context_buf: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0
    sink_token_length: int = 0
    num_tokens: int = 0
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
    seq_offset: int = 0
    tokens_per_block: int = 64
    mask_type: int = 1
    kv_cache_quant_mode: int = 0
    position_embedding_type: int = 0
    layer_idx: int = 0
    global_layer_idx: int = 0
    kv_scale_orig_quant: Optional[torch.Tensor] = None
    kv_scale_quant_orig: Optional[torch.Tensor] = None
    attention_output_orig_quant: Optional[torch.Tensor] = None
    bmm1_scale: float = 1.0
    bmm2_scale: float = 1.0
    rotary_inv_freq: Optional[torch.Tensor] = None
    rotary_cos_sin: Optional[torch.Tensor] = None
    attention_chunk_size: int = 0
    fp8_context_fmha: bool = False
    remove_padding: bool = True
    cross_attention: bool = False
    position_shift_enabled: bool = False
    paged_context_fmha: bool = False
    attention_sinks: Optional[torch.Tensor] = None
    is_mla_enable: bool = False
    # MLA parameters
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    q_scaling: float = 1.0
    latent_cache: Optional[torch.Tensor] = None
    num_layers: int = 0


@dataclass
class EnqueueContextParams(EnqueueParams):
    batch_size: int = 0
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None
    # MLA context: separate K, V inputs (non-absorption mode)
    k_input: Optional[torch.Tensor] = None
    v_input: Optional[torch.Tensor] = None
    absorption_mode: bool = False
    # Chunked prefill: softmax stats for incremental merge
    softmax_stats_tensor: Optional[torch.Tensor] = None


@dataclass
class EnqueueGenerationParams(EnqueueParams):
    beam_width: int = 1
    num_requests: int = 0
    predicted_tokens_per_seq: int = 1
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    spec_decoding_packed_mask: Optional[torch.Tensor] = None


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

    @staticmethod
    def _compute_window_left(
        cyclic_attention_window_size: int,
        max_kv_length: int,
        attention_chunk_size: int,
    ) -> int:
        """Compute flashinfer window_left with TRTLLM semantics alignment.

        TRTLLM's attention_window_size is exclusive, while flashinfer's
        window_left is inclusive. Keep the same user-visible behavior by
        subtracting 1 when sliding-window attention is enabled.
        """
        if attention_chunk_size != 0 and cyclic_attention_window_size < max_kv_length:
            raise ValueError(
                "Chunked-attention and sliding-window-attention should not be enabled at the same time."
            )
        if 0 < cyclic_attention_window_size < max_kv_length:
            return cyclic_attention_window_size - 1
        return -1

    def _build_block_tables(
        self,
        kv_cache_block_offsets,
        host_kv_cache_pool_mapping,
        layer_idx: int,
        global_layer_idx: int,
        batch_start: int,
        batch_size: int,
    ):
        """Convert C++ KVBlockArray block offsets to FlashInfer page indices.

        The C++ qkv_preprocessing kernel writes K/V via KVBlockArray, which
        addresses the pool as: pool_base + block_offset * bytes_per_single_kv_block.
        Each K/V block is a separate entry (K at offset N, V at N+1).

        FlashInfer indexes into the tensor returned by get_buffers(layer_idx),
        where each "page" spans one K block + one V block.  The page stride
        (get_buffers().stride(0)) varies by KV cache manager:

        - KVCacheManager (V1): strided view over multi-layer pool.
          stride(0) = num_layers * single_kv_block_elems, so
          divisor = num_layers (e.g. 72).
          k_offset = page * num_layers -> page = k_offset // 72.
        - KVCacheManagerV2: contiguous per-layer tensor.
          stride(0) = kv_factor * single_kv_block_elems, so
          divisor = kv_factor (= 2).
          k_offset = page * num_layers * kv_factor + layer * kv_factor
          -> page_in_tensor = k_offset // 2.

        The unified formula k_offsets // (stride(0) // single_kv_block_elems)
        handles both layouts without branching.

        TODO: This conversion exists because FlashInfer's trtllm-gen FMHA
        kernels use a shared paged KV cache index (one page index covers both
        K and V), while TRT-LLM's KVBlockArray uses separate K/V global
        block offsets (K at offset N, V at offset N+1). Once the trtllm-gen
        kernels natively support TRT-LLM's separate K/V index layout (i.e.,
        accepting kv_cache_block_offsets directly with independent K and V
        columns and pool-pointer-based addressing), this entire method can be
        removed and kv_cache_block_offsets can be passed through directly.
        Tracking: https://github.com/flashinfer-ai/flashinfer/issues/2694

        Args:
            layer_idx: Local layer offset used to index host_kv_cache_pool_mapping.
            global_layer_idx: Global layer index used as key in
                kv_cache_manager.get_buffers() / layer_offsets.
        """
        if kv_cache_block_offsets is None:
            return None
        pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
        k_offsets = kv_cache_block_offsets[pool_idx, batch_start : batch_start + batch_size, 0, :]
        kv_buf = self._kv_cache_manager.get_buffers(global_layer_idx, kv_layout=DEFAULT_KV_LAYOUT)
        single_kv_block_elems = kv_buf.shape[2] * kv_buf.shape[3] * kv_buf.shape[4]
        divisor = kv_buf.stride(0) // single_kv_block_elems
        return k_offsets // divisor

    def run_context(self, params: EnqueueContextParams):
        block_tables = self._build_block_tables(
            params.kv_cache_block_offsets,
            params.host_kv_cache_pool_mapping,
            params.layer_idx,
            params.global_layer_idx,
            params.seq_offset,
            params.batch_size,
        )
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            params.attention_chunk_size,
        )

        ctx_ws = WorkspaceManager.split_context_workspace(
            workspace=params.workspace,
            dtype=params.attention_input.dtype,
            batch_size=params.batch_size,
            num_tokens=params.num_tokens,
            num_heads=params.num_heads,
            head_size=params.head_size,
            rotary_embedding_dim=params.rotary_embedding_dim,
            separate_q_kv_input=True,
            fp8_context_fmha=params.fp8_context_fmha,
        )

        torch.ops.trtllm.build_decoder_info(
            seq_q_offsets=ctx_ws.cu_q_seqlens,
            seq_kv_offsets=ctx_ws.cu_kv_seqlens,
            padding_offsets=None,
            tokens_info=ctx_ws.tokens_info,
            encoder_padding_offsets=None,
            packed_mask_row_offsets=ctx_ws.cu_mask_rows,
            seq_cp_partial_offsets=None,
            attention_mask=None,
            seq_q_lengths=params.context_lengths,
            seq_kv_lengths=params.sequence_lengths,
            fmha_tile_counter=ctx_ws.fmha_tile_counter,
            dequant_scale_qkv=params.kv_scale_quant_orig,
            quant_scale_o=params.attention_output_orig_quant,
            fmha_bmm1_scale=ctx_ws.fmha_bmm1_scale,
            fmha_bmm2_scale=ctx_ws.fmha_bmm2_scale,
            rotary_embedding_inv_freq=ctx_ws.rotary_inv_freq_buf,
            rotary_embedding_inv_freq_cache=params.rotary_inv_freq,
            cp_size=1,
            separate_qkv_scales=QuantMode(params.kv_cache_quant_mode).has_fp4_kv_cache(),
            fmha_host_bmm1_scale=params.bmm1_scale,
            batch_size=params.batch_size,
            max_q_seq_length=params.input_seq_length,
            max_encoder_q_seq_length=0,
            attention_window_size=params.cyclic_attention_window_size,
            sink_token_length=params.sink_token_length,
            num_tokens=params.num_tokens,
            remove_padding=params.remove_padding,
            attention_mask_type=params.mask_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_scaling_type=params.rotary_embedding_scale_type,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
        )

        separate_q_kv_output = params.paged_context_fmha or params.cross_attention

        ctx_qkv_args = dict(
            qkv_input=params.qkv_input,
            cross_kv_input=None,
            quantized_qkv_output=None,
            q_output=ctx_ws.q_buf,
            kv_cache_block_offsets=params.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
            qkv_bias=None,
            qkv_scale_quant_orig=params.kv_scale_quant_orig,
            qkv_scale_orig_quant=params.kv_scale_orig_quant,
            o_scale_orig_quant=params.attention_output_orig_quant,
            fmha_bmm1_scale=ctx_ws.fmha_bmm1_scale,
            fmha_bmm2_scale=ctx_ws.fmha_bmm2_scale,
            fmha_tile_counter=ctx_ws.fmha_tile_counter,
            logn_scaling=None,
            tokens_info=ctx_ws.tokens_info,
            seq_lens=params.context_lengths,
            cache_seq_lens=params.sequence_lengths,
            encoder_seq_lens=None,
            cu_seq_lens=ctx_ws.cu_q_seqlens,
            cu_kv_seq_lens=ctx_ws.cu_kv_seqlens,
            sparse_kv_offsets=None,
            sparse_kv_indices=None,
            rotary_embedding_inv_freq=ctx_ws.rotary_inv_freq_buf,
            rotary_coef_cache_buffer=params.rotary_cos_sin,
            spec_decoding_position_offsets=None,
            mrope_rotary_cos_sin=params.mrope_rotary_cos_sin,
            mrope_position_deltas=None,
            batch_size=params.batch_size,
            max_input_seq_len=params.input_seq_length,
            max_kv_seq_len=params.max_past_kv_length,
            cyclic_kv_cache_len=params.cyclic_attention_window_size,
            sink_token_len=params.sink_token_length,
            token_num=params.num_tokens,
            remove_padding=params.remove_padding,
            is_last_chunk=(params.attention_chunk_size == 0)
            or (params.input_seq_length == params.max_past_kv_length),
            cross_attention=params.cross_attention,
            head_num=params.num_heads,
            kv_head_num=params.num_kv_heads,
            qheads_per_kv_head=params.num_heads // params.num_kv_heads,
            size_per_head=params.head_size,
            fmha_host_bmm1_scale=params.bmm1_scale,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_scaling_type=params.rotary_embedding_scale_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
            position_embedding_type=params.position_embedding_type,
            position_shift_enabled=params.position_shift_enabled,
            separate_q_kv_output=separate_q_kv_output,
            quantized_fp8_output=params.fp8_context_fmha,
            generation_phase=False,
            rotary_vision_start=0,
            rotary_vision_length=0,
            layer_idx=params.layer_idx,
            tokens_per_block=params.tokens_per_block,
            max_attention_window_size=params.max_attention_window_size,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
            cyclic_attention_window_size=params.cyclic_attention_window_size,
            beam_width=0,
            sink_token_length=params.sink_token_length,
            is_mla_enable=params.is_mla_enable,
        )
        torch.ops.trtllm.qkv_preprocessing(**ctx_qkv_args)

        # Extract Q for FlashInfer prefill. Two cases depending on whether
        # qkv_preprocessing wrote Q to a separate buffer:
        #
        # separate_q_kv_output=True (FP8/FP4 quantized, or cross-attention):
        #   qkv_preprocessing wrote RoPE'd Q into ctx_ws.q_buf. Read from it.
        #
        # separate_q_kv_output=False (BF16/FP16 non-quantized):
        #   Q stays in the packed QKV buffer after RoPE. Extract Q as a
        #   zero-copy strided view: qkv_input[:, :q_hidden_size].
        #   The view's stride(0) = full QKV hidden size, which the trtllm-gen
        #   kernel handles correctly via qStrideTokens / qStrideHeads.
        #   This matches thop's context-phase behavior where FMHA reads Q
        #   directly from the packed QKV buffer (fmhaParams.qkvPtr).
        if separate_q_kv_output:
            q_processed = ctx_ws.q_buf.view(params.num_tokens, params.num_heads, params.head_size)
        else:
            q_size = params.num_heads * params.head_size
            q_processed = params.qkv_input[:, :q_size].view(-1, params.num_heads, params.head_size)
        ctx_ws.trtllm_gen_workspace.zero_()

        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q_processed,
            kv_cache=self._kv_cache_manager.get_buffers(
                params.global_layer_idx, kv_layout=DEFAULT_KV_LAYOUT
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
            sinks=params.attention_sinks,
        )

        torch.ops.trtllm.kv_cache_postprocessing(**ctx_qkv_args)

    def run_mla_context(self, params: EnqueueContextParams):
        """
        Non-absorption MLA context attention.

          1. build_decoder_info  -- compute cu_seqlens, rotary inv_freq, etc.
          2. mla_rope_context    -- apply RoPE to Q/K, write latent KV to paged cache
          3. FMHA (SeparateQkv)  -- ragged Q/K/V attention via trtllm-gen kernel

        TODO: Support absorption MLA context attention.
        """
        head_dim_qk = params.qk_nope_head_dim + params.qk_rope_head_dim
        head_dim_v = params.qk_nope_head_dim

        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            params.attention_chunk_size,
        )

        ctx_ws = WorkspaceManager.split_context_workspace(
            workspace=params.workspace,
            dtype=params.attention_input.dtype,
            batch_size=params.batch_size,
            num_tokens=params.num_tokens,
            num_heads=params.num_heads,
            head_size=params.head_size,
            rotary_embedding_dim=params.rotary_embedding_dim,
            separate_q_kv_input=True,
            fp8_context_fmha=False,
        )

        mla_bmm1_scale = 1.0 / (params.q_scaling * math.sqrt(float(head_dim_qk)))

        torch.ops.trtllm.build_decoder_info(
            seq_q_offsets=ctx_ws.cu_q_seqlens,
            seq_kv_offsets=ctx_ws.cu_kv_seqlens,
            padding_offsets=None,
            tokens_info=ctx_ws.tokens_info,
            encoder_padding_offsets=None,
            packed_mask_row_offsets=ctx_ws.cu_mask_rows,
            seq_cp_partial_offsets=None,
            attention_mask=None,
            seq_q_lengths=params.context_lengths,
            seq_kv_lengths=params.sequence_lengths,
            fmha_tile_counter=ctx_ws.fmha_tile_counter,
            dequant_scale_qkv=params.kv_scale_quant_orig,
            quant_scale_o=params.attention_output_orig_quant,
            fmha_bmm1_scale=ctx_ws.fmha_bmm1_scale,
            fmha_bmm2_scale=ctx_ws.fmha_bmm2_scale,
            rotary_embedding_inv_freq=ctx_ws.rotary_inv_freq_buf,
            rotary_embedding_inv_freq_cache=params.rotary_inv_freq,
            cp_size=1,
            separate_qkv_scales=False,
            fmha_host_bmm1_scale=mla_bmm1_scale,
            batch_size=params.batch_size,
            max_q_seq_length=params.input_seq_length,
            max_encoder_q_seq_length=0,
            attention_window_size=params.cyclic_attention_window_size,
            sink_token_length=params.sink_token_length,
            num_tokens=params.num_tokens,
            remove_padding=params.remove_padding,
            attention_mask_type=params.mask_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_scaling_type=params.rotary_embedding_scale_type,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
        )

        # Step 2: MLA RoPE -- apply RoPE to Q rope-part and K rope-part,
        # and write latent KV (c_kv + k_pe) into paged KV cache.
        # Only invoke MLA RoPE when latent_cache is provided.
        # When latent_cache is None (e.g. forward_context_with_cached_kv path),
        # RoPE and KV cache writes were already done upstream; skip this step.
        # Aligns with C++ attentionOp.cpp guard: if (params.mla_param->latent_cache != nullptr)
        if params.latent_cache is not None:
            torch.ops.trtllm.mla_rope_context(
                latent_cache=params.latent_cache,
                q_buf=params.qkv_input,
                k_buf=params.k_input,
                v_buf=params.v_input,
                quant_q_buf=None,
                quant_k_buf=None,
                quant_v_buf=None,
                context_buf=params.context_buf,
                q_pe=None,
                cos_sin_cache=params.rotary_cos_sin,
                workspace=None,
                cache_seq_lens=params.sequence_lengths,
                seq_q_offset=None,
                fmha_tile_counter=ctx_ws.fmha_tile_counter,
                cu_q_seqlens=ctx_ws.cu_q_seqlens,
                cu_kv_seqlens=ctx_ws.cu_kv_seqlens,
                block_ids_per_seq=None,
                bmm1_scale=ctx_ws.fmha_bmm1_scale,
                bmm2_scale=ctx_ws.fmha_bmm2_scale,
                quant_scale_o=params.attention_output_orig_quant,
                quant_scale_q=params.kv_scale_orig_quant,
                quant_scale_kv=params.kv_scale_orig_quant,
                dequant_scale_q=params.kv_scale_quant_orig,
                dequant_scale_kv=params.kv_scale_quant_orig,
                quant_scale_qkv=None,
                helix_position_offsets=None,
                helix_is_inactive_rank=None,
                batch_size=params.batch_size,
                acc_q_len=params.num_tokens,
                head_num=params.num_heads,
                max_input_seq_len=params.input_seq_length,
                q_pe_ld=0,
                q_pe_stride=0,
                q_lora_rank=0,
                kv_lora_rank=params.kv_lora_rank,
                qk_nope_head_dim=params.qk_nope_head_dim,
                qk_rope_head_dim=params.qk_rope_head_dim,
                v_head_dim=params.v_head_dim,
                predicted_tokens_per_seq=1,
                num_layers=params.num_layers,
                host_bmm1_scale=mla_bmm1_scale,
                absorption_mode=False,
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                layer_idx=params.layer_idx,
                tokens_per_block=params.tokens_per_block,
                kv_head_num=1,
                size_per_head=params.kv_lora_rank + params.qk_rope_head_dim,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                cyclic_attention_window_size=params.cyclic_attention_window_size,
                max_attention_window_size=params.max_attention_window_size,
                sink_token_length=params.sink_token_length,
                beam_width=0,
                seq_offset=params.seq_offset,
            )

        out = params.context_buf
        if out.dim() == 2:
            out = out.view(params.num_tokens, params.num_heads, head_dim_v)

        ctx_ws.trtllm_gen_workspace.zero_()

        # mask_type=0 (padding/FULL) for chunked prefill intermediate chunks,
        # mask_type=1 (causal) for normal context and final chunk.
        is_causal = params.mask_type != 0
        return_lse = params.softmax_stats_tensor is not None

        import os

        use_ragged = os.environ.get("TRTLLM_MLA_CONTEXT_USE_RAGGED", "0") == "1"

        if use_ragged:
            q_3d = params.qkv_input.view(-1, params.num_heads, head_dim_qk)
            k_3d = params.k_input.view(-1, params.num_heads, head_dim_qk)
            v_3d = params.v_input.contiguous().view(-1, params.num_heads, head_dim_v)
            flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q_3d,
                key=k_3d,
                value=v_3d,
                workspace_buffer=ctx_ws.trtllm_gen_workspace,
                seq_lens=params.sequence_lengths,
                max_q_len=params.input_seq_length,
                max_kv_len=params.max_past_kv_length,
                bmm1_scale=mla_bmm1_scale,
                bmm2_scale=1.0,
                o_sf_scale=0.0,
                batch_size=params.batch_size,
                window_left=window_left,
                cum_seq_lens_q=ctx_ws.cu_q_seqlens,
                cum_seq_lens_kv=ctx_ws.cu_kv_seqlens,
                enable_pdl=None,
                is_causal=is_causal,
                return_lse=return_lse,
                attention_sinks=params.attention_sinks,
                out=out,
                lse=params.softmax_stats_tensor,
            )
        else:
            flashinfer.prefill.trtllm_batch_context_with_kv_cache_mla(
                query=params.qkv_input,
                key=params.k_input,
                value=params.v_input,
                workspace_buffer=ctx_ws.trtllm_gen_workspace,
                seq_lens=params.sequence_lengths,
                max_q_len=params.input_seq_length,
                max_kv_len=params.max_past_kv_length,
                bmm1_scale=mla_bmm1_scale,
                bmm2_scale=1.0,
                batch_size=params.batch_size,
                cum_seq_lens_q=ctx_ws.cu_q_seqlens,
                cum_seq_lens_kv=ctx_ws.cu_kv_seqlens,
                num_qo_heads=params.num_heads,
                num_kv_heads=params.num_heads,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
                head_dim_qk_nope=params.qk_nope_head_dim,
                scale_q=params.q_scaling,
                window_left=window_left,
                is_causal=is_causal,
                out=out,
                sinks=params.attention_sinks,
                return_lse=return_lse,
                lse=params.softmax_stats_tensor,
            )

    def run_generation(self, params: EnqueueGenerationParams):
        batch_beam = params.num_requests * params.beam_width
        block_tables = self._build_block_tables(
            params.kv_cache_block_offsets,
            params.host_kv_cache_pool_mapping,
            params.layer_idx,
            params.global_layer_idx,
            params.seq_offset,
            batch_beam,
        )
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            params.attention_chunk_size,
        )

        is_multi_token_gen = (
            params.spec_decoding_generation_lengths is not None
            and params.predicted_tokens_per_seq > 1
        )

        gen_ws = WorkspaceManager.split_generation_workspace(
            workspace=params.workspace,
            dtype=params.attention_input.dtype,
            batch_beam=batch_beam,
            num_tokens=params.num_tokens,
            num_heads=params.num_heads,
            head_size=params.head_size,
            rotary_embedding_dim=params.rotary_embedding_dim,
            num_kv_heads=params.num_kv_heads,
        )

        is_build_decoder_info_kernel_needed = torch.ops.trtllm.build_decoder_info(
            seq_q_offsets=gen_ws.cu_seqlens,
            seq_kv_offsets=gen_ws.cu_kv_seqlens,
            padding_offsets=None,
            tokens_info=gen_ws.tokens_info,
            encoder_padding_offsets=None,
            packed_mask_row_offsets=None,
            seq_cp_partial_offsets=None,
            attention_mask=None,
            seq_q_lengths=(params.spec_decoding_generation_lengths if is_multi_token_gen else None),
            seq_kv_lengths=params.sequence_lengths,
            fmha_tile_counter=None,
            dequant_scale_qkv=None,
            quant_scale_o=None,
            fmha_bmm1_scale=None,
            fmha_bmm2_scale=None,
            rotary_embedding_inv_freq=gen_ws.rotary_inv_freq,
            rotary_embedding_inv_freq_cache=params.rotary_inv_freq,
            cp_size=1,
            separate_qkv_scales=QuantMode(params.kv_cache_quant_mode).has_fp4_kv_cache(),
            fmha_host_bmm1_scale=params.bmm1_scale,
            batch_size=batch_beam,
            max_q_seq_length=params.input_seq_length,
            max_encoder_q_seq_length=0,
            attention_window_size=0,
            sink_token_length=0,
            num_tokens=params.num_tokens,
            remove_padding=True,
            attention_mask_type=0,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_scaling_type=params.rotary_embedding_scale_type,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
        )

        if is_build_decoder_info_kernel_needed:
            rotary_inv_freq_buf = gen_ws.rotary_inv_freq
            cu_seqlens = gen_ws.cu_seqlens
            cu_kv_seqlens = gen_ws.cu_kv_seqlens
        else:
            rotary_inv_freq_buf = params.rotary_inv_freq
            cu_seqlens = None
            cu_kv_seqlens = None

        torch.ops.trtllm.qkv_preprocessing(
            qkv_input=params.qkv_input,
            cross_kv_input=None,
            quantized_qkv_output=None,
            q_output=gen_ws.q_buf,
            kv_cache_block_offsets=params.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
            qkv_bias=None,
            qkv_scale_quant_orig=params.kv_scale_quant_orig,
            qkv_scale_orig_quant=params.kv_scale_orig_quant,
            o_scale_orig_quant=params.attention_output_orig_quant,
            fmha_bmm1_scale=gen_ws.bmm1_scale,
            fmha_bmm2_scale=gen_ws.bmm2_scale,
            fmha_tile_counter=None,
            logn_scaling=None,
            tokens_info=(gen_ws.tokens_info if is_multi_token_gen else None),
            seq_lens=(params.spec_decoding_generation_lengths if is_multi_token_gen else None),
            cache_seq_lens=params.sequence_lengths,
            encoder_seq_lens=None,
            cu_seq_lens=cu_seqlens,
            cu_kv_seq_lens=cu_kv_seqlens,
            sparse_kv_offsets=None,
            sparse_kv_indices=None,
            rotary_embedding_inv_freq=rotary_inv_freq_buf,
            rotary_coef_cache_buffer=params.rotary_cos_sin,
            spec_decoding_position_offsets=(
                params.spec_decoding_position_offsets if is_multi_token_gen else None
            ),
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            batch_size=batch_beam,
            max_input_seq_len=params.input_seq_length,
            max_kv_seq_len=params.max_past_kv_length,
            cyclic_kv_cache_len=params.cyclic_attention_window_size,
            sink_token_len=params.sink_token_length,
            token_num=params.num_tokens,
            remove_padding=params.remove_padding,
            is_last_chunk=False,
            cross_attention=params.cross_attention,
            head_num=params.num_heads,
            kv_head_num=params.num_kv_heads,
            qheads_per_kv_head=params.num_heads // params.num_kv_heads,
            size_per_head=params.head_size,
            fmha_host_bmm1_scale=params.bmm1_scale,
            rotary_embedding_dim=params.rotary_embedding_dim,
            rotary_embedding_base=params.rotary_embedding_base,
            rotary_scaling_type=params.rotary_embedding_scale_type,
            rotary_embedding_scale=params.rotary_embedding_scale,
            rotary_embedding_max_positions=params.rotary_embedding_max_positions,
            position_embedding_type=params.position_embedding_type,
            position_shift_enabled=params.position_shift_enabled,
            separate_q_kv_output=True,
            quantized_fp8_output=params.fp8_context_fmha,
            generation_phase=True,
            rotary_vision_start=0,
            rotary_vision_length=0,
            layer_idx=params.layer_idx,
            tokens_per_block=params.tokens_per_block,
            max_attention_window_size=params.max_attention_window_size,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
            cyclic_attention_window_size=params.cyclic_attention_window_size,
            beam_width=params.beam_width,
            sink_token_length=params.sink_token_length,
            seq_offset=params.seq_offset,
            is_mla_enable=params.is_mla_enable,
        )

        q_processed = gen_ws.q_buf.view(params.num_tokens, params.num_heads, params.head_size)
        gen_ws.trtllm_gen_workspace.zero_()

        # FlashInfer's trtllm-gen decode kernel needs to know the actual
        # number of query tokens per request to correctly derive batch_size
        # from the flattened query tensor:
        #
        #   query.shape = [num_tokens, num_heads, head_dim]
        #     where num_tokens = batch_size * input_seq_length
        #
        # FlashInfer computes batch_size internally as:
        #   - If q_len_per_req is set:  batch_size = num_tokens / q_len_per_req
        #   - If q_len_per_req is None: batch_size = cum_seq_lens_q.size(0) - 1
        #
        # This is critical for speculative decoding (e.g., Eagle3) where each
        # request may have multiple query tokens (input_seq_length > 1) even when
        # predicted_tokens_per_seq == 1 (as on Blackwell where
        # is_spec_decoding_enabled is forced False). Without the correct
        # q_len_per_req, FlashInfer would compute batch_size = num_tokens
        # (assuming 1 token/req), causing out-of-bounds reads on block_tables
        # which only has batch_size rows.
        #
        # Two branches handle different query-length distributions:
        #   - is_multi_token_gen (variable lengths): use cum_seq_lens_q from
        #     build_decoder_info, which encodes per-request query lengths.
        #   - else (uniform lengths): use q_len_per_req = input_seq_length.
        #     This covers both normal single-token decode (input_seq_length=1)
        #     and uniform multi-token decode (input_seq_length>1).
        if is_multi_token_gen:
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q_processed,
                kv_cache=self._kv_cache_manager.get_buffers(
                    params.global_layer_idx, kv_layout=DEFAULT_KV_LAYOUT
                ),
                workspace_buffer=gen_ws.trtllm_gen_workspace,
                block_tables=block_tables,
                seq_lens=params.sequence_lengths,
                max_seq_len=params.max_past_kv_length,
                out=params.context_buf,
                bmm1_scale=params.bmm1_scale,
                bmm2_scale=params.bmm2_scale,
                window_left=window_left,
                kv_layout=self._layout,
                sinks=params.attention_sinks,
                q_len_per_req=None,
                max_q_len=params.input_seq_length,
                cum_seq_lens_q=cu_seqlens,
            )
        else:
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q_processed,
                kv_cache=self._kv_cache_manager.get_buffers(
                    params.global_layer_idx, kv_layout=DEFAULT_KV_LAYOUT
                ),
                workspace_buffer=gen_ws.trtllm_gen_workspace,
                block_tables=block_tables,
                seq_lens=params.sequence_lengths,
                max_seq_len=params.max_past_kv_length,
                out=params.context_buf,
                bmm1_scale=params.bmm1_scale,
                bmm2_scale=params.bmm2_scale,
                window_left=window_left,
                kv_layout=self._layout,
                sinks=params.attention_sinks,
                q_len_per_req=params.input_seq_length,
            )

    def run_mla_generation(self, params: EnqueueGenerationParams):
        """MLA generation decode using flashinfer MLA kernel."""
        import math

        batch_beam = params.num_requests * params.beam_width
        block_tables = self._build_block_tables(
            params.kv_cache_block_offsets,
            params.host_kv_cache_pool_mapping,
            params.layer_idx,
            params.global_layer_idx,
            params.seq_offset,
            batch_beam,
        )

        pages_per_superblock = 128 // params.tokens_per_block
        if pages_per_superblock > 1:
            num_blocks = block_tables.size(-1)
            remainder = num_blocks % pages_per_superblock
            if remainder != 0:
                pad = pages_per_superblock - remainder
                block_tables = torch.nn.functional.pad(block_tables, (0, pad), value=0)

        mla_head_dim_qk = params.kv_lora_rank + params.qk_rope_head_dim
        q_len_per_req = params.num_tokens // batch_beam if batch_beam > 0 else 1

        query = params.qkv_input.view(batch_beam, q_len_per_req, params.num_heads, mla_head_dim_qk)

        kv_cache = self._kv_cache_manager.get_buffers(
            params.global_layer_idx, kv_layout=DEFAULT_KV_LAYOUT
        )
        if kv_cache.ndim == 5:
            kv_cache = kv_cache.squeeze(2)

        bmm1_scale = 1.0 / (
            params.q_scaling * math.sqrt(params.qk_nope_head_dim + params.qk_rope_head_dim)
        )
        bmm2_scale = 1.0

        # context_buf: [B*q_len, H, kv_lora_rank]
        # flashinfer MLA decode out check only accepts [B, H, D] (3D),
        # but kernel outputs [B, q_len, H, D] when q_len > 1.
        # For q_len=1: pass context_buf directly (in-place, zero copy).
        # For q_len>1: let flashinfer allocate, then copy back.
        out_buf = params.context_buf if q_len_per_req == 1 else None

        mla_out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=params.workspace.view(-1, 4),
            qk_nope_head_dim=params.qk_nope_head_dim,
            kv_lora_rank=params.kv_lora_rank,
            qk_rope_head_dim=params.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_seq_len=params.max_past_kv_length,
            out=out_buf,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=params.attention_sinks,
        )
        if q_len_per_req > 1:
            params.context_buf.copy_(mla_out.reshape_as(params.context_buf))


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
    tokens_per_block: Optional[int] = 64,
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
    sparse_kv_indices: Optional[torch.Tensor] = None,
    sparse_attn_indices: Optional[torch.Tensor] = None,
    skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
    skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
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
        quant_config: Quantization configuration (QuantConfig).
        kv_cache_manager: KV cache manager (its .dtype provides KV cache DataType).
        phase: Phase to check ("context", "generation", or "both").
        sparse_kv_indices: Sparse KV indices tensor for context phase.
        sparse_attn_indices: Sparse attention indices tensor for generation phase.
        skip_softmax_threshold_scale_factor_prefill: Scale factor for the skip-softmax
            threshold in prefill phase. Non-None indicates skip-softmax is enabled.
        skip_softmax_threshold_scale_factor_decode: Scale factor for the skip-softmax
            threshold in decode phase. Non-None indicates skip-softmax is enabled.

    Returns:
        Tuple of (is_supported, reason_if_not_supported).
    """
    kv_cache_dtype = torch_dtype_to_binding(q.dtype)
    if kv_cache_manager is not None:
        kv_cache_dtype = kv_cache_manager.dtype

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
        sparse_kv_indices=sparse_kv_indices,
        sparse_attn_indices=sparse_attn_indices,
        skip_softmax_threshold_scale_factor_prefill=skip_softmax_threshold_scale_factor_prefill,
        skip_softmax_threshold_scale_factor_decode=skip_softmax_threshold_scale_factor_decode,
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
    helix_tensor_params: List[Optional[torch.Tensor]],
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
    global_layer_idx: Optional[int] = None,
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
        helix_tensor_params: [helix_position_offsets, helix_is_inactive_rank]
            for Helix parallelism. Unused; accepted for interface parity.
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

    Returns:
        None. Results are written to the output tensor in-place.
    """
    logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

    backend = FlashInferTrtllmGenAttention(
        kv_cache_manager=kv_cache_manager, quant_config=quant_config
    )

    is_fp8_out = output.dtype == torch.float8_e4m3fn
    is_fp4_out = output.dtype == torch.uint8
    kv_cache_quant_mode = QuantMode(quant_mode)

    has_kv_cache_quant = kv_cache_quant_mode.has_kv_cache_quant()
    resolved_kv_scale_orig_quant = None
    resolved_kv_scale_quant_orig = None
    if has_kv_cache_quant and kv_scale_orig_quant is not None and kv_scale_quant_orig is not None:
        resolved_kv_scale_orig_quant = kv_scale_orig_quant
        resolved_kv_scale_quant_orig = kv_scale_quant_orig
        if kv_cache_quant_mode.has_fp4_kv_cache():
            assert resolved_kv_scale_orig_quant.size(0) == 3, (
                f"kv_scale_orig_quant must have size(0)==3 for FP4, got {resolved_kv_scale_orig_quant.size(0)}"
            )
            assert resolved_kv_scale_quant_orig.size(0) == 3, (
                f"kv_scale_quant_orig must have size(0)==3 for FP4, got {resolved_kv_scale_quant_orig.size(0)}"
            )

    num_tokens = q.size(0)

    attn_input_type = AttentionInputType.mixed
    if attention_input_type is not None:
        attn_input_type = AttentionInputType(attention_input_type)

    num_contexts, num_generations = _parse_request_types(host_request_types)

    is_gen_only = attn_input_type == AttentionInputType.generation_only
    is_ctx_only = attn_input_type == AttentionInputType.context_only

    if is_gen_only:
        num_ctx_tokens = 0
        num_gen_tokens = num_tokens
    elif is_ctx_only:
        num_ctx_tokens = num_tokens
        num_gen_tokens = 0
    else:
        num_ctx_tokens = int(host_context_lengths[:num_contexts].sum()) if num_contexts > 0 else 0
        num_gen_tokens = num_tokens - num_ctx_tokens

    # Prepare Workspace
    # Use upper-bound token counts for workspace sizing to avoid repeated
    # resizes during inference. The workspace contains variable-size buffers
    # (q_buf, tokens_info) whose sizes depend on actual token counts. Using
    # upper bounds allocates workspace to its maximum from the first call:
    #   - max_context_length bounds total tokens (= configured max_num_tokens)
    #   - max_num_requests bounds generation tokens (1 token per gen request)
    workspace_max_tokens = max(num_tokens, max_context_length)
    workspace_max_gen_tokens = max(num_gen_tokens, max_num_requests)
    required_workspace_size = WorkspaceManager.get_workspace_size(
        dtype=q.dtype,
        num_tokens=workspace_max_tokens,
        num_gen_tokens=workspace_max_gen_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        max_num_requests=max_num_requests,
        rotary_embedding_dim=rotary_embedding_dim,
    )

    # Check if we need to create/resize workspace
    current_workspace_size = (
        workspace.numel() * workspace.element_size() if workspace is not None else 0
    )

    if current_workspace_size < required_workspace_size:
        logger.warning(
            f"Attention workspace size is not enough, increase the size from "
            f"{current_workspace_size} bytes to {required_workspace_size} bytes"
        )
        if workspace is None:
            workspace = torch.zeros(required_workspace_size, device=q.device, dtype=torch.uint8)
        else:
            workspace.resize_(required_workspace_size)
            workspace.zero_()

    if is_mla_enable and is_gen_only and kv_lora_rank:
        out_head_size = kv_lora_rank
    elif is_mla_enable and v_head_dim:
        out_head_size = v_head_dim
    else:
        out_head_size = head_size
    out_tensor = output.view(num_tokens, num_heads, out_head_size)

    max_attn_window_size = (
        attention_window_size
        if beam_width == 1
        else (cache_indirection.size(2) if cache_indirection is not None else attention_window_size)
    )
    cyclic_attn_window_size = attention_window_size

    common_params = dict(
        workspace=workspace,
        max_attention_window_size=max_attn_window_size,
        cyclic_attention_window_size=cyclic_attn_window_size,
        sink_token_length=sink_token_length,
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
        tokens_per_block=tokens_per_block if tokens_per_block is not None else 64,
        mask_type=mask_type,
        kv_cache_quant_mode=quant_mode,
        position_embedding_type=position_embedding_type,
        layer_idx=layer_idx,
        global_layer_idx=global_layer_idx if global_layer_idx is not None else layer_idx,
        kv_scale_orig_quant=resolved_kv_scale_orig_quant,
        kv_scale_quant_orig=resolved_kv_scale_quant_orig,
        attention_output_orig_quant=out_scale,
        bmm1_scale=1.0 / (math.sqrt(head_size) * q_scaling),
        bmm2_scale=1.0,
        rotary_inv_freq=rotary_inv_freq,
        rotary_cos_sin=rotary_cos_sin,
        attention_chunk_size=attention_chunk_size if attention_chunk_size is not None else 0,
        fp8_context_fmha=is_fp8_out
        or is_fp4_out
        or (kv_cache_quant_mode.has_fp8_kv_cache() and use_paged_context_fmha),
        remove_padding=True,
        cross_attention=False,
        position_shift_enabled=False,
        paged_context_fmha=use_paged_context_fmha,
        attention_sinks=attention_sinks,
        is_mla_enable=is_mla_enable,
        kv_lora_rank=kv_lora_rank or 0,
        qk_nope_head_dim=qk_nope_head_dim or 0,
        qk_rope_head_dim=qk_rope_head_dim or 0,
        v_head_dim=v_head_dim or 0,
        q_scaling=q_scaling,
        latent_cache=latent_cache,
        num_layers=host_kv_cache_pool_mapping.size(0)
        if host_kv_cache_pool_mapping is not None
        else 0,
    )

    # Context Phase
    if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
        seq_offset = 0
        token_offset = 0
        num_seqs = num_contexts

        max_context_q_len = int(host_context_lengths[seq_offset : seq_offset + num_seqs].max())
        max_past_kv_len = int(host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max())

        ctx_params = EnqueueContextParams(
            **common_params,
            attention_input=q[token_offset : token_offset + num_ctx_tokens],
            qkv_input=q[token_offset : token_offset + num_ctx_tokens],
            context_buf=out_tensor[token_offset : token_offset + num_ctx_tokens],
            sequence_lengths=sequence_length[seq_offset:],
            context_lengths=context_lengths[seq_offset:],
            max_past_kv_length=max_past_kv_len,
            num_tokens=num_ctx_tokens,
            seq_offset=seq_offset,
            input_seq_length=max_context_q_len,
            batch_size=num_seqs,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            k_input=k if k is not None else None,
            v_input=v if v is not None else None,
            absorption_mode=False,
            softmax_stats_tensor=softmax_stats_tensor,
        )
        if is_mla_enable and not is_fused_qkv:
            backend.run_mla_context(ctx_params)
        else:
            backend.run_context(ctx_params)

    # Generation Phase
    if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
        seq_offset = num_contexts
        token_offset = 0 if is_gen_only else num_ctx_tokens
        num_seqs = num_generations

        max_past_kv_len = int(host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max())
        input_seq_length = num_gen_tokens // num_seqs if num_seqs > 0 else 1

        spec_gen_lengths = None
        spec_pos_offsets = None
        spec_packed_mask = None
        if (
            spec_decoding_bool_params
            and len(spec_decoding_bool_params) >= 2
            and spec_decoding_bool_params[0]
            and spec_decoding_bool_params[1]
            and predicted_tokens_per_seq > 1
        ):
            if spec_decoding_tensor_params and len(spec_decoding_tensor_params) >= 3:
                spec_gen_lengths = spec_decoding_tensor_params[0]
                spec_pos_offsets = spec_decoding_tensor_params[1]
                spec_packed_mask = spec_decoding_tensor_params[2]

        gen_params = EnqueueGenerationParams(
            **common_params,
            attention_input=q[token_offset : token_offset + num_gen_tokens],
            qkv_input=q[token_offset : token_offset + num_gen_tokens],
            context_buf=out_tensor[token_offset : token_offset + num_gen_tokens],
            sequence_lengths=sequence_length[seq_offset:],
            context_lengths=context_lengths[seq_offset:],
            max_past_kv_length=max_past_kv_len,
            num_tokens=num_gen_tokens,
            seq_offset=seq_offset,
            input_seq_length=input_seq_length,
            beam_width=beam_width,
            num_requests=num_seqs // beam_width,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            spec_decoding_generation_lengths=spec_gen_lengths,
            spec_decoding_position_offsets=spec_pos_offsets,
            spec_decoding_packed_mask=spec_packed_mask,
        )
        if is_mla_enable:
            backend.run_mla_generation(gen_params)
        else:
            backend.run_generation(gen_params)

    logger.debug(f"trtllm_gen_attention stops at layer {layer_idx}")
