"""
TrtLLM-Gen Attention Backend

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a drop-in replacement for thop.attention() with support for trtllm-gen
kernel only (Blackwell architecture: SM100/SM103).

Architecture Overview:
    1. AttentionConfig - Configuration dataclass for attention parameters
    2. TrtllmGenSupportChecker - Validates if configuration is supported
    4. FlashInferTrtllmGenAttention - FlashInfer implementation using trtllm-gen
    5. trtllm_gen_attention - Main entry point function
    6. is_supported - Check if configuration is supported

Usage:
    # Check if configuration is supported
    supported, reason = is_supported(num_heads=32, num_kv_heads=8, ...)
    if supported:
        trtllm_gen_attention(q, k, v, output, ...)
    else:
        Fallback to thop.attention()
"""

import math
from dataclasses import dataclass, field
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

# Alignment for workspace buffers (256 bytes)
WORKSPACE_ALIGNMENT = 256

# Default KV layout for flashinfer
# HND = [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]
DEFAULT_KV_LAYOUT = "HND"

# Default backend name for flashinfer
DEFAULT_BACKEND = "trtllm-gen"


def _get_kv_cache_dtype_from_quant_config(
    quant_config,
    input_dtype: torch.dtype,
) -> torch.dtype:
    """
    Get KV cache dtype based on quant_config.

    This is a convenience function that wraps AttentionConfig.get_kv_cache_dtype_from_quant_config().

    Args:
        quant_config: Quantization configuration object (QuantConfig).
                     Can be None for no quantization.
        input_dtype: Input data type to fallback to if no quantization.

    Returns:
        torch.dtype: The KV cache dtype.
                    - torch.uint8 if has_fp4_kv_cache()
                    - torch.float8_e4m3fn if has_fp8_kv_cache()
                    - input_dtype otherwise
    """
    # Forward to the static method in AttentionConfig
    # This is defined after AttentionConfig, so we can't use it directly here
    # Instead, we'll define it inline for now and update after the class definition
    if quant_config is None:
        return input_dtype

    if quant_config.layer_quant_mode.has_fp4_kv_cache():
        return torch.uint8
    elif quant_config.layer_quant_mode.has_fp8_kv_cache():
        return torch.float8_e4m3fn
    else:
        return input_dtype


@dataclass
class AttentionConfig:
    """
    Configuration for attention computation.

    Encapsulates all parameters needed for attention to enable
    clean parameter passing and validation.
    """

    # Basic attention parameters
    num_heads: int
    num_kv_heads: int
    head_size: int
    layer_idx: int = 0

    # KV Cache parameters
    use_paged_kv_cache: bool = True
    tokens_per_block: int = 64
    max_num_requests: int = 256
    max_context_length: int = 8192
    attention_window_size: int = -1  # -1 means unlimited

    # Data types
    dtype: torch.dtype = torch.float16
    out_dtype: Optional[torch.dtype] = None

    # Quantization config
    quant_config: Optional[QuantConfig] = None

    # RoPE parameters
    position_embedding_type: int = 0
    rotary_embedding_dim: int = 0
    rotary_embedding_base: float = 10000.0
    rotary_embedding_scale_type: int = 0
    rotary_embedding_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rotary_embedding_max_position_info: List[int] = field(default_factory=lambda: [8192, 8192])

    # Attention mask and features
    mask_type: int = 1  # CAUSAL by default
    q_scaling: float = 1.0
    beam_width: int = 1
    sink_token_length: int = 0

    # Advanced features (not supported by trtllm-gen)
    is_mla_enable: bool = False
    is_fused_qkv: bool = True
    update_kv_cache: bool = True
    cross_attention: bool = False
    is_spec_decoding: bool = False
    has_alibi: bool = False
    is_padded: bool = False
    position_shift_enabled: bool = False

    # Input tensors
    q: Optional[torch.Tensor] = None

    @property
    def kv_cache_dtype(self) -> torch.dtype:
        """
        Get KV cache dtype based on quant_config.

        Returns:
            torch.dtype: The KV cache dtype.
                        - torch.uint8 if has_fp4_kv_cache()
                        - torch.float8_e4m3fn if has_fp8_kv_cache()
                        - dtype otherwise
        """
        return _get_kv_cache_dtype_from_quant_config(self.quant_config, self.dtype)

    @property
    def has_fp4_kv_cache(self) -> bool:
        """
        Check if FP4 KV cache is enabled.

        Returns:
            bool: True if FP4 KV cache is enabled via quant_config, False otherwise.
        """
        if self.quant_config is not None:
            return self.quant_config.layer_quant_mode.has_fp4_kv_cache()
        return self.dtype == torch.uint8

    @property
    def heads_ratio(self) -> int:
        """Get ratio of query heads to KV heads (for GQA)."""
        return self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1


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

    # Unsupported head sizes for context FMHA
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80}

    # Maximum heads ratio for generation
    MAX_HEADS_RATIO_GENERATION = 16

    # Minimum tokens per block
    MIN_TOKENS_PER_BLOCK = 8

    # Supported tokens_per_block values for trtllm-gen kernels
    SUPPORTED_TOKENS_PER_BLOCK = {32}

    @classmethod
    def check_hardware(cls) -> Tuple[bool, str]:
        """Check if hardware supports trtllm-gen (Blackwell SM100/SM103)."""
        sm = get_sm_version()
        if not is_sm_100f(sm):
            return (False, f"trtllm-gen requires SM100 or SM103 (Blackwell). Current: SM{sm}.")
        return True, ""

    @classmethod
    def check_basic_features(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check basic feature requirements."""
        if config.is_mla_enable:
            return False, "MLA is not supported by trtllm-gen backend."

        if not config.is_fused_qkv:
            return False, "Only fused QKV is supported by trtllm-gen backend."

        if not config.update_kv_cache:
            return False, "KV cache update must be enabled for trtllm-gen backend."

        if config.cross_attention:
            return False, "Cross attention is not supported by trtllm-gen backend."

        if config.is_spec_decoding:
            return False, "Speculative decoding is not supported by trtllm-gen backend."

        return True, ""

    @classmethod
    def check_dtypes(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check if data types are supported."""

        if config.has_fp4_kv_cache:
            return False, "NVFP4 KV cache is not supported by flashinfer trtllm-gen kernels."

        if config.dtype not in cls.SUPPORTED_INPUT_DTYPES:
            return (
                False,
                f"Input dtype {config.dtype} not supported. Supported: FP16, BF16, FP8 (E4M3).",
            )

        if config.kv_cache_dtype is not None:
            if config.kv_cache_dtype not in cls.SUPPORTED_KV_CACHE_DTYPES:
                return (
                    False,
                    f"KV cache dtype {config.kv_cache_dtype} not supported. "
                    f"Supported: FP16, BF16, FP8.",
                )

        if config.out_dtype is not None:
            if config.out_dtype not in cls.SUPPORTED_OUT_DTYPES:
                return (
                    False,
                    f"Output dtype {config.out_dtype} not supported. Supported: FP16, BF16, FP8.",
                )

        return True, ""

    @classmethod
    def check_head_config(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check head configuration validity."""
        assert config.num_heads > 0, "num_heads must be positive."
        assert config.num_kv_heads > 0, "num_kv_heads must be positive."

        if config.num_heads % config.num_kv_heads != 0:
            return (
                False,
                f"num_heads ({config.num_heads}) must be divisible by "
                f"num_kv_heads ({config.num_kv_heads}).",
            )

        return True, ""

    @classmethod
    def check_context_phase(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check context (prefill) phase specific requirements."""
        if config.head_size in cls.UNSUPPORTED_HEAD_SIZES_CONTEXT:
            return (False, f"[Context] Head size {config.head_size} is not supported.")

        try:
            mask_type_enum = AttentionMaskType(config.mask_type)
            if mask_type_enum == AttentionMaskType.custom_mask:
                return False, "[Context] Custom mask is not supported."
        except ValueError:
            return False, f"[Context] Invalid mask_type: {config.mask_type}."

        if config.has_alibi:
            return False, "[Context] ALiBi is not supported."

        if config.is_padded:
            return False, "[Context] Padded input is not supported."

        return True, ""

    @classmethod
    def check_generation_phase(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check generation (decode) phase specific requirements."""
        if config.beam_width != 1:
            return (
                False,
                f"[Generation] Beam search (beam_width={config.beam_width}) "
                "is not supported. Must be 1.",
            )

        if config.position_shift_enabled:
            return False, "[Generation] Position shift is not supported."

        if config.sink_token_length != 0:
            return (
                False,
                f"[Generation] StreamingLLM (sink_token_length="
                f"{config.sink_token_length}) is not supported.",
            )

        if config.tokens_per_block < cls.MIN_TOKENS_PER_BLOCK:
            return (
                False,
                f"[Generation] tokens_per_block ({config.tokens_per_block}) "
                f"must be >= {cls.MIN_TOKENS_PER_BLOCK}.",
            )

        if config.heads_ratio > cls.MAX_HEADS_RATIO_GENERATION:
            return (
                False,
                f"[Generation] num_heads/num_kv_heads ratio ({config.heads_ratio}) "
                f"must be <= {cls.MAX_HEADS_RATIO_GENERATION}.",
            )

        if config.has_alibi:
            return False, "[Generation] ALiBi is not supported."

        return True, ""

    @classmethod
    def check_paged_kv_cache(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check paged KV cache configuration."""
        if config.use_paged_kv_cache:
            if config.tokens_per_block <= 0:
                return False, "tokens_per_block must be positive."

            # Must be power of 2
            if config.tokens_per_block & (config.tokens_per_block - 1) != 0:
                return (False, f"tokens_per_block ({config.tokens_per_block}) must be power of 2.")

            # Check if tokens_per_block is supported by trtllm-gen kernels
            if config.tokens_per_block not in cls.SUPPORTED_TOKENS_PER_BLOCK:
                return (
                    False,
                    f"tokens_per_block ({config.tokens_per_block}) is not supported by "
                    f"trtllm-gen kernels. Supported values: {sorted(cls.SUPPORTED_TOKENS_PER_BLOCK)}.",
                )

        return True, ""

    @classmethod
    def is_supported(cls, config: AttentionConfig, phase: str = "both") -> Tuple[bool, str]:
        """
        Comprehensive check if configuration is supported.

        Args:
            config: Attention configuration to validate.
            phase: Which phase to check - "context", "generation", or "both".

        Returns:
            Tuple of (is_supported, reason_if_not_supported).
        """
        # Hardware check
        ok, reason = cls.check_hardware()
        if not ok:
            return False, reason

        # Basic features check
        ok, reason = cls.check_basic_features(config)
        if not ok:
            return False, reason

        # Data type check
        ok, reason = cls.check_dtypes(config)
        if not ok:
            return False, reason

        # Head configuration check
        ok, reason = cls.check_head_config(config)
        if not ok:
            return False, reason

        # Phase-specific checks
        if phase in ("context", "both"):
            ok, reason = cls.check_context_phase(config)
            if not ok:
                return False, reason

        if phase in ("generation", "both"):
            ok, reason = cls.check_generation_phase(config)
            if not ok:
                return False, reason

        # Paged KV cache check
        ok, reason = cls.check_paged_kv_cache(config)
        if not ok:
            return False, reason

        return True, ""


class WorkspaceManager:
    """
    Manages workspace allocation for attention computation.

    Aligned with C++ AttentionOp::getWorkspaceSize*() methods.
    """

    ALIGNMENT = WORKSPACE_ALIGNMENT

    @staticmethod
    def _align_size(size: int) -> int:
        """Align size to boundary."""
        alignment = WorkspaceManager.ALIGNMENT
        return ((size + alignment - 1) // alignment) * alignment

    @classmethod
    def get_context_workspace_size(
        cls,
        dtype: torch.dtype,
        max_num_seq: int,
        max_num_tokens: int,
        num_heads: int,
        head_size: int,
        rotary_embedding_dim: int = 0,
    ) -> int:
        """Calculate workspace size for context (prefill) phase."""
        if max_num_tokens == 0:
            return 0

        # Convert torch dtype to binding dtype for get_size_in_bytes
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)
        local_hidden_units_qo = num_heads * head_size

        # Q buffer for paged context FMHA
        q_buf_size = dtype_size * max_num_tokens * local_hidden_units_qo

        # Cumulative sequence lengths
        cu_seqlens_size = 4 * (max_num_seq + 1)  # sizeof(int)

        # Rotary inv freq buffer
        rotary_inv_freq_size = (
            4 * max_num_seq * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )

        # Tokens info: (batch_idx, token_idx_in_seq) per token
        tokens_info_size = 8 * max_num_tokens  # sizeof(int2)

        # FMHA scheduler counter
        fmha_scheduler_counter = 4  # sizeof(uint32_t)

        # BMM scales for FP8
        fmha_bmm1_scale_size = 4 * 2  # sizeof(float) * 2
        fmha_bmm2_scale_size = 4  # sizeof(float)

        # Calculate total with alignment
        workspace_size = 0
        workspace_size += cls._align_size(q_buf_size)
        workspace_size += cls._align_size(cu_seqlens_size) * 3  # q, kv, mask_rows
        workspace_size += cls._align_size(rotary_inv_freq_size)
        workspace_size += cls._align_size(tokens_info_size)
        workspace_size += cls._align_size(fmha_scheduler_counter)
        workspace_size += cls._align_size(fmha_bmm1_scale_size)
        workspace_size += cls._align_size(fmha_bmm2_scale_size)

        return workspace_size

    @classmethod
    def get_generation_workspace_size(
        cls,
        dtype: torch.dtype,
        max_num_seq: int,
        max_num_tokens: int,
        num_heads: int,
        head_size: int,
        multi_processor_count: int,
        rotary_embedding_dim: int = 0,
    ) -> int:
        """Calculate workspace size for generation (decode) phase."""
        if max_num_tokens == 0:
            return 0

        # Convert torch dtype to binding dtype for get_size_in_bytes
        binding_dtype = torch_dtype_to_binding(dtype)
        dtype_size = get_size_in_bytes(dtype=binding_dtype, num_elements=1)
        batch_beam = max_num_seq

        multi_processor_count = torch.cuda.get_device_properties(
            device=torch.cuda.current_device()
        ).multi_processor_count
        # Estimate max sequence length tile
        max_seq_len_tile = max(
            1, (multi_processor_count + batch_beam * num_heads - 1) // (batch_beam * num_heads)
        )
        max_seq_len_tile = max(max_seq_len_tile, 4)

        # Partial output/sum/max buffers for multi-block attention
        partial_out_size = dtype_size * batch_beam * num_heads * head_size * max_seq_len_tile
        partial_sum_size = 4 * batch_beam * num_heads * max_seq_len_tile
        partial_max_size = 4 * batch_beam * num_heads * max_seq_len_tile

        # XQA workspace components
        cu_seqlens_size = 4 * (batch_beam + 1)
        cu_kv_seqlens_size = 4 * (batch_beam + 1)
        rotary_inv_freq_size = (
            4 * batch_beam * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )
        tokens_info_size = 8 * max_num_tokens

        # Scales for trtllm-gen kernels
        bmm1_scale_size = 4 * 2
        bmm2_scale_size = 4

        # Calculate total with alignment
        workspace_size = 0
        workspace_size += cls._align_size(partial_out_size)
        workspace_size += cls._align_size(partial_sum_size)
        workspace_size += cls._align_size(partial_max_size)
        workspace_size += cls._align_size(cu_seqlens_size)
        workspace_size += cls._align_size(cu_kv_seqlens_size)
        workspace_size += cls._align_size(rotary_inv_freq_size)
        workspace_size += cls._align_size(tokens_info_size)
        workspace_size += cls._align_size(bmm1_scale_size)
        workspace_size += cls._align_size(bmm2_scale_size)

        return workspace_size

    @classmethod
    def get_workspace_size(
        cls,
        config: AttentionConfig,
        num_tokens: int,
        num_gen_tokens: int,
    ) -> int:
        """
        Calculate total workspace size.

        Returns max(context_workspace, generation_workspace).
        """
        context_size = cls.get_context_workspace_size(
            dtype=config.dtype,
            max_num_seq=config.max_num_requests,
            max_num_tokens=num_tokens,
            num_heads=config.num_heads,
            head_size=config.head_size,
            rotary_embedding_dim=config.rotary_embedding_dim,
        )

        device = config.q.device if config.q is not None else torch.cuda.current_device()
        multi_processor_count = torch.cuda.get_device_properties(
            device=device
        ).multi_processor_count

        generation_size = cls.get_generation_workspace_size(
            dtype=config.dtype,
            max_num_seq=config.max_num_requests,
            max_num_tokens=num_gen_tokens,
            num_heads=config.num_heads,
            head_size=config.head_size,
            multi_processor_count=multi_processor_count,
            rotary_embedding_dim=config.rotary_embedding_dim,
        )

        return max(context_size, generation_size)


class FlashInferTrtllmGenAttention:
    """
    An attention backend using pure trtllm-gen kernels from flashinfer.
    """

    def __init__(self):
        self._checker = TrtllmGenSupportChecker()
        self._layout = DEFAULT_KV_LAYOUT

    @property
    def layout(self) -> str:
        """KV cache layout (HND or NHD)."""
        return self._layout

    def is_supported(self, config: AttentionConfig, phase: str = "both") -> Tuple[bool, str]:
        """Check if configuration is supported by this backend."""
        if not IS_FLASHINFER_AVAILABLE:
            return False, "flashinfer package is not installed."
        return self._checker.is_supported(config)

    def _compute_scales(
        self,
        config: AttentionConfig,
        kv_scale_quant_orig: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Compute BMM scales for attention.

        Args:
            config: Attention configuration.
            kv_scale_quant_orig: KV cache dequantization scales.

        Returns:
            Tuple of (bmm1_scale, bmm2_scale).
        """
        # Base softmax scale
        if config.q_scaling != 1.0:
            softmax_scale = config.q_scaling / math.sqrt(config.head_size)
        else:
            softmax_scale = 1.0 / math.sqrt(config.head_size)

        bmm1_scale = softmax_scale
        bmm2_scale = 1.0

        # Incorporate KV cache dequantization scales
        # flashinfer accepts torch.Tensor for bmm1_scale and bmm2_scale
        # This avoids GPU sync during CUDA graph capture
        if kv_scale_quant_orig is not None and kv_scale_quant_orig.numel() >= 2:
            # Return tensor scales - flashinfer handles tensor multiplication internally
            k_dequant_scale = kv_scale_quant_orig[0:1].to(torch.float32)
            v_dequant_scale = kv_scale_quant_orig[1:2].to(torch.float32)
            bmm1_scale = softmax_scale * k_dequant_scale
            bmm2_scale = v_dequant_scale

        return bmm1_scale, bmm2_scale

    def run_context(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        workspace: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        batch_size: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int = -1,
        out: torch.Tensor = None,
    ):
        """
        Execute context (prefill) phase using flashinfer.

        Calls flashinfer.prefill.trtllm_batch_context_with_kv_cache.
        """
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=batch_size,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            window_left=window_left,
            out=out,
            kv_layout=self._layout,
        )

    def run_generation(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace: torch.Tensor,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int = -1,
        out: torch.Tensor = None,
    ):
        """
        Execute generation (decode) phase using flashinfer.

        Calls flashinfer.decode.trtllm_batch_decode_with_kv_cache.
        """
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_kv_len,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
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


def _get_block_tables(
    kv_cache_block_offsets: torch.Tensor,
    pool_index: int,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    """
    Extract block tables for a range of sequences.

    kv_cache_block_offsets shape: (num_pools, batch_size, 2, max_blocks_per_seq)
    where the "2" dimension is [primary_pool, secondary_pool].

    flashinfer expects block_tables shape: (batch_size, max_blocks_per_seq) with dtype int32.

    Args:
        kv_cache_block_offsets: Full block offsets tensor.
        pool_index: KV cache pool index.
        start_idx: Start sequence index.
        end_idx: End sequence index.

    Returns:
        Block tables tensor for the specified range, shape (num_seqs, max_blocks_per_seq), dtype int32.
    """
    if kv_cache_block_offsets.dim() == 4:
        # Shape: (num_pools, batch_size, 2, max_blocks_per_seq)
        # Extract primary pool (index 0) block offsets
        result = kv_cache_block_offsets[pool_index, start_idx:end_idx, 0, :].contiguous()
    elif kv_cache_block_offsets.dim() == 3:
        # Shape: (batch_size, 2, max_blocks_per_seq)
        result = kv_cache_block_offsets[start_idx:end_idx, 0, :].contiguous()
    else:
        # Shape: (batch_size, max_blocks_per_seq)
        result = kv_cache_block_offsets[start_idx:end_idx].contiguous()

    # flashinfer requires int32 block_tables
    return result.to(torch.int32)


def is_supported(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
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
    phase: str = "both",
) -> Tuple[bool, str]:
    """
    Check if trtllm-gen backend supports the given configuration.

    This is the compatibility function that wraps TrtllmGenSupportChecker.

    Args:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads.
        head_size: Size of each attention head.
        dtype: Input data type.
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
        cyclic_attention_window_size: Cyclic attention window size.
        max_attention_window_size: Max attention window size.
        is_spec_decoding: Whether speculative decoding is enabled.
        is_mla_enable: Whether MLA is enabled.
        is_fused_qkv: Whether QKV is fused.
        update_kv_cache: Whether KV cache update is enabled.
        has_rotary_inv_freq: Whether rotary_inv_freq is provided.
        has_rotary_cos_sin: Whether rotary_cos_sin is provided.
        has_kv_scale: Whether KV scales are provided.
        has_cross_kv: Whether cross KV is provided.
        quant_config: Quantization configuration (QuantConfig). If provided and kv_cache_dtype
                     is None, will automatically determine kv_cache_dtype based on
                     has_fp8_kv_cache() or has_fp4_kv_cache().
        phase: Phase to check ("context", "generation", or "both").

    Returns:
        Tuple of (is_supported, reason_if_not_supported).
    """
    # Build config from parameters
    # Note: kv_cache_dtype will be auto-calculated in __post_init__ if quant_config is provided
    config = AttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        out_dtype=out_dtype,
        quant_config=quant_config,
        mask_type=mask_type if mask_type is not None else 1,
        has_alibi=has_alibi,
        is_padded=is_padded,
        use_paged_kv_cache=use_paged_kv_cache,
        tokens_per_block=tokens_per_block,
        beam_width=beam_width,
        position_shift_enabled=position_shift_enabled,
        sink_token_length=sink_token_length,
        cross_attention=cross_attention or has_cross_kv,
        is_spec_decoding=is_spec_decoding,
        is_mla_enable=is_mla_enable,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=update_kv_cache,
    )

    return FlashInferTrtllmGenAttention().is_supported(config, phase)


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
) -> None:
    """
    TrtLLM-Gen attention using flashinfer backend.

    This function is a drop-in replacement for thop.attention() but only
    supports trtllm-gen kernel (Blackwell architecture).

    It uses flashinfer's batch attention APIs:
    - flashinfer.prefill.trtllm_batch_context_with_kv_cache for context phase
    - flashinfer.decode.trtllm_batch_decode_with_kv_cache for generation phase

    IMPORTANT: Call is_supported() first to check if this backend can handle
    your configuration. If not supported, fallback to thop.attention().

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

    Returns:
        None. Results are written to the output tensor in-place.
    """
    logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

    # ========== 1. Build Configuration ==========
    page_size = tokens_per_block if tokens_per_block is not None else 64

    config = AttentionConfig(
        q=q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        layer_idx=layer_idx,
        dtype=q.dtype,
        tokens_per_block=page_size,
        max_num_requests=max_num_requests,
        max_context_length=max_context_length,
        attention_window_size=attention_window_size,
        mask_type=mask_type,
        q_scaling=q_scaling,
        beam_width=beam_width,
        sink_token_length=sink_token_length,
        position_embedding_type=position_embedding_type,
        rotary_embedding_dim=rotary_embedding_dim,
        rotary_embedding_base=rotary_embedding_base,
        rotary_embedding_scale_type=rotary_embedding_scale_type,
        rotary_embedding_scales=rotary_embedding_scales,
        rotary_embedding_max_position_info=rotary_embedding_max_position_info,
        is_mla_enable=is_mla_enable,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=update_kv_cache,
        quant_config=quant_config,
    )

    kv_cache = None
    if kv_cache_manager is not None and not config.has_fp4_kv_cache:
        kv_cache = kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")

    # ========== 2. Get Backend ==========
    backend = FlashInferTrtllmGenAttention()

    # ========== 3. Parse Request Types ==========
    num_tokens = q.size(0)

    attn_input_type = AttentionInputType.mixed
    if attention_input_type is not None:
        attn_input_type = AttentionInputType(attention_input_type)

    num_contexts, num_generations = _parse_request_types(host_request_types)

    # Calculate token counts
    host_ctx_lens = host_context_lengths
    num_ctx_tokens = int(host_ctx_lens[:num_contexts].sum()) if num_contexts > 0 else 0
    num_gen_tokens = num_tokens - num_ctx_tokens

    # ========== 4. Compute Scales ==========
    bmm1_scale, bmm2_scale = backend._compute_scales(config, kv_scale_quant_orig)

    # ========== 5. Prepare Workspace ==========
    # trtllm-gen backend needs at least 16MB for counter workspace and scratch
    min_workspace_size = 16 * 1024 * 1024  # 16 MB

    required_workspace_size = WorkspaceManager.get_workspace_size(
        config=config,
        num_tokens=num_tokens,
        num_gen_tokens=num_gen_tokens,
    )
    required_workspace_size = max(required_workspace_size, min_workspace_size)

    # Check if we need to create/resize workspace
    current_workspace_size = (
        workspace.numel() * workspace.element_size() if workspace is not None else 0
    )

    if current_workspace_size < required_workspace_size:
        workspace.resize_(required_workspace_size)

    # ========== 6. Reshape Tensors ==========
    # Input q shape: [num_tokens, (num_heads + 2*num_kv_heads) * head_size] for fused QKV
    # Need: [num_tokens, num_heads, head_size]
    if is_fused_qkv:
        q_tensor = q.view(num_tokens, num_heads + 2 * num_kv_heads, head_size)
        query = q_tensor[:, :num_heads, :].contiguous()
    else:
        query = q.view(num_tokens, num_heads, head_size)

    out_tensor = output.view(num_tokens, num_heads, head_size)

    # Determine window_left for sliding window attention
    window_left = attention_window_size if attention_window_size < max_context_length else -1

    # Check KV cache availability
    # kv_cache is the actual tensor from kv_cache_manager.get_buffers()
    has_kv_cache = (
        kv_cache_block_offsets is not None
        and host_kv_cache_pool_pointers is not None
        and host_kv_cache_pool_mapping is not None
        and kv_cache is not None
    )

    # ========== 7. Context Phase (Prefill) ==========
    if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
        logger.debug(
            f"[Layer {layer_idx}] Context phase: {num_contexts} requests, {num_ctx_tokens} tokens"
        )

        ctx_query = query[:num_ctx_tokens]
        ctx_output = out_tensor[:num_ctx_tokens]

        # Build cumulative sequence lengths
        ctx_lens = host_ctx_lens[:num_contexts].to(torch.int32)
        cum_seq_lens_q = torch.zeros(num_contexts + 1, dtype=torch.int32, device=q.device)
        torch.cumsum(ctx_lens.to(q.device), dim=0, out=cum_seq_lens_q[1:])

        # KV sequence lengths
        ctx_kv_lens = sequence_length[:num_contexts].to(torch.int32)
        cum_seq_lens_kv = torch.zeros(num_contexts + 1, dtype=torch.int32, device=q.device)
        torch.cumsum(ctx_kv_lens.to(q.device), dim=0, out=cum_seq_lens_kv[1:])

        # Use host tensors to avoid device-to-host sync during CUDA graph capture
        # ctx_lens is already on CPU (from host_ctx_lens)
        max_q_len = int(ctx_lens.max())
        # Use max_context_length as upper bound to avoid GPU sync
        max_kv_len = max_context_length

        if has_kv_cache and kv_cache is not None:
            # host_kv_cache_pool_mapping is on CPU, direct indexing is safe
            pool_index = int(host_kv_cache_pool_mapping[layer_idx, 0])
            ctx_block_tables = _get_block_tables(
                kv_cache_block_offsets, pool_index, 0, num_contexts
            )

            # Calculate number of blocks needed per sequence for context
            ctx_kv_lens_device = ctx_kv_lens.to(q.device)

            # Skip block_tables truncation during CUDA graph capture to avoid GPU-to-CPU sync.
            # The clamp operation below ensures safety anyway.
            if not torch.cuda.is_current_stream_capturing():
                num_blocks_per_seq = (ctx_kv_lens_device + page_size - 1) // page_size
                max_num_blocks = int(num_blocks_per_seq.max()) if num_contexts > 0 else 0

                # Truncate block_tables to only include valid blocks
                if max_num_blocks > 0 and max_num_blocks < ctx_block_tables.shape[1]:
                    ctx_block_tables = ctx_block_tables[:, :max_num_blocks].contiguous()

            # Clamp block indices to valid range to prevent illegal memory access
            max_pages = kv_cache.shape[0]
            ctx_block_tables = ctx_block_tables.clamp(0, max_pages - 1)

            # Run context phase
            backend.run_context(
                query=ctx_query,
                kv_cache=kv_cache,
                block_tables=ctx_block_tables,
                seq_lens=ctx_kv_lens_device,
                cum_seq_lens_q=cum_seq_lens_q,
                cum_seq_lens_kv=cum_seq_lens_kv,
                workspace=workspace,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                batch_size=num_contexts,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=window_left if window_left > 0 else -1,
                out=ctx_output,
            )

    # ========== 8. Generation Phase (Decode) ==========
    if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
        logger.debug(
            f"[Layer {layer_idx}] Generation phase: "
            f"{num_generations} requests, {num_gen_tokens} tokens"
        )

        gen_query = query[num_ctx_tokens:]
        gen_output = out_tensor[num_ctx_tokens:]

        # KV sequence lengths for generation
        gen_kv_lens = sequence_length[num_contexts : num_contexts + num_generations].to(torch.int32)
        # Use max_context_length as upper bound to avoid GPU sync during CUDA graph capture
        max_kv_len = max_context_length

        if has_kv_cache and kv_cache is not None:
            # host_kv_cache_pool_mapping is on CPU, direct indexing is safe
            pool_index = int(host_kv_cache_pool_mapping[layer_idx, 0])
            gen_block_tables = _get_block_tables(
                kv_cache_block_offsets,
                pool_index,
                num_contexts,
                num_contexts + num_generations,
            )

            # Calculate number of blocks needed per sequence for generation
            gen_kv_lens_device = gen_kv_lens.to(q.device)

            # Skip block_tables truncation during CUDA graph capture to avoid GPU-to-CPU sync.
            # The clamp operation below ensures safety anyway.
            if not torch.cuda.is_current_stream_capturing():
                num_blocks_per_seq = (gen_kv_lens_device + page_size - 1) // page_size
                max_num_blocks = int(num_blocks_per_seq.max()) if num_generations > 0 else 0

                # Truncate block_tables to only include valid blocks
                if max_num_blocks > 0 and max_num_blocks < gen_block_tables.shape[1]:
                    gen_block_tables = gen_block_tables[:, :max_num_blocks].contiguous()

            # Clamp block indices to valid range to prevent illegal memory access
            max_pages = kv_cache.shape[0]
            gen_block_tables = gen_block_tables.clamp(0, max_pages - 1)

            # Run generation phase
            backend.run_generation(
                query=gen_query,
                kv_cache=kv_cache,
                block_tables=gen_block_tables,
                seq_lens=gen_kv_lens_device,
                workspace=workspace,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=window_left if window_left > 0 else -1,
                out=gen_output,
            )

    logger.debug(f"trtllm_gen_attention stops at layer {layer_idx}")
