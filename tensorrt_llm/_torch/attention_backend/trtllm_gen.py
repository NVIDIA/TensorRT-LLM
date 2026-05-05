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
    FlashInferTrtllmGenAttention.is_supported()  - Check if trtllm-gen can handle the given config.
    FlashInferTrtllmGenAttention.attention()      - Main attention method (called from TrtllmAttention.run).

Example:
    backend = FlashInferTrtllmGenAttention(kv_cache_manager=..., quant_config=...)
    supported, reason = backend.is_supported(num_heads=32, num_kv_heads=8, ...)
    if supported:
        backend.attention(q, k, v, output, ...)
    else:
        Fallback to thop.attention()
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_sm_version, is_sm_100f, nvtx_range
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantMode


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

    # Unsupported head sizes for context FMHA.
    # 96 is excluded because trtllm-gen kernel library does not ship
    # context kernels for headDim=96 (affects Phi-3 family models).
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80, 96}

    # Maximum heads ratio for generation.
    MAX_HEADS_RATIO_GENERATION = 32

    # Minimum tokens per block, tokens_per_block < 8 is not supported by TRTLLM-GEN kernels.
    MIN_TOKENS_PER_BLOCK = 8

    # Supported tokens_per_block values for trtllm-gen kernels
    SUPPORTED_TOKENS_PER_BLOCK = {16, 32, 64}

    # MLA shapes accepted by FlashInfer's trtllm-gen wrapper/launcher. The decode API
    # uses kv_lora_rank as headDimV and kv_lora_rank + qk_rope_head_dim as headDimQk.
    SUPPORTED_MLA_GENERATION_HEAD_DIMS = {
        (320, 256),
        (576, 512),
    }

    # Known FlashInfer package gap: this shape can pass the coarse checks but then fail
    # in the TRTLLM-GEN launcher with "Missing TRTLLM-GEN kernel".
    MISSING_MLA_GENERATION_KERNELS = {
        (576, 512, 32),
    }

    @classmethod
    def _check_mla_generation_support(
        cls,
        head_size: int,
        tokens_per_block: int,
        kv_lora_rank: Optional[int],
        qk_rope_head_dim: Optional[int],
    ) -> Tuple[bool, str]:
        missing_params = [
            name
            for name, value in (
                ("kv_lora_rank", kv_lora_rank),
                ("qk_rope_head_dim", qk_rope_head_dim),
            )
            if value is None or value <= 0
        ]
        if missing_params:
            return (
                False,
                f"[Generation][MLA] Missing required MLA parameter(s): {', '.join(missing_params)}.",
            )

        kv_rank = int(kv_lora_rank)
        qk_rope_dim = int(qk_rope_head_dim)
        head_dim_qk = kv_rank + qk_rope_dim
        head_dim_v = kv_rank
        if head_size != head_dim_qk:
            return (
                False,
                f"[Generation][MLA] head_size ({head_size}) must match "
                f"kv_lora_rank + qk_rope_head_dim ({head_dim_qk}).",
            )

        if (head_dim_qk, head_dim_v) not in cls.SUPPORTED_MLA_GENERATION_HEAD_DIMS:
            supported = sorted(cls.SUPPORTED_MLA_GENERATION_HEAD_DIMS)
            return (
                False,
                f"[Generation][MLA] Unsupported head dimensions: "
                f"headDimQk={head_dim_qk}, headDimV={head_dim_v}. Supported: {supported}.",
            )

        if (head_dim_qk, head_dim_v, tokens_per_block) in cls.MISSING_MLA_GENERATION_KERNELS:
            return (
                False,
                f"[Generation][MLA] Missing TRTLLM-GEN decode kernel for "
                f"headDimQk={head_dim_qk}, headDimV={head_dim_v}, "
                f"tokens_per_block={tokens_per_block}.",
            )

        return True, ""

    @classmethod
    def is_supported(
        cls,
        q_dtype: torch.dtype,
        kv_cache_dtype: DataType,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        attention_input_type: Optional[int] = None,
        out_dtype: Optional[torch.dtype] = None,
        mask_type: int = 1,
        beam_width: int = 1,
        sink_token_length: int = 0,
        tokens_per_block: Optional[int] = 64,
        use_paged_kv_cache: bool = True,
        is_mla_enable: bool = False,
        kv_lora_rank: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
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
        has_context_phase = True
        has_generation_phase = True
        if attention_input_type is not None:
            attn_input_type = AttentionInputType(attention_input_type)
            has_context_phase = attn_input_type != AttentionInputType.generation_only
            has_generation_phase = attn_input_type != AttentionInputType.context_only

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
        if is_mla_enable and has_context_phase:
            return False, (
                "MLA context and mixed phases fall back to thop.attention until "
                "FlashInfer context support is ready."
            )
        if is_mla_enable and not is_fused_qkv:
            return False, "MLA context (separate Q/K/V) falls back to thop."
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

        check_context_phase = has_context_phase and not is_mla_enable
        if check_context_phase:
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

        if has_generation_phase:
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
            heads_ratio = num_heads // num_kv_heads
            if not is_mla_enable and heads_ratio > cls.MAX_HEADS_RATIO_GENERATION:
                return (
                    False,
                    f"[Generation] heads ratio ({heads_ratio}) exceeds maximum ({cls.MAX_HEADS_RATIO_GENERATION}).",
                )
            if has_alibi:
                return False, "[Generation] ALiBi is not supported."
            if (q_dtype, kv_cache_dtype, o_dtype) not in cls.SUPPORTED_DTYPE_COMBOS_GENERATION:
                return False, (
                    f"[Generation] Unsupported dtype combination: Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )
            if is_mla_enable:
                supported, reason = cls._check_mla_generation_support(
                    head_size=head_size,
                    tokens_per_block=tokens_per_block,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                )
                if not supported:
                    return False, reason

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


@lru_cache(maxsize=128)
def _get_context_workspace_layout(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
) -> dict[str, int]:
    return thop.get_trtllm_gen_context_workspace_layout(
        dtype,
        batch_size,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        True,
        False,
    )


@lru_cache(maxsize=128)
def _get_context_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
) -> int:
    if max_num_tokens == 0:
        return 0
    layout = _get_context_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def _get_generation_workspace_layout(
    dtype: torch.dtype,
    batch_beam: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> dict[str, int]:
    return thop.get_trtllm_gen_generation_workspace_layout(
        dtype,
        batch_beam,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        num_kv_heads,
    )


@lru_cache(maxsize=128)
def _get_generation_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> int:
    if max_num_tokens == 0:
        return 0
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    layout = _get_generation_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def _get_workspace_size(
    dtype: torch.dtype,
    num_tokens: int,
    num_gen_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_num_requests: int,
    rotary_embedding_dim: int,
) -> int:
    context_size = _get_context_workspace_size(
        dtype,
        max_num_requests,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
    )
    generation_size = _get_generation_workspace_size(
        dtype,
        max_num_requests,
        num_gen_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return max(context_size, generation_size)


@dataclass(slots=True)
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
    multi_processor_count: int = 0
    batch_size: int = 0
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None
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

    # Default KV layout for flashinfer
    # HND = [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]
    DEFAULT_KV_LAYOUT = "HND"

    def __init__(
        self,
        kv_cache_manager: Optional[KVCacheManager] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        self._checker = TrtllmGenSupportChecker()
        self._layout = self.DEFAULT_KV_LAYOUT
        self._kv_cache_manager = kv_cache_manager
        self._quant_config = quant_config
        self._kv_pool_cache = {}
        self._pool_idx_cache = {}
        self._block_tables_cache = {}
        self._multi_processor_count_cache = {}
        self._enable_pdl = get_env_enable_pdl()
        missing_ops = self._missing_fused_nanobind_ops()
        if missing_ops:
            raise RuntimeError(
                f"trtllm-gen requires fused nanobind ops, missing: {', '.join(missing_ops)}."
            )

    @property
    def layout(self) -> str:
        """KV cache layout."""
        return self._layout

    def is_supported(self, **kwargs) -> Tuple[bool, str]:
        if not IS_FLASHINFER_AVAILABLE:
            return False, "flashinfer package is not installed."
        if self._kv_cache_manager is None:
            return False, "trtllm-gen requires a KVCacheManager."
        if not kwargs.get("use_paged_kv_cache", True):
            return False, "trtllm-gen requires paged KV cache."
        return self._checker.is_supported(**kwargs)

    def _get_multi_processor_count(self, device: torch.device) -> int:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("trtllm-gen requires CUDA tensors.")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        multi_processor_count = self._multi_processor_count_cache.get(device_index)
        if multi_processor_count is None:
            multi_processor_count = torch.cuda.get_device_properties(
                device_index
            ).multi_processor_count
            self._multi_processor_count_cache[device_index] = multi_processor_count
        return multi_processor_count

    def attention(
        self,
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
        num_contexts: int,
        num_ctx_tokens: int,
        global_layer_idx: Optional[int] = None,
    ) -> None:
        logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

        is_fp8_out = output.dtype == torch.float8_e4m3fn
        is_fp4_out = output.dtype == torch.uint8
        kv_cache_quant_mode = QuantMode(quant_mode)

        has_kv_cache_quant = kv_cache_quant_mode.has_kv_cache_quant()
        resolved_kv_scale_orig_quant = None
        resolved_kv_scale_quant_orig = None
        if (
            has_kv_cache_quant
            and kv_scale_orig_quant is not None
            and kv_scale_quant_orig is not None
        ):
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

        is_gen_only = attn_input_type == AttentionInputType.generation_only

        num_generations = host_request_types.size(0) - num_contexts
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid trtllm-gen attention token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}, attention_input_type={attn_input_type}."
            )

        workspace_max_tokens = max(num_tokens, max_context_length)
        workspace_max_gen_tokens = max(num_gen_tokens, max_num_requests)
        required_workspace_size = _get_workspace_size(
            q.dtype,
            workspace_max_tokens,
            workspace_max_gen_tokens,
            num_heads,
            num_kv_heads,
            head_size,
            max_num_requests,
            rotary_embedding_dim,
        )

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
            else (
                cache_indirection.size(2)
                if cache_indirection is not None
                else attention_window_size
            )
        )
        cyclic_attn_window_size = attention_window_size
        params = EnqueueParams(
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
            multi_processor_count=self._get_multi_processor_count(q.device),
        )

        if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
            seq_offset = 0
            token_offset = 0
            num_seqs = num_contexts

            max_context_q_len = int(host_context_lengths[seq_offset : seq_offset + num_seqs].max())
            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )

            params.attention_input = q[token_offset : token_offset + num_ctx_tokens]
            params.qkv_input = q[token_offset : token_offset + num_ctx_tokens]
            params.context_buf = out_tensor[token_offset : token_offset + num_ctx_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.context_lengths = context_lengths[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_ctx_tokens
            params.seq_offset = seq_offset
            params.input_seq_length = max_context_q_len
            params.batch_size = num_seqs
            params.mrope_rotary_cos_sin = mrope_rotary_cos_sin
            self.run_context(params)

        if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
            seq_offset = num_contexts
            token_offset = 0 if is_gen_only else num_ctx_tokens
            num_seqs = num_generations

            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )
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

            params.attention_input = q[token_offset : token_offset + num_gen_tokens]
            params.qkv_input = q[token_offset : token_offset + num_gen_tokens]
            params.context_buf = out_tensor[token_offset : token_offset + num_gen_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.context_lengths = context_lengths[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_gen_tokens
            params.seq_offset = seq_offset
            params.input_seq_length = input_seq_length
            params.beam_width = beam_width
            params.num_requests = num_seqs // beam_width
            params.predicted_tokens_per_seq = predicted_tokens_per_seq
            params.spec_decoding_generation_lengths = spec_gen_lengths
            params.spec_decoding_position_offsets = spec_pos_offsets
            params.spec_decoding_packed_mask = spec_packed_mask
            if is_mla_enable:
                self.run_mla_generation(params)
            else:
                self.run_generation(params)

        logger.debug(f"trtllm_gen_attention stops at layer {layer_idx}")

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

    def _get_kv_cache_and_block_tables(
        self,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        layer_idx: int,
        num_kv_heads: int,
        tokens_per_block: int,
        head_dim: int,
        kv_cache_quant_mode: int,
        batch_start: int,
        batch_size: int,
        dtype: torch.dtype,
        is_mla_enable: bool = False,
    ):
        """Get FlashInfer kv_cache and block_tables.

        The kv_cache tensor is a flat-block view of the per-layer KV cache pool.

        Shape: [total_blocks, num_kv_heads, tokens_per_block, head_dim]
        where each dim-0 element = one single K or V block.

        FlashInfer receives kv_cache as a tuple (kv_pool, kv_pool) where
        K and V share the same pool tensor. With uses_shared_paged_kv_idx=
        False (flashinfer PR #2770), the kernel reads K/V offsets separately
        from block_tables[batch, 2, max_blocks] and computes:
            K_addr = pool_ptr + K_offset * block_size
            V_addr = pool_ptr + V_offset * block_size

        Raw block offsets are used directly as indices -- no division needed.

        block_tables is a lightweight Python tensor slice (no C++ op call),
        computed fresh each forward pass since batch composition changes.
        """
        if kv_cache_block_offsets is None:
            return None, None

        kv_pool = self._get_or_build_kv_pool(
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            layer_idx=layer_idx,
            num_kv_heads=num_kv_heads,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            kv_cache_quant_mode=kv_cache_quant_mode,
            dtype=dtype,
            is_mla_enable=is_mla_enable,
        )
        block_tables = self._slice_block_tables(
            kv_cache_block_offsets=kv_cache_block_offsets,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            layer_idx=layer_idx,
            batch_start=batch_start,
            batch_size=batch_size,
        )

        return (kv_pool, kv_pool), block_tables

    def _get_or_build_kv_pool(
        self,
        *,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        layer_idx: int,
        num_kv_heads: int,
        tokens_per_block: int,
        head_dim: int,
        kv_cache_quant_mode: int,
        dtype: torch.dtype,
        is_mla_enable: bool = False,
    ) -> torch.Tensor:
        cache_scope = self._get_kv_pool_cache_scope(
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            num_kv_heads=num_kv_heads,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            kv_cache_quant_mode=kv_cache_quant_mode,
            dtype=dtype,
            is_mla_enable=is_mla_enable,
        )
        cache_key = (layer_idx, cache_scope)
        cached_kv_pool = self._kv_pool_cache.get(cache_key)
        if cached_kv_pool is not None:
            return cached_kv_pool

        kv_cache = self._kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
        block_elems = kv_cache.shape[2] * kv_cache.shape[3] * kv_cache.shape[4]
        storage_numel = kv_cache.untyped_storage().nbytes() // kv_cache.element_size()
        available_blocks = (storage_numel - kv_cache.storage_offset()) // block_elems

        if kv_cache.shape[2] != num_kv_heads or kv_cache.shape[3] != tokens_per_block:
            raise RuntimeError(
                "Invalid trtllm-gen KV cache view shape: "
                f"shape={tuple(kv_cache.shape)}, num_kv_heads={num_kv_heads}, "
                f"tokens_per_block={tokens_per_block}."
            )

        kv_pool = kv_cache.as_strided(
            (available_blocks, kv_cache.shape[2], kv_cache.shape[3], kv_cache.shape[4]),
            (block_elems, kv_cache.shape[3] * kv_cache.shape[4], kv_cache.shape[4], 1),
        )
        if kv_pool.data_ptr() != kv_cache.data_ptr():
            raise RuntimeError("Failed to create a no-copy trtllm-gen KV cache pool view.")
        self._kv_pool_cache[cache_key] = kv_pool
        return kv_pool

    def _get_kv_pool_cache_scope(
        self,
        *,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        num_kv_heads: int,
        tokens_per_block: int,
        head_dim: int,
        kv_cache_quant_mode: int,
        dtype: torch.dtype,
        is_mla_enable: bool,
    ):
        blocks_in_primary_pool = getattr(self._kv_cache_manager, "blocks_in_primary_pool", None)
        if blocks_in_primary_pool is None:
            blocks_per_window = getattr(self._kv_cache_manager, "blocks_per_window", None)
            if blocks_per_window:
                blocks_in_primary_pool = max(
                    int(primary_blocks) for primary_blocks, _ in blocks_per_window.values()
                )
        if blocks_in_primary_pool is not None:
            blocks_in_primary_pool = int(blocks_in_primary_pool)

        return (
            host_kv_cache_pool_pointers.data_ptr(),
            host_kv_cache_pool_pointers._version,
            host_kv_cache_pool_mapping.data_ptr(),
            host_kv_cache_pool_mapping._version,
            blocks_in_primary_pool,
            int(getattr(self._kv_cache_manager, "num_local_layers", 0) or 0),
            num_kv_heads,
            tokens_per_block,
            head_dim,
            kv_cache_quant_mode,
            dtype,
            is_mla_enable,
        )

    def _slice_block_tables(
        self,
        *,
        kv_cache_block_offsets,
        host_kv_cache_pool_mapping,
        layer_idx: int,
        batch_start: int,
        batch_size: int,
    ):
        cache_key = (host_kv_cache_pool_mapping.data_ptr(), layer_idx)
        pool_idx = self._pool_idx_cache.get(cache_key)
        if pool_idx is None:
            pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
            self._pool_idx_cache[cache_key] = pool_idx
        block_tables_key = (
            kv_cache_block_offsets.data_ptr(),
            tuple(kv_cache_block_offsets.shape),
            tuple(kv_cache_block_offsets.stride()),
            pool_idx,
            batch_start,
            batch_size,
        )
        block_tables = self._block_tables_cache.get(block_tables_key)
        if block_tables is None:
            block_tables = kv_cache_block_offsets[pool_idx, batch_start : batch_start + batch_size]
            self._block_tables_cache[block_tables_key] = block_tables
        return block_tables

    @staticmethod
    def _missing_fused_nanobind_ops() -> List[str]:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "get_trtllm_gen_generation_workspace_layout",
            "trtllm_gen_context_preprocess",
            "trtllm_gen_context_postprocess",
            "trtllm_gen_generation_preprocess",
        )
        return [op for op in required_ops if not hasattr(thop, op)]

    def run_context(self, params: EnqueueParams):
        kv_factor = 0
        total_num_blocks = 0
        # KV metadata is cached in Python; these satisfy the fused op signature.
        with nvtx_range("trtllm_gen.context.kv_metadata", color="blue"):
            kv_cache, block_tables = self._get_kv_cache_and_block_tables(
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                layer_idx=params.layer_idx,
                num_kv_heads=params.num_kv_heads,
                tokens_per_block=params.tokens_per_block,
                head_dim=params.head_size,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                batch_start=params.seq_offset,
                batch_size=params.batch_size,
                dtype=params.qkv_input.dtype,
                is_mla_enable=params.is_mla_enable,
            )
        with nvtx_range("trtllm_gen.context.preprocess", color="purple"):
            (
                q_processed,
                _kv_pool,
                _block_tables,
                fmha_workspace,
                cu_q_seqlens,
                cu_kv_seqlens,
                max_q_len,
                max_kv_len,
                window_left,
            ) = thop.trtllm_gen_context_preprocess(
                qkv_input=params.qkv_input,
                workspace=params.workspace,
                sequence_lengths=params.sequence_lengths,
                context_lengths=params.context_lengths,
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=params.kv_scale_orig_quant,
                kv_scale_quant_orig=params.kv_scale_quant_orig,
                attention_output_orig_quant=params.attention_output_orig_quant,
                rotary_inv_freq=params.rotary_inv_freq,
                rotary_cos_sin=params.rotary_cos_sin,
                mrope_rotary_cos_sin=params.mrope_rotary_cos_sin,
                layer_idx=params.layer_idx,
                num_heads=params.num_heads,
                num_kv_heads=params.num_kv_heads,
                head_size=params.head_size,
                tokens_per_block=params.tokens_per_block,
                mask_type=params.mask_type,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                max_attention_window_size=params.max_attention_window_size,
                cyclic_attention_window_size=params.cyclic_attention_window_size,
                sink_token_length=params.sink_token_length,
                num_tokens=params.num_tokens,
                batch_size=params.batch_size,
                input_seq_length=params.input_seq_length,
                max_past_kv_length=params.max_past_kv_length,
                rotary_embedding_dim=params.rotary_embedding_dim,
                rotary_embedding_base=params.rotary_embedding_base,
                rotary_embedding_scale_type=params.rotary_embedding_scale_type,
                rotary_embedding_scale=params.rotary_embedding_scale,
                rotary_embedding_max_positions=params.rotary_embedding_max_positions,
                position_embedding_type=params.position_embedding_type,
                bmm1_scale=params.bmm1_scale,
                bmm2_scale=params.bmm2_scale,
                attention_chunk_size=params.attention_chunk_size,
                fp8_context_fmha=params.fp8_context_fmha,
                paged_context_fmha=params.paged_context_fmha,
                is_mla_enable=params.is_mla_enable,
                multi_processor_count=params.multi_processor_count,
                total_num_blocks=total_num_blocks,
                kv_factor=kv_factor,
                need_build_kv_cache_metadata=False,
            )

        with nvtx_range("trtllm_gen.context.flashinfer", color="green"):
            flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q_processed,
                kv_cache=kv_cache,
                workspace_buffer=fmha_workspace,
                block_tables=block_tables,
                seq_lens=params.sequence_lengths,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                bmm1_scale=params.bmm1_scale,
                bmm2_scale=params.bmm2_scale,
                batch_size=params.batch_size,
                cum_seq_lens_q=cu_q_seqlens,
                cum_seq_lens_kv=cu_kv_seqlens,
                window_left=window_left,
                out=params.context_buf,
                kv_layout=self._layout,
                sinks=params.attention_sinks,
                uses_shared_paged_kv_idx=False,
                enable_pdl=self._enable_pdl,
            )

        with nvtx_range("trtllm_gen.context.postprocess", color="orange"):
            thop.trtllm_gen_context_postprocess(
                qkv_input=params.qkv_input,
                workspace=params.workspace,
                sequence_lengths=params.sequence_lengths,
                context_lengths=params.context_lengths,
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=params.kv_scale_orig_quant,
                kv_scale_quant_orig=params.kv_scale_quant_orig,
                attention_output_orig_quant=params.attention_output_orig_quant,
                rotary_cos_sin=params.rotary_cos_sin,
                mrope_rotary_cos_sin=params.mrope_rotary_cos_sin,
                layer_idx=params.layer_idx,
                num_heads=params.num_heads,
                num_kv_heads=params.num_kv_heads,
                head_size=params.head_size,
                tokens_per_block=params.tokens_per_block,
                mask_type=params.mask_type,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                max_attention_window_size=params.max_attention_window_size,
                cyclic_attention_window_size=params.cyclic_attention_window_size,
                sink_token_length=params.sink_token_length,
                num_tokens=params.num_tokens,
                batch_size=params.batch_size,
                input_seq_length=params.input_seq_length,
                max_past_kv_length=params.max_past_kv_length,
                rotary_embedding_dim=params.rotary_embedding_dim,
                rotary_embedding_base=params.rotary_embedding_base,
                rotary_embedding_scale_type=params.rotary_embedding_scale_type,
                rotary_embedding_scale=params.rotary_embedding_scale,
                rotary_embedding_max_positions=params.rotary_embedding_max_positions,
                position_embedding_type=params.position_embedding_type,
                bmm1_scale=params.bmm1_scale,
                fp8_context_fmha=params.fp8_context_fmha,
                paged_context_fmha=params.paged_context_fmha,
                is_mla_enable=params.is_mla_enable,
                attention_chunk_size=params.attention_chunk_size,
                multi_processor_count=params.multi_processor_count,
            )

    def run_generation(self, params: EnqueueParams):
        batch_beam = params.num_requests * params.beam_width
        kv_factor = 0
        total_num_blocks = 0
        # KV metadata is cached in Python; these satisfy the fused op signature.
        with nvtx_range("trtllm_gen.generation.kv_metadata", color="blue"):
            kv_cache, block_tables = self._get_kv_cache_and_block_tables(
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                layer_idx=params.layer_idx,
                num_kv_heads=params.num_kv_heads,
                tokens_per_block=params.tokens_per_block,
                head_dim=params.head_size,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                batch_start=params.seq_offset,
                batch_size=batch_beam,
                dtype=params.qkv_input.dtype,
                is_mla_enable=params.is_mla_enable,
            )
        with nvtx_range("trtllm_gen.generation.preprocess", color="purple"):
            (
                q_processed,
                _kv_pool,
                _block_tables,
                fmha_workspace,
                cu_seqlens,
                max_q_len,
                max_kv_len,
                window_left,
                is_multi_token_gen,
            ) = thop.trtllm_gen_generation_preprocess(
                qkv_input=params.qkv_input,
                workspace=params.workspace,
                sequence_lengths=params.sequence_lengths,
                spec_decoding_generation_lengths=params.spec_decoding_generation_lengths,
                spec_decoding_position_offsets=params.spec_decoding_position_offsets,
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=params.kv_scale_orig_quant,
                kv_scale_quant_orig=params.kv_scale_quant_orig,
                attention_output_orig_quant=params.attention_output_orig_quant,
                rotary_inv_freq=params.rotary_inv_freq,
                rotary_cos_sin=params.rotary_cos_sin,
                layer_idx=params.layer_idx,
                seq_offset=params.seq_offset,
                num_heads=params.num_heads,
                num_kv_heads=params.num_kv_heads,
                head_size=params.head_size,
                tokens_per_block=params.tokens_per_block,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                max_attention_window_size=params.max_attention_window_size,
                cyclic_attention_window_size=params.cyclic_attention_window_size,
                sink_token_length=params.sink_token_length,
                num_tokens=params.num_tokens,
                batch_beam=batch_beam,
                input_seq_length=params.input_seq_length,
                max_past_kv_length=params.max_past_kv_length,
                rotary_embedding_dim=params.rotary_embedding_dim,
                rotary_embedding_base=params.rotary_embedding_base,
                rotary_embedding_scale_type=params.rotary_embedding_scale_type,
                rotary_embedding_scale=params.rotary_embedding_scale,
                rotary_embedding_max_positions=params.rotary_embedding_max_positions,
                position_embedding_type=params.position_embedding_type,
                bmm1_scale=params.bmm1_scale,
                bmm2_scale=params.bmm2_scale,
                fp8_context_fmha=params.fp8_context_fmha,
                predicted_tokens_per_seq=params.predicted_tokens_per_seq,
                attention_chunk_size=params.attention_chunk_size,
                multi_processor_count=params.multi_processor_count,
                total_num_blocks=total_num_blocks,
                kv_factor=kv_factor,
                need_build_kv_cache_metadata=False,
            )

        q_len_per_req = None if is_multi_token_gen else params.input_seq_length
        decode_max_q_len = max_q_len if is_multi_token_gen else None
        decode_cu_seqlens = cu_seqlens if is_multi_token_gen else None
        with nvtx_range("trtllm_gen.generation.flashinfer", color="green"):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q_processed,
                kv_cache=kv_cache,
                workspace_buffer=fmha_workspace,
                block_tables=block_tables,
                seq_lens=params.sequence_lengths,
                max_seq_len=max_kv_len,
                out=params.context_buf,
                bmm1_scale=params.bmm1_scale,
                bmm2_scale=params.bmm2_scale,
                window_left=window_left,
                kv_layout=self._layout,
                sinks=params.attention_sinks,
                q_len_per_req=q_len_per_req,
                max_q_len=decode_max_q_len,
                cum_seq_lens_q=decode_cu_seqlens,
                uses_shared_paged_kv_idx=False,
                enable_pdl=self._enable_pdl,
                backend="trtllm-gen",
            )

    def run_mla_generation(self, params: EnqueueParams) -> None:
        """MLA generation decode using flashinfer MLA kernel."""
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            raise NotImplementedError(
                "Sliding-window attention is not supported by MLA decode path."
            )
        if params.attention_chunk_size != 0:
            raise NotImplementedError("Chunked-attention is not supported by MLA decode path.")

        batch_beam = params.num_requests * params.beam_width
        with nvtx_range("trtllm_gen.mla.kv_metadata", color="blue"):
            kv_cache_tuple, block_tables = self._get_kv_cache_and_block_tables(
                kv_cache_block_offsets=params.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
                layer_idx=params.layer_idx,
                num_kv_heads=params.num_kv_heads,
                tokens_per_block=params.tokens_per_block,
                head_dim=params.head_size,
                kv_cache_quant_mode=params.kv_cache_quant_mode,
                batch_start=params.seq_offset,
                batch_size=batch_beam,
                dtype=params.attention_input.dtype,
                is_mla_enable=params.is_mla_enable,
            )

            # MLA decode API takes a single kv tensor [num_pages, 1, page_size, head_dim],
            # not the (K_pool, V_pool) tuple.
            kv_cache = kv_cache_tuple[0]

        with nvtx_range("trtllm_gen.mla.block_table_pad", color="yellow"):
            pages_per_superblock = 128 // params.tokens_per_block
            if pages_per_superblock > 1:
                num_blocks = block_tables.size(-1)
                remainder = num_blocks % pages_per_superblock
                if remainder != 0:
                    pad = pages_per_superblock - remainder
                    block_tables = torch.nn.functional.pad(block_tables, (0, pad), value=0)

        with nvtx_range("trtllm_gen.mla.prepare", color="purple"):
            mla_head_dim_qk = params.kv_lora_rank + params.qk_rope_head_dim
            q_len_per_req = params.num_tokens // batch_beam if batch_beam > 0 else 1

            query = params.qkv_input.view(
                batch_beam, q_len_per_req, params.num_heads, mla_head_dim_qk
            )

            bmm1_scale = 1.0 / (
                params.q_scaling * math.sqrt(params.qk_nope_head_dim + params.qk_rope_head_dim)
            )
            bmm2_scale = 1.0

        with nvtx_range("trtllm_gen.mla.flashinfer", color="green"):
            flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=params.workspace.view(-1, 4),
                qk_nope_head_dim=params.qk_nope_head_dim,
                kv_lora_rank=params.kv_lora_rank,
                qk_rope_head_dim=params.qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=params.sequence_lengths,
                max_seq_len=params.max_past_kv_length,
                out=params.context_buf.view(
                    batch_beam, q_len_per_req, params.num_heads, params.kv_lora_rank
                ),
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=params.attention_sinks,
                uses_shared_paged_kv_idx=False,
                enable_pdl=self._enable_pdl,
                backend="trtllm-gen",
            )
