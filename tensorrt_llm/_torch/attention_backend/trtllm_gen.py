"""
TrtLLM-Gen Attention Backend

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a drop-in replacement for thop.attention() with support for trtllm-gen
kernel only (Blackwell architecture: SM100/SM103). Enabled via TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1.

Architecture:
    - QKV preprocessing & RoPE: C++ kernels via tensorrt_llm.bindings.internal.thop,
      same as thop.attention. Writes K/V to paged KV cache via pool pointers.
    - Attention: flashinfer trtllm-gen FMHA kernels, reading KV cache through
      the KV cache manager carried by attention metadata.

Entry points:
    FlashInferTrtllmGenAttention.is_supported()  - Check if trtllm-gen can handle the given config.
    FlashInferTrtllmGenAttention.attention()      - Main attention method (called from TrtllmAttention.run).

Example:
    backend = FlashInferTrtllmGenAttention(attention_layer=...)
    supported, reason = backend.is_supported(
        q, metadata=..., forward_args=..., ...)
    if supported:
        backend.attention(q, metadata=..., forward_args=..., ...)
    else:
        Fallback to thop.attention()
"""

import math
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantMode

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


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
        tokens_per_block: Optional[int] = 64,
        use_paged_kv_cache: bool = True,
        is_mla_enable: bool = False,
        kv_lora_rank: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        cross_attention: bool = False,
        is_spec_decoding: bool = False,
        has_alibi: bool = False,
        is_padded: bool = False,
        position_shift_enabled: bool = False,
        quant_config: Optional[QuantConfig] = None,
        has_sparse_attention: bool = False,
        has_skip_softmax_attention: bool = False,
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

        if has_skip_softmax_attention:
            return (
                False,
                "Skip-softmax attention is not supported by trtllm-gen backend.",
            )

        if has_sparse_attention:
            return False, "Sparse attention is not supported by trtllm-gen backend."
        if is_mla_enable and has_context_phase:
            return False, (
                "MLA context and mixed phases fall back to thop.attention until "
                "FlashInfer context support is ready."
            )
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
    fp8_context_fmha: bool,
) -> dict[str, int]:
    return thop.get_trtllm_gen_context_workspace_layout(
        dtype,
        batch_size,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        True,
        fp8_context_fmha,
    )


@lru_cache(maxsize=128)
def _get_context_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
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
        fp8_context_fmha,
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
    fp8_context_fmha: bool,
) -> int:
    context_size = _get_context_workspace_size(
        dtype,
        max_num_requests,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        fp8_context_fmha,
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
    """Per-call dynamic parameters for trtllm-gen attention.

    Layer-static properties (num_heads, head_size, rotary params, etc.) are
    read directly from ``FlashInferTrtllmGenAttention`` cached attributes
    to avoid redundant copies on every forward call.
    """

    forward: AttentionForwardArgs
    attention_input: Optional[torch.Tensor] = None
    qkv_input: Optional[torch.Tensor] = None
    context_buf: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    kv_cache_block_offsets: Optional[torch.Tensor] = None
    host_kv_cache_pool_pointers: Optional[torch.Tensor] = None
    host_kv_cache_pool_mapping: Optional[torch.Tensor] = None
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0
    num_tokens: int = 0
    seq_offset: int = 0
    tokens_per_block: int = 64
    mask_type: int = 1
    kv_cache_quant_mode: int = 0
    layer_idx: int = 0
    fp8_context_fmha: bool = False
    paged_context_fmha: bool = False
    kv_factor: int = 0
    total_num_blocks: int = 0
    # Context-only fields
    batch_size: int = 0
    # Generation-only fields
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
    # Keep shared paged indices disabled to match the current TensorRT-LLM
    # block-table layout used by the fused preprocessing path.
    USE_SHARED_PAGED_KV_IDX = False

    def __init__(
        self,
        attention_layer: "TrtllmAttention",
    ):
        self._attention_layer_ref = weakref.ref(attention_layer)
        self._checker = TrtllmGenSupportChecker()
        self._layout = self.DEFAULT_KV_LAYOUT
        # Read once so the hot path is not sensitive to later environment changes.
        self._enable_pdl = get_env_enable_pdl()
        missing_ops = self._missing_fused_nanobind_ops()
        if missing_ops:
            raise RuntimeError(
                f"trtllm-gen requires fused nanobind ops, missing: {', '.join(missing_ops)}."
            )

        # Cache layer-static properties to avoid repeated attribute lookups
        # through the weakref on every layer forward call.
        self._num_heads = attention_layer.num_heads
        self._num_kv_heads = attention_layer.num_kv_heads
        self._head_dim = attention_layer.head_dim
        self._quant_mode = attention_layer.quant_mode
        self._q_scaling = attention_layer.q_scaling
        self._position_embedding_type = attention_layer.position_embedding_type
        self._is_mla_enable = attention_layer.is_mla_enable
        self._kv_lora_rank = attention_layer.kv_lora_rank or 0
        self._qk_nope_head_dim = attention_layer.qk_nope_head_dim or 0
        self._qk_rope_head_dim = attention_layer.qk_rope_head_dim or 0
        self._v_head_dim = attention_layer.v_head_dim
        self._predicted_tokens_per_seq = attention_layer.predicted_tokens_per_seq
        self._rotary_embedding_dim = attention_layer.rope_params.dim
        self._rotary_embedding_base = attention_layer.rope_params.theta
        self._rotary_embedding_scale_type = int(attention_layer.rope_params.scale_type)
        self._rotary_embedding_scale = attention_layer.rope_params.scale
        self._rotary_embedding_max_positions = attention_layer.rope_params.max_positions
        self._bmm1_scale = 1.0 / (math.sqrt(self._head_dim) * self._q_scaling)
        self._rotary_inv_freq = attention_layer.rotary_inv_freq
        self._rotary_cos_sin = attention_layer.rotary_cos_sin
        self._attention_chunk_size = (
            attention_layer.attention_chunk_size
            if attention_layer.attention_chunk_size is not None
            else 0
        )

        # Static keyword args shared across preprocess / postprocess C++ calls.
        # Built once to avoid dict construction on every forward call.
        self._static_kw: dict[str, object] = dict(
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_dim,
            rotary_embedding_dim=self._rotary_embedding_dim,
            rotary_embedding_base=self._rotary_embedding_base,
            rotary_embedding_scale_type=self._rotary_embedding_scale_type,
            rotary_embedding_scale=self._rotary_embedding_scale,
            rotary_embedding_max_positions=self._rotary_embedding_max_positions,
            position_embedding_type=self._position_embedding_type,
            bmm1_scale=self._bmm1_scale,
            attention_chunk_size=self._attention_chunk_size,
        )

        # Cached is_supported() result.  None means not yet checked;
        # a positive result is stable (model-static) and cached permanently.
        self._support_result: Optional[Tuple[bool, str]] = None
        # Lazily set on the first attention() call from the query device.
        self._multi_processor_count: Optional[int] = None

    @property
    def layout(self) -> str:
        """KV cache layout."""
        return self._layout

    def _get_attention_layer(self) -> "TrtllmAttention":
        attention_layer = self._attention_layer_ref()
        if attention_layer is None:
            raise RuntimeError("trtllm-gen attention layer has been destroyed.")
        return attention_layer

    def _get_kv_scale_params(
        self,
        forward_args: AttentionForwardArgs,
        quant_mode: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        kv_cache_quant_mode = QuantMode(quant_mode)
        kv_scale_orig_quant = forward_args.kv_scale_orig_quant
        kv_scale_quant_orig = forward_args.kv_scale_quant_orig
        if (
            not kv_cache_quant_mode.has_kv_cache_quant()
            or kv_scale_orig_quant is None
            or kv_scale_quant_orig is None
        ):
            return None, None

        if kv_cache_quant_mode.has_fp4_kv_cache():
            assert kv_scale_orig_quant.size(0) == 3, (
                f"kv_scale_orig_quant must have size(0)==3 for FP4, got {kv_scale_orig_quant.size(0)}"
            )
            assert kv_scale_quant_orig.size(0) == 3, (
                f"kv_scale_quant_orig must have size(0)==3 for FP4, got {kv_scale_quant_orig.size(0)}"
            )

        return kv_scale_orig_quant, kv_scale_quant_orig

    @staticmethod
    def _get_mrope_rotary_cos_sin(
        forward_args: AttentionForwardArgs,
    ) -> Optional[torch.Tensor]:
        return forward_args.mrope_rotary_cos_sin

    def is_supported(
        self,
        q: torch.Tensor,
        *,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        mask_type: int,
        active_helix: bool,
        use_sage_attn: bool,
    ) -> Tuple[bool, str]:
        if use_sage_attn:
            return False, "trtllm-gen does not support sage attention."
        if active_helix:
            return False, "trtllm-gen does not support helix parallelism."
        # Return cached positive result after the first supported call.
        if self._support_result is not None:
            return self._support_result

        if not IS_FLASHINFER_AVAILABLE:
            return False, "flashinfer package is not installed."
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None:
            return False, "trtllm-gen requires a KVCacheManager."
        use_paged_kv_cache = metadata.kv_cache_block_offsets is not None
        if not use_paged_kv_cache:
            return False, "trtllm-gen requires paged KV cache."

        output = forward_args.output
        if output is None:
            return False, "trtllm-gen requires forward_args.output."

        attention_layer = self._get_attention_layer()
        sparse_attention_config = attention_layer.sparse_attention_config
        has_skip_softmax_attention = (
            getattr(sparse_attention_config, "algorithm", None) == "skip_softmax"
        )
        has_sparse_attention = (
            sparse_attention_config is not None and not has_skip_softmax_attention
        )
        result = self._checker.is_supported(
            q_dtype=q.dtype,
            kv_cache_dtype=kv_cache_manager.dtype,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_dim,
            attention_input_type=int(forward_args.attention_input_type),
            out_dtype=output.dtype,
            mask_type=mask_type,
            beam_width=metadata.beam_width,
            tokens_per_block=metadata.tokens_per_block,
            use_paged_kv_cache=use_paged_kv_cache,
            is_mla_enable=self._is_mla_enable,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            cross_attention=False,
            is_spec_decoding=metadata.is_spec_decoding_enabled,
            has_alibi=self._position_embedding_type in (4, 5),
            is_padded=False,
            position_shift_enabled=False,
            quant_config=attention_layer.quant_config,
            has_sparse_attention=has_sparse_attention,
            has_skip_softmax_attention=has_skip_softmax_attention,
        )
        if result[0]:
            self._support_result = result
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_multi_processor_count_for_device(device_index: int) -> int:
        return torch.cuda.get_device_properties(device_index).multi_processor_count

    def _get_multi_processor_count(self, device: torch.device) -> int:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("trtllm-gen requires CUDA tensors.")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return self._get_multi_processor_count_for_device(device_index)

    def attention(
        self,
        q: torch.Tensor,
        *,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        mask_type: int,
        use_paged_context_fmha: bool,
    ) -> None:
        attention_layer = self._get_attention_layer()
        layer_idx = attention_layer.get_local_layer_idx(metadata)
        logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

        output = forward_args.output
        if output is None:
            raise RuntimeError("trtllm-gen attention requires forward_args.output.")

        workspace = (
            metadata.workspace if not metadata.is_cuda_graph else metadata.cuda_graph_workspace
        )

        # Lazily cache the SM count from the first query tensor's device.
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(q.device)

        # Use cached layer-static properties.
        num_heads = self._num_heads
        num_kv_heads = self._num_kv_heads
        head_size = self._head_dim
        quant_mode = self._quant_mode
        is_mla_enable = self._is_mla_enable
        kv_lora_rank = self._kv_lora_rank
        v_head_dim = self._v_head_dim

        # Per-call dynamic values from metadata / forward_args.
        tokens_per_block = metadata.tokens_per_block
        max_num_requests = metadata.max_num_requests
        max_context_length = min(metadata.max_seq_len - 1, metadata.max_num_tokens)
        attention_window_size = forward_args.attention_window_size or metadata.max_seq_len
        beam_width = metadata.beam_width
        attention_input_type = int(forward_args.attention_input_type)

        is_fp8_out = output.dtype == torch.float8_e4m3fn
        is_fp4_out = output.dtype == torch.uint8
        kv_cache_quant_mode = QuantMode(quant_mode)
        fp8_context_fmha = (
            is_fp8_out
            or is_fp4_out
            or (kv_cache_quant_mode.has_fp8_kv_cache() and use_paged_context_fmha)
        )

        num_tokens = q.size(0)
        attn_input_type = AttentionInputType(attention_input_type)
        is_gen_only = attn_input_type == AttentionInputType.generation_only

        num_contexts = metadata.num_contexts
        num_ctx_tokens = metadata.num_ctx_tokens
        num_generations = metadata.host_request_types_runtime.size(0) - num_contexts
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
            self._rotary_embedding_dim,
            fp8_context_fmha,
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

        cache_indirection = metadata.cache_indirection
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
        kv_factor, total_num_blocks = self._get_kv_cache_metadata(metadata, is_mla_enable)
        params = EnqueueParams(
            forward=forward_args,
            workspace=workspace,
            max_attention_window_size=max_attn_window_size,
            cyclic_attention_window_size=cyclic_attn_window_size,
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            tokens_per_block=tokens_per_block if tokens_per_block is not None else 64,
            mask_type=mask_type,
            kv_cache_quant_mode=quant_mode,
            layer_idx=layer_idx,
            fp8_context_fmha=fp8_context_fmha,
            paged_context_fmha=use_paged_context_fmha,
            kv_factor=kv_factor,
            total_num_blocks=total_num_blocks,
        )

        sequence_length = metadata.kv_lens_cuda_runtime
        host_past_key_value_lengths = metadata.kv_lens_runtime
        context_lengths = metadata.prompt_lens_cuda_runtime
        host_context_lengths = metadata.prompt_lens_cpu_runtime

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
            self.run_context(params)

        if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
            seq_offset = num_contexts
            token_offset = 0 if is_gen_only else num_ctx_tokens
            num_seqs = num_generations

            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )
            input_seq_length = num_gen_tokens // num_seqs if num_seqs > 0 else 1

            predicted_tokens_per_seq = self._predicted_tokens_per_seq
            spec_gen_lengths = None
            spec_pos_offsets = None
            spec_packed_mask = None
            if (
                metadata.is_spec_decoding_enabled
                and metadata.use_spec_decoding
                and predicted_tokens_per_seq > 1
            ):
                spec_gen_lengths = metadata.spec_decoding_generation_lengths
                position_offsets_for_cpp = metadata.spec_decoding_position_offsets
                if position_offsets_for_cpp is not None and position_offsets_for_cpp.dim() == 1:
                    position_offsets_for_cpp = position_offsets_for_cpp.view(
                        metadata.max_num_requests, -1
                    )
                spec_pos_offsets = position_offsets_for_cpp
                spec_packed_mask = metadata.spec_decoding_packed_mask

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

    @staticmethod
    def _missing_fused_nanobind_ops() -> List[str]:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "get_trtllm_gen_generation_workspace_layout",
            "trtllm_gen_context_preprocess",
            "trtllm_gen_context_postprocess",
            "trtllm_gen_generation_preprocess",
            "build_trtllm_gen_kv_cache_metadata",
        )
        return [op for op in required_ops if not hasattr(thop, op)]

    def _get_kv_cache_metadata(
        self,
        metadata: "TrtllmAttentionMetadata",
        is_mla_enable: bool,
    ) -> Tuple[int, int]:
        """Return (kv_factor, total_num_blocks) for building KV cache views."""
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None:
            raise RuntimeError("trtllm-gen requires a KVCacheManager.")

        kv_factor = 1 if is_mla_enable else 2
        blocks_in_primary_pool = getattr(kv_cache_manager, "blocks_in_primary_pool", None)
        if blocks_in_primary_pool is None:
            blocks_per_window = getattr(kv_cache_manager, "blocks_per_window", None)
            if blocks_per_window:
                blocks_in_primary_pool = max(
                    int(primary) for primary, _ in blocks_per_window.values()
                )
        total_num_blocks = (
            int(blocks_in_primary_pool) * kv_cache_manager.num_local_layers * kv_factor
        )
        return kv_factor, total_num_blocks

    def run_context(
        self,
        params: EnqueueParams,
    ):
        kv_scale_orig_quant, kv_scale_quant_orig = self._get_kv_scale_params(
            params.forward, params.kv_cache_quant_mode
        )
        attention_output_orig_quant = params.forward.out_scale
        mrope_rotary_cos_sin = self._get_mrope_rotary_cos_sin(params.forward)

        (
            q_processed,
            kv_pool,
            block_tables,
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
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            attention_output_orig_quant=attention_output_orig_quant,
            rotary_inv_freq=self._rotary_inv_freq,
            rotary_cos_sin=self._rotary_cos_sin,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            layer_idx=params.layer_idx,
            tokens_per_block=params.tokens_per_block,
            mask_type=params.mask_type,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
            max_attention_window_size=params.max_attention_window_size,
            cyclic_attention_window_size=params.cyclic_attention_window_size,
            num_tokens=params.num_tokens,
            batch_size=params.batch_size,
            input_seq_length=params.input_seq_length,
            max_past_kv_length=params.max_past_kv_length,
            bmm2_scale=1.0,
            fp8_context_fmha=params.fp8_context_fmha,
            paged_context_fmha=params.paged_context_fmha,
            is_mla_enable=self._is_mla_enable,
            total_num_blocks=params.total_num_blocks,
            kv_factor=params.kv_factor,
            need_build_kv_cache_metadata=True,
            multi_processor_count=self._multi_processor_count,
            **self._static_kw,
        )

        # FlashInfer accepts a split K/V tuple; TensorRT-LLM stores both views
        # in one flat paged KV pool, so both tuple entries intentionally alias.
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q_processed,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=fmha_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self._bmm1_scale,
            bmm2_scale=1.0,
            batch_size=params.batch_size,
            cum_seq_lens_q=cu_q_seqlens,
            cum_seq_lens_kv=cu_kv_seqlens,
            window_left=window_left,
            out=params.context_buf,
            kv_layout=self._layout,
            sinks=params.forward.attention_sinks,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            enable_pdl=self._enable_pdl,
        )

        thop.trtllm_gen_context_postprocess(
            qkv_input=params.qkv_input,
            workspace=params.workspace,
            sequence_lengths=params.sequence_lengths,
            context_lengths=params.context_lengths,
            kv_cache_block_offsets=params.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            attention_output_orig_quant=attention_output_orig_quant,
            rotary_cos_sin=self._rotary_cos_sin,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            layer_idx=params.layer_idx,
            tokens_per_block=params.tokens_per_block,
            mask_type=params.mask_type,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
            max_attention_window_size=params.max_attention_window_size,
            cyclic_attention_window_size=params.cyclic_attention_window_size,
            num_tokens=params.num_tokens,
            batch_size=params.batch_size,
            input_seq_length=params.input_seq_length,
            max_past_kv_length=params.max_past_kv_length,
            fp8_context_fmha=params.fp8_context_fmha,
            paged_context_fmha=params.paged_context_fmha,
            is_mla_enable=self._is_mla_enable,
            multi_processor_count=self._multi_processor_count,
            **self._static_kw,
        )

    def run_generation(
        self,
        params: EnqueueParams,
    ):
        batch_beam = params.num_requests * params.beam_width
        kv_scale_orig_quant, kv_scale_quant_orig = self._get_kv_scale_params(
            params.forward, params.kv_cache_quant_mode
        )
        attention_output_orig_quant = params.forward.out_scale
        (
            q_processed,
            kv_pool,
            block_tables,
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
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            attention_output_orig_quant=attention_output_orig_quant,
            rotary_inv_freq=self._rotary_inv_freq,
            rotary_cos_sin=self._rotary_cos_sin,
            layer_idx=params.layer_idx,
            seq_offset=params.seq_offset,
            tokens_per_block=params.tokens_per_block,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
            max_attention_window_size=params.max_attention_window_size,
            cyclic_attention_window_size=params.cyclic_attention_window_size,
            num_tokens=params.num_tokens,
            batch_beam=batch_beam,
            input_seq_length=params.input_seq_length,
            max_past_kv_length=params.max_past_kv_length,
            bmm2_scale=1.0,
            fp8_context_fmha=params.fp8_context_fmha,
            predicted_tokens_per_seq=params.predicted_tokens_per_seq,
            multi_processor_count=self._multi_processor_count,
            total_num_blocks=params.total_num_blocks,
            kv_factor=params.kv_factor,
            need_build_kv_cache_metadata=True,
            **self._static_kw,
        )

        q_len_per_req = None if is_multi_token_gen else params.input_seq_length
        decode_max_q_len = max_q_len if is_multi_token_gen else None
        decode_cu_seqlens = cu_seqlens if is_multi_token_gen else None
        # FlashInfer accepts a split K/V tuple; TensorRT-LLM stores both views
        # in one flat paged KV pool, so both tuple entries intentionally alias.
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q_processed,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=fmha_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_seq_len=max_kv_len,
            out=params.context_buf,
            bmm1_scale=self._bmm1_scale,
            bmm2_scale=1.0,
            window_left=window_left,
            kv_layout=self._layout,
            sinks=params.forward.attention_sinks,
            q_len_per_req=q_len_per_req,
            max_q_len=decode_max_q_len,
            cum_seq_lens_q=decode_cu_seqlens,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            enable_pdl=self._enable_pdl,
            backend="trtllm-gen",
        )

    def run_mla_generation(
        self,
        params: EnqueueParams,
    ) -> None:
        """MLA generation decode using flashinfer MLA kernel."""
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            raise NotImplementedError(
                "Sliding-window attention is not supported by MLA decode path."
            )
        if self._attention_chunk_size != 0:
            raise NotImplementedError("Chunked-attention is not supported by MLA decode path.")

        batch_beam = params.num_requests * params.beam_width
        if params.attention_input is None:
            raise RuntimeError("MLA generation requires attention_input.")
        kv_cache, block_tables = thop.build_trtllm_gen_kv_cache_metadata(
            host_kv_cache_pool_pointers=params.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=params.host_kv_cache_pool_mapping,
            kv_cache_block_offsets=params.kv_cache_block_offsets,
            layer_idx=params.layer_idx,
            num_kv_heads=self._num_kv_heads,
            tokens_per_block=params.tokens_per_block,
            head_dim=self._head_dim,
            kv_factor=params.kv_factor,
            total_num_blocks=params.total_num_blocks,
            kv_cache_quant_mode=params.kv_cache_quant_mode,
            batch_start=params.seq_offset,
            batch_size=batch_beam,
            dtype=params.attention_input.dtype,
        )

        pages_per_superblock = 128 // params.tokens_per_block
        if pages_per_superblock > 1:
            num_blocks = block_tables.size(-1)
            remainder = num_blocks % pages_per_superblock
            if remainder != 0:
                pad = pages_per_superblock - remainder
                block_tables = torch.nn.functional.pad(block_tables, (0, pad), value=0)

        kv_lora_rank = self._kv_lora_rank
        qk_nope_head_dim = self._qk_nope_head_dim
        qk_rope_head_dim = self._qk_rope_head_dim
        mla_head_dim_qk = kv_lora_rank + qk_rope_head_dim
        q_len_per_req = params.num_tokens // batch_beam if batch_beam > 0 else 1

        query = params.qkv_input.view(batch_beam, q_len_per_req, self._num_heads, mla_head_dim_qk)

        bmm1_scale = 1.0 / (self._q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))

        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=params.workspace.view(-1, 4),
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_seq_len=params.max_past_kv_length,
            out=params.context_buf.view(batch_beam, q_len_per_req, self._num_heads, kv_lora_rank),
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            sinks=params.forward.attention_sinks,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            enable_pdl=self._enable_pdl,
            backend="trtllm-gen",
        )
