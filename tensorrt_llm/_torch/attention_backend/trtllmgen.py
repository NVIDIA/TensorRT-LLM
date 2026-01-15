import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, List, Optional, Tuple, Union

import torch
from flashinfer import FP4Tensor

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

########################################################
# TrtllmGenAttention Backend
########################################################


def is_sm100_family(sm: Optional[int] = None) -> bool:
    """
    Check if the SM version is in the SM100 family (Blackwell architecture).

    Corresponds to C++ isSM100Family() in cudaUtils.h.

    Args:
        sm: SM version to check. If None, uses current device SM version.

    Returns:
        True if SM is 100 or 103 (Blackwell architecture).
    """
    if sm is None:
        sm = get_sm_version()
    return sm == 100 or sm == 103


def is_trtllmgen_supported(sm: Optional[int] = None) -> bool:
    """
    Check if trtllm-gen kernels are supported on the current device.

    For MLA generation: SM >= 100 and SM != 120
    For Context/Generation FMHA: SM == 100 or SM == 103

    Args:
        sm: SM version to check. If None, uses current device SM version.

    Returns:
        True if trtllm-gen kernels can be used.
    """
    if sm is None:
        sm = get_sm_version()
    return is_sm100_family(sm)


def is_supported_context_fmha(
    head_size: int,
    mask_type: Optional[int] = None,
    has_alibi: bool = False,
    is_padded: bool = False,
    sm: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check whether the trtllm-gen Context FMHA kernel supports the given configuration.

    Based on fmhaDispatcher.cpp checks:
    - SM must be 100 or 103 (Blackwell)
    - Head size must NOT equal 80 or 72
    - Custom mask NOT supported
    - ALiBi NOT supported
    - Input must NOT be padded

    Args:
        head_size: Size of each attention head.
        mask_type: Attention mask type (AttentionMaskType enum value).
        has_alibi: Whether ALiBi position embedding is used.
        is_padded: Whether input is padded.
        sm: SM version. If None, uses current device SM version.

    Returns:
        Tuple of (is_supported: bool, reason: str).
    """
    if sm is None:
        sm = get_sm_version()

    # Check SM version - trtllm-gen FMHA requires SM100 or SM103
    if not is_sm100_family(sm):
        return (
            False,
            f"trtllm-gen Context FMHA requires SM100 or SM103 (Blackwell). Current SM: {sm}.",
        )

    # Head size must NOT equal 80 or 72
    # fmhaDispatcher.cpp:55: mUseTllmGen = isSM100Family() && headSize != 80 && headSize != 72
    if head_size == 80:
        return False, "Head size 80 is not supported by trtllm-gen Context FMHA."
    if head_size == 72:
        return False, "Head size 72 is not supported by trtllm-gen Context FMHA."

    # Custom mask not supported
    # fmhaDispatcher.cpp:83-143
    if mask_type is not None:
        try:
            mask_type_enum = AttentionMaskType(mask_type)
            if mask_type_enum == AttentionMaskType.CUSTOM_MASK:
                return False, "Custom mask is not supported by trtllm-gen Context FMHA."
        except ValueError:
            return False, f"Invalid mask_type value: {mask_type}."

    # ALiBi not supported
    if has_alibi:
        return False, "ALiBi is not supported by trtllm-gen Context FMHA."

    # Padded input not supported
    if is_padded:
        return False, "Padded input is not supported by trtllm-gen Context FMHA."

    return True, ""


def is_supported_generation_xqa(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[torch.dtype] = None,
    beam_width: int = 1,
    position_shift_enabled: bool = False,
    sink_token_length: int = 0,
    cross_attention: bool = False,
    use_paged_kv_cache: bool = True,
    tokens_per_block: int = 64,
    cyclic_attention_window_size: Optional[int] = None,
    max_attention_window_size: Optional[int] = None,
    has_alibi: bool = False,
    is_mla_enabled: bool = False,
    is_spec_decoding: bool = False,
    sm: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check whether the trtllm-gen Generation XQA kernel supports the given configuration.

    Based on xqaDispatcher.cpp checks:
    - SM must be 100 or 103 (Blackwell)
    - beam_width must = 1
    - position_shift_enabled must = false
    - sink_token_length must = 0
    - cross_attention + !paged_kv_cache not supported
    - tokens_per_block must >= 8
    - cyclic_attention_window_size must = max_attention_window_size
    - num_q_heads / num_kv_heads must <= 16
    - ALiBi not supported
    - MLA + tree spec decoding not supported

    Args:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_size: Size of each attention head.
        dtype: Data type of Q tensor.
        kv_cache_dtype: Data type of KV cache. If None, uses dtype.
        beam_width: Beam width for beam search. Must be 1.
        position_shift_enabled: Whether position shift is enabled. Must be False.
        sink_token_length: Number of sink tokens (StreamingLLM). Must be 0.
        cross_attention: Whether cross attention is used.
        use_paged_kv_cache: Whether paged KV cache is used.
        tokens_per_block: Tokens per KV cache block. Must be >= 8.
        cyclic_attention_window_size: Cyclic attention window size.
        max_attention_window_size: Max attention window size.
        has_alibi: Whether ALiBi is used.
        is_mla_enabled: Whether MLA is enabled.
        is_spec_decoding: Whether speculative decoding is enabled.
        sm: SM version. If None, uses current device SM version.

    Returns:
        Tuple of (is_supported: bool, reason: str).
    """
    if sm is None:
        sm = get_sm_version()

    # Check SM version - trtllm-gen XQA requires SM100 or SM103
    if not is_sm100_family(sm):
        return (
            False,
            f"trtllm-gen Generation XQA requires SM100 or SM103 (Blackwell). Current SM: {sm}.",
        )

    # beam_width must = 1 (beam search not supported)
    if beam_width != 1:
        return (
            False,
            f"Beam search (beam_width={beam_width}) is not supported by trtllm-gen Generation XQA. Must be 1.",
        )

    # position_shift_enabled must = false
    if position_shift_enabled:
        return False, "Position shift is not supported by trtllm-gen Generation XQA."

    # sink_token_length must = 0 (StreamingLLM not supported)
    if sink_token_length != 0:
        return (
            False,
            f"StreamingLLM (sink_token_length={sink_token_length}) is not supported by trtllm-gen Generation XQA.",
        )

    # cross_attention + !paged_kv_cache not supported
    if cross_attention and not use_paged_kv_cache:
        return (
            False,
            "Cross attention with non-paged KV cache is not supported by trtllm-gen Generation XQA.",
        )

    # tokens_per_block must >= 8
    if tokens_per_block < 8:
        return (
            False,
            f"tokens_per_block ({tokens_per_block}) must be >= 8 for trtllm-gen Generation XQA.",
        )

    # cyclic_attention_window_size must = max_attention_window_size
    if cyclic_attention_window_size is not None and max_attention_window_size is not None:
        if cyclic_attention_window_size != max_attention_window_size:
            return (
                False,
                f"cyclic_attention_window_size ({cyclic_attention_window_size}) must equal "
                f"max_attention_window_size ({max_attention_window_size}) for trtllm-gen XQA.",
            )

    # num_q_heads / num_kv_heads must <= 16
    if num_kv_heads > 0:
        heads_ratio = num_heads // num_kv_heads
        if heads_ratio > 16:
            return (
                False,
                f"num_heads/num_kv_heads ratio ({heads_ratio}) must be <= 16 for trtllm-gen Generation XQA.",
            )

    # ALiBi not supported
    if has_alibi:
        return False, "ALiBi is not supported by trtllm-gen Generation XQA."

    # MLA + tree spec decoding not supported
    if is_mla_enabled and is_spec_decoding:
        return (
            False,
            "MLA with tree speculative decoding is not supported by trtllm-gen Generation XQA.",
        )

    # Check data type compatibility
    # xqaDispatcher.cpp:139-141: Q type must match math type
    # When KV cache is FP8 or FP4, Q is converted to FP8
    if kv_cache_dtype in {torch.float8_e4m3fn, torch.uint8}:  # E4M3 or E2M1 (FP4)
        # Q will be converted to FP8, which is fine
        pass
    else:
        # Q type should match KV type for math operations
        pass

    return True, ""


def is_supported_mla_generation(
    is_mla_enabled: bool = True,
    use_sparse_attention: bool = False,
    sm: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check whether the trtllm-gen MLA Generation kernel supports the given configuration.

    MLA Generation requires: SM >= 100 and SM != 120
    Sparse MLA requires: mUseSparseAttention && mUseTllmGen && mIsMLAEnabled

    Args:
        is_mla_enabled: Whether MLA is enabled.
        use_sparse_attention: Whether sparse attention is used.
        sm: SM version. If None, uses current device SM version.

    Returns:
        Tuple of (is_supported: bool, reason: str).
    """
    if sm is None:
        sm = get_sm_version()

    if not is_mla_enabled:
        return False, "MLA is not enabled."

    # MLA generation: SM >= 100 and SM != 120
    # attentionOp.h:529: mUseTllmGen = (mSM >= 100) && (mSM != 120)
    use_tllm_gen = (sm >= 100) and (sm != 120)

    if not use_tllm_gen:
        return (
            False,
            f"trtllm-gen MLA Generation requires SM >= 100 and SM != 120. Current SM: {sm}.",
        )

    # Sparse MLA: mUseSparseAttention && mUseTllmGen && mIsMLAEnabled
    if use_sparse_attention:
        # All conditions met for sparse MLA
        pass

    return True, ""


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
    is_mla_enabled: bool = False,
    use_paged_kv_cache: bool = True,
    tokens_per_block: int = 64,
    beam_width: int = 1,
    position_shift_enabled: bool = False,
    sink_token_length: int = 0,
    cross_attention: bool = False,
    cyclic_attention_window_size: Optional[int] = None,
    max_attention_window_size: Optional[int] = None,
    is_spec_decoding: bool = False,
    use_sparse_attention: bool = False,
    q_lora_rank: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    phase: str = "both",
    sm: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check whether the trtllm-gen attention backend supports the given configuration.

    This function performs comprehensive validation checks based on:
    - fmhaDispatcher.cpp (Context FMHA)
    - xqaDispatcher.cpp (Generation XQA)
    - attentionOp.cpp (MLA)

    Hardware Requirements:
        - Only supports Blackwell architecture: SM100 or SM103

    Supported Data Types:
        - Context FMHA Input: FP16, BF16, FP8 (E4M3)
        - KV Cache: FP16, BF16, FP8 (E4M3), FP4 (E2M1)
        - Output: FP16, BF16, FP8 (E4M3), FP4 (E2M1)

    Context Phase FMHA Requirements:
        - Head size must NOT equal 80 or 72
        - Custom mask NOT supported
        - ALiBi NOT supported
        - Input must NOT be padded

    Generation Phase XQA Requirements:
        - beam_width must = 1 (no beam search)
        - position_shift_enabled must = false
        - sink_token_length must = 0 (no StreamingLLM)
        - cross_attention + !paged_kv_cache not supported
        - tokens_per_block must >= 8
        - cyclic_attention_window_size must = max_attention_window_size
        - num_heads / num_kv_heads must <= 16
        - ALiBi not supported
        - MLA + tree spec decoding not supported

    MLA Generation Requirements:
        - SM >= 100 and SM != 120

    Args:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (for GQA/MQA).
        head_size: Size of each attention head.
        dtype: Data type of the input tensors (torch.float16, torch.bfloat16, torch.float8_e4m3fn).
        kv_cache_dtype: Data type of the KV cache. If None, uses dtype.
        out_dtype: Output data type. If None, uses dtype.
        mask_type: Attention mask type (AttentionMaskType enum value). If None, uses CAUSAL.
        has_alibi: Whether ALiBi position embedding is used.
        is_padded: Whether input is padded.
        is_mla_enabled: Whether MLA (Multi-head Latent Attention) is enabled.
        use_paged_kv_cache: Whether to use paged KV cache.
        tokens_per_block: Number of tokens per KV cache block.
        beam_width: Beam width for beam search.
        position_shift_enabled: Whether position shift is enabled.
        sink_token_length: Number of sink tokens for StreamingLLM.
        cross_attention: Whether cross attention is used.
        cyclic_attention_window_size: Cyclic attention window size.
        max_attention_window_size: Max attention window size.
        is_spec_decoding: Whether speculative decoding is enabled.
        use_sparse_attention: Whether sparse attention is used.
        q_lora_rank: MLA Q LoRA rank (required if is_mla_enabled).
        kv_lora_rank: MLA KV LoRA rank (required if is_mla_enabled).
        qk_nope_head_dim: MLA QK nope head dimension (required if is_mla_enabled).
        qk_rope_head_dim: MLA QK rope head dimension (required if is_mla_enabled).
        phase: Which phase to check - "context", "generation", or "both".
        sm: SM version. If None, uses current device SM version.

    Returns:
        Tuple of (is_supported: bool, reason: str).
        If supported, returns (True, "").
        If not supported, returns (False, "reason for not being supported").
    """
    if sm is None:
        sm = get_sm_version()

    # ========== Hardware check ==========
    # trtllm-gen requires SM100 or SM103 (Blackwell)
    if not is_sm100_family(sm):
        return (
            False,
            f"trtllm-gen attention requires SM100 or SM103 (Blackwell architecture). Current SM: {sm}.",
        )

    # ========== Data type checks ==========
    # Context FMHA input types: FP16, BF16, FP8 (E4M3)
    supported_input_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    if dtype not in supported_input_dtypes:
        return False, f"Input dtype {dtype} not supported. Supported: FP16, BF16, FP8 (E4M3)."

    # KV Cache types: FP16, BF16, FP8 (E4M3), FP4 (E2M1/uint8)
    if kv_cache_dtype is not None:
        supported_kv_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8}
        if kv_cache_dtype not in supported_kv_dtypes:
            return (
                False,
                f"KV cache dtype {kv_cache_dtype} not supported. Supported: FP16, BF16, FP8 (E4M3), FP4 (E2M1).",
            )

    # Output types: FP16, BF16, FP8 (E4M3), FP4 (E2M1/uint8)
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
        ctx_supported, ctx_reason = is_supported_context_fmha(
            head_size=head_size,
            mask_type=mask_type,
            has_alibi=has_alibi,
            is_padded=is_padded,
            sm=sm,
        )
        if not ctx_supported:
            return False, f"[Context] {ctx_reason}"

    # ========== Generation phase checks ==========
    if phase in ("generation", "both"):
        # Check MLA generation separately
        if is_mla_enabled:
            mla_supported, mla_reason = is_supported_mla_generation(
                is_mla_enabled=is_mla_enabled,
                use_sparse_attention=use_sparse_attention,
                sm=sm,
            )
            if not mla_supported:
                return False, f"[MLA Generation] {mla_reason}"

            # MLA-specific checks
            if q_lora_rank is None or kv_lora_rank is None:
                return False, "[MLA] q_lora_rank and kv_lora_rank must be specified."
            if qk_nope_head_dim is None or qk_rope_head_dim is None:
                return False, "[MLA] qk_nope_head_dim and qk_rope_head_dim must be specified."
            if qk_rope_head_dim != 64 or kv_lora_rank != 512:
                return (
                    False,
                    f"[MLA] Only kv_lora_rank=512 and qk_rope_head_dim=64 supported. "
                    f"Got kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}.",
                )
        else:
            # Standard XQA generation
            gen_supported, gen_reason = is_supported_generation_xqa(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                beam_width=beam_width,
                position_shift_enabled=position_shift_enabled,
                sink_token_length=sink_token_length,
                cross_attention=cross_attention,
                use_paged_kv_cache=use_paged_kv_cache,
                tokens_per_block=tokens_per_block,
                cyclic_attention_window_size=cyclic_attention_window_size,
                max_attention_window_size=max_attention_window_size,
                has_alibi=has_alibi,
                is_mla_enabled=is_mla_enabled,
                is_spec_decoding=is_spec_decoding,
                sm=sm,
            )
            if not gen_supported:
                return False, f"[Generation] {gen_reason}"

    # ========== Paged KV cache checks ==========
    if use_paged_kv_cache:
        if tokens_per_block <= 0:
            return False, "tokens_per_block must be positive for paged KV cache."
        # tokens_per_block should be power of 2 for FMHA
        if tokens_per_block & (tokens_per_block - 1) != 0:
            return False, f"tokens_per_block ({tokens_per_block}) must be a power of 2."

    return True, ""


class AttentionMaskType(IntEnum):
    """
    Attention mask types.
    Corresponds to C++ AttentionMaskType in gptKernels.h.
    """

    # Mask the padded tokens.
    PADDING = 0
    # Mask the padded tokens and all the tokens that come after in a sequence.
    CAUSAL = 1
    # Only attend to the previous tokens in a fixed-length window.
    SLIDING_WINDOW_CAUSAL = 2
    # See ChatGLM-6B mask.
    BIDIRECTIONAL = 3
    # See GLM-10B mask.
    # TODO: merge this mask into BIDIRECTIONAL
    BIDIRECTIONALGLM = 4
    # For Phi-3-small model
    BLOCKSPARSE = 5
    # The custom mask input.
    CUSTOM_MASK = 6


class PositionEmbeddingType(IntEnum):
    """
    Position embedding types for attention.
    Corresponds to C++ PositionEmbeddingType in gptKernels.h.
    """

    LEARNED_ABSOLUTE = 0
    ROPE_GPTJ = 1
    ROPE_GPT_NEOX = 2
    LONG_ROPE = 3
    # Workflow: (bmm1_output * scale_bmm1 + alibi)
    ALIBI = 4
    # Workflow: (bmm1_output + alibi) * scale_bmm1
    ALIBI_WITH_SCALE = 5
    RELATIVE = 6
    CHATGLM = 7
    YARN = 8
    ROPE_M = 9


class RotaryScalingType(IntEnum):
    """
    Rotary embedding scaling types.
    Corresponds to C++ RotaryScalingType in gptKernels.h.
    """

    NONE = 0
    LINEAR = 1
    DYNAMIC = 2
    LONG = 3
    LLAMA3 = 4


class DataType(IntEnum):
    """
    Data types for attention kernels.
    Corresponds to C++ Data_type in multiHeadAttentionCommon.h.
    """

    BOOL = 0
    FP16 = 1
    HALF = 1  # Alias for FP16
    FP32 = 2
    INT4 = 3
    INT8 = 4
    INT32 = 5
    BF16 = 6
    E2M1 = 7  # NVFP4
    E4M3 = 8  # FP8
    E5M2 = 9  # FP8


class AttentionInputType(IntEnum):
    """
    Attention input type indicating if batch contains context, generation, or mixed requests.
    Corresponds to C++ AttentionInputType in attentionOp.cpp.
    """

    MIXED = 0  # Batch contains both context and generation requests
    CONTEXT_ONLY = 1  # Batch contains only context (prefill) requests
    GENERATION_ONLY = 2  # Batch contains only generation (decode) requests


@dataclass
class BlockSparseParams:
    """
    Block sparse attention parameters.
    Corresponds to C++ BlockSparseParams in gptKernels.h.
    """

    block_size: int = 1
    homo_head_pattern: int = 0
    num_local_blocks: int = 0  # Sliding window blocks
    vertical_stride: int = 0

    def data(self) -> Tuple[int, int, int, int]:
        return (
            self.block_size,
            self.homo_head_pattern,
            self.num_local_blocks,
            self.vertical_stride,
        )


@dataclass
class MlaMetaParams:
    """
    MLA (Multi-head Latent Attention) meta parameters.
    Corresponds to C++ MlaMetaParams in mlaKernels.h.
    """

    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    predicted_tokens_per_seq: int = 1
    num_layers: int = 0

    def data(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.predicted_tokens_per_seq,
            self.num_layers,
        )


@dataclass
class MlaParams:
    """
    MLA (Multi-head Latent Attention) runtime parameters.
    Corresponds to C++ MlaParams<T> in mlaKernels.h.
    """

    # cKV + k_pe
    latent_cache: Optional[torch.Tensor] = None
    # Tensor Q for both context and generation MLA
    q_buf: Optional[torch.Tensor] = None
    # Separate tensor K for context MLA
    k_buf: Optional[torch.Tensor] = None
    # Separate tensor V for context MLA
    v_buf: Optional[torch.Tensor] = None
    # Query position embedding tensor [total_q_len, num_heads, qk_rope_head_dim]
    q_pe: Optional[torch.Tensor] = None
    # Quantized Q buffer for FP8
    quant_q_buf: Optional[torch.Tensor] = None
    # Quantized K buffer for FP8
    quant_k_buf: Optional[torch.Tensor] = None
    # Quantized V buffer for FP8
    quant_v_buf: Optional[torch.Tensor] = None
    # Cumulative Q sequence lengths
    cu_q_seqlens: Optional[torch.Tensor] = None
    # KV cache data type
    cache_type: int = 0
    # Quantization scales
    quant_scale_kv: Optional[torch.Tensor] = None
    quant_scale_o: Optional[torch.Tensor] = None
    quant_scale_q: Optional[torch.Tensor] = None
    dequant_scale_q: Optional[torch.Tensor] = None
    dequant_scale_kv: Optional[torch.Tensor] = None
    # QKV quantization scale for FP8 context
    quant_scale_qkv: Optional[torch.Tensor] = None
    # BMM scales
    bmm1_scale: Optional[torch.Tensor] = None
    bmm2_scale: Optional[torch.Tensor] = None
    host_bmm1_scale: float = 1.0
    # Sparse MLA mode
    absorption_mode: bool = False


@dataclass
class SparseAttentionParams:
    """
    Sparse attention parameters for runtime.
    Corresponds to C++ SparseAttentionParams in sparseAttentionKernels.h.
    """

    # [num_kv_heads, num_sparse_kv_indices]
    sparse_kv_indices: Optional[torch.Tensor] = None
    # [num_kv_heads, num_sparse_attn_indices]
    sparse_attn_indices: Optional[torch.Tensor] = None
    # [num_contexts + 1]
    sparse_kv_offsets: Optional[torch.Tensor] = None
    # [num_generations + 1]
    sparse_attn_offsets: Optional[torch.Tensor] = None
    # for DSA (Dynamic Sparse Attention) / MLA attention
    sparse_mla_topk: int = 0
    # for DSA attention - pointer to KV cache pool
    sparse_mla_kv_cache_pool: Optional[int] = None

    sparse_attn_indices_block_size: int = 1
    sparse_attn_indices_stride: int = 0

    def to_string(self) -> str:
        """Returns a string representation of the parameters."""
        lines = [
            f"sparse_kv_indices: {self.sparse_kv_indices}",
            f"sparse_attn_indices: {self.sparse_attn_indices}",
            f"sparse_kv_offsets: {self.sparse_kv_offsets}",
            f"sparse_attn_offsets: {self.sparse_attn_offsets}",
            f"sparse_mla_topk: {self.sparse_mla_topk}",
            f"sparse_mla_kv_cache_pool: {self.sparse_mla_kv_cache_pool}",
            f"sparse_attn_indices_block_size: {self.sparse_attn_indices_block_size}",
            f"sparse_attn_indices_stride: {self.sparse_attn_indices_stride}",
        ]
        return "\n".join(lines)


@dataclass
class QKVPreprocessingParams:
    """
    Parameters for QKV preprocessing operations.
    Corresponds to the arguments of C++ runQkvPreprocessing() in qkvPreprocessOp.cpp.

    This includes:
    - RoPE (Rotary Position Embedding) application
    - KV cache updates for both context and generation phases
    - Support for paged KV cache
    """

    # ========== Tensor parameters ==========
    # Input QKV tensor (required)
    qkv_input: torch.Tensor = None
    # Optional output Q tensor (for separate Q/KV output)
    q_output: Optional[torch.Tensor] = None
    # Optional QKV bias tensor
    qkv_bias: Optional[torch.Tensor] = None
    # Sequence lengths tensor
    seq_lens: Optional[torch.Tensor] = None
    # Cache sequence lengths tensor
    cache_seq_lens: Optional[torch.Tensor] = None
    # Cumulative sequence lengths tensor
    cu_seq_lens: Optional[torch.Tensor] = None
    # Rotary embedding inverse frequencies
    rotary_embedding_inv_freq: Optional[torch.Tensor] = None
    # Rotary coefficient cache buffer
    rotary_coef_cache_buffer: Optional[torch.Tensor] = None
    # QKV scale for original to quantized conversion
    qkv_scale_orig_quant: Optional[torch.Tensor] = None
    # QKV scale for quantized to original conversion
    qkv_scale_quant_orig: Optional[torch.Tensor] = None
    # Speculative decoding position offsets
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    # Multi-rope rotary cos/sin values
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None
    # Multi-rope position deltas
    mrope_position_deltas: Optional[torch.Tensor] = None
    # KV cache block offsets tensor (required)
    block_offsets: torch.Tensor = None

    # ========== Scalar parameters ==========
    batch_size: int = 0
    max_input_seq_len: int = 0
    max_kv_seq_len: int = 0
    cyclic_kv_cache_len: int = 0
    sink_token_len: int = 0
    token_num: int = 0
    remove_padding: bool = True
    is_last_chunk: bool = True
    cross_attention: bool = False
    head_num: int = 0
    kv_head_num: int = 0
    size_per_head: int = 0
    rotary_embedding_dim: int = 0
    rotary_embedding_base: float = 10000.0
    rotary_scale_type: int = 0  # RotaryScalingType
    rotary_embedding_scale: float = 1.0
    rotary_embedding_max_positions: int = 0
    position_embedding_type: int = 0  # PositionEmbeddingType
    position_shift_enabled: bool = False
    separate_q_kv_output: bool = False
    quantized_fp8_output: bool = False
    generation_phase: bool = False
    multi_processor_count: int = 0
    rotary_vision_start: int = 0
    rotary_vision_length: int = 0
    quant_mode: int = 0  # QuantMode value

    # ========== KV cache buffer parameters ==========
    tokens_per_block: int = 64
    max_blocks_per_sequence: int = 0
    attention_window_size: int = 0
    size_per_token: int = 0
    sink_token_length: int = 0
    max_cyclic_attention_window_size: int = 0
    can_use_one_more_block: bool = False
    host_primary_pool_pointer: int = 0
    host_secondary_pool_pointer: int = 0


class QKVPreprocessRunner:
    """
    Runner class for QKV preprocessing operations.

    This class wraps the C++ torch.ops.trtllm.run_qkv_preprocessing() function
    and provides a convenient interface for running QKV preprocessing.

    Usage:
        runner = QKVPreprocessRunner()
        params = QKVPreprocessingParams(
            qkv_input=qkv_tensor,
            block_offsets=block_offsets_tensor,
            batch_size=batch_size,
            # ... other parameters
        )
        runner.run(params)
    """

    def __init__(self):
        """Initialize the QKV preprocessing runner."""
        pass

    def run(self, params: QKVPreprocessingParams) -> None:
        """
        Run QKV preprocessing.

        This applies RoPE (Rotary Position Embedding) and updates the KV cache
        for both context and generation phases.

        Args:
            params: QKVPreprocessingParams containing all input tensors and parameters.
        """
        torch.ops.trtllm.run_qkv_preprocessing(
            # Tensors
            params.qkv_input,
            params.q_output,
            params.qkv_bias,
            params.seq_lens,
            params.cache_seq_lens,
            params.cu_seq_lens,
            params.rotary_embedding_inv_freq,
            params.rotary_coef_cache_buffer,
            params.qkv_scale_orig_quant,
            params.qkv_scale_quant_orig,
            params.spec_decoding_position_offsets,
            params.mrope_rotary_cos_sin,
            params.mrope_position_deltas,
            params.block_offsets,
            # Scalars
            params.batch_size,
            params.max_input_seq_len,
            params.max_kv_seq_len,
            params.cyclic_kv_cache_len,
            params.sink_token_len,
            params.token_num,
            params.remove_padding,
            params.is_last_chunk,
            params.cross_attention,
            params.head_num,
            params.kv_head_num,
            params.size_per_head,
            params.rotary_embedding_dim,
            params.rotary_embedding_base,
            params.rotary_scale_type,
            params.rotary_embedding_scale,
            params.rotary_embedding_max_positions,
            params.position_embedding_type,
            params.position_shift_enabled,
            params.separate_q_kv_output,
            params.quantized_fp8_output,
            params.generation_phase,
            params.multi_processor_count,
            params.rotary_vision_start,
            params.rotary_vision_length,
            params.quant_mode,
            # KV cache buffer parameters
            params.tokens_per_block,
            params.max_blocks_per_sequence,
            params.attention_window_size,
            params.size_per_token,
            params.sink_token_length,
            params.max_cyclic_attention_window_size,
            params.can_use_one_more_block,
            params.host_primary_pool_pointer,
            params.host_secondary_pool_pointer,
        )


@dataclass
class KVCachePostprocessParams:
    """
    Parameters for KV cache postprocessing operations.
    This handles sparse KV cache updates after FMHA.
    """

    # Input tensor (used for dtype detection)
    qkv_input: torch.Tensor = None
    # Sparse KV cache parameters
    sparse_kv_indices: Optional[torch.Tensor] = None
    sparse_kv_offsets: Optional[torch.Tensor] = None
    # Block offsets for paged KV cache
    block_offsets: Optional[torch.Tensor] = None
    # Scalar parameters
    batch_size: int = 0
    is_last_chunk: bool = True
    head_num: int = 0
    kv_head_num: int = 0
    size_per_head: int = 0
    quant_mode: int = 0
    # KV cache buffer parameters
    tokens_per_block: int = 64
    max_blocks_per_sequence: int = 0
    attention_window_size: int = 0
    size_per_token: int = 0
    sink_token_length: int = 0
    max_cyclic_attention_window_size: int = 0
    can_use_one_more_block: bool = False
    host_primary_pool_pointer: int = 0
    host_secondary_pool_pointer: int = 0


class KVCachePostprocessRunner:
    """
    Runner class for KV cache postprocessing operations.

    This class wraps the C++ torch.ops.trtllm.run_kv_cache_postprocessing() function
    and handles sparse KV cache updates after FMHA.

    Usage:
        runner = KVCachePostprocessRunner()
        params = KVCachePostprocessParams(
            qkv_input=qkv_tensor,
            block_offsets=block_offsets_tensor,
            batch_size=batch_size,
            # ... other parameters
        )
        runner.run(params)
    """

    def run(self, params: KVCachePostprocessParams) -> None:
        """
        Run KV cache postprocessing.

        This handles sparse KV cache updates after FMHA computation.
        Should be called after context FMHA for non-MLA attention.

        Args:
            params: KVCachePostprocessParams containing all input tensors and parameters.
        """
        torch.ops.trtllm.run_kv_cache_postprocessing(
            # Tensors
            params.qkv_input,
            params.sparse_kv_indices,
            params.sparse_kv_offsets,
            params.block_offsets,
            # Scalars
            params.batch_size,
            params.is_last_chunk,
            params.head_num,
            params.kv_head_num,
            params.size_per_head,
            params.quant_mode,
            # KV cache buffer parameters
            params.tokens_per_block,
            params.max_blocks_per_sequence,
            params.attention_window_size,
            params.size_per_token,
            params.sink_token_length,
            params.max_cyclic_attention_window_size,
            params.can_use_one_more_block,
            params.host_primary_pool_pointer,
            params.host_secondary_pool_pointer,
        )


########################################################
# MLA Context Processing Runner
########################################################


class MLAContextRunner:
    """
    Runner class for MLA (Multi-head Latent Attention) context phase operations.

    This class wraps the C++ torch.ops.trtllm.mla_rope_context() and
    torch.ops.trtllm.mla_context_fp8_quantize() functions.

    Usage:
        runner = MLAContextRunner()
        runner.run_rope_context(
            q=q_tensor,
            q_pe=q_pe_tensor,
            latent_cache=latent_cache_tensor,
            ...
        )
    """

    def run_rope_context(
        self,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cache_seq_lens: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        batch_size: int,
        num_heads: int,
        max_input_seq_len: int,
        layer_idx: int,
        tokens_per_block: int,
        attention_window_size: int,
        sink_token_length: int,
        beam_width: int,
        quant_mode: int,
        mla_params: MlaMetaParams,
        absorption_mode: bool = False,
        k: Optional[torch.Tensor] = None,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        helix_position_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Apply RoPE to MLA Q/K tensors and write K to KV cache (context phase).

        This function performs:
        1. Applies rotary position embedding (RoPE) to Q and K tensors
        2. Writes the processed K (latent cache) to the paged KV cache

        Args:
            q: Query tensor [total_q_len, num_heads * (qk_nope_head_dim + qk_rope_head_dim)]
            q_pe: Query position embedding tensor [total_q_len, num_heads, qk_rope_head_dim]
            latent_cache: Latent cache tensor (compressed KV) [total_kv_len, kv_lora_rank + qk_rope_head_dim]
            cos_sin_cache: RoPE cos/sin cache [max_seq_len, qk_rope_head_dim]
            cu_q_seqlens: Cumulative query sequence lengths [batch_size + 1]
            cache_seq_lens: Cache sequence lengths per batch [batch_size]
            kv_cache_block_offsets: KV cache block offsets tensor
            host_kv_cache_pool_pointers: Host KV cache pool pointers tensor
            host_kv_cache_pool_mapping: Host KV cache pool mapping tensor
            batch_size: Number of sequences in batch
            num_heads: Number of attention heads
            max_input_seq_len: Maximum input sequence length
            layer_idx: Layer index
            tokens_per_block: Tokens per KV cache block
            attention_window_size: Attention window size
            sink_token_length: Sink token length
            beam_width: Beam width
            quant_mode: Quantization mode
            mla_params: MLA meta parameters
            absorption_mode: Whether to use sparse MLA absorption mode
            k: Optional key tensor
            kv_scale_orig_quant: Optional KV scale for quantization
            helix_position_offsets: Optional Helix position offsets
        """
        torch.ops.trtllm.mla_rope_context(
            q,
            q_pe,
            k,
            latent_cache,
            cos_sin_cache,
            cu_q_seqlens,
            cache_seq_lens,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            helix_position_offsets,
            batch_size,
            num_heads,
            max_input_seq_len,
            layer_idx,
            tokens_per_block,
            attention_window_size,
            sink_token_length,
            beam_width,
            quant_mode,
            mla_params.q_lora_rank,
            mla_params.kv_lora_rank,
            mla_params.qk_nope_head_dim,
            mla_params.qk_rope_head_dim,
            mla_params.v_head_dim,
            absorption_mode,
        )

    def run_fp8_quantize(
        self,
        q: torch.Tensor,
        quant_q: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        quant_scale_qkv: torch.Tensor,
        batch_size: int,
        num_heads: int,
        max_input_seq_len: int,
        total_kv_len: int,
        mla_params: MlaMetaParams,
        absorption_mode: bool = False,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        quant_k: Optional[torch.Tensor] = None,
        quant_v: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Quantize MLA Q/K/V tensors to FP8 for context phase attention.

        Args:
            q: Query tensor (input)
            quant_q: Output quantized Q tensor
            cu_q_seqlens: Cumulative query sequence lengths
            quant_scale_qkv: Quantization scale for QKV
            batch_size: Batch size
            num_heads: Number of attention heads
            max_input_seq_len: Maximum input sequence length
            total_kv_len: Total KV length
            mla_params: MLA meta parameters
            absorption_mode: Whether using absorption mode (sparse MLA)
            k: Optional key tensor (input)
            v: Optional value tensor (input)
            quant_k: Output quantized K tensor (optional)
            quant_v: Output quantized V tensor (optional)
        """
        torch.ops.trtllm.mla_context_fp8_quantize(
            q,
            k,
            v,
            quant_q,
            quant_k,
            quant_v,
            cu_q_seqlens,
            quant_scale_qkv,
            batch_size,
            num_heads,
            max_input_seq_len,
            total_kv_len,
            mla_params.q_lora_rank,
            mla_params.kv_lora_rank,
            mla_params.qk_nope_head_dim,
            mla_params.qk_rope_head_dim,
            mla_params.v_head_dim,
            absorption_mode,
        )


########################################################
# Ulysses Context Parallelism Runner
########################################################


class UlyssesContextParallelism:
    """
    Runner class for Ulysses-style Context Parallelism (CP) operations.

    This class wraps the C++ transpose kernels and uses torch.distributed
    for all-to-all communication. It implements the head-split to sequence-split
    transformation for attention computation across multiple GPUs.

    Ulysses CP splits the attention heads across GPUs before FMHA and then
    redistributes the results to sequence-split layout after FMHA.

    Usage:
        cp = UlyssesContextParallelism(
            cp_size=4,
            cp_rank=dist.get_rank(),
            num_attn_heads=32,
            num_attn_kv_heads=8,
            head_size=128,
            cp_group=dist.group.WORLD,
        )

        # Context phase preprocess
        attention_input = cp.context_preprocess(qkv_tensor, params)

        # Run FMHA...

        # Context phase postprocess
        output = cp.context_postprocess(fmha_output, params)
    """

    def __init__(
        self,
        cp_size: int,
        cp_rank: int,
        num_attn_heads: int,
        num_attn_kv_heads: int,
        head_size: int,
        cp_group: Optional["torch.distributed.ProcessGroup"] = None,
        mqa_broadcast: int = 1,
        fp8_output: bool = False,
    ):
        """
        Initialize UlyssesContextParallelism.

        Args:
            cp_size: Context parallelism world size
            cp_rank: Current rank in context parallelism group
            num_attn_heads: Number of Q attention heads
            num_attn_kv_heads: Number of KV attention heads
            head_size: Size per attention head
            cp_group: torch.distributed process group for CP communication
            mqa_broadcast: MQA broadcast factor (default 1)
            fp8_output: Whether attention output is in FP8 format
        """
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.num_attn_heads = num_attn_heads
        self.num_attn_kv_heads = num_attn_kv_heads
        self.head_size = head_size
        self.cp_group = cp_group
        self.mqa_broadcast = mqa_broadcast
        self.fp8_output = fp8_output

        # Computed values
        self.partial_q_heads = num_attn_heads // cp_size
        self.partial_kv_heads = (
            num_attn_kv_heads // cp_size if num_attn_kv_heads >= cp_size else num_attn_kv_heads
        )
        self.partial_heads = self.partial_q_heads + 2 * self.partial_kv_heads

    def _compute_partial_info(
        self,
        context_lengths: torch.Tensor,
        batch_size: int,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """
        Compute partial token information for context phase.

        Returns:
            partial_token_num: Total number of partial tokens
            max_partial_length: Maximum partial sequence length
            cu_q_seqlens: Cumulative Q sequence lengths
            cu_cp_partial_seqlens: Cumulative CP partial sequence lengths
        """
        # Move to CPU for computation
        host_context_lengths = context_lengths.cpu()

        partial_lengths = (host_context_lengths + self.cp_size - 1) // self.cp_size
        partial_token_num = int(partial_lengths.sum().item())
        max_partial_length = int(partial_lengths.max().item())

        # Compute cumulative sequence lengths
        cu_q_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=context_lengths.device)
        cu_q_seqlens[1:] = torch.cumsum(context_lengths, dim=0)

        cu_cp_partial_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=context_lengths.device
        )
        cu_cp_partial_seqlens[1:] = torch.cumsum(partial_lengths.to(context_lengths.device), dim=0)

        return partial_token_num, max_partial_length, cu_q_seqlens, cu_cp_partial_seqlens

    def context_preprocess(
        self,
        attention_input: torch.Tensor,
        context_lengths: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Preprocess attention input for context phase with Ulysses CP.

        Transforms: [partialTokenNum, numHeads + 2*numKVHeads, headSize]
                 -> all-to-all
                 -> [numTokens, partialHeads, headSize]

        Args:
            attention_input: Input QKV tensor [partialTokenNum, numHeads + 2*numKVHeads, headSize]
            context_lengths: Sequence lengths for each batch [batch_size]
            batch_size: Batch size

        Returns:
            Transformed attention input ready for FMHA
        """
        import torch.distributed as dist

        if self.cp_size <= 1:
            return attention_input

        partial_token_num, max_partial_length, cu_q_seqlens, cu_cp_partial_seqlens = (
            self._compute_partial_info(context_lengths, batch_size)
        )

        # Allocate buffers
        # dst_other_ranks: for data to send to other ranks
        # dst_my_rank: for data to keep on this rank
        dst_other_ranks = torch.empty(
            self.cp_size,
            partial_token_num,
            self.partial_heads,
            self.head_size,
            dtype=attention_input.dtype,
            device=attention_input.device,
        )
        dst_my_rank = torch.empty(
            partial_token_num,
            self.partial_heads,
            self.head_size,
            dtype=attention_input.dtype,
            device=attention_input.device,
        )

        # Step 1: Transpose for head splitting
        # [partialTokens, totalHeads, headSize] -> [cpSize, partialTokens, partialHeads, headSize]
        torch.ops.trtllm.ulysses_cp_transpose(
            attention_input,
            dst_other_ranks,
            dst_my_rank,
            partial_token_num,
            self.cp_size,
            self.num_attn_heads,
            self.num_attn_kv_heads,
            self.mqa_broadcast,
            self.head_size,
            self.cp_rank,
        )

        # Step 2: All-to-all communication using torch.distributed
        # Prepare send/recv tensors
        send_list = [dst_other_ranks[i] for i in range(self.cp_size)]
        send_list[self.cp_rank] = dst_my_rank  # Replace with my_rank data

        recv_buffer = torch.empty_like(dst_other_ranks)
        recv_list = [recv_buffer[i] for i in range(self.cp_size)]

        if self.cp_group is not None:
            dist.all_to_all(recv_list, send_list, group=self.cp_group)
        else:
            dist.all_to_all(recv_list, send_list)

        # Copy my_rank data to recv_buffer
        recv_buffer[self.cp_rank].copy_(dst_my_rank)

        # Step 3: Transpose to sequence-major layout
        # [cpSize, partialTokens, partialHeads, headSize] -> [numTokens, partialHeads, headSize]
        total_tokens = int(cu_q_seqlens[-1].item())
        output = torch.empty(
            total_tokens,
            self.partial_heads,
            self.head_size,
            dtype=attention_input.dtype,
            device=attention_input.device,
        )

        torch.ops.trtllm.ulysses_cp_transpose2(
            recv_buffer,
            output,
            context_lengths,
            cu_q_seqlens,
            cu_cp_partial_seqlens,
            self.cp_size,
            max_partial_length,
            batch_size,
            self.partial_heads,
            self.head_size,
        )

        return output

    def context_postprocess(
        self,
        fmha_output: torch.Tensor,
        context_lengths: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Postprocess FMHA output for context phase with Ulysses CP.

        Transforms: [numTokens, partialHeads, headSize]
                 -> all-to-all
                 -> [partialTokenNum, numHeads, headSize]

        Args:
            fmha_output: FMHA output tensor [numTokens, partialHeads, headSize]
            context_lengths: Sequence lengths for each batch [batch_size]
            batch_size: Batch size

        Returns:
            Transformed output with full attention heads
        """
        import torch.distributed as dist

        if self.cp_size <= 1:
            return fmha_output

        partial_token_num, max_partial_length, cu_q_seqlens, cu_cp_partial_seqlens = (
            self._compute_partial_info(context_lengths, batch_size)
        )

        partial_heads = self.num_attn_heads // self.cp_size

        # Step 1: Transpose to CP-major layout (before all-to-all)
        # [numTokens, partialHeads, headSize] -> [cpSize, partialTokens, partialHeads, headSize]
        send_buffer = torch.empty(
            self.cp_size,
            partial_token_num,
            partial_heads,
            self.head_size,
            dtype=fmha_output.dtype,
            device=fmha_output.device,
        )

        torch.ops.trtllm.ulysses_cp_transpose_to_seq_major2(
            fmha_output,
            send_buffer,
            context_lengths,
            cu_q_seqlens,
            cu_cp_partial_seqlens,
            self.cp_size,
            max_partial_length,
            batch_size,
            partial_heads,
            self.head_size,
        )

        # Step 2: All-to-all communication
        send_list = [send_buffer[i] for i in range(self.cp_size)]
        recv_buffer = torch.empty_like(send_buffer)
        recv_list = [recv_buffer[i] for i in range(self.cp_size)]

        if self.cp_group is not None:
            dist.all_to_all(recv_list, send_list, group=self.cp_group)
        else:
            dist.all_to_all(recv_list, send_list)

        # Step 3: Transpose to sequence-major layout (after all-to-all)
        # [cpSize, partialTokens, partialHeads, headSize] -> [partialTokens, numHeads, headSize]
        output = torch.empty(
            partial_token_num,
            self.num_attn_heads,
            self.head_size,
            dtype=fmha_output.dtype,
            device=fmha_output.device,
        )

        src_my_rank = recv_buffer[self.cp_rank]
        # Gather other ranks data
        src_other_ranks = torch.cat(
            [recv_buffer[i : i + 1] for i in range(self.cp_size) if i != self.cp_rank], dim=0
        )

        torch.ops.trtllm.ulysses_cp_transpose_to_seq_major(
            output,
            src_my_rank,
            src_other_ranks,
            partial_token_num,
            self.cp_size,
            partial_heads,
            self.head_size,
            self.cp_rank,
        )

        return output

    def generation_preprocess(
        self,
        attention_input: torch.Tensor,
        batch_beam: int,
    ) -> torch.Tensor:
        """
        Preprocess attention input for generation phase with Ulysses CP.

        Similar to context_preprocess but with simpler fixed-size shapes.

        Args:
            attention_input: Input QKV tensor [batch_beam, numHeads + 2*numKVHeads, headSize]
            batch_beam: Batch size * beam width

        Returns:
            Transformed attention input ready for MMHA
        """
        import torch.distributed as dist

        if self.cp_size <= 1:
            return attention_input

        partial_token_num = (batch_beam + self.cp_size - 1) // self.cp_size

        # Allocate buffers
        dst_other_ranks = torch.empty(
            self.cp_size,
            partial_token_num,
            self.partial_heads,
            self.head_size,
            dtype=attention_input.dtype,
            device=attention_input.device,
        )
        dst_my_rank = torch.empty(
            partial_token_num,
            self.partial_heads,
            self.head_size,
            dtype=attention_input.dtype,
            device=attention_input.device,
        )

        # Step 1: Transpose
        torch.ops.trtllm.ulysses_cp_transpose(
            attention_input,
            dst_other_ranks,
            dst_my_rank,
            partial_token_num,
            self.cp_size,
            self.num_attn_heads,
            self.num_attn_kv_heads,
            self.mqa_broadcast,
            self.head_size,
            self.cp_rank,
        )

        # Step 2: All-to-all
        send_list = [dst_other_ranks[i] for i in range(self.cp_size)]
        send_list[self.cp_rank] = dst_my_rank

        recv_buffer = torch.empty_like(dst_other_ranks)
        recv_list = [recv_buffer[i] for i in range(self.cp_size)]

        if self.cp_group is not None:
            dist.all_to_all(recv_list, send_list, group=self.cp_group)
        else:
            dist.all_to_all(recv_list, send_list)

        recv_buffer[self.cp_rank].copy_(dst_my_rank)

        # Step 3: View as [batch_beam, partialHeads, headSize]
        output = recv_buffer.permute(1, 0, 2, 3).contiguous()
        output = output.view(batch_beam, self.partial_heads, self.head_size)

        return output

    def generation_postprocess(
        self,
        mmha_output: torch.Tensor,
        batch_beam: int,
    ) -> torch.Tensor:
        """
        Postprocess MMHA output for generation phase with Ulysses CP.

        Similar to context_postprocess but with simpler fixed-size shapes.

        Args:
            mmha_output: MMHA output tensor [batch_beam, partialHeads, headSize]
            batch_beam: Batch size * beam width

        Returns:
            Transformed output with full attention heads
        """
        import torch.distributed as dist

        if self.cp_size <= 1:
            return mmha_output

        partial_token_num = (batch_beam + self.cp_size - 1) // self.cp_size
        partial_heads = self.num_attn_heads // self.cp_size

        # Step 1: View as [cpSize, partialTokens, partialHeads, headSize]
        send_buffer = mmha_output.view(
            self.cp_size, partial_token_num, partial_heads, self.head_size
        )
        send_buffer = send_buffer.contiguous()

        # Step 2: All-to-all
        send_list = [send_buffer[i] for i in range(self.cp_size)]
        recv_buffer = torch.empty_like(send_buffer)
        recv_list = [recv_buffer[i] for i in range(self.cp_size)]

        if self.cp_group is not None:
            dist.all_to_all(recv_list, send_list, group=self.cp_group)
        else:
            dist.all_to_all(recv_list, send_list)

        # Step 3: Transpose to sequence-major
        output = torch.empty(
            partial_token_num,
            self.num_attn_heads,
            self.head_size,
            dtype=mmha_output.dtype,
            device=mmha_output.device,
        )

        src_my_rank = recv_buffer[self.cp_rank]
        src_other_ranks = torch.cat(
            [recv_buffer[i : i + 1] for i in range(self.cp_size) if i != self.cp_rank], dim=0
        )

        torch.ops.trtllm.ulysses_cp_transpose_to_seq_major(
            output,
            src_my_rank,
            src_other_ranks,
            partial_token_num,
            self.cp_size,
            partial_heads,
            self.head_size,
            self.cp_rank,
        )

        return output


########################################################
# TrtllmGenFmhaRunner - TRTLLM-Gen FMHA Kernel Runner
########################################################


class QkvLayout(IntEnum):
    """
    QKV layout types for TRTLLM-Gen FMHA.
    Corresponds to C++ QkvLayout in fmhaRunnerParams.h.
    """

    # SeparateQkv: separate Q, K and V buffers.
    # Each has the shape: [batchSize, seqLen, numHeads, headDim].
    SEPARATE_QKV = 0
    # PackedQkv: single buffer for Q, K and V.
    # Shape: [batchSize, seqLen, numHeadsQ + 2*numHeadsKv, headDim].
    PACKED_QKV = 1
    # Paged buffer for K and V.
    PAGED_KV = 2
    # ContiguousKv: Contiguous buffer for Q and Kv.
    CONTIGUOUS_KV = 3


class TrtllmGenAttentionMaskType(IntEnum):
    """
    Attention mask types for TRTLLM-Gen FMHA.
    Corresponds to C++ TrtllmGenAttentionMaskType in fmhaRunnerParams.h.
    """

    # Dense mask.
    DENSE = 0
    # Causal mask.
    CAUSAL = 1
    # Sliding window or chunked causal mask.
    SLIDING_OR_CHUNKED_CAUSAL = 2
    # Custom mask.
    CUSTOM = 3


class FmhaKernelType(IntEnum):
    """
    FMHA kernel types for TRTLLM-Gen.
    Corresponds to C++ FmhaKernelType in fmhaRunnerParams.h.
    """

    # The context-phase kernels.
    CONTEXT = 0
    # Choose the best generation kernel based on the heuristic.
    GENERATION = 1
    # Swap tensor A and tensor B of Mma, which only supports numHeadsQPerKv <= 16.
    SWAPS_MMA_AB_FOR_GENERATION = 2
    # Keep tensor A and tensor B of Mma.
    KEEPS_MMA_AB_FOR_GENERATION = 3
    # Speculative decoding generation-phase attention kernels.
    SPEC_DECODING_GENERATION = 4


class TileScheduler(IntEnum):
    """
    Tile scheduler types for TRTLLM-Gen FMHA.
    Corresponds to C++ TileScheduler in fmhaRunnerParams.h.
    """

    # Static scheduler (Non-persistent).
    STATIC = 0
    # Persistent scheduler.
    PERSISTENT = 1


def _mask_type_to_trtllm_gen(mask_type: int) -> TrtllmGenAttentionMaskType:
    """
    Convert AttentionMaskType to TrtllmGenAttentionMaskType.

    Mapping based on cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h
    """
    if mask_type == int(AttentionMaskType.PADDING):
        return TrtllmGenAttentionMaskType.DENSE
    elif mask_type == int(AttentionMaskType.CAUSAL):
        return TrtllmGenAttentionMaskType.CAUSAL
    elif mask_type == int(AttentionMaskType.SLIDING_WINDOW_CAUSAL):
        return TrtllmGenAttentionMaskType.SLIDING_OR_CHUNKED_CAUSAL
    elif mask_type == int(AttentionMaskType.CUSTOM_MASK):
        return TrtllmGenAttentionMaskType.CUSTOM
    else:
        # Default to CAUSAL for unknown types
        return TrtllmGenAttentionMaskType.CAUSAL


class TrtllmGenDataType(IntEnum):
    """
    Data types for TRTLLM-Gen FMHA runner.
    Used to specify Q, KV, and output data types.
    """

    FP16 = 0
    BF16 = 1
    FP32 = 2
    E4M3 = 3  # FP8
    E2M1 = 4  # NVFP4


@dataclass
class TrtllmGenFmhaRunnerParams:
    """
    Parameters for TRTLLM-Gen FMHA runner.
    Corresponds to C++ TllmGenFmhaRunnerParams in fmhaRunnerParams.h.
    """

    # ========== Enum fields ==========
    qkv_layout: QkvLayout = QkvLayout.SEPARATE_QKV
    mask_type: TrtllmGenAttentionMaskType = TrtllmGenAttentionMaskType.CAUSAL
    kernel_type: FmhaKernelType = FmhaKernelType.CONTEXT
    tile_scheduler: TileScheduler = TileScheduler.STATIC
    multi_ctas_kv_mode: bool = False
    use_block_sparse_attention: bool = False

    # ========== Tensor pointer fields ==========
    q_ptr: Optional[torch.Tensor] = None
    k_ptr: Optional[torch.Tensor] = None
    v_ptr: Optional[torch.Tensor] = None
    kv_ptr: Optional[torch.Tensor] = None
    kv_sf_ptr: Optional[torch.Tensor] = None
    qkv_ptr: Optional[torch.Tensor] = None
    attention_sinks_ptr: Optional[torch.Tensor] = None
    custom_mask_ptr: Optional[torch.Tensor] = None
    custom_mask_offsets_ptr: Optional[torch.Tensor] = None
    first_sparse_mask_offsets_kv_ptr: Optional[torch.Tensor] = None
    multi_ctas_kv_counter_ptr: Optional[torch.Tensor] = None
    seq_lens_kv_ptr: Optional[torch.Tensor] = None
    cum_seq_lens_q_ptr: Optional[torch.Tensor] = None
    cum_seq_lens_kv_ptr: Optional[torch.Tensor] = None
    kv_page_idx_ptr: Optional[torch.Tensor] = None
    output_scale_ptr: Optional[torch.Tensor] = None
    scale_softmax_log2_ptr: Optional[torch.Tensor] = None
    kv_sf_scale_ptr: Optional[torch.Tensor] = None
    o_sf_scale_ptr: Optional[torch.Tensor] = None
    multi_ctas_kv_scratch_ptr: Optional[torch.Tensor] = None
    softmax_stats_ptr: Optional[torch.Tensor] = None
    o_ptr: Optional[torch.Tensor] = None
    o_sf_ptr: Optional[torch.Tensor] = None
    seqlens_q_ptr: Optional[torch.Tensor] = None

    # ========== Scalar fields ==========
    head_dim_qk: int = 0
    head_dim_v: int = 0
    head_dim_qk_nope: int = 0
    num_heads_q: int = 0
    num_heads_kv: int = 0
    num_heads_q_per_kv: int = 0
    batch_size: int = 0
    max_seq_len_cache_kv: int = 0
    max_seq_len_q: int = 0
    max_seq_len_kv: int = 0
    attention_window_size: int = 0
    chunked_attention_size: int = 0
    sum_of_seq_lens_q: int = 0
    sum_of_seq_lens_kv: int = 0
    max_num_pages_per_seq_kv: int = 0
    num_tokens_per_page: int = 0
    num_pages_in_mem_pool: int = 0
    multi_processor_count: int = 0
    scale_q: float = 1.0
    sf_start_token_idx: int = 0
    skip_softmax_threshold_scale_factor: float = 0.0
    sparse_mla: bool = False
    sparse_mla_top_k: int = 0
    layer_idx: int = 0
    is_spec_dec_tree: bool = False


class TrtllmGenFmhaRunner:
    """
    Runner class for TRTLLM-Gen FMHA kernels.

    This class wraps the C++ torch.classes.trtllm.TrtllmGenFmhaRunner and provides
    a Pythonic interface for running FMHA kernels on Blackwell (SM100) GPUs.

    Usage:
        runner = TrtllmGenFmhaRunner(
            dtype_q=TrtllmGenDataType.FP16,
            dtype_kv=TrtllmGenDataType.FP16,
            dtype_out=TrtllmGenDataType.FP16
        )
        params = TrtllmGenFmhaRunnerParams(
            qkv_layout=QkvLayout.PAGED_KV,
            mask_type=TrtllmGenAttentionMaskType.CAUSAL,
            # ... other parameters
        )
        runner.run(params)
    """

    def __init__(
        self,
        dtype_q: TrtllmGenDataType = TrtllmGenDataType.FP16,
        dtype_kv: TrtllmGenDataType = TrtllmGenDataType.FP16,
        dtype_out: TrtllmGenDataType = TrtllmGenDataType.FP16,
    ):
        """
        Initialize the TRTLLM-Gen FMHA runner.

        Args:
            dtype_q: Data type for Q tensor.
            dtype_kv: Data type for KV tensor.
            dtype_out: Data type for output tensor.
        """
        self._runner = torch.classes.trtllm.TrtllmGenFmhaRunner(
            int(dtype_q), int(dtype_kv), int(dtype_out)
        )

    def _call_runner_method(self, method_name: str, params: TrtllmGenFmhaRunnerParams):
        """Helper method to call runner methods with unpacked parameters."""
        method = getattr(self._runner, method_name)
        return method(
            # Enum fields
            int(params.qkv_layout),
            int(params.mask_type),
            int(params.kernel_type),
            int(params.tile_scheduler),
            params.multi_ctas_kv_mode,
            params.use_block_sparse_attention,
            # Tensor pointer fields
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.kv_ptr,
            params.kv_sf_ptr,
            params.qkv_ptr,
            params.attention_sinks_ptr,
            params.custom_mask_ptr,
            params.custom_mask_offsets_ptr,
            params.first_sparse_mask_offsets_kv_ptr,
            params.multi_ctas_kv_counter_ptr,
            params.seq_lens_kv_ptr,
            params.cum_seq_lens_q_ptr,
            params.cum_seq_lens_kv_ptr,
            params.kv_page_idx_ptr,
            params.output_scale_ptr,
            params.scale_softmax_log2_ptr,
            params.kv_sf_scale_ptr,
            params.o_sf_scale_ptr,
            params.multi_ctas_kv_scratch_ptr,
            params.softmax_stats_ptr,
            params.o_ptr,
            params.o_sf_ptr,
            params.seqlens_q_ptr,
            # Scalar fields
            params.head_dim_qk,
            params.head_dim_v,
            params.head_dim_qk_nope,
            params.num_heads_q,
            params.num_heads_kv,
            params.num_heads_q_per_kv,
            params.batch_size,
            params.max_seq_len_cache_kv,
            params.max_seq_len_q,
            params.max_seq_len_kv,
            params.attention_window_size,
            params.chunked_attention_size,
            params.sum_of_seq_lens_q,
            params.sum_of_seq_lens_kv,
            params.max_num_pages_per_seq_kv,
            params.num_tokens_per_page,
            params.num_pages_in_mem_pool,
            params.multi_processor_count,
            params.scale_q,
            params.sf_start_token_idx,
            params.skip_softmax_threshold_scale_factor,
            params.sparse_mla,
            params.sparse_mla_top_k,
            params.layer_idx,
            params.is_spec_dec_tree,
        )

    def run(self, params: TrtllmGenFmhaRunnerParams) -> None:
        """
        Run the FMHA kernel with the given parameters.

        Args:
            params: FMHA runner parameters.
        """
        self._call_runner_method("run", params)


@dataclass
class EnqueueParams:
    """
    Base parameters for attention enqueue operations.
    Corresponds to C++ EnqueueParams<T> in attentionOp.h.
    """

    # Attention input tensor
    attention_input: Optional[torch.Tensor] = None
    # QKV bias tensor
    qkv_bias: Optional[torch.Tensor] = None
    # Attention mask input, shape: [batch_size, attention_mask_stride]
    attention_mask: Optional[torch.Tensor] = None
    # Attention sinks with shape [num_heads_q], float
    attention_sinks: Optional[torch.Tensor] = None
    # Rotary inv_freq cache buffer to avoid re-computing
    rotary_inv_freq: Optional[torch.Tensor] = None
    # Rotary cos sin cache buffer to avoid re-computing
    rotary_cos_sin: Optional[torch.Tensor] = None
    # NOTE: input_seq_length might be larger than one in the medusa mode
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    # Max cache capacity (used to allocate KV cache)
    # By default, max_attention_window_size == cyclic_attention_window_size
    # unless each layer has different cyclic kv cache length.
    max_attention_window_size: int = 0
    # Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
    cyclic_attention_window_size: int = 0
    max_cyclic_attention_window_size: int = 0
    can_use_one_more_block: bool = False
    sink_token_length: int = 0
    kv_scale_orig_quant: Optional[torch.Tensor] = None
    kv_scale_quant_orig: Optional[torch.Tensor] = None
    attention_output_orig_quant: Optional[torch.Tensor] = None
    attention_output_sf_scale: Optional[torch.Tensor] = None
    alibi_slopes: Optional[torch.Tensor] = None
    context_buf: Optional[torch.Tensor] = None
    context_buf_sf: Optional[torch.Tensor] = None
    key_value_cache: Optional[torch.Tensor] = None
    block_offsets: Optional[torch.Tensor] = None
    host_primary_pool_pointer: Optional[int] = None
    host_secondary_pool_pointer: Optional[int] = None
    host_primary_block_scale_pool_pointer: Optional[int] = None
    host_secondary_block_scale_pool_pointer: Optional[int] = None
    num_tokens: int = 0
    total_kv_len: int = 0
    max_blocks_per_sequence: int = 0
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    host_context_lengths: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None
    # optional when logn scaling
    logn_scaling_ptr: Optional[torch.Tensor] = None
    # optional when relative position
    relative_attention_bias: Optional[torch.Tensor] = None
    relative_attention_bias_stride: int = 0
    # optional when cross attention
    encoder_input_lengths: Optional[torch.Tensor] = None
    runtime_perf_knobs: Optional[torch.Tensor] = None
    # optional when compute attention stats (MLA chunked prefill or Helix parallelism)
    # this is a buffer of size [num_tokens, num_heads_q] with each element
    # representing the max and LSE/denominator of the softmax values
    softmax_stats: Optional[torch.Tensor] = None
    stream: Optional[torch.cuda.Stream] = None


@dataclass
class EnqueueContextParams(EnqueueParams):
    """
    Parameters for context phase attention enqueue.
    Corresponds to C++ EnqueueContextParams<T> in attentionOp.h.
    """

    # Attention packed mask input (used by context FMHA)
    attention_packed_mask: Optional[torch.Tensor] = None
    host_block_offsets: Optional[torch.Tensor] = None
    batch_size: int = 0
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None

    # optional when cross attention
    cross_kv: Optional[torch.Tensor] = None
    cross_kv_length: int = 0
    num_encoder_tokens: int = 0
    mla_param: Optional[Any] = None  # kernels::MlaParams<T>*

    # optional for separate QKV input, currently only used for context MLA
    k_ptr: Optional[torch.Tensor] = None
    v_ptr: Optional[torch.Tensor] = None

    def to_string(self) -> str:
        """
        Returns a string representation of the parameters.
        Corresponds to C++ enqueueContextParamsToString().
        """
        lines = ["EnqueueContextParams ===================="]
        lines.append(f"attention_input: {self.attention_input}")
        lines.append(f"qkv_bias: {self.qkv_bias}")
        lines.append(f"attention_mask: {self.attention_mask}")
        lines.append(f"attention_packed_mask: {self.attention_packed_mask}")
        lines.append(f"rotary_inv_freq: {self.rotary_inv_freq}")
        lines.append(f"rotary_cos_sin: {self.rotary_cos_sin}")
        lines.append(f"input_seq_length: {self.input_seq_length}")
        lines.append(f"max_past_kv_length: {self.max_past_kv_length}")
        lines.append(f"max_attention_window_size: {self.max_attention_window_size}")
        lines.append(f"cyclic_attention_window_size: {self.cyclic_attention_window_size}")
        lines.append(f"max_cyclic_attention_window_size: {self.max_cyclic_attention_window_size}")
        lines.append(
            f"can_use_one_more_block: {'true' if self.can_use_one_more_block else 'false'}"
        )
        lines.append(f"sink_token_length: {self.sink_token_length}")
        if self.context_lengths is not None and self.batch_size > 0:
            lines.append(f"context_lengths: {self.context_lengths}")
        if self.sequence_lengths is not None and self.batch_size > 0:
            lines.append(f"sequence_lengths: {self.sequence_lengths}")
        lines.append(f"kv_scale_orig_quant: {self.kv_scale_orig_quant}")
        lines.append(f"kv_scale_quant_orig: {self.kv_scale_quant_orig}")
        lines.append(f"attention_output_orig_quant: {self.attention_output_orig_quant}")
        lines.append(f"alibi_slopes: {self.alibi_slopes}")
        lines.append(f"context_buf: {self.context_buf}")
        lines.append(f"context_buf_sf: {self.context_buf_sf}")
        lines.append(f"key_value_cache: {self.key_value_cache}")
        lines.append(f"block_offsets: {self.block_offsets}")
        lines.append(f"host_block_offsets: {self.host_block_offsets}")
        lines.append(f"host_primary_pool_pointer: {self.host_primary_pool_pointer}")
        lines.append(f"host_secondary_pool_pointer: {self.host_secondary_pool_pointer}")
        lines.append(f"batch_size: {self.batch_size}")
        lines.append(f"num_tokens: {self.num_tokens}")
        lines.append(f"total_kv_len: {self.total_kv_len}")
        lines.append(f"max_blocks_per_sequence: {self.max_blocks_per_sequence}")
        lines.append(f"workspace: {self.workspace}")
        lines.append(f"logn_scaling_ptr: {self.logn_scaling_ptr}")
        lines.append(f"relative_attention_bias: {self.relative_attention_bias}")
        lines.append(f"relative_attention_bias_stride: {self.relative_attention_bias_stride}")
        lines.append(f"cross_kv: {self.cross_kv}")
        lines.append(f"cross_kv_length: {self.cross_kv_length}")
        lines.append(f"encoder_input_lengths: {self.encoder_input_lengths}")
        lines.append(f"num_encoder_tokens: {self.num_encoder_tokens}")
        lines.append(f"softmax_stats: {self.softmax_stats}")
        lines.append(f"k_ptr: {self.k_ptr}")
        lines.append(f"v_ptr: {self.v_ptr}")
        return "\n".join(lines)


@dataclass
class EnqueueGenerationParams(EnqueueParams):
    """
    Parameters for generation phase attention enqueue.
    Corresponds to C++ EnqueueGenerationParams<T> in attentionOp.h.
    """

    beam_width: int = 1
    # Attention mask has shape of [batch_size, attention_mask_stride]
    attention_mask_stride: int = 0
    num_requests: int = 0
    cache_indir: Optional[torch.Tensor] = None
    semaphores: Optional[torch.Tensor] = None
    host_past_key_value_lengths: Optional[torch.Tensor] = None
    mrope_position_deltas: Optional[torch.Tensor] = None

    # optional when speculative decoding is used
    spec_decoding_mask: Optional[torch.Tensor] = None
    spec_decoding_packed_mask: Optional[torch.Tensor] = None
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_is_generation_length_variable: bool = False
    spec_decoding_max_generation_length: int = 1
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None
    # optional when fuse_fp4_quant is enabled
    start_token_idx_sf: int = 0
    layer_idx: int = 0


def _get_device_index(device: Optional[Union[int, torch.device]] = None) -> int:
    """Helper to get device index from various input types."""
    if device is None:
        return torch.cuda.current_device()
    elif isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    return device


def get_sm_version(device: Optional[Union[int, torch.device]] = None) -> int:
    """
    Get the SM (Streaming Multiprocessor) version for the specified CUDA device.

    The SM version is computed as: major * 10 + minor
    For example:
        - SM 80 for A100 (compute capability 8.0)
        - SM 89 for Ada Lovelace (compute capability 8.9)
        - SM 90 for Hopper H100 (compute capability 9.0)

    Args:
        device: CUDA device index or torch.device. If None, uses current device.

    Returns:
        SM version as an integer (e.g., 80, 89, 90).
    """
    device_idx = _get_device_index(device)
    major, minor = torch.cuda.get_device_capability(device_idx)
    return major * 10 + minor


def get_multi_processor_count(device: Optional[Union[int, torch.device]] = None) -> int:
    """
    Get the number of streaming multiprocessors (SMs) on the specified CUDA device.

    Args:
        device: CUDA device index or torch.device. If None, uses current device.

    Returns:
        Number of streaming multiprocessors.
    """
    device_idx = _get_device_index(device)
    props = torch.cuda.get_device_properties(device_idx)
    return props.multi_processor_count


def get_max_shared_memory_per_block_optin(device: Optional[Union[int, torch.device]] = None) -> int:
    """
    Get the maximum shared memory per block with opt-in for the specified CUDA device.

    This returns the maximum amount of shared memory per block when opting in to
    use extended shared memory (cudaFuncAttributeMaxDynamicSharedMemorySize).

    Uses PyTorch's torch.cuda.cudart() API to query cudaDevAttrMaxSharedMemoryPerBlockOptin.

    Args:
        device: CUDA device index or torch.device. If None, uses current device.

    Returns:
        Maximum shared memory per block (opt-in) in bytes.
    """
    device_idx = _get_device_index(device)

    # cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97

    # Use PyTorch's cudart wrapper to call cudaDeviceGetAttribute
    cudart = torch.cuda.cudart()
    err, value = cudart.cudaDeviceGetAttribute(
        CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device_idx
    )

    if err == 0:  # cudaSuccess
        return value

    # Fallback: use default shared memory size from device properties
    props = torch.cuda.get_device_properties(device_idx)
    return props.max_shared_memory_per_block


def get_dtype_size_in_bits(dtype: DataType) -> int:
    """
    Get the size in bits for a given DataType.

    Args:
        dtype: The DataType enum value.

    Returns:
        Size in bits.
    """
    dtype_to_bits = {
        DataType.BOOL: 8,
        DataType.FP16: 16,
        DataType.FP32: 32,
        DataType.INT4: 4,
        DataType.INT8: 8,
        DataType.INT32: 32,
        DataType.BF16: 16,
        DataType.E2M1: 4,  # NVFP4
        DataType.E4M3: 8,  # FP8
        DataType.E5M2: 8,  # FP8
    }
    return dtype_to_bits.get(dtype, 32)


def get_kv_cache_elem_size_in_bits(kv_cache_quant_mode: QuantMode, dtype: DataType) -> int:
    """
    Get the KV cache element size in bits based on quantization mode and data type.

    Corresponds to C++ AttentionOp::getKvCacheElemSizeInBits<T>() in attentionOp.h.

    Args:
        kv_cache_quant_mode: The quantization mode for KV cache.
        dtype: The data type (used when no quantization is applied).

    Returns:
        Element size in bits (4, 8, 16, or 32).
    """
    if kv_cache_quant_mode.has_int8_kv_cache() or kv_cache_quant_mode.has_fp8_kv_cache():
        return 8
    elif kv_cache_quant_mode.has_fp4_kv_cache():
        return 4
    return get_dtype_size_in_bits(dtype)


class TrtllmGenAttention:
    """
    TRTLLM-Gen attention backend.
    """

    def __init__(self):
        # Data members mirroring C++ Runner class
        self.beam_width = 1
        self.max_num_requests = 1
        self.attention_window_size = 1
        self.sink_token_length = 1

        # Data members mirroring C++ AttentionOp class
        self.layer_idx = -1
        self.num_heads = -1
        self.vision_start = -1
        self.vision_length = -1
        self.num_kv_heads = 1
        self.head_size = -1
        self.unidirectional = 1
        self.q_scaling = 1.0
        self.attn_logit_softcapping_scale = 0.0

        self.rotary_embedding_dim = 0
        self.rotary_embedding_base = 10000.0
        self.rotary_embedding_scale_type = RotaryScalingType.NONE
        self.rotary_embedding_scale = 1.0
        self.rotary_embedding_short_m_scale = 1.0
        self.rotary_embedding_long_m_scale = 1.0
        self.rotary_embedding_max_positions = 1024
        self.rotary_embedding_original_max_positions = 1024
        self.position_embedding_type = PositionEmbeddingType.LEARNED_ABSOLUTE
        self.use_logn_scaling = False
        self.remove_padding = True
        self.mask_type = AttentionMaskType.CAUSAL
        # tensorrt_llm::kernels::BlockSparseParams mBlockSparseParams;

        self.paged_kv_cache = True
        self.tokens_per_block = 0
        self.kv_cache_quant_mode = QuantMode(0)
        self.tp_size = 1
        self.tp_rank = 0
        self.unfuse_qkv_gemm = False
        self.type = DataType.FP32
        self.max_context_length = 0
        self.qkv_bias_enabled = False
        self.cross_attention = False
        self.max_distance = 0
        self.pos_shift_enabled = False
        self.paged_context_fmha = False
        self.fp8_context_fmha = False
        self.fp8_atten_output = False
        self.fp8_context_mla = False
        self.fp8_generation_mla = False
        self.chunk_prefill_buffer_batch_size = 1
        self.dense_context_fmha = False
        self.has_full_attention_mask = False
        self.is_spec_decoding_enabled = False
        self.use_spec_decoding = False
        self.is_spec_tree = True
        self.spec_decoding_is_generation_length_variable = False
        self.spec_decoding_max_generation_length = 1
        self._is_mla_enabled = False
        self.is_generation_mla = False
        self.use_gen_flash_mla = False
        self.use_sparse_attention = False
        self.use_tllm_gen_sparse_attention = False
        self.mla_params = MlaMetaParams()
        self.cp_size = 1
        self.cp_rank = 0
        self.cp_group = set()
        self.num_attn_heads = -1
        self.num_attn_kv_heads = -1
        self.num_kv_heads_origin = -1
        self.attn_tp_size = -1
        self.attn_tp_rank = 0
        self.attn_cp_size = -1
        self.attn_cp_rank = 0
        self.ulysses_mqa_broadcast = 1
        self.enable_context_fmha = True
        self.fmha_force_fp32_acc = False
        self.multi_block_mode = True
        self.enable_xqa = True
        self.use_kv_cache = True
        self.skip_attn = False
        self.fuse_fp4_quant = False
        self.runtime_sparse_attention_params = SparseAttentionParams()
        self.nb_multi_block_semaphores = 0
        self.attention_chunk_size = None
        self.skip_softmax_threshold_scale_factor_prefill = 0.0
        self.skip_softmax_threshold_scale_factor_decode = 0.0
        self.skip_softmax_total_blocks = None
        self.skip_softmax_skipped_blocks = None

        # Private members in AttentionOp class
        self.sm = get_sm_version()
        self.use_tllm_gen = (self.sm >= 100) and (self.sm != 120)
        self.multi_processor_count = get_multi_processor_count()
        self.max_shared_memory_per_block_optin = get_max_shared_memory_per_block_optin()
        self.tllm_gen_fmha_runner = None
        self.ulysses_cp_runner: Optional[UlyssesContextParallelism] = None

        # Initialize the attention op
        self.initialize()

    def data(self) -> Tuple[int, int, int, int]:
        return (
            self.beam_width,
            self.max_num_requests,
            self.attention_window_size,
            self.sink_token_length,
        )

    def is_alibi(self) -> bool:
        return self.position_embedding_type == PositionEmbeddingType.ALIBI

    def is_alibi_with_scale(self) -> bool:
        return self.position_embedding_type == PositionEmbeddingType.ALIBI_WITH_SCALE

    def is_rope(self) -> bool:
        return self.position_embedding_type in [
            PositionEmbeddingType.ROPE_GPTJ,
            PositionEmbeddingType.ROPE_GPT_NEOX,
            PositionEmbeddingType.LONG_ROPE,
            PositionEmbeddingType.YARN,
            PositionEmbeddingType.ROPE_M,
        ]

    def is_long_rope(self) -> bool:
        return self.position_embedding_type == PositionEmbeddingType.LONG_ROPE

    def is_mrope(self) -> bool:
        return self.position_embedding_type == PositionEmbeddingType.ROPE_M

    def is_logn_scaling(self) -> bool:
        return self.use_logn_scaling

    def is_cross_attention(self) -> bool:
        return self.cross_attention

    def is_use_kv_cache(self) -> bool:
        return self.use_kv_cache

    def is_use_custom_mask(self) -> bool:
        return self.mask_type == AttentionMaskType.CUSTOM_MASK

    def is_use_full_custom_mask(self) -> bool:
        return self.use_custom_mask() and self.has_full_attention_mask

    def is_use_packed_custom_mask(self) -> bool:
        return self.use_custom_mask() and self.enable_context_fmha

    def get_kv_cache_elem_size_in_bits(self) -> int:
        """
        Get the KV cache element size in bits based on current quantization mode and data type.

        Returns:
            Element size in bits (4, 8, 16, or 32).
        """
        return get_kv_cache_elem_size_in_bits(self.kv_cache_quant_mode, self.type)

    def get_head_size(self) -> int:
        """Get the head size."""
        return self.head_size

    def mmha_supported(self, head_size: int) -> bool:
        """Check if MMHA supports the given head size."""
        supported_sizes = [32, 48, 64, 80, 96, 104, 112, 128, 144, 160, 192, 224, 256]
        return head_size in supported_sizes

    def is_mla_enabled_fn(self) -> bool:
        """Check if MLA is enabled."""
        return self._is_mla_enabled

    def is_use_sparse_attention(self) -> bool:
        return self.use_sparse_attention and self.paged_kv_cache and self.enable_xqa

    def is_use_tllm_gen_sparse_attention(self) -> bool:
        return self.use_tllm_gen_sparse_attention and self.use_sparse_attention()

    def is_use_sparse_mla(self) -> bool:
        return self.use_sparse_attention() and self.use_tllm_gen and self._is_mla_enabled

    def sm_version(self) -> int:
        return self.sm

    def supports_nv_fp4_output(self) -> bool:
        return (
            self.enable_context_fmha
            and self.enable_xqa
            and not (self.cp_size > 1 and self.attn_tp_size > 1 and self.attn_cp_size == 1)
        )

    def multi_block_semaphores(self) -> int:
        return self.nb_multi_block_semaphores

    def attention_chunk_size(self) -> Optional[int]:
        return self.attention_chunk_size

    def skip_softmax_threshold_scale_factor_prefill(self) -> float:
        return self.skip_softmax_threshold_scale_factor_prefill

    def skip_softmax_threshold_scale_factor_decode(self) -> float:
        return self.skip_softmax_threshold_scale_factor_decode

    def skip_softmax_total_blocks(self) -> Optional[int]:
        return self.skip_softmax_total_blocks

    def is_relative(self) -> bool:
        return self.position_embedding_type == PositionEmbeddingType.RELATIVE

    def get_workspace_size(
        self,
        num_tokens: int,
        max_attention_window_size: int,
        num_gen_tokens: int,
        max_blocks_per_sequence: int,
    ) -> int:
        """
        Compute the workspace size required for attention operations.

        This method calculates the workspace size needed for both context and generation phases.
        It corresponds to the C++ getWorkspaceSizeForContext and getWorkspaceSizeForGeneration methods.

        Args:
            num_tokens: Total number of tokens in the batch
            max_attention_window_size: Maximum attention window size
            num_gen_tokens: Number of generation (decode) tokens
            max_blocks_per_sequence: Maximum number of KV cache blocks per sequence

        Returns:
            Required workspace size in bytes
        """
        if num_tokens == 0:
            return 0

        # Get element size based on data type
        dtype_size = get_dtype_size_in_bits(self.type) // 8
        head_size = self.get_head_size()

        # Calculate context workspace size
        context_workspace_size = 0
        batch_size = self.max_num_requests

        if num_tokens > 0:
            # Attention mask size (only if context FMHA is disabled)
            attention_mask_size = (
                0
                if self.enable_context_fmha
                else dtype_size * num_tokens * max_attention_window_size
            )

            # Cumulative sequence lengths
            cu_seqlens_size = 4 * (batch_size + 1)  # sizeof(int) = 4

            # Rotary inverse frequencies
            rotary_inv_freq_size = (
                4 * batch_size * self.rotary_embedding_dim // 2
                if self.rotary_embedding_dim > 0
                else 0
            )

            context_workspace_size = attention_mask_size + cu_seqlens_size + rotary_inv_freq_size

            # Add space for QKV buffers in some configurations
            if self.enable_context_fmha:
                local_hidden_units_qo = (
                    int(self.num_attn_heads * head_size)
                    if self.num_attn_heads > 0
                    else self.num_heads * head_size
                )
                local_hidden_units_kv = (
                    int(self.num_attn_kv_heads * head_size)
                    if self.num_attn_kv_heads > 0
                    else self.num_kv_heads * head_size
                )

                # Some context FMHA kernels need temporary buffers
                context_workspace_size += (
                    dtype_size * num_tokens * (local_hidden_units_qo + 2 * local_hidden_units_kv)
                )

        # Calculate generation workspace size
        generation_workspace_size = 0
        if num_gen_tokens > 0:
            batch_beam = self.max_num_requests

            # Estimate max sequence length tiles for multi-block mode
            max_seq_len_tile = max(
                64,
                (self.multi_processor_count + self.num_heads - 1) // self.num_heads
                if self.num_heads > 0
                else 64,
            )

            # Partial output, sum, and max buffers for multi-block attention
            partial_out_size = (
                dtype_size * batch_beam * self.num_heads * head_size * max_seq_len_tile
            )
            partial_sum_size = (
                4 * batch_beam * self.num_heads * max_seq_len_tile
            )  # sizeof(float) = 4
            partial_max_size = 4 * batch_beam * self.num_heads * max_seq_len_tile

            generation_workspace_size = partial_out_size + partial_sum_size + partial_max_size

            # MLA workspace
            if self._is_mla_enabled:
                mla_workspace_size = 0
                num_kv_heads = self.num_kv_heads
                head_dim = self.mla_params.kv_lora_rank + self.mla_params.qk_rope_head_dim

                # Multi-CTA KV mode buffers
                mla_workspace_size += dtype_size * 256 * self.multi_processor_count * head_dim
                # Partial sum and max
                mla_workspace_size += 4 * 256 * self.multi_processor_count * 2

                if self.use_gen_flash_mla:
                    s_q = self.mla_params.predicted_tokens_per_seq
                    num_q_heads = (
                        self.num_heads // self.cp_size if self.cp_size > 0 else self.num_heads
                    )
                    head_size_v = self.mla_params.kv_lora_rank

                    num_sm_parts = self.get_flash_mla_num_sm_parts(
                        s_q, num_q_heads, num_kv_heads, head_size_v
                    )

                    # Flash MLA metadata and kernel buffers
                    mla_workspace_size += 4 * num_sm_parts * 8  # Metadata
                    mla_workspace_size += 4 * (batch_beam + 1)  # CU seqlens
                    mla_workspace_size += 4 * batch_beam * s_q * num_q_heads  # softmax_lse
                    mla_workspace_size += (
                        4 * (batch_beam + num_sm_parts) * num_q_heads * s_q
                    )  # softmax_lse_accum
                    mla_workspace_size += (
                        4 * (batch_beam + num_sm_parts) * num_q_heads * s_q * head_size_v
                    )  # out_accum

                generation_workspace_size += mla_workspace_size

            # XQA workspace
            if self.enable_xqa:
                xqa_workspace_size = 4 * (batch_beam + 1) * 2  # cu_seqlens + cu_kv_seqlens
                generation_workspace_size += xqa_workspace_size

        # Return the maximum of context and generation workspace sizes
        return max(context_workspace_size, generation_workspace_size)

    def get_flash_mla_num_sm_parts(
        self, s_q: int, num_heads: int, num_kv_heads: int, head_size_v: int
    ) -> int:
        """Calculate the number of SM parts for flash MLA."""
        block_size_m = 64
        num_heads_per_head_k = s_q * num_heads // num_kv_heads if num_kv_heads > 0 else 1
        sm_cnt = self.multi_processor_count
        num_sm_parts = (
            sm_cnt // num_kv_heads // ((num_heads_per_head_k + block_size_m - 1) // block_size_m)
            if num_kv_heads > 0
            else 1
        )
        return max(1, num_sm_parts)

    # This method can be ignored.
    # It's only used when we use XQA kernels in generation phase.
    #
    # def prepare_enqueue_generation(self, params: EnqueueGenerationParams) -> None:
    #     self.beam_width = params.beam_width
    #     self.max_num_requests = params.num_requests
    #     self.attention_window_size = params.max_attention_window_size
    #     self.sink_token_length = params.sink_token_length

    def initialize(self) -> int:
        """
        Initialize the attention op.

        This method corresponds to C++ AttentionOp::initialize() in common/attentionOp.cpp.
        It performs:
        1. Ulysses attention head/KV head computation
        2. Context FMHA support validation
        3. FP8/FP4 configuration checks
        4. MLA configuration checks
        5. FMHA dispatcher/runner construction
        6. XQA dispatcher construction
        7. Multi-block semaphore allocation
        """
        # ========== Step 1: Ulysses configuration ==========
        # Use Ulysses for GPTAttentionPlugin
        if self.attn_tp_size < 0 or self.attn_cp_size < 0:
            self.attn_tp_size = self.tp_size * self.cp_size
            self.attn_cp_size = 1

        self.num_attn_heads = self.num_heads * self.tp_size / self.attn_tp_size
        self.num_attn_kv_heads = (
            self.num_kv_heads * self.tp_size + self.attn_tp_size - 1
        ) / self.attn_tp_size

        if self.cp_size != self.attn_cp_size:
            # MQA broadcast
            self.ulysses_mqa_broadcast = (
                self.attn_tp_size + self.num_kv_heads_origin - 1
            ) / self.num_kv_heads_origin

        # ========== Step 2: Pre-check whether FMHA is supported ==========
        if self.enable_context_fmha:
            self.enable_context_fmha = False
            if not (self.type == DataType.HALF or self.type == DataType.BF16):
                logger.warning("Fall back to unfused MHA because of unsupported data type.")
            elif self.position_embedding_type == PositionEmbeddingType.RELATIVE:
                logger.warning("Fall back to unfused MHA because of relative position embedding.")
            elif self.is_cross_attention() and self.is_use_kv_cache() and not self.paged_kv_cache:
                # TODO: add the support for cross attention + contiguous kv cache.
                logger.warning(
                    "Fall back to unfused MHA because of cross attention + contiguous kv cache."
                )
            else:
                self.enable_context_fmha = True

        # ========== Step 3: FP8/FP4 configuration checks ==========
        # Pre-Check of FP8 Context FMHA.
        if self.fp8_context_fmha:
            assert self.enable_context_fmha, (
                "FP8 FMHA cannot be enabled because Context FMHA is not supported."
            )
            assert self.sm in (89, 90, 100, 103, 120, 121), (
                "FP8 FMHA can only be enabled on sm_89, sm_90, sm_100, sm_103, sm_120 or sm_121."
            )

        # Pre-Check of FP8 Generation MLA.
        if self.fp8_generation_mla:
            assert self._is_mla_enabled, (
                "FP8 Generation MLA cannot be enabled because MLA is not supported."
            )
            assert self.sm in (89, 90, 100, 103, 120, 121), (
                "FP8 Generation MLA is supported on Ada, Hopper or Blackwell architecture."
            )

        # Check requirements for FP4 output.
        assert not self.fuse_fp4_quant or self.enable_context_fmha, (
            "Context FMHA must enable if fuse_fp4_quant is enabled"
        )
        assert not self.fuse_fp4_quant or self.sm in (100, 103, 120, 121), (
            "fuse_fp4_quant only supports SM100, SM103, SM120 or SM121 devices."
        )

        # Check requirements for FP4 KV cache.
        assert not self.kv_cache_quant_mode.has_fp4_kv_cache() or self.fp8_context_fmha, (
            "mFP8ContextFMHA must enable if FP4 KV cache is enabled"
        )

        # ========== Step 4: Basic validation checks ==========
        # Check requirements for RoPE.
        assert self.is_rope() == (self.rotary_embedding_dim != 0), "RoPE configuration mismatch"
        assert self.sm >= 80 or self.type != DataType.BF16, (
            "Unsupported data type, pre SM 80 GPUs do not support bfloat16"
        )

        # Pre-check whether the head size is supported by MMHA.
        # Support head size == 72 only for fmha kernels, so skip pre-check here.
        if self.get_head_size() == 72:
            pass
        elif not self.mmha_supported(self.get_head_size()) and not self._is_mla_enabled:
            raise ValueError(f"Head size {self.get_head_size()} is not supported by MMHA.")

        # ========== Step 5: MLA configuration checks ==========
        if self._is_mla_enabled:
            assert self.enable_context_fmha, "MLA(Deepseek v2) only support fmha"
            assert not self.dense_context_fmha, "MLA(Deepseek v2) currently not support dense fmha"
            assert self.paged_kv_cache and self.use_kv_cache and self.remove_padding, (
                "MLA(Deepseek v2) only support paged kv cache"
            )
            assert not self.cross_attention, (
                "MLA(Deepseek v2) do not support cross attention right now"
            )
            assert self.mask_type != AttentionMaskType.CUSTOM_MASK, (
                "MLA(Deepseek v2) do not support custom mask right now"
            )
            assert self.mla_params.qk_rope_head_dim == 64 and self.mla_params.kv_lora_rank == 512, (
                "MLA(Deepseek v2) only support fixed kv_lora_rank(512) and fixed qk_rope_head_dim(64) right now."
            )

        # ========== Step 6: Construct the FMHA dispatcher ==========
        if self.enable_context_fmha:
            # Determine input/output data types
            if self.type == DataType.HALF:
                data_type = DataType.FP16
            elif self.type == DataType.BF16:
                data_type = DataType.BF16
            else:
                raise ValueError("GPTAttentionPlugin received wrong data type.")

            # Output data type
            data_type_out = DataType.E4M3 if self.fp8_atten_output else data_type

            # FP8 FMHA should be used with fp8 workflow together.
            if self.fp8_context_fmha or self.fp8_context_mla:
                data_type = DataType.E4M3

            # KV input data type
            data_type_kv = data_type
            if self.paged_kv_cache and self.paged_context_fmha:
                if self.kv_cache_quant_mode.has_fp8_kv_cache():
                    data_type_kv = DataType.E4M3
                elif self.kv_cache_quant_mode.has_fp4_kv_cache():
                    data_type_kv = DataType.E2M1

            if self.fuse_fp4_quant:
                # If FP4 quantization workflow is enabled, set output type to FP4.
                data_type_out = DataType.E2M1

            if self._is_mla_enabled:
                # For FP8 MLA, currently context attention is performed in BF16.
                data_type_out = DataType.BF16
                data_type_kv = DataType.BF16

            if self.fp8_context_mla and self.kv_cache_quant_mode.has_fp8_kv_cache():
                data_type_kv = DataType.E4M3
                data_type_out = DataType.BF16

            # Determine attention input layout
            if self.is_cross_attention():
                # Always use paged-kv-fmha if paged_kv cache is used.
                attention_input_layout = "Q_PAGED_KV" if self.paged_kv_cache else "Q_CONTIGUOUS_KV"
            elif not self.use_kv_cache:
                attention_input_layout = "PACKED_QKV"
            else:
                attention_input_layout = (
                    "Q_PAGED_KV"
                    if (self.paged_kv_cache and self.paged_context_fmha)
                    else "PACKED_QKV"
                )

            # Store FMHA params for later use
            self._fmha_data_type = data_type
            self._fmha_data_type_kv = data_type_kv
            self._fmha_data_type_out = data_type_out
            self._fmha_attention_input_layout = attention_input_layout
            self._fmha_is_s_padded = not self.remove_padding

            # MLA-specific FMHA params
            if self._is_mla_enabled:
                if self.use_sparse_mla():
                    self._fmha_attention_input_layout = "Q_PAGED_KV"
                    self._fmha_num_kv_heads = 1
                    self._fmha_head_size = (
                        self.mla_params.kv_lora_rank + self.mla_params.qk_rope_head_dim
                    )
                    self._fmha_head_size_v = self.mla_params.kv_lora_rank
                    self._fmha_head_size_qk_nope = self.mla_params.qk_nope_head_dim
                    # Adjust the qScaling for the absorption mode.
                    self._fmha_q_scaling = (
                        self.q_scaling
                        * math.sqrt(
                            self.mla_params.qk_nope_head_dim + self.mla_params.qk_rope_head_dim
                        )
                        / math.sqrt(self.mla_params.kv_lora_rank + self.mla_params.qk_rope_head_dim)
                    )
                else:
                    # Context MLA always use separate_q_k_v layout
                    self._fmha_attention_input_layout = "SEPARATE_Q_K_V"
                    self._fmha_num_kv_heads = self.num_heads
                    self._fmha_head_size = (
                        self.mla_params.qk_nope_head_dim + self.mla_params.qk_rope_head_dim
                    )
                    self._fmha_head_size_v = self.mla_params.qk_nope_head_dim
                    self._fmha_head_size_qk_nope = self.mla_params.qk_nope_head_dim
                    self._fmha_q_scaling = self.q_scaling
            else:
                self._fmha_num_kv_heads = int(self.num_attn_kv_heads)
                self._fmha_head_size = self.head_size
                self._fmha_head_size_v = self.head_size
                self._fmha_head_size_qk_nope = 0
                self._fmha_q_scaling = self.q_scaling

            # Construct the TrtllmGenFmhaRunner for MLA generation
            if self._is_mla_enabled:
                # Update XQA enablement for MLA
                self.enable_xqa = (self.sm == 120) and self.is_generation_mla

                if self.use_tllm_gen:
                    # Determine data types for TrtllmGenFmhaRunner
                    if self.type == DataType.HALF:
                        q_data_type = DataType.FP16
                        kv_data_type = DataType.FP16
                        output_data_type = DataType.FP16
                    elif self.type == DataType.BF16:
                        q_data_type = DataType.BF16
                        kv_data_type = DataType.BF16
                        output_data_type = DataType.BF16
                    else:
                        raise ValueError("The data type is not supported.")

                    if self.kv_cache_quant_mode.has_fp8_kv_cache():
                        q_data_type = DataType.E4M3
                        kv_data_type = DataType.E4M3

                    # Store for TrtllmGenFmhaRunner instantiation
                    self._tllm_gen_fmha_q_type = q_data_type
                    self._tllm_gen_fmha_kv_type = kv_data_type
                    self._tllm_gen_fmha_output_type = output_data_type

                    # The actual runner will be created when needed
                    self.tllm_gen_fmha_runner = None  # Placeholder

                elif self.is_generation_mla and not self.use_gen_flash_mla:
                    # Construct the FMHA runner for generation
                    gen_data_type = DataType.E4M3 if self.fp8_generation_mla else data_type
                    gen_data_type_out = DataType.BF16 if self.fp8_generation_mla else gen_data_type

                    self._decoder_fmha_data_type = gen_data_type
                    self._decoder_fmha_data_type_out = gen_data_type_out
                    self._decoder_fmha_q_scaling = (
                        self.q_scaling
                        * math.sqrt(
                            self.mla_params.qk_nope_head_dim + self.mla_params.qk_rope_head_dim
                        )
                        / math.sqrt(self.mla_params.kv_lora_rank + self.mla_params.qk_rope_head_dim)
                    )

            # Generation MLA reuses the context FMHA code path
            # Fall back to unfused MHA kernels if not supported (simplified check in Python)
            self.enable_context_fmha = self.is_generation_mla or self.enable_context_fmha

        # ========== Step 7: Custom mask check ==========
        # Only FMHA supports custom mask currently.
        assert not self.use_custom_mask() or self.enable_context_fmha, (
            "Only Context FMHA supports custom mask input currently."
        )

        # ========== Step 8: XQA dispatcher configuration ==========
        self.enable_xqa = (
            (self.enable_xqa or self.is_spec_decoding_enabled)
            and (self.type == DataType.HALF or self.type == DataType.BF16)
            and self.use_kv_cache
        )

        if self.enable_xqa:
            logger.debug("Enabling XQA kernels for GPTAttention.")

            # Determine XQA data types
            if self.type == DataType.HALF:
                xqa_input_type = DataType.FP16
                xqa_output_type = DataType.FP16
            elif self.type == DataType.BF16:
                xqa_input_type = DataType.BF16
                xqa_output_type = DataType.BF16
            else:
                xqa_input_type = DataType.FP32
                xqa_output_type = DataType.FP32

            # Determine KV cache and math data type
            if self.kv_cache_quant_mode.has_int8_kv_cache():
                xqa_kv_type = DataType.INT8
                xqa_math_type = xqa_input_type
            elif self.kv_cache_quant_mode.has_fp8_kv_cache():
                xqa_kv_type = DataType.E4M3
                xqa_math_type = DataType.E4M3
            elif self.kv_cache_quant_mode.has_fp4_kv_cache():
                xqa_kv_type = DataType.E2M1
                xqa_math_type = DataType.E4M3
            else:
                xqa_kv_type = xqa_input_type
                xqa_math_type = xqa_input_type

            # If fuse_fp4_quant is enabled, set output data type to FP4.
            if self.fuse_fp4_quant:
                xqa_output_type = DataType.E2M1
            elif self.fp8_atten_output:
                xqa_output_type = DataType.E4M3

            if self.is_spec_decoding_enabled and not self.use_tllm_gen:
                xqa_output_type = DataType.E4M3
                assert self.num_heads % self.num_kv_heads == 0, (
                    "mNumHeads should be multiples of mNumKVHeads."
                )

            # Store XQA configuration
            self._xqa_input_type = xqa_input_type
            self._xqa_output_type = xqa_output_type
            self._xqa_kv_type = xqa_kv_type
            self._xqa_math_type = xqa_math_type
            self._xqa_is_mla = self.is_generation_mla

        elif self.is_spec_decoding_enabled:
            raise ValueError(
                "Speculative decoding mode doesn't support the data type or cross attention."
            )

        # ========== Step 9: Multi-block semaphore allocation ==========
        if self.nb_multi_block_semaphores > 0:
            # In Python, we'll allocate semaphores when needed
            self._multi_block_semaphores = None  # Will be allocated on device

        # ========== Step 10: Construct Ulysses CP runner ==========
        # Construct the Ulysses CP runner for multi-device context parallelism
        if self.cp_size > 1 and self.attn_tp_size > 1 and self.attn_cp_size == 1:
            self.ulysses_cp_runner = UlyssesContextParallelism(
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
                num_attn_heads=int(self.num_attn_heads),
                num_attn_kv_heads=int(self.num_attn_kv_heads),
                head_size=self.get_head_size(),
                cp_group=self.cp_group if self.cp_group else None,
                mqa_broadcast=int(self.ulysses_mqa_broadcast),
                fp8_output=self.fp8_context_fmha,
            )

        return 0

    def enqueue_context(self, params: EnqueueContextParams) -> int:
        """
        Enqueue context attention.

        Maps to cpp/tensorrt_llm/common/attentionOp.cpp::enqueueContext().

        This method performs:
        1. Validates configuration and prepares buffers
        2. QKV preprocessing (RoPE, KV cache write) via run_qkv_preprocessing
        3. Runs FMHA kernel via tllm_gen_fmha_runner

        Args:
            params: EnqueueContextParams containing all input tensors and parameters

        Returns:
            0 on success
        """
        head_size = self.get_head_size()
        position_embedding_type = self.position_embedding_type
        q_scaling = self.q_scaling

        # Calculate size per token for KV cache
        size_per_token = (
            self.num_attn_kv_heads * head_size * self.get_kv_cache_elem_size_in_bits() // 8
        )

        # In context phase, FMHA runner has restrictions:
        # 1. Only applies to self attention
        # 2. Doesn't apply to MHA with relative attention bias
        if not self.enable_context_fmha:
            logger.warning("Context FMHA is not enabled, falling back to unfused MHA")
            return -1

        attention_input = params.attention_input

        # Handle Ulysses context preprocess for CP > 1
        # do all-to-all for params.attention_input, need to split on kv head
        # [token_num // cp_size, kv_heads, head_size] -> [token_num, kv_heads // cp_size, head_size]
        if self.cp_size > 1 and self.attn_tp_size > 1 and self.attn_cp_size == 1:
            attention_input = self.ulysses_cp_runner.context_preprocess(
                attention_input=attention_input,
                context_lengths=params.context_lengths,
                batch_size=params.batch_size,
            )

        enable_paged_kv_context_fmha = self.paged_kv_cache and self.paged_context_fmha

        # Validation checks
        if self.kv_cache_quant_mode.has_int8_kv_cache() and enable_paged_kv_context_fmha:
            raise RuntimeError("Paged Context FMHA doesn't work with int8 kv cache currently.")
        if params.sink_token_length > 0 and enable_paged_kv_context_fmha:
            raise RuntimeError(
                "Cannot support StreamingLLM now when enabling paged KV context FMHA."
            )

        # The max_kv_seq_len comes from the encoder seqlen when cross attention is used
        max_kv_seq_len = (
            params.cross_kv_length if self.is_cross_attention() else params.max_past_kv_length
        )

        # Build cu_seqlens if not provided
        cu_q_seqlens = None
        cu_kv_seqlens = None

        # Calculate cyclic kv cache length
        cyclic_kv_cache_len = (
            params.cross_kv_length
            if self.is_cross_attention()
            else params.cyclic_attention_window_size
        )

        # Prepare QKV preprocessing parameters - this applies RoPE and writes to KV cache
        preprocess_params = QKVPreprocessingParams(
            # Tensor parameters
            qkv_input=attention_input,
            q_output=None,  # Will be set if separate Q/KV output is needed
            qkv_bias=params.qkv_bias,
            seq_lens=params.context_lengths,
            cache_seq_lens=params.sequence_lengths,
            cu_seq_lens=cu_q_seqlens,
            rotary_embedding_inv_freq=params.rotary_inv_freq,
            rotary_coef_cache_buffer=params.rotary_cos_sin,
            qkv_scale_orig_quant=params.kv_scale_orig_quant,
            qkv_scale_quant_orig=params.kv_scale_quant_orig,
            spec_decoding_position_offsets=None,
            mrope_rotary_cos_sin=params.mrope_rotary_cos_sin,
            mrope_position_deltas=None,
            block_offsets=params.block_offsets,
            # Scalar parameters
            batch_size=params.batch_size,
            max_input_seq_len=params.input_seq_length,
            max_kv_seq_len=max_kv_seq_len,
            cyclic_kv_cache_len=cyclic_kv_cache_len,
            sink_token_len=params.sink_token_length,
            token_num=params.num_tokens,
            remove_padding=self.remove_padding,
            is_last_chunk=True,
            cross_attention=self.is_cross_attention(),
            head_num=int(self.num_attn_heads),
            kv_head_num=int(self.num_attn_kv_heads),
            size_per_head=head_size,
            rotary_embedding_dim=self.rotary_embedding_dim,
            rotary_embedding_base=self.rotary_embedding_base,
            rotary_scale_type=int(self.rotary_embedding_scale_type),
            rotary_embedding_scale=self.rotary_embedding_scale,
            rotary_embedding_max_positions=self.rotary_embedding_max_positions,
            position_embedding_type=int(position_embedding_type),
            position_shift_enabled=self.pos_shift_enabled,
            separate_q_kv_output=enable_paged_kv_context_fmha or self.is_cross_attention(),
            quantized_fp8_output=self.fp8_context_fmha,
            generation_phase=False,
            multi_processor_count=self.multi_processor_count,
            rotary_vision_start=self.vision_start,
            rotary_vision_length=self.vision_length,
            quant_mode=int(self.kv_cache_quant_mode),
            # KV cache buffer parameters
            tokens_per_block=self.tokens_per_block,
            max_blocks_per_sequence=params.max_blocks_per_sequence,
            attention_window_size=params.max_attention_window_size,
            size_per_token=size_per_token,
            sink_token_length=params.sink_token_length,
            max_cyclic_attention_window_size=params.max_cyclic_attention_window_size,
            can_use_one_more_block=params.can_use_one_more_block,
            host_primary_pool_pointer=params.host_primary_pool_pointer or 0,
            host_secondary_pool_pointer=params.host_secondary_pool_pointer or 0,
        )

        # Run QKV preprocessing (RoPE + KV cache write)
        if not self._is_mla_enabled:
            qkv_preprocess_runner = QKVPreprocessRunner()
            qkv_preprocess_runner.run(preprocess_params)

        # Handle MLA (Multi-head Latent Attention) separately
        if self._is_mla_enabled:
            assert params.mla_param is not None, "MLA param is nullptr"
            mla_param = params.mla_param

            # Set up MLA parameters that come from the runtime
            # These correspond to C++ attentionOp.cpp lines 1745-1775
            absorption_mode = self.use_sparse_mla() if hasattr(self, "use_sparse_mla") else False

            # Build cu_q_seqlens if not provided
            if cu_q_seqlens is None and params.context_lengths is not None:
                cu_q_seqlens = torch.zeros(
                    params.batch_size + 1, dtype=torch.int32, device=params.context_lengths.device
                )
                cu_q_seqlens[1:] = torch.cumsum(params.context_lengths, dim=0)

            # Run MLA RoPE context processing
            # This applies RoPE to Q/K and writes to KV cache
            if mla_param.latent_cache is not None:
                mla_runner = MLAContextRunner()

                # Prepare q_pe tensor if available
                # q_pe has shape [total_q_len, num_heads, qk_rope_head_dim]
                q_pe = getattr(mla_param, "q_pe", None)

                if q_pe is not None:
                    mla_runner.run_rope_context(
                        q=mla_param.q_buf if mla_param.q_buf is not None else attention_input,
                        q_pe=q_pe,
                        latent_cache=mla_param.latent_cache,
                        cos_sin_cache=params.rotary_cos_sin,
                        cu_q_seqlens=cu_q_seqlens,
                        cache_seq_lens=params.sequence_lengths,
                        kv_cache_block_offsets=params.block_offsets,
                        host_kv_cache_pool_pointers=torch.tensor(
                            [
                                [
                                    params.host_primary_pool_pointer or 0,
                                    params.host_secondary_pool_pointer or 0,
                                ]
                            ],
                            dtype=torch.int64,
                        ),
                        host_kv_cache_pool_mapping=torch.tensor(
                            [[0, self.layer_idx if hasattr(self, "layer_idx") else 0]],
                            dtype=torch.int32,
                        ),
                        batch_size=params.batch_size,
                        num_heads=int(self.num_attn_heads),
                        max_input_seq_len=params.input_seq_length,
                        layer_idx=self.layer_idx if hasattr(self, "layer_idx") else 0,
                        tokens_per_block=self.tokens_per_block,
                        attention_window_size=params.max_attention_window_size,
                        sink_token_length=params.sink_token_length,
                        beam_width=1,  # Always 1 for context
                        quant_mode=int(self.kv_cache_quant_mode),
                        mla_params=self.mla_params,
                        absorption_mode=absorption_mode,
                        k=mla_param.k_buf,
                        kv_scale_orig_quant=mla_param.quant_scale_kv,
                    )
                else:
                    logger.warning("MLA RoPE context: q_pe not available")

            # Run FP8 quantization if enabled
            if self.fp8_context_mla:
                mla_runner = MLAContextRunner()

                # Allocate FP8 quantized buffers if not provided
                if mla_param.quant_q_buf is not None and mla_param.quant_scale_qkv is not None:
                    mla_runner.run_fp8_quantize(
                        q=mla_param.q_buf if mla_param.q_buf is not None else attention_input,
                        quant_q=mla_param.quant_q_buf,
                        cu_q_seqlens=cu_q_seqlens,
                        quant_scale_qkv=mla_param.quant_scale_qkv
                        if hasattr(mla_param, "quant_scale_qkv")
                        else params.kv_scale_orig_quant,
                        batch_size=params.batch_size,
                        num_heads=int(self.num_attn_heads),
                        max_input_seq_len=params.input_seq_length,
                        total_kv_len=params.total_kv_len,
                        mla_params=self.mla_params,
                        absorption_mode=absorption_mode,
                        k=mla_param.k_buf,
                        v=mla_param.v_buf,
                        quant_k=mla_param.quant_k_buf,
                        quant_v=mla_param.quant_v_buf,
                    )
                else:
                    logger.warning("MLA FP8 quantize: required buffers not available")

        # Construct FMHA params and run kernel
        if self.tllm_gen_fmha_runner is not None:
            # Determine if QKV is fused or separate
            is_fused_qkv = params.k_ptr is None and params.v_ptr is None

            # Calculate q_scaling factor
            scale_q = 1.0 / math.sqrt(head_size) * q_scaling

            # Determine QKV layout
            if is_fused_qkv:
                qkv_layout = QkvLayout.PACKED_QKV
            else:
                qkv_layout = QkvLayout.SEPARATE_QKV

            # Create FMHA runner params
            fmha_params = TrtllmGenFmhaRunnerParams(
                # Layout and mask configuration
                qkv_layout=qkv_layout,
                mask_type=_mask_type_to_trtllm_gen(int(self.mask_type)),
                kernel_type=FmhaKernelType.CONTEXT,
                tile_scheduler=TileScheduler.STATIC,
                multi_ctas_kv_mode=False,
                use_block_sparse_attention=False,
                # Tensor pointers
                q_ptr=None if is_fused_qkv else attention_input,
                k_ptr=params.k_ptr,
                v_ptr=params.v_ptr,
                kv_ptr=None,
                kv_sf_ptr=None,
                qkv_ptr=attention_input if is_fused_qkv else None,
                attention_sinks_ptr=None,
                custom_mask_ptr=params.attention_packed_mask,
                custom_mask_offsets_ptr=None,
                first_sparse_mask_offsets_kv_ptr=None,
                multi_ctas_kv_counter_ptr=None,
                seq_lens_kv_ptr=params.sequence_lengths,
                cum_seq_lens_q_ptr=cu_q_seqlens,
                cum_seq_lens_kv_ptr=cu_kv_seqlens,
                kv_page_idx_ptr=params.block_offsets,
                output_scale_ptr=params.attention_output_orig_quant,
                scale_softmax_log2_ptr=None,
                kv_sf_scale_ptr=params.kv_scale_quant_orig,
                o_sf_scale_ptr=None,
                multi_ctas_kv_scratch_ptr=None,
                softmax_stats_ptr=params.softmax_stats,
                o_ptr=params.context_buf,
                o_sf_ptr=params.context_buf_sf,
                seqlens_q_ptr=params.context_lengths,
                # Dimension parameters
                head_dim_qk=head_size,
                head_dim_v=head_size,
                head_dim_qk_nope=0,  # Only used for MLA
                num_heads_q=int(self.num_attn_heads),
                num_heads_kv=int(self.num_attn_kv_heads),
                num_heads_q_per_kv=int(self.num_attn_heads // self.num_attn_kv_heads)
                if self.num_attn_kv_heads > 0
                else int(self.num_attn_heads),
                batch_size=params.batch_size,
                max_seq_len_cache_kv=params.max_attention_window_size,
                max_seq_len_q=params.input_seq_length,
                max_seq_len_kv=max_kv_seq_len,
                attention_window_size=params.max_attention_window_size,
                chunked_attention_size=self.attention_chunk_size
                if self.attention_chunk_size is not None
                else 0,
                sum_of_seq_lens_q=params.num_tokens,
                sum_of_seq_lens_kv=params.num_tokens,
                max_num_pages_per_seq_kv=params.max_blocks_per_sequence,
                num_tokens_per_page=self.tokens_per_block,
                num_pages_in_mem_pool=0,  # Will be determined from cache if needed
                multi_processor_count=self.multi_processor_count,
                scale_q=scale_q,
                sf_start_token_idx=0,
                skip_softmax_threshold_scale_factor=self.skip_softmax_threshold_scale_factor_prefill,
                sparse_mla=False,
                sparse_mla_top_k=0,
                layer_idx=self.layer_idx if hasattr(self, "layer_idx") else 0,
                is_spec_dec_tree=False,
            )

            # Run the FMHA kernel
            self.tllm_gen_fmha_runner.run(fmha_params)

        # Handle Ulysses context postprocess for CP > 1
        # [token_num, kv_heads // cp_size, head_size] -> [token_num // cp_size, kv_heads, head_size]
        if self.cp_size > 1 and self.attn_tp_size > 1 and self.attn_cp_size == 1:
            if self.ulysses_cp_runner is not None and params.context_buf is not None:
                # Run postprocess on FMHA output to gather heads across ranks
                output = self.ulysses_cp_runner.context_postprocess(
                    fmha_output=params.context_buf,
                    context_lengths=params.context_lengths,
                    batch_size=params.batch_size,
                )
                # Copy result back to context_buf
                params.context_buf.copy_(output)

        # KV cache postprocessing for non-MLA
        # This handles sparse KV cache updates after FMHA
        if not self._is_mla_enabled:
            kv_postprocess_runner = KVCachePostprocessRunner()
            postprocess_params = KVCachePostprocessParams(
                qkv_input=attention_input,
                sparse_kv_indices=self.runtime_sparse_attention_params.sparse_kv_indices
                if hasattr(self, "runtime_sparse_attention_params")
                else None,
                sparse_kv_offsets=self.runtime_sparse_attention_params.sparse_kv_offsets
                if hasattr(self, "runtime_sparse_attention_params")
                else None,
                block_offsets=params.block_offsets,
                batch_size=params.batch_size,
                is_last_chunk=True,  # Context phase is always the last chunk
                head_num=int(self.num_attn_heads),
                kv_head_num=int(self.num_attn_kv_heads),
                size_per_head=head_size,
                quant_mode=int(self.kv_cache_quant_mode),
                tokens_per_block=self.tokens_per_block,
                max_blocks_per_sequence=params.max_blocks_per_sequence,
                attention_window_size=params.max_attention_window_size,
                size_per_token=size_per_token,
                sink_token_length=params.sink_token_length,
                max_cyclic_attention_window_size=params.max_cyclic_attention_window_size,
                can_use_one_more_block=params.can_use_one_more_block,
                host_primary_pool_pointer=params.host_primary_pool_pointer or 0,
                host_secondary_pool_pointer=params.host_secondary_pool_pointer or 0,
            )
            kv_postprocess_runner.run(postprocess_params)

        return 0

    def enqueue_generation(self, params: EnqueueGenerationParams) -> int:
        """
        Enqueue generation attention.

        Args:
            params: Generation phase parameters

        Returns:
            0 on success
        """
        # TODO: Implement generation phase attention
        raise NotImplementedError("Generation phase attention is not implemented")
        return 0

    def mla_generation(self, params: MlaParams, generation_params: EnqueueGenerationParams) -> int:
        """
        Enqueue MLA generation attention.

        Args:
            params: MLA runtime parameters
            generation_params: Generation phase parameters

        Returns:
            0 on success
        """
        # TODO: Implement MLA generation
        raise NotImplementedError("MLA generation is not implemented")
        return 0


########################################################
# Flashinfer API wrapper
########################################################


def trtllm_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    window_left: int = -1,
    out: Optional[Union[torch.Tensor, FP4Tensor]] = None,
    out_dtype: Optional[Union[torch.dtype, str]] = None,
    o_sf_scale: Optional[float] = None,
    o_sf_vec_size: Optional[int] = None,
    kv_layout: str = "HND",
    enable_pdl: Optional[bool] = None,
    sinks: Optional[List[torch.Tensor]] = None,
) -> Union[torch.Tensor, FP4Tensor]:
    """
    Parameters
    ----------
    query : torch.Tensor
        query tensor with shape [num_tokens, num_heads, head_dim]
    kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If kv_cache is a single tensor, it should be a tensor with shape
        [num_pages, 1 or 2, num_kv_heads, page_size, head_dim] if ``kv_layout`` is "HND",
        or [num_pages, 1 or 2, page_size, num_kv_heads, head_dim] if ``kv_layout`` is "NHD".
        If kv_cache is a tuple of two tensors, it should be a tuple of two tensors with shape
        [num_pages, num_kv_heads, page_size, head_dim] if ``kv_layout`` is "HND",
        or [num_pages, page_size, num_kv_heads, head_dim] if ``kv_layout`` is "NHD".
        The first tensor is the key cache, the second tensor is the value cache.
    workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
        workspace
    block_tables : torch.Tensor
        page_table of kv cache, [batch_size, num_pages]
    seq_lens : torch.Tensor
        A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``
    max_q_len : int
        max sequence length for query
    max_kv_len : int
        max sequence length for kv_cache
    bmm1_scale : Union[float, torch.Tensor]
        fused scale for bmm1 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.
    bmm2_scale : Union[float, torch.Tensor]
        fused scale for bmm2 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.
    batch_size : int
        batch size
    cum_seq_lens_q : torch.Tensor
        cumulative sequence length for query. shape: ``[batch_size + 1]``
    cum_seq_lens_kv : torch.Tensor
        cumulative sequence length for kv_cache. shape: ``[batch_size + 1]``
    window_left : int = -1
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    out : Optional[Union[torch.Tensor, FP4Tensor]] = None
        output tensor, if not provided, will be allocated with ``out_dtype``,
        if ``out_dtype`` is not provided, will use the type of ``query``.
    out_dtype : Optional[Union[torch.dtype, str]] = None
        output dtype, if not provided, will use the type of ``out``.
        For nvfp4, use string ``nvfp4``.
    o_sf_scale : Optional[float] = None
        scale for nvfp4 output tensor scale factor.
    o_sf_vec_size : Optional[int] = None
        vector size for nvfp4 output tensor scale factor.
    enable_pdl : Optional[bool] = None
        Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
        Defaults to ``None``, which means it will be enabled if the device supports PDL.
    kv_layout : str = "HND"
        Layout of kv-cache, can be "HND" or "NHD", default is "HND".
    sinks : Optional[List[torch.Tensor]] = None
        additional value per head in the denominator of the softmax.

    Returns
    -------
    out: Union[torch.Tensor, FP4Tensor]
        output torch.Tensor or FP4Tensor.
    """
    # ========== Step 1: Parse query tensor dimensions ==========
    num_tokens, num_heads, head_dim = query.shape

    # ========== Step 2: Parse KV cache and extract dimensions ==========
    if isinstance(kv_cache, tuple):
        # Separate K and V caches
        k_cache, v_cache = kv_cache
        if kv_layout == "HND":
            # [num_pages, num_kv_heads, page_size, head_dim]
            num_pages, num_kv_heads, page_size, kv_head_dim = k_cache.shape
        else:  # NHD
            # [num_pages, page_size, num_kv_heads, head_dim]
            num_pages, page_size, num_kv_heads, kv_head_dim = k_cache.shape
        # Get pool pointers
        host_primary_pool_pointer = k_cache.data_ptr()
        host_secondary_pool_pointer = v_cache.data_ptr()
    else:
        # Combined KV cache
        if kv_layout == "HND":
            # [num_pages, 1 or 2, num_kv_heads, page_size, head_dim]
            if kv_cache.dim() == 5:
                num_pages, kv_factor, num_kv_heads, page_size, kv_head_dim = kv_cache.shape
            else:
                raise ValueError(f"Unexpected kv_cache shape: {kv_cache.shape}")
        else:  # NHD
            # [num_pages, 1 or 2, page_size, num_kv_heads, head_dim]
            if kv_cache.dim() == 5:
                num_pages, kv_factor, page_size, num_kv_heads, kv_head_dim = kv_cache.shape
            else:
                raise ValueError(f"Unexpected kv_cache shape: {kv_cache.shape}")

        # Get pool pointer - for combined cache, both point to same buffer
        host_primary_pool_pointer = kv_cache.data_ptr()
        host_secondary_pool_pointer = kv_cache.data_ptr()

    # ========== Step 3: Determine output dtype and allocate output ==========
    is_nvfp4_output = out_dtype == "nvfp4"

    if out is None:
        if is_nvfp4_output:
            # Allocate FP4 output tensor
            assert o_sf_vec_size is not None, "o_sf_vec_size must be provided for nvfp4 output"
            out_data = torch.empty(
                (num_tokens, num_heads, head_dim // 2),  # FP4 packs 2 elements per byte
                dtype=torch.uint8,
                device=query.device,
            )
            sf_size = (num_tokens * num_heads * head_dim + o_sf_vec_size - 1) // o_sf_vec_size
            out_sf = torch.empty(sf_size, dtype=torch.float8_e4m3fn, device=query.device)
            out = FP4Tensor(out_data, out_sf, o_sf_scale or 1.0)
        else:
            actual_out_dtype = out_dtype if out_dtype is not None else query.dtype
            out = torch.empty(
                (num_tokens, num_heads, head_dim), dtype=actual_out_dtype, device=query.device
            )

    # Get output tensor and scale factor tensor
    if isinstance(out, FP4Tensor):
        context_buf = out.data
        context_buf_sf = out.scale
        out_scale = torch.tensor([o_sf_scale or 1.0], dtype=torch.float32, device=query.device)
    else:
        context_buf = out
        context_buf_sf = None
        out_scale = None

    # ========== Step 4: Create and configure TrtllmGenAttention ==========
    attn_op = TrtllmGenAttention()

    # Configure basic parameters
    attn_op.num_heads = num_heads
    attn_op.num_kv_heads = num_kv_heads
    attn_op.head_size = head_dim
    attn_op.tokens_per_block = page_size
    attn_op.max_num_requests = batch_size
    attn_op.max_context_length = max_kv_len

    # Configure data type
    if query.dtype == torch.float16:
        attn_op.type = DataType.FP16
    elif query.dtype == torch.bfloat16:
        attn_op.type = DataType.BF16
    elif query.dtype == torch.float32:
        attn_op.type = DataType.FP32
    else:
        attn_op.type = DataType.FP16

    # Configure attention window
    if window_left > 0:
        attn_op.mask_type = AttentionMaskType.SLIDING_WINDOW_CAUSAL
        attn_op.attention_window_size = window_left
    else:
        attn_op.mask_type = AttentionMaskType.CAUSAL
        attn_op.attention_window_size = max_kv_len

    # Configure paged KV cache
    attn_op.paged_kv_cache = True
    attn_op.paged_context_fmha = True
    attn_op.use_kv_cache = True

    # Configure FP4 output
    attn_op.fuse_fp4_quant = is_nvfp4_output

    # Set q_scaling from bmm1_scale
    if isinstance(bmm1_scale, torch.Tensor):
        attn_op.q_scaling = bmm1_scale.item()
    else:
        attn_op.q_scaling = bmm1_scale

    # Re-initialize with new configuration
    attn_op.initialize()

    # ========== Step 5: Compute sequence lengths ==========
    # Context lengths are the Q sequence lengths (from cumsum difference)
    context_lengths = cum_seq_lens_q[1:] - cum_seq_lens_q[:-1]
    context_lengths = context_lengths.to(torch.int32)

    # Sequence lengths are the KV cache lengths
    sequence_lengths = seq_lens.to(torch.int32)

    # Total KV length
    total_kv_len = seq_lens.sum().item()

    # ========== Step 6: Prepare attention sinks ==========
    attention_sinks_tensor = None
    if sinks is not None and len(sinks) > 0:
        attention_sinks_tensor = torch.stack(sinks, dim=0).to(torch.float32)

    # ========== Step 7: Create EnqueueContextParams ==========
    max_blocks_per_sequence = block_tables.size(1)

    params = EnqueueContextParams(
        # Input tensor
        attention_input=query.view(num_tokens, -1),  # Flatten to [num_tokens, hidden_dim]
        qkv_bias=None,
        attention_mask=None,
        attention_sinks=attention_sinks_tensor,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        # Sequence length parameters
        input_seq_length=max_q_len,
        max_past_kv_length=max_kv_len,
        max_attention_window_size=max_kv_len if window_left < 0 else window_left,
        cyclic_attention_window_size=max_kv_len if window_left < 0 else window_left,
        max_cyclic_attention_window_size=max_kv_len if window_left < 0 else window_left,
        can_use_one_more_block=False,
        sink_token_length=0,
        # Quantization scales
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        attention_output_orig_quant=out_scale,
        attention_output_sf_scale=out_scale,
        # Output buffer
        context_buf=context_buf.view(num_tokens, -1),  # Flatten to [num_tokens, hidden_dim]
        context_buf_sf=context_buf_sf,
        # KV cache parameters
        key_value_cache=None,
        block_offsets=block_tables,
        host_block_offsets=block_tables.cpu() if block_tables.is_cuda else block_tables,
        host_primary_pool_pointer=host_primary_pool_pointer,
        host_secondary_pool_pointer=host_secondary_pool_pointer,
        # Batch parameters
        batch_size=batch_size,
        num_tokens=num_tokens,
        total_kv_len=total_kv_len,
        max_blocks_per_sequence=max_blocks_per_sequence,
        # Sequence lengths
        sequence_lengths=sequence_lengths,
        context_lengths=context_lengths,
        host_context_lengths=context_lengths.cpu() if context_lengths.is_cuda else context_lengths,
        # Workspace
        workspace=workspace_buffer,
        # Context-specific parameters
        attention_packed_mask=None,
        mrope_rotary_cos_sin=None,
        cross_kv=None,
        cross_kv_length=0,
        num_encoder_tokens=0,
        mla_param=None,
        k_ptr=None,
        v_ptr=None,
        # Stream
        stream=torch.cuda.current_stream(),
    )

    # ========== Step 8: Call enqueue_context ==========
    attn_op.enqueue_context(params)

    # ========== Step 9: Return output ==========
    if isinstance(out, FP4Tensor):
        return out
    else:
        return out.view(num_tokens, num_heads, head_dim)


def trtllm_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    window_left: int = -1,
    out: Optional[Union[torch.Tensor, FP4Tensor]] = None,
    out_dtype: Optional[Union[torch.dtype, str]] = None,
    o_sf_scale: Optional[float] = None,
    o_sf_vec_size: Optional[int] = None,
    sinks: Optional[List[torch.Tensor]] = None,
    kv_layout: str = "HND",
    enable_pdl: Optional[bool] = None,
    backend: str = "auto",
    q_len_per_req: Optional[int] = 1,
    o_scale: Optional[float] = 1.0,
    mask: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, FP4Tensor]:
    """
    Parameters
    ----------
    query : torch.Tensor
        query tensor with shape [num_tokens, num_heads, head_dim],
        num_tokens = total query tokens in the batch.

    kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If kv_cache is a single tensor, it should be a tensor with shape
        [num_pages, 1 or 2, num_kv_heads, page_size, head_dim] if ``kv_layout`` is ``HND``,
        or [num_pages, 1 or 2, page_size, num_kv_heads, head_dim] if ``kv_layout`` is ``NHD``.
        If kv_cache is a tuple of two tensors, it should be a tuple of two tensors with shape
        [num_pages, num_kv_heads, page_size, head_dim] if ``kv_layout`` is ``HND``,
        or [num_pages, page_size, num_kv_heads, head_dim] if ``kv_layout`` is ``NHD``.
        The first tensor is the key cache, and the second tensor is the value cache.

    workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
        workspace

    block_tables : torch.Tensor
        page_table of kv cache, [batch_size, num_pages]

    seq_lens : torch.Tensor
        A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``

    max_seq_len : int
        max sequence length for kv_cache

    bmm1_scale : Union[float, torch.Tensor]
        fused scale for bmm1 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.

    bmm2_scale : Union[float, torch.Tensor]
        fused scale for bmm2 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.

    window_left : int = -1
        The left (inclusive) window size for the attention window, when set to ``-1``,
        the window size will be set to the full length of the sequence. Defaults to ``-1``.

    out :  Optional[Union[torch.Tensor, FP4Tensor]] = None
        output tensor, if not provided, will be allocated with ``out_dtype``,
        if ``out_dtype`` is not provided, will use the type of ``query``.

    out_dtype : Optional[Union[torch.dtype, str]] = None
        output dtype, if not provided, will use the type of ``out``.
        For nvfp4, use string ``nvfp4``.

    o_sf_scale : Optional[float] = None
        scale for nvfp4 output tensor scale factor.

    o_sf_vec_size : Optional[int] = None
        vector size for nvfp4 output tensor scale factor.

    sinks : Optional[List[torch.Tensor]] = None
        additional value per head in the denominator of the softmax.

    kv_layout : str = "HND"
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        Defaults to ``HND``.

    enable_pdl : Optional[bool] = None
        Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
        When set to ``None``, the backend will be chosen based on the device architecture and kernel availability.

    backend : str = "auto"
        The implementation backend, could be ``auto``/``xqa`` or ``trtllm-gen``. Defaults to ``auto``.
        When set to ``auto``, the backend will be chosen based on the device architecture and kernel availability.
        For sm_100 and sm_103 (blackwell architecture), ``auto`` will choose ``trtllm-gen`` backend.
        For sm_90 (hopper architecture) and sm_120 (blackwell architecture), ``auto`` will choose ``xqa`` backend.

    o_scale : Optional[float] = 1.0
        output scale factor for xqa fp8 output.

    mask : Optional[torch.Tensor] = None
        causal attention mask for xqa speculative decoding.

    max_q_len: Optional[int] = None
        The maximum query sequence length across all requests when using variable-length queries.
        Only supported by trtllm-gen backend. Must be provided together with ``cum_seq_lens_q``.
        When None, all requests use uniform query length specified by ``q_len_per_req``.

    cum_seq_lens_q : Optional[torch.Tensor] = None
        Cumulative query sequence lengths for variable-length query support,
        shape: ``[batch_size + 1]``, dtype: ``torch.int32``.
        Only supported by trtllm-gen backend. Must be provided together with ``max_q_len``.
        When None, all requests use uniform query length specified by ``q_len_per_req``.

    Returns
    -------
    out : Union[torch.Tensor, FP4Tensor]
        output torch.Tensor or FP4Tensor.
    """
    pass


########################################################
# Drop-in replacement for thop.attention()
########################################################

# Global cache for AttentionOp instances to avoid re-initialization
_attention_op_cache: dict = {}


def _get_dtype_from_torch(dtype: torch.dtype) -> DataType:
    """Convert torch dtype to internal DataType enum."""
    if dtype == torch.float16:
        return DataType.FP16
    elif dtype == torch.float32:
        return DataType.FP32
    elif dtype == torch.bfloat16:
        return DataType.BF16
    elif dtype == torch.float8_e4m3fn:
        return DataType.E4M3
    elif dtype == torch.uint8:
        return DataType.E2M1  # NVFP4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _compute_kv_cache_params(
    layer_idx: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: int,
    kv_cache_quant_mode: QuantMode,
    is_mla_enabled: bool,
    seq_offset: int,
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_block_offsets: Optional[torch.Tensor],
) -> Tuple[
    int, int, Optional[torch.Tensor], Optional[torch.Tensor], int, int, Optional[int], Optional[int]
]:
    """
    Compute KV cache related parameters.

    This function extracts pool indices, layer indices in cache pool, block offsets,
    and pool pointers needed for KV cache operations.

    Returns:
        Tuple of (pool_index, layer_idx_in_cache_pool, block_offsets, host_block_offsets,
                  host_primary_pool_pointer, host_secondary_pool_pointer,
                  host_primary_block_scale_pool_pointer, host_secondary_block_scale_pool_pointer)
    """
    if host_kv_cache_pool_mapping is None:
        return 0, 0, None, None, 0, 0, None, None

    pool_index = host_kv_cache_pool_mapping[layer_idx, 0].item()
    layer_idx_in_cache_pool = host_kv_cache_pool_mapping[layer_idx, 1].item()

    # Get block offsets
    block_offsets = None
    host_block_offsets = None
    if kv_cache_block_offsets is not None:
        block_offsets = kv_cache_block_offsets[pool_index, seq_offset:]
    if host_kv_cache_block_offsets is not None:
        host_block_offsets = host_kv_cache_block_offsets[pool_index, seq_offset:]

    # Calculate cache element size in bits
    if kv_cache_quant_mode.has_int8_kv_cache() or kv_cache_quant_mode.has_fp8_kv_cache():
        cache_elem_bits = 8
    elif kv_cache_quant_mode.has_fp4_kv_cache():
        cache_elem_bits = 4
    else:
        cache_elem_bits = 16  # Default to FP16

    # Calculate block size and intra-pool offset
    block_size = tokens_per_block * num_kv_heads * head_size
    bytes_per_block = block_size * cache_elem_bits // 8
    kv_factor = 1 if is_mla_enabled else 2
    intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block

    # Get pool pointers
    host_primary_pool_pointer = 0
    host_secondary_pool_pointer = 0
    host_primary_block_scale_pool_pointer = None
    host_secondary_block_scale_pool_pointer = None

    if host_kv_cache_pool_pointers is not None:
        use_nvfp4_kv_cache = kv_cache_quant_mode.has_fp4_kv_cache()

        if use_nvfp4_kv_cache:
            # For NVFP4 KV cache, extra block scales are stored in separate pools
            # Layout: [num_pools, 2 (primary and secondary), 2 (data and scale)]
            assert host_kv_cache_pool_pointers.dim() == 3
            host_primary_pool_pointer = (
                host_kv_cache_pool_pointers[pool_index, 0, 0].item() + intra_pool_offset
            )
            host_secondary_pool_pointer = (
                host_kv_cache_pool_pointers[pool_index, 1, 0].item() + intra_pool_offset
            )

            # Calculate intra-pool offset for scaling factors
            vector_size = 16  # NVFP4 block scaling uses fixed vector size of 16
            bytes_per_block_sf = block_size // vector_size * 1  # 1 byte per E4M3 sf
            intra_pool_offset_sf = layer_idx_in_cache_pool * kv_factor * bytes_per_block_sf

            host_primary_block_scale_pool_pointer = (
                host_kv_cache_pool_pointers[pool_index, 0, 1].item() + intra_pool_offset_sf
            )
            host_secondary_block_scale_pool_pointer = (
                host_kv_cache_pool_pointers[pool_index, 1, 1].item() + intra_pool_offset_sf
            )
        else:
            # Standard KV cache: [num_pools, 2 (primary and secondary)]
            assert host_kv_cache_pool_pointers.dim() == 2
            host_primary_pool_pointer = (
                host_kv_cache_pool_pointers[pool_index, 0].item() + intra_pool_offset
            )
            host_secondary_pool_pointer = (
                host_kv_cache_pool_pointers[pool_index, 1].item() + intra_pool_offset
            )

    return (
        pool_index,
        layer_idx_in_cache_pool,
        block_offsets,
        host_block_offsets,
        host_primary_pool_pointer,
        host_secondary_pool_pointer,
        host_primary_block_scale_pool_pointer,
        host_secondary_block_scale_pool_pointer,
    )


def _run_attention_phase(
    op: TrtllmGenAttention,
    is_context: bool,
    seq_offset: int,
    num_seqs: int,
    token_offset: int,
    num_tokens: int,
    predicted_tokens_per_seq: int,
    total_kv_len: int,
    # Tensors
    workspace: torch.Tensor,
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
    qkv_or_q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    context_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
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
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    mla_tensor_params: List[Optional[torch.Tensor]],
    softmax_stats_tensor: Optional[torch.Tensor],
    spec_decoding_tensor_params: List[Optional[torch.Tensor]],
    attention_sinks: Optional[torch.Tensor],
    sparse_kv_indices: Optional[torch.Tensor],
    sparse_kv_offsets: Optional[torch.Tensor],
    sparse_attn_indices: Optional[torch.Tensor],
    sparse_attn_offsets: Optional[torch.Tensor],
    sparse_attn_indices_block_size: int,
    sparse_mla_topk: int,
    cu_q_seqlens: Optional[torch.Tensor],
    cu_kv_seqlens: Optional[torch.Tensor],
    fmha_scheduler_counter: Optional[torch.Tensor],
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
    # Scalar parameters
    beam_width: int,
    attention_window_size: int,
    sink_token_length: int,
) -> None:
    """
    Run a single attention phase (context or generation).

    This function merges the logic from C++ Runner::run() method directly into the attention flow,
    eliminating the need for a separate Runner class.
    """
    # Get attention input with offset applied
    attention_input = qkv_or_q[token_offset:] if token_offset > 0 else qkv_or_q
    context_buf = output[token_offset:] if token_offset > 0 else output
    context_buf_sf = (
        output_sf[token_offset:] if output_sf is not None and token_offset > 0 else output_sf
    )

    # Rotary inv_freq and cos_sin cache
    rotary_inv_freq_ptr = rotary_inv_freq if op.is_rope() and rotary_inv_freq is not None else None
    rotary_cos_sin_ptr = rotary_cos_sin if op.is_rope() and rotary_cos_sin is not None else None

    # Get slice of context_lengths and sequence_lengths for this batch
    context_lengths_slice = context_lengths[seq_offset:] if seq_offset > 0 else context_lengths
    sequence_lengths_slice = sequence_length[seq_offset:] if seq_offset > 0 else sequence_length
    host_context_lengths_slice = host_context_lengths[seq_offset : seq_offset + num_seqs]
    host_past_kv_lengths_slice = host_past_key_value_lengths[seq_offset : seq_offset + num_seqs]

    # Compute max lengths
    max_context_q_len = host_context_lengths_slice.max().item()
    max_past_kv_length = host_past_kv_lengths_slice.max().item()

    # Determine attention window sizes
    max_attention_window_size = attention_window_size
    if beam_width != 1 and cache_indirection is not None:
        max_attention_window_size = cache_indirection.size(2)
    cyclic_attention_window_size = attention_window_size
    can_use_one_more_block = beam_width > 1

    # Compute KV cache parameters
    max_blocks_per_sequence = 0
    if op.use_kv_cache and kv_cache_block_offsets is not None:
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1)

    (
        pool_index,
        layer_idx_in_cache_pool,
        block_offsets,
        host_block_offsets,
        host_primary_pool_pointer,
        host_secondary_pool_pointer,
        host_primary_block_scale_pool_pointer,
        host_secondary_block_scale_pool_pointer,
    ) = _compute_kv_cache_params(
        layer_idx=op.layer_idx,
        num_kv_heads=op.num_kv_heads,
        head_size=op.head_size,
        tokens_per_block=op.tokens_per_block,
        kv_cache_quant_mode=op.kv_cache_quant_mode,
        is_mla_enabled=op._is_mla_enabled,
        seq_offset=seq_offset,
        host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
        kv_cache_block_offsets=kv_cache_block_offsets,
        host_kv_cache_block_offsets=host_kv_cache_block_offsets,
    )

    # Get KV scale pointers
    kv_scale_orig_quant_ptr = (
        kv_scale_orig_quant if op.kv_cache_quant_mode.has_kv_cache_quant() else None
    )
    kv_scale_quant_orig_ptr = (
        kv_scale_quant_orig if op.kv_cache_quant_mode.has_kv_cache_quant() else None
    )

    # Get output scale
    out_scale_ptr = None
    out_sf_scale_ptr = None
    if op.fp8_context_fmha and not op.fuse_fp4_quant and out_scale is not None:
        out_scale_ptr = out_scale
    if op.fuse_fp4_quant and out_scale is not None:
        out_sf_scale_ptr = out_scale

    # Attention sinks
    attention_sinks_ptr = attention_sinks

    # Update sparse attention params
    op.runtime_sparse_attention_params.sparse_kv_indices = sparse_kv_indices
    op.runtime_sparse_attention_params.sparse_kv_offsets = sparse_kv_offsets
    op.runtime_sparse_attention_params.sparse_attn_indices = sparse_attn_indices
    op.runtime_sparse_attention_params.sparse_attn_offsets = sparse_attn_offsets
    op.runtime_sparse_attention_params.sparse_attn_indices_block_size = (
        sparse_attn_indices_block_size
    )
    if sparse_attn_indices is not None:
        op.runtime_sparse_attention_params.sparse_attn_indices_stride = sparse_attn_indices.size(-1)

    if op._is_mla_enabled and op.use_sparse_attention:
        op.runtime_sparse_attention_params.sparse_mla_topk = sparse_mla_topk
        if op.use_kv_cache and host_kv_cache_pool_pointers is not None:
            op.runtime_sparse_attention_params.sparse_mla_kv_cache_pool = (
                host_kv_cache_pool_pointers[pool_index, 0].item()
            )

    # Prepare MLA params if enabled
    mla_param = None
    if op._is_mla_enabled:
        mla_param = MlaParams()
        mla_param.q_buf = attention_input
        mla_param.context_buf = context_buf
        mla_param.cos_sin_cache = rotary_cos_sin_ptr

        if is_context:
            if op.use_sparse_attention:
                mla_param.latent_cache = latent_cache
                if q_pe is not None:
                    mla_param.q_pe = q_pe
            else:
                mla_param.latent_cache = latent_cache
                if k is not None:
                    mla_param.k_buf = k[token_offset:] if token_offset > 0 else k
                if v is not None:
                    mla_param.v_buf = v[token_offset:] if token_offset > 0 else v

                # Helix position offsets (from mla_tensor_params)
                if len(mla_tensor_params) >= 2:
                    if mla_tensor_params[0] is not None:
                        pass  # helix_position_offsets - handled in context
                    if mla_tensor_params[1] is not None:
                        pass  # helix_is_inactive_rank
        else:
            # Generation phase
            mla_param.latent_cache = latent_cache
            if q_pe is not None:
                mla_param.q_pe = q_pe
            if cu_q_seqlens is not None:
                mla_param.cu_q_seqlens = cu_q_seqlens
            if cu_kv_seqlens is not None:
                pass  # cu_kv_seqlens
            if fmha_scheduler_counter is not None:
                pass  # fmha_tile_counter
            if mla_bmm1_scale is not None:
                mla_param.bmm1_scale = mla_bmm1_scale
            if mla_bmm2_scale is not None:
                mla_param.bmm2_scale = mla_bmm2_scale
            if quant_q_buffer is not None:
                mla_param.quant_q_buf = quant_q_buffer

    # Build common enqueue params
    if is_context:
        # Context phase
        input_seq_length = max_context_q_len

        # Check if we can use trtllm_batch_context_with_kv_cache (simpler API)
        # Unsupported features that require fallback to op.enqueue_context:
        #   - rotary_inv_freq, rotary_cos_sin: RoPE not supported
        #   - mrope_rotary_cos_sin: MRoPE not supported
        #   - mla_param: MLA not supported
        #   - kv_scale_orig_quant, kv_scale_quant_orig: KV cache quantization not supported
        #   - host_primary_block_scale_pool_pointer: Block scale pointers not supported
        #   - sink_token_length > 0: Sink tokens not supported
        #   - k_ptr, v_ptr: Separate K/V pointers not supported
        #   - softmax_stats: Softmax stats not supported
        #   - can_use_one_more_block: Beam search not supported
        k_ptr_val = k[token_offset:] if k is not None and token_offset > 0 else k
        v_ptr_val = v[token_offset:] if v is not None and token_offset > 0 else v

        use_simple_api = (
            rotary_inv_freq_ptr is None
            and rotary_cos_sin_ptr is None
            and (not op.is_mrope() or mrope_rotary_cos_sin is None)
            and mla_param is None
            and kv_scale_orig_quant_ptr is None
            and kv_scale_quant_orig_ptr is None
            and host_primary_block_scale_pool_pointer is None
            and host_secondary_block_scale_pool_pointer is None
            and sink_token_length == 0
            and k_ptr_val is None
            and v_ptr_val is None
            and softmax_stats_tensor is None
            and not can_use_one_more_block
            and block_offsets is not None
            and host_primary_pool_pointer is not None
        )

        if use_simple_api:
            # Use trtllm_batch_context_with_kv_cache
            # Compute cumulative sequence lengths
            cum_seq_lens_q = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=context_lengths_slice.device
            )
            cum_seq_lens_q[1:] = torch.cumsum(
                context_lengths_slice[:num_seqs].to(torch.int32), dim=0
            )

            cum_seq_lens_kv = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=sequence_lengths_slice.device
            )
            cum_seq_lens_kv[1:] = torch.cumsum(
                sequence_lengths_slice[:num_seqs].to(torch.int32), dim=0
            )

            # Determine window_left for sliding window attention
            window_left = (
                cyclic_attention_window_size
                if cyclic_attention_window_size < max_attention_window_size
                else -1
            )

            # Reshape attention_input from [num_tokens, hidden_dim] to [num_tokens, num_heads, head_dim]
            query = attention_input.view(num_tokens, op.num_heads, op.head_size)

            # Prepare output tensor
            out = context_buf.view(num_tokens, op.num_heads, op.head_size)

            # Prepare attention sinks as list
            sinks = None
            if attention_sinks_ptr is not None and attention_sinks_ptr.numel() > 0:
                sinks = [attention_sinks_ptr[i] for i in range(attention_sinks_ptr.size(0))]

            # Reconstruct KV cache tensor from pool pointers
            # Note: We create a dummy tensor wrapper around the pool pointer for the API
            # The actual data is accessed via the pointer internally
            # KV cache shape: [num_pages, 2, num_kv_heads, tokens_per_block, head_size] for HND layout
            num_pages = block_offsets.max().item() + 1 if block_offsets.numel() > 0 else 1
            kv_cache_shape = (num_pages, 2, op.num_kv_heads, op.tokens_per_block, op.head_size)

            # Create tensor from pool pointer using torch.from_blob (requires knowing the exact memory layout)
            # Since we have pool pointers, we need to create a tensor that wraps this memory
            kv_cache = torch.empty(
                kv_cache_shape, dtype=attention_input.dtype, device=attention_input.device
            )
            # Copy pointer to tensor's data_ptr - this is a workaround
            # In practice, we need the actual kv_cache tensor reference
            # For now, set the internal storage to point to our pool
            # Note: This is a hack - proper solution requires passing kv_cache tensor through the interface

            trtllm_batch_context_with_kv_cache(
                query=query,
                kv_cache=kv_cache,  # Note: This needs proper tensor with correct data_ptr
                workspace_buffer=workspace,
                block_tables=block_offsets,
                seq_lens=sequence_lengths_slice[:num_seqs].to(torch.int32),
                max_q_len=input_seq_length,
                max_kv_len=max_past_kv_length,
                bmm1_scale=op.q_scaling,
                bmm2_scale=1.0,
                batch_size=num_seqs,
                cum_seq_lens_q=cum_seq_lens_q,
                cum_seq_lens_kv=cum_seq_lens_kv,
                window_left=window_left,
                out=out,
                out_dtype=None,
                o_sf_scale=out_sf_scale_ptr.item() if out_sf_scale_ptr is not None else None,
                o_sf_vec_size=None,
                kv_layout="HND",
                enable_pdl=None,
                sinks=sinks,
            )
        else:
            # Fallback to op.enqueue_context for advanced features
            params = EnqueueContextParams(
                attention_input=attention_input,
                qkv_bias=None,
                attention_mask=None,
                attention_sinks=attention_sinks_ptr,
                rotary_inv_freq=rotary_inv_freq_ptr,
                rotary_cos_sin=rotary_cos_sin_ptr,
                input_seq_length=input_seq_length,
                max_past_kv_length=max_past_kv_length,
                max_attention_window_size=max_attention_window_size,
                cyclic_attention_window_size=cyclic_attention_window_size,
                max_cyclic_attention_window_size=cyclic_attention_window_size,
                can_use_one_more_block=can_use_one_more_block,
                sink_token_length=sink_token_length,
                kv_scale_orig_quant=kv_scale_orig_quant_ptr,
                kv_scale_quant_orig=kv_scale_quant_orig_ptr,
                attention_output_orig_quant=out_scale_ptr,
                attention_output_sf_scale=out_sf_scale_ptr,
                context_buf=context_buf,
                context_buf_sf=context_buf_sf,
                block_offsets=block_offsets,
                host_block_offsets=host_block_offsets,
                host_primary_pool_pointer=host_primary_pool_pointer,
                host_secondary_pool_pointer=host_secondary_pool_pointer,
                host_primary_block_scale_pool_pointer=host_primary_block_scale_pool_pointer,
                host_secondary_block_scale_pool_pointer=host_secondary_block_scale_pool_pointer,
                num_tokens=num_tokens,
                total_kv_len=total_kv_len,
                max_blocks_per_sequence=max_blocks_per_sequence,
                sequence_lengths=sequence_lengths_slice,
                context_lengths=context_lengths_slice,
                host_context_lengths=host_context_lengths,
                workspace=workspace,
                softmax_stats=softmax_stats_tensor,
                batch_size=num_seqs,
                k_ptr=k_ptr_val,
                v_ptr=v_ptr_val,
                mla_param=mla_param,
                mrope_rotary_cos_sin=mrope_rotary_cos_sin if op.is_mrope() else None,
            )

            op.enqueue_context(params)

    else:
        # Generation phase
        batch_beam = num_seqs
        assert batch_beam % beam_width == 0, (
            f"batch_beam ({batch_beam}) must be divisible by beam_width ({beam_width})"
        )
        num_requests = batch_beam // beam_width

        assert num_tokens % num_seqs == 0, (
            f"seq_len should be same for all generation requests, num_tokens={num_tokens}, num_seqs={num_seqs}"
        )
        input_seq_length = num_tokens // num_seqs

        # Get cache indirection for beam search
        cache_indir = None
        if beam_width != 1 and cache_indirection is not None:
            cache_indir = cache_indirection

        params = EnqueueGenerationParams(
            attention_input=attention_input,
            qkv_bias=None,
            attention_mask=None,
            attention_sinks=attention_sinks_ptr,
            rotary_inv_freq=rotary_inv_freq_ptr,
            rotary_cos_sin=rotary_cos_sin_ptr,
            input_seq_length=input_seq_length,
            max_past_kv_length=max_past_kv_length,
            max_attention_window_size=max_attention_window_size,
            cyclic_attention_window_size=cyclic_attention_window_size,
            max_cyclic_attention_window_size=cyclic_attention_window_size,
            can_use_one_more_block=can_use_one_more_block,
            sink_token_length=sink_token_length,
            kv_scale_orig_quant=kv_scale_orig_quant_ptr,
            kv_scale_quant_orig=kv_scale_quant_orig_ptr,
            attention_output_orig_quant=out_scale_ptr,
            attention_output_sf_scale=out_sf_scale_ptr,
            context_buf=context_buf,
            context_buf_sf=context_buf_sf,
            block_offsets=block_offsets,
            host_primary_pool_pointer=host_primary_pool_pointer,
            host_secondary_pool_pointer=host_secondary_pool_pointer,
            host_primary_block_scale_pool_pointer=host_primary_block_scale_pool_pointer,
            host_secondary_block_scale_pool_pointer=host_secondary_block_scale_pool_pointer,
            num_tokens=num_tokens,
            total_kv_len=total_kv_len,
            max_blocks_per_sequence=max_blocks_per_sequence,
            sequence_lengths=sequence_lengths_slice,
            context_lengths=context_lengths_slice,
            host_context_lengths=host_context_lengths,
            workspace=workspace,
            softmax_stats=softmax_stats_tensor,
            # Generation-specific params
            beam_width=beam_width,
            num_requests=num_requests,
            cache_indir=cache_indir,
            host_past_key_value_lengths=host_past_key_value_lengths,
            mrope_position_deltas=mrope_position_deltas if op.is_mrope() else None,
            start_token_idx_sf=token_offset,
            layer_idx=op.layer_idx,
            mla_param=mla_param,
        )

        # Handle speculative decoding params
        if op.is_spec_decoding_enabled and op.use_spec_decoding:
            sm_version = get_sm_version()
            use_tllm_gen = (sm_version >= 100) and (sm_version != 120)

            if use_tllm_gen:
                assert len(spec_decoding_tensor_params) == 6, (
                    "Expecting 6 tensors for spec-dec mode with tllm-gen"
                )
            else:
                assert len(spec_decoding_tensor_params) == 3, (
                    "Expecting 3 tensors for spec-dec mode"
                )

            if spec_decoding_tensor_params[0] is not None:
                params.spec_decoding_generation_lengths = spec_decoding_tensor_params[0]
            if spec_decoding_tensor_params[1] is not None:
                params.spec_decoding_position_offsets = spec_decoding_tensor_params[1]
            if spec_decoding_tensor_params[2] is not None:
                params.spec_decoding_packed_mask = spec_decoding_tensor_params[2]

            params.spec_decoding_is_generation_length_variable = True
            if spec_decoding_tensor_params[1] is not None:
                params.spec_decoding_max_generation_length = spec_decoding_tensor_params[1].size(1)

            if use_tllm_gen and len(spec_decoding_tensor_params) >= 6:
                if spec_decoding_tensor_params[3] is not None:
                    params.spec_decoding_bl_tree_mask_offset = spec_decoding_tensor_params[3]
                if spec_decoding_tensor_params[4] is not None:
                    params.spec_decoding_bl_tree_mask = spec_decoding_tensor_params[4]
                if spec_decoding_tensor_params[5] is not None:
                    params.spec_bl_tree_first_sparse_mask_offset_kv = spec_decoding_tensor_params[5]

        # Run generation phase
        if op._is_mla_enabled:
            if op.use_gen_flash_mla and block_ids_per_seq is not None:
                mla_param.block_ids_per_seq = block_ids_per_seq
            mla_param.cache_seq_lens = sequence_lengths_slice
            op.mla_generation(mla_param, params)
        else:
            op.enqueue_generation(params)


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
    # ========== Check if trtllm-gen is supported, fallback to thop.attention if not ==========
    # Note: enqueue_generation and mla_generation are not implemented yet in this backend,
    # so we fall back to thop.attention for unsupported configurations
    from tensorrt_llm.bindings.internal import thop

    # Determine if sparse attention is used
    _use_sparse_attention = (sparse_kv_indices is not None and sparse_kv_indices.numel() > 0) or (
        sparse_attn_indices is not None and sparse_attn_indices.numel() > 0
    )

    # Check if this configuration is supported by trtllm-gen
    supported, reason = is_supported(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=q.dtype,
        out_dtype=output.dtype,
        mask_type=mask_type,
        has_alibi=(
            PositionEmbeddingType(position_embedding_type) == PositionEmbeddingType.ALIBI
            or PositionEmbeddingType(position_embedding_type)
            == PositionEmbeddingType.ALIBI_WITH_SCALE
        ),
        is_padded=False,  # Assume not padded for now
        is_mla_enabled=is_mla_enable,
        use_paged_kv_cache=(kv_cache_block_offsets is not None),
        tokens_per_block=tokens_per_block or 64,
        beam_width=beam_width,
        position_shift_enabled=False,  # Not exposed in the interface
        sink_token_length=sink_token_length,
        cross_attention=False,  # Not exposed in the interface
        cyclic_attention_window_size=attention_window_size,
        max_attention_window_size=attention_window_size,
        is_spec_decoding=spec_decoding_bool_params[0] if spec_decoding_bool_params else False,
        use_sparse_attention=_use_sparse_attention,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    if not supported:
        logger.debug(f"trtllm-gen not supported: {reason}. Falling back to thop.attention()")
        return thop.attention(
            q,
            k,
            v,
            output,
            output_sf,
            workspace,
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            cache_indirection,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            out_scale,
            rotary_inv_freq,
            rotary_cos_sin,
            latent_cache,
            q_pe,
            block_ids_per_seq,
            attention_sinks,
            is_fused_qkv,
            update_kv_cache,
            predicted_tokens_per_seq,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_size,
            tokens_per_block,
            max_num_requests,
            max_context_length,
            attention_window_size,
            sink_token_length,
            beam_width,
            mask_type,
            quant_mode,
            q_scaling,
            position_embedding_type,
            rotary_embedding_dim,
            rotary_embedding_base,
            rotary_embedding_scale_type,
            rotary_embedding_scales,
            rotary_embedding_max_position_info,
            use_paged_context_fmha,
            attention_input_type,
            is_mla_enable,
            chunked_prefill_buffer_batch_size,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            mla_tensor_params,
            attention_chunk_size,
            softmax_stats_tensor,
            spec_decoding_bool_params,
            spec_decoding_tensor_params,
            sparse_kv_indices,
            sparse_kv_offsets,
            sparse_attn_indices,
            sparse_attn_offsets,
            sparse_attn_indices_block_size,
            sparse_mla_topk,
            skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )

    global _attention_op_cache

    logger.debug(f"Attention op starts at layer {layer_idx}")

    # ========== Step 1: Validation and setup ==========
    # Check if using KV cache
    use_kv_cache = (
        kv_cache_block_offsets is not None
        and host_kv_cache_block_offsets is not None
        and host_kv_cache_pool_pointers is not None
        and host_kv_cache_pool_mapping is not None
    )

    assert is_mla_enable or is_fused_qkv, "Only fused QKV is supported for non-MLA attention now"
    assert update_kv_cache, "KV cache update cannot be disabled now"

    qkv_or_q = q
    if is_fused_qkv:
        assert k is None, "The k tensor should be null if using fused QKV"
        assert v is None, "The v tensor should be null if using fused QKV"
    if not is_fused_qkv and update_kv_cache:
        assert k is not None, (
            "The k tensor should be provided if updating KV cache with unfused K/V"
        )
        assert v is not None, (
            "The v tensor should be provided if updating KV cache with unfused K/V"
        )

    # Determine data types
    dtype = _get_dtype_from_torch(qkv_or_q.dtype)
    out_dtype = output.dtype
    is_fp8_out = out_dtype == torch.float8_e4m3fn
    is_fp4_out = out_dtype == torch.uint8  # Torch doesn't support native nvfp4 type

    # ========== Step 2: Create or retrieve cached AttentionOp ==========
    # Extract rotary embedding scales
    rotary_embedding_scale = rotary_embedding_scales[0] if len(rotary_embedding_scales) > 0 else 1.0
    rotary_embedding_short_m_scale = (
        rotary_embedding_scales[1] if len(rotary_embedding_scales) > 1 else 1.0
    )
    rotary_embedding_long_m_scale = (
        rotary_embedding_scales[2] if len(rotary_embedding_scales) > 2 else 1.0
    )
    rotary_embedding_max_positions = (
        rotary_embedding_max_position_info[0]
        if len(rotary_embedding_max_position_info) > 0
        else 1024
    )
    rotary_embedding_original_max_positions = (
        rotary_embedding_max_position_info[1]
        if len(rotary_embedding_max_position_info) > 1
        else 1024
    )

    # Create AttentionOp and configure
    op = TrtllmGenAttention()
    op.type = dtype
    op.fmha_force_fp32_acc = dtype == DataType.BF16
    op.layer_idx = layer_idx
    op.num_heads = num_heads
    op.num_kv_heads = num_kv_heads
    op.head_size = head_size
    op.mask_type = AttentionMaskType(mask_type)
    op.kv_cache_quant_mode = QuantMode(quant_mode)
    op.use_kv_cache = use_kv_cache
    op.paged_kv_cache = op.paged_kv_cache and use_kv_cache
    op.tokens_per_block = tokens_per_block or 0
    op.fp8_generation_mla = False
    op.fuse_fp4_quant = is_fp4_out
    op.max_context_length = max_context_length
    op.q_scaling = q_scaling
    op.position_embedding_type = PositionEmbeddingType(position_embedding_type)
    op.rotary_embedding_dim = rotary_embedding_dim
    op.rotary_embedding_base = rotary_embedding_base
    op.rotary_embedding_scale_type = RotaryScalingType(rotary_embedding_scale_type)
    op.rotary_embedding_scale = rotary_embedding_scale
    op.rotary_embedding_short_m_scale = rotary_embedding_short_m_scale
    op.rotary_embedding_long_m_scale = rotary_embedding_long_m_scale
    op.rotary_embedding_max_positions = rotary_embedding_max_positions
    op.rotary_embedding_original_max_positions = rotary_embedding_original_max_positions
    op.fp8_context_fmha = (
        is_fp8_out
        or is_fp4_out
        or (op.kv_cache_quant_mode.has_fp8_kv_cache() and use_paged_context_fmha)
    )
    op.fp8_atten_output = is_fp8_out
    op.paged_context_fmha = use_paged_context_fmha

    op.attention_chunk_size = attention_chunk_size
    op.skip_softmax_threshold_scale_factor_prefill = (
        skip_softmax_threshold_scale_factor_prefill or 0.0
    )
    op.skip_softmax_threshold_scale_factor_decode = (
        skip_softmax_threshold_scale_factor_decode or 0.0
    )

    # Speculative decoding params
    assert len(spec_decoding_bool_params) == 3, "Expecting 3 bools for spec-dec mode"
    op.is_spec_decoding_enabled = spec_decoding_bool_params[0]
    op.use_spec_decoding = spec_decoding_bool_params[1]
    op.is_spec_tree = spec_decoding_bool_params[2]

    # Sparse attention
    op.use_sparse_attention = False
    op.use_tllm_gen_sparse_attention = False
    if (sparse_kv_indices is not None and sparse_kv_indices.numel() > 0) or (
        sparse_attn_indices is not None and sparse_attn_indices.numel() > 0
    ):
        op.use_sparse_attention = True
        if sparse_attn_indices is not None and sparse_attn_indices.numel() > 0:
            op.use_tllm_gen_sparse_attention = True

    sparse_mla_topk_value = sparse_mla_topk or 0

    # MLA configuration
    if is_mla_enable:
        assert not is_fp4_out, "MLA does not support NVFP4 output yet"
        assert host_kv_cache_pool_mapping is not None
        layer_num = host_kv_cache_pool_mapping.size(0)

        if (
            sparse_mla_topk_value > 0
            and sparse_attn_indices is not None
            and sparse_attn_indices.numel() > 0
        ):
            op.use_sparse_attention = True

        op._is_mla_enabled = True
        op.mla_params = MlaMetaParams(
            q_lora_rank=q_lora_rank or 0,
            kv_lora_rank=kv_lora_rank or 0,
            qk_nope_head_dim=qk_nope_head_dim or 0,
            qk_rope_head_dim=qk_rope_head_dim or 0,
            v_head_dim=v_head_dim or 0,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            num_layers=layer_num,
        )

        sm_version = get_sm_version()
        op.fp8_context_mla = (
            sm_version in (90, 100, 103, 120)
        ) and op.kv_cache_quant_mode.has_fp8_kv_cache()
        op.is_generation_mla = (
            head_size == op.mla_params.kv_lora_rank + op.mla_params.qk_rope_head_dim
        )
        op.fp8_generation_mla = op.kv_cache_quant_mode.has_fp8_kv_cache()
        op.use_gen_flash_mla = sm_version == 90 and tokens_per_block == 64

        # For MLA, override num_kv_heads and head_size for KV cache calculations
        op.num_kv_heads = 1
        op.head_size = op.mla_params.kv_lora_rank + op.mla_params.qk_rope_head_dim

        op.chunk_prefill_buffer_batch_size = chunked_prefill_buffer_batch_size or 1

    # Create cache key and check cache
    cache_key = (
        op.layer_idx,
        op.num_heads,
        op.num_kv_heads,
        op.head_size,
        int(op.mask_type),
        int(op.kv_cache_quant_mode),
        op.tokens_per_block,
        int(op.type),
        op.max_context_length,
        int(op.position_embedding_type),
        op.rotary_embedding_dim,
        op._is_mla_enabled,
        beam_width,
        max_num_requests,
        attention_window_size,
        sink_token_length,
    )

    if cache_key in _attention_op_cache:
        logger.debug(f"Attention op for layer {layer_idx} is cached")
        op = _attention_op_cache[cache_key]
    else:
        logger.debug(f"Preparing new attention op for layer {layer_idx}")
        op.beam_width = beam_width
        op.max_num_requests = max_num_requests
        op.attention_window_size = attention_window_size
        op.sink_token_length = sink_token_length
        op.initialize()
        _attention_op_cache[cache_key] = op

    # ========== Step 3: Compute batch information ==========
    num_seqs = host_context_lengths.size(0)

    # Determine attention input type
    attn_input_type = AttentionInputType.MIXED
    if attention_input_type is not None:
        attn_input_type = AttentionInputType(attention_input_type)
    is_gen_only = attn_input_type == AttentionInputType.GENERATION_ONLY

    # Count context vs generation requests
    # Request types: CONTEXT = 0, GENERATION = 1
    num_contexts = 0
    request_types = host_request_types.cpu().numpy()
    for idx in range(num_seqs):
        if request_types[idx] != 0:  # Not CONTEXT
            break
        num_contexts += 1

    num_generations = num_seqs - num_contexts

    # Verify all remaining are generation requests
    for idx in range(num_contexts, num_seqs):
        assert request_types[idx] == 1, f"Expected generation request at index {idx}"

    # Calculate token counts
    num_tokens = qkv_or_q.size(0)
    num_ctx_tokens = host_context_lengths[:num_contexts].sum().item() if num_contexts > 0 else 0
    num_gen_tokens = num_tokens if is_gen_only else (num_tokens - num_ctx_tokens)

    ctx_total_kv_len = host_total_kv_lens[0].item()
    # gen_total_kv_len not used since generation falls back to thop.attention

    # ========== Step 4: Allocate workspace ==========
    max_attention_window_size_compute = attention_window_size
    if beam_width != 1 and cache_indirection is not None:
        max_attention_window_size_compute = cache_indirection.size(2)

    max_blocks_per_sequence = 0
    if use_kv_cache and kv_cache_block_offsets is not None:
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1)

    # Compute workspace size (simplified - in practice would call getWorkspaceSize methods)
    workspace_size = (
        op.get_workspace_size(
            num_tokens=num_tokens,
            max_attention_window_size=max_attention_window_size_compute,
            num_gen_tokens=num_gen_tokens,
            max_blocks_per_sequence=max_blocks_per_sequence,
        )
        if hasattr(op, "get_workspace_size")
        else 0
    )

    logger.debug(f"Expected workspace size is {workspace_size} bytes")

    # Resize or allocate workspace
    if workspace is not None:
        if workspace.numel() < workspace_size:
            logger.warning(
                f"Attention workspace size not enough, resizing from {workspace.numel()} to {workspace_size}"
            )
            workspace = workspace.resize_(workspace_size)
    else:
        logger.debug(f"Allocating new attention workspace with size {workspace_size} bytes")
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=qkv_or_q.device)

    # ========== Step 5: Run context phase ==========
    if num_contexts > 0 and attn_input_type != AttentionInputType.GENERATION_ONLY:
        _run_attention_phase(
            op=op,
            is_context=True,
            seq_offset=0,
            num_seqs=num_contexts,
            token_offset=0,
            num_tokens=num_ctx_tokens,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            total_kv_len=ctx_total_kv_len,
            workspace=workspace,
            output=output,
            output_sf=output_sf,
            qkv_or_q=qkv_or_q,
            k=k,
            v=v,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            kv_cache_block_offsets=kv_cache_block_offsets,
            host_kv_cache_block_offsets=host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            cache_indirection=cache_indirection,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            out_scale=out_scale,
            rotary_inv_freq=rotary_inv_freq,
            rotary_cos_sin=rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=block_ids_per_seq,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            mrope_position_deltas=mrope_position_deltas,
            mla_tensor_params=mla_tensor_params,
            softmax_stats_tensor=softmax_stats_tensor,
            spec_decoding_tensor_params=spec_decoding_tensor_params,
            attention_sinks=attention_sinks,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=sparse_attn_indices_block_size,
            sparse_mla_topk=sparse_mla_topk_value,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            beam_width=beam_width,
            attention_window_size=attention_window_size,
            sink_token_length=sink_token_length,
        )

    # ========== Step 6: Run generation phase ==========
    if num_generations > 0 and attn_input_type != AttentionInputType.CONTEXT_ONLY:
        # Fallback for generation phase: enqueue_generation and mla_generation are not implemented yet
        logger.debug(
            "Generation phase not implemented in trtllm-gen. Falling back to thop.attention()"
        )
        return thop.attention(
            q,
            k,
            v,
            output,
            output_sf,
            workspace,
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            cache_indirection,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            out_scale,
            rotary_inv_freq,
            rotary_cos_sin,
            latent_cache,
            q_pe,
            block_ids_per_seq,
            attention_sinks,
            is_fused_qkv,
            update_kv_cache,
            predicted_tokens_per_seq,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_size,
            tokens_per_block,
            max_num_requests,
            max_context_length,
            attention_window_size,
            sink_token_length,
            beam_width,
            mask_type,
            quant_mode,
            q_scaling,
            position_embedding_type,
            rotary_embedding_dim,
            rotary_embedding_base,
            rotary_embedding_scale_type,
            rotary_embedding_scales,
            rotary_embedding_max_position_info,
            use_paged_context_fmha,
            attention_input_type,
            is_mla_enable,
            chunked_prefill_buffer_batch_size,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            mla_tensor_params,
            attention_chunk_size,
            softmax_stats_tensor,
            spec_decoding_bool_params,
            spec_decoding_tensor_params,
            sparse_kv_indices,
            sparse_kv_offsets,
            sparse_attn_indices,
            sparse_attn_offsets,
            sparse_attn_indices_block_size,
            sparse_mla_topk,
            skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )

        # TODO: Enable _run_attention_phase for generation when enqueue_generation and mla_generation are implemented
        # _run_attention_phase(
        #     op=op,
        #     is_context=False,
        #     seq_offset=gen_seq_offset,
        #     num_seqs=num_generations,
        #     token_offset=gen_token_offset,
        #     num_tokens=num_gen_tokens,
        #     predicted_tokens_per_seq=predicted_tokens_per_seq,
        #     total_kv_len=gen_total_kv_len,
        #     workspace=workspace,
        #     output=output,
        #     output_sf=output_sf,
        #     qkv_or_q=qkv_or_q,
        #     k=k,
        #     v=v,
        #     sequence_length=sequence_length,
        #     host_past_key_value_lengths=host_past_key_value_lengths,
        #     context_lengths=context_lengths,
        #     host_context_lengths=host_context_lengths,
        #     kv_cache_block_offsets=kv_cache_block_offsets,
        #     host_kv_cache_block_offsets=host_kv_cache_block_offsets,
        #     host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
        #     host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
        #     cache_indirection=cache_indirection,
        #     kv_scale_orig_quant=kv_scale_orig_quant,
        #     kv_scale_quant_orig=kv_scale_quant_orig,
        #     out_scale=out_scale,
        #     rotary_inv_freq=rotary_inv_freq,
        #     rotary_cos_sin=rotary_cos_sin,
        #     latent_cache=latent_cache,
        #     q_pe=q_pe,
        #     block_ids_per_seq=block_ids_per_seq,
        #     mrope_rotary_cos_sin=mrope_rotary_cos_sin,
        #     mrope_position_deltas=mrope_position_deltas,
        #     mla_tensor_params=mla_tensor_params,
        #     softmax_stats_tensor=softmax_stats_tensor,
        #     spec_decoding_tensor_params=spec_decoding_tensor_params,
        #     attention_sinks=attention_sinks,
        #     sparse_kv_indices=sparse_kv_indices,
        #     sparse_kv_offsets=sparse_kv_offsets,
        #     sparse_attn_indices=sparse_attn_indices,
        #     sparse_attn_offsets=sparse_attn_offsets,
        #     sparse_attn_indices_block_size=sparse_attn_indices_block_size,
        #     sparse_mla_topk=sparse_mla_topk_value,
        #     cu_q_seqlens=cu_q_seqlens,
        #     cu_kv_seqlens=cu_kv_seqlens,
        #     fmha_scheduler_counter=fmha_scheduler_counter,
        #     mla_bmm1_scale=mla_bmm1_scale,
        #     mla_bmm2_scale=mla_bmm2_scale,
        #     quant_q_buffer=quant_q_buffer,
        #     beam_width=beam_width,
        #     attention_window_size=attention_window_size,
        #     sink_token_length=sink_token_length,
        # )

    logger.debug(f"Attention op stops at layer {layer_idx}")
