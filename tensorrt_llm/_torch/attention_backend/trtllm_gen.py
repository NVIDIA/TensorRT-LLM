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

Entry point:
    FlashInferTrtllmGenFmha.forward() - Main attention method selected by
    TrtllmAttention's unified FMHA capability evaluator.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch.attention_backend.fmha import (
    AcceptedIntegerValues,
    BaseFmha,
    DTypeCombination,
    FmhaCapabilities,
    FmhaFeature,
    FmhaPhase,
    FmhaSupportContext,
    MlaCapabilities,
    MlaGenerationCase,
    PhaseCapabilities,
)
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_TRTLLM_GEN_REQUIRED_THOP_OPS = (
    "get_trtllm_gen_context_workspace_layout",
    "get_trtllm_gen_generation_workspace_layout",
    "trtllm_gen_context_preprocess",
    "trtllm_gen_context_postprocess",
    "trtllm_gen_generation_preprocess",
    "build_trtllm_gen_kv_cache_metadata",
)

_TRTLLM_GEN_CONTEXT_DTYPE_COMBINATIONS = frozenset(
    {
        DTypeCombination(torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        DTypeCombination(torch.float16, DataType.HALF, torch.float16),
        DTypeCombination(torch.bfloat16, DataType.BF16, torch.bfloat16),
        DTypeCombination(torch.float8_e4m3fn, DataType.FP8, torch.float16),
        DTypeCombination(torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        DTypeCombination(torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        DTypeCombination(torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        DTypeCombination(torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }
)

_TRTLLM_GEN_GENERATION_DTYPE_COMBINATIONS = frozenset(
    {
        DTypeCombination(torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        DTypeCombination(torch.float16, DataType.HALF, torch.float16),
        DTypeCombination(torch.bfloat16, DataType.BF16, torch.bfloat16),
        DTypeCombination(torch.float8_e4m3fn, DataType.FP8, torch.float16),
        DTypeCombination(torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        DTypeCombination(torch.bfloat16, DataType.FP8, torch.bfloat16),
        DTypeCombination(torch.float16, DataType.FP8, torch.float16),
        DTypeCombination(torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        DTypeCombination(torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        DTypeCombination(torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }
)

_TRTLLM_GEN_RUNTIME_FEATURES = frozenset(
    {
        FmhaFeature.kv_cache_manager,
        FmhaFeature.paged_kv_cache,
        FmhaFeature.output_buffer,
    }
)

_TRTLLM_GEN_CONTEXT_HEAD_SIZES = frozenset(
    {
        # TRTLLM-GEN context kernels currently cover common power-of-two LLM head
        # sizes. HeadDim=80 has a padded SMEM layout in the C++ export path but is
        # not accepted here until the flashinfer package ships the matching context
        # kernels for this path.
        16,
        32,
        64,
        128,
        256,
        512,
    }
)

_TRTLLM_GEN_POSITION_EMBEDDING_TYPES = frozenset(
    {
        position_embedding_type
        for position_embedding_type in PositionEmbeddingType
        if not position_embedding_type.is_alibi()
    }
)

_TRTLLM_GEN_CONTEXT_MASK_TYPES = frozenset(
    {mask_type for mask_type in AttentionMaskType if mask_type != AttentionMaskType.custom_mask}
)

_TRTLLM_GEN_MLA_GENERATION_CASES = frozenset(
    {
        MlaGenerationCase(320, 256, 16),
        MlaGenerationCase(320, 256, 32),
        MlaGenerationCase(320, 256, 64),
        MlaGenerationCase(576, 512, 16),
        MlaGenerationCase(576, 512, 64),
    }
)

FLASHINFER_TRTLLM_GEN_CAPABILITIES = FmhaCapabilities(
    name="flashinfer_trtllm_gen",
    sm_versions=AcceptedIntegerValues(values=frozenset({100, 103})),
    attention_input_types=frozenset(AttentionInputType),
    runtime_features=_TRTLLM_GEN_RUNTIME_FEATURES,
    required_runtime_features=_TRTLLM_GEN_RUNTIME_FEATURES,
    tokens_per_block=AcceptedIntegerValues(values=frozenset({16, 32, 64})),
    generation_beam_widths=AcceptedIntegerValues(values=frozenset({1})),
    generation_head_ratios=AcceptedIntegerValues(min_value=1, max_value=32),
    context=PhaseCapabilities(
        dtype_combinations=_TRTLLM_GEN_CONTEXT_DTYPE_COMBINATIONS,
        head_sizes=AcceptedIntegerValues(values=_TRTLLM_GEN_CONTEXT_HEAD_SIZES),
        mask_types=_TRTLLM_GEN_CONTEXT_MASK_TYPES,
        position_embedding_types=_TRTLLM_GEN_POSITION_EMBEDDING_TYPES,
        features=frozenset(),
    ),
    generation=PhaseCapabilities(
        dtype_combinations=_TRTLLM_GEN_GENERATION_DTYPE_COMBINATIONS,
        head_sizes=AcceptedIntegerValues(min_value=1),
        mask_types=frozenset(AttentionMaskType),
        position_embedding_types=_TRTLLM_GEN_POSITION_EMBEDDING_TYPES,
        features=frozenset(),
    ),
    mla=MlaCapabilities(
        phases=frozenset({FmhaPhase.generation}),
        generation_cases=_TRTLLM_GEN_MLA_GENERATION_CASES,
    ),
)


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
    read directly from ``FlashInferTrtllmGenFmha`` cached attributes
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


class FlashInferTrtllmGenFmha(BaseFmha):
    """
    An FMHA implementation using pure trtllm-gen kernels from flashinfer.
    """

    capabilities = FLASHINFER_TRTLLM_GEN_CAPABILITIES

    # Default KV layout for flashinfer
    # HND = [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]
    DEFAULT_KV_LAYOUT = "HND"
    # Keep shared paged indices disabled to match the current TensorRT-LLM
    # block-table layout used by the fused preprocessing path.
    USE_SHARED_PAGED_KV_IDX = False

    @classmethod
    def is_supported(cls, context: FmhaSupportContext) -> bool:
        if not super().is_supported(context):
            return False
        if os.environ.get("TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION", "0") != "1":
            return cls._not_supported("disabled by TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION.")
        if not IS_FLASHINFER_AVAILABLE:
            return cls._not_supported("flashinfer package is not installed.")
        missing_ops = cls._missing_fused_nanobind_ops()
        if missing_ops:
            return cls._not_supported(f"missing fused nanobind ops: {', '.join(missing_ops)}.")
        return True

    def __init__(
        self,
        attention_layer: "TrtllmAttention",
    ):
        super().__init__(attention_layer)
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

        # Lazily set on the first attention() call from the query device.
        self._multi_processor_count: Optional[int] = None

    @property
    def layout(self) -> str:
        """KV cache layout."""
        return self._layout

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

    def forward(
        self,
        attn: "TrtllmAttention",
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        self.attention(
            q,
            metadata=metadata,
            forward_args=forward_args,
            mask_type=int(forward_args.mask_type),
            use_paged_context_fmha=metadata.use_paged_context_fmha,
        )

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
            or (
                (kv_cache_quant_mode.has_fp8_kv_cache() or kv_cache_quant_mode.has_fp4_kv_cache())
                and use_paged_context_fmha
            )
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
        return [op for op in _TRTLLM_GEN_REQUIRED_THOP_OPS if not hasattr(thop, op)]

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
        if blocks_in_primary_pool is None:
            raise RuntimeError(
                "trtllm-gen could not determine blocks_in_primary_pool from the KVCacheManager."
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
        mrope_rotary_cos_sin = params.forward.mrope_rotary_cos_sin

        (
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
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
        kv_cache_sf = None
        if kv_scale_pool is not None:
            kv_cache_sf = (kv_scale_pool, kv_scale_pool)

        has_fp4_kv = QuantMode(params.kv_cache_quant_mode).has_fp4_kv_cache()
        if has_fp4_kv:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * self._num_heads * self._head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, self._num_heads, self._head_dim)
            )
        ctx_bmm1_scale = bmm1_scale if has_fp4_kv and bmm1_scale is not None else self._bmm1_scale
        ctx_bmm2_scale = bmm2_scale if has_fp4_kv and bmm2_scale is not None else 1.0

        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q_processed,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=fmha_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=ctx_bmm1_scale,
            bmm2_scale=ctx_bmm2_scale,
            batch_size=params.batch_size,
            cum_seq_lens_q=cu_q_seqlens,
            cum_seq_lens_kv=cu_kv_seqlens,
            window_left=window_left,
            out=params.context_buf,
            kv_layout=self._layout,
            sinks=params.forward.attention_sinks,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            kv_cache_sf=kv_cache_sf,
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
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
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
        kv_cache_sf = None
        if kv_scale_pool is not None:
            kv_cache_sf = (kv_scale_pool, kv_scale_pool)

        has_fp4_kv = QuantMode(params.kv_cache_quant_mode).has_fp4_kv_cache()
        if has_fp4_kv:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * self._num_heads * self._head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, self._num_heads, self._head_dim)
            )
        gen_bmm1_scale = bmm1_scale if has_fp4_kv else self._bmm1_scale
        gen_bmm2_scale = bmm2_scale if has_fp4_kv else 1.0

        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q_processed,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=fmha_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_seq_len=max_kv_len,
            out=params.context_buf,
            bmm1_scale=gen_bmm1_scale,
            bmm2_scale=gen_bmm2_scale,
            window_left=window_left,
            kv_layout=self._layout,
            sinks=params.forward.attention_sinks,
            q_len_per_req=q_len_per_req,
            max_q_len=decode_max_q_len,
            cum_seq_lens_q=decode_cu_seqlens,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            kv_cache_sf=kv_cache_sf,
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
        kv_cache, block_tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
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


FlashInferTrtllmGenAttention = FlashInferTrtllmGenFmha
