# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TrtLLM-Gen Attention Backend

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a drop-in replacement for thop.attention() with support for trtllm-gen
kernel only (Blackwell architecture: SM100/SM103). Enabled via TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1.

Architecture:
    - QKV preprocessing & RoPE: C++ kernels via tensorrt_llm.bindings.internal.thop,
      same as thop.attention. Writes K/V to paged KV cache via pool pointers.
    - Attention: flashinfer trtllm-gen FMHA kernels, reading KV cache through
      the paged KV cache fields carried by FmhaParams.

Entry points:
    FlashInferTrtllmGenAttention.is_supported()  - Check if trtllm-gen can handle the given config.
    FlashInferTrtllmGenAttention.forward()       - Main attention method.

Example:
    backend = FlashInferTrtllmGenAttention(attention_layer=...)
    supported, reason = backend.is_supported(q, k, v, attn=..., meta=..., fwd=...)
    if supported:
        backend.forward(q, k, v, attn=..., meta=..., fwd=...)
    else:
        Fallback to thop.attention()
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._utils import get_sm_version, is_sm_100f, torch_dtype_to_binding
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.quantization.mode import QuantMode

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _clear_multi_ctas_kv_counter_workspace(
    fmha_workspace: torch.Tensor,
    num_heads: int,
    max_num_requests: int,
    multi_processor_count: Optional[int],
) -> None:
    counter_size = _get_multi_ctas_kv_counter_size(
        num_heads,
        max_num_requests,
        multi_processor_count,
    )
    fmha_workspace.narrow(0, 0, counter_size).zero_()


def _get_multi_ctas_kv_counter_size(
    num_heads: int,
    max_num_requests: int,
    multi_processor_count: Optional[int],
) -> int:
    return max(num_heads * max_num_requests, multi_processor_count or 0)


def _get_bmm1_scale_log2(bmm1_scale: torch.Tensor) -> torch.Tensor:
    if bmm1_scale.numel() < 2:
        raise RuntimeError("trtllm-gen bmm1_scale workspace must contain raw and log2 scales.")
    return bmm1_scale.narrow(0, 1, 1)


def _trtllm_gen_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_pool: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    window_left: int,
    out: torch.Tensor,
    sinks: Optional[torch.Tensor],
    enable_pdl: bool,
    q_len_per_req: Optional[int],
    max_q_len: Optional[int],
    cum_seq_lens_q: Optional[torch.Tensor],
    kv_scale_pool: Optional[torch.Tensor],
    uses_shared_paged_kv_idx: bool,
) -> None:
    if q_len_per_req is not None:
        decode_max_q_len = q_len_per_req
        batch_size = query.size(0) // q_len_per_req
    else:
        if max_q_len is None or cum_seq_lens_q is None:
            raise RuntimeError(
                "trtllm-gen multi-token generation requires max_q_len and cum_seq_lens_q."
            )
        decode_max_q_len = max_q_len
        batch_size = cum_seq_lens_q.size(0) - 1

    bmm1_scale_arg = (
        _get_bmm1_scale_log2(bmm1_scale) if isinstance(bmm1_scale, torch.Tensor) else bmm1_scale
    )

    run_func = flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
    sm_count = flashinfer.decode.get_device_sm_count(query.device)
    run_func(
        out,
        None,  # out_scale_factor
        query,
        kv_pool,
        kv_pool,
        workspace_buffer,
        block_tables,
        seq_lens,
        decode_max_q_len,
        max_seq_len,
        bmm1_scale_arg,
        bmm2_scale,
        -1.0,  # o_sf_scale
        -1,  # o_sf_vec_size
        0,  # o_sf_start_index
        batch_size,
        window_left,
        0,  # sparse_mla_top_k
        sm_count,
        enable_pdl,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        sinks,
        cum_seq_lens_q,
        kv_scale_pool,  # k_block_scales
        kv_scale_pool,  # v_block_scales
        None,  # skip_softmax_threshold_scale_factor
        uses_shared_paged_kv_idx,
        None,  # lse
        0,  # lse_stride_tokens
        0,  # lse_stride_heads
    )


def _trtllm_gen_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_pool: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    window_left: int,
    out: torch.Tensor,
    sinks: Optional[torch.Tensor],
    enable_pdl: bool,
    kv_scale_pool: Optional[torch.Tensor],
    uses_shared_paged_kv_idx: bool,
) -> None:
    bmm1_scale_arg = (
        _get_bmm1_scale_log2(bmm1_scale) if isinstance(bmm1_scale, torch.Tensor) else bmm1_scale
    )

    run_func = flashinfer.prefill.get_trtllm_gen_fmha_module().trtllm_paged_attention_context
    sm_count = flashinfer.prefill.get_device_sm_count(query.device)
    run_func(
        out,
        None,  # out_scale_factor
        query,
        kv_pool,
        kv_pool,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale_arg,
        bmm2_scale,
        -1.0,  # o_sf_scale
        -1,  # o_sf_vec_size
        0,  # o_sf_start_index
        batch_size,
        window_left,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        sm_count,
        enable_pdl,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        sinks,
        kv_scale_pool,  # key_block_scales
        kv_scale_pool,  # value_block_scales
        None,  # skip_softmax_threshold_scale_factor
        uses_shared_paged_kv_idx,
        True,  # causal
        None,  # lse
        0,  # lse_stride_tokens
        0,  # lse_stride_heads
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
class FmhaParams:
    attn: "TrtllmAttention"
    meta: "TrtllmAttentionMetadata"
    fwd: AttentionForwardArgs
    workspace: torch.Tensor
    attention_input: Optional[torch.Tensor] = None
    qkv_input: Optional[torch.Tensor] = None
    context_buf: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0
    num_tokens: int = 0
    seq_offset: int = 0
    tokens_per_block: int = 64
    fp8_context_fmha: bool = False
    kv_factor: int = 0
    total_num_blocks: int = 0
    # Context-only fields
    batch_size: int = 0
    # Generation-only fields
    num_requests: int = 0
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_position_offsets: Optional[torch.Tensor] = None


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

    # Supported data types
    SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_KV_CACHE_DTYPES = {DataType.HALF, DataType.BF16, DataType.FP8, DataType.NVFP4}
    SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    # Supported Q:KV:O dtype combinations for trtllm-gen kernels
    # Format: (q_dtype: torch.dtype, kv_dtype: DataType, o_dtype: torch.dtype)
    SUPPORTED_DTYPE_COMBOS_CONTEXT = {
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        # e4m3:nvfp4:*
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }
    SUPPORTED_DTYPE_COMBOS_GENERATION = {
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        (torch.bfloat16, DataType.FP8, torch.bfloat16),
        (torch.float16, DataType.FP8, torch.float16),
        # e4m3:nvfp4:*
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }

    # 96 is excluded because trtllm-gen does not ship context kernels for it.
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80, 96}
    MAX_HEADS_RATIO_GENERATION = 32
    MIN_TOKENS_PER_BLOCK = 8
    SUPPORTED_TOKENS_PER_BLOCK = {16, 32, 64}
    SUPPORTED_MLA_GENERATION_HEAD_DIMS = {
        (320, 256),
        (576, 512),
    }
    MISSING_MLA_GENERATION_KERNELS = {
        (576, 512, 32),
    }

    def __init__(
        self,
        attention_layer: "TrtllmAttention",
    ):
        self._layout = self.DEFAULT_KV_LAYOUT
        # Read once so the hot path is not sensitive to later environment changes.
        self._enable_pdl = get_env_enable_pdl()
        missing_ops = self._missing_fused_nanobind_ops()
        if missing_ops:
            raise RuntimeError(
                f"trtllm-gen requires fused nanobind ops, missing: {', '.join(missing_ops)}."
            )

        # Lazily set on the first forward() call from the query device.
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
    def _get_kv_cache_dtype_and_total_blocks(
        meta: "TrtllmAttentionMetadata",
        is_mla_enable: bool,
    ) -> Tuple[Optional[DataType], int]:
        kv_cache_dtype = None
        total_num_blocks = 0
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is not None:
            kv_cache_dtype = kv_cache_manager.dtype
            kv_factor = 1 if is_mla_enable else 2
            blocks_in_primary_pool = getattr(kv_cache_manager, "blocks_in_primary_pool", None)
            if blocks_in_primary_pool is None:
                blocks_per_window = getattr(kv_cache_manager, "blocks_per_window", None)
                if blocks_per_window:
                    blocks_in_primary_pool = max(
                        int(primary) for primary, _ in blocks_per_window.values()
                    )
            if blocks_in_primary_pool is not None:
                total_num_blocks = (
                    int(blocks_in_primary_pool) * kv_cache_manager.num_local_layers * kv_factor
                )
        return kv_cache_dtype, total_num_blocks

    @staticmethod
    def _get_kv_factor(attn: "TrtllmAttention") -> int:
        return 1 if attn.is_mla_enable else 2

    @staticmethod
    def _get_generation_out_head_size(attn: "TrtllmAttention") -> int:
        kv_lora_rank = attn.kv_lora_rank or 0
        if attn.is_mla_enable and kv_lora_rank:
            return kv_lora_rank
        return attn.head_dim

    @staticmethod
    def _get_context_out_head_size(attn: "TrtllmAttention") -> int:
        if attn.is_mla_enable and attn.v_head_dim:
            return attn.v_head_dim
        return attn.head_dim

    @staticmethod
    def _get_bmm1_scale(attn: "TrtllmAttention") -> float:
        return 1.0 / (math.sqrt(attn.head_dim) * attn.q_scaling)

    @staticmethod
    def _get_attention_chunk_size(attn: "TrtllmAttention") -> int:
        return attn.attention_chunk_size if attn.attention_chunk_size is not None else 0

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
                "[Generation][MLA] Missing required MLA parameter(s): "
                f"{', '.join(missing_params)}.",
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

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Tuple[bool, str]:
        is_mla_enable = attn.is_mla_enable
        has_skip_softmax = (
            attn.skip_softmax_threshold_scale_factor_prefill is not None
            or attn.skip_softmax_threshold_scale_factor_decode is not None
        )
        if (
            fwd.sage_attn_num_elts_per_blk_q > 0
            or fwd.sage_attn_num_elts_per_blk_k > 0
            or fwd.sage_attn_num_elts_per_blk_v > 0
        ):
            return False, "trtllm-gen does not support sage attention."
        if meta.helix_position_offsets is not None:
            return False, "trtllm-gen does not support helix parallelism."
        sparse_kv_indices = fwd.sparse.sparse_kv_indices
        sparse_attn_indices = fwd.sparse.sparse_attn_indices
        if (
            (sparse_kv_indices is not None and sparse_kv_indices.numel() > 0)
            or (sparse_attn_indices is not None and sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
        ):
            return False, "trtllm-gen does not support sparse attention."
        if has_skip_softmax:
            return False, "trtllm-gen does not support skip-softmax attention."
        if meta.use_spec_decoding and meta.is_spec_dec_tree:
            return (
                False,
                "FlashInfer trtllm-gen does not support spec-dec tree/custom masks.",
            )
        if is_mla_enable and fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "trtllm-gen MLA supports generation-only attention."

        if not IS_FLASHINFER_AVAILABLE:
            return False, "flashinfer package is not installed."
        if meta.kv_cache_block_offsets is None:
            return False, "trtllm-gen requires paged KV cache."
        output = fwd.output
        if output is None:
            return False, "trtllm-gen requires output."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block is None:
            tokens_per_block = 0

        attn_input_type = fwd.attention_input_type
        has_context_phase = attn_input_type != AttentionInputType.generation_only
        has_generation_phase = attn_input_type != AttentionInputType.context_only
        q_dtype = q.dtype
        o_dtype = output.dtype

        sm = get_sm_version()
        if not is_sm_100f(sm):
            return False, (f"trtllm-gen requires SM100 or SM103 (Blackwell). Current: SM{sm}.")

        if q_dtype not in self.SUPPORTED_INPUT_DTYPES:
            return False, (
                f"Input dtype {q_dtype} not supported. Supported: FP16, BF16, FP8 (E4M3)."
            )

        kv_cache_dtype, _ = self._get_kv_cache_dtype_and_total_blocks(meta, is_mla_enable)
        if kv_cache_dtype is None:
            kv_cache_dtype = torch_dtype_to_binding(q_dtype)

        is_fp8_out = output.dtype == torch.float8_e4m3fn
        is_fp4_out = output.dtype == torch.uint8
        has_fp8_kv = kv_cache_dtype == DataType.FP8
        has_fp4_kv = kv_cache_dtype == DataType.NVFP4
        fp8_context_fmha = (
            is_fp8_out or is_fp4_out or has_fp4_kv or (has_fp8_kv and has_context_phase)
        )
        if has_fp4_kv or fp8_context_fmha:
            q_dtype = torch.float8_e4m3fn

        if kv_cache_dtype not in self.SUPPORTED_KV_CACHE_DTYPES:
            return False, (
                f"KV cache dtype {kv_cache_dtype} not supported. Supported: FP16, BF16, FP8, NVFP4."
            )
        if o_dtype not in self.SUPPORTED_OUT_DTYPES:
            return False, f"Output dtype {o_dtype} not supported. Supported: FP16, BF16, FP8."

        assert attn.num_heads > 0, "num_heads must be positive."
        assert attn.num_kv_heads > 0, "num_kv_heads must be positive."
        if attn.num_heads % attn.num_kv_heads != 0:
            return (
                False,
                f"num_heads ({attn.num_heads}) must be divisible by "
                f"num_kv_heads ({attn.num_kv_heads}).",
            )

        has_alibi = attn.position_embedding_type in (4, 5)
        check_context_phase = has_context_phase and not is_mla_enable
        if check_context_phase:
            if attn.head_dim in self.UNSUPPORTED_HEAD_SIZES_CONTEXT:
                return False, f"[Context] Head size {attn.head_dim} is not supported."
            try:
                if AttentionMaskType(fwd.mask_type) == AttentionMaskType.custom_mask:
                    return False, "[Context] Custom mask is not supported."
            except ValueError:
                return False, f"[Context] Invalid mask_type: {fwd.mask_type}."
            if has_alibi:
                return False, "[Context] ALiBi is not supported."
            if (q_dtype, kv_cache_dtype, o_dtype) not in self.SUPPORTED_DTYPE_COMBOS_CONTEXT:
                return False, (
                    f"[Context] Unsupported dtype combination: "
                    f"Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )

        if has_generation_phase:
            if meta.beam_width != 1:
                return (
                    False,
                    f"[Generation] Beam search (beam_width={meta.beam_width}) "
                    "is not supported. Must be 1.",
                )
            sink_token_length = 0
            if sink_token_length != 0:
                return (
                    False,
                    f"[Generation] StreamingLLM "
                    f"(sink_token_length={sink_token_length}) is not supported.",
                )
            if tokens_per_block < self.MIN_TOKENS_PER_BLOCK:
                return (
                    False,
                    f"[Generation] tokens_per_block ({tokens_per_block}) "
                    f"must be >= {self.MIN_TOKENS_PER_BLOCK}.",
                )
            heads_ratio = attn.num_heads // attn.num_kv_heads
            if not is_mla_enable and heads_ratio > self.MAX_HEADS_RATIO_GENERATION:
                return (
                    False,
                    f"[Generation] heads ratio ({heads_ratio}) exceeds maximum "
                    f"({self.MAX_HEADS_RATIO_GENERATION}).",
                )
            if has_alibi:
                return False, "[Generation] ALiBi is not supported."
            if (q_dtype, kv_cache_dtype, o_dtype) not in self.SUPPORTED_DTYPE_COMBOS_GENERATION:
                return False, (
                    f"[Generation] Unsupported dtype combination: "
                    f"Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )
            if is_mla_enable:
                supported, reason = self._check_mla_generation_support(
                    head_size=attn.head_dim,
                    tokens_per_block=tokens_per_block,
                    kv_lora_rank=attn.kv_lora_rank,
                    qk_rope_head_dim=attn.qk_rope_head_dim,
                )
                if not supported:
                    return False, reason

        if tokens_per_block <= 0:
            return False, "tokens_per_block must be positive."
        if tokens_per_block & (tokens_per_block - 1) != 0:
            return False, f"tokens_per_block ({tokens_per_block}) must be power of 2."
        if tokens_per_block not in self.SUPPORTED_TOKENS_PER_BLOCK:
            supported = sorted(self.SUPPORTED_TOKENS_PER_BLOCK)
            return (
                False,
                f"tokens_per_block ({tokens_per_block}) is not supported "
                f"by trtllm-gen kernels. Supported: {supported}.",
            )

        return True, ""

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
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> None:
        output = fwd.output
        if output is None:
            raise RuntimeError("trtllm-gen attention requires output.")
        if meta.kv_cache_block_offsets is None:
            raise RuntimeError("trtllm-gen attention requires paged KV cache.")

        workspace = meta.effective_workspace
        if workspace is None:
            workspace = torch.empty((0,), device=q.device, dtype=torch.int8)

        # Lazily cache the SM count from the first query tensor's device.
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(q.device)

        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_size = attn.head_dim
        quant_mode = attn.quant_mode
        is_mla_enable = attn.is_mla_enable
        tokens_per_block = meta.tokens_per_block
        max_num_requests = meta.max_num_requests
        max_context_length = meta.max_context_length
        attention_window_size = fwd.attention_window_size
        beam_width = meta.beam_width

        num_tokens = q.size(0)
        attn_input_type = fwd.attention_input_type
        is_gen_only = attn_input_type == AttentionInputType.generation_only
        is_fp8_out = output.dtype == torch.float8_e4m3fn
        is_fp4_out = output.dtype == torch.uint8
        kv_cache_quant_mode = QuantMode(quant_mode)
        fp8_context_fmha = (
            is_fp8_out
            or is_fp4_out
            or kv_cache_quant_mode.has_fp4_kv_cache()
            or (kv_cache_quant_mode.has_fp8_kv_cache() and not is_gen_only)
        )

        num_contexts = meta.num_contexts
        num_ctx_tokens = meta.num_ctx_tokens
        num_generations = meta.num_generations
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid trtllm-gen attention token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}, attention_input_type={attn_input_type}."
            )

        workspace_max_tokens = max(num_tokens, max_context_length)
        workspace_max_gen_tokens = max(num_gen_tokens, max_num_requests)
        required_workspace_size = _get_workspace_size(
            dtype=q.dtype,
            num_tokens=workspace_max_tokens,
            num_gen_tokens=workspace_max_gen_tokens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            max_num_requests=max_num_requests,
            rotary_embedding_dim=attn.rope_dim,
            fp8_context_fmha=fp8_context_fmha,
        )

        current_workspace_size = workspace.numel() * workspace.element_size()
        if current_workspace_size < required_workspace_size:
            if meta.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Attention CUDA graph workspace is smaller than the "
                    "required size for trtllm-gen."
                )
            required_workspace_numel = math.ceil(required_workspace_size / workspace.element_size())
            workspace.resize_((required_workspace_numel,))

        out_head_size = (
            self._get_generation_out_head_size(attn)
            if is_gen_only
            else self._get_context_out_head_size(attn)
        )
        out_tensor = output.view(num_tokens, num_heads, out_head_size)

        cache_indirection = meta.cache_indirection
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
        tokens_per_block = tokens_per_block if tokens_per_block is not None else 64
        _, total_num_blocks = self._get_kv_cache_dtype_and_total_blocks(meta, is_mla_enable)

        params = FmhaParams(
            attn=attn,
            meta=meta,
            fwd=fwd,
            workspace=workspace,
            max_attention_window_size=max_attn_window_size,
            cyclic_attention_window_size=cyclic_attn_window_size,
            tokens_per_block=tokens_per_block,
            fp8_context_fmha=fp8_context_fmha,
            kv_factor=self._get_kv_factor(attn),
            total_num_blocks=total_num_blocks,
        )

        sequence_length = meta.kv_lens_cuda_runtime
        host_past_key_value_lengths = meta.kv_lens_runtime

        if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
            seq_offset = 0
            token_offset = 0
            num_seqs = num_contexts

            context_lengths = meta.prompt_lens_cuda_runtime
            host_context_lengths = meta.prompt_lens_cpu_runtime
            max_context_q_len = int(host_context_lengths[seq_offset : seq_offset + num_seqs].max())
            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )

            params.attention_input = q[token_offset : token_offset + num_ctx_tokens]
            params.qkv_input = params.attention_input
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

            predicted_tokens_per_seq = attn.predicted_tokens_per_seq
            spec_gen_lengths = None
            spec_pos_offsets = None
            if meta.is_spec_decoding_enabled and predicted_tokens_per_seq > 1:
                spec_gen_lengths = meta.spec_decoding_generation_lengths
                position_offsets_for_cpp = meta.spec_decoding_position_offsets_for_cpp
                if position_offsets_for_cpp is not None and position_offsets_for_cpp.dim() == 1:
                    position_offsets_for_cpp = position_offsets_for_cpp.view(max_num_requests, -1)
                spec_pos_offsets = position_offsets_for_cpp

            params.attention_input = q[token_offset : token_offset + num_gen_tokens]
            params.qkv_input = params.attention_input
            params.context_buf = out_tensor[token_offset : token_offset + num_gen_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_gen_tokens
            params.seq_offset = seq_offset
            params.input_seq_length = input_seq_length
            params.num_requests = num_seqs // beam_width
            params.spec_decoding_generation_lengths = spec_gen_lengths
            params.spec_decoding_position_offsets = spec_pos_offsets
            if is_mla_enable:
                self.run_mla_generation(params)
            else:
                self.run_generation(params)
        return

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
                "Chunked-attention and sliding-window-attention should not "
                "be enabled at the same time."
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

    def run_context(
        self,
        params: FmhaParams,
    ) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        bmm1_scale_static = self._get_bmm1_scale(attn)
        attention_chunk_size = self._get_attention_chunk_size(attn)

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
            params.qkv_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.context_lengths,  # context_lengths
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_inv_freq,  # rotary_inv_freq
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_rotary_cos_sin,  # mrope_rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            fwd.mask_type,  # mask_type
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            params.batch_size,  # batch_size
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            1.0,  # bmm2_scale
            attention_chunk_size,  # attention_chunk_size
            params.fp8_context_fmha,  # fp8_context_fmha
            meta.use_paged_context_fmha,  # paged_context_fmha
            attn.is_mla_enable,  # is_mla_enable
            self._multi_processor_count,  # multi_processor_count
            params.total_num_blocks,  # total_num_blocks
            params.kv_factor,  # kv_factor
            True,  # need_build_kv_cache_metadata
        )

        has_fp4_kv = QuantMode(attn.quant_mode).has_fp4_kv_cache()
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or params.fp8_context_fmha:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, attn.num_heads, attn.head_dim)
            )
        ctx_bmm1_scale = (
            bmm1_scale if params.fp8_context_fmha and bmm1_scale is not None else bmm1_scale_static
        )
        ctx_bmm2_scale = bmm2_scale if params.fp8_context_fmha and bmm2_scale is not None else 1.0

        _trtllm_gen_batch_context_with_kv_cache(
            q_processed,  # query
            kv_pool,  # kv_pool
            fmha_workspace,  # workspace_buffer
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            max_q_len,  # max_q_len
            max_kv_len,  # max_kv_len
            ctx_bmm1_scale,  # bmm1_scale
            ctx_bmm2_scale,  # bmm2_scale
            params.batch_size,  # batch_size
            cu_q_seqlens,  # cum_seq_lens_q
            cu_kv_seqlens,  # cum_seq_lens_kv
            window_left,  # window_left
            params.context_buf,  # out
            fwd.attention_sinks,  # sinks
            self._enable_pdl,  # enable_pdl
            kv_scale_pool,  # kv_scale_pool
            self.USE_SHARED_PAGED_KV_IDX,  # uses_shared_paged_kv_idx
        )

        thop.trtllm_gen_context_postprocess(
            params.qkv_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.context_lengths,  # context_lengths
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_rotary_cos_sin,  # mrope_rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            fwd.mask_type,  # mask_type
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            params.batch_size,  # batch_size
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            params.fp8_context_fmha,  # fp8_context_fmha
            meta.use_paged_context_fmha,  # paged_context_fmha
            attn.is_mla_enable,  # is_mla_enable
            attention_chunk_size,  # attention_chunk_size
            self._multi_processor_count,  # multi_processor_count
        )

    def run_generation(
        self,
        params: FmhaParams,
    ) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        bmm1_scale_static = self._get_bmm1_scale(attn)
        attention_chunk_size = self._get_attention_chunk_size(attn)
        batch_beam = params.num_requests * meta.beam_width
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
            params.qkv_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.spec_decoding_generation_lengths,  # spec_decoding_generation_lengths
            params.spec_decoding_position_offsets,  # spec_decoding_position_offsets
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_inv_freq,  # rotary_inv_freq
            attn.rotary_cos_sin,  # rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            params.seq_offset,  # seq_offset
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            batch_beam,  # batch_beam
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            1.0,  # bmm2_scale
            params.fp8_context_fmha,  # fp8_context_fmha
            attn.predicted_tokens_per_seq,  # predicted_tokens_per_seq
            attention_chunk_size,  # attention_chunk_size
            self._multi_processor_count,  # multi_processor_count
            params.total_num_blocks,  # total_num_blocks
            params.kv_factor,  # kv_factor
            True,  # need_build_kv_cache_metadata
        )

        # FIXME: Flashinfer trtllm-gen API doesn't support a separate
        # multi CTAs counter buffer. We have to clear a small buffer
        # before trtllm_gen_batch_decode_with_kv_cache.
        #
        # We must also avoid clearing the workspace only when it is
        # resized. The warmup phase may have already cached the workspace
        # pointer; if the capture phase skips the zeroing step, the
        # CUDA graph will not include the counter initialization. We
        # have already verified—specifically in the context of the GPTOSS-20B
        # test graph replay scenario—that this skipping logic is unsafe.
        #
        # https://github.com/flashinfer-ai/flashinfer/issues/3433
        _clear_multi_ctas_kv_counter_workspace(
            fmha_workspace, attn.num_heads, meta.max_num_requests, self._multi_processor_count
        )

        q_len_per_req = None if is_multi_token_gen else params.input_seq_length
        decode_max_q_len = max_q_len if is_multi_token_gen else None
        decode_cu_seqlens = cu_seqlens if is_multi_token_gen else None

        has_fp4_kv = QuantMode(attn.quant_mode).has_fp4_kv_cache()
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or params.fp8_context_fmha:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, attn.num_heads, attn.head_dim)
            )
        gen_bmm1_scale = (
            bmm1_scale if params.fp8_context_fmha and bmm1_scale is not None else bmm1_scale_static
        )
        gen_bmm2_scale = bmm2_scale if params.fp8_context_fmha and bmm2_scale is not None else 1.0

        _trtllm_gen_batch_decode_with_kv_cache(
            q_processed,  # query
            kv_pool,  # kv_pool
            fmha_workspace,  # workspace_buffer
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            max_kv_len,  # max_seq_len
            gen_bmm1_scale,  # bmm1_scale
            gen_bmm2_scale,  # bmm2_scale
            window_left,  # window_left
            params.context_buf,  # out
            fwd.attention_sinks,  # sinks
            self._enable_pdl,  # enable_pdl
            q_len_per_req,  # q_len_per_req
            decode_max_q_len,  # max_q_len
            decode_cu_seqlens,  # cum_seq_lens_q
            kv_scale_pool,  # kv_scale_pool
            self.USE_SHARED_PAGED_KV_IDX,  # uses_shared_paged_kv_idx
        )

    def run_mla_generation(
        self,
        params: FmhaParams,
    ) -> None:
        """MLA generation decode using flashinfer MLA kernel."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            raise NotImplementedError(
                "Sliding-window attention is not supported by MLA decode path."
            )
        if self._get_attention_chunk_size(attn) != 0:
            raise NotImplementedError("Chunked-attention is not supported by MLA decode path.")

        batch_beam = params.num_requests * meta.beam_width
        if params.attention_input is None:
            raise RuntimeError("MLA generation requires attention_input.")
        kv_cache, block_tables = thop.build_trtllm_gen_kv_cache_metadata(
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            attn.local_layer_idx,  # layer_idx
            attn.num_kv_heads,  # num_kv_heads
            params.tokens_per_block,  # tokens_per_block
            attn.head_dim,  # head_dim
            params.kv_factor,  # kv_factor
            params.total_num_blocks,  # total_num_blocks
            attn.quant_mode,  # kv_cache_quant_mode
            params.seq_offset,  # batch_start
            batch_beam,  # batch_size
            params.attention_input.dtype,  # dtype
        )

        pages_per_superblock = 128 // params.tokens_per_block
        if pages_per_superblock > 1:
            num_blocks = block_tables.size(-1)
            remainder = num_blocks % pages_per_superblock
            if remainder != 0:
                pad = pages_per_superblock - remainder
                block_tables = torch.nn.functional.pad(block_tables, (0, pad), value=0)

        kv_lora_rank = attn.kv_lora_rank or 0
        qk_nope_head_dim = attn.qk_nope_head_dim or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        mla_head_dim_qk = kv_lora_rank + qk_rope_head_dim
        q_len_per_req = params.num_tokens // batch_beam if batch_beam > 0 else 1

        query = params.qkv_input.view(batch_beam, q_len_per_req, attn.num_heads, mla_head_dim_qk)

        bmm1_scale = 1.0 / (attn.q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))

        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query,  # query
            kv_cache,  # kv_cache
            params.workspace.view(-1, 4),  # workspace_buffer
            qk_nope_head_dim,  # qk_nope_head_dim
            kv_lora_rank,  # kv_lora_rank
            qk_rope_head_dim,  # qk_rope_head_dim
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            params.max_past_kv_length,  # max_seq_len
            0,  # sparse_mla_top_k
            params.context_buf.view(batch_beam, q_len_per_req, attn.num_heads, kv_lora_rank),  # out
            bmm1_scale,  # bmm1_scale
            1.0,  # bmm2_scale
            fwd.attention_sinks,  # sinks
            None,  # skip_softmax_threshold_scale_factor
            self._enable_pdl,  # enable_pdl
            "trtllm-gen",  # backend
            True,  # is_var_seq
            self.USE_SHARED_PAGED_KV_IDX,  # uses_shared_paged_kv_idx
        )
