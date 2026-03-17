# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""TRT-LLM MLA attention backend for Auto-Deploy.

Wraps TRT-LLM's ``thop.attention`` kernel with ``is_mla_enable=True`` for use in
Auto-Deploy, following the same design pattern as the standard TRT-LLM attention
backend (``trtllm_attention.py``).

MLA stores compressed latent representations instead of separate K and V.

Phase-specific details:

- **Prefill** (``attention_input_type=context_only``):
  Expands compressed KV via ``kv_b_proj_weight`` to get separate K_nope, V, then builds
  ``Q = [q_nope | q_pe]``, ``K = [k_nope | k_pe]``. Each prefill sequence is processed
  independently (matching the PyExecutor pattern) because with
  ``use_paged_context_fmha=False`` the kernel treats all tokens as one sequence.
  Calls ``thop.attention`` with ``is_fused_qkv=False`` and ``head_size=qk_head_dim``.
  Output is directly in ``v_head_dim`` space.

  **Dimension constraint**: The C++ FMHA context kernel derives ``headSizeV``
  as ``qk_nope_head_dim``, so ``v_head_dim`` must equal ``qk_nope_head_dim``.
  Models where they differ (e.g. GLM-4.7-Flash) are rejected at graph-build
  time with a clear error — use the ``flashinfer_mla`` backend instead.

- **Decode** (``attention_input_type=generation_only``):
  Uses weight absorption: ``q_absorbed = q_nope @ W_kn``, then
  ``fused_q = [q_absorbed | q_pe]``. Calls ``thop.attention`` with
  ``is_fused_qkv=True`` and ``head_size=gen_head_size``. Output is in latent space
  and projected back to ``v_head_dim`` via ``W_v``.

Cache layout:
    kv_cache: paged pool storing [compressed_kv | k_pe] per token
    - num_kv_heads=1, head_dim=kv_lora_rank+qk_rope_head_dim, kv_factor=1 (HND)
    - kv_factor=1 because MLA K and V are the same latent data; this matches
      the C++ thop.attention expectation and enables the flash-MLA kernel on SM90
"""

import math
from typing import List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.quantization import QuantMode

from .....llmapi.llm_args import KvCacheConfig
from ...utils.cuda_graph import cuda_graph_state
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    KVPagedResourceHandler,
    MHACallable,
    PrepareMetadataCallable,
    PrepareMetadataHostCallable,
    ResourceHandlerDict,
)

# =============================================================================
# Helpers
# =============================================================================


def _mla_padded_num_heads(num_heads: int, num_kv_heads: int) -> int:
    """Return padded num_heads so that the Q/KV ratio is a power of 2.

    The trtllm-gen FMHA MLA decode kernel on SM100 (Blackwell) requires that
    ``num_heads // num_kv_heads`` is a power of two.  If it already is, this
    returns ``num_heads`` unchanged.  Otherwise it rounds the ratio up to the
    next power of two and returns ``rounded_ratio * num_kv_heads``.
    """
    ratio = num_heads // num_kv_heads
    if ratio > 0 and (ratio & (ratio - 1)) == 0:
        return num_heads
    padded_ratio = 1 << (ratio - 1).bit_length()
    return padded_ratio * num_kv_heads


# =============================================================================
# Module-level planner
# =============================================================================


class _TrtllmMLAPlanner:
    """Planner for TRT-LLM MLA attention backend.

    Mirrors ``_TrtllmPlanner`` from the standard trtllm backend.  Only stores data
    that cannot be derived from ``SequenceInfo`` or tensor shapes:

    - ``workspace``: scratch for thop.attention
    - Host metadata not in SequenceInfo: ``host_request_types``, ``host_total_kv_lens``,
      ``host_past_kv_lengths``, ``host_context_lengths``
    - ``block_offsets`` / ``block_ids_per_seq``: filled by the device-side
      ``prepare_trtllm_mla_metadata`` op using a Triton kernel
    - Per-layer pool pointers (``_per_layer_pool_ptrs``)
    - Decode scratch buffers for MLA weight absorption / V projection

    Two entry points:
    - ``plan_host()``: fills pinned host tensors (runs outside CUDA graph)
    - ``plan_device()``: computes block_offsets/block_ids_per_seq via Triton (runs
      inside the graph as part of ``prepare_trtllm_mla_metadata``)
    """

    def __init__(self):
        self.workspace: Optional[torch.Tensor] = None
        self._per_layer_pool_ptrs: dict = {}
        self.host_pool_mapping: Optional[torch.Tensor] = None
        self.host_request_types: Optional[torch.Tensor] = None
        self.host_total_kv_lens: Optional[torch.Tensor] = None
        self.host_past_kv_lengths: Optional[torch.Tensor] = None
        self.host_context_lengths: Optional[torch.Tensor] = None
        self.block_offsets: Optional[torch.Tensor] = None
        self.block_ids_per_seq: Optional[torch.Tensor] = None
        self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        self.kv_scale_quant_orig: Optional[torch.Tensor] = None

        # Decode-path buffers reused across layers (allocated on first use)
        self.cu_q_decode: Optional[torch.Tensor] = None
        self.cu_kv_decode: Optional[torch.Tensor] = None
        self.fmha_scheduler_counter_decode: Optional[torch.Tensor] = None
        self.output_latent: Optional[torch.Tensor] = None
        self.fused_q_flat: Optional[torch.Tensor] = None
        self.q_pe_padded_buf: Optional[torch.Tensor] = None
        self.latent_cache_buf: Optional[torch.Tensor] = None
        self._cu_kv_decode_host: Optional[torch.Tensor] = None
        self._decode_buf_max_tokens: int = 0
        self._decode_buf_num_heads: int = 0
        self._decode_buf_kv_lora_rank: int = 0
        self._decode_buf_rope_dim: int = 0
        self._decode_buf_dtype: Optional[torch.dtype] = None
        self._decode_padded_num_heads: int = 0
        self._tokens_per_block: int = 0
        self._cache_rows_per_block: int = 0

        # Pre-allocated V-projection output buffer (allocated in ensure_decode_buffers)
        self.v_proj_output: Optional[torch.Tensor] = None

        # Prefill RoPE table (created on first prefill call if not fused-rope)
        self._prefill_rotary_cos_sin: Optional[torch.Tensor] = None
        self._prefill_rcs_max_pos: int = 0
        self._prefill_rcs_dim: int = 0

    def get_or_create_rotary_cos_sin(
        self,
        max_positions: int,
        dim: int,
        theta: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a cached rotary_cos_sin table, creating it if needed.

        The C++ kernel's ``invokeMLARopeContext`` requires a valid cos/sin table
        even when the model has already applied RoPE externally.  This method
        lazily creates one using the same format as the PT backend
        (``RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin``).
        """
        if (
            self._prefill_rotary_cos_sin is not None
            and self._prefill_rcs_max_pos >= max_positions
            and self._prefill_rcs_dim == dim
        ):
            return self._prefill_rotary_cos_sin

        from tensorrt_llm.functional import RopeEmbeddingUtils

        _, cos_sin_np = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            max_positions, dim, theta
        )
        self._prefill_rotary_cos_sin = torch.tensor(cos_sin_np, dtype=torch.float32, device=device)
        self._prefill_rcs_max_pos = max_positions
        self._prefill_rcs_dim = dim
        return self._prefill_rotary_cos_sin

    def reset(self, device: torch.device, max_batch: int, max_blocks_per_seq: int) -> None:
        """One-time allocation of ALL persistent buffers."""
        if self.workspace is not None:
            return

        self.workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.host_pool_mapping = torch.zeros(
            1, 2, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_total_kv_lens = torch.zeros(
            2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_request_types = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.block_offsets = torch.zeros(
            1, max_batch, 2, max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self.host_past_kv_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_context_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.block_ids_per_seq = torch.zeros(
            max_batch, max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self.cu_kv_decode = torch.zeros(max_batch + 1, dtype=torch.int32, device=device)
        self._cu_kv_decode_host = torch.zeros(
            max_batch + 1, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.fmha_scheduler_counter_decode = torch.zeros(1, dtype=torch.uint32, device=device)

    def ensure_decode_buffers(
        self,
        device: torch.device,
        max_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Allocate or grow decode-path scratch buffers (once, then reused).

        On SM100+ the FMHA MLA decode kernel requires a power-of-2 Q/KV head
        ratio.  Buffers are sized for the *padded* head count so that padding
        inside ``_handle_decode_impl`` is a zero-copy view.
        """
        need_alloc = (
            self.cu_q_decode is None
            or max_tokens > self._decode_buf_max_tokens
            or num_heads != self._decode_buf_num_heads
            or kv_lora_rank != self._decode_buf_kv_lora_rank
            or qk_rope_head_dim != self._decode_buf_rope_dim
            or dtype != self._decode_buf_dtype
        )
        if not need_alloc:
            return

        sm = get_sm_version()
        padded = _mla_padded_num_heads(num_heads, num_kv_heads) if sm >= 100 else num_heads

        self._decode_buf_max_tokens = max_tokens
        self._decode_buf_num_heads = num_heads
        self._decode_buf_kv_lora_rank = kv_lora_rank
        self._decode_buf_rope_dim = qk_rope_head_dim
        self._decode_buf_dtype = dtype
        self._decode_padded_num_heads = padded
        gen_head_size = kv_lora_rank + qk_rope_head_dim

        self.cu_q_decode = torch.arange(max_tokens + 1, dtype=torch.int32, device=device) * padded
        self.output_latent = torch.empty(
            max_tokens, padded * kv_lora_rank, dtype=dtype, device=device
        )
        alloc_fn = torch.zeros if padded != num_heads else torch.empty
        self.fused_q_flat = alloc_fn(max_tokens, padded * gen_head_size, dtype=dtype, device=device)
        if padded != num_heads:
            self.q_pe_padded_buf = torch.zeros(
                max_tokens, padded, qk_rope_head_dim, dtype=dtype, device=device
            )
        else:
            self.q_pe_padded_buf = None
        self.latent_cache_buf = torch.empty(max_tokens, gen_head_size, dtype=dtype, device=device)
        self.v_proj_output = torch.empty(
            max_tokens, num_heads, v_head_dim, dtype=dtype, device=device
        )

    def plan_host(
        self,
        num_prefill: int,
        num_decode: int,
        max_context_length: int,
        seq_len_with_cache_host: torch.Tensor,
        input_pos_host: torch.Tensor,
        seq_len_host: torch.Tensor,
    ) -> None:
        """Per-forward HOST metadata: pinned tensors for thop.attention.

        Called from ``prepare_trtllm_mla_metadata_host`` before every forward
        (including CUDA graph replays).  Mirrors ``_TrtllmPlanner.plan_host``.
        """
        num_seq = num_prefill + num_decode

        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

        is_capturing = torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up()
        if is_capturing:
            self.host_total_kv_lens[0] = max_context_length * num_prefill
            self.host_total_kv_lens[1] = max_context_length * num_decode
            self.host_past_kv_lengths[:num_seq].fill_(max_context_length)
            self.host_context_lengths[:num_seq].fill_(max_context_length)
            if num_decode > 0:
                cu_kv = self._cu_kv_decode_host
                for i in range(num_decode):
                    cu_kv[i + 1] = (i + 1) * max_context_length
                self.cu_kv_decode[: num_decode + 1].copy_(
                    cu_kv[: num_decode + 1], non_blocking=True
                )
        else:
            self.host_total_kv_lens[0] = seq_len_with_cache_host[:num_prefill].sum()
            self.host_total_kv_lens[1] = seq_len_with_cache_host[num_prefill:num_seq].sum()
            self.host_past_kv_lengths[:num_seq] = input_pos_host[:num_seq]
            self.host_context_lengths[:num_seq] = seq_len_host[:num_seq]
            if num_decode > 0:
                cu_kv = self._cu_kv_decode_host
                decode_lens = seq_len_with_cache_host[num_prefill:num_seq]
                cu_kv[1 : num_decode + 1] = decode_lens.cumsum(0).int()
                self.cu_kv_decode[: num_decode + 1].copy_(
                    cu_kv[: num_decode + 1], non_blocking=True
                )

    def plan_device(
        self,
        num_seq: int,
        block_offset_multiplier: int,
        cu_num_pages: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> None:
        """Per-forward DEVICE metadata: block_offsets and block_ids_per_seq via Triton.

        Called from the ``prepare_trtllm_mla_metadata`` custom op (in the graph).
        Uses ``ragged_to_block_table_triton`` to scatter ``cache_loc`` pages into
        ``block_ids_per_seq`` on the GPU, then derives ``block_offsets``.
        """
        if self.host_pool_mapping.size(0) != block_offset_multiplier:
            self.host_pool_mapping = torch.zeros(
                block_offset_multiplier,
                2,
                dtype=torch.int32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )

        self.block_ids_per_seq[:num_seq].fill_(0)
        torch.ops.auto_deploy.ragged_to_block_table_triton(
            cache_loc, cu_num_pages, self.block_ids_per_seq, num_seq
        )

        k_slice = self.block_offsets[0, :num_seq, 0, :]
        k_slice.copy_(self.block_ids_per_seq[:num_seq])
        k_slice.mul_(block_offset_multiplier)
        self.block_offsets[0, :num_seq, 1, :] = k_slice

    def get_pool_pointers_for_layer(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Return a per-layer ``host_pool_pointers`` tensor for this kv_cache view."""
        ptr = kv_cache.data_ptr()
        t = self._per_layer_pool_ptrs.get(ptr)
        if t is not None:
            return t
        t = torch.zeros(1, 2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned())
        t[0, 0] = ptr
        self._per_layer_pool_ptrs[ptr] = t
        return t


_GlobalTrtllmMLAPlanner = _TrtllmMLAPlanner()


# =============================================================================
# Host-side prepare function
# =============================================================================


def prepare_trtllm_mla_metadata_host(
    batch_info_host: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_host: torch.Tensor,
) -> None:
    """Fill thop-specific HOST metadata (pinned tensors for thop.attention).

    Runs OUTSIDE the CUDA graph before every forward (including replays).
    Mirrors ``prepare_trtllm_metadata_host`` from the standard trtllm backend.
    """
    num_prefill, _, num_decode = batch_info_host.tolist()
    max_context_length, max_blocks_per_seq, _, max_batch_size = max_seq_info_host.tolist()

    _GlobalTrtllmMLAPlanner.reset(torch.device("cuda"), max_batch_size, max_blocks_per_seq)

    _GlobalTrtllmMLAPlanner.plan_host(
        num_prefill=num_prefill,
        num_decode=num_decode,
        max_context_length=max_context_length,
        seq_len_with_cache_host=seq_len_with_cache_host,
        input_pos_host=input_pos_host,
        seq_len_host=seq_len_host,
    )


# =============================================================================
# Device-side prepare function (inserted into the graph)
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_mla_prepare_metadata", mutates_args=())
def prepare_trtllm_mla_metadata(
    batch_info_host: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute block_offsets and block_ids_per_seq for MLA (device-side, part of graph).

    Uses ``ragged_to_block_table_triton`` to scatter ``cache_loc`` pages into
    ``block_ids_per_seq`` on the GPU, then derives ``block_offsets``.
    Returns ``[block_offsets]`` which flows through the graph to each attention op.
    """
    num_prefill, _, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    _, _, block_offset_multiplier, _ = max_seq_info_host.tolist()

    _GlobalTrtllmMLAPlanner.plan_device(
        num_seq=num_seq,
        block_offset_multiplier=block_offset_multiplier,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
    )

    return [_GlobalTrtllmMLAPlanner.block_offsets]


@prepare_trtllm_mla_metadata.register_fake
def prepare_trtllm_mla_metadata_fake(
    batch_info_host: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> List[torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    _, max_blocks_per_seq, _, max_batch_size = max_seq_info_host.tolist()
    return [
        torch.empty(
            1, max_batch_size, 2, max_blocks_per_seq, dtype=torch.int32, device=cache_loc.device
        )
    ]


# =============================================================================
# Helper: call thop.attention with MLA parameters
# =============================================================================


def _call_thop_attention_mla(
    qkv_or_q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor],
    q_pe: Optional[torch.Tensor],
    is_fused_qkv: bool,
    attention_input_type: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
    scale: float,
    quant_mode: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    sequence_length: torch.Tensor,
    context_lengths: torch.Tensor,
    host_past_kv_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    host_total_kv_lens: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    cu_q_seqlens: Optional[torch.Tensor] = None,
    cu_kv_seqlens: Optional[torch.Tensor] = None,
    fmha_scheduler_counter: Optional[torch.Tensor] = None,
    update_kv_cache: bool = True,
    block_ids_per_seq: Optional[torch.Tensor] = None,
    position_embedding_type: int = 0,
    rotary_embedding_dim: int = 0,
    rotary_embedding_base: float = 10000.0,
    rotary_cos_sin: Optional[torch.Tensor] = None,
) -> None:
    """Call thop.attention with MLA parameters enabled.

    Args:
        attention_input_type: 1 = context_only (prefill), 2 = generation_only (decode).
            MLA does not support mixed (0).
        block_ids_per_seq: Override for the planner's block_ids_per_seq. If None,
            uses _GlobalTrtllmMLAPlanner.block_ids_per_seq.
        position_embedding_type: RoPE type for the C++ kernel.  0 = disabled (RoPE
            already applied externally or not needed), 2 = rope_gpt_neox (C++ kernel
            applies RoPE internally from position metadata).
        rotary_embedding_dim: RoPE dimension (qk_rope_head_dim when enabled).
        rotary_embedding_base: RoPE base frequency.
        rotary_cos_sin: Pre-computed [1, max_pos * dim] cos/sin table for the C++
            kernel.  Required when position_embedding_type != 0.
    """
    if block_ids_per_seq is None:
        block_ids_per_seq = _GlobalTrtllmMLAPlanner.block_ids_per_seq

    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    sm_version = get_sm_version()
    if sm_version >= 89:
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    thop.attention(
        qkv_or_q,  # q / fused QKV
        k,  # k
        v,  # v
        output,  # output
        None,  # output_sf
        _GlobalTrtllmMLAPlanner.workspace,  # workspace
        sequence_length,  # sequence_length
        host_past_kv_lengths,  # host_past_key_value_lengths
        host_total_kv_lens,  # host_total_kv_lens
        context_lengths,  # context_lengths
        host_context_lengths,  # host_context_lengths
        host_request_types,  # host_request_types
        kv_cache_block_offsets,  # kv_cache_block_offsets
        host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
        host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
        None,  # cache_indirection
        _GlobalTrtllmMLAPlanner.kv_scale_orig_quant,  # kv_scale_orig_quant
        _GlobalTrtllmMLAPlanner.kv_scale_quant_orig,  # kv_scale_quant_orig
        None,  # out_scale
        None,  # rotary_inv_freq
        rotary_cos_sin,  # rotary_cos_sin
        latent_cache,  # latent_cache (MLA)
        q_pe,  # q_pe (MLA generation)
        block_ids_per_seq,  # block_ids_per_seq
        None,  # attention_sinks
        is_fused_qkv,  # is_fused_qkv
        update_kv_cache,  # update_kv_cache
        1,  # predicted_tokens_per_seq
        0,  # layer_idx
        num_heads,  # num_heads
        num_kv_heads,  # num_kv_heads
        head_size,  # head_size
        tokens_per_block,  # tokens_per_block
        max_num_requests,  # max_num_requests
        max_context_length,  # max_context_length
        max_context_length,  # attention_window_size
        0,  # sink_token_length
        1,  # beam_width
        int(AttentionMaskType.causal),  # mask_type
        quant_mode,  # quant_mode
        scale,  # q_scaling
        position_embedding_type,  # position_embedding_type
        rotary_embedding_dim,  # rotary_embedding_dim
        rotary_embedding_base,  # rotary_embedding_base
        0,  # rotary_embedding_scale_type
        rotary_embedding_scales,  # rotary_embedding_scales
        rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
        False,  # use_paged_context_fmha
        attention_input_type,  # attention_input_type
        True,  # is_mla_enable
        max_num_requests,  # chunked_prefill_buffer_batch_size
        0,  # q_lora_rank
        kv_lora_rank,  # kv_lora_rank
        qk_nope_head_dim,  # qk_nope_head_dim
        qk_rope_head_dim,  # qk_rope_head_dim
        v_head_dim,  # v_head_dim
        None,  # mrope_rotary_cos_sin
        None,  # mrope_position_deltas
        mla_tensor_params,  # mla_tensor_params
        None,  # attention_chunk_size
        None,  # softmax_stats_tensor
        spec_decoding_bool_params,  # spec_decoding_bool_params
        spec_decoding_tensor_params,  # spec_decoding_tensor_params
        None,  # sparse_kv_indices
        None,  # sparse_kv_offsets
        None,  # sparse_attn_indices
        None,  # sparse_attn_offsets
        1,  # sparse_attn_indices_block_size
        0,  # sparse_mla_topk
        None,  # skip_softmax_threshold_scale_factor_prefill
        None,  # skip_softmax_threshold_scale_factor_decode
        None,  # skip_softmax_stat
        cu_q_seqlens,  # cu_q_seqlens
        cu_kv_seqlens,  # cu_kv_seqlens
        fmha_scheduler_counter,  # fmha_scheduler_counter
        None,  # mla_bmm1_scale
        None,  # mla_bmm2_scale
        None,  # quant_q_buffer
    )


# =============================================================================
# Phase-specific handlers
# =============================================================================


def _handle_prefill_thop(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    compressed_kv_flat: torch.Tensor,
    kpe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    latent_cache: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
    quant_mode: int,
    sequence_length: torch.Tensor,
    context_lengths: torch.Tensor,
    host_past_kv_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    host_total_kv_lens: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    position_embedding_type: int = 2,
    rotary_embedding_dim: int = 0,
    rotary_cos_sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched prefill via a single thop.attention call (SEPARATE_Q_K_V layout).

    Expands compressed KV via kv_b_proj, assembles full Q/K/V, and passes the
    entire batch to thop.attention with attention_input_type=context_only.
    The C++ kernel handles multi-sequence batching internally via
    host_context_lengths and writes latent_cache to the paged KV cache.

    RoPE handling depends on ``position_embedding_type``:
    - 2 (rope_gpt_neox): the C++ kernel's ``invokeMLARopeContext`` applies
      RoPE to k_pe (from latent_cache) before writing to the KV cache, and
      to Q's rope portion for FMHA.  Requires a valid ``rotary_cos_sin``
      table and **un-RoPE'd** q_pe, kpe, and latent_cache.
    - 0 (disabled): RoPE already applied externally.  Q/K/V carry post-RoPE
      values and latent_cache is written to the cache as-is.

    Uses q_scaling=1.0 so the C++ kernel computes softmax_scale = 1/sqrt(head_size),
    matching the standard 1/sqrt(qk_head_dim) model scale.
    """
    dtype = q_nope_flat.dtype
    device = q_nope_flat.device

    q = torch.empty(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    q[:, :, :qk_nope_head_dim] = q_nope_flat
    q[:, :, qk_nope_head_dim:] = q_pe_flat
    q = q.view(num_tokens, num_heads * qk_head_dim)

    kv = torch.nn.functional.linear(compressed_kv_flat, kv_b_proj_weight)
    kv = kv.view(num_tokens, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope = kv[:, :, :qk_nope_head_dim]
    v = kv[:, :, qk_nope_head_dim:].contiguous().view(num_tokens, num_heads * v_head_dim)

    k = torch.empty(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    k[:, :, :qk_nope_head_dim] = k_nope
    k[:, :, qk_nope_head_dim:] = kpe_flat.view(num_tokens, 1, qk_rope_head_dim)
    k = k.view(num_tokens, num_heads * qk_head_dim)

    output = torch.empty(num_tokens, num_heads * v_head_dim, dtype=dtype, device=device)

    _call_thop_attention_mla(
        q,
        k,
        v,
        output,
        latent_cache,
        None,
        False,
        1,
        num_heads,
        num_kv_heads,
        qk_head_dim,
        tokens_per_block,
        max_num_requests,
        max_context_length,
        1.0,
        quant_mode,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        sequence_length,
        context_lengths,
        host_past_kv_lengths,
        host_context_lengths,
        host_request_types,
        host_total_kv_lens,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        position_embedding_type=position_embedding_type,
        rotary_embedding_dim=rotary_embedding_dim,
        rotary_cos_sin=rotary_cos_sin,
    )

    return output


def _get_cache_2d_view(kv_cache: torch.Tensor) -> tuple:
    """Return (kv_cache_2d, rows_per_block) for flat index_copy_ writes."""
    planner = _GlobalTrtllmMLAPlanner
    if planner._cache_rows_per_block == 0:
        planner._cache_rows_per_block = kv_cache.stride(0) // kv_cache.stride(3)
    rpb = planner._cache_rows_per_block
    D = kv_cache.shape[-1]
    tpb = kv_cache.shape[3]
    num_rows = (kv_cache.shape[0] - 1) * rpb + tpb
    kv_cache_2d = torch.as_strided(
        kv_cache,
        size=(num_rows, D),
        stride=(kv_cache.stride(3), kv_cache.stride(4)),
    )
    return kv_cache_2d, rpb


def _write_decode_latent_to_cache(
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    num_decode: int,
    num_prefill: int,
    seq_len_with_cache: torch.Tensor,
    tokens_per_block: int,
) -> None:
    """Write decode tokens' latent_cache to the paged KV cache.

    Computes write indices from ``seq_len_with_cache`` and ``block_ids_per_seq``
    (filled by the device-side ``prepare_trtllm_mla_metadata`` op), then uses
    ``as_strided`` + ``index_copy_`` for an efficient single-kernel scatter.
    """
    planner = _GlobalTrtllmMLAPlanner
    device = kv_cache.device
    kv_cache_2d, rpb = _get_cache_2d_view(kv_cache)

    num_seq = num_prefill + num_decode
    positions = seq_len_with_cache[num_prefill:num_seq] - 1
    page_in_seq = positions // tokens_per_block
    slot = positions % tokens_per_block

    seq_range = torch.arange(num_decode, device=device, dtype=torch.long)
    page_ids = planner.block_ids_per_seq[num_prefill:num_seq][seq_range, page_in_seq.long()]
    flat_idx = page_ids.long() * rpb + slot

    kv_cache_2d.index_copy_(0, flat_idx, latent_cache)


def _apply_rope_from_table(
    q_pe: torch.Tensor,
    kpe: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    positions: torch.Tensor,
    qk_rope_head_dim: int,
) -> tuple:
    """Apply GPTJ-style (interleaved) RoPE using a flat cos/sin table.

    The fuse_rope_into_trtllm_mla transform reverses the AD weight
    de-interleaving at compile time, so q_pe / kpe arrive in GPTJ
    (interleaved) layout at runtime.  This function applies GPTJ rotation
    (pairing adjacent elements (2j, 2j+1)) to match the mla_rope_generation
    decode kernel, ensuring prefill and decode produce consistent cache data.

    Args:
        q_pe: [T, H, D] pre-RoPE query positional component (GPTJ layout).
        kpe: [T, D] pre-RoPE key positional encoding (GPTJ layout).
        rotary_cos_sin: [1, max_pos * D * 2] flat float32 table with
            D float2 (cos, sin) entries per position.
        positions: [T] int position IDs for each token.
        qk_rope_head_dim: D, the RoPE head dimension.

    Returns:
        (q_pe_rotated, kpe_rotated) with GPTJ RoPE applied.
    """
    half = qk_rope_head_dim // 2
    table = rotary_cos_sin.view(-1, half, 2)  # [max_pos, D/2, 2]
    cos_half = table[positions.long(), :, 0].to(q_pe.dtype)  # [T, D/2]
    sin_half = table[positions.long(), :, 1].to(q_pe.dtype)  # [T, D/2]

    def _rotate_interleaved(x, cos_h, sin_h):
        pairs = x.unflatten(-1, (-1, 2))  # [..., D/2, 2]
        even, odd = pairs[..., 0], pairs[..., 1]
        r_even = even * cos_h - odd * sin_h
        r_odd = even * sin_h + odd * cos_h
        return torch.stack([r_even, r_odd], dim=-1).flatten(-2)

    cos_q = cos_half.unsqueeze(1)  # [T, 1, D/2]
    sin_q = sin_half.unsqueeze(1)  # [T, 1, D/2]
    q_pe_rotated = _rotate_interleaved(q_pe, cos_q, sin_q)
    kpe_rotated = _rotate_interleaved(kpe, cos_half, sin_half)

    return q_pe_rotated, kpe_rotated


def _handle_decode_impl(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    num_tokens: int,
    num_prefill: int,
    num_heads: int,
    num_kv_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    head_size: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
    scale: float,
    quant_mode: int,
    sequence_length: torch.Tensor,
    context_lengths: torch.Tensor,
    host_past_kv_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    host_total_kv_lens: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    *,
    rotary_cos_sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Handle decode: weight absorption + latent-space attention + output projection.

    Follows the PyTorch backend pattern: ``bmm_out`` writes the Q absorption
    result directly into a pre-allocated ``fused_q`` slice (no intermediate
    tensor, no ``torch.cat``), and the V projection also uses ``bmm_out`` into
    a pre-allocated buffer.

    On SM100+ the FMHA MLA decode kernel requires a power-of-2 Q/KV head ratio.
    When ``num_heads / num_kv_heads`` is not a power of two (e.g. GLM-4.7-Flash
    with 20 heads), the Q tensor is zero-padded to the next valid head count and
    the attention output is sliced back before the V projection.

    When ``rotary_cos_sin`` is provided, uses ``mla_rope_generation`` to fuse
    cache write + RoPE + q_pe copy + scheduler fill into one kernel.  Otherwise
    copies q_pe manually and zeros the scheduler counter.
    """
    planner = _GlobalTrtllmMLAPlanner
    gen_head_size = kv_lora_rank + qk_rope_head_dim
    padded_num_heads = planner._decode_padded_num_heads
    needs_head_padding = padded_num_heads != num_heads

    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_kn = weight_reshaped[:, :qk_nope_head_dim, :].contiguous()
    w_v_t = weight_reshaped[:, qk_nope_head_dim:, :].transpose(1, 2).contiguous()

    # Build fused_q = [q_absorbed | q_pe] into pre-allocated buffer.
    # bmm_out writes q_absorbed directly into the left slice of fused_q.
    fused_q_flat = planner.fused_q_flat[:num_tokens]
    fused_q_view = fused_q_flat.view(
        num_tokens,
        padded_num_heads if needs_head_padding else num_heads,
        gen_head_size,
    )

    q_nope_t = q_nope_flat.transpose(0, 1)
    q_absorbed_target = fused_q_view[:, :num_heads, :kv_lora_rank].transpose(0, 1)
    torch.ops.trtllm.bmm_out(q_nope_t, w_kn, q_absorbed_target)

    cu_q = planner.cu_q_decode[: num_tokens + 1]
    cu_kv = planner.cu_kv_decode[: num_tokens + 1]

    if needs_head_padding:
        q_pe_padded = planner.q_pe_padded_buf[:num_tokens]
        q_pe_padded[:, :num_heads, :] = q_pe_flat
    else:
        q_pe_padded = q_pe_flat

    if rotary_cos_sin is not None:
        # Fused path: mla_rope_generation fills q_pe in fused_q, writes cache
        # with RoPE on kpe, and fills cu_q/cu_kv/scheduler in one kernel.
        # thop.attention with update_kv_cache=True also writes the cache
        # (required by C++ assertion), so the cache is written twice — this
        # benign double-write is cheaper than separate Python kernels.
        torch.ops.trtllm.mla_rope_generation(
            fused_q_view,
            q_pe_padded,
            latent_cache,
            rotary_cos_sin,
            cu_q,
            cu_kv,
            planner.fmha_scheduler_counter_decode,
            None,  # mla_bmm1_scale (non-FP8)
            None,  # mla_bmm2_scale (non-FP8)
            None,  # quant_q_buffer (non-FP8)
            sequence_length,
            host_past_kv_lengths,
            host_context_lengths,
            num_prefill,  # num_contexts
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            planner.kv_scale_orig_quant,
            planner.kv_scale_quant_orig,
            None,  # out_scale
            planner.block_ids_per_seq,
            [None, None],  # mla_tensor_params (helix)
            1,  # predicted_tokens_per_seq
            0,  # layer_idx
            padded_num_heads,
            num_kv_heads,
            gen_head_size,
            tokens_per_block,
            max_context_length,  # attention_window_size
            0,  # sink_token_length
            1,  # beam_width
            quant_mode,
            scale,  # q_scaling
            0,  # q_lora_rank
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,  # v_head_dim (in latent space = kv_lora_rank)
        )
    else:
        # Non-fused path: copy q_pe into the rope portion of fused_q and
        # zero the scheduler counter manually.
        fused_q_view[:, :num_heads, kv_lora_rank:] = q_pe_flat
        planner.fmha_scheduler_counter_decode.fill_(0)

    output_latent = planner.output_latent[:num_tokens]

    _call_thop_attention_mla(
        fused_q_flat,
        None,
        None,
        output_latent,
        latent_cache,
        q_pe_padded,
        True,
        2,  # attention_input_type = generation_only
        padded_num_heads,
        num_kv_heads,
        gen_head_size,
        tokens_per_block,
        max_num_requests,
        max_context_length,
        scale,
        quant_mode,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        sequence_length,
        context_lengths,
        host_past_kv_lengths,
        host_context_lengths,
        host_request_types,
        host_total_kv_lens,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        cu_q_seqlens=cu_q,
        cu_kv_seqlens=cu_kv,
        fmha_scheduler_counter=planner.fmha_scheduler_counter_decode,
    )

    # V projection: project from latent space back to v_head_dim.
    output_reshaped = output_latent.view(num_tokens, padded_num_heads, kv_lora_rank)
    if needs_head_padding:
        output_reshaped = output_reshaped[:, :num_heads, :].contiguous()
    v_proj_out = planner.v_proj_output[:num_tokens]
    torch.ops.trtllm.bmm_out(
        output_reshaped.transpose(0, 1),
        w_v_t,
        v_proj_out.transpose(0, 1),
    )
    return v_proj_out.reshape(num_tokens, num_heads * v_head_dim)


# =============================================================================
# Shared MLA attention implementation
# =============================================================================


def _mla_with_cache_impl(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
    rotary_cos_sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shared implementation for both MLA attention ops (with and without fused RoPE).

    Host-side metadata (``host_past_kv_lengths``, ``host_context_lengths``, etc.)
    is read from the planner singleton (filled by ``plan_host``).
    ``kv_cache_block_offsets`` flows through the graph from the device-side
    ``prepare_trtllm_mla_metadata`` op.

    When ``rotary_cos_sin`` is None, q_pe/kpe arrive post-RoPE and latent_cache
    is written to the paged KV cache as-is.  When provided, q_pe/kpe are pre-RoPE:
    prefill applies RoPE in Python; decode uses ``mla_rope_generation`` to fuse
    cache write + RoPE + q_pe copy + scheduler fill into one kernel.
    """
    fused_rope = rotary_cos_sin is not None

    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode
    max_context_length = int(max_seq_info_host[0])
    max_num_requests = int(max_seq_info_host[3])

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    num_kv_heads = 1
    tokens_per_block = kv_cache.shape[3]  # HND: [blocks, kv_factor, heads, tpb, head_dim]

    gen_head_size = kv_lora_rank + qk_rope_head_dim

    planner = _GlobalTrtllmMLAPlanner
    if planner._tokens_per_block != tokens_per_block:
        planner._tokens_per_block = tokens_per_block
    host_kv_cache_pool_pointers = planner.get_pool_pointers_for_layer(kv_cache)

    quant_mode = 0
    if kv_cache.dtype == torch.float8_e4m3fn:
        if planner.kv_scale_orig_quant is None:
            planner.kv_scale_orig_quant = torch.tensor(
                [1.0], dtype=torch.float32, device=q_nope.device
            )
            planner.kv_scale_quant_orig = torch.tensor(
                [1.0], dtype=torch.float32, device=q_nope.device
            )
        quant_mode = int(QuantMode.FP8_KV_CACHE)

    # Flatten from [B, S, ...] to [bs, ...] using actual tensor dims (not
    # num_tokens from metadata) because piecewise CUDA graphs may pad inputs
    # to bucket size.  Subsequent slicing with num_tokens / num_prefill_tokens
    # selects only the real tokens within the padded buffer.
    bs = b * s
    q_nope_c = q_nope if q_nope.is_contiguous() else q_nope.contiguous()
    q_pe_c = q_pe if q_pe.is_contiguous() else q_pe.contiguous()
    compressed_kv_c = compressed_kv if compressed_kv.is_contiguous() else compressed_kv.contiguous()
    kpe_c = kpe if kpe.is_contiguous() else kpe.contiguous()

    q_nope_flat = q_nope_c.view(bs, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe_c.view(bs, num_heads, qk_rope_head_dim)
    compressed_kv_flat = compressed_kv_c.view(bs, kv_lora_rank)
    kpe_flat = kpe_c.view(bs, qk_rope_head_dim)

    if num_decode > 0:
        planner.ensure_decode_buffers(
            q_nope.device,
            max_num_requests,
            num_heads,
            num_kv_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            v_head_dim,
            q_nope.dtype,
        )

    # Build latent_cache: [bs, kv_lora_rank + qk_rope_head_dim]
    # Use torch.cat with out= for decode-only to write into pre-allocated buffer
    # with a single kernel instead of two slice-assign copies.
    if num_prefill == 0:
        latent_buf = planner.latent_cache_buf[:num_tokens]
        torch.cat([compressed_kv_flat[:num_tokens], kpe_flat[:num_tokens]], dim=-1, out=latent_buf)
        latent_cache = latent_buf
    else:
        latent_cache = torch.cat(
            [compressed_kv_flat[:num_tokens], kpe_flat[:num_tokens]], dim=-1
        ).contiguous()

    sequence_length = seq_len_with_cache[:num_seq]
    context_lengths = seq_len[:num_seq]
    host_past_kv_lengths = planner.host_past_kv_lengths[:num_seq]
    host_context_lengths = planner.host_context_lengths[:num_seq]
    host_request_types = planner.host_request_types[:num_seq]
    host_total_kv_lens = planner.host_total_kv_lens
    host_kv_cache_pool_mapping = planner.host_pool_mapping

    # thop.attention's MLA generation runners compute the effective softmax
    # scale from q_scaling and qk_head_dim (NOT gen_head_size):
    #   FlashMLA / XQA: softmax_scale = 1 / (q_scaling * sqrt(qk_head_dim))
    #   trtllm-gen:     mScaleQ = q_scaling * sqrt(qk_head_dim) / sqrt(gen_head_size)
    # To get effective softmax_scale == model's `scale`:
    #   q_scaling = 1 / (scale * sqrt(qk_head_dim))
    thop_q_scaling = 1.0 / (scale * math.sqrt(qk_head_dim))

    def _make_shared_metadata():
        return (
            num_heads,
            num_kv_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            gen_head_size,
            tokens_per_block,
            max_num_requests,
            max_context_length,
            thop_q_scaling,
            quant_mode,
            sequence_length,
            context_lengths,
            host_past_kv_lengths,
            host_context_lengths,
            host_request_types,
            host_total_kv_lens,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        )

    # -- Prefill helper -------------------------------------------------------

    def _prefill_with_cache_write(q_n, q_p, ckv, kpe_pre, lc, n_tok, n_pf):
        """Run thop.attention prefill (handles FMHA + cache write).

        The C++ kernel's ``invokeMLARopeContext`` ALWAYS requires
        ``position_embedding_type=2`` and a valid ``rotary_cos_sin`` table for
        MLA context — it uses them to apply RoPE to the kpe portion of
        ``latent_cache`` before writing to the paged KV cache.  Q and K
        tensors for FMHA carry whatever RoPE state the model already applied.
        """
        if fused_rope:
            rope_table = rotary_cos_sin
        else:
            rope_table = planner.get_or_create_rotary_cos_sin(
                max_context_length,
                qk_rope_head_dim,
                10000.0,
                q_n.device,
            )

        return _handle_prefill_thop(
            q_n,
            q_p,
            ckv,
            kpe_pre,
            kv_b_proj_weight,
            lc,
            n_tok,
            num_heads,
            num_kv_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            tokens_per_block,
            max_num_requests,
            max_context_length,
            quant_mode,
            sequence_length,
            context_lengths,
            host_past_kv_lengths,
            host_context_lengths,
            host_request_types,
            host_total_kv_lens,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            position_embedding_type=2,
            rotary_embedding_dim=qk_rope_head_dim,
            rotary_cos_sin=rope_table,
        )

    # -- Decode helper --------------------------------------------------------

    def _do_decode(q_n, q_p, lc, n_tok, n_pf):
        """Write cache (non-fused only) and run decode attention."""
        if not fused_rope:
            _write_decode_latent_to_cache(
                lc, kv_cache, n_tok, n_pf, seq_len_with_cache, tokens_per_block
            )
        return _handle_decode_impl(
            q_n,
            q_p,
            kv_b_proj_weight,
            lc,
            kv_cache,
            n_tok,
            n_pf,
            *_make_shared_metadata(),
            rotary_cos_sin=rotary_cos_sin,
        )

    # -- 3-way dispatch -------------------------------------------------------
    # Allocate output with bs rows (may include CG padding); only real-token
    # positions are filled — padding stays zero, matching the FI MLA pattern.
    y = torch.zeros(bs, num_heads * v_head_dim, dtype=q_nope_flat.dtype, device=q_nope_flat.device)

    if num_prefill > 0 and num_decode > 0:
        y[:num_prefill_tokens] = _prefill_with_cache_write(
            q_nope_flat[:num_prefill_tokens],
            q_pe_flat[:num_prefill_tokens],
            compressed_kv_flat[:num_prefill_tokens],
            kpe_flat[:num_prefill_tokens],
            latent_cache[:num_prefill_tokens],
            num_prefill_tokens,
            num_prefill,
        )
        y[num_prefill_tokens:num_tokens] = _do_decode(
            q_nope_flat[num_prefill_tokens:num_tokens],
            q_pe_flat[num_prefill_tokens:num_tokens],
            latent_cache[num_prefill_tokens:num_tokens],
            num_decode,
            num_prefill,
        )
    elif num_prefill > 0:
        y[:num_tokens] = _prefill_with_cache_write(
            q_nope_flat[:num_tokens],
            q_pe_flat[:num_tokens],
            compressed_kv_flat[:num_tokens],
            kpe_flat[:num_tokens],
            latent_cache,
            num_prefill_tokens,
            num_prefill,
        )
    else:
        y[:num_tokens] = _do_decode(
            q_nope_flat[:num_tokens],
            q_pe_flat[:num_tokens],
            latent_cache,
            num_tokens,
            0,
        )

    return y.view(b, s, num_heads, v_head_dim)


def _mla_with_cache_fake_impl(
    q_nope: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Shared fake implementation for torch.compile tracing."""
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim
    return q_nope.new_empty(
        q_nope.shape[0],
        q_nope.shape[1],
        q_nope.shape[2],
        v_head_dim,
    ).contiguous()


# =============================================================================
# Cached MLA attention ops (thin wrappers around _mla_with_cache_impl)
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_mla_with_cache", mutates_args=("kv_cache",))
def trtllm_mla_with_cache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    # Standard metadata (4 args — matches trtllm attention pattern)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # Extra metadata from prepare_trtllm_mla_metadata (1 arg)
    kv_cache_block_offsets: torch.Tensor,
    # Cache
    kv_cache: torch.Tensor,
    # Constants
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """TRT-LLM MLA attention with paged latent cache (post-RoPE inputs)."""
    return _mla_with_cache_impl(
        q_nope,
        q_pe,
        compressed_kv,
        kpe,
        kv_b_proj_weight,
        batch_info_host,
        seq_len,
        seq_len_with_cache,
        max_seq_info_host,
        kv_cache_block_offsets,
        kv_cache,
        scale,
        kv_lora_rank,
    )


@trtllm_mla_with_cache.register_fake
def trtllm_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return _mla_with_cache_fake_impl(q_nope, kv_b_proj_weight)


@torch.library.custom_op(
    "auto_deploy::trtllm_mla_fused_rope_with_cache", mutates_args=("kv_cache",)
)
def trtllm_mla_fused_rope_with_cache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    # Standard metadata (4 args — matches trtllm attention pattern)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # Extra metadata from prepare_trtllm_mla_metadata (1 arg)
    kv_cache_block_offsets: torch.Tensor,
    # Cache
    kv_cache: torch.Tensor,
    # Constants
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """TRT-LLM MLA attention with fused RoPE and paged latent cache (pre-RoPE inputs)."""
    return _mla_with_cache_impl(
        q_nope,
        q_pe,
        compressed_kv,
        kpe,
        kv_b_proj_weight,
        batch_info_host,
        seq_len,
        seq_len_with_cache,
        max_seq_info_host,
        kv_cache_block_offsets,
        kv_cache,
        scale,
        kv_lora_rank,
        rotary_cos_sin=rotary_cos_sin,
    )


@trtllm_mla_fused_rope_with_cache.register_fake
def trtllm_mla_fused_rope_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return _mla_with_cache_fake_impl(q_nope, kv_b_proj_weight)


# =============================================================================
# AttentionDescriptor
# =============================================================================


@AttentionRegistry.register("trtllm_mla")
class TrtllmMLAAttention(AttentionDescriptor):
    """TRT-LLM MLA attention backend for Auto-Deploy.

    Uses thop.attention with is_mla_enable=True for both prefill and decode.
    Stores compressed latent representations (compressed_kv + k_pe) in paged KV cache.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 5

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.trtllm_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return [
            "batch_info_host",
            "seq_len",
            "seq_len_with_cache",
            "max_seq_info_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls,
        any_source_attn_node: Node,
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Register the device-side prepare op that computes block_offsets via Triton."""
        return (
            torch.ops.auto_deploy.trtllm_mla_prepare_metadata.default,
            1,
            [],
        )

    @classmethod
    def get_cache_initializers(
        cls,
        source_attn_node: Node,
        cache_config: KvCacheConfig,
    ) -> ResourceHandlerDict:
        """Return KV cache handler for MLA latent cache.

        MLA stores [compressed_kv | k_pe] per token with num_kv_heads=1.
        """
        q_nope_fake: FakeTensor = source_attn_node.args[0].meta["val"]
        compressed_kv_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        kpe_fake: FakeTensor = source_attn_node.args[3].meta["val"]
        kv_b_proj_fake: FakeTensor = source_attn_node.args[4].meta["val"]

        num_heads = q_nope_fake.shape[2]
        qk_nope_head_dim = q_nope_fake.shape[-1]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]
        kv_head_dim = kv_b_proj_fake.shape[0] // num_heads
        v_head_dim = kv_head_dim - qk_nope_head_dim

        if v_head_dim != qk_nope_head_dim:
            raise RuntimeError(
                f"trtllm_mla backend requires v_head_dim == qk_nope_head_dim, "
                f"but got v_head_dim={v_head_dim}, qk_nope_head_dim={qk_nope_head_dim}. "
                f"The C++ FMHA context kernel derives headSizeV from qk_nope_head_dim, "
                f"which produces incorrect results when they differ. "
                f"Use the flashinfer_mla backend for this model instead."
            )

        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads=1,
                head_dim=kv_lora_rank + qk_rope_head_dim,
                dtype=cls.resolve_cache_dtype(
                    cache_config.dtype,
                    compressed_kv_fake.dtype,
                ),
                kv_factor=1,
                kv_layout="HND",
            )
        }

    @classmethod
    def get_host_prepare_metadata_function(
        cls,
    ) -> Optional[PrepareMetadataHostCallable]:
        return prepare_trtllm_mla_metadata_host

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        # scale is at positional index 6 in torch_mla(q_nope, q_pe, compressed_kv,
        # kpe, kv_b_proj_weight, is_causal, scale, layout); fall back to kwargs.
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)
        return [scale, kv_lora_rank]
