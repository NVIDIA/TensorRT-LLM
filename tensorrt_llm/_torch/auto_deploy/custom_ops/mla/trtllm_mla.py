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

  **SDPA fallback**: All MLA prefill uses PyTorch SDPA.  The AD thop.attention
  context call has a parameter mismatch vs the PyTorch backend's TrtllmAttention
  wrapper that causes an illegal memory access on both SM90 and SM100 (root-cause
  TBD).  The thop MLA *decode* kernel works correctly.

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
from typing import List, Optional

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
    PrepareMetadataHostCallable,
    ResourceHandlerDict,
)

# =============================================================================
# Helpers
# =============================================================================


# =============================================================================
# Module-level planner
# =============================================================================


class _TrtllmMLAPlanner:
    """Minimal planner for TRT-LLM MLA attention backend.

    Mirrors ``_TrtllmPlanner`` from the standard trtllm backend. Manages persistent
    buffers for thop.attention metadata and per-layer pool pointers for CUDA graph
    compatibility.

    Decode-specific buffers (cu_q_decode, cu_kv_decode, fmha_scheduler_counter_decode,
    output_latent, fused_q_flat, latent_cache_buf) are pre-allocated once and reused
    across all layers within a step, eliminating per-layer arange/zeros/cat/empty
    kernel launches.

    Cache write indices (decode_page_idx, decode_slot_idx) are pre-computed on the
    host during plan() and copied to GPU, eliminating per-layer subtract/div/mod/arange
    kernel launches in the decode hot path.  Page IDs are resolved once per step
    and combined with the pool stride ratio into a flat write index so that
    per-layer cache writes use a single ``index_copy_`` via ``as_strided``,
    replacing multi-dim ``index_put`` and its internal bf16 copy overhead.
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
        self.latent_cache_buf: Optional[torch.Tensor] = None
        self._cu_kv_decode_host: Optional[torch.Tensor] = None
        self._decode_buf_max_tokens: int = 0
        self._decode_buf_num_heads: int = 0
        self._decode_buf_kv_lora_rank: int = 0
        self._decode_buf_rope_dim: int = 0
        self._decode_buf_dtype: Optional[torch.dtype] = None

        # Pre-computed cache write indices (host→GPU, computed in plan())
        self.decode_page_idx: Optional[torch.Tensor] = None
        self.decode_slot_idx: Optional[torch.Tensor] = None
        self._decode_page_idx_host: Optional[torch.Tensor] = None
        self._decode_slot_idx_host: Optional[torch.Tensor] = None
        self._tokens_per_block: int = 0

        # Pre-resolved page IDs and flat write indices for cache writes.
        # page_ids are resolved once per step in plan(); the flat write
        # index (page_id * rows_per_block + slot) is computed once per step
        # (either in plan() or lazily on first layer) so that per-layer
        # writes use a single index_copy_ via an as_strided view.
        self.decode_cache_page_ids: Optional[torch.Tensor] = None
        self.decode_flat_write_idx: Optional[torch.Tensor] = None
        self._seq_range_buf: Optional[torch.Tensor] = None
        self._cache_rows_per_block: int = 0
        self._flat_write_idx_dirty: bool = True

        # Per-layer weight cache: dict[data_ptr → (w_kn, w_v_t)].
        # Each layer has a different kv_b_proj_weight; caching by data_ptr
        # ensures the .contiguous() materialisation runs only once (at warmup).
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Pre-allocated V-projection output buffer (allocated in ensure_decode_buffers)
        self.v_proj_output: Optional[torch.Tensor] = None

        # Pre-computed RoPE cos/sin table in the flat interleaved format expected
        # by torch.ops.trtllm.mla_rope_generation:
        #   [1, max_pos * qk_rope_head_dim * 2] float32
        # Set by the fuse_rope_into_trtllm_mla graph transform.
        self.rotary_cos_sin: Optional[torch.Tensor] = None

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
        self.cu_q_seqlens: Optional[torch.Tensor] = None
        self.cu_kv_seqlens: Optional[torch.Tensor] = None
        self.fmha_scheduler_counter = torch.zeros(1, dtype=torch.int32, device=device)

        self.decode_page_idx = torch.zeros(max_batch, dtype=torch.int64, device=device)
        self.decode_slot_idx = torch.zeros(max_batch, dtype=torch.int64, device=device)
        self._decode_page_idx_host = torch.zeros(
            max_batch, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self._decode_slot_idx_host = torch.zeros(
            max_batch, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self.decode_cache_page_ids = torch.zeros(max_batch, dtype=torch.int32, device=device)
        self.decode_flat_write_idx = torch.zeros(max_batch, dtype=torch.int64, device=device)
        self._seq_range_buf = torch.arange(max_batch, dtype=torch.int64, device=device)

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
        """Allocate or grow decode-path scratch buffers (once, then reused)."""
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

        self._decode_buf_max_tokens = max_tokens
        self._decode_buf_num_heads = num_heads
        self._decode_buf_kv_lora_rank = kv_lora_rank
        self._decode_buf_rope_dim = qk_rope_head_dim
        self._decode_buf_dtype = dtype
        gen_head_size = kv_lora_rank + qk_rope_head_dim

        self.cu_q_decode = (
            torch.arange(max_tokens + 1, dtype=torch.int32, device=device) * num_heads
        )
        self.output_latent = torch.empty(
            max_tokens, num_heads * kv_lora_rank, dtype=dtype, device=device
        )
        self.fused_q_flat = torch.empty(
            max_tokens, num_heads * gen_head_size, dtype=dtype, device=device
        )
        self.latent_cache_buf = torch.empty(max_tokens, gen_head_size, dtype=dtype, device=device)
        self.v_proj_output = torch.empty(
            max_tokens, num_heads, v_head_dim, dtype=dtype, device=device
        )

    def get_weight_matrices(
        self,
        kv_b_proj_weight: torch.Tensor,
        num_heads: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
    ):
        """Return cached contiguous w_kn and w_v_t weight slices.

        Uses a per-``data_ptr`` dict so every layer's weight is materialised
        exactly once (at warmup) instead of every step.
        """
        ptr = kv_b_proj_weight.data_ptr()
        cached = self._weight_cache.get(ptr)
        if cached is not None:
            return cached
        weight_reshaped = kv_b_proj_weight.view(
            num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )
        # w_kn: [H, qk_nope_head_dim, kv_lora_rank] — used by bmm_out
        w_kn = weight_reshaped[:, :qk_nope_head_dim, :].contiguous()
        # w_v_t: [H, kv_lora_rank, v_head_dim] — used by bmm_out
        w_v_t = weight_reshaped[:, qk_nope_head_dim:, :].transpose(1, 2).contiguous()
        self._weight_cache[ptr] = (w_kn, w_v_t)
        return w_kn, w_v_t

    def plan(
        self,
        num_prefill: int,
        num_decode: int,
        max_context_length: int,
        block_offset_multiplier: int,
        seq_len_with_cache_host: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        input_pos_host: torch.Tensor,
        seq_len_host: torch.Tensor,
    ) -> None:
        """Per-forward host metadata: fills host_request_types, block_offsets, etc.

        Also pre-computes decode cache write indices (page_idx, slot_idx) on the host
        and copies them to GPU, eliminating per-layer subtract/div/mod GPU kernels.
        """
        num_seq = num_prefill + num_decode

        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

        if self.host_pool_mapping.size(0) != block_offset_multiplier:
            self.host_pool_mapping = torch.zeros(
                block_offset_multiplier,
                2,
                dtype=torch.int32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )

        block_offsets = self.block_offsets
        total_pages = int(cu_num_pages_host[num_seq])
        base_offsets = cache_loc[:total_pages] * block_offset_multiplier

        pages_per_seq = cu_num_pages_host[1 : num_seq + 1] - cu_num_pages_host[:num_seq]
        seq_idx = torch.repeat_interleave(torch.arange(num_seq, dtype=torch.int), pages_per_seq)
        pg_idx = torch.cat([torch.arange(n, dtype=torch.int) for n in pages_per_seq.tolist()])

        block_offsets[0, seq_idx, 0, pg_idx] = base_offsets
        block_offsets[0, seq_idx, 1, pg_idx] = base_offsets

        self.block_ids_per_seq.fill_(0)
        self.block_ids_per_seq[seq_idx, pg_idx] = cache_loc[:total_pages]

        tokens_per_block = self._tokens_per_block

        is_capturing = torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up()
        if is_capturing:
            self.host_total_kv_lens[0] = max_context_length * num_prefill
            self.host_total_kv_lens[1] = max_context_length * num_decode
            self.host_past_kv_lengths[:num_seq].fill_(max_context_length)
            self.host_context_lengths[:num_seq].fill_(max_context_length)
            if num_decode > 0 and tokens_per_block > 0:
                positions = max_context_length - 1
                self._decode_page_idx_host[:num_decode].fill_(positions // tokens_per_block)
                self._decode_slot_idx_host[:num_decode].fill_(positions % tokens_per_block)
                self.decode_page_idx[:num_decode].copy_(
                    self._decode_page_idx_host[:num_decode], non_blocking=True
                )
                self.decode_slot_idx[:num_decode].copy_(
                    self._decode_slot_idx_host[:num_decode], non_blocking=True
                )
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
            if num_decode > 0 and tokens_per_block > 0:
                positions = seq_len_with_cache_host[num_prefill:num_seq] - 1
                self._decode_page_idx_host[:num_decode] = positions // tokens_per_block
                self._decode_slot_idx_host[:num_decode] = positions % tokens_per_block
                self.decode_page_idx[:num_decode].copy_(
                    self._decode_page_idx_host[:num_decode], non_blocking=True
                )
                self.decode_slot_idx[:num_decode].copy_(
                    self._decode_slot_idx_host[:num_decode], non_blocking=True
                )
            if num_decode > 0:
                cu_kv = self._cu_kv_decode_host
                decode_lens = seq_len_with_cache_host[num_prefill:num_seq]
                cu_kv[1 : num_decode + 1] = decode_lens.cumsum(0).int()
                self.cu_kv_decode[: num_decode + 1].copy_(
                    cu_kv[: num_decode + 1], non_blocking=True
                )

        # Pre-resolve page IDs once per step (outside the CUDA graph) so
        # per-layer cache writes skip the 2D gather from block_ids_per_seq.
        # When the stride ratio is known (after the first forward pass),
        # also compute the flat write index here to avoid per-layer arithmetic.
        self._flat_write_idx_dirty = True
        if num_decode > 0 and tokens_per_block > 0:
            seq_range = self._seq_range_buf[num_prefill : num_prefill + num_decode]
            page_idx = self.decode_page_idx[:num_decode]
            slot_idx = self.decode_slot_idx[:num_decode]
            page_ids = self.block_ids_per_seq[seq_range, page_idx]
            self.decode_cache_page_ids[:num_decode] = page_ids
            if self._cache_rows_per_block > 0:
                self.decode_flat_write_idx[:num_decode] = (
                    page_ids.long() * self._cache_rows_per_block + slot_idx
                )
                self._flat_write_idx_dirty = False

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
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_host: torch.Tensor,
) -> None:
    """Fill thop-specific host metadata and compute block_offsets for MLA.

    Identical signature to ``prepare_trtllm_metadata_host`` but operates on the
    MLA-specific planner instance. Runs OUTSIDE the CUDA graph before every forward.
    """
    num_prefill, _, num_decode = batch_info_host.tolist()
    max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size = (
        max_seq_info_host.tolist()
    )

    _GlobalTrtllmMLAPlanner.reset(cache_loc.device, max_batch_size, max_blocks_per_seq)

    _GlobalTrtllmMLAPlanner.plan(
        num_prefill=num_prefill,
        num_decode=num_decode,
        max_context_length=max_context_length,
        block_offset_multiplier=block_offset_multiplier,
        seq_len_with_cache_host=seq_len_with_cache_host,
        cu_num_pages_host=cu_num_pages_host,
        cache_loc=cache_loc,
        input_pos_host=input_pos_host,
        seq_len_host=seq_len_host,
    )


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
) -> None:
    """Call thop.attention with MLA parameters enabled.

    Args:
        attention_input_type: 1 = context_only (prefill), 2 = generation_only (decode).
            MLA does not support mixed (0).
        block_ids_per_seq: Override for the planner's block_ids_per_seq. If None,
            uses _GlobalTrtllmMLAPlanner.block_ids_per_seq.
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
        None,  # rotary_cos_sin
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
        0,  # position_embedding_type
        0,  # rotary_embedding_dim
        10000.0,  # rotary_embedding_base
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


def _write_latent_cache_to_paged_kv(
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids_per_seq: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    num_prefill: int,
    tokens_per_block: int,
) -> None:
    """Write latent_cache tokens into the paged KV cache during prefill.

    Uses block_ids_per_seq (raw pool page indices) for direct indexing into
    the per-layer kv_cache view, matching the decode path approach.

    Fully vectorized: all indexing happens on GPU to avoid device-to-host syncs.
    """
    device = kv_cache.device
    total_tokens = latent_cache.shape[0]

    seq_lens = seq_len_host[:num_prefill].to(device=device, dtype=torch.int32)
    seq_starts = input_pos_host[:num_prefill].to(device=device, dtype=torch.int32)

    seq_indices = torch.arange(num_prefill, device=device, dtype=torch.int32).repeat_interleave(
        seq_lens
    )

    cu_seqlens = torch.zeros(num_prefill + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.cumsum(0)
    token_ids = torch.arange(total_tokens, device=device, dtype=torch.int32)
    offset_in_seq = token_ids - cu_seqlens[seq_indices.long()]
    abs_positions = seq_starts[seq_indices.long()] + offset_in_seq

    page_in_seq = (abs_positions // tokens_per_block).long()
    slots = (abs_positions % tokens_per_block).long()
    seq_idx_long = seq_indices.long()

    page_ids = block_ids_per_seq[seq_idx_long, page_in_seq]
    kv_cache[page_ids, 0, 0, slots] = latent_cache


def _read_latent_cache_from_paged_kv(
    kv_cache: torch.Tensor,
    block_ids_per_seq: torch.Tensor,
    seq_idx: int,
    num_tokens_to_read: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Read latent cache tokens from paged KV cache for a single sequence.

    Inverse of _write_latent_cache_to_paged_kv — uses block_ids_per_seq
    for direct indexing into the per-layer kv_cache view.

    Returns: [num_tokens_to_read, latent_dim] tensor.
    """
    device = kv_cache.device

    positions = torch.arange(num_tokens_to_read, device=device, dtype=torch.int64)
    page_indices = positions // tokens_per_block
    slots = positions % tokens_per_block

    page_ids = block_ids_per_seq[seq_idx, page_indices]
    return kv_cache[page_ids, 0, 0, slots]


def _handle_prefill(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    latent_cache: torch.Tensor,
    num_tokens: int,
    num_prefill: int,
    num_heads: int,
    num_kv_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    gen_head_size: int,
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
) -> torch.Tensor:
    """Handle prefill: expand compressed KV + thop.attention with separate Q, K, V.

    On SM < 100, the thop kernel requires is_fused_qkv=False for context_only MLA,
    meaning Q, K, V must be provided as separate expanded tensors (no weight absorption).

    With use_paged_context_fmha=False (required for MLA on SM90), the kernel treats all
    provided Q/K/V tokens as a single sequence when there is only one context request.
    For multi-batch prefill the caller must either loop per-sequence or provide
    cu_q_seqlens / cu_kv_seqlens.

    The output is directly in v_head_dim space — no W_v projection is needed.
    """
    dtype = q_nope_flat.dtype
    device = q_nope_flat.device

    # Build Q = [q_nope | q_pe] -> [num_tokens, num_heads * qk_head_dim]
    q = torch.empty(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    q[:, :, :qk_nope_head_dim] = q_nope_flat
    q[:, :, qk_nope_head_dim:] = q_pe_flat
    q = q.view(num_tokens, num_heads * qk_head_dim)

    # Expand compressed_kv via kv_b_proj_weight to get per-head k_nope and v.
    # The weight rows are grouped per-head: each head's (qk_nope_head_dim + v_head_dim)
    # rows are contiguous.  We must reshape to per-head FIRST, then split, to match
    # the reference torch_mla which does view(B, S, N, kv_head_dim) then split(dim=-1).
    compressed_kv = latent_cache[:, :kv_lora_rank]
    k_pe = latent_cache[:, kv_lora_rank:]

    kv = torch.nn.functional.linear(compressed_kv, kv_b_proj_weight)
    kv = kv.view(num_tokens, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope = kv[:, :, :qk_nope_head_dim]
    v = kv[:, :, qk_nope_head_dim:].contiguous().view(num_tokens, num_heads * v_head_dim)

    k = torch.empty(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    k[:, :, :qk_nope_head_dim] = k_nope
    k[:, :, qk_nope_head_dim:] = k_pe.view(num_tokens, 1, qk_rope_head_dim)
    k = k.view(num_tokens, num_heads * qk_head_dim)

    output = torch.empty(num_tokens, num_heads * v_head_dim, dtype=dtype, device=device)

    planner = _GlobalTrtllmMLAPlanner

    if num_prefill <= 1:
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
            num_heads,
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
        )
    else:
        # Multiple prefill sequences: process each independently.
        # With use_paged_context_fmha=False the kernel treats all tokens as one
        # sequence, so we must call per-sequence to avoid cross-sequence attention.
        token_offset = 0
        for seq_idx in range(num_prefill):
            seq_len = int(host_context_lengths[seq_idx])
            seq_kv_len = int(sequence_length[seq_idx])
            t_end = token_offset + seq_len

            seq_host_total_kv_lens = torch.zeros(
                2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
            )
            seq_host_total_kv_lens[0] = seq_kv_len

            seq_block_ids = planner.block_ids_per_seq[seq_idx : seq_idx + 1].contiguous()
            seq_block_offsets = kv_cache_block_offsets[:, seq_idx : seq_idx + 1].contiguous()

            seq_output = torch.empty(seq_len, num_heads * v_head_dim, dtype=dtype, device=device)

            _call_thop_attention_mla(
                q[token_offset:t_end].contiguous(),
                k[token_offset:t_end].contiguous(),
                v[token_offset:t_end].contiguous(),
                seq_output,
                latent_cache[token_offset:t_end].contiguous(),
                None,
                False,
                1,
                num_heads,
                num_heads,
                qk_head_dim,
                tokens_per_block,
                1,
                seq_kv_len,
                1.0,
                quant_mode,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                sequence_length[seq_idx : seq_idx + 1].contiguous(),
                context_lengths[seq_idx : seq_idx + 1].contiguous(),
                host_past_kv_lengths[seq_idx : seq_idx + 1].contiguous(),
                host_context_lengths[seq_idx : seq_idx + 1].contiguous(),
                host_request_types[seq_idx : seq_idx + 1].contiguous(),
                seq_host_total_kv_lens,
                seq_block_offsets,
                host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping,
                block_ids_per_seq=seq_block_ids,
            )
            output[token_offset:t_end] = seq_output
            token_offset = t_end

    return output


def _batched_fresh_prefill_sdpa(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    compressed_kv_flat: torch.Tensor,
    kpe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    scale: float,
    num_prefill: int,
    seq_lens_list: list,
) -> torch.Tensor:
    """Batched SDPA prefill for fresh sequences (all input_pos == 0).

    Replaces the per-sequence loop with a single batched SDPA call.
    For uniform-length sequences this is a simple reshape; for variable
    lengths, sequences are padded to max_len.  The causal mask naturally
    prevents valid queries from attending to any padded key positions
    (ki <= qi < seq_len), so no custom mask is needed.
    """
    num_tokens = q_nope_flat.shape[0]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    kv_expanded = torch.matmul(compressed_kv_flat, kv_b_proj_weight.t())
    kv_expanded = kv_expanded.view(num_tokens, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope = kv_expanded[:, :, :qk_nope_head_dim]
    v_states = kv_expanded[:, :, qk_nope_head_dim:]

    kpe_expanded = kpe_flat.unsqueeze(1).expand(-1, num_heads, -1)
    k_full = torch.cat([k_nope, kpe_expanded], dim=-1)
    q_full = torch.cat([q_nope_flat, q_pe_flat], dim=-1)

    B = num_prefill
    uniform_len = len(set(seq_lens_list)) == 1

    if uniform_len:
        S = int(seq_lens_list[0])
        q_s = q_full.view(B, S, num_heads, qk_head_dim).transpose(1, 2)
        k_s = k_full.view(B, S, num_heads, qk_head_dim).transpose(1, 2).contiguous()
        v_s = v_states.view(B, S, num_heads, v_head_dim).transpose(1, 2).contiguous()

        out = torch.nn.functional.scaled_dot_product_attention(
            q_s,
            k_s,
            v_s,
            is_causal=True,
            scale=scale,
        )
        return out.transpose(1, 2).reshape(num_tokens, num_heads * v_head_dim).contiguous()

    max_len = max(int(s) for s in seq_lens_list)
    q_padded = q_full.new_zeros(B, max_len, num_heads, qk_head_dim)
    k_padded = k_full.new_zeros(B, max_len, num_heads, qk_head_dim)
    v_padded = v_states.new_zeros(B, max_len, num_heads, v_head_dim)

    offset = 0
    for i, slen in enumerate(seq_lens_list):
        slen = int(slen)
        q_padded[i, :slen] = q_full[offset : offset + slen]
        k_padded[i, :slen] = k_full[offset : offset + slen]
        v_padded[i, :slen] = v_states[offset : offset + slen]
        offset += slen

    q_s = q_padded.transpose(1, 2)
    k_s = k_padded.transpose(1, 2).contiguous()
    v_s = v_padded.transpose(1, 2).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        q_s,
        k_s,
        v_s,
        is_causal=True,
        scale=scale,
    )
    out = out.transpose(1, 2)  # [B, S, N, V]
    results = []
    for i, slen in enumerate(seq_lens_list):
        slen = int(slen)
        results.append(out[i, :slen].reshape(slen, num_heads * v_head_dim))
    return torch.cat(results, dim=0).contiguous()


def _handle_chunked_prefill(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    compressed_kv_flat: torch.Tensor,
    kpe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    scale: float,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids_per_seq: torch.Tensor,
    tokens_per_block: int,
    num_prefill: int,
) -> torch.Tensor:
    """Handle chunked prefill via SDPA with cached latent tokens.

    When all sequences are fresh (input_pos == 0), dispatches to a single
    batched SDPA call.  When input_pos > 0, reads previously cached latent
    tokens from the paged KV cache, combines with the current chunk, and
    runs SDPA with a bottom-right-aligned causal mask per sequence.
    """
    seq_lens_list = seq_len_host[:num_prefill].tolist()
    input_pos_list = input_pos_host[:num_prefill].tolist()

    all_fresh = all(ip == 0 for ip in input_pos_list)
    if all_fresh and num_prefill > 1:
        return _batched_fresh_prefill_sdpa(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            kv_b_proj_weight,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            scale,
            num_prefill,
            seq_lens_list,
        )

    outputs = []
    q_offset = 0

    for seq_idx, slen in enumerate(seq_lens_list):
        slen = int(slen)
        input_pos = int(input_pos_list[seq_idx])
        q_nope_seq = q_nope_flat[q_offset : q_offset + slen]
        q_pe_seq = q_pe_flat[q_offset : q_offset + slen]
        ckv_seq = compressed_kv_flat[q_offset : q_offset + slen]
        kpe_seq = kpe_flat[q_offset : q_offset + slen]

        if input_pos > 0:
            cached_latent = _read_latent_cache_from_paged_kv(
                kv_cache,
                block_ids_per_seq,
                seq_idx,
                input_pos,
                tokens_per_block,
            )
            ckv_seq = torch.cat([cached_latent[:, :kv_lora_rank], ckv_seq], dim=0)
            kpe_seq = torch.cat([cached_latent[:, kv_lora_rank:], kpe_seq], dim=0)

        kv_expanded = torch.matmul(ckv_seq, kv_b_proj_weight.t())
        kv_expanded = kv_expanded.view(-1, num_heads, qk_nope_head_dim + v_head_dim)
        k_nope = kv_expanded[:, :, :qk_nope_head_dim]
        v_states = kv_expanded[:, :, qk_nope_head_dim:]

        kpe_expanded = kpe_seq.unsqueeze(1).expand(-1, num_heads, -1)
        k_full = torch.cat([k_nope, kpe_expanded], dim=-1)
        q_full = torch.cat([q_nope_seq, q_pe_seq], dim=-1)

        total_kv = k_full.shape[0]
        q_s = q_full.unsqueeze(0).transpose(1, 2)
        k_s = k_full.unsqueeze(0).transpose(1, 2).contiguous()
        v_s = v_states.unsqueeze(0).transpose(1, 2).contiguous()

        if input_pos == 0:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                is_causal=True,
                scale=scale,
            )
        else:
            attn_mask = torch.full(
                (slen, total_kv),
                float("-inf"),
                device=q_nope_flat.device,
                dtype=q_nope_flat.dtype,
            )
            for qi in range(slen):
                kv_end = input_pos + qi + 1
                attn_mask[qi, :kv_end] = 0.0
            out = torch.nn.functional.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                scale=scale,
            )

        outputs.append(out.transpose(1, 2).reshape(slen, num_heads * v_head_dim))
        q_offset += slen

    return torch.cat(outputs, dim=0).contiguous()


def _write_decode_latent_to_cache(
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    num_decode: int,
) -> None:
    """Write decode tokens' latent_cache to the paged KV cache.

    Uses an ``as_strided`` 2D view of the (possibly non-contiguous) paged
    cache with a pre-computed flat index and ``index_copy_``.  This replaces
    the multi-dim ``index_put`` (which internally spawns an extra bf16 copy
    kernel per call) with a single efficient scatter.

    The flat index is computed once per step — either in ``plan()`` (after
    the stride ratio is learned) or lazily on the first layer call.
    """
    planner = _GlobalTrtllmMLAPlanner

    # Learn stride ratio from the first kv_cache encountered.
    if planner._cache_rows_per_block == 0:
        planner._cache_rows_per_block = kv_cache.stride(0) // kv_cache.stride(3)

    # Compute flat write index once per step; subsequent layers reuse it.
    if planner._flat_write_idx_dirty:
        page_ids = planner.decode_cache_page_ids[:num_decode]
        slot_idx = planner.decode_slot_idx[:num_decode]
        planner.decode_flat_write_idx[:num_decode] = (
            page_ids.long() * planner._cache_rows_per_block + slot_idx
        )
        planner._flat_write_idx_dirty = False

    flat_idx = planner.decode_flat_write_idx[:num_decode]

    # as_strided maps (block, slot) pairs to correct storage offsets even
    # when the pool has kv_factor > 1, making the 5D tensor non-contiguous.
    # Use a tight row count: the last block only needs tokens_per_block rows
    # (not the full rows_per_block which includes kv_factor/kv_heads gaps),
    # avoiding overflow past the pool's storage when there is a storage_offset.
    D = kv_cache.shape[-1]
    rpb = planner._cache_rows_per_block
    tpb = kv_cache.shape[3]
    num_rows = (kv_cache.shape[0] - 1) * rpb + tpb
    kv_cache_2d = torch.as_strided(
        kv_cache,
        size=(num_rows, D),
        stride=(kv_cache.stride(3), kv_cache.stride(4)),
    )
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
    table = rotary_cos_sin.view(-1, qk_rope_head_dim, 2)
    half = qk_rope_head_dim // 2
    cos_half = table[positions.long(), :half, 0].to(q_pe.dtype)  # [T, D/2]
    sin_half = table[positions.long(), :half, 1].to(q_pe.dtype)  # [T, D/2]

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


def _handle_fused_rope_decode(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
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
) -> torch.Tensor:
    """Decode with fused mla_rope_generation kernel.

    Replaces _handle_decode's 3 separate kernels (q_pe copy, cache write,
    scheduler fill) with a single ``mla_rope_generation`` call, then runs
    the same Q absorption + thop.attention + V projection pipeline.

    Note: thop.attention with update_kv_cache=True also writes the cache
    (required by C++ assertion), so the cache is written twice.  This benign
    double-write is cheaper than the alternative of separate Python kernels.
    """
    planner = _GlobalTrtllmMLAPlanner
    gen_head_size = kv_lora_rank + qk_rope_head_dim

    w_kn, w_v_t = planner.get_weight_matrices(
        kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )

    # Build fused_q — Q absorption via bmm_out into the left slice.
    fused_q_flat = planner.fused_q_flat[:num_tokens]
    fused_q_view = fused_q_flat.view(num_tokens, num_heads, gen_head_size)

    q_nope_t = q_nope_flat.transpose(0, 1)
    q_absorbed_target = fused_q_view[:, :, :kv_lora_rank].transpose(0, 1)
    torch.ops.trtllm.bmm_out(q_nope_t, w_kn, q_absorbed_target)

    # mla_rope_generation fills the right slice of fused_q with RoPE'd q_pe,
    # writes latent_cache to the paged KV cache with RoPE on kpe, and fills
    # cu_q_seqlens / cu_kv_seqlens / fmha_scheduler_counter.
    cu_q = planner.cu_q_decode[: num_tokens + 1]
    cu_kv = planner.cu_kv_decode[: num_tokens + 1]

    torch.ops.trtllm.mla_rope_generation(
        fused_q_view,
        q_pe_flat,
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
        num_heads,
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

    output_latent = planner.output_latent[:num_tokens]

    _call_thop_attention_mla(
        fused_q_flat,
        None,
        None,
        output_latent,
        latent_cache,
        q_pe_flat,
        True,
        2,  # attention_input_type = generation_only
        num_heads,
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

    # V projection
    output_reshaped = output_latent.view(num_tokens, num_heads, kv_lora_rank)
    v_proj_out = planner.v_proj_output[:num_tokens]
    torch.ops.trtllm.bmm_out(
        output_reshaped.transpose(0, 1),
        w_v_t,
        v_proj_out.transpose(0, 1),
    )
    return v_proj_out.reshape(num_tokens, num_heads * v_head_dim)


def _handle_decode(
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
) -> torch.Tensor:
    """Handle decode: weight absorption + latent-space attention + output projection.

    Follows the PyTorch backend pattern: ``bmm_out`` writes the Q absorption
    result directly into a pre-allocated ``fused_q`` slice (no intermediate
    tensor, no ``torch.cat``), and the V projection also uses ``bmm_out`` into
    a pre-allocated buffer.
    """
    planner = _GlobalTrtllmMLAPlanner
    gen_head_size = kv_lora_rank + qk_rope_head_dim

    w_kn, w_v_t = planner.get_weight_matrices(
        kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )

    # Build fused_q = [q_absorbed | q_pe] into pre-allocated buffer.
    # bmm_out writes q_absorbed directly into the left slice of fused_q,
    # then q_pe is copied into the right slice — no torch.cat needed.
    fused_q_flat = planner.fused_q_flat[:num_tokens]
    fused_q_view = fused_q_flat.view(num_tokens, num_heads, gen_head_size)

    q_nope_t = q_nope_flat.transpose(0, 1)
    q_absorbed_target = fused_q_view[:, :, :kv_lora_rank].transpose(0, 1)
    torch.ops.trtllm.bmm_out(q_nope_t, w_kn, q_absorbed_target)

    fused_q_view[:, :, kv_lora_rank:] = q_pe_flat

    output_latent = planner.output_latent[:num_tokens]

    cu_q = planner.cu_q_decode[: num_tokens + 1]
    cu_kv = planner.cu_kv_decode[: num_tokens + 1]

    planner.fmha_scheduler_counter_decode.fill_(0)

    _call_thop_attention_mla(
        fused_q_flat,
        None,
        None,
        output_latent,
        latent_cache,
        q_pe_flat,
        True,
        2,  # attention_input_type = generation_only
        num_heads,
        num_kv_heads,
        head_size,
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

    # Project from latent space back to v_head_dim using bmm_out into
    # a pre-allocated buffer (matching the PT backend pattern).
    output_reshaped = output_latent.view(num_tokens, num_heads, kv_lora_rank)
    v_proj_out = planner.v_proj_output[:num_tokens]
    torch.ops.trtllm.bmm_out(
        output_reshaped.transpose(0, 1),
        w_v_t,
        v_proj_out.transpose(0, 1),
    )
    return v_proj_out.reshape(num_tokens, num_heads * v_head_dim)


# =============================================================================
# Cached MLA attention op
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_mla_with_cache", mutates_args=("kv_cache",))
def trtllm_mla_with_cache(
    # 5 MLA tensor args (matching torch_mla source op)
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    # Standard metadata (SequenceInfo fields)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # Cache
    kv_cache: torch.Tensor,
    # Constants
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """TRT-LLM MLA attention with paged latent cache for Auto-Deploy.

    Prefill: expands compressed KV to separate Q, K, V and calls thop.attention with
    is_fused_qkv=False and attention_input_type=context_only.

    Decode: uses weight absorption (q_absorbed = q_nope @ W_kn) and calls thop.attention
    with is_fused_qkv=True and attention_input_type=generation_only, then projects output
    from latent space back to v_head_dim via W_v.
    """
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

    # Flatten from [B, S, ...] to [num_tokens, ...], only copy if non-contiguous
    q_nope_c = q_nope if q_nope.is_contiguous() else q_nope.contiguous()
    q_pe_c = q_pe if q_pe.is_contiguous() else q_pe.contiguous()
    compressed_kv_c = compressed_kv if compressed_kv.is_contiguous() else compressed_kv.contiguous()
    kpe_c = kpe if kpe.is_contiguous() else kpe.contiguous()

    q_nope_flat = q_nope_c.view(num_tokens, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe_c.view(num_tokens, num_heads, qk_rope_head_dim)
    compressed_kv_flat = compressed_kv_c.view(num_tokens, kv_lora_rank)
    kpe_flat = kpe_c.view(num_tokens, qk_rope_head_dim)

    # Ensure decode-path scratch buffers are allocated
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

    # Build latent_cache: [num_tokens, kv_lora_rank + qk_rope_head_dim]
    # Use torch.cat with out= for decode-only to write into pre-allocated buffer
    # with a single kernel instead of two slice-assign copies.
    if num_prefill == 0:
        latent_buf = planner.latent_cache_buf[:num_tokens]
        torch.cat([compressed_kv_flat, kpe_flat], dim=-1, out=latent_buf)
        latent_cache = latent_buf
    else:
        latent_cache = torch.cat([compressed_kv_flat, kpe_flat], dim=-1).contiguous()

    # Metadata from planner
    sequence_length = seq_len_with_cache[:num_seq]
    context_lengths = seq_len[:num_seq]
    host_past_kv_lengths = planner.host_past_kv_lengths[:num_seq]
    host_context_lengths = planner.host_context_lengths[:num_seq]
    host_request_types = planner.host_request_types[:num_seq]
    host_total_kv_lens = planner.host_total_kv_lens
    kv_cache_block_offsets = planner.block_offsets
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

    # Always use SDPA for MLA prefill.  The AD thop.attention context call
    # has a parameter mismatch vs the PyTorch backend's TrtllmAttention
    # wrapper that causes an illegal memory access on both SM90 and SM100.
    # Root-cause TBD — known differences already fixed (latent_cache,
    # num_kv_heads=num_heads) but the kernel still crashes.  The thop MLA
    # *decode* kernel (weight-absorption path) works correctly.
    # SDPA avoids the issue: attention is computed via PyTorch and the
    # latent cache is written separately by _write_latent_cache_to_paged_kv.

    def _do_prefill(q_n, q_p, lc, n_tok, n_pf):
        """Dispatch prefill via SDPA (handles both fresh and chunked prefill)."""
        return _handle_chunked_prefill(
            q_n,
            q_p,
            compressed_kv_flat[:n_tok],
            kpe_flat[:n_tok],
            kv_b_proj_weight,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
            seq_len_host,
            input_pos_host,
            kv_cache,
            planner.block_ids_per_seq,
            tokens_per_block,
            n_pf,
        )

    if num_prefill > 0 and num_decode > 0:
        _write_latent_cache_to_paged_kv(
            latent_cache[:num_prefill_tokens],
            kv_cache,
            planner.block_ids_per_seq,
            seq_len_host,
            input_pos_host,
            num_prefill,
            tokens_per_block,
        )
        y = torch.empty(
            num_tokens,
            num_heads * v_head_dim,
            dtype=q_nope_flat.dtype,
            device=q_nope_flat.device,
        )
        y[:num_prefill_tokens] = _do_prefill(
            q_nope_flat[:num_prefill_tokens],
            q_pe_flat[:num_prefill_tokens],
            latent_cache[:num_prefill_tokens],
            num_prefill_tokens,
            num_prefill,
        )
        _write_decode_latent_to_cache(
            latent_cache[num_prefill_tokens:num_tokens],
            kv_cache,
            num_decode,
        )
        y[num_prefill_tokens:num_tokens] = _handle_decode(
            q_nope_flat[num_prefill_tokens:num_tokens],
            q_pe_flat[num_prefill_tokens:num_tokens],
            kv_b_proj_weight,
            latent_cache[num_prefill_tokens:num_tokens],
            kv_cache,
            num_decode,
            num_prefill,
            *_make_shared_metadata(),
        )
    elif num_prefill > 0:
        _write_latent_cache_to_paged_kv(
            latent_cache,
            kv_cache,
            planner.block_ids_per_seq,
            seq_len_host,
            input_pos_host,
            num_prefill,
            tokens_per_block,
        )
        y = _do_prefill(
            q_nope_flat,
            q_pe_flat,
            latent_cache,
            num_prefill_tokens,
            num_prefill,
        )
    else:
        _write_decode_latent_to_cache(
            latent_cache,
            kv_cache,
            num_tokens,
        )
        y = _handle_decode(
            q_nope_flat,
            q_pe_flat,
            kv_b_proj_weight,
            latent_cache,
            kv_cache,
            num_tokens,
            0,
            *_make_shared_metadata(),
        )

    return y.view(b, s, num_heads, v_head_dim)


@trtllm_mla_with_cache.register_fake
def trtllm_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
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
# Fused RoPE + MLA cached attention op
# =============================================================================


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
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """TRT-LLM MLA attention with fused RoPE, cache write, and paged latent cache.

    Same as ``trtllm_mla_with_cache`` but receives **pre-RoPE** q_pe/kpe and a
    ``rotary_cos_sin`` table.  For decode, calls ``mla_rope_generation`` which
    fuses cache write + RoPE + q_pe copy + scheduler fill into one kernel.
    For prefill, applies RoPE in Python before SDPA.
    """
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
    tokens_per_block = kv_cache.shape[3]

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

    q_nope_c = q_nope if q_nope.is_contiguous() else q_nope.contiguous()
    q_pe_c = q_pe if q_pe.is_contiguous() else q_pe.contiguous()
    compressed_kv_c = compressed_kv if compressed_kv.is_contiguous() else compressed_kv.contiguous()
    kpe_c = kpe if kpe.is_contiguous() else kpe.contiguous()

    q_nope_flat = q_nope_c.view(num_tokens, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe_c.view(num_tokens, num_heads, qk_rope_head_dim)
    compressed_kv_flat = compressed_kv_c.view(num_tokens, kv_lora_rank)
    kpe_flat = kpe_c.view(num_tokens, qk_rope_head_dim)

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

    # Build latent_cache: [num_tokens, kv_lora_rank + qk_rope_head_dim]
    # For the fused op, kpe is pre-RoPE — mla_rope_generation applies RoPE
    # to the kpe portion of latent_cache during the cache write.
    if num_prefill == 0:
        latent_buf = planner.latent_cache_buf[:num_tokens]
        torch.cat([compressed_kv_flat, kpe_flat], dim=-1, out=latent_buf)
        latent_cache = latent_buf
    else:
        latent_cache = torch.cat([compressed_kv_flat, kpe_flat], dim=-1).contiguous()

    sequence_length = seq_len_with_cache[:num_seq]
    context_lengths = seq_len[:num_seq]
    host_past_kv_lengths = planner.host_past_kv_lengths[:num_seq]
    host_context_lengths = planner.host_context_lengths[:num_seq]
    host_request_types = planner.host_request_types[:num_seq]
    host_total_kv_lens = planner.host_total_kv_lens
    kv_cache_block_offsets = planner.block_offsets
    host_kv_cache_pool_mapping = planner.host_pool_mapping

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

    def _apply_prefill_rope(q_p, kpe_pre, n_tok, n_pf):
        """Compute RoPE for prefill tokens; returns (q_pe_rot, kpe_rot)."""
        positions = torch.arange(n_tok, device=q_p.device, dtype=torch.int32)
        offset = 0
        for i in range(n_pf):
            slen = int(seq_len_host[i])
            ipos = int(input_pos_host[i])
            positions[offset : offset + slen] = torch.arange(
                ipos, ipos + slen, device=q_p.device, dtype=torch.int32
            )
            offset += slen
        return _apply_rope_from_table(q_p, kpe_pre, rotary_cos_sin, positions, qk_rope_head_dim)

    def _do_prefill(q_n, q_p_rot, ckv_pre, kpe_rot, n_pf):
        """Prefill with already-rotated inputs: dispatch to SDPA."""
        return _handle_chunked_prefill(
            q_n,
            q_p_rot,
            ckv_pre,
            kpe_rot,
            kv_b_proj_weight,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
            seq_len_host,
            input_pos_host,
            kv_cache,
            planner.block_ids_per_seq,
            tokens_per_block,
            n_pf,
        )

    def _do_decode(q_n, q_p, lc, n_tok, n_pf):
        """Decode with fused mla_rope_generation."""
        return _handle_fused_rope_decode(
            q_n,
            q_p,
            kv_b_proj_weight,
            lc,
            kv_cache,
            rotary_cos_sin,
            n_tok,
            n_pf,
            *_make_shared_metadata(),
        )

    if num_prefill > 0 and num_decode > 0:
        # Apply RoPE for prefill tokens to get rotated kpe for cache write.
        q_pe_rot_pf, kpe_rot_pf = _apply_prefill_rope(
            q_pe_flat[:num_prefill_tokens],
            kpe_flat[:num_prefill_tokens],
            num_prefill_tokens,
            num_prefill,
        )
        # Write [compressed_kv | kpe_rotated] to the paged cache so that
        # decode steps (which read raw cached kpe) see consistent RoPE.
        latent_for_cache = torch.cat([compressed_kv_flat[:num_prefill_tokens], kpe_rot_pf], dim=-1)
        _write_latent_cache_to_paged_kv(
            latent_for_cache,
            kv_cache,
            planner.block_ids_per_seq,
            seq_len_host,
            input_pos_host,
            num_prefill,
            tokens_per_block,
        )
        y = torch.empty(
            num_tokens,
            num_heads * v_head_dim,
            dtype=q_nope_flat.dtype,
            device=q_nope_flat.device,
        )
        y[:num_prefill_tokens] = _do_prefill(
            q_nope_flat[:num_prefill_tokens],
            q_pe_rot_pf,
            compressed_kv_flat[:num_prefill_tokens],
            kpe_rot_pf,
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
        # Apply RoPE for prefill tokens.
        q_pe_rot_pf, kpe_rot_pf = _apply_prefill_rope(
            q_pe_flat, kpe_flat, num_prefill_tokens, num_prefill
        )
        latent_for_cache = torch.cat([compressed_kv_flat, kpe_rot_pf], dim=-1)
        _write_latent_cache_to_paged_kv(
            latent_for_cache,
            kv_cache,
            planner.block_ids_per_seq,
            seq_len_host,
            input_pos_host,
            num_prefill,
            tokens_per_block,
        )
        y = _do_prefill(
            q_nope_flat,
            q_pe_rot_pf,
            compressed_kv_flat,
            kpe_rot_pf,
            num_prefill,
        )
    else:
        y = _do_decode(
            q_nope_flat,
            q_pe_flat,
            latent_cache,
            num_tokens,
            0,
        )

    return y.view(b, s, num_heads, v_head_dim)


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
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
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
            "seq_len_host",
            "input_pos_host",
            "seq_len_with_cache",
            "max_seq_info_host",
        ]

    @classmethod
    def get_cache_initializers(
        cls,
        source_attn_node: Node,
        cache_config: KvCacheConfig,
    ) -> ResourceHandlerDict:
        """Return KV cache handler for MLA latent cache.

        MLA stores [compressed_kv | k_pe] per token with num_kv_heads=1.
        """
        compressed_kv_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        kpe_fake: FakeTensor = source_attn_node.args[3].meta["val"]

        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]

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
        scale = source_attn_node.kwargs.get("scale", None)
        return [scale, kv_lora_rank]
