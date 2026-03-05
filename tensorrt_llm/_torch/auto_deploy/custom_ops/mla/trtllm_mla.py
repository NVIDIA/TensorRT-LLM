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

MLA stores compressed latent representations instead of separate K and V:
- Prefill: expands compressed_kv via kv_b_proj_weight, builds full QKV,
  computes attention via SDPA, and writes latent_cache to the paged KV cache.
- Chunked prefill: when input_pos > 0, reads previously cached latent tokens
  from the paged KV cache, combines with the current chunk's tokens, and runs
  SDPA with an explicit bottom-right-aligned causal mask.
- Decode: uses weight absorption (q_nope @ W_kn^T) to avoid expanding cached KV,
  calls thop.attention in latent space, then projects output back via W_v.

Cache layout:
    kv_cache: paged pool storing [compressed_kv | k_pe] per token
    - num_kv_heads=1, head_dim=kv_lora_rank+qk_rope_head_dim, kv_factor=2 (HND)
"""

import math
from typing import List, Optional

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from tensorrt_llm._utils import get_sm_version
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
# Module-level planner
# =============================================================================


class _TrtllmMLAPlanner:
    """Minimal planner for TRT-LLM MLA attention backend.

    Mirrors ``_TrtllmPlanner`` from the standard trtllm backend. Manages persistent
    buffers for thop.attention metadata and per-layer pool pointers for CUDA graph
    compatibility.
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

    def reset(self, device: torch.device, max_batch: int, max_blocks_per_seq: int) -> None:
        """One-time allocation of ALL persistent buffers."""
        if self.workspace is not None:
            return

        self.workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.host_pool_mapping = torch.zeros(1, 2, dtype=torch.int32, device="cpu", pin_memory=True)
        self.host_total_kv_lens = torch.zeros(2, dtype=torch.int64, device="cpu", pin_memory=True)
        self.host_request_types = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.block_offsets = torch.zeros(
            1, max_batch, 2, max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self.host_past_kv_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.host_context_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.block_ids_per_seq = torch.zeros(
            max_batch, max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self.cu_q_seqlens: Optional[torch.Tensor] = None
        self.cu_kv_seqlens: Optional[torch.Tensor] = None
        self.fmha_scheduler_counter = torch.zeros(1, dtype=torch.int32, device=device)

    def plan(
        self,
        num_prefill: int,
        num_decode: int,
        max_context_length: int,
        block_offset_multiplier: int,
        seq_len_with_cache_host: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        page_seq_indices: torch.Tensor,
        page_in_seq: torch.Tensor,
        input_pos_host: torch.Tensor,
        seq_len_host: torch.Tensor,
    ) -> None:
        """Per-forward host metadata: fills host_request_types, block_offsets, etc."""
        num_seq = num_prefill + num_decode

        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

        block_offsets = self.block_offsets
        total_pages = int(cu_num_pages_host[num_seq])
        base_offsets = cache_loc[:total_pages] * block_offset_multiplier
        seq_idx = page_seq_indices[:total_pages]
        pg_idx = page_in_seq[:total_pages]
        block_offsets[0, seq_idx, 0, pg_idx] = base_offsets
        block_offsets[0, seq_idx, 1, pg_idx] = base_offsets + 1

        self.block_ids_per_seq.fill_(0)
        self.block_ids_per_seq[seq_idx, pg_idx] = cache_loc[:total_pages]

        is_capturing = torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up()
        if is_capturing:
            self.host_total_kv_lens[0] = max_context_length * num_prefill
            self.host_total_kv_lens[1] = max_context_length * num_decode
            self.host_past_kv_lengths[:num_seq].fill_(max_context_length)
            self.host_context_lengths[:num_seq].fill_(max_context_length)
        else:
            self.host_total_kv_lens[0] = seq_len_with_cache_host[:num_prefill].sum()
            self.host_total_kv_lens[1] = seq_len_with_cache_host[num_prefill:num_seq].sum()
            self.host_past_kv_lengths[:num_seq] = input_pos_host[:num_seq]
            self.host_context_lengths[:num_seq] = seq_len_host[:num_seq]

    def get_pool_pointers_for_layer(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Return a per-layer ``host_pool_pointers`` tensor for this kv_cache view."""
        ptr = kv_cache.data_ptr()
        t = self._per_layer_pool_ptrs.get(ptr)
        if t is not None:
            return t
        t = torch.zeros(1, 2, dtype=torch.int64, device="cpu", pin_memory=True)
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
    page_seq_indices: torch.Tensor,
    page_in_seq: torch.Tensor,
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
        page_seq_indices=page_seq_indices,
        page_in_seq=page_in_seq,
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
    latent_cache: torch.Tensor,
    q_pe: Optional[torch.Tensor],
    is_fused_qkv: bool,
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
) -> None:
    """Call thop.attention with MLA parameters enabled."""
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    sm_version = get_sm_version()
    if sm_version >= 89:
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    # MLA requires context_only (1) or generation_only (2), never mixed (0)
    attention_input_type = 2 if is_fused_qkv else 1

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
        _GlobalTrtllmMLAPlanner.block_ids_per_seq,  # block_ids_per_seq
        None,  # attention_sinks
        is_fused_qkv,  # is_fused_qkv
        True,  # update_kv_cache
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
    kv_cache_block_offsets: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    num_prefill: int,
    tokens_per_block: int,
) -> None:
    """Write latent_cache tokens into the paged KV cache during prefill.

    The kv_cache has HND layout: [num_pages, kv_factor, num_kv_heads, page_size, head_dim].
    Block offsets are physical indices into the interleaved pool (all layers share one
    pool), so they include a stride of num_layers * kv_factor per logical page. We
    recover the page index and kv slot by dividing by the stride ratio between dim 0
    and dim 1 of the (potentially non-contiguous) per-layer view.
    For MLA both K and V slots store the same latent data.
    """
    blocks_per_page = kv_cache.stride(0) // kv_cache.stride(1)
    num_kv_h = kv_cache.shape[2]
    token_offset = 0
    for seq_idx in range(num_prefill):
        seq_start = int(input_pos_host[seq_idx])
        seq_len = int(seq_len_host[seq_idx])
        for t in range(seq_len):
            abs_pos = seq_start + t
            page_idx = abs_pos // tokens_per_block
            slot_in_page = abs_pos % tokens_per_block
            k_block = int(kv_cache_block_offsets[0, seq_idx, 0, page_idx])
            p, r = k_block // blocks_per_page, k_block % blocks_per_page
            kv_cache[p, r // num_kv_h, r % num_kv_h, slot_in_page] = latent_cache[token_offset]
            v_block = int(kv_cache_block_offsets[0, seq_idx, 1, page_idx])
            p, r = v_block // blocks_per_page, v_block % blocks_per_page
            kv_cache[p, r // num_kv_h, r % num_kv_h, slot_in_page] = latent_cache[token_offset]
            token_offset += 1


def _read_latent_cache_from_paged_kv(
    kv_cache: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    seq_idx: int,
    num_tokens_to_read: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Read latent cache tokens from paged KV cache for a single sequence.

    Inverse of _write_latent_cache_to_paged_kv — reads from the K slot of
    the HND paged cache.

    Returns: [num_tokens_to_read, latent_dim] tensor.
    """
    blocks_per_page = kv_cache.stride(0) // kv_cache.stride(1)
    num_kv_h = kv_cache.shape[2]
    latent_dim = kv_cache.shape[4]

    result = torch.empty(
        num_tokens_to_read, latent_dim, dtype=kv_cache.dtype, device=kv_cache.device
    )
    for t in range(num_tokens_to_read):
        page_idx = t // tokens_per_block
        slot_in_page = t % tokens_per_block
        k_block = int(kv_cache_block_offsets[0, seq_idx, 0, page_idx])
        p, r = k_block // blocks_per_page, k_block % blocks_per_page
        result[t] = kv_cache[p, r // num_kv_h, r % num_kv_h, slot_in_page]
    return result


def _handle_prefill(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
    compressed_kv_flat: torch.Tensor,
    kpe_flat: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    scale: float,
    seq_lens: torch.Tensor,
    input_pos_host: torch.Tensor,
    kv_cache: Optional[torch.Tensor] = None,
    kv_cache_block_offsets: Optional[torch.Tensor] = None,
    tokens_per_block: int = 0,
    kv_lora_rank: int = 0,
    num_prefill: int = 0,
) -> torch.Tensor:
    """Handle prefill: expand compressed_kv, compute attention via SDPA.

    For regular prefill (input_pos == 0), Q and K/V are the same tokens with
    a standard causal mask.

    For chunked prefill (input_pos > 0), previously cached latent tokens are
    read from the paged KV cache, combined with the current chunk, and SDPA
    uses a bottom-right-aligned causal mask (is_causal=True with q_len < kv_len).
    """

    outputs = []
    q_offset = 0
    for seq_idx, slen in enumerate(seq_lens.tolist()):
        slen = int(slen)
        input_pos = int(input_pos_host[seq_idx])
        q_nope_seq = q_nope_flat[q_offset : q_offset + slen]
        q_pe_seq = q_pe_flat[q_offset : q_offset + slen]
        ckv_seq = compressed_kv_flat[q_offset : q_offset + slen]
        kpe_seq = kpe_flat[q_offset : q_offset + slen]

        if input_pos > 0 and kv_cache is not None:
            cached_latent = _read_latent_cache_from_paged_kv(
                kv_cache, kv_cache_block_offsets, seq_idx, input_pos, tokens_per_block
            )
            cached_ckv = cached_latent[:, :kv_lora_rank]
            cached_kpe = cached_latent[:, kv_lora_rank:]
            all_ckv = torch.cat([cached_ckv, ckv_seq], dim=0)
            all_kpe = torch.cat([cached_kpe, kpe_seq], dim=0)
        else:
            all_ckv = ckv_seq
            all_kpe = kpe_seq

        kv_len = all_ckv.shape[0]

        kv_expanded = torch.matmul(all_ckv, kv_b_proj_weight.t())
        kv_expanded = kv_expanded.view(kv_len, num_heads, qk_nope_head_dim + v_head_dim)
        k_nope = kv_expanded[:, :, :qk_nope_head_dim]
        v_states = kv_expanded[:, :, qk_nope_head_dim:]

        kpe_expanded = all_kpe.unsqueeze(1).expand(-1, num_heads, -1)
        k_full = torch.cat([k_nope, kpe_expanded], dim=-1)

        q_full = torch.cat([q_nope_seq, q_pe_seq], dim=-1)

        # [1, num_heads, seq_len, head_dim] for SDPA
        q_s = q_full.unsqueeze(0).transpose(1, 2)
        k_s = k_full.unsqueeze(0).transpose(1, 2).contiguous()
        v_s = v_states.unsqueeze(0).transpose(1, 2).contiguous()

        if input_pos == 0:
            out_seq = torch.nn.functional.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                is_causal=True,
                scale=scale,
            )
        else:
            # Chunked prefill: build an explicit bottom-right-aligned causal
            # mask because is_causal=True may not align correctly when
            # q_len < kv_len in all SDPA backends.
            q_positions = torch.arange(input_pos, input_pos + slen, device=q_nope_flat.device)
            k_positions = torch.arange(kv_len, device=q_nope_flat.device)
            attn_mask = q_positions.unsqueeze(-1) >= k_positions.unsqueeze(0)
            out_seq = torch.nn.functional.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                scale=scale,
            )
        out_seq = out_seq.transpose(1, 2).squeeze(0)

        outputs.append(out_seq.reshape(slen, num_heads * v_head_dim))
        q_offset += slen

    return torch.cat(outputs, dim=0).contiguous()


def _handle_decode(
    q_nope_flat: torch.Tensor,
    q_pe_flat: torch.Tensor,
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
    """Handle decode: weight absorption + latent-space attention + output projection."""
    weight_reshaped = kv_b_proj_weight.view(
        num_heads,
        qk_nope_head_dim + v_head_dim,
        kv_lora_rank,
    )
    w_kn = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]

    # q_absorbed = q_nope @ W_kn -> [num_tokens, N, kv_lora_rank]
    q_absorbed = torch.einsum("bnd,ndk->bnk", q_nope_flat, w_kn).contiguous()

    # fused_q: [num_tokens, N, kv_lora_rank + qk_rope_head_dim]
    fused_q = torch.cat([q_absorbed, q_pe_flat], dim=-1)
    fused_q_flat = fused_q.reshape(
        num_tokens,
        num_heads * (kv_lora_rank + qk_rope_head_dim),
    ).contiguous()

    # thop.attention outputs in latent space for MLA decode
    output_latent = torch.empty(
        num_tokens,
        num_heads * kv_lora_rank,
        dtype=q_nope_flat.dtype,
        device=q_nope_flat.device,
    )

    cu_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=q_nope_flat.device) * num_heads
    cu_kv = torch.zeros(
        num_tokens + 1,
        dtype=torch.int32,
        device=q_nope_flat.device,
    )
    cu_kv[1:] = sequence_length[:num_tokens].cumsum(0)

    _call_thop_attention_mla(
        fused_q_flat,
        None,
        None,
        output_latent,
        latent_cache,
        q_pe_flat,
        True,
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
        fmha_scheduler_counter=_GlobalTrtllmMLAPlanner.fmha_scheduler_counter,
    )

    # Project from latent space back to v_head_dim
    output_latent = output_latent.view(num_tokens, num_heads, kv_lora_rank)
    output = torch.einsum("bnk,nvk->bnv", output_latent, w_v)
    return output.reshape(num_tokens, num_heads * v_head_dim).contiguous()


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

    Prefill: expands compressed_kv to full K, V, computes attention via PyTorch SDPA,
    and writes the latent cache into the paged KV cache.

    Decode: uses weight absorption (q_nope @ W_kn^T) to build fused_q in latent space,
    calls thop.attention which reads from the paged latent cache, then projects output
    back to v_head_dim using W_v.
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
    tokens_per_block = kv_cache.shape[3]  # HND: [blocks, 2, heads, tpb, head_dim]

    gen_head_size = kv_lora_rank + qk_rope_head_dim

    host_kv_cache_pool_pointers = _GlobalTrtllmMLAPlanner.get_pool_pointers_for_layer(kv_cache)

    quant_mode = 0
    if kv_cache.dtype == torch.float8_e4m3fn:
        if _GlobalTrtllmMLAPlanner.kv_scale_orig_quant is None:
            _GlobalTrtllmMLAPlanner.kv_scale_orig_quant = torch.tensor(
                [1.0], dtype=torch.float32, device=q_nope.device
            )
            _GlobalTrtllmMLAPlanner.kv_scale_quant_orig = torch.tensor(
                [1.0], dtype=torch.float32, device=q_nope.device
            )
        quant_mode = int(QuantMode.FP8_KV_CACHE)

    # Flatten from [B, S, ...] to [num_tokens, ...]
    q_nope_flat = q_nope.contiguous().view(num_tokens, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe.contiguous().view(num_tokens, num_heads, qk_rope_head_dim)
    compressed_kv_flat = compressed_kv.contiguous().view(num_tokens, kv_lora_rank)
    kpe_flat = kpe.contiguous().view(num_tokens, qk_rope_head_dim)

    # latent_cache: [num_tokens, kv_lora_rank + qk_rope_head_dim]
    latent_cache = torch.cat([compressed_kv_flat, kpe_flat], dim=-1).contiguous()

    # Metadata from planner
    sequence_length = seq_len_with_cache[:num_seq]
    context_lengths = seq_len[:num_seq]
    host_past_kv_lengths = _GlobalTrtllmMLAPlanner.host_past_kv_lengths[:num_seq]
    host_context_lengths = _GlobalTrtllmMLAPlanner.host_context_lengths[:num_seq]
    host_request_types = _GlobalTrtllmMLAPlanner.host_request_types[:num_seq]
    host_total_kv_lens = _GlobalTrtllmMLAPlanner.host_total_kv_lens
    kv_cache_block_offsets = _GlobalTrtllmMLAPlanner.block_offsets
    host_kv_cache_pool_mapping = _GlobalTrtllmMLAPlanner.host_pool_mapping

    # thop.attention applies q_scaling / sqrt(head_size) internally, where
    # head_size = gen_head_size = kv_lora_rank + qk_rope_head_dim.
    # The correct MLA scale is 1/sqrt(qk_head_dim), so we compensate:
    #   q_scaling / sqrt(gen_head_size) = 1/sqrt(qk_head_dim)
    #   => q_scaling = sqrt(gen_head_size / qk_head_dim)
    thop_q_scaling = math.sqrt(gen_head_size / qk_head_dim)

    def _make_decode_shared():
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

    prefill_seq_lens = seq_len_host[:num_prefill]

    def _run_prefill(q_nope_sl, q_pe_sl, ckv_sl, kpe_sl, lc_sl):
        # Write cache BEFORE attention so that chunked prefill can read
        # previously cached tokens when building the full KV sequence.
        _write_latent_cache_to_paged_kv(
            lc_sl,
            kv_cache,
            kv_cache_block_offsets,
            seq_len_host,
            input_pos_host,
            num_prefill,
            tokens_per_block,
        )
        y_prefill = _handle_prefill(
            q_nope_sl,
            q_pe_sl,
            ckv_sl,
            kpe_sl,
            kv_b_proj_weight,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            scale,
            prefill_seq_lens,
            input_pos_host,
            kv_cache=kv_cache,
            kv_cache_block_offsets=kv_cache_block_offsets,
            tokens_per_block=tokens_per_block,
            kv_lora_rank=kv_lora_rank,
            num_prefill=num_prefill,
        )
        return y_prefill

    if num_prefill > 0 and num_decode > 0:
        y = torch.empty(
            num_tokens,
            num_heads * v_head_dim,
            dtype=q_nope_flat.dtype,
            device=q_nope_flat.device,
        )
        y[:num_prefill_tokens] = _run_prefill(
            q_nope_flat[:num_prefill_tokens],
            q_pe_flat[:num_prefill_tokens],
            compressed_kv_flat[:num_prefill_tokens],
            kpe_flat[:num_prefill_tokens],
            latent_cache[:num_prefill_tokens],
        )
        y[num_prefill_tokens:num_tokens] = _handle_decode(
            q_nope_flat[num_prefill_tokens:num_tokens],
            q_pe_flat[num_prefill_tokens:num_tokens],
            kv_b_proj_weight,
            latent_cache[num_prefill_tokens:num_tokens],
            num_decode,
            *_make_decode_shared(),
        )
    elif num_prefill > 0:
        y = _run_prefill(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            latent_cache,
        )
    else:
        y = _handle_decode(
            q_nope_flat,
            q_pe_flat,
            kv_b_proj_weight,
            latent_cache,
            num_tokens,
            *_make_decode_shared(),
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
                kv_factor=2,
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
