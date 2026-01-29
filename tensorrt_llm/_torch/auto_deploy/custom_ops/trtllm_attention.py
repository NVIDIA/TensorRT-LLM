# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""TRT-LLM attention backend for Auto-Deploy.

This module provides a TRT-LLM attention backend that wraps the optimized
`thop.attention` kernel for use in Auto-Deploy.

Architecture Overview:
---------------------
TRT-LLM's thop.attention expects:
- Combined KV cache or separate K/V with specific layout
- Many metadata tensors (sequence_length, context_lengths, request_types, etc.)
- Pool pointers for multi-pool KV cache management
- Per-layer wrapper state

AD provides:
- Separate K/V caches per layer: [num_pages, page_size, num_kv_heads, head_dim]
- Simpler metadata: batch_info, cu_seqlen, cu_num_pages, cache_loc, etc.

This implementation bridges the gap by:
1. Converting AD's metadata to TRT-LLM's format
2. Using AD's separate K/V caches with TRT-LLM's paged context FMHA
3. Managing per-layer state through global state dictionary

Cache Backend Options:
---------------------
- SimpleCacheBackend (default): Per-layer cache allocation, Python metadata prep
- PTCacheBackend: Unified pool via PT's KVCacheManager, C++ fast path for metadata

Set `use_pt_cache_backend=True` in TrtllmAttentionConfig to use PTCacheBackend.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType

from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BufferInitializerDict,
    CacheConfig,
    CacheInitializerDict,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    PrepareMetadataHostCallable,
    SequenceInfo,
)

# Import cache backends

# PTCacheBackend is optional - only import if available
try:
    from .pt_cache_backend import PTCacheBackend, PTCacheConfig

    _HAS_PT_CACHE_BACKEND = True
except ImportError:
    _HAS_PT_CACHE_BACKEND = False
    PTCacheBackend = None
    PTCacheConfig = None


@dataclass
class TrtllmLayerState:
    """Per-layer state for TRT-LLM attention wrapper."""

    layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    tokens_per_block: int
    max_num_requests: int
    max_context_length: int

    # Pre-allocated tensors for metadata translation
    # Device tensors
    sequence_length: torch.Tensor = field(default=None)
    context_lengths: torch.Tensor = field(default=None)
    kv_cache_block_offsets: torch.Tensor = field(default=None)

    # Host tensors (pinned for async H2D)
    host_past_key_value_lengths: torch.Tensor = field(default=None)
    host_context_lengths: torch.Tensor = field(default=None)
    host_request_types: torch.Tensor = field(default=None)
    host_total_kv_lens: torch.Tensor = field(default=None)
    host_kv_cache_pool_pointers: torch.Tensor = field(default=None)
    host_kv_cache_pool_mapping: torch.Tensor = field(default=None)

    def __post_init__(self):
        """Allocate pre-sized tensors."""
        if self.sequence_length is None:
            device = "cuda"

            # Device tensors
            self.sequence_length = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device=device
            )
            self.context_lengths = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device=device
            )

            # Host tensors (pinned memory for async transfers)
            self.host_past_key_value_lengths = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.host_context_lengths = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.host_request_types = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.host_total_kv_lens = torch.zeros(
                2, dtype=torch.int64, device="cpu", pin_memory=True
            )
            # Pool pointers: [num_pools, 2] where each row is [k_cache_ptr, v_cache_ptr]
            # thop.attention expects 2D tensor: [num_pools, 2]
            self.host_kv_cache_pool_pointers = torch.zeros(
                1, 2, dtype=torch.int64, device="cpu", pin_memory=True
            )
            # Pool mapping: 2D [num_layers, 2] format expected by thop.attention
            # pool_mapping[layer, 0] = pool_idx (0 for single pool)
            # pool_mapping[layer, 1] = layer_offset (0 when using per-layer pointers)
            # Use max 256 layers to cover most models
            max_layers = 256
            self.host_kv_cache_pool_mapping = torch.zeros(
                max_layers, 2, dtype=torch.int32, device="cpu", pin_memory=True
            )


class TrtllmAttentionGlobalState:
    """Global state manager for TRT-LLM attention layers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._layer_states: Dict[int, TrtllmLayerState] = {}
            cls._instance._workspace: Optional[torch.Tensor] = None
            cls._instance._max_blocks_per_seq: int = 0
        return cls._instance

    def get_or_create_layer_state(
        self,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tokens_per_block: int,
        max_num_requests: int,
        max_context_length: int,
    ) -> TrtllmLayerState:
        """Get or create per-layer state."""
        if layer_idx not in self._layer_states:
            self._layer_states[layer_idx] = TrtllmLayerState(
                layer_idx=layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_num_requests=max_num_requests,
                max_context_length=max_context_length,
            )
        return self._layer_states[layer_idx]

    def init_workspace(self, buffer: torch.Tensor) -> None:
        """Initialize the global workspace buffer."""
        self._workspace = buffer

    @property
    def workspace(self) -> Optional[torch.Tensor]:
        return self._workspace

    def set_max_blocks_per_seq(self, max_blocks: int) -> None:
        """Set max blocks per sequence (needed for block offset tensor sizing)."""
        self._max_blocks_per_seq = max(self._max_blocks_per_seq, max_blocks)

    @property
    def max_blocks_per_seq(self) -> int:
        return self._max_blocks_per_seq

    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        self._layer_states.clear()
        self._workspace = None
        self._max_blocks_per_seq = 0


# Global state singleton
_global_state = TrtllmAttentionGlobalState()


def _prepare_trtllm_metadata(
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    state: TrtllmLayerState,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Prepare TRT-LLM metadata from AD metadata.

    Converts AD's metadata format to what thop.attention expects.

    Args:
        batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
        cu_seqlen_host: Cumulative sequence lengths [num_seq + 1]
        cu_num_pages: Cumulative page counts [num_seq + 1]
        cu_num_pages_host: Same as cu_num_pages but on host
        cache_loc: Flat page indices for all sequences
        last_page_len: Tokens in last page per sequence
        last_page_len_host: Same on host
        seq_len_with_cache_host: Total seq length including cached tokens
        state: Per-layer TRT-LLM state
        k_cache: K cache tensor [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: V cache tensor [num_pages, page_size, num_kv_heads, head_dim]

    Returns:
        Tuple of tensors needed by thop.attention
    """
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # Compute input sequence lengths from cumulative sums
    # cu_seqlen_host[i+1] - cu_seqlen_host[i] = seq_len[i]
    input_seq_lens = (cu_seqlen_host[1 : num_seq + 1] - cu_seqlen_host[:num_seq]).int()

    # TRT-LLM metadata:
    # - sequence_length: total length including KV cache (same as seq_len_with_cache)
    # - context_lengths: current input lengths (input_seq_lens)
    # - past_key_value_lengths: cached tokens = seq_len_with_cache - input_seq_lens

    seq_len_with_cache = seq_len_with_cache_host[:num_seq].int()
    past_kv_lens = seq_len_with_cache - input_seq_lens.cpu()

    # Copy to pre-allocated tensors
    state.sequence_length[:num_seq].copy_(seq_len_with_cache.cuda())
    state.context_lengths[:num_seq].copy_(input_seq_lens.cuda())
    state.host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
    state.host_context_lengths[:num_seq].copy_(input_seq_lens.cpu())

    # Request types: 0 = context (prefill), 1 = generation (decode)
    state.host_request_types[:num_prefill].fill_(0)
    state.host_request_types[num_prefill:num_seq].fill_(1)

    # Total KV lens for context and generation requests
    context_total_kv = seq_len_with_cache[:num_prefill].sum().item() if num_prefill > 0 else 0
    gen_total_kv = seq_len_with_cache[num_prefill:num_seq].sum().item() if num_decode > 0 else 0
    state.host_total_kv_lens[0] = context_total_kv
    state.host_total_kv_lens[1] = gen_total_kv

    # Set up KV cache pool pointers
    # The C++ kernel uses a SINGLE interleaved buffer for K/V storage:
    # - pool_pointers[0, 0] = Primary pool pointer (interleaved K/V on GPU)
    # - pool_pointers[0, 1] = Secondary pool pointer (CPU offloading, 0 if unused)
    #
    # Interleaved format: [num_blocks, 2, page_size * kv_dim]
    # - block_data[block, 0, :] = K data for block
    # - block_data[block, 1, :] = V data for block
    #
    # We need to create an interleaved buffer from AD's separate K/V caches.

    # Get cache dimensions
    num_blocks = k_cache.shape[0]
    page_size = k_cache.shape[1]
    num_kv_heads_cache = k_cache.shape[2]
    head_dim_cache = k_cache.shape[3]

    # Kernel expects cache block layout: [numHeads, tokensPerBlock, headDim]
    # Each block size = num_kv_heads * page_size * head_dim
    block_size = num_kv_heads_cache * page_size * head_dim_cache

    # Allocate interleaved buffer with layout: [total_kv_blocks, block_size]
    # where total_kv_blocks = 2 * num_blocks (K and V blocks interleaved)
    # Block order: [K0, V0, K1, V1, ...] where Ki is K block i, Vi is V block i
    total_kv_blocks = 2 * num_blocks
    if (
        not hasattr(state, "interleaved_kv_cache")
        or state.interleaved_kv_cache is None
        or state.interleaved_kv_cache.shape[0] < total_kv_blocks
    ):
        state.interleaved_kv_cache = torch.zeros(
            total_kv_blocks, block_size, dtype=k_cache.dtype, device=k_cache.device
        )

    # k_cache shape: [num_blocks, page_size, num_kv_heads, head_dim]
    # Kernel expects: [numHeads, tokensPerBlock, headDim] per block
    # Need to transpose: [block, tokens, heads, dim] -> [block, heads, tokens, dim]
    k_for_kernel = k_cache.permute(0, 2, 1, 3).reshape(num_blocks, block_size)
    v_for_kernel = v_cache.permute(0, 2, 1, 3).reshape(num_blocks, block_size)

    # Copy K blocks to even indices (0, 2, 4, ...) and V blocks to odd indices (1, 3, 5, ...)
    state.interleaved_kv_cache[0::2, :].copy_(k_for_kernel[:num_blocks])  # K at even indices
    state.interleaved_kv_cache[1::2, :].copy_(v_for_kernel[:num_blocks])  # V at odd indices

    # Set pool pointers to point to interleaved buffer
    # Position 0 = primary pool (interleaved K/V), Position 1 = secondary pool (0 = unused)
    state.host_kv_cache_pool_pointers[0, 0] = state.interleaved_kv_cache.data_ptr()
    state.host_kv_cache_pool_pointers[0, 1] = 0  # No secondary pool

    # Pool mapping: 2D tensor is pre-initialized to zeros which is correct
    # pool_mapping[layer, 0] = 0 means pool 0, pool_mapping[layer, 1] = 0 means no offset

    # Block offsets: convert flat cache_loc to per-sequence block indices
    # Shape: [num_pools, num_seq, max_blocks_per_seq]
    pages_per_seq = (cu_num_pages_host[1 : num_seq + 1] - cu_num_pages_host[:num_seq]).int()
    max_blocks = pages_per_seq.max().item() if num_seq > 0 else 1
    _global_state.set_max_blocks_per_seq(max_blocks)

    # Allocate or resize block offsets if needed
    # Shape: [num_pools, batch, 2, max_blocks] - TRT-LLM expected layout
    if (
        state.kv_cache_block_offsets is None
        or state.kv_cache_block_offsets.shape[1] < num_seq
        or state.kv_cache_block_offsets.shape[3] < max_blocks
    ):
        state.kv_cache_block_offsets = torch.zeros(
            1,  # num_pools (single pool)
            max(num_seq, state.max_num_requests),
            2,  # K and V share same block indices
            max(max_blocks, _global_state.max_blocks_per_seq),
            dtype=torch.int32,
            device=cache_loc.device,
        )

    # Fill block offsets from cache_loc
    # For interleaved format, K and V use different block indices:
    # - K blocks at even indices: 0, 2, 4, ...
    # - V blocks at odd indices: 1, 3, 5, ...
    # The interleaved pool layout is: [K_block0, V_block0, K_block1, V_block1, ...]
    state.kv_cache_block_offsets.zero_()
    offset = 0
    for i in range(num_seq):
        n_pages = pages_per_seq[i].item()
        if n_pages > 0:
            k_block_indices = cache_loc[offset : offset + n_pages] * 2  # Even indices for K
            v_block_indices = cache_loc[offset : offset + n_pages] * 2 + 1  # Odd indices for V
            state.kv_cache_block_offsets[0, i, 0, :n_pages] = k_block_indices
            state.kv_cache_block_offsets[0, i, 1, :n_pages] = v_block_indices
            offset += n_pages

    return (
        state.sequence_length[:num_seq],
        state.host_past_key_value_lengths[:num_seq],
        state.host_total_kv_lens,
        state.context_lengths[:num_seq],
        state.host_context_lengths[:num_seq],
        state.host_request_types[:num_seq],
        state.kv_cache_block_offsets[:, :num_seq, :, :max_blocks],  # 4D: [pools, seq, 2, blocks]
        state.host_kv_cache_pool_pointers,
        state.host_kv_cache_pool_mapping,  # 2D: [layers, 2], not sliced
    )


@torch.library.custom_op(
    "auto_deploy::trtllm_attention_mha_with_cache", mutates_args=("k_cache", "v_cache")
)
def trtllm_mha_with_cache(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (AD format)
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    # CACHES (AD format: separate K and V)
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    layer_idx: int,
    scale: Optional[float],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
) -> torch.Tensor:
    """TRT-LLM attention with KV cache for Auto-Deploy.

    This wraps thop.attention() with AD's metadata and cache interface.

    Note: This implementation assumes:
    - RoPE is applied OUTSIDE this kernel (AD's pattern)
    - No speculative decoding
    - No MLA
    - Causal attention mask
    """
    # Get batch dimensions
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    # Reshape inputs to TRT-LLM expected format: [num_tokens, hidden_dim]
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    # Reshape Q, K, V to [num_tokens, num_heads * head_dim]
    q_flat = q.reshape(b * s, num_heads * head_dim)[:num_tokens]
    k_flat = k.reshape(b * s, num_kv_heads * head_dim)[:num_tokens]
    v_flat = v.reshape(b * s, num_kv_heads * head_dim)[:num_tokens]

    # TRT-LLM requires FUSED QKV: concatenate Q, K, V along hidden dimension
    # Shape: [num_tokens, (num_heads + 2 * num_kv_heads) * head_dim]
    qkv_fused = torch.cat([q_flat, k_flat, v_flat], dim=-1)

    # Prepare output tensor
    output = torch.empty(num_tokens, num_heads * head_dim, dtype=q.dtype, device=q.device)

    # Check if PTCacheBackend is active - if so, use its metadata
    pt_backend = _trtllm_config.pt_cache_backend
    if pt_backend is not None:
        # PTCacheBackend's metadata is prepared via two mechanisms:
        # 1. During forward (warmup/capture/normal): host_prepare_fn is called here
        # 2. During inference: run_host_prepare_for_attention_forward() calls registered fn
        #    BEFORE graph replay (updates device tensors before replay)
        #
        # CUDA graph handling:
        # - HOST tensor VALUES are baked into the graph at capture time
        # - DEVICE tensor ADDRESSES are captured (data can be updated before replay)
        #
        # Therefore:
        # - During capture: call host_prepare_fn with skip_device_ops=True
        #   (sets host tensors correctly for capture, skips H2D copies that aren't allowed)
        # - Outside capture: call host_prepare_fn normally (sets all tensors)
        is_capturing = torch.cuda.is_current_stream_capturing()
        host_prepare_fn = pt_backend.get_host_prepare_metadata_function()
        if host_prepare_fn is not None:
            host_prepare_fn(
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages_host,
                cache_loc,
                seq_len_with_cache_host,
                skip_device_ops=is_capturing,  # Skip H2D during capture
            )

        # Get metadata from PTCacheBackend
        # NOTE: We slice tensors to num_seq because the TRT-LLM kernel uses
        # tensor SIZE to determine batch size. This is incompatible with CUDA
        # graphs since slicing creates different addresses each time.
        sequence_length = pt_backend.sequence_length[:num_seq]
        host_past_key_value_lengths = pt_backend.host_past_key_value_lengths[:num_seq]
        host_total_kv_lens = pt_backend.host_total_kv_lens
        context_lengths = pt_backend.context_lengths[:num_seq]
        host_context_lengths = pt_backend.host_context_lengths[:num_seq]
        host_request_types = pt_backend.host_request_types[:num_seq]

        # Get block offsets from PTCacheBackend - shape [1, num_seq, 2, max_blocks]
        # PTCacheBackend already sets K/V at different indices (*2 for K, *2+1 for V)
        # in _fill_block_offsets_from_cache_loc(), so use directly
        kv_cache_block_offsets = pt_backend.kv_cache_block_offsets[:, :num_seq, :, :]

        # Get pool pointers directly from C++ KVCacheManager: [[base_ptr, 0]]
        # The C++ pool stores data in [heads, tokens, dim] layout per block,
        # which matches what the kernel expects - no transpose needed!
        host_kv_cache_pool_pointers = pt_backend.get_pool_pointers()
        host_kv_cache_pool_mapping = pt_backend.get_pool_mapping()
    else:
        # Fall back to original metadata preparation
        state = _global_state.get_or_create_layer_state(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_num_requests=max_num_requests,
            max_context_length=max_context_length,
        )

        # Prepare TRT-LLM metadata using fallback
        (
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        ) = _prepare_trtllm_metadata(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            state,
            k_cache,
            v_cache,
        )

    # Compute softmax scale
    # sm_scale = scale if scale is not None else (1.0 / (head_dim**0.5))

    # Attention window (full attention)
    attention_window_size = max_context_length

    # Pack parameters for thop.attention
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    # Add extra params for newer TRT-LLM versions
    sm_version = get_sm_version()
    if sm_version >= 89:  # Ada/Hopper
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    try:
        thop.attention(
            qkv_fused,  # q (actually fused QKV)
            None,  # k (None when using fused QKV)
            None,  # v (None when using fused QKV)
            output,  # output
            None,  # output_sf (NVFP4)
            workspace_buffer,  # workspace
            sequence_length,  # sequence_length
            host_past_key_value_lengths,  # host_past_key_value_lengths
            host_total_kv_lens,  # host_total_kv_lens
            context_lengths,  # context_lengths
            host_context_lengths,  # host_context_lengths
            host_request_types,  # host_request_types
            kv_cache_block_offsets,  # kv_cache_block_offsets
            host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            None,  # cache_indirection (beam search)
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            None,  # out_scale
            None,  # rotary_inv_freq
            None,  # rotary_cos_sin
            None,  # latent_cache (MLA)
            None,  # q_pe (MLA)
            None,  # block_ids_per_seq
            None,  # attention_sinks
            True,  # is_fused_qkv (Q contains [Q,K,V] concatenated)
            True,  # update_kv_cache
            1,  # predicted_tokens_per_seq
            layer_idx,  # layer_idx
            num_heads,  # num_heads
            num_kv_heads,  # num_kv_heads
            head_dim,  # head_size
            tokens_per_block,  # tokens_per_block
            max_num_requests,  # max_num_requests
            max_context_length,  # max_context_length
            attention_window_size,  # attention_window_size
            0,  # sink_token_length
            1,  # beam_width
            int(AttentionMaskType.causal),  # mask_type
            0,  # quant_mode
            1.0,  # q_scaling (scaling factor applied to Q, typically 1.0)
            0,  # position_embedding_type (none - RoPE applied outside)
            0,  # rotary_embedding_dim
            10000.0,  # rotary_embedding_base
            0,  # rotary_embedding_scale_type
            rotary_embedding_scales,  # rotary_embedding_scales
            rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
            True,  # use_paged_context_fmha - True for paged KV cache
            None,  # attention_input_type
            False,  # is_mla_enable
            max(1, num_prefill),  # chunked_prefill_buffer_batch_size
            None,  # q_lora_rank (MLA)
            None,  # kv_lora_rank (MLA)
            None,  # qk_nope_head_dim (MLA)
            None,  # qk_rope_head_dim (MLA)
            None,  # v_head_dim (MLA)
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
            None,  # cu_q_seqlens
            None,  # cu_kv_seqlens
            None,  # fmha_scheduler_counter
            None,  # mla_bmm1_scale
            None,  # mla_bmm2_scale
            None,  # quant_q_buffer
        )
    except Exception as e:
        ad_logger.error(f"TRT-LLM attention failed at layer {layer_idx}: {e}")
        ad_logger.error(f"  num_seq={num_seq}, num_tokens={num_tokens}")
        ad_logger.error(f"  q_flat.shape={q_flat.shape}, k_flat.shape={k_flat.shape}")
        ad_logger.error(f"  k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}")
        raise

    # Deinterleave: Copy data from interleaved buffer back to AD's separate K/V caches
    # Only needed for fallback path (when state has interleaved_kv_cache)
    if (
        "state" in dir()
        and hasattr(state, "interleaved_kv_cache")
        and state.interleaved_kv_cache is not None
    ):
        num_blocks_cache = k_cache.shape[0]
        page_size_cache = k_cache.shape[1]
        num_kv_heads_cache = k_cache.shape[2]
        head_dim_cache = k_cache.shape[3]
        # block_size_cache = num_kv_heads_cache * page_size_cache * head_dim_cache

        # K at even indices (0, 2, 4, ...), V at odd indices (1, 3, 5, ...)
        # Kernel layout: [numHeads, tokensPerBlock, headDim]
        # AD layout: [tokensPerBlock, numHeads, headDim]
        k_from_kernel = state.interleaved_kv_cache[0::2, :].reshape(
            num_blocks_cache, num_kv_heads_cache, page_size_cache, head_dim_cache
        )
        v_from_kernel = state.interleaved_kv_cache[1::2, :].reshape(
            num_blocks_cache, num_kv_heads_cache, page_size_cache, head_dim_cache
        )

        # Transpose back: [block, heads, tokens, dim] -> [block, tokens, heads, dim]
        k_cache.copy_(k_from_kernel.permute(0, 2, 1, 3))
        v_cache.copy_(v_from_kernel.permute(0, 2, 1, 3))

    # Reshape output back to AD format [b, s, num_heads * head_dim]
    # Pad back to original batch*seq size if needed
    if output.shape[0] < b * s:
        output_padded = torch.zeros(b * s, num_heads * head_dim, dtype=q.dtype, device=q.device)
        output_padded[:num_tokens] = output
        output = output_padded

    return output.view(b, s, num_heads * head_dim)


@trtllm_mha_with_cache.register_fake
def trtllm_mha_with_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    layer_idx: int,
    scale: Optional[float],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(q.contiguous())


class TrtllmAttentionConfig:
    """Configuration holder for TRT-LLM attention backend.

    This class stores runtime configuration that's set during cache initialization
    and used when constructing the attention op constants.

    Attributes:
        use_pt_cache_backend: If True, use PTCacheBackend with PT's KVCacheManager
            for efficient C++ metadata preparation. If False (default), use
            SimpleCacheBackend with Python-based metadata preparation.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        """Reset configuration to defaults."""
        self.page_size: int = 64
        self.max_batch_size: int = 64
        self.max_seq_len: int = 65536
        self.max_num_tokens: int = 2048
        self.is_configured: bool = False

        # Cache backend configuration
        self.use_pt_cache_backend: bool = False
        self._pt_cache_backend: Optional["PTCacheBackend"] = None
        self._num_layers: int = 0
        self._num_kv_heads_per_layer: List[int] = []
        self._head_dim: int = 0
        self._dtype: torch.dtype = torch.float16

    def configure(self, si: SequenceInfo):
        """Configure from SequenceInfo."""
        self.page_size = si.page_size
        self.max_batch_size = si.max_batch_size
        self.max_seq_len = si.max_seq_len
        self.max_num_tokens = si.max_num_tokens
        self.is_configured = True
        ad_logger.info(
            f"[TRT-LLM Attention Config] page_size={self.page_size}, "
            f"max_batch_size={self.max_batch_size}, max_seq_len={self.max_seq_len}, "
            f"max_num_tokens={self.max_num_tokens}, use_pt_cache_backend={self.use_pt_cache_backend}"
        )

    def set_model_config(
        self,
        num_layers: int,
        num_kv_heads_per_layer: List[int],
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Set model configuration for PTCacheBackend.

        This should be called during model analysis before cache initialization.

        Args:
            num_layers: Total number of attention layers
            num_kv_heads_per_layer: Number of KV heads for each layer
            head_dim: Head dimension
            dtype: Cache data type
        """
        self._num_layers = num_layers
        self._num_kv_heads_per_layer = num_kv_heads_per_layer
        self._head_dim = head_dim
        self._dtype = dtype

    def get_or_create_pt_cache_backend(self, si: SequenceInfo) -> Optional["PTCacheBackend"]:
        """Get or create the PTCacheBackend instance.

        This is called during cache initialization if use_pt_cache_backend is True.

        Args:
            si: SequenceInfo with cache configuration

        Returns:
            PTCacheBackend instance, or None if not configured to use it
        """
        if not self.use_pt_cache_backend:
            return None

        if not _HAS_PT_CACHE_BACKEND:
            ad_logger.warning(
                "[TRT-LLM] PTCacheBackend requested but not available. "
                "Falling back to SimpleCacheBackend."
            )
            return None

        if self._pt_cache_backend is not None:
            return self._pt_cache_backend

        # Validate we have model config
        if self._num_layers == 0 or not self._num_kv_heads_per_layer:
            ad_logger.error(
                "[TRT-LLM] Cannot create PTCacheBackend: model config not set. "
                "Call set_model_config() first."
            )
            return None

        # Create PTCacheConfig - use AD's calculated num_pages to avoid OOM
        config = PTCacheConfig(
            num_layers=self._num_layers,
            num_kv_heads_per_layer=self._num_kv_heads_per_layer,
            head_dim=self._head_dim,
            tokens_per_block=si.page_size,
            max_num_sequences=si.max_batch_size,
            max_seq_len=si.max_seq_len,
            num_pages=si.num_pages,  # Use AD's calculated pages, not theoretical max
            dtype=self._dtype,
        )

        # Create and initialize backend
        self._pt_cache_backend = PTCacheBackend(config)
        self._pt_cache_backend.initialize(si, si.device)

        ad_logger.info(
            f"[TRT-LLM] Created PTCacheBackend: num_layers={self._num_layers}, "
            f"num_pages={si.num_pages}, tokens_per_block={si.page_size}"
        )

        return self._pt_cache_backend

    @property
    def pt_cache_backend(self) -> Optional["PTCacheBackend"]:
        """Get the PTCacheBackend instance if available."""
        return self._pt_cache_backend


# Global config singleton
_trtllm_config = TrtllmAttentionConfig()


@AttentionRegistry.register("trtllm")
class TrtllmAttention(AttentionDescriptor):
    """TRT-LLM attention backend for Auto-Deploy.

    This backend uses the optimized thop.attention kernel from TRT-LLM,
    providing better performance than FlashInfer on certain workloads.

    Note: This backend assumes RoPE is applied outside the attention kernel,
    which matches AD's current pattern.

    Cache Backend Options:
        - SimpleCacheBackend (default): Per-layer cache allocation
        - PTCacheBackend: Uses PT's KVCacheManager with C++ fast path

    To enable PTCacheBackend, set:
        TrtllmAttentionConfig().use_pt_cache_backend = True

    Usage:
        Set `backend: trtllm` in your AD config under `insert_cached_attention`.
    """

    # Class-level counter for layer indices
    _layer_counter: int = 0

    # Track num_kv_heads per layer for PTCacheBackend config
    _num_kv_heads_per_layer: List[int] = []
    _head_dim: int = 0
    _dtype: torch.dtype = torch.float16

    @classmethod
    def _get_next_layer_idx(cls) -> int:
        """Get the next layer index and increment counter."""
        idx = cls._layer_counter
        cls._layer_counter += 1
        return idx

    @classmethod
    def _reset_layer_counter(cls) -> None:
        """Reset layer counter (for testing or new model builds)."""
        cls._layer_counter = 0
        cls._num_kv_heads_per_layer = []
        cls._head_dim = 0
        cls._dtype = torch.float16
        _global_state.reset()
        _trtllm_config.reset()

    @classmethod
    def _track_layer_config(cls, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> None:
        """Track layer configuration for PTCacheBackend setup.

        This is called for each layer during graph analysis to collect
        the per-layer KV head counts needed by PTCacheBackend.
        """
        cls._num_kv_heads_per_layer.append(num_kv_heads)
        cls._head_dim = head_dim
        cls._dtype = dtype

    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op."""
        return torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments."""
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages",
            "cu_num_pages_host",
            "cache_loc",
            "last_page_len",
            "last_page_len_host",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Get the prepare_metadata op info."""
        # TRT-LLM doesn't need extra metadata preparation like FlashInfer
        return (None, 0, [])

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        """Provide cache initializer functions.

        If PTCacheBackend is enabled, returns initializers that get cache views
        from the unified pool. Otherwise, allocates separate per-layer caches.
        """
        # Extract tensor shapes from source node
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        dtype = cache_config.dtype or k_fake.dtype

        # Get current layer index for PTCacheBackend
        layer_idx = cls._layer_counter  # Current layer being processed

        def _get_cache_simple(si: SequenceInfo):
            """Simple cache allocation (default path)."""
            # Configure global state from SequenceInfo (first time only)
            if not _trtllm_config.is_configured:
                _trtllm_config.configure(si)

            # AD cache format: [num_pages, page_size, num_kv_heads, head_dim]
            cache = torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                head_dim,
                device=si.device,
                dtype=dtype,
            )
            ad_logger.debug(
                f"[TRT-LLM] Created cache: shape={cache.shape}, dtype={cache.dtype}, "
                f"device={cache.device}"
            )
            return cache

        def _get_k_cache_pt(si: SequenceInfo):
            """Get K cache from PTCacheBackend."""
            # Configure global state first
            if not _trtllm_config.is_configured:
                _trtllm_config.configure(si)

            # Set model config before creating backend
            if _trtllm_config._num_layers == 0:
                _trtllm_config.set_model_config(
                    num_layers=len(cls._num_kv_heads_per_layer),
                    num_kv_heads_per_layer=cls._num_kv_heads_per_layer,
                    head_dim=cls._head_dim,
                    dtype=cls._dtype,
                )

            # Get or create PT backend
            pt_backend = _trtllm_config.get_or_create_pt_cache_backend(si)

            if pt_backend is None:
                # Fall back to simple allocation
                return _get_cache_simple(si)

            # Register host prepare function ONCE (only for layer 0)
            # This function is called by run_host_prepare_for_attention_forward()
            # BEFORE each forward pass, outside of CUDA graph capture
            if layer_idx == 0:
                host_fn = pt_backend.get_host_prepare_metadata_function()
                if host_fn is not None:
                    host_args = pt_backend.get_host_prepare_metadata_args()
                    si.register_host_prepare_for_attention_forward(host_fn, host_args)

            return pt_backend.get_cache("k_cache", layer_idx)

        def _get_v_cache_pt(si: SequenceInfo):
            """Get V cache from PTCacheBackend."""
            pt_backend = _trtllm_config.pt_cache_backend
            if pt_backend is None:
                return _get_cache_simple(si)
            return pt_backend.get_cache("v_cache", layer_idx)

        # Use PT backend initializers if enabled
        if _trtllm_config.use_pt_cache_backend and _HAS_PT_CACHE_BACKEND:
            return {"k_cache": _get_k_cache_pt, "v_cache": _get_v_cache_pt}
        else:
            return {"k_cache": _get_cache_simple, "v_cache": _get_cache_simple}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        """Provide global buffer initializer functions."""

        def _init_workspace(si: SequenceInfo) -> torch.Tensor:
            # TRT-LLM workspace - size based on PT backend (64 MB)
            buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=si.device)
            _global_state.init_workspace(buffer)
            ad_logger.debug(
                f"[TRT-LLM] Initialized workspace: {buffer.shape}, device={buffer.device}"
            )
            return buffer

        return {"workspace_buffer": _init_workspace}

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Get function that performs host-side prep for attention.

        If PTCacheBackend is enabled, returns the backend's efficient
        metadata preparation function. Otherwise, returns None (metadata
        is prepared inside the kernel call).
        """
        # Check if we're using PTCacheBackend
        if _trtllm_config.use_pt_cache_backend and _trtllm_config.pt_cache_backend is not None:
            return _trtllm_config.pt_cache_backend.get_host_prepare_metadata_function()

        # Default: metadata is prepared inside the kernel call
        return None

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Provide constant arguments to be passed to the attention op."""
        # Sanity check layout
        layout = source_attn_node.kwargs.get("layout", None)
        if (
            layout is None
            and len(source_attn_node.args) > 0
            and isinstance(source_attn_node.args[-1], str)
        ):
            layout = source_attn_node.args[-1]
        if layout != "bsnd":
            raise RuntimeError(
                f"Expected torch_attention layout='bsnd' but got {layout!r} "
                f"for node: {source_attn_node.format_node()}"
            )

        # Extract attention parameters
        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                f"Unsupported attention arguments for {source_attn_node=}: "
                f"{attn_mask=}, {dropout_p=}, {is_causal=}"
            )

        # Get scale
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        # Extract tensor shapes for constants
        q_fake: FakeTensor = source_attn_node.args[0].meta["val"]
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]

        num_heads = q_fake.shape[2]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        dtype = k_fake.dtype

        # Track layer configuration for PTCacheBackend
        cls._track_layer_config(num_kv_heads, head_dim, dtype)

        # Get layer index
        layer_idx = cls._get_next_layer_idx()

        # Use configured values if available, otherwise defaults
        tokens_per_block = _trtllm_config.page_size
        max_num_requests = _trtllm_config.max_batch_size
        max_context_length = _trtllm_config.max_seq_len

        ad_logger.debug(
            f"[TRT-LLM] Layer {layer_idx} constants: num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, scale={scale}, "
            f"tokens_per_block={tokens_per_block}, max_num_requests={max_num_requests}, "
            f"max_context_length={max_context_length}"
        )

        # Return constants in order expected by trtllm_mha_with_cache
        return [
            layer_idx,  # layer_idx
            scale,  # scale
            num_heads,  # num_heads
            num_kv_heads,  # num_kv_heads
            head_dim,  # head_dim
            tokens_per_block,  # tokens_per_block (from AD's page_size)
            max_num_requests,  # max_num_requests (from AD's max_batch_size)
            max_context_length,  # max_context_length (from AD's max_seq_len)
        ]


# =============================================================================
# Public API Functions
# =============================================================================


def enable_pt_cache_backend(enable: bool = True) -> None:
    """Enable or disable PTCacheBackend for TRT-LLM attention.

    When enabled, the TRT-LLM attention backend uses PT's KVCacheManager
    for efficient metadata preparation via C++ code paths.

    Benefits of PTCacheBackend:
    - ~50% faster metadata preparation (C++ vs Python loops)
    - Pre-allocated tensors for CUDA graph compatibility
    - Direct access to unified pool pointers for thop.attention

    Limitations (current implementation):
    - Does not support block reuse (AD manages page assignments)
    - Does not support host offloading

    Usage:
        # Before building the model with AD
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            enable_pt_cache_backend
        )
        enable_pt_cache_backend(True)

        # Then proceed with AD model building
        # ...

    Args:
        enable: Whether to enable PTCacheBackend (default: True)
    """
    if enable and not _HAS_PT_CACHE_BACKEND:
        ad_logger.warning(
            "PTCacheBackend is not available (missing TensorRT-LLM bindings). "
            "Falling back to SimpleCacheBackend."
        )
        return

    _trtllm_config.use_pt_cache_backend = enable
    ad_logger.info(f"[TRT-LLM] PTCacheBackend {'enabled' if enable else 'disabled'}")


def get_pt_cache_backend() -> Optional["PTCacheBackend"]:
    """Get the current PTCacheBackend instance if available.

    Returns:
        The PTCacheBackend instance, or None if not initialized or disabled.
    """
    return _trtllm_config.pt_cache_backend


def is_pt_cache_backend_enabled() -> bool:
    """Check if PTCacheBackend is enabled.

    Returns:
        True if PTCacheBackend is enabled and available.
    """
    return _trtllm_config.use_pt_cache_backend and _HAS_PT_CACHE_BACKEND


def reset_trtllm_attention_state() -> None:
    """Reset all TRT-LLM attention state.

    Call this before building a new model to ensure clean state.
    This resets:
    - Layer counter
    - Global state (per-layer states, workspace)
    - Configuration (page size, batch size, etc.)
    - PTCacheBackend instance
    """
    TrtllmAttention._reset_layer_counter()
    ad_logger.info("[TRT-LLM] Attention state reset")
