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

"""TRT-LLM attention backend for Auto-Deploy.

This module wraps TRT-LLM's optimized ``thop.attention`` kernel for use in Auto-Deploy,
following the same design pattern as the FlashInfer backend:

- Minimal module-level state (``_TrtllmPlanner``, analogous to ``_FlashInferPlanner``)
- SequenceInfo fields used directly as thop.attention metadata
- Pool pointers derived lazily from ``kv_cache.data_ptr()``
- Workspace managed as module-level state (not a ResourceHandler / graph input)
- All possible "constants" inferred from tensor shapes at runtime
"""

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
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args
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
# Module-level planner (analogous to _GlobalFlashInferPlanner)
# =============================================================================


class _TrtllmPlanner:
    """Minimal planner for TRT-LLM attention backend.

    Analogous to ``_FlashInferPlanner`` in the FlashInfer backend. Only stores
    data that cannot be derived from SequenceInfo or tensor shapes.

    Two main entry points:
    - ``reset()``: one-time allocation of ALL persistent buffers.
    - ``plan()``: per-forward host metadata (host_request_types, block_offsets, host_total_kv_lens).

    Pool pointer management:
    Each attention layer needs its own ``host_pool_pointers`` tensor so that CUDA graph
    replay reads the correct (layer-specific) pool base from a stable tensor address.
    Tensors are created lazily on first access per layer and never re-allocated.
    """

    def __init__(self):
        self.workspace: Optional[torch.Tensor] = None
        # Per-layer pool pointer tensors, keyed by kv_cache.data_ptr().
        # Each is [1, 2] int64 pinned, created lazily on first access per layer.
        # This ensures each layer's attention kernel in a CUDA graph is captured
        # with its own stable tensor address, avoiding the issue where a shared
        # tensor would hold only the last-set layer's pointer during graph replay.
        self._per_layer_pool_ptrs: dict = {}
        # pool_mapping: fixed [1, 2] all zeros since we always pass layer_idx=0
        # and pool_pointers already encodes the layer offset via kv_cache.data_ptr()
        self.host_pool_mapping: Optional[torch.Tensor] = None  # [1, 2] int32 pinned
        # thop-specific host metadata NOT available from SequenceInfo
        self.host_request_types: Optional[torch.Tensor] = None  # [max_batch] int32 pinned
        self.host_total_kv_lens: Optional[torch.Tensor] = None  # [2] int64 pinned
        # thop variant of input_pos_host and seq_len_host
        # keeping a separate copy here since we sometimes have to overwrite the original values
        self.host_past_kv_lengths: Optional[torch.Tensor] = None  # [max_batch] int32 pinned
        self.host_context_lengths: Optional[torch.Tensor] = None  # [max_batch] int32 pinned
        # Persistent block_offsets buffer for CUDA graph compatibility.
        # Pre-allocated to max size so the tensor address is stable across replays.
        self.block_offsets: Optional[torch.Tensor] = None
        # FP8 scale tensors (lazily initialized from constants on first FP8 use)
        self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        self.kv_scale_quant_orig: Optional[torch.Tensor] = None

    def reset(self, device: torch.device, max_batch: int, max_blocks_per_seq: int) -> None:
        """One-time allocation of ALL persistent buffers.

        Guards against double-init. Called lazily from ``prepare_trtllm_metadata_host``
        on the first forward pass after cache initialization.
        """
        if self.workspace is not None:
            return  # already initialized

        # Workspace: pre-allocate a modest initial buffer (like flashinfer's 320MB).
        # thop.attention auto-resizes via resize_() if more space is needed during warm-up.
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
        """Per-forward host metadata: fills host_request_types, block_offsets, host_total_kv_lens.

        Called from ``prepare_trtllm_metadata_host`` before every forward (including replays).
        """
        num_seq = num_prefill + num_decode

        # host_request_types: 0 = prefill (context), 1 = decode (generation)
        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

        # Compute block_offsets for thop.attention using pre-computed page indices.
        block_offsets = self.block_offsets
        total_pages = int(cu_num_pages_host[num_seq])
        base_offsets = cache_loc[:total_pages] * block_offset_multiplier
        seq_idx = page_seq_indices[:total_pages]
        pg_idx = page_in_seq[:total_pages]
        block_offsets[0, seq_idx, 0, pg_idx] = base_offsets  # K
        block_offsets[0, seq_idx, 1, pg_idx] = base_offsets + 1  # V

        # host_total_kv_lens: [context_total_kv, gen_total_kv]
        is_capturing = torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up()
        if is_capturing:
            # CUDA graph capture: set host tensors to MAX values so the kernel captures
            # the worst-case execution pattern.
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
        """Return a per-layer ``host_pool_pointers`` tensor for this kv_cache view.

        Each attention layer receives a different ``kv_cache`` tensor (a strided view
        into the pool).  We create one pinned [1, 2] int64 tensor per unique
        ``data_ptr`` and cache it forever.  This guarantees that each layer's
        ``thop.attention`` call in a CUDA graph is captured with a *stable, distinct*
        tensor address, so graph replay reads the correct pool base for every layer.
        """
        ptr = kv_cache.data_ptr()
        t = self._per_layer_pool_ptrs.get(ptr)
        if t is not None:
            return t

        t = torch.zeros(1, 2, dtype=torch.int64, device="cpu", pin_memory=True)
        t[0, 0] = ptr
        self._per_layer_pool_ptrs[ptr] = t
        return t


_GlobalTrtllmPlanner = _TrtllmPlanner()


# =============================================================================
# Host-side prepare function (analogous to prepare_flashinfer_metadata_host)
# =============================================================================


def prepare_trtllm_metadata_host(
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
    """Fill thop-specific host metadata and compute block_offsets.

    This runs OUTSIDE the CUDA graph before every forward (including replays).
    Block offsets MUST be computed here (not in the device-side prepare_metadata op)
    because they are batch-dependent and need to be updated before each replay.

    All max-size constants are read from ``max_seq_info_host`` which is set once via
    ``SequenceInfo.update_cache_information()`` after cache initialization:
    ``[max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size]``

    ``page_seq_indices`` and ``page_in_seq`` are pre-computed in SequenceInfo from
    ``pages_per_seq`` and avoid the expensive GPU searchsorted that was previously needed.
    """
    num_prefill, _, num_decode = batch_info_host.tolist()

    # Read all max-size constants from max_seq_info_host (set at cache init time)
    max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size = (
        max_seq_info_host.tolist()
    )

    # One-time allocation of all persistent buffers (lazy, guards against double-init)
    _GlobalTrtllmPlanner.reset(cache_loc.device, max_batch_size, max_blocks_per_seq)

    # Per-forward: fill host_request_types, block_offsets, host_total_kv_lens
    _GlobalTrtllmPlanner.plan(
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
# Cached attention op (analogous to flashinfer_mha_with_cache)
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_attention_mha_with_cache", mutates_args=("kv_cache",))
def trtllm_mha_with_cache(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (SequenceInfo fields used directly by thop)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # CACHE
    kv_cache: torch.Tensor,
    # CONSTANTS (only truly un-inferable values)
    scale: Optional[float],
    sliding_window: Optional[int] = None,
    kv_scale_orig_quant: float = 1.0,
    kv_scale_quant_orig: float = 1.0,
) -> torch.Tensor:
    """TRT-LLM attention with paged KV cache for Auto-Deploy.

    Infers num_heads, num_kv_heads, head_dim, and tokens_per_block from tensor shapes.
    All max-size constants (max_num_requests, max_context_length) are read from
    ``max_seq_info_host`` which is set once via ``SequenceInfo.update_cache_information()``.

    Note: ``prepare_trtllm_metadata_host`` is guaranteed to be called before this op,
    so all persistent planner buffers are already initialized.

    Note: layer_idx is always passed as 0 to thop.attention because
    the kv_cache tensor is already a strided view for the correct layer,
    pool_pointers encodes kv_cache.data_ptr() (layer-specific), and
    pool_mapping is all zeros. See module docstring for details.
    """
    # Infer dimensions from tensor shapes (bsnd layout)
    num_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[3]
    tokens_per_block = kv_cache.shape[3]  # HND: [blocks, 2, heads, tpb, head_dim]

    # Get batch dimensions and model-level constants from host tensors (no device sync)
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode
    max_context_length = int(max_seq_info_host[0])
    max_num_requests = int(max_seq_info_host[3])
    # Use sliding_window for attention_window_size if provided, else full context length
    attention_window_size = (
        sliding_window
        if isinstance(sliding_window, int) and sliding_window > 0
        else max_context_length
    )

    # Get per-layer pool pointer tensor (stable address for CUDA graph replay)
    host_kv_cache_pool_pointers = _GlobalTrtllmPlanner.get_pool_pointers_for_layer(kv_cache)

    # FP8 KV cache: lazily create scale tensors from float constants on first use
    if kv_cache.dtype == torch.float8_e4m3fn:
        if _GlobalTrtllmPlanner.kv_scale_orig_quant is None:
            _GlobalTrtllmPlanner.kv_scale_orig_quant = torch.tensor(
                [kv_scale_orig_quant], dtype=torch.float32, device=q.device
            )
            _GlobalTrtllmPlanner.kv_scale_quant_orig = torch.tensor(
                [kv_scale_quant_orig], dtype=torch.float32, device=q.device
            )
        quant_mode = int(QuantMode.FP8_KV_CACHE)
    else:
        quant_mode = 0

    # Reshape Q, K, V to [num_tokens, num_heads * head_dim] and fuse.
    # Input is [bs, 1] (generate-only) or [1, total_seq_len] (prefill/mixed).
    # With piecewise CUDA graphs the tensor may be padded to a bucket size
    # (b*s > num_tokens), so flatten first and slice to the real token count.
    q_shape_og = q.shape
    q_flat = q.reshape(-1, num_heads * head_dim)[:num_tokens]
    k_flat = k.reshape(-1, num_kv_heads * head_dim)[:num_tokens]
    v_flat = v.reshape(-1, num_kv_heads * head_dim)[:num_tokens]
    qkv_fused = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()

    # Prepare output
    output = torch.empty(num_tokens, num_heads * head_dim, dtype=q.dtype, device=q.device)

    # Map SequenceInfo fields to thop.attention args
    sequence_length = seq_len_with_cache[:num_seq]  # device
    context_lengths = seq_len[:num_seq]  # device
    host_past_kv_lengths = _GlobalTrtllmPlanner.host_past_kv_lengths[:num_seq]  # host (pinned)
    host_context_lengths = _GlobalTrtllmPlanner.host_context_lengths[:num_seq]  # host (pinned)

    # thop-specific metadata from _GlobalTrtllmPlanner
    host_request_types = _GlobalTrtllmPlanner.host_request_types[:num_seq]
    host_total_kv_lens = _GlobalTrtllmPlanner.host_total_kv_lens

    # Block offsets from host_prepare
    kv_cache_block_offsets = _GlobalTrtllmPlanner.block_offsets

    # Pool mapping (shared, always zeros since layer offset is in pool_pointers)
    host_kv_cache_pool_mapping = _GlobalTrtllmPlanner.host_pool_mapping

    # Pack parameters for thop.attention
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    sm_version = get_sm_version()
    if sm_version >= 89:  # Ada/Hopper
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    thop.attention(
        qkv_fused,  # q (actually fused QKV)
        None,  # k (None when using fused QKV)
        None,  # v (None when using fused QKV)
        output,  # output
        None,  # output_sf (NVFP4)
        _GlobalTrtllmPlanner.workspace,  # workspace (module-level, like flashinfer)
        sequence_length,  # sequence_length
        host_past_kv_lengths,  # host_past_key_value_lengths
        host_total_kv_lens,  # host_total_kv_lens
        context_lengths,  # context_lengths
        host_context_lengths,  # host_context_lengths
        host_request_types,  # host_request_types
        kv_cache_block_offsets,  # kv_cache_block_offsets
        host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
        host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
        None,  # cache_indirection (beam search)
        _GlobalTrtllmPlanner.kv_scale_orig_quant,  # kv_scale_orig_quant
        _GlobalTrtllmPlanner.kv_scale_quant_orig,  # kv_scale_quant_orig
        None,  # out_scale
        None,  # rotary_inv_freq
        None,  # rotary_cos_sin
        None,  # latent_cache (MLA)
        None,  # q_pe (MLA)
        None,  # block_ids_per_seq
        None,  # attention_sinks
        True,  # is_fused_qkv
        True,  # update_kv_cache
        1,  # predicted_tokens_per_seq
        0,  # layer_idx (always 0; pool_pointers already encodes the layer offset)
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
        quant_mode,  # quant_mode
        1.0,  # q_scaling
        0,  # position_embedding_type
        0,  # rotary_embedding_dim
        10000.0,  # rotary_embedding_base
        0,  # rotary_embedding_scale_type
        rotary_embedding_scales,  # rotary_embedding_scales
        rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
        True,  # use_paged_context_fmha
        0,  # attention_input_type
        False,  # is_mla_enable
        max_num_requests,  # chunked_prefill_buffer_batch_size
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

    # If input was padded (piecewise CG), embed the real output into a padded
    # tensor so downstream static segments see the expected bucket-sized shape.
    total_padded_tokens = q_shape_og[0] * q_shape_og[1]
    if total_padded_tokens > num_tokens:
        padded_output = torch.zeros(
            total_padded_tokens, num_heads * head_dim, dtype=q.dtype, device=q.device
        )
        padded_output[:num_tokens] = output
        return padded_output.view(*q_shape_og)
    return output.view(*q_shape_og)


@trtllm_mha_with_cache.register_fake
def trtllm_mha_with_cache_fake(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (SequenceInfo fields used directly by thop)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # CACHE
    kv_cache: torch.Tensor,
    # CONSTANTS (only truly un-inferable values)
    scale: Optional[float],
    sliding_window: Optional[int] = None,
    kv_scale_orig_quant: float = 1.0,
    kv_scale_quant_orig: float = 1.0,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(q.contiguous())


# =============================================================================
# AttentionDescriptor (analogous to FlashInferAttention)
# =============================================================================


@AttentionRegistry.register("trtllm")
class TrtllmAttention(AttentionDescriptor):
    """TRT-LLM attention backend for Auto-Deploy.

    Follows the same stateless descriptor pattern as ``FlashInferAttention``.
    """

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
        """Get the list of standard metadata arguments from SequenceInfo."""
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
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Return only KV cache handler (no workspace handler, managed like flashinfer)."""
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]

        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads,
                head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype),
                kv_factor=2,
                kv_layout="HND",
            )
        }

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Return host-side prepare function for thop-specific metadata."""
        return prepare_trtllm_metadata_host

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Extract constants from the source attention node.

        Returns scale, sliding_window, kv_scale_orig_quant, and kv_scale_quant_orig.
        Everything else (num_heads, head_dim, max_context_length, etc.) is inferred
        from tensor shapes or SequenceInfo metadata at runtime.
        """
        # Sanity check: layout == "bsnd"
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

        # Check other arguments
        _attn_mask, _dropout_p, _is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )

        # Get scale
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        # Get sliding_window from source attention node
        sliding_window = extract_op_args(source_attn_node, "sliding_window")[0]

        return [
            scale,
            sliding_window,
            1.0,  # kv_scale_orig_quant (hard-coded, same as FlashInfer)
            1.0,  # kv_scale_quant_orig (hard-coded, same as FlashInfer)
        ]
