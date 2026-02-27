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
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args
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
# Module-level planner (analogous to _GlobalFlashInferPlanner)
# =============================================================================


class _TrtllmPlanner:
    """Minimal planner for TRT-LLM attention backend.

    Analogous to ``_FlashInferPlanner`` in the FlashInfer backend. Only stores
    data that cannot be derived from SequenceInfo or tensor shapes.

    Two main entry points:
    - ``reset()``: one-time allocation of ALL persistent buffers.
    - ``plan()``: per-forward host metadata (host_request_types, block_offsets, host_total_kv_lens).
    """

    def __init__(self):
        self.workspace: Optional[torch.Tensor] = None
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

    def reset(self, device: torch.device, max_batch: int, max_blocks_per_seq: int) -> None:
        """One-time allocation of ALL persistent buffers.

        Guards against double-init. Called lazily from ``prepare_trtllm_metadata``
        on the first forward pass after cache initialization.
        """
        if self.workspace is not None:
            return  # already initialized

        # Workspace: pre-allocate a modest initial buffer (like flashinfer's 320MB).
        # thop.attention auto-resizes via resize_() if more space is needed during warm-up.
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

        Called from ``prepare_trtllm_metadata_host`` before every forward
        (including CUDA graph replays).
        """
        num_seq = num_prefill + num_decode

        # host_request_types: 0 = prefill (context), 1 = decode (generation)
        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

        # host_total_kv_lens: [context_total_kv, gen_total_kv]
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

    def plan_device(
        self,
        num_seq: int,
        block_offset_multiplier: int,
        cu_num_pages: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> None:
        """Per-forward DEVICE metadata: block_offsets via Triton kernel (pure GPU).

        Called from the ``prepare_trtllm_metadata`` custom op (in the graph).
        """
        k_slice = self.block_offsets[0, :, 0, :]  # [max_batch, M], stride [2*M, 1]
        torch.ops.auto_deploy.ragged_to_block_table_triton(
            cache_loc, cu_num_pages, k_slice, num_seq
        )
        self.block_offsets[0, :num_seq, 0, :].mul_(block_offset_multiplier)
        self.block_offsets[0, :num_seq, 1, :] = self.block_offsets[0, :num_seq, 0, :] + 1


_GlobalTrtllmPlanner = _TrtllmPlanner()


# =============================================================================
# Host-side prepare function (runs outside CUDA graph, before every forward)
# =============================================================================


def prepare_trtllm_metadata_host(
    batch_info_host: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_host: torch.Tensor,
) -> None:
    """Fill thop-specific HOST metadata (pinned tensors for thop.attention).

    Runs OUTSIDE the CUDA graph before every forward (including replays).
    Handles host_request_types, host_total_kv_lens, host_past_kv_lengths,
    host_context_lengths.
    """
    num_prefill, _, num_decode = batch_info_host.tolist()
    max_context_length, max_blocks_per_seq, _, max_batch_size = max_seq_info_host.tolist()

    _GlobalTrtllmPlanner.reset(torch.device("cuda"), max_batch_size, max_blocks_per_seq)
    _GlobalTrtllmPlanner.plan_host(
        num_prefill=num_prefill,
        num_decode=num_decode,
        max_context_length=max_context_length,
        seq_len_with_cache_host=seq_len_with_cache_host,
        input_pos_host=input_pos_host,
        seq_len_host=seq_len_host,
    )


# =============================================================================
# Device-side prepare function (inserted into the graph, analogous to
# prepare_flashinfer_metadata)
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_attention_prepare_metadata", mutates_args=())
def prepare_trtllm_metadata(
    batch_info_host: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute block_offsets for thop.attention (device-side, part of the traced graph).

    Uses the ``ragged_to_block_table_triton`` Triton kernel to scatter ``cache_loc``
    pages into the block_offsets table. All operations are pure GPU.

    Returns ``block_offsets`` which flows through the graph to each attention op,
    creating an explicit data dependency.
    """
    num_prefill, _, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    _, _, block_offset_multiplier, _ = max_seq_info_host.tolist()

    _GlobalTrtllmPlanner.plan_device(
        num_seq=num_seq,
        block_offset_multiplier=block_offset_multiplier,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
    )

    return [_GlobalTrtllmPlanner.block_offsets]


@prepare_trtllm_metadata.register_fake
def prepare_trtllm_metadata_fake(
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
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # EXTRA METADATA (from prepare_trtllm_metadata device-side op)
    kv_cache_block_offsets: torch.Tensor,
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

    ``kv_cache_block_offsets`` is computed by the ``prepare_trtllm_metadata`` device-side
    op and flows through the graph to create an explicit data dependency.

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

    # Per-layer pool pointer tensor (created per invocation from kv_cache.data_ptr())
    host_kv_cache_pool_pointers = torch.zeros(
        1, 2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
    )
    host_kv_cache_pool_pointers[0, 0] = kv_cache.data_ptr()

    # FP8 KV cache scale tensors
    if kv_cache.dtype == torch.float8_e4m3fn:
        kv_scale_oq = torch.tensor([kv_scale_orig_quant], dtype=torch.float32, device=q.device)
        kv_scale_qo = torch.tensor([kv_scale_quant_orig], dtype=torch.float32, device=q.device)
        quant_mode = int(QuantMode.FP8_KV_CACHE)
    else:
        kv_scale_oq = None
        kv_scale_qo = None
        quant_mode = 0

    # Reshape Q, K, V to [num_tokens, num_heads * head_dim] and fuse
    # Input is always [bs, 1] (generate-only) or [1, total_seq_len] (prefill/mixed),
    # so b * s == num_tokens always holds.
    q_shape_og = q.shape
    q_flat = q.reshape(num_tokens, num_heads * head_dim)
    k_flat = k.reshape(num_tokens, num_kv_heads * head_dim)
    v_flat = v.reshape(num_tokens, num_kv_heads * head_dim)
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
        kv_scale_oq,  # kv_scale_orig_quant
        kv_scale_qo,  # kv_scale_quant_orig
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
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # EXTRA METADATA (from prepare_trtllm_metadata device-side op)
    kv_cache_block_offsets: torch.Tensor,
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
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Return the device-side prepare_metadata op for block_offsets computation."""
        return (torch.ops.auto_deploy.trtllm_attention_prepare_metadata.default, 1, [])

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Return host-side prepare function for thop-specific pinned tensors."""
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
