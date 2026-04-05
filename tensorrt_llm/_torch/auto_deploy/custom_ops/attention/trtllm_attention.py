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

import functools
import math
from typing import List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import GraphModule, Node

from tensorrt_llm._utils import get_sm_version, nvtx_range, prefer_pinned
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
    BatchInfo,
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
    - ``init_spec_decoding()``: initialize parameters specific to spec-decoding kernels once when configured.
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
        self.context_lengths_gpu: Optional[torch.Tensor] = None  # [max_batch] int32 device
        # Persistent block_offsets buffer for CUDA graph compatibility.
        # Pre-allocated to max size so the tensor address is stable across replays.
        self.block_offsets: Optional[torch.Tensor] = None
        # Per-layer cache for tensors that must survive CUDA graph replay.
        # Keyed by kv_cache.data_ptr() (stable and unique per layer).
        self._layer_cache: dict[
            int, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = {}
        # Spec-dec state for Eagle linear chain.
        # Initialized once in prepare_trtllm_metadata_host when spec_config is provided.
        self.is_spec_decoding_enabled: bool = False
        self.predicted_tokens_per_seq: int = 1
        self.spec_decoding_generation_lengths: Optional[torch.Tensor] = None
        self.spec_decoding_position_offsets: Optional[torch.Tensor] = None
        self.spec_decoding_packed_mask: Optional[torch.Tensor] = None

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
        self.context_lengths_gpu = torch.zeros(max_batch, dtype=torch.int32, device=device)

    def init_spec_decoding(self, max_batch: int, spec_config=None) -> None:
        """Initialize persistent spec-decoding tensors once when configured."""
        self.predicted_tokens_per_seq = 1

        if spec_config is None or self.is_spec_decoding_enabled:
            return

        draft_len = spec_config.max_draft_len
        self.is_spec_decoding_enabled = True
        self.spec_decoding_generation_lengths = torch.full(
            (max_batch,), draft_len + 1, dtype=torch.int, device="cuda"
        )
        self.spec_decoding_position_offsets = _generate_spec_decoding_position_offsets(
            max_batch, draft_len
        )
        self.spec_decoding_packed_mask = _generate_spec_decoding_packed_mask(max_batch, draft_len)

    def get_layer_tensors(
        self,
        kv_cache: torch.Tensor,
        kv_scale_orig_quant: float,
        kv_scale_quant_orig: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """Return (pool_pointers, kv_scale_oq, kv_scale_qo, quant_mode) for a layer.

        Lazily creates and caches the tensors on first call per layer to avoid
        dynamic allocations during CUDA graph capture.
        """
        key = kv_cache.data_ptr()
        if key not in self._layer_cache:
            pool_ptrs = torch.zeros(
                1, 2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
            )
            pool_ptrs[0, 0] = key
            if kv_cache.dtype == torch.float8_e4m3fn:
                scale_oq = torch.tensor(
                    [kv_scale_orig_quant], dtype=torch.float32, device=kv_cache.device
                )
                scale_qo = torch.tensor(
                    [kv_scale_quant_orig], dtype=torch.float32, device=kv_cache.device
                )
            else:
                scale_oq = None
                scale_qo = None
            self._layer_cache[key] = (pool_ptrs, scale_oq, scale_qo)

        pool_ptrs, scale_oq, scale_qo = self._layer_cache[key]
        quant_mode = int(QuantMode.FP8_KV_CACHE) if scale_oq is not None else 0
        return pool_ptrs, scale_oq, scale_qo, quant_mode

    def plan_host(
        self,
        num_prefill: int,
        num_decode: int,
        max_context_length: int,
        seq_len_with_cache_host: torch.Tensor,
        input_pos_host: torch.Tensor,
        seq_len_host: torch.Tensor,
        prompt_lens_host: Optional[torch.Tensor] = None,
    ) -> None:
        """Per-forward HOST metadata: pinned tensors for thop.attention.

        Called from ``prepare_trtllm_metadata_host`` before every forward
        (including CUDA graph replays).
        """
        num_seq = num_prefill + num_decode

        # host_request_types: 0 = prefill (context), 1 = decode (generation)
        with nvtx_range("ad_trtllm_plan_host_request_types"):
            self.host_request_types[:num_prefill].fill_(0)
            self.host_request_types[num_prefill:num_seq].fill_(1)

        # Use prompt_lens if available (original context length, constant across
        # iterations). Falls back to seq_len_host (correct for prefill, wrong for
        # extend/decode on subsequent iterations). Matches PyTorch backend's
        # prompt_lens which is set once per request and never changes.
        with nvtx_range("ad_trtllm_plan_host_context_lens"):
            context_lens = prompt_lens_host if prompt_lens_host is not None else seq_len_host

        # host_total_kv_lens: [context_total_kv, gen_total_kv]
        is_capturing = torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up()
        if is_capturing:
            with nvtx_range("ad_trtllm_plan_host_capturing"):
                self.host_total_kv_lens[0] = max_context_length * num_prefill
                self.host_total_kv_lens[1] = max_context_length * num_decode
                self.host_past_kv_lengths[:num_seq].fill_(max_context_length)
                self.host_context_lengths[:num_seq].fill_(max_context_length)
                self.context_lengths_gpu[:num_seq].fill_(max_context_length)
        else:
            with nvtx_range("ad_trtllm_plan_host_total_kv_lens"):
                self.host_total_kv_lens[0] = seq_len_with_cache_host[:num_prefill].sum()
                self.host_total_kv_lens[1] = seq_len_with_cache_host[num_prefill:num_seq].sum()
            with nvtx_range("ad_trtllm_plan_host_past_kv_lengths"):
                # Match PyTorch backend: host_past_kv_lengths = total KV cache length
                # after this forward (cached_token_lens + seq_lens_kv).
                self.host_past_kv_lengths[:num_seq] = seq_len_with_cache_host[:num_seq]
            with nvtx_range("ad_trtllm_plan_host_context_lengths"):
                self.host_context_lengths[:num_seq] = context_lens[:num_seq]
            with nvtx_range("ad_trtllm_plan_host_context_lengths_gpu"):
                self.context_lengths_gpu[:num_seq].copy_(context_lens[:num_seq], non_blocking=True)

    def update_host_request_types(self, batch_info: BatchInfo) -> None:
        """
        Refresh host_request_types from the current batch info.

        When running speculative decoding, we change batch_info
        to all-decode after target verification and before drafting.
        Since this is mid-forward, we need to refresh host_request_types before
        each kernel call.
        """
        num_seq = batch_info.get_total_num_sequences()
        num_prefill = batch_info.get_num_sequences()[0]
        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

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
_TRTLLM_ATTN_FP8_INPUT_SCALE_KEY = "trtllm_attention_input_scale"
_TRTLLM_ATTN_OUT_SCALE_KEY = "trtllm_attention_out_scale"


def set_trtllm_attention_fp8_input_scale(attn_node: Node, input_scale: Node) -> None:
    attn_node.meta[_TRTLLM_ATTN_FP8_INPUT_SCALE_KEY] = input_scale


def get_trtllm_attention_fp8_input_scale(attn_node: Node) -> Optional[Node]:
    scale = attn_node.meta.get(_TRTLLM_ATTN_FP8_INPUT_SCALE_KEY)
    return scale if isinstance(scale, Node) else None


def clear_trtllm_attention_fp8_input_scale(attn_node: Node) -> None:
    attn_node.meta.pop(_TRTLLM_ATTN_FP8_INPUT_SCALE_KEY, None)


@functools.cache
def _generate_spec_decoding_position_offsets(max_num_requests: int, draft_len: int) -> torch.Tensor:
    width = draft_len + 1
    row = torch.arange(width, dtype=torch.int, device="cuda")
    return row.unsqueeze(0).expand(max_num_requests, -1).contiguous()


@functools.cache
def _generate_spec_decoding_packed_mask(max_num_requests: int, draft_len: int) -> torch.Tensor:
    """Build the packed int32 causal mask expected by TRT-LLM spec-decoding.

    The last dimension stores consecutive 32-token chunks, with each int32
    encoding a prefix mask for positions in that chunk:
      1 -> 0b1, 3 -> 0b11, 7 -> 0b111, ...
    This matches the PyTorch backend helper and the kernel's input contract.
    """
    width = draft_len + 1
    num_blocks = math.ceil(width / 32)
    mask = torch.zeros([max_num_requests, width, num_blocks], dtype=torch.int, device="cuda")
    remaining = width
    for blk in range(num_blocks):
        if remaining <= 0:
            break
        n = min(32, remaining)
        vals = (torch.pow(2, torch.arange(n) + 1) - 1).int()
        mask[:, blk * 32 : blk * 32 + n, blk] = vals
        remaining -= 32
    return mask


# =============================================================================
# Host-side prepare function (runs outside CUDA graph, before every forward)
# =============================================================================


def prepare_trtllm_metadata_host(
    batch_info_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    prompt_lens_host: Optional[torch.Tensor] = None,
    spec_config=None,
) -> None:
    """Fill thop-specific HOST metadata (pinned tensors for thop.attention).

    Runs OUTSIDE the CUDA graph before every forward (including replays).
    Handles host_request_types, host_total_kv_lens, host_past_kv_lengths,
    host_context_lengths. Also performs one-time spec-dec tensor initialization
    when spec_config is provided and spec-dec hasn't been set up yet.
    """
    with nvtx_range("ad_trtllm_host_batch_info"):
        batch_info = BatchInfo(batch_info_host)
        num_prefill, num_extend, num_decode = batch_info.get_num_sequences()

    with nvtx_range("ad_trtllm_host_init_spec_decoding"):
        _GlobalTrtllmPlanner.init_spec_decoding(batch_info.get_max_batch_size(), spec_config)

    with nvtx_range("ad_trtllm_host_reset"):
        _GlobalTrtllmPlanner.reset(
            torch.device("cuda"),
            batch_info.get_max_batch_size(),
            batch_info.get_max_blocks_per_seq(),
        )

    # Extend (spec decoding) requests are expected as decode requests by the kernel.
    with nvtx_range("ad_trtllm_host_plan_host"):
        _GlobalTrtllmPlanner.plan_host(
            num_prefill=num_prefill,
            num_decode=num_decode + num_extend,
            max_context_length=batch_info.get_max_context_length(),
            seq_len_with_cache_host=seq_len_with_cache_host,
            input_pos_host=input_pos_host,
            seq_len_host=seq_len_host,
            prompt_lens_host=prompt_lens_host,
        )


# =============================================================================
# Device-side prepare function (inserted into the graph, analogous to
# prepare_flashinfer_metadata)
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_attention_prepare_metadata", mutates_args=())
def prepare_trtllm_metadata(
    batch_info_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute block_offsets for thop.attention (device-side, part of the traced graph).

    Uses the ``ragged_to_block_table_triton`` Triton kernel to scatter ``cache_loc``
    pages into the block_offsets table. All operations are pure GPU.

    Returns ``block_offsets`` which flows through the graph to each attention op,
    creating an explicit data dependency.
    """
    batch_info = BatchInfo(batch_info_host)
    block_offset_multiplier = batch_info.get_block_offset_multiplier()

    _GlobalTrtllmPlanner.plan_device(
        num_seq=batch_info.get_total_num_sequences(),
        block_offset_multiplier=block_offset_multiplier,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
    )

    return [_GlobalTrtllmPlanner.block_offsets]


@prepare_trtllm_metadata.register_fake
def prepare_trtllm_metadata_fake(
    batch_info_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> List[torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    batch_info = BatchInfo(batch_info_host)
    max_blocks_per_seq = batch_info.get_max_blocks_per_seq()
    max_batch_size = batch_info.get_max_batch_size()
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
    # EXTRA METADATA (from prepare_trtllm_metadata device-side op)
    kv_cache_block_offsets: torch.Tensor,
    # CACHE
    kv_cache: torch.Tensor,
    # CONSTANTS (only truly un-inferable values)
    scale: Optional[float],
    sliding_window: Optional[int] = None,
    kv_scale_orig_quant: float = 1.0,
    kv_scale_quant_orig: float = 1.0,
    out_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """TRT-LLM attention with paged KV cache for Auto-Deploy.

    Infers num_heads, num_kv_heads, head_dim, and tokens_per_block from tensor shapes.
    All max-size constants (max_num_requests, max_context_length) are read from
    ``batch_info_host`` via ``BatchInfo`` which is set once via
    ``SequenceInfo.update_cache_information()``.

    ``kv_cache_block_offsets`` is computed by the ``prepare_trtllm_metadata`` device-side
    op and flows through the graph to create an explicit data dependency.

    Note: layer_idx is always passed as 0 to thop.attention because
    the kv_cache tensor is already a strided view for the correct layer,
    pool_pointers encodes kv_cache.data_ptr() (layer-specific), and
    pool_mapping is all zeros. See module docstring for details.
    """
    # Get batch dimensions and model-level constants from host tensors (no device sync)
    batch_info = BatchInfo(batch_info_host)
    num_tokens = batch_info.get_total_num_tokens()
    max_context_length = batch_info.get_max_context_length()
    max_num_requests = batch_info.get_max_batch_size()
    _GlobalTrtllmPlanner.update_host_request_types(batch_info)

    # Infer dimensions from tensor shapes (bsnd layout)
    num_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[3]
    tokens_per_block = kv_cache.shape[3]  # HND: [blocks, 2, heads, tpb, head_dim]

    num_seq = batch_info.get_total_num_sequences()
    num_tokens = batch_info.get_total_num_tokens()
    max_context_length = batch_info.get_max_context_length()
    max_num_requests = batch_info.get_max_batch_size()
    # Use sliding_window for attention_window_size if provided, else full context length
    attention_window_size = (
        sliding_window
        if isinstance(sliding_window, int) and sliding_window > 0
        else max_context_length
    )

    # Per-layer tensors (pool pointers + optional FP8 scale tensors).
    # Cached on first call to avoid allocations during CUDA graph capture.
    host_kv_cache_pool_pointers, kv_scale_oq, kv_scale_qo, quant_mode = (
        _GlobalTrtllmPlanner.get_layer_tensors(kv_cache, kv_scale_orig_quant, kv_scale_quant_orig)
    )

    # Reshape Q, K, V to [num_tokens, num_heads * head_dim] and fuse.
    # Input is [bs, 1] (generate-only) or [1, total_seq_len] (prefill/mixed).
    # With piecewise CUDA graphs the tensor may be padded to a bucket size
    # (b*s > num_tokens), so flatten first and slice to the real token count.
    q_shape_og = q.shape
    q_flat = q.reshape(-1, num_heads * head_dim)[:num_tokens]
    k_flat = k.reshape(-1, num_kv_heads * head_dim)[:num_tokens]
    v_flat = v.reshape(-1, num_kv_heads * head_dim)[:num_tokens]
    qkv_fused = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()

    # Prepare output: if caller provided an `out` buffer, write directly into it
    total_padded_tokens = q_shape_og[0] * q_shape_og[1]
    # If out_scale is set, attention quantizes output to FP8.
    out_dtype = torch.float8_e4m3fn if out_scale is not None else q.dtype
    if out is not None:
        out_flat = out.view(-1, num_heads * head_dim)
        output = out_flat[:num_tokens]
    else:
        output = torch.zeros(
            total_padded_tokens, num_heads * head_dim, dtype=out_dtype, device=q.device
        )
    # Map SequenceInfo fields to thop.attention args
    sequence_length = seq_len_with_cache[:num_seq]  # device
    context_lengths = _GlobalTrtllmPlanner.context_lengths_gpu[:num_seq]

    host_request_types = _GlobalTrtllmPlanner.host_request_types[:num_seq]
    host_total_kv_lens = _GlobalTrtllmPlanner.host_total_kv_lens

    # Pool mapping (shared, always zeros since layer offset is in pool_pointers)
    host_kv_cache_pool_mapping = _GlobalTrtllmPlanner.host_pool_mapping

    # Pack parameters for thop.attention
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]

    # Spec-dec parameters for thop.attention.
    #
    # spec_decoding_bool_params:
    #   [0] is_spec_decoding_enabled: runtime supports spec-dec (True if spec-dec configured)
    #   [1] use_spec_decoding: use spec-dec THIS forward
    #   [2] is_spec_dec_tree: draft is a tree structure (False for linear Eagle chain)
    #
    # spec_decoding_tensor_params:
    #   [0] generation_lengths: [max_requests] tokens per request this step
    #   [1] position_offsets: [max_requests, draft_len+1] position offsets per token
    #   [2] packed_mask: [max_requests, draft_len+1, ceil((draft_len+1)/32)] causal mask

    num_extend = batch_info.get_num_sequences()[1]
    is_spec_decoding_enabled = _GlobalTrtllmPlanner.is_spec_decoding_enabled
    use_spec_decoding = num_extend > 0
    spec_decoding_bool_params = [is_spec_decoding_enabled, use_spec_decoding, False]
    spec_decoding_tensor_params = [
        _GlobalTrtllmPlanner.spec_decoding_generation_lengths,
        _GlobalTrtllmPlanner.spec_decoding_position_offsets,
        _GlobalTrtllmPlanner.spec_decoding_packed_mask,
    ]

    # Append Blackwell tree mask params on Blackwell+ SM only (>= 100, excluding 120/121).
    # Pre-Blackwell expects exactly 3 spec-dec tensor params.
    # Matches TrtllmAttentionWrapper.is_sm_version_trtllm_gen_kernel (trtllm.py:773).
    sm_version = get_sm_version()
    if not (sm_version < 100 or sm_version in (120, 121)):
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    thop.attention(
        qkv_fused,  # q (actually fused QKV)
        None,  # k (None when using fused QKV)
        None,  # v (None when using fused QKV)
        output[:num_tokens],  # output
        None,  # output_sf (NVFP4)
        _GlobalTrtllmPlanner.workspace,  # workspace (module-level, like flashinfer)
        sequence_length,  # sequence_length
        _GlobalTrtllmPlanner.host_past_kv_lengths[:num_seq],  # host_past_key_value_lengths
        host_total_kv_lens,  # host_total_kv_lens
        context_lengths,  # context_lengths
        _GlobalTrtllmPlanner.host_context_lengths[:num_seq],  # host_context_lengths
        host_request_types,  # host_request_types
        kv_cache_block_offsets,  # kv_cache_block_offsets
        host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
        host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
        None,  # cache_indirection (beam search)
        kv_scale_oq,  # kv_scale_orig_quant
        kv_scale_qo,  # kv_scale_quant_orig
        out_scale,  # out_scale
        None,  # rotary_inv_freq
        None,  # rotary_cos_sin
        None,  # latent_cache (MLA)
        None,  # q_pe (MLA)
        None,  # block_ids_per_seq
        None,  # attention_sinks
        True,  # is_fused_qkv
        True,  # update_kv_cache
        _GlobalTrtllmPlanner.predicted_tokens_per_seq,  # predicted_tokens_per_seq
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

    if out is not None:
        if total_padded_tokens > num_tokens:
            out_flat[num_tokens:].zero_()
        return out.new_empty(0)

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
    # EXTRA METADATA (from prepare_trtllm_metadata device-side op)
    kv_cache_block_offsets: torch.Tensor,
    # CACHE
    kv_cache: torch.Tensor,
    # CONSTANTS (only truly un-inferable values)
    scale: Optional[float],
    sliding_window: Optional[int] = None,
    kv_scale_orig_quant: float = 1.0,
    kv_scale_quant_orig: float = 1.0,
    out_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    if out is not None:
        return out.new_empty(0)
    out_dtype = torch.float8_e4m3fn if out_scale is not None else q.dtype
    return torch.empty_like(q.contiguous(), dtype=out_dtype)


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
    def prepare_node_for_cache_insertion(cls, gm: GraphModule, attn_node: Node) -> None:
        """Materialize optional out_scale node for FP8 output path if contract exists."""
        input_scale = get_trtllm_attention_fp8_input_scale(attn_node)
        if input_scale is None:
            attn_node.meta.pop(_TRTLLM_ATTN_OUT_SCALE_KEY, None)
            return

        existing_out_scale = attn_node.meta.get(_TRTLLM_ATTN_OUT_SCALE_KEY)
        if isinstance(existing_out_scale, Node):
            return

        with gm.graph.inserting_before(attn_node):
            out_scale = gm.graph.call_function(
                torch.ops.aten.reciprocal.default, args=(input_scale,)
            )
        attn_node.meta[_TRTLLM_ATTN_OUT_SCALE_KEY] = out_scale

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Extract constants from the source attention node.

        Returns scale, sliding_window, kv_scale_orig_quant, kv_scale_quant_orig,
        and optional output quant scale for FP8 linear consumers.
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

        # Optional out_scale is injected by prepare_node_for_cache_insertion when available.
        out_scale = source_attn_node.meta.get(_TRTLLM_ATTN_OUT_SCALE_KEY)
        if not isinstance(out_scale, Node):
            out_scale = None

        return [
            scale,
            sliding_window,
            1.0,  # kv_scale_orig_quant (hard-coded, same as FlashInfer)
            1.0,  # kv_scale_quant_orig (hard-coded, same as FlashInfer)
            out_scale,
        ]
