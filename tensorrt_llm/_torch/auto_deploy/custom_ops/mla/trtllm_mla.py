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

Mixed batches (both prefill and decode tokens present) dispatch the prefill and
decode paths back-to-back — matching the PT backend's ``forward_impl`` pattern.

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
from ....attention_backend.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    PositionEmbeddingType,
    RopeParams,
    RotaryScalingType,
)
from ....attention_backend.trtllm import TrtllmAttentionWrapper
from ...utils.cuda_graph import cuda_graph_state
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
# Helpers
# =============================================================================

# Fixed offset added to layer_idx for context (prefill) attention calls so
# they use separate C++ kernel state from decode calls on the same layer.
# Decode uses layer_idx ∈ [0, N-1], context uses [_CONTEXT_LAYER_OFFSET,
# _CONTEXT_LAYER_OFFSET + N - 1].  Both ranges map to the same physical KV
# cache layer via the pool mapping (modulo _CONTEXT_LAYER_OFFSET).
# Must be ≥ the maximum number of model layers (1000 covers all known models).
_CONTEXT_LAYER_OFFSET = 1000


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

        # Flash MLA metadata (required on SM90 for FP8 KV cache + MLA decode)
        self.flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None
        self.flash_mla_num_splits: Optional[torch.Tensor] = None

        # RoPE tables for direct thop.attention calls (created once, reused)
        self.rotary_inv_freq: Optional[torch.Tensor] = None
        self.rotary_cos_sin: Optional[torch.Tensor] = None
        self._rope_initialized: bool = False

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
        self._tokens_per_block: int = 0

        # Pre-allocated V-projection output buffer (allocated in ensure_decode_buffers)
        self.v_proj_output: Optional[torch.Tensor] = None

        # Prefill RoPE table (created on first prefill call if not fused-rope)
        self._prefill_rotary_cos_sin: Optional[torch.Tensor] = None
        self._prefill_rcs_max_pos: int = 0
        self._prefill_rcs_dim: int = 0

        # Per-layer TrtllmAttentionWrapper instances (created on first use)
        self._attn_wrappers: dict = {}

    def get_or_create_wrapper(
        self,
        layer_idx: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        *,
        for_context: bool = False,
    ) -> TrtllmAttentionWrapper:
        """Return a cached TrtllmAttentionWrapper for this layer, creating if needed.

        The PT backend creates TWO wrappers per layer for MLA:
        - **context** (self.mha): head_size=qk_head_dim, num_kv_heads=num_heads, v_head_dim=v_head_dim
        - **decode** (self.mqa): head_size=kv_lora_rank+qk_rope_head_dim, num_kv_heads=1, v_head_dim=kv_lora_rank

        Args:
            for_context: If True, return context wrapper; if False, decode wrapper.
        """
        key = (layer_idx, for_context)
        w = self._attn_wrappers.get(key)
        if w is not None:
            return w

        # Build RopeParams with YaRN if available (matching standard backend).
        max_pos = getattr(self, "yarn_max_positions", 8192)
        orig_max_pos = getattr(self, "yarn_original_max_positions", 4096)
        factor = getattr(self, "yarn_factor", 1.0)
        rope_params = RopeParams(
            dim=qk_rope_head_dim,
            theta=10000.0,
            max_positions=min(max_pos, 16384),
            original_max_positions=orig_max_pos,
        )
        if factor > 1.0:
            rope_params.scale_type = RotaryScalingType.yarn
            rope_params.scale = factor
            short_m = getattr(self, "yarn_short_m_scale", 1.0)
            long_m = getattr(self, "yarn_long_m_scale", 1.0)
            rope_params.short_m_scale = short_m
            rope_params.long_m_scale = long_m
            rope_params.mscale = short_m
            rope_params.mscale_all_dim = long_m
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=rope_params,
            is_neox=False,
        )
        # q_scaling = 1/(mscale^2) matching standard backend (attention.py L1336)
        q_lora = getattr(self, "q_lora_rank", 0)
        q_scaling = 1.0
        if factor > 1.0:
            mscale_all_dim = getattr(self, "yarn_long_m_scale", 1.0)
            q_scaling = 1.0 / (mscale_all_dim * mscale_all_dim) if mscale_all_dim > 0 else 1.0

        if for_context:
            mla_params = MLAParams(
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=v_head_dim,
            )
            w = TrtllmAttentionWrapper(
                num_heads=num_heads,
                head_size=qk_nope_head_dim + qk_rope_head_dim,
                num_kv_heads=num_heads,
                pos_embd_params=pos_embd_params,
                q_scaling=q_scaling,
                mla_params=mla_params,
            )
        else:
            mla_params = MLAParams(
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=kv_lora_rank,
            )
            w = TrtllmAttentionWrapper(
                num_heads=num_heads,
                head_size=kv_lora_rank + qk_rope_head_dim,
                num_kv_heads=1,
                pos_embd_params=pos_embd_params,
                q_scaling=q_scaling,
                mla_params=mla_params,
            )

        w.update_quant_config()
        self._attn_wrappers[key] = w
        return w

    def _init_ctx_workspace(self, device: torch.device) -> torch.Tensor:
        """Create a separate workspace for context attention (not shared with decode)."""
        self._ctx_workspace = torch.empty(0, dtype=torch.int8, device=device)
        return self._ctx_workspace

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

        # Pre-allocate workspace large enough to avoid cudaMalloc during
        # CUDA graph capture (matching the standard trtllm_attention backend).
        self.workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        # Shape: [pool_mapping_size, 2] — maps [layer_idx] to [pool_idx, layer_within_pool].
        # All zeros: each layer's pool pointer already points to that layer's
        # cache tensor, so layer_within_pool must be 0 (no additional offset).
        # This matches the standard trtllm_attention backend convention.
        pool_mapping_size = _CONTEXT_LAYER_OFFSET * 2
        self.host_pool_mapping = torch.zeros(
            pool_mapping_size, 2, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
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

        # Flash MLA: allocate metadata buffers for SM90 + head_dim=576.
        # The standard FMHA doesn't support FP8 KV cache + MLA on SM90;
        # Flash MLA provides this capability.
        if torch.cuda.get_device_capability() == (9, 0):
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            self.flash_mla_tile_scheduler_metadata = torch.zeros(
                sm_count * 8, dtype=torch.int32, device=device
            )
            self.flash_mla_num_splits = torch.zeros(max_batch + 1, dtype=torch.int32, device=device)

    def ensure_decode_buffers(
        self,
        device: torch.device,
        max_tokens: int,
        num_heads: int,
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

        # FP8 generation MLA buffers: quantized Q, BMM scales.
        # The mla_rope_generation kernel fills these; thop.attention reads them.
        # Allocated unconditionally (small cost) so CUDA graph addresses are stable.
        self.mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        self.mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        self.quant_q_buffer = torch.empty(
            max_tokens, num_heads, gen_head_size, dtype=torch.uint8, device=device
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
        # Ensure pool_mapping has enough rows for block_offset_multiplier
        if self.host_pool_mapping.size(0) < block_offset_multiplier:
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

    def ensure_rope_tables(self, qk_rope_head_dim: int):
        """Create RoPE inv_freq + cos_sin tables (once, reused across calls).

        YaRN scaling parameters default to 1.0 / model ``max_position_embeddings``.
        The fused-RoPE transform precomputes the cos/sin table that the kernel
        actually consumes, so these defaults are fine for the fall-back path.
        """
        if self._rope_initialized:
            return
        # Cap max_positions to avoid huge table allocations. The kernel only
        # needs positions up to the actual sequence length.
        max_pos = min(getattr(self, "yarn_max_positions", 8192), 16384)
        orig_max_pos = getattr(self, "yarn_original_max_positions", 4096)
        factor = getattr(self, "yarn_factor", 1.0)
        short_m = getattr(self, "yarn_short_m_scale", 1.0)
        long_m = getattr(self, "yarn_long_m_scale", 1.0)

        rope = RopeParams(
            dim=qk_rope_head_dim,
            theta=10000.0,
            max_positions=max_pos,
            original_max_positions=orig_max_pos,
        )
        if factor > 1.0:
            rope.scale_type = RotaryScalingType.yarn
            rope.scale = factor
            rope.short_m_scale = short_m
            rope.long_m_scale = long_m
            rope.mscale = short_m
            rope.mscale_all_dim = long_m
        self.rotary_inv_freq, self.rotary_cos_sin = rope.create_rope_const_params()
        self._rope_initialized = True

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
    seq_len_with_cache_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_host: torch.Tensor,
) -> None:
    """Fill thop-specific HOST metadata (pinned tensors for thop.attention).

    Runs OUTSIDE the CUDA graph before every forward (including replays).
    Mirrors ``prepare_trtllm_metadata_host`` from the standard trtllm backend.
    """
    batch_info = BatchInfo(batch_info_host)
    num_prefill, _, num_decode = batch_info.get_absorbed_info()
    max_context_length = batch_info.get_max_context_length()
    max_blocks_per_seq = batch_info.get_max_blocks_per_seq()
    max_batch_size = batch_info.get_max_batch_size()

    planner = _GlobalTrtllmMLAPlanner
    planner.reset(torch.device("cuda"), max_batch_size, max_blocks_per_seq)

    planner.plan_host(
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
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute block_offsets and block_ids_per_seq for MLA (device-side, part of graph).

    Uses ``ragged_to_block_table_triton`` to scatter ``cache_loc`` pages into
    ``block_ids_per_seq`` on the GPU, then derives ``block_offsets``.
    Returns ``[block_offsets]`` which flows through the graph to each attention op.
    """
    batch_info = BatchInfo(batch_info_host)
    num_prefill, _, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    block_offset_multiplier = batch_info.get_block_offset_multiplier()

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
    layer_idx: int = 0,
    scale: float = 1.0,
    fused_rope: bool = False,
    rotary_cos_sin: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched prefill via C++ FMHA.

    Expands compressed_kv via kv_b_proj into separate K/V, assembles the
    Q/K/V tensors, and calls thop.attention with attention_input_type=
    context_only so the kernel runs FMHA and writes latent_cache to the
    paged KV cache (with RoPE applied by mla_rope_context when
    rotary_cos_sin carries the model's table, or a no-op identity table
    when q_pe/kpe already have RoPE baked in).
    """
    dtype = q_nope_flat.dtype
    device = q_nope_flat.device
    planner = _GlobalTrtllmMLAPlanner

    # Final output: [num_tokens, num_heads * v_head_dim]
    output = torch.empty(num_tokens, num_heads * v_head_dim, dtype=dtype, device=device)

    # Expand compressed KV via kv_b_proj to get separate K, V for FMHA.
    #
    # The FP8 MLA context FMHA quant kernel (invokeMLAContextFp8Quantize in
    # mlaKernels.cu) reads V with token_stride = N*(qk_nope + v) and
    # head_stride = V_HEAD_DIM, i.e. it assumes a "grouped-by-dim" memory
    # layout where all heads' nope are packed first, then all heads' v.
    # The PT backend builds kv_b_proj in that layout at checkpoint load
    # (modeling_deepseekv3.py: cat(k_nope_weight.reshape(N*nope, ...),
    # v_weight.reshape(N*v, ...)) along dim=0).  HF stores it per-head
    # (head0_nope | head0_v | head1_nope | ...), so a straight linear +
    # 2D split gives scrambled data (first N*nope columns contain nope+v
    # of the first N/2 heads).  Permute the weight here (cached per-layer
    # on the planner) so the downstream split + kernel reads match.
    w_src = (
        kv_b_proj_weight.to(dtype)
        if kv_b_proj_weight.dtype == torch.float8_e4m3fn
        else kv_b_proj_weight
    )
    perm_cache_key = (kv_b_proj_weight.data_ptr(), dtype)
    perm_cache = getattr(planner, "_kv_b_proj_grouped_cache", None)
    if perm_cache is None:
        perm_cache = {}
        planner._kv_b_proj_grouped_cache = perm_cache
    w_grouped = perm_cache.get(perm_cache_key)
    if w_grouped is None:
        w_reshaped = w_src.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
        w_nope = w_reshaped[:, :qk_nope_head_dim, :].reshape(
            num_heads * qk_nope_head_dim, kv_lora_rank
        )
        w_v = w_reshaped[:, qk_nope_head_dim:, :].reshape(num_heads * v_head_dim, kv_lora_rank)
        w_grouped = torch.cat([w_nope, w_v], dim=0).contiguous()
        perm_cache[perm_cache_key] = w_grouped
    kv = torch.nn.functional.linear(compressed_kv_flat, w_grouped)
    # kv layout: [T, N*qk_nope | N*v] (grouped-by-dim). Split on last dim.
    # V stays a non-contiguous view with token stride = N*(qk_nope+v) to
    # match the FMHA kernel's stride assumption (see comment above).
    kv_2d = kv.view(num_tokens, num_heads * (qk_nope_head_dim + v_head_dim))
    k_nope_2d, v = kv_2d.split([num_heads * qk_nope_head_dim, num_heads * v_head_dim], dim=-1)
    k_nope = k_nope_2d.view(num_tokens, num_heads, qk_nope_head_dim)

    # -- Let C++ handle RoPE, FP8 cache write, and FP8 FMHA --
    num_prefill = int((host_request_types == 0).sum().item())
    pf = num_prefill

    kpe_expanded = kpe_flat.view(num_tokens, 1, qk_rope_head_dim).expand(-1, num_heads, -1)
    k = torch.cat([k_nope, kpe_expanded], dim=-1).view(num_tokens, num_heads * qk_head_dim)
    q = torch.cat([q_nope_flat, q_pe_flat], dim=-1).view(num_tokens, num_heads * qk_head_dim)

    planner.ensure_rope_tables(qk_rope_head_dim)

    # When fused_rope=False, q_pe and kpe already have RoPE from the model.
    # Use identity cos_sin (cos=1, sin=0) so invokeMLARopeContext doesn't
    # re-apply RoPE (which would double-apply, corrupting decode).
    # When fused_rope=True, use the model's rotary_cos_sin (passed in via
    # the custom op parameter, computed by fuse_rope_into_trtllm_mla from
    # the model's actual RoPE buffer). CRITICAL: the same table must be
    # used for prefill cache writes and decode RoPE — otherwise prefill
    # writes RoPE'd kpe with table A and decode computes Q's RoPE with
    # table B, producing wrong attention scores.
    if not fused_rope:
        if not hasattr(planner, "_identity_cos_sin"):
            identity = torch.zeros_like(planner.rotary_cos_sin)
            identity.view(-1, qk_rope_head_dim, 2)[:, :, 0] = 1.0
            planner._identity_cos_sin = identity
        thop_cos_sin = planner._identity_cos_sin
    else:
        thop_cos_sin = rotary_cos_sin if rotary_cos_sin is not None else planner.rotary_cos_sin

    # The MLA context FMHA uses FP8 quantized Q/K/V when quant_mode has
    # FP8 bits (mFP8ContextMLA=true). This is the only supported path for
    # MLA context on SM100+ — the BF16 FMHA path doesn't have cubin
    # support for MLA's different Q/K (192) and V (128) head sizes.
    # The FP8 quantization causes ~15% GSM8K accuracy loss vs BF16
    # attention, inherent to the FP8 context MLA precision (the PT backend
    # shows a similar MMLU gap: 84.7% vs ~94% with BF16).
    quant_mode = 0
    cache_is_fp8 = kv_cache is not None and kv_cache.dtype == torch.float8_e4m3fn
    if cache_is_fp8:
        quant_mode = int(QuantMode.FP8_1x128_128x128) | int(QuantMode.FP8_KV_CACHE)
        if planner.kv_scale_orig_quant is None:
            planner.kv_scale_orig_quant = torch.tensor(
                [1.0], dtype=torch.float32, device=kv_b_proj_weight.device
            )
            planner.kv_scale_quant_orig = torch.tensor(
                [1.0], dtype=torch.float32, device=kv_b_proj_weight.device
            )

    # Use kv_cache_block_offsets (device-filled by prepare_trtllm_mla_metadata)
    # so prefill writes land on the same blocks that decode later reads from.
    ctx_block_offsets = kv_cache_block_offsets

    # host_total_kv_lens for the context-only call: sum of NEW tokens per
    # prefill seq (matches what the fresh-prefill kernel path expects).
    ctx_total_kv_lens = planner.host_total_kv_lens.clone()
    ctx_total_kv_lens[0] = host_context_lengths[:pf].sum().item() if pf > 0 else 0
    ctx_total_kv_lens[1] = 0  # context-only call touches no decode tokens

    # Treat every prefill seq as fresh (past_kv=0, sequence_length=new tokens).
    # When AD's scheduler does chunked prefill or cache reuse, the host metadata
    # carries past_kv>0 and sequence_length includes those cached tokens.  The
    # thop context FMHA's cached-KV path produces garbage output under that
    # config (output magnitude blows up by ~10×num_decode); forcing the fresh
    # semantics here (attention over new tokens only) recovers accuracy.
    past_kv_zeroed = torch.zeros_like(host_past_kv_lengths)
    new_token_seq_len = context_lengths

    sm_version = get_sm_version()
    rotary_embedding_scales = [
        getattr(_GlobalTrtllmMLAPlanner, "yarn_factor", 1.0),
        getattr(_GlobalTrtllmMLAPlanner, "yarn_short_m_scale", 1.0),
        getattr(_GlobalTrtllmMLAPlanner, "yarn_long_m_scale", 1.0),
    ]
    rotary_embedding_max_position_info = [
        getattr(_GlobalTrtllmMLAPlanner, "yarn_max_positions", max_context_length),
        getattr(_GlobalTrtllmMLAPlanner, "yarn_original_max_positions", max_context_length),
    ]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]
    if sm_version >= 89:
        spec_decoding_tensor_params.extend([None, None, None])
    mla_tensor_params = [None, None]

    # NOTE: Do NOT clone V here. The V tensor must be a non-contiguous view
    # from kv.split() with token stride = numHeads * (qk_nope + v_head_dim).
    # The C++ FP8 quantize kernel and FMHA kernel read V using this stride.
    # Cloning would make V contiguous with the wrong stride, causing
    # out-of-bounds reads and wrong output on layers 1+.

    thop.attention(
        q,  # q
        k,  # k
        v,  # v
        output,  # output
        None,  # output_sf
        planner.workspace,  # workspace
        new_token_seq_len[:pf],  # sequence_length (new tokens only)
        past_kv_zeroed[:pf],  # host_past_key_value_lengths (zeroed)
        ctx_total_kv_lens.int(),  # host_total_kv_lens
        context_lengths[:pf],  # context_lengths
        host_context_lengths[:pf],  # host_context_lengths
        host_request_types[:pf],  # host_request_types
        ctx_block_offsets,  # kv_cache_block_offsets
        host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
        planner.host_pool_mapping,  # host_kv_cache_pool_mapping
        None,  # cache_indirection
        planner.kv_scale_orig_quant,  # kv_scale_orig_quant
        planner.kv_scale_quant_orig,  # kv_scale_quant_orig
        None,  # out_scale
        planner.rotary_inv_freq,  # rotary_inv_freq
        thop_cos_sin,  # rotary_cos_sin (identity when no_fuse_rope)
        latent_cache,  # latent_cache — C++ handles RoPE + cache write
        None,  # q_pe
        planner.block_ids_per_seq,  # block_ids_per_seq
        None,  # attention_sinks
        False,  # is_fused_qkv
        True,  # update_kv_cache
        1,  # predicted_tokens_per_seq
        layer_idx + _CONTEXT_LAYER_OFFSET,  # layer_idx
        num_heads,  # num_heads
        num_heads,  # num_kv_heads (context: expanded)
        qk_nope_head_dim + qk_rope_head_dim,  # head_size (192)
        tokens_per_block,  # tokens_per_block
        max_num_requests,  # max_num_requests
        max_context_length,  # max_context_length
        max_context_length,  # attention_window_size
        0,  # sink_token_length
        1,  # beam_width
        int(AttentionMaskType.causal),  # mask_type
        quant_mode,  # quant_mode
        1.0 / (scale * math.sqrt(qk_nope_head_dim + qk_rope_head_dim)),  # q_scaling
        int(PositionEmbeddingType.yarn),  # position_embedding_type
        qk_rope_head_dim,  # rotary_embedding_dim
        10000.0,  # rotary_embedding_base
        5,  # rotary_embedding_scale_type (YaRN)
        rotary_embedding_scales,  # rotary_embedding_scales
        rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
        True,  # use_paged_context_fmha
        int(AttentionInputType.context_only),  # attention_input_type
        True,  # is_mla_enable
        1,  # chunked_prefill_buffer_batch_size
        getattr(_GlobalTrtllmMLAPlanner, "q_lora_rank", 0),  # q_lora_rank
        kv_lora_rank,  # kv_lora_rank
        qk_nope_head_dim,  # qk_nope_head_dim
        qk_rope_head_dim,  # qk_rope_head_dim
        v_head_dim,  # v_head_dim
        None,  # mrope_rotary_cos_sin
        None,  # mrope_position_deltas
        mla_tensor_params,  # helix_tensor_params
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
        None,  # flash_mla_tile_scheduler_metadata
        None,  # flash_mla_num_splits
        pf,  # num_contexts
        num_tokens,  # num_ctx_tokens
    )

    return output


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
    # Table layout: [max_pos, D, 2] where D=qk_rope_head_dim (NeoX-doubled).
    # Reshape with full D so each position is one row, then take first D/2
    # frequencies (second D/2 are NeoX duplicates).
    table = rotary_cos_sin.view(-1, qk_rope_head_dim, 2)  # [max_pos, D, 2]
    # Keep cos/sin in float32 to match the C++ mla_rope_generation kernel
    # and FlashInfer, which both compute rotation in float32.  Casting to
    # BF16 here loses ~3 decimal digits that accumulate across 61 layers.
    cos_half = table[positions.long(), :half, 0]  # [T, D/2] float32
    sin_half = table[positions.long(), :half, 1]  # [T, D/2] float32

    def _rotate_interleaved(x, cos_h, sin_h):
        # Compute in float32 (matching C++ kernel), cast back to input dtype.
        dtype = x.dtype
        x_f = x.float()
        pairs = x_f.unflatten(-1, (-1, 2))  # [..., D/2, 2]
        even, odd = pairs[..., 0], pairs[..., 1]
        r_even = even * cos_h - odd * sin_h
        r_odd = even * sin_h + odd * cos_h
        return torch.stack([r_even, r_odd], dim=-1).flatten(-2).to(dtype)

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
    v_head_dim: int,
    kv_lora_rank: int,
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
    layer_idx: int = 0,
) -> torch.Tensor:
    """Handle decode: weight absorption + latent-space attention + output projection.

    Follows the PyTorch backend pattern: ``bmm_out`` writes the Q absorption
    result directly into a pre-allocated ``fused_q`` slice (no intermediate
    tensor, no ``torch.cat``), and the V projection also uses ``bmm_out`` into
    a pre-allocated buffer.

    When ``rotary_cos_sin`` is provided, uses ``mla_rope_generation`` to fuse
    cache write + RoPE + q_pe copy + scheduler fill into one kernel.  Otherwise
    copies q_pe manually and zeros the scheduler counter.
    """
    planner = _GlobalTrtllmMLAPlanner
    gen_head_size = kv_lora_rank + qk_rope_head_dim

    # Cast FP8 weights to compute dtype for BMM (FP8 checkpoints store kv_b_proj in FP8).
    # Cache the casted/reshaped weights so CUDA graph replay uses stable addresses.
    ptr = kv_b_proj_weight.data_ptr()
    cached = planner._per_layer_pool_ptrs.get(("w_kn_v_t", ptr))
    if cached is not None:
        w_kn, w_v_t = cached
    else:
        if kv_b_proj_weight.dtype == torch.float8_e4m3fn:
            kv_b_proj_weight = kv_b_proj_weight.to(q_nope_flat.dtype)
        weight_reshaped = kv_b_proj_weight.view(
            num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )
        w_kn = weight_reshaped[:, :qk_nope_head_dim, :].contiguous()
        w_v_t = weight_reshaped[:, qk_nope_head_dim:, :].transpose(1, 2).contiguous()
        planner._per_layer_pool_ptrs[("w_kn_v_t", ptr)] = (w_kn, w_v_t)

    # NOTE: Context wrapper warmup removed — now using direct thop.attention()
    # calls which don't need wrapper-based warmup. RoPE tables and workspace
    # are managed by the planner.
    # Ensure decode wrapper exists (needed for warmup plan/run above)
    planner.get_or_create_wrapper(
        layer_idx,
        num_heads,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    )

    # Flash MLA metadata (required on SM90 for FP8 KV cache + MLA decode).
    flash_mla_meta = None
    flash_mla_splits = None
    if planner.flash_mla_tile_scheduler_metadata is not None:
        decode_kv_lens = sequence_length[num_prefill:]
        flash_mla_splits = planner.flash_mla_num_splits[: num_tokens + 1]
        thop.compute_flash_mla_metadata(
            decode_kv_lens,
            planner.flash_mla_tile_scheduler_metadata,
            flash_mla_splits,
            num_tokens,
            1,  # s_q (1 token per decode request)
            num_heads,
            1,  # num_kv_heads
            kv_lora_rank,
        )
        flash_mla_meta = planner.flash_mla_tile_scheduler_metadata

    fused_q_flat = planner.fused_q_flat[:num_tokens]
    fused_q_view = fused_q_flat.view(num_tokens, num_heads, gen_head_size)

    # BMM: [num_heads, num_tokens, qk_nope_head_dim] @ [num_heads, qk_nope_head_dim, kv_lora_rank]
    q_absorbed = torch.bmm(q_nope_flat.transpose(0, 1), w_kn)
    fused_q_view[:, :, :kv_lora_rank] = q_absorbed.transpose(0, 1)

    cu_q = planner.cu_q_decode[: num_tokens + 1]
    cu_kv = planner.cu_kv_decode[: num_tokens + 1]

    # FP8 generation MLA: pass quantized Q buffer and BMM scales when FP8 KV
    # cache is active.  The mla_rope_generation kernel fills these tensors;
    # thop.attention reads them in the XQA/FMHA decode path.
    fp8_kv = QuantMode(quant_mode).has_fp8_kv_cache()
    mla_bmm1 = planner.mla_bmm1_scale if fp8_kv else None
    mla_bmm2 = planner.mla_bmm2_scale if fp8_kv else None
    quant_q = planner.quant_q_buffer[:num_tokens] if fp8_kv else None

    # mla_rope_generation handles: cache write (with FP8 block quantization),
    # RoPE on q_pe, cu_q/cu_kv fill, scheduler counter, and FP8 BMM scales.
    # Always called — for no-fuse mode, rotary_cos_sin is an identity table
    # (cos=1, sin=0) so RoPE is a no-op on pre-RoPE'd q_pe/kpe.
    torch.ops.trtllm.mla_rope_generation(
        fused_q_view,
        q_pe_flat,
        latent_cache,
        rotary_cos_sin,
        cu_q,
        cu_kv,
        planner.fmha_scheduler_counter_decode,
        mla_bmm1,
        mla_bmm2,
        quant_q,
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
        layer_idx,  # layer_idx
        num_heads,
        num_kv_heads,
        gen_head_size,
        tokens_per_block,
        max_context_length,  # attention_window_size
        0,  # sink_token_length
        1,  # beam_width
        quant_mode,
        # q_scaling — must match the q_scaling passed to thop.attention because
        # mla_rope_generation (dsv3RopeOp.cpp:218) computes
        #   host_bmm1_scale = 1 / (q_scaling * sqrt(qk_head_dim))
        # and writes it into mla_bmm1_scale, which the decode FMHA uses as its
        # softmax scale.  Hardcoding 1.0 here drops mscale^2 from the YaRN
        # softmax, halving the attention scale on DeepSeek R1 (mscale≈1.37).
        scale,
        0,  # q_lora_rank
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,  # v_head_dim (in latent space = kv_lora_rank)
    )

    output_latent = planner.output_latent[:num_tokens]

    # Call thop.attention directly (CG-safe, no wrapper.plan()/run()).
    # Follows the same pattern as trtllm_attention.py:424-503.
    planner.ensure_rope_tables(qk_rope_head_dim)

    sm_version = get_sm_version()
    rotary_embedding_scales = [
        getattr(_GlobalTrtllmMLAPlanner, "yarn_factor", 1.0),
        getattr(_GlobalTrtllmMLAPlanner, "yarn_short_m_scale", 1.0),
        getattr(_GlobalTrtllmMLAPlanner, "yarn_long_m_scale", 1.0),
    ]
    rotary_embedding_max_position_info = [
        getattr(_GlobalTrtllmMLAPlanner, "yarn_max_positions", max_context_length),
        getattr(_GlobalTrtllmMLAPlanner, "yarn_original_max_positions", max_context_length),
    ]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]
    if sm_version >= 89:
        spec_decoding_tensor_params.extend([None, None, None])
    mla_tensor_params = [None, None]

    thop.attention(
        fused_q_flat,  # q (fused Q for decode)
        None,  # k
        None,  # v
        output_latent,  # output
        None,  # output_sf
        planner.workspace,  # workspace
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
        planner.kv_scale_orig_quant,  # kv_scale_orig_quant
        planner.kv_scale_quant_orig,  # kv_scale_quant_orig
        None,  # out_scale
        planner.rotary_inv_freq,  # rotary_inv_freq
        planner.rotary_cos_sin,  # rotary_cos_sin
        latent_cache,  # latent_cache
        q_pe_flat,  # q_pe
        planner.block_ids_per_seq,  # block_ids_per_seq
        None,  # attention_sinks
        True,  # is_fused_qkv
        True,  # update_kv_cache
        1,  # predicted_tokens_per_seq
        layer_idx,  # layer_idx
        num_heads,  # num_heads
        num_kv_heads,  # num_kv_heads
        gen_head_size,  # head_size
        tokens_per_block,  # tokens_per_block
        max_num_requests,  # max_num_requests
        max_context_length,  # max_context_length
        max_context_length,  # attention_window_size
        0,  # sink_token_length
        1,  # beam_width
        int(AttentionMaskType.causal),  # mask_type
        quant_mode,  # quant_mode
        scale,  # q_scaling
        int(PositionEmbeddingType.yarn),  # position_embedding_type
        qk_rope_head_dim,  # rotary_embedding_dim
        10000.0,  # rotary_embedding_base
        5,  # rotary_embedding_scale_type (YaRN)
        rotary_embedding_scales,  # rotary_embedding_scales
        rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
        False,  # use_paged_context_fmha
        int(AttentionInputType.generation_only),  # attention_input_type
        True,  # is_mla_enable
        1,  # chunked_prefill_buffer_batch_size
        0,  # q_lora_rank
        kv_lora_rank,  # kv_lora_rank
        qk_nope_head_dim,  # qk_nope_head_dim
        qk_rope_head_dim,  # qk_rope_head_dim
        kv_lora_rank,  # v_head_dim (latent space)
        None,  # mrope_rotary_cos_sin
        None,  # mrope_position_deltas
        mla_tensor_params,  # helix_tensor_params
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
        cu_q,  # cu_q_seqlens
        cu_kv,  # cu_kv_seqlens
        planner.fmha_scheduler_counter_decode,  # fmha_scheduler_counter
        mla_bmm1,  # mla_bmm1_scale
        mla_bmm2,  # mla_bmm2_scale
        quant_q,  # quant_q_buffer
        flash_mla_meta,  # flash_mla_tile_scheduler_metadata
        flash_mla_splits,  # flash_mla_num_splits
        num_prefill,  # num_contexts (leading context entries in the full batch)
        0,  # num_ctx_tokens (ignored when attention_input_type=generation_only)
    )

    output_reshaped = output_latent.view(num_tokens, num_heads, kv_lora_rank)
    v_proj_result = torch.bmm(output_reshaped.transpose(0, 1), w_v_t)
    return v_proj_result.transpose(0, 1).reshape(num_tokens, num_heads * v_head_dim)


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
    kv_cache_block_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
    rotary_cos_sin: Optional[torch.Tensor] = None,
    layer_idx: int = 0,
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

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode
    max_context_length = batch_info.get_max_context_length()
    max_num_requests = batch_info.get_max_batch_size()

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    num_kv_heads = 1
    tokens_per_block = kv_cache.shape[3]  # HND: [blocks, kv_factor, heads, tpb, head_dim]

    planner = _GlobalTrtllmMLAPlanner
    if planner._tokens_per_block != tokens_per_block:
        planner._tokens_per_block = tokens_per_block
    host_kv_cache_pool_pointers = planner.get_pool_pointers_for_layer(kv_cache)

    quant_mode = 0
    cache_is_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    if cache_is_fp8:
        # FP8_1x128_128x128 enables block-scaled FP8 quantization.
        # FP8_KV_CACHE tells the C++ kernels that the KV cache stores FP8
        # elements so they compute correct memory offsets and select the right
        # code paths (XQA MLA on SM120, flash-MLA on SM90, etc.).
        quant_mode = int(QuantMode.FP8_1x128_128x128) | int(QuantMode.FP8_KV_CACHE)
        if planner.kv_scale_orig_quant is None:
            planner.kv_scale_orig_quant = torch.tensor(
                [1.0], dtype=torch.float32, device=q_nope.device
            )
            planner.kv_scale_quant_orig = torch.tensor(
                [1.0], dtype=torch.float32, device=q_nope.device
            )
    # Set quant_mode on all wrappers unconditionally.
    if quant_mode != 0:
        for _key, w in planner._attn_wrappers.items():
            w.quant_mode = quant_mode

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
            v_head_dim,
            kv_lora_rank,
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
        """Run prefill via C++ FMHA with cache writes."""
        return _prefill_with_cache_write_thop(
            q_n,
            q_p,
            ckv,
            kpe_pre,
            lc,
            n_tok,
            n_pf,
        )

    def _prefill_with_cache_write_thop(q_n, q_p, ckv, kpe_pre, lc, n_tok, n_pf):
        """Run prefill via C++ FMHA with Python-side RoPE and cache writes."""
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
            layer_idx=layer_idx,
            scale=scale,
            fused_rope=fused_rope,
            rotary_cos_sin=rotary_cos_sin,
            kv_cache=kv_cache,
        )

    # -- Decode helper --------------------------------------------------------

    def _do_decode(q_n, q_p, lc, n_tok, n_pf):
        """Run decode attention via mla_rope_generation + thop.attention."""
        return _handle_decode_impl(
            q_n,
            q_p,
            kv_b_proj_weight,
            lc,
            kv_cache,
            n_tok,
            n_pf,
            *_make_shared_metadata(),
            rotary_cos_sin=decode_rotary_cos_sin,
            layer_idx=layer_idx,
        )

    # -- Decode RoPE table ---------------------------------------------------
    # mla_rope_generation must ALWAYS be called for decode because it fills
    # critical buffers: cu_q, cu_kv, fmha_scheduler_counter, and (for FP8)
    # mla_bmm1_scale, mla_bmm2_scale, quant_q_buffer.  It also writes
    # latent_cache to the paged KV cache with proper FP8 block quantization.
    #
    # When fused_rope=True, pass the actual cos_sin table (RoPE applied by kernel).
    # When fused_rope=False, q_pe/kpe already have RoPE from the model, so pass
    # an identity table (cos=1, sin=0) to make the kernel's RoPE a no-op.
    planner.ensure_rope_tables(qk_rope_head_dim)
    if fused_rope:
        decode_rotary_cos_sin = rotary_cos_sin
    else:
        if not hasattr(planner, "_identity_cos_sin"):
            identity = torch.zeros_like(planner.rotary_cos_sin)
            identity.view(-1, qk_rope_head_dim, 2)[:, :, 0] = 1.0
            planner._identity_cos_sin = identity
        decode_rotary_cos_sin = planner._identity_cos_sin

    # -- Dispatch: prefill and/or decode ----------------------------------------
    # Allocate output with bs rows (may include CG padding); only real-token
    # positions are filled — padding stays zero, matching the FI MLA pattern.
    y = torch.zeros(bs, num_heads * v_head_dim, dtype=q_nope_flat.dtype, device=q_nope_flat.device)

    if num_prefill > 0 and num_decode > 0:
        # Mixed batch: run thop.attention prefill + decode back-to-back,
        # matching the PT backend's forward_impl pattern.
        # Note: we pass the FULL latent_cache (not sliced to prefill tokens)
        # because thop internally indexes it by the full batch extent.
        y[:num_prefill_tokens] = _prefill_with_cache_write(
            q_nope_flat[:num_prefill_tokens],
            q_pe_flat[:num_prefill_tokens],
            compressed_kv_flat[:num_prefill_tokens],
            kpe_flat[:num_prefill_tokens],
            latent_cache,
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
    # Standard metadata (3 args — matches trtllm attention pattern)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    # Extra metadata from prepare_trtllm_mla_metadata (1 arg)
    kv_cache_block_offsets: torch.Tensor,
    # Cache
    kv_cache: torch.Tensor,
    # Constants
    scale: Optional[float],
    kv_lora_rank: int,
    layer_idx: int = 0,
    out: Optional[torch.Tensor] = None,
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
        kv_cache_block_offsets,
        kv_cache,
        scale,
        kv_lora_rank,
        layer_idx=layer_idx,
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
    kv_cache_block_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
    layer_idx: int = 0,
    out: Optional[torch.Tensor] = None,
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
    # Standard metadata (3 args — matches trtllm attention pattern)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    # Extra metadata from prepare_trtllm_mla_metadata (1 arg)
    kv_cache_block_offsets: torch.Tensor,
    # Cache
    kv_cache: torch.Tensor,
    # Constants
    scale: Optional[float],
    kv_lora_rank: int,
    layer_idx: int = 0,
    out: Optional[torch.Tensor] = None,
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
        kv_cache_block_offsets,
        kv_cache,
        scale,
        kv_lora_rank,
        rotary_cos_sin=rotary_cos_sin,
        layer_idx=layer_idx,
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
    kv_cache_block_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
    layer_idx: int = 0,
    out: Optional[torch.Tensor] = None,
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
    def _has_fp8_model_weights(cls, source_attn_node: Node) -> bool:
        """Detect if the model uses FP8 weights.

        Checks the graph module's parameters for any FP8 tensors.  This runs
        during the ``cache_init`` stage which is after ``weight_load``, so
        parameters carry their actual loaded dtypes.
        """
        gm = source_attn_node.graph.owning_module
        if gm is None:
            return False
        for param in gm.parameters():
            if param.dtype == torch.float8_e4m3fn:
                return True
        return False

    @classmethod
    def get_cache_initializers(
        cls,
        source_attn_node: Node,
        cache_config: KvCacheConfig,
    ) -> ResourceHandlerDict:
        """Return KV cache handler for MLA latent cache.

        MLA stores [compressed_kv | k_pe] per token with num_kv_heads=1.

        For FP8 models, the KV cache is allocated in FP8 dtype when
        ``cache_config.dtype`` is ``"auto"``.  This matches the PT backend
        (which uses ``quant_mode=FP8_KV_CACHE``) and routes to the
        ``mFP8ContextMLA=true`` C++ code path, avoiding the batch-size
        dependent crash in the BF16 context FMHA path.
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

        # Use FP8 KV cache when the model has FP8 weights (matching the PT
        # backend which auto-enables FP8 KV cache via quant_mode).
        if cache_config.dtype == "auto" and cls._has_fp8_model_weights(source_attn_node):
            cache_dtype = torch.float8_e4m3fn
        else:
            cache_dtype = cls.resolve_cache_dtype(
                cache_config.dtype,
                compressed_kv_fake.dtype,
            )

        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads=1,
                head_dim=kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
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
        (scale,) = extract_op_args(source_attn_node, "scale")
        return [scale, kv_lora_rank]

    @classmethod
    def needs_layer_idx(cls) -> bool:
        return True
