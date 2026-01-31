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

"""FlashInfer TRT-LLM attention backend for Auto-Deploy.

This module provides a TRT-LLM attention backend through FlashInfer's interface,
which is much simpler than direct thop.attention usage.

Benefits:
- Simpler integration (~300 lines vs ~1200 lines)
- FlashInfer handles metadata translation internally
- Built-in CUDA graph support
- No need for PT cache backend complexity
- Unified interface with FlashInfer backend

Architecture:
- Uses flashinfer.decode.trtllm_batch_decode_with_kv_cache with backend="trtllm-gen"
- FlashInfer translates AD's metadata to TRT-LLM format internally
- Handles both prefill and decode through FlashInfer's unified interface
"""

from typing import Dict, List, Optional, Tuple

import flashinfer
import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

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
    SequenceInfo,
)


class _FlashInferTrtllmPlanner:
    """Planner for FlashInfer TRT-LLM backend.

    This is simpler than the direct thop.attention approach because FlashInfer
    handles metadata translation and planning internally.
    """

    workspace_buffer: Optional[torch.Tensor]

    # Cached block tables for CUDA graph support
    _cached_block_tables: Dict[int, torch.Tensor]
    _cached_seq_lens: Dict[int, torch.Tensor]

    def __init__(self):
        self.workspace_buffer = None
        self._cached_block_tables = {}
        self._cached_seq_lens = {}

    def init_workspace(self, workspace_buffer: torch.Tensor):
        """Initialize workspace buffer."""
        self.workspace_buffer = workspace_buffer
        # Zero out for first use (required by FlashInfer)
        workspace_buffer.zero_()

    def reset(self) -> None:
        """Reset planner state."""
        # No persistent state to reset
        pass

    def get_or_create_block_table(
        self, batch_size: int, max_pages: int, device: torch.device
    ) -> torch.Tensor:
        """Get or create a block table tensor for CUDA graph support.

        Args:
            batch_size: Batch size
            max_pages: Maximum number of pages per sequence
            device: Device

        Returns:
            Block table tensor [batch_size, max_pages]
        """
        key = (batch_size, max_pages)
        if key not in self._cached_block_tables:
            self._cached_block_tables[key] = torch.zeros(
                batch_size, max_pages, dtype=torch.int32, device=device
            )
        return self._cached_block_tables[key]

    def get_or_create_seq_lens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get or create a seq_lens tensor for CUDA graph support.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Seq lens tensor [batch_size]
        """
        if batch_size not in self._cached_seq_lens:
            self._cached_seq_lens[batch_size] = torch.zeros(
                batch_size, dtype=torch.int32, device=device
            )
        return self._cached_seq_lens[batch_size]


_GlobalFlashInferTrtllmPlanner = _FlashInferTrtllmPlanner()


@torch.library.custom_op(
    "auto_deploy::flashinfer_trtllm_attention_prepare_metadata", mutates_args=()
)
def prepare_flashinfer_trtllm_metadata(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare metadata for FlashInfer TRT-LLM attention.

    Converts AD's metadata format to FlashInfer's expected format:
    - block_tables: [batch_size, max_pages_per_seq]
    - seq_lens: [batch_size]

    Args:
        input_ids: Input IDs [batch_size, seq_len]
        position_ids: Position IDs
        seq_len: Sequence lengths [batch_size]
        input_pos: Input positions (offsets into KV cache)
        cache_loc: Flat page indices for all sequences
        pages_per_seq: Pages per sequence [batch_size]

    Returns:
        Tuple of (block_tables, seq_lens)
    """
    # Reset planner
    _GlobalFlashInferTrtllmPlanner.reset()

    # Get number of sequences from input_ids
    num_seq = input_ids.shape[0]
    seq_len = seq_len[:num_seq]

    # Prepare FlashInfer metadata
    offsets = input_pos[:num_seq].clone()

    # Compute KV sequence lengths (including cached tokens)
    # seq_lens_kv[i] = offset[i] + seq_len[i]
    seq_lens_kv = (offsets + seq_len).int()

    # Compute max pages per sequence from pages_per_seq
    max_pages_per_seq = pages_per_seq[:num_seq].max().item() if num_seq > 0 else 1

    # Build block tables from cache_loc
    # block_tables[i, j] = cache_loc[offset + j] where offset is the cumulative pages before seq i
    device = cache_loc.device

    # Get or create cached tensors for CUDA graph support
    is_capturing = torch.cuda.is_current_stream_capturing()
    if is_capturing:
        block_tables = _GlobalFlashInferTrtllmPlanner.get_or_create_block_table(
            num_seq, max_pages_per_seq, device
        )
        seq_lens_out = _GlobalFlashInferTrtllmPlanner.get_or_create_seq_lens(num_seq, device)
        # Zero out before filling
        block_tables.zero_()
    else:
        block_tables = torch.zeros(num_seq, max_pages_per_seq, dtype=torch.int32, device=device)
        seq_lens_out = torch.zeros(num_seq, dtype=torch.int32, device=device)

    # Fill block tables
    offset = 0
    for i in range(num_seq):
        n_pages = pages_per_seq[i].item()
        if n_pages > 0:
            block_tables[i, :n_pages] = cache_loc[offset : offset + n_pages]
            offset += n_pages
        seq_lens_out[i] = seq_lens_kv[i]

    return block_tables, seq_lens_out


@prepare_flashinfer_trtllm_metadata.register_fake
def prepare_flashinfer_trtllm_metadata_fake(
    input_ids,
    position_ids,
    seq_len,
    input_pos,
    cache_loc,
    pages_per_seq,
):
    """Fake implementation for torch.compile."""
    num_seq = input_ids.shape[0]
    seq_len = seq_len[:num_seq]
    max_pages_per_seq = pages_per_seq[:num_seq].max().item() if num_seq > 0 else 1
    block_tables = torch.empty(num_seq, max_pages_per_seq, dtype=torch.int32, device=seq_len.device)
    seq_lens_out = torch.empty(num_seq, dtype=torch.int32, device=seq_len.device)
    return block_tables, seq_lens_out


@torch.library.custom_op(
    "auto_deploy::flashinfer_trtllm_attention_mha_with_cache", mutates_args=("k_cache", "v_cache")
)
def flashinfer_trtllm_mha_with_cache(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (passed through but not used directly)
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    # EXTRA METADATA (from prepare_metadata)
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
) -> torch.Tensor:
    """FlashInfer TRT-LLM attention with KV cache.

    This uses FlashInfer's trtllm_batch_decode_with_kv_cache which internally
    calls TRT-LLM kernels but handles all metadata translation.

    Args:
        q: Query tensor [b, s, n_heads, head_dim]
        k: Key tensor [b, s, n_kv_heads, head_dim]
        v: Value tensor [b, s, n_kv_heads, head_dim]
        input_ids: Input IDs (standard metadata, passed through)
        position_ids: Position IDs (standard metadata, passed through)
        seq_len: Sequence lengths (standard metadata, passed through)
        input_pos: Input positions (standard metadata, passed through)
        cache_loc: Cache locations (standard metadata, passed through)
        pages_per_seq: Pages per sequence (standard metadata, passed through)
        block_tables: Block table [batch_size, max_pages_per_seq] (from prepare_metadata)
        seq_lens: Sequence lengths [batch_size] (from prepare_metadata)
        k_cache: K cache [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: V cache [num_pages, page_size, num_kv_heads, head_dim]
        workspace_buffer: Workspace buffer
        scale: Softmax scale

    Returns:
        Output tensor [b, s, n_heads, head_dim]
    """
    # Get dimensions
    q_shape_og = q.shape
    b, s, n_heads, head_dim = q_shape_og
    n_kv_heads = k.shape[2]

    # Get page_size and max_seq_len from cache shape
    page_size = k_cache.shape[1]
    max_seq_len = seq_lens.max().item()

    # Reshape to [num_tokens, n_heads, head_dim]
    num_tokens = b * s
    q_flat = q.reshape(num_tokens, n_heads, head_dim)
    k_flat = k.reshape(num_tokens, n_kv_heads, head_dim)
    v_flat = v.reshape(num_tokens, n_kv_heads, head_dim)

    # Compute batch indices and positions for KV cache append
    # For decode: one token per sequence at the end
    # For prefill: multiple tokens per sequence
    batch_indices = torch.arange(b, dtype=torch.int32, device=q.device).repeat_interleave(s)

    # Compute positions within each sequence
    # positions[i*s + j] = (seq_lens[i] - s) + j for j in range(s)
    # This represents the position in the KV cache where token j of sequence i should be written
    base_positions = (seq_lens - s).unsqueeze(1)  # [b, 1]
    offsets = torch.arange(s, dtype=torch.int32, device=q.device).unsqueeze(0)  # [1, s]
    positions = (base_positions + offsets).reshape(-1)  # [b*s]

    # Prepare page indices for append operation
    # FlashInfer needs flattened page indices
    paged_kv_indptr = torch.zeros(b + 1, dtype=torch.int32, device=q.device)
    # Count non-zero entries in block_tables per sequence
    pages_per_seq = (block_tables != 0).sum(dim=1).int()
    paged_kv_indptr[1:] = torch.cumsum(pages_per_seq, 0)

    # Flatten block_tables to get paged_kv_indices
    paged_kv_indices = block_tables[block_tables != 0].contiguous()

    # Compute last page lengths
    # Last page len = (seq_lens - 1) % page_size + 1
    paged_kv_last_page_len = ((seq_lens - 1) % page_size + 1).int()

    # Append K, V to cache
    flashinfer.page.append_paged_kv_cache(
        k_flat,
        v_flat,
        batch_indices,
        positions,
        (k_cache, v_cache),
        paged_kv_indices,
        paged_kv_indptr,
        paged_kv_last_page_len,
    )

    # Compute softmax scale
    sm_scale = scale if scale is not None else (1.0 / (head_dim**0.5))

    # Run FlashInfer TRT-LLM attention
    # Use backend="trtllm-gen" to explicitly use TRT-LLM kernels
    # or backend="auto" to let FlashInfer choose based on architecture
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q_flat,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        window_left=-1,  # Full attention
        kv_layout="NHD",  # AD uses [num_pages, page_size, num_kv_heads, head_dim]
        backend="trtllm-gen",  # Use TRT-LLM backend explicitly
        q_len_per_req=s,  # Query length per request
    )

    # Reshape back to original shape
    return output.view(q_shape_og)


@flashinfer_trtllm_mha_with_cache.register_fake
def flashinfer_trtllm_mha_with_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    scale: Optional[float],
) -> torch.Tensor:
    """Fake implementation for torch.compile."""
    return torch.empty_like(q.contiguous())


@AttentionRegistry.register("flashinfer_trtllm")
class FlashInferTrtllmAttention(AttentionDescriptor):
    """FlashInfer TRT-LLM attention backend.

    This backend uses FlashInfer's TRT-LLM interface, which provides:
    - Simpler integration than direct thop.attention
    - Built-in CUDA graph support
    - Automatic metadata translation
    - Unified interface with FlashInfer

    Usage:
        Set `backend: flashinfer_trtllm` in your AD config under `insert_cached_attention`.
    """

    @classmethod
    def _get_planner(cls) -> _FlashInferTrtllmPlanner:
        return _GlobalFlashInferTrtllmPlanner

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
        return torch.ops.auto_deploy.flashinfer_trtllm_attention_mha_with_cache

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Get extra metadata preparation info.

        Returns the prepare_metadata function that converts standard metadata
        to FlashInfer TRT-LLM format (block_tables, seq_lens).
        """
        return (
            torch.ops.auto_deploy.flashinfer_trtllm_attention_prepare_metadata.default,
            2,  # Number of outputs: (block_tables, seq_lens)
            [],  # No constants needed
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        """Provide cache initializer functions."""
        # Extract tensor shapes from source node
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]

        def _get_cache(si: SequenceInfo):
            # AD cache format: [num_pages, page_size, num_kv_heads, head_dim]
            # This matches FlashInfer's NHD layout
            return torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                head_dim,
                device=si.device,
                dtype=cache_config.dtype or k_fake.dtype,
            )

        return {"k_cache": _get_cache, "v_cache": _get_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        """Provide global buffer initializer functions."""

        def _init_workspace(si: SequenceInfo) -> torch.Tensor:
            # FlashInfer workspace - 128 MB recommended
            buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=si.device)
            cls._get_planner().init_workspace(buffer)
            ad_logger.info(
                f"[FlashInfer TRT-LLM] Initialized workspace: {buffer.shape}, device={buffer.device}"
            )
            return buffer

        return {"workspace_buffer": _init_workspace}

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments."""
        return [
            "input_ids",
            "position_ids",
            "seq_len",
            "input_pos",
            "cache_loc",
            "pages_per_seq",
        ]

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

        ad_logger.debug(f"[FlashInfer TRT-LLM] Constants: scale={scale}")

        return [scale]  # Only scale is needed as a constant
