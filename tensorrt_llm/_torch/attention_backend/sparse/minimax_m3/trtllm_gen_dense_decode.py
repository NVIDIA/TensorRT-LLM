# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-gen decode attention for MiniMax-M3's dense layers (0-2).

Those layers attend the whole page table, so nothing about them needs MSA;
they only run there because ``MsaSparseGqaFmha`` claims every M3 layer. MSA's
kernel uses the context schedule, spending a 128-row Q tile on one decode
token, while trtllm-gen has a generation tile scheduler for exactly this shape.

``FlashInferTrtllmGenFmha`` cannot be reused as-is. It reaches the pool through
``build_trtllm_gen_kv_cache_metadata``, which assumes each layer contributes
exactly K+V to a pool slot. M3 packs K+V for every layer of a group into one
slot, and sparse layers add an index-K sub-page on top, so there is no uniform
per-layer stride and ``_kv_pool_mapping_offset`` is only a ranking, not an
addressable offset. This module goes around that: it builds the same flat
sub-page pool and ``[batch, 2, max_blocks]`` block table the kernel expects
directly out of M3's own slot geometry, then calls the same flashinfer entry
point the generic path calls.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

# Zeroed on every call rather than only at allocation: a few KB of memset per
# dense layer costs nothing, and it removes any dependence on the kernel
# leaving the counters back at zero.
_counter_buffers: dict[Tuple[int, int], torch.Tensor] = {}


def _multi_processor_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_properties(index).multi_processor_count


def _counter_buffer(device: torch.device, num_heads: int, max_num_requests: int) -> torch.Tensor:
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _get_multi_ctas_kv_counter_size,
    )

    size = _get_multi_ctas_kv_counter_size(
        num_heads, max_num_requests, _multi_processor_count(device)
    )
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, size)
    buffer = _counter_buffers.get(key)
    if buffer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The trtllm-gen multi-CTA KV counter buffer must be allocated "
                "before CUDA graph capture."
            )
        buffer = torch.zeros(size, dtype=torch.uint8, device=device)
        _counter_buffers[key] = buffer
    return buffer


def _workspace(q_dtype: torch.dtype, num_heads: int, head_dim: int, num_kv_heads: int) -> int:
    """Byte size of the trtllm-gen scratch slab.

    It is a fixed slab (``kTrtllmGenWorkspaceSize``), independent of the batch,
    but the size is read from the C++ layout rather than hardcoded.
    """
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _get_generation_workspace_layout,
    )

    layout = _get_generation_workspace_layout(q_dtype, 1, 1, num_heads, head_dim, num_kv_heads, 0)
    return int(layout["trtllm_gen_workspace_size"])


def _subpage_block_table(
    block_table: torch.Tensor, subpages_per_slot: int, reserve: bool
) -> torch.Tensor:
    """Expand a slot table into trtllm-gen's separate K and V page rows.

    ``uses_shared_paged_kv_idx`` is False for TensorRT-LLM, so the kernel takes
    ``[batch, 2, max_blocks]`` and indexes K and V independently. Rooting the
    pool at this layer's K (see ``get_kv_subpage_pool``) puts slot ``s``'s K at
    ``s * subpages_per_slot`` and its V one sub-page later.
    """
    batch, max_blocks = block_table.shape
    out = get_memory_buffers().get_buffer(
        [batch, 2, max_blocks],
        torch.int32,
        buffer_name="m3_trtllm_gen_subpage_block_table",
        reserve_buffer=reserve,
    )
    torch.mul(block_table, subpages_per_slot, out=out[:, 0])
    torch.add(out[:, 0], 1, out=out[:, 1])
    return out


def minimax_m3_trtllm_gen_dense_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache_manager,
    layer_idx: int,
    block_table: torch.Tensor,  # [batch, max_blocks] slot ids
    seq_lens: torch.Tensor,  # [batch] int32
    *,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    max_seq_len: int,
    max_num_requests: int,
    enable_pdl: bool = True,
) -> None:
    """Full-context decode attention through trtllm-gen, in place into ``output``."""
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _trtllm_gen_batch_decode_with_kv_cache,
    )

    kv_pool, subpages_per_slot = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
    num_heads = int(q.shape[1])

    # The kernel variant is picked from the Q dtype and shares one dtype across
    # q/k/v, so an FP8 pool needs FP8 Q. M3 stores unscaled E4M3, so this is a
    # plain cast and a no-op when the fused producer already emitted FP8.
    if kv_pool.dtype == torch.float8_e4m3fn and q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    reserve = torch.cuda.is_current_stream_capturing()
    workspace = get_memory_buffers().get_buffer(
        [_workspace(q.dtype, num_heads, int(q.shape[2]), int(kv_pool.shape[1]))],
        torch.uint8,
        buffer_name="m3_trtllm_gen_workspace",
        reserve_buffer=reserve,
    )
    counters = _counter_buffer(q.device, num_heads, max_num_requests)
    counters.zero_()

    _trtllm_gen_batch_decode_with_kv_cache(
        q,  # query
        kv_pool,  # kv_pool
        workspace,  # workspace_buffer
        counters,  # multi_ctas_kv_counter_buffer
        _subpage_block_table(block_table, subpages_per_slot, reserve),  # block_tables
        seq_lens,  # seq_lens
        max_seq_len,  # max_seq_len
        sm_scale,  # bmm1_scale
        1.0,  # bmm2_scale
        -1,  # window_left: M3 dense layers are fully causal
        output,  # out
        None,  # sinks
        enable_pdl,  # enable_pdl
        decode_query_len,  # q_len_per_req
        None,  # max_q_len
        None,  # cum_seq_lens_q
        None,  # kv_scale_pool: M3 stores unscaled E4M3
        False,  # uses_shared_paged_kv_idx
    )


def dense_decode_sm_scale(head_dim: int, q_scaling: float) -> float:
    """bmm1 scale in trtllm-gen's convention, matching FlashInferTrtllmGenFmha."""
    return 1.0 / (math.sqrt(head_dim) * q_scaling)


def dense_decode_supported(kv_cache_manager, q: torch.Tensor) -> Optional[str]:
    """Return None when the geometry is supported, else why it is not."""
    if not hasattr(kv_cache_manager, "get_kv_subpage_pool"):
        return "the KV cache manager does not expose a flat sub-page pool."
    if q.shape[-1] != 128:
        return f"head_dim {int(q.shape[-1])}; only 128 has trtllm-gen H128 cubins."
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "flashinfer is not installed."
    return None


__all__ = [
    "dense_decode_sm_scale",
    "dense_decode_supported",
    "minimax_m3_trtllm_gen_dense_decode",
]
