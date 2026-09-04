# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-gen decode attention for MiniMax-M3's dense layers (0-2).

Those layers attend the whole page table, so nothing about them needs MSA;
they only run there because MsaPrefillFmha claims every M3 layer. MSA's kernel
uses the context schedule, spending a 128-row Q tile on one decode token, while
trtllm-gen has a generation tile scheduler for exactly this shape.

FlashInferTrtllmGenFmha cannot be reused as-is. It reaches the pool through
build_trtllm_gen_kv_cache_metadata, which assumes each layer contributes
exactly K+V to a pool slot. M3 packs K+V for every layer of a group into one
slot, and sparse layers add an index-K sub-page on top, so there is no uniform
per-layer stride and _kv_pool_mapping_offset is only a ranking, not an
addressable offset. This module goes around that: it builds the same flat
sub-page pool and [batch, 2, max_blocks] block table the kernel expects
directly out of M3's own slot geometry, then calls the same flashinfer entry
point the generic path calls.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers


@functools.lru_cache(maxsize=None)
def _counter_size(num_heads: int, max_num_requests: int, device_index: int) -> int:
    """Byte size of the multi-CTA KV counter block.

    Cached like the workspace size below: it depends only on the head count,
    the request bound and the device's SM count, none of which move during a
    run, while computing it reaches C++ through a deferred import and a
    device-properties query on every dense layer of every step.
    """
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _get_multi_ctas_kv_counter_size,
    )

    multi_processor_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return int(_get_multi_ctas_kv_counter_size(num_heads, max_num_requests, multi_processor_count))


def _device_index(device: torch.device) -> int:
    """Device ordinal of device, resolving the current one where it has none."""
    return device.index if device.index is not None else torch.cuda.current_device()


def _counter_buffer(
    device: torch.device, num_heads: int, max_num_requests: int, reserve: bool
) -> torch.Tensor:
    """Zeroed multi-CTA KV counters for one call.

    Taken from the shared arena, like the workspace beside it, so the block
    joins the graph memory pool and clear_memory_buffers(). The arena hands
    back uninitialized memory, so the zeroing is per call rather than per
    allocation; a few KB of memset costs nothing next to the kernel it feeds.
    """
    counters = get_memory_buffers().get_buffer(
        [_counter_size(num_heads, max_num_requests, _device_index(device))],
        torch.uint8,
        buffer_name="m3_trtllm_gen_kv_counters",
        reserve_buffer=reserve,
    )
    counters.zero_()
    return counters


@functools.lru_cache(maxsize=None)
def _workspace(q_dtype: torch.dtype, num_heads: int, head_dim: int, num_kv_heads: int) -> int:
    """Byte size of the trtllm-gen scratch slab.

    It is a fixed slab (kTrtllmGenWorkspaceSize), independent of the batch, but
    the size is read from the C++ layout rather than hardcoded. Cached so that
    read, and the layout dict it builds, happen once instead of once per dense
    layer per step.
    """
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _get_generation_workspace_layout,
    )

    layout = _get_generation_workspace_layout(q_dtype, 1, 1, num_heads, head_dim, num_kv_heads, 0)
    return int(layout["trtllm_gen_workspace_size"])


def _workspace_buffer(size: int, reserve: bool) -> torch.Tensor:
    return get_memory_buffers().get_buffer(
        [size],
        torch.uint8,
        buffer_name="m3_trtllm_gen_workspace",
        reserve_buffer=reserve,
    )


def reserve_dense_decode_workspace(
    *,
    q_dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    max_num_requests: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Take the scratch slab and the zeroed KV counters, ahead of the kernel.

    Both are sized by the head geometry and the request bound alone, none of
    which move during a run, so a caller that knows those before the phase runs
    can settle the allocation there. Returns them with their combined byte
    size, which the caller can compare across steps to catch a slab that grew
    after a CUDA graph captured it.
    """
    reserve = torch.cuda.is_current_stream_capturing()
    workspace_size = _workspace(q_dtype, num_heads, head_dim, num_kv_heads)
    workspace = _workspace_buffer(workspace_size, reserve)
    counters = _counter_buffer(device, num_heads, max_num_requests, reserve)
    return workspace, counters, workspace_size + int(counters.numel())


def subpage_block_table(
    block_table: torch.Tensor, subpages_per_slot: int, reserve: bool = False
) -> torch.Tensor:
    """Expand a slot table into trtllm-gen's separate K and V page rows.

    uses_shared_paged_kv_idx is False for TensorRT-LLM, so the kernel takes
    [batch, 2, max_blocks] and indexes K and V independently. Rooting the pool
    at this layer's K (see get_kv_subpage_pool) puts slot s's K at
    s * subpages_per_slot and its V one sub-page later.

    The result is a function of the slot table and that factor alone, so every
    dense layer of a step would compute the same one. prepare() therefore
    stages it once into a graph-stable buffer, and this runs only where it
    could not: a manager whose layers disagree on the factor, or a caller that
    skipped prepare().
    """
    batch, max_blocks = block_table.shape
    out = get_memory_buffers().get_buffer(
        [batch, 2, max_blocks],
        torch.int32,
        buffer_name="m3_trtllm_gen_subpage_block_table",
        reserve_buffer=reserve,
    )
    write_subpage_block_table(block_table, subpages_per_slot, out)
    return out


def write_subpage_block_table(
    block_table: torch.Tensor, subpages_per_slot: int, out: torch.Tensor
) -> None:
    """Write the K and V sub-page rows of block_table into out."""
    torch.mul(block_table, subpages_per_slot, out=out[:, 0])
    torch.add(out[:, 0], 1, out=out[:, 1])


def uniform_subpages_per_slot(kv_cache_manager) -> int:
    """Sub-pages per slot when every layer of the pool agrees, else 0.

    The factor is a property of a layer group, so a single-group model has one
    for the whole pool and prepare() can expand the block table without naming
    a layer (see subpage_block_table). A manager with no sub-page pool, or one
    whose groups disagree, reports 0 rather than a guess.
    """
    get_pool = getattr(kv_cache_manager, "get_kv_subpage_pool", None)
    layer_offsets = getattr(kv_cache_manager, "layer_offsets", None)
    if get_pool is None or not layer_offsets:
        return 0
    factors = {int(get_pool(layer_idx, "HND")[1]) for layer_idx in layer_offsets}
    return factors.pop() if len(factors) == 1 else 0


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
    staged_subpage_table: Optional[torch.Tensor] = None,
    staged_subpages_per_slot: int = 0,
    workspace: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
) -> None:
    """Full-context decode attention through trtllm-gen, in place into output.

    staged_subpage_table is the expansion of block_table prepare() already
    staged, used when staged_subpages_per_slot matches this layer's factor and
    expanded here otherwise; see subpage_block_table.

    workspace and counters are the scratch a caller already reserved (see
    reserve_dense_decode_workspace); both are taken from the arena here when a
    caller has not, as the standalone kernel tests do. A supplied buffer is
    checked against the size this call needs, since the caller reserves it from
    a geometry it derives for itself.
    """
    import flashinfer

    kv_pool, subpages_per_slot = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
    num_heads = int(q.shape[1])

    # The kernel variant is picked from the Q dtype and shares one dtype across
    # q/k/v, so an FP8 pool needs FP8 Q. M3 stores unscaled E4M3, so this is a
    # plain cast and a no-op when the fused producer already emitted FP8.
    if kv_pool.dtype == torch.float8_e4m3fn and q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    reserve = torch.cuda.is_current_stream_capturing()
    workspace_size = _workspace(q.dtype, num_heads, int(q.shape[2]), int(kv_pool.shape[1]))
    if workspace is None:
        workspace = _workspace_buffer(workspace_size, reserve)
    elif int(workspace.numel()) < workspace_size:
        raise ValueError(
            f"MiniMax-M3 dense decode was handed a {int(workspace.numel())}-byte "
            f"trtllm-gen workspace, but this call needs {workspace_size}. It was "
            "reserved from a different head geometry than the kernel sees."
        )
    counter_size = _counter_size(num_heads, max_num_requests, _device_index(q.device))
    if counters is None:
        counters = _counter_buffer(q.device, num_heads, max_num_requests, reserve)
    elif int(counters.numel()) < counter_size:
        raise ValueError(
            f"MiniMax-M3 dense decode was handed {int(counters.numel())} bytes of "
            f"multi-CTA KV counters, but this call needs {counter_size}. They were "
            "reserved from a different head or request bound than the kernel sees."
        )
    if staged_subpage_table is None or staged_subpages_per_slot != subpages_per_slot:
        staged_subpage_table = subpage_block_table(block_table, subpages_per_slot, reserve)

    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        # The tuple form takes the K and V page views separately, which is what
        # a flat sub-page pool is: both rows of the block table index into it.
        kv_cache=(kv_pool, kv_pool),
        workspace_buffer=workspace,
        block_tables=staged_subpage_table,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        window_left=-1,  # M3 dense layers are fully causal
        out=output,
        sinks=None,
        kv_layout="HND",
        enable_pdl=enable_pdl,
        backend="trtllm-gen",
        q_len_per_req=decode_query_len,
        max_q_len=None,
        cum_seq_lens_q=None,
        kv_cache_sf=None,  # M3 stores unscaled E4M3
        uses_shared_paged_kv_idx=False,
        multi_ctas_kv_counter_buffer=counters,
    )


@functools.lru_cache(maxsize=1)
def flashinfer_available() -> bool:
    """Whether flashinfer can be imported, resolved once for the process.

    The verdict below is consulted by ensure_msa_available and again per dense
    layer, so the import statement would otherwise be on the hot path.
    """
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


def dense_decode_unsupported_reason(kv_cache_manager, head_dim: int) -> Optional[str]:
    """Return None when the geometry is supported, else why it is not.

    Takes head_dim rather than a query tensor so the metadata's up-front
    validation can reach the same verdict as the call site without one, and so
    the two cannot drift.
    """
    if not hasattr(kv_cache_manager, "get_kv_subpage_pool"):
        return "the KV cache manager does not expose a flat sub-page pool."
    if int(head_dim) != 128:
        return f"head_dim {int(head_dim)}; only 128 has trtllm-gen H128 cubins."
    if not flashinfer_available():
        return "flashinfer is not installed."
    return None


__all__ = [
    "dense_decode_unsupported_reason",
    "flashinfer_available",
    "minimax_m3_trtllm_gen_dense_decode",
    "reserve_dense_decode_workspace",
    "subpage_block_table",
    "uniform_subpages_per_slot",
    "write_subpage_block_table",
]
