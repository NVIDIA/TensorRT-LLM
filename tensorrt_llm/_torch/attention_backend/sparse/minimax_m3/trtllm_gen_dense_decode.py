# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-gen context and decode attention for MiniMax-M3 dense layers.

Those layers attend the whole page table, so nothing about them needs MSA;
they only run there because MsaSparseGqaFmha claims every M3 target layer.
MSA's kernel uses the context schedule, spending a 128-row Q tile on one
decode token, while trtllm-gen has generation scheduling for exactly this
shape. The NVFP4 path also uses trtllm-gen context attention because its
shipped cubins consume physical P32 packed data and block-scale pages.

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
from typing import Optional

import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm.bindings import DataType

from .msa_utils import check_decode_span_shape


def _layer_uses_nvfp4(kv_cache_manager, layer_idx: int) -> bool:
    predicate = getattr(kv_cache_manager, "is_nvfp4_layer", None)
    if predicate is not None:
        return bool(predicate(layer_idx))
    return getattr(kv_cache_manager, "dtype", None) == DataType.NVFP4


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


def _counter_buffer(
    device: torch.device, num_heads: int, max_num_requests: int, reserve: bool
) -> torch.Tensor:
    """Zeroed multi-CTA KV counters for one call.

    Taken from the shared arena, like the workspace beside it, so the block
    joins the graph memory pool and clear_memory_buffers(). The arena hands
    back uninitialized memory, so the zeroing is per call rather than per
    allocation; a few KB of memset costs nothing next to the kernel it feeds.
    """
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    counters = get_memory_buffers().get_buffer(
        [_counter_size(num_heads, max_num_requests, device_index)],
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


@functools.lru_cache(maxsize=None)
def _context_workspace(
    q_dtype: torch.dtype,
    max_num_requests: int,
    max_num_tokens: int,
    num_heads: int,
    head_dim: int,
) -> int:
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _get_context_workspace_size,
    )

    return int(
        _get_context_workspace_size(
            q_dtype,
            max_num_requests,
            max_num_tokens,
            num_heads,
            head_dim,
            0,
            True,
        )
    )


def _dense_kv_inputs(
    q: torch.Tensor,
    kv_cache_manager,
    layer_idx: int,
    *,
    sm_scale: float,
    kv_scale_quant_orig: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    int,
    int,
    float | torch.Tensor,
    float | torch.Tensor,
]:
    """Resolve direct TRTLLM-gen inputs for one M3 dense layer."""
    get_dense_pool = getattr(kv_cache_manager, "get_dense_kv_subpage_pool", None)
    if get_dense_pool is None:
        kv_pool, subpages_per_slot = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
        pages_per_role = 1
    else:
        kv_pool, subpages_per_slot, pages_per_role = get_dense_pool(layer_idx)
    kv_scale_pool = None
    bmm1_scale: float | torch.Tensor = sm_scale
    bmm2_scale: float | torch.Tensor = 1.0

    if _layer_uses_nvfp4(kv_cache_manager, layer_idx):
        if kv_scale_quant_orig is None:
            raise RuntimeError("MiniMax-M3 dense NVFP4 attention requires [Q, K, V] scales")
        if kv_scale_quant_orig.dtype != torch.float32 or kv_scale_quant_orig.numel() < 3:
            raise ValueError("MiniMax-M3 dense NVFP4 scales must be FP32 [Q, K, V]")
        kv_scale_pool, scale_factor, scale_pages = kv_cache_manager.get_dense_kv_scale_subpage_pool(
            layer_idx
        )
        if (int(scale_factor), int(scale_pages)) != (
            int(subpages_per_slot),
            int(pages_per_role),
        ):
            raise RuntimeError(
                "MiniMax-M3 NVFP4 data/scale pools have different block-table geometry"
            )
        q = q.to(torch.float8_e4m3fn)
        kv_pool = kv_pool.view(torch.uint8)
        kv_scale_pool = kv_scale_pool.view(torch.float8_e4m3fn)
        raw_bmm1 = kv_scale_quant_orig[1:2] * float(sm_scale)
        bmm1_scale = torch.cat((raw_bmm1, raw_bmm1 * 1.4426950408889634))
        bmm2_scale = kv_scale_quant_orig[2:3]
    elif kv_pool.dtype == torch.float8_e4m3fn and q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    return (
        q,
        kv_pool,
        kv_scale_pool,
        int(subpages_per_slot),
        int(pages_per_role),
        bmm1_scale,
        bmm2_scale,
    )


def subpage_block_table(
    block_table: torch.Tensor,
    subpages_per_slot: int,
    reserve: bool = False,
    pages_per_role: int = 1,
) -> torch.Tensor:
    """Expand a slot table into trtllm-gen's separate K and V page rows.

    uses_shared_paged_kv_idx is False for TensorRT-LLM, so the kernel takes
    [batch, 2, max_blocks] and indexes K and V independently. With physical
    P32 NVFP4 pages, each logical slot expands to ``pages_per_role`` entries.

    The result is a function of the slot table and that factor alone, so every
    dense layer of a step would compute the same one. prepare() therefore
    stages it once into a graph-stable buffer, and this runs only where it
    could not: a manager whose layers disagree on the factor, or a caller that
    skipped prepare().
    """
    batch, max_blocks = block_table.shape
    out = get_memory_buffers().get_buffer(
        [batch, 2, max_blocks * pages_per_role],
        torch.int32,
        buffer_name="m3_trtllm_gen_subpage_block_table",
        reserve_buffer=reserve,
    )
    write_subpage_block_table(block_table, subpages_per_slot, out, pages_per_role)
    return out


def write_subpage_block_table(
    block_table: torch.Tensor,
    subpages_per_slot: int,
    out: torch.Tensor,
    pages_per_role: int = 1,
) -> None:
    """Write the K and V sub-page rows of block_table into out."""
    if pages_per_role == 1:
        torch.mul(block_table, subpages_per_slot, out=out[:, 0])
        torch.add(out[:, 0], 1, out=out[:, 1])
        return
    offsets = torch.arange(pages_per_role, dtype=block_table.dtype, device=block_table.device)
    key_pages = (block_table.unsqueeze(-1) * subpages_per_slot + offsets).flatten(1)
    out[:, 0].copy_(key_pages)
    torch.add(out[:, 0], pages_per_role, out=out[:, 1])


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


def uniform_dense_subpage_geometry(kv_cache_manager) -> tuple[int, int]:
    """Common dense slot stride and pages per role, or ``(0, 0)``."""
    get_pool = getattr(kv_cache_manager, "get_dense_kv_subpage_pool", None)
    layer_offsets = getattr(kv_cache_manager, "layer_offsets", None)
    if not layer_offsets:
        return 0, 0
    dense_layers = [
        layer_idx
        for layer_idx in layer_offsets
        if not getattr(kv_cache_manager, "sparse_layer_ids", ())
        or layer_idx not in kv_cache_manager.sparse_layer_ids
    ]
    if not dense_layers:
        return 0, 0
    if get_pool is None:
        legacy = getattr(kv_cache_manager, "get_kv_subpage_pool", None)
        if legacy is None:
            return 0, 0
        geometries = {(int(legacy(layer_idx, "HND")[1]), 1) for layer_idx in dense_layers}
    else:
        geometries = set()
        for layer_idx in dense_layers:
            _pool, slot_stride, pages_per_role = get_pool(layer_idx)
            geometries.add((int(slot_stride), int(pages_per_role)))
    return geometries.pop() if len(geometries) == 1 else (0, 0)


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
    kv_scale_quant_orig: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
) -> None:
    """Full-context decode attention through trtllm-gen, in place into output.

    staged_subpage_table is the expansion of block_table prepare() already
    staged, used when staged_subpages_per_slot matches this layer's factor and
    expanded here otherwise; see subpage_block_table.
    """
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _trtllm_gen_batch_decode_with_kv_cache,
    )

    # The multi-CTA KV counters are sized against max_num_requests, so a batch
    # read out of a longer q would undersize them.
    check_decode_span_shape(
        "MiniMax-M3 trtllm-gen dense decode",
        int(q.shape[0]),
        int(seq_lens.shape[0]),
        decode_query_len,
    )

    (
        q,
        kv_pool,
        kv_scale_pool,
        subpages_per_slot,
        pages_per_role,
        bmm1_scale,
        bmm2_scale,
    ) = _dense_kv_inputs(
        q,
        kv_cache_manager,
        layer_idx,
        sm_scale=sm_scale,
        kv_scale_quant_orig=kv_scale_quant_orig,
    )
    num_heads = int(q.shape[1])

    reserve = torch.cuda.is_current_stream_capturing()
    workspace = get_memory_buffers().get_buffer(
        [_workspace(q.dtype, num_heads, int(q.shape[2]), int(kv_pool.shape[1]))],
        torch.uint8,
        buffer_name="m3_trtllm_gen_workspace",
        reserve_buffer=reserve,
    )
    if staged_subpage_table is None or staged_subpages_per_slot != subpages_per_slot:
        staged_subpage_table = subpage_block_table(
            block_table, subpages_per_slot, reserve, pages_per_role
        )

    _trtllm_gen_batch_decode_with_kv_cache(
        q,  # query
        kv_pool,  # kv_pool
        workspace,  # workspace_buffer
        _counter_buffer(
            q.device, num_heads, max_num_requests, reserve
        ),  # multi_ctas_kv_counter_buffer
        staged_subpage_table,  # block_tables
        seq_lens,  # seq_lens
        max_seq_len,  # max_seq_len
        bmm1_scale,  # bmm1_scale
        bmm2_scale,  # bmm2_scale
        -1,  # window_left: M3 dense layers are fully causal
        output,  # out
        None,  # sinks
        enable_pdl,  # enable_pdl
        decode_query_len,  # q_len_per_req
        None,  # max_q_len
        None,  # cum_seq_lens_q
        kv_scale_pool,  # NVFP4 E4M3 block scales, otherwise None
        False,  # uses_shared_paged_kv_idx
    )


def minimax_m3_trtllm_gen_dense_context(
    q: torch.Tensor,
    kv_cache_manager,
    layer_idx: int,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    *,
    sm_scale: float,
    output: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    max_num_requests: int,
    kv_scale_quant_orig: Optional[torch.Tensor] = None,
    staged_subpage_table: Optional[torch.Tensor] = None,
    staged_subpages_per_slot: int = 0,
    enable_pdl: bool = True,
) -> None:
    """Full-context attention for M3 dense layers, including NVFP4 KV."""
    from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
        _trtllm_gen_batch_context_with_kv_cache,
    )

    (
        q,
        kv_pool,
        kv_scale_pool,
        subpages_per_slot,
        pages_per_role,
        bmm1_scale,
        bmm2_scale,
    ) = _dense_kv_inputs(
        q,
        kv_cache_manager,
        layer_idx,
        sm_scale=sm_scale,
        kv_scale_quant_orig=kv_scale_quant_orig,
    )
    batch_size = int(seq_lens.shape[0])
    num_heads = int(q.shape[1])
    reserve = torch.cuda.is_current_stream_capturing()
    workspace = get_memory_buffers().get_buffer(
        [
            _context_workspace(
                q.dtype, max_num_requests, int(q.shape[0]), num_heads, int(q.shape[2])
            )
        ],
        torch.uint8,
        buffer_name="m3_trtllm_gen_context_workspace",
        reserve_buffer=reserve,
    )
    if staged_subpage_table is None or staged_subpages_per_slot != subpages_per_slot:
        staged_subpage_table = subpage_block_table(
            block_table, subpages_per_slot, reserve, pages_per_role
        )

    _trtllm_gen_batch_context_with_kv_cache(
        q,
        kv_pool,
        workspace,
        _counter_buffer(q.device, num_heads, max_num_requests, reserve),
        staged_subpage_table,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        cu_q_lens,
        cu_kv_lens,
        -1,
        output,
        None,
        enable_pdl,
        kv_scale_pool,
        False,
        True,
    )


def minimax_m3_trtllm_gen_dense_attention(
    q: torch.Tensor,
    kv_cache_manager,
    layer_idx: int,
    metadata,
    *,
    sm_scale: float,
    output: torch.Tensor,
    kv_scale_quant_orig: Optional[torch.Tensor],
) -> None:
    """Dispatch the packed M3 dense batch by context/generation phase."""
    qo_lens = metadata.msa_qo_lens_cpu
    kv_lens = metadata.msa_kv_lens_cpu
    if qo_lens is None or kv_lens is None:
        raise RuntimeError("MiniMax-M3 dense attention metadata was not prepared")
    num_contexts = int(metadata.num_contexts or 0)
    batch_size = int(qo_lens.shape[0])
    ctx_tokens = int(qo_lens[:num_contexts].sum().item()) if num_contexts else 0

    staged_rows = getattr(metadata, "msa_subpage_rows", None)
    if num_contexts:
        staged_table, staged_factor = (
            staged_rows(0, num_contexts) if staged_rows is not None else (None, 0)
        )
        minimax_m3_trtllm_gen_dense_context(
            q[:ctx_tokens],
            kv_cache_manager,
            layer_idx,
            metadata.msa_block_table[:num_contexts],
            metadata.msa_seq_lens_cuda[:num_contexts],
            metadata.msa_cu_q_lens[: num_contexts + 1],
            metadata.msa_cu_kv_lens[: num_contexts + 1],
            sm_scale=sm_scale,
            output=output[:ctx_tokens],
            max_q_len=int(qo_lens[:num_contexts].max().item()),
            max_kv_len=int(kv_lens[:num_contexts].max().item()),
            max_num_requests=int(metadata.max_num_requests),
            kv_scale_quant_orig=kv_scale_quant_orig,
            staged_subpage_table=staged_table,
            staged_subpages_per_slot=staged_factor,
        )

    if num_contexts == batch_size:
        return
    gen_qo = qo_lens[num_contexts:]
    decode_query_len = int(gen_qo[0].item())
    if not torch.equal(gen_qo, torch.full_like(gen_qo, decode_query_len)):
        raise NotImplementedError("MiniMax-M3 dense generation requires a uniform query length")
    staged_table, staged_factor = (
        staged_rows(num_contexts, batch_size) if staged_rows is not None else (None, 0)
    )
    minimax_m3_trtllm_gen_dense_decode(
        q[ctx_tokens:],
        kv_cache_manager,
        layer_idx,
        metadata.msa_block_table[num_contexts:batch_size],
        metadata.msa_seq_lens_cuda[num_contexts:batch_size],
        sm_scale=sm_scale,
        output=output[ctx_tokens:],
        decode_query_len=decode_query_len,
        max_seq_len=int(kv_lens[num_contexts:].max().item()),
        max_num_requests=int(metadata.max_num_requests),
        staged_subpage_table=staged_table,
        staged_subpages_per_slot=staged_factor,
        kv_scale_quant_orig=kv_scale_quant_orig,
    )


@functools.lru_cache(maxsize=1)
def _flashinfer_available() -> bool:
    """Whether flashinfer can be imported, resolved once for the process.

    The verdict below is consulted once per step by prepare() and again per
    dense layer, so the import statement would otherwise be on the hot path.
    """
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


def dense_decode_unsupported_reason(kv_cache_manager, head_dim: int) -> Optional[str]:
    """Return None when the geometry is supported, else why it is not.

    Takes head_dim rather than a query tensor so prepare() can reach the same
    verdict as the call site without one, and so the two cannot drift.
    """
    if not hasattr(kv_cache_manager, "get_kv_subpage_pool"):
        return "the KV cache manager does not expose a flat sub-page pool."
    if int(head_dim) != 128:
        return f"head_dim {int(head_dim)}; only 128 has trtllm-gen H128 cubins."
    dense_layers = [
        layer_idx
        for layer_idx in getattr(kv_cache_manager, "layer_offsets", {})
        if layer_idx not in getattr(kv_cache_manager, "sparse_layer_ids", ())
    ]
    probe_layer = dense_layers[0] if dense_layers else 0
    if _layer_uses_nvfp4(kv_cache_manager, probe_layer):
        if not hasattr(kv_cache_manager, "get_dense_kv_scale_subpage_pool"):
            return "the NVFP4 manager does not expose a dense block-scale pool."
        if not dense_layers:
            return "the NVFP4 manager has no dense attention layer."
        dense_pool, _stride, _pages = kv_cache_manager.get_dense_kv_subpage_pool(dense_layers[0])
        if int(dense_pool.shape[2]) != 32:
            return (
                f"the NVFP4 dense pool uses P{int(dense_pool.shape[2])}; "
                "the shipped trtllm-gen cubins require P32."
            )
    if not _flashinfer_available():
        return "flashinfer is not installed."
    return None


__all__ = [
    "dense_decode_unsupported_reason",
    "minimax_m3_trtllm_gen_dense_attention",
    "minimax_m3_trtllm_gen_dense_context",
    "minimax_m3_trtllm_gen_dense_decode",
    "subpage_block_table",
    "uniform_dense_subpage_geometry",
    "uniform_subpages_per_slot",
    "write_subpage_block_table",
]
