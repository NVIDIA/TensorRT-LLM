# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe decode kernels for MiniMax-M3 sparse attention.

Replaces MSA's host-centric `fmha_sm100_plan` / `fmha_sm100` launch for
the decode path while invoking the same JIT-compiled SM100 kernel
binaries via `fmha_sm100.jit.get_fmha_variant`. MSA's plan bakes
host-side values into the launch, which CUDA graph replays would freeze,
so only the launch is replaced and the kernels are reused.

The persistent buffers live in an `M3DecodeState` that the attention
metadata builds once (outside capture) and owns, so their `data_ptr()`
stays stable across CUDA graph replays. The kernel launches are the
module-level `decode_*` functions that operate on that state.

Design contract:

* No plan/run split: every call assembles launch args directly.
* Everything per-step-varying is a device tensor: `seq_lens`,
  `kv_page_indptr`, `kv_indices`, `kv_block_indexes`, and the `max_score`
  contents.
* Host-baked values are geometry or per-batch-size constants only: head
  counts, pack factor, page size, `max_k_tiles` capacity, and worklists
  (a pure function of batch size for decode).
* Every function is callable inside a CUDA graph capture and yields
  correct results at replay: no `.item()`, `.cpu()`, or `.tolist()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from .topk import select_topk_blocks
from .worklist import build_decode_worklist

if TYPE_CHECKING:
    from ..metadata import MiniMaxM3SparseConfig

# Mirrors fmha_sm100.jit._PACK_FACTORS.
_PACK_FACTORS = (1, 2, 4, 6, 8, 16)
_QO_TILE_SIZE = 128
_WORKSPACE_BYTES = 32 * 1024 * 1024


def _compute_pack_factor(max_qo_len: int, num_qo_heads: int, num_kv_heads: int) -> int:
    """Verbatim port of `fmha_sm100.api._compute_pack_factor`."""
    if num_kv_heads == -1:
        return 1
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1


def _max_k_tiles_capacity(max_kv_len: int) -> int:
    """MSA's `max_k_tiles` formula at max capacity.

    Number of 128-token KV blocks rounded up to a multiple of 128 (the
    kernel's max-score tile stride granularity).
    """
    return math.ceil(math.ceil(max_kv_len / 128) / 128) * 128


@dataclass(frozen=True)
class M3DecodeGeometry:
    """Compile/alloc-time constants for one M3 layer family (per rank)."""

    num_q_heads: int
    num_kv_heads: int
    num_index_heads: int
    head_dim: int
    page_size: int
    topk: int
    init_blocks: int
    local_blocks: int
    max_batch: int
    max_kv_len: int

    def __post_init__(self):
        if self.head_dim != 128 or self.page_size != 128:
            raise NotImplementedError(
                "MSA SM100 decode kernels require head_dim=128 and page_size=128; "
                f"got head_dim={self.head_dim}, page_size={self.page_size}."
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.num_index_heads % self.num_kv_heads != 0:
            raise ValueError("num_index_heads must be divisible by num_kv_heads")

    @classmethod
    def from_config(
        cls,
        config: "MiniMaxM3SparseConfig",
        *,
        max_batch: int,
        max_kv_len: int,
        page_size: Optional[int] = None,
    ) -> "M3DecodeGeometry":
        """Build the decode alloc-time key from the layer config.

        Adds the two runtime dims the driver needs to size its buffers
        (`max_batch`, `max_kv_len`). `page_size` defaults to `block_size`.
        """
        return cls(
            num_q_heads=int(config.num_q_heads),
            num_kv_heads=int(config.num_kv_heads),
            num_index_heads=int(config.num_index_heads),
            head_dim=int(config.head_dim),
            page_size=int(config.block_size if page_size is None else page_size),
            topk=int(config.topk),
            init_blocks=int(config.init_blocks),
            local_blocks=int(config.local_blocks),
            max_batch=int(max_batch),
            max_kv_len=int(max_kv_len),
        )


@dataclass
class M3DecodeState:
    """Persistent decode buffers and compiled kernels for one geometry.

    Data only: the kernel launches are the module-level `decode_*`
    functions below. The attention metadata builds this once (outside
    capture) and owns it, so every persistent buffer keeps a stable
    `data_ptr()` across CUDA graph replays. Per-call views are prefix or
    strided views of these buffers.
    """

    geom: M3DecodeGeometry
    device: torch.device
    pf_proxy: int
    heads_packed_proxy: int
    pf_sparse: int
    heads_packed_sparse: int
    max_k_tiles: int
    num_ctas: int
    proxy_module: object
    sparse_module: object
    workspace_buffer: torch.Tensor
    max_score_flat: torch.Tensor
    kv_block_indexes: torch.Tensor
    out: torch.Tensor
    kv_segment_offsets: torch.Tensor
    valid_pages: torch.Tensor
    qo_offset: torch.Tensor
    # Per-batch-size host constants, built once per shape (outside capture,
    # bounded by the distinct batch sizes seen: CUDA graph buckets + eager).
    qo_const_cache: dict = field(default_factory=dict)
    worklist_cache: dict = field(default_factory=dict)


def build_m3_decode_state(geometry: M3DecodeGeometry, device: torch.device) -> M3DecodeState:
    """Allocate the persistent decode buffers and compile the kernels.

    Runs outside any CUDA graph capture (from the metadata's `prepare()`),
    using the same JIT-compiled SM100 binaries as the MSA api path.
    """
    import fmha_sm100  # noqa: F401  (hard dependency of the decode kernels)
    from fmha_sm100.jit import _dlpack_dtype_code, get_fmha_variant

    g = geometry
    # Pack factors and packed head counts (decode: qo_len == 1).
    pf_proxy = _compute_pack_factor(1, g.num_index_heads, 1)
    pf_sparse = _compute_pack_factor(1, g.num_q_heads, g.num_kv_heads)
    max_k_tiles = _max_k_tiles_capacity(g.max_kv_len)
    bf16_code = _dlpack_dtype_code(torch.bfloat16)
    return M3DecodeState(
        geom=g,
        device=device,
        pf_proxy=pf_proxy,
        heads_packed_proxy=g.num_index_heads // pf_proxy,
        pf_sparse=pf_sparse,
        heads_packed_sparse=g.num_q_heads // pf_sparse,
        max_k_tiles=max_k_tiles,
        num_ctas=int(torch.cuda.get_device_properties(device).multi_processor_count),
        # Proxy: OnlyScore (sparse_mode=2). Sparse GQA: Sparse (sparse_mode=0).
        proxy_module=get_fmha_variant(
            bf16_code, _QO_TILE_SIZE, True, 2, g.page_size, False, pf_proxy
        ),
        sparse_module=get_fmha_variant(
            bf16_code, _QO_TILE_SIZE, True, 0, g.page_size, False, pf_sparse
        ),
        workspace_buffer=torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
        max_score_flat=torch.empty(
            g.num_index_heads * max_k_tiles * g.max_batch, dtype=torch.float32, device=device
        ),
        kv_block_indexes=torch.full(
            (g.max_batch, g.num_kv_heads, g.topk), -1, dtype=torch.int32, device=device
        ),
        out=torch.empty(
            g.max_batch, g.num_q_heads, g.head_dim, dtype=torch.bfloat16, device=device
        ),
        kv_segment_offsets=torch.zeros(g.max_batch + 1, dtype=torch.int32, device=device),
        valid_pages=torch.zeros(g.max_batch, dtype=torch.int32, device=device),
        qo_offset=torch.zeros(g.max_batch, dtype=torch.int32, device=device),
    )


def resolve_decode_state(
    m3_meta,
    geometry: M3DecodeGeometry,
    device: torch.device,
) -> M3DecodeState:
    """Return the decode state owning the CUDA-graph-stable buffers.

    The attention metadata's `prepare()` builds and attaches the state as
    `m3_meta.decode_state` (outside capture), so its buffers keep a stable
    `data_ptr()` across replays. The eager and test paths may reach here
    without one; there we build a per-call state and cache it on the
    metadata for reuse.
    """
    state = getattr(m3_meta, "decode_state", None)
    if state is not None and state.geom == geometry and state.device == device:
        return state
    state = build_m3_decode_state(geometry, device)
    try:
        m3_meta.decode_state = state
    except AttributeError:
        # Metadata forbids attribute assignment (e.g. a slotted/frozen
        # test double); the eager path still works with a per-call state.
        pass
    return state


# ---------------------------------------------------------------------------
# Cached per-batch-size constants (host work happens once per shape, outside
# any capture; callers warm shapes up before capturing).
# ---------------------------------------------------------------------------


def _qo_consts(
    state: M3DecodeState, batch: int, pack_factor: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(qo_segment_lens, qo_segment_offsets) for packed decode lens."""
    key = (batch, pack_factor)
    cached = state.qo_const_cache.get(key)
    if cached is None:
        lens = torch.full((batch,), pack_factor, dtype=torch.int32, device=state.device)
        offsets = (
            (torch.arange(batch + 1, dtype=torch.int64) * pack_factor)
            .to(torch.int32)
            .to(state.device)
        )
        cached = (lens, offsets)
        state.qo_const_cache[key] = cached
    return cached


def _worklist(
    state: M3DecodeState, batch: int, num_packed_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (batch, num_packed_heads)
    cached = state.worklist_cache.get(key)
    if cached is None:
        cached = build_decode_worklist(
            batch_size=batch,
            num_packed_heads=num_packed_heads,
            num_ctas=state.num_ctas,
            device=state.device,
        )
        state.worklist_cache[key] = cached
    return cached


def _kv_offsets_view(state: M3DecodeState, seq_lens: torch.Tensor, batch: int) -> torch.Tensor:
    """Cumulative KV lengths into the persistent buffer (device op)."""
    view = state.kv_segment_offsets[: batch + 1]
    torch.cumsum(seq_lens, 0, dtype=torch.int32, out=view[1:])
    return view


def _qo_offset_view(state: M3DecodeState, seq_lens: torch.Tensor, batch: int) -> torch.Tensor:
    """Per-request causal offset `kv_len - 1` (device op).

    The kernel's causal bound is inclusive (attend positions
    `<= offset + q_idx`); with one query token at position `kv_len - 1`
    this unmasks exactly the `kv_len` cached positions. `kv_len` itself
    would leak one stale slot from a partially-filled last page in sparse
    mode, which has no secondary seqlen clip.
    """
    view = state.qo_offset[:batch]
    torch.sub(seq_lens, 1, out=view)
    return view


# ---------------------------------------------------------------------------
# Proxy MQA pass (indexer): per-KV-block max scores
# ---------------------------------------------------------------------------


def decode_proxy_max_score(
    state: M3DecodeState,
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Dense MQA proxy pass, OnlyScore mode.

    Parameters
    ----------
    idx_q : `[batch, num_index_heads, 128]` bf16 (decode: 1 token/req).
    idx_k_paged : `[num_pages, 1, page_size, 128]` bf16 (HND).
    seq_lens : `[batch]` int32 device; per-request KV length.
    kv_page_indptr : `[batch + 1]` int32 device.
    kv_indices : `[total_pages]` int32 device page table.
    sm_scale : softmax scale (does not affect max-score ranking).

    Returns
    -------
    `[num_index_heads, max_k_tiles, batch]` fp32 view into the persistent
    max-score buffer; unwritten tiles are -inf.
    """
    g = state.geom
    batch = idx_q.shape[0]
    seq_lens = seq_lens[:batch]

    max_score = torch.as_strided(
        state.max_score_flat,
        (g.num_index_heads, state.max_k_tiles, batch),
        (state.max_k_tiles * batch, batch, 1),
    )
    max_score.fill_(float("-inf"))

    qo_lens, qo_offsets = _qo_consts(state, batch, state.pf_proxy)
    work_range, work_info = _worklist(state, batch, state.heads_packed_proxy)
    kv_offsets = _kv_offsets_view(state, seq_lens, batch)
    qo_offset = _qo_offset_view(state, seq_lens, batch)

    state.proxy_module.run(
        state.workspace_buffer,
        idx_q,
        idx_k_paged,
        idx_k_paged,
        qo_lens,
        seq_lens,
        qo_offsets,
        kv_offsets,
        work_range,
        work_info,
        None,  # out (OnlyScore)
        float(sm_scale),
        1.0,
        1.0,
        1.0,
        1.0,
        state.pf_proxy,  # max_qo_len after packing
        qo_offset,  # kv_len - 1: inclusive causal bound = last cached pos
        1,  # num_kv_splits
        None,
        None,
        None,  # kv_tile_begin / end / split
        None,
        None,  # workspace_o / workspace_lse
        None,  # num_kv_splits_per_row
        _QO_TILE_SIZE,
        kv_indices,
        kv_page_indptr,
        max_score,
        state.max_k_tiles,
        None,  # kv_block_indexes
        state.pf_proxy,
        True,  # qo_len_uniform
        torch.cuda.current_stream().cuda_stream,
    )
    return max_score


# ---------------------------------------------------------------------------
# Top-k block selection (device-driven)
# ---------------------------------------------------------------------------


def decode_select_blocks(
    state: M3DecodeState,
    max_score: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Group-reduce index-head scores to KV heads and pick top-k blocks.

    Returns `[batch, num_kv_heads, topk]` int32 view into the persistent
    `kv_block_indexes` buffer.
    """
    g = state.geom
    batch = max_score.shape[2]
    seq_lens = seq_lens[:batch]

    group = g.num_index_heads // g.num_kv_heads
    if group > 1:
        max_score_kv = max_score.view(g.num_kv_heads, group, max_score.shape[1], batch).amax(dim=1)
    else:
        max_score_kv = max_score

    valid_pages = state.valid_pages[:batch]
    torch.div(seq_lens + (g.page_size - 1), g.page_size, rounding_mode="floor", out=valid_pages)

    out = state.kv_block_indexes[:batch]
    return select_topk_blocks(
        max_score_kv,
        valid_pages,
        topk=g.topk,
        init_blocks=g.init_blocks,
        local_blocks=g.local_blocks,
        out=out,
    )


# ---------------------------------------------------------------------------
# Sparse block-GQA main pass
# ---------------------------------------------------------------------------


def decode_sparse_attention(
    state: M3DecodeState,
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Block-sparse paged GQA over the selected KV blocks.

    Parameters
    ----------
    q : `[batch, num_q_heads, 128]` bf16.
    k_paged / v_paged : `[num_pages, num_kv_heads, page_size, 128]` bf16.
    kv_block_indexes : `[batch, num_kv_heads, topk]` int32 ascending,
        -1 tail padded.
    seq_lens / kv_page_indptr / kv_indices : as in `decode_proxy_max_score`.

    Returns
    -------
    `[batch, num_q_heads, 128]` bf16 view into the persistent output buffer.
    """
    batch = q.shape[0]
    seq_lens = seq_lens[:batch]

    out = state.out[:batch]
    qo_lens, qo_offsets = _qo_consts(state, batch, state.pf_sparse)
    work_range, work_info = _worklist(state, batch, state.heads_packed_sparse)
    kv_offsets = _kv_offsets_view(state, seq_lens, batch)
    qo_offset = _qo_offset_view(state, seq_lens, batch)

    state.sparse_module.run(
        state.workspace_buffer,
        q,
        k_paged,
        v_paged,
        qo_lens,
        seq_lens,
        qo_offsets,
        kv_offsets,
        work_range,
        work_info,
        out,
        float(sm_scale),
        1.0,
        1.0,
        1.0,
        1.0,
        state.pf_sparse,  # max_qo_len after packing
        qo_offset,  # kv_len - 1 (see _qo_offset_view)
        1,  # num_kv_splits
        None,
        None,
        None,
        None,
        None,
        None,
        _QO_TILE_SIZE,
        kv_indices,
        kv_page_indptr,
        None,  # max_score
        -1,  # max_k_tiles
        kv_block_indexes,
        state.pf_sparse,
        True,  # qo_len_uniform
        torch.cuda.current_stream().cuda_stream,
    )
    return out


__all__ = [
    "M3DecodeGeometry",
    "M3DecodeState",
    "build_m3_decode_state",
    "resolve_decode_state",
    "decode_proxy_max_score",
    "decode_select_blocks",
    "decode_sparse_attention",
]
