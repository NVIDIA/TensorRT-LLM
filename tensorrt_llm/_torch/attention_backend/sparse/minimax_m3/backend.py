# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention backend.

MiniMax-M3 layers 3..N use a two-step block-sparse attention:

1. **Index attention** projects ``hidden_states`` through a small index
   branch:

     * ``index_q_proj``  produces a per-head Q vector of width
       ``sparse_index_dim`` for each of ``sparse_num_index_heads`` index
       heads, shape ``[num_tokens, num_idx_heads, sparse_index_dim]``.
     * ``index_k_proj``  produces a **single replicated** K vector of
       width ``sparse_index_dim`` per token, shape
       ``[num_tokens, 1, sparse_index_dim]``. The same K is broadcast
       across every index head during scoring.

   Each token's per-block score is the max (``score_type='max'``) of the
   per-position dot product against the cached index K within the block.
   The first ``init_blocks`` blocks always receive the highest priority,
   the last ``local_blocks`` blocks the next-highest; among the
   remaining blocks the top-``topk`` highest-scoring blocks are
   selected. When ``sparse_num_index_heads > num_kv_heads`` the
   per-index-head top-k indices are unioned to the kv-head granularity.

2. **Sparse GQA** runs standard GQA attention over the main K/V cache,
   but only over the selected blocks. The output is reshaped to match
   the dense path's ``[num_tokens, num_q_heads * head_dim]``.

The cache layout follows SGLang's MiniMax sparse backend and is
compatible with :class:`~tensorrt_llm._torch.pyexecutor.resource_manager.KVCacheManagerV2`:

    main K/V cache : ``[num_slots, num_kv_heads, head_dim]``
    index K cache  : ``[num_slots, 1, sparse_index_dim]``
    index V cache  : ``[num_slots, 1, sparse_index_dim]``  -- only when
                     ``disable_index_value=False``. The M3 checkpoint
                     sets ``disable_index_value=True`` on every sparse
                     layer, so for the bring-up the index V cache is
                     skipped entirely.

The ``req_to_token`` mapping is the same paged indirection
``KVCacheManagerV2`` produces: ``req_to_token[req_idx, pos]`` is the
slot index into ``k_cache`` (and the index caches) for that token.

CUDA-graph safety
-----------------

All scalar max lengths (``max_seqlen_q``, ``max_seqlen_k``) are
pre-computed CPU-side in :meth:`MiniMaxM3SparseAttentionMetadata.prepare`
and stored as plain Python ints. The hot path uses only batched
tensor ops with static shapes derived from those CPU-side scalars;
no ``.item()`` or other GPU-CPU sync runs inside the forward
functions. The algorithm captures cleanly under a CUDA graph and
replays bit-identical output.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .kernels import triton_block_max_score, triton_sparse_softmax
from .metadata import (
    MiniMaxM3SparseAttentionMetadata,
    MiniMaxM3SparseConfig,
    ensure_metadata_on_device,
    get_minimax_m3_attention_metadata_cls,
)

if TYPE_CHECKING:
    from .cache_manager import MiniMaxM3SparseIndexCache
    from .metadata import MiniMaxM3SparseParams

# Sentinel block score for blocks that init / local priority forces into
# the top-k regardless of their numerical score.
_INIT_SCORE = 1e30
_LOCAL_SCORE = 1e29


# ---------------------------------------------------------------------------
# Vectorized algorithm — CUDA-graph safe
# ---------------------------------------------------------------------------


def _gather_paged_batched(
    cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    max_k: int,
) -> torch.Tensor:
    """Gather ``[batch, max_k, *channel_dims]`` from a paged cache.

    Supports two cache layouts:

      * **3-D flat-slot** ``[num_slots, *channel_dims]``. Used by the
        side index caches (``idx_k_cache`` / ``idx_v_cache``), which the
        :class:`MiniMaxM3KVCacheManagerV2` allocates as plain
        ``torch.zeros((num_total_slots, 1, sparse_index_dim))`` tensors,
        and by focused unit tests that construct flat-slot test
        tensors directly.
      * **4-D paged multi-dim** ``[num_pages, tokens_per_block, *channel_dims]``.
        Used when the cache is a multi-dim view of the main K/V pool
        (``kv_pool[:, 0]`` for K, ``kv_pool[:, 1]`` for V). The slot id
        ``s`` is decomposed as ``page = s // tokens_per_block`` and
        ``within = s % tokens_per_block``, and the gather uses
        multi-dim fancy indexing to read directly from the pool. This
        path is required for correctness: the previously used
        ``kv_pool[:, 0].reshape(-1, num_kv_heads, head_dim)`` silently
        copies the K slice (the dim-0 stride is 2× the contiguous
        stride because dim 1 separates K from V), so the prior call's
        ``index_copy_`` writes never propagated back to ``kv_pool`` and
        the next forward read zeros. Routing through the multi-dim
        view via fancy indexing preserves shared storage.

    Out-of-range positions read slot 0 (their scores will be masked
    out by the caller via the ``seq_lens`` bound). No CPU sync.

    ``req_to_token`` and ``slot_ids`` must already live on the cache
    device — the production
    :class:`MiniMaxM3AttentionMetadata.prepare` builds them there, and
    test callers do the same. ``.to(dtype=torch.long)`` below is a
    same-device dtype conversion (capture-safe).
    """
    batch = int(slot_ids.shape[0])
    slot_ids_long = slot_ids.to(dtype=torch.long)
    # [batch, max_k] int64 slot indices into the cache.
    slot_rows = req_to_token.index_select(0, slot_ids_long)[:, :max_k]
    if cache.ndim >= 4:
        # Multi-dim paged: dim 0 is num_pages, dim 1 is tokens_per_block.
        tokens_per_block = int(cache.shape[1])
        slot_long = slot_rows.to(torch.long)
        page = slot_long // tokens_per_block
        within = slot_long % tokens_per_block
        # Multi-dim fancy indexing produces a new contiguous tensor
        # with the gathered values; result shape is
        # ``[batch, max_k, *cache.shape[2:]]``.
        return cache[page, within]
    flat = cache.index_select(0, slot_rows.reshape(-1).to(torch.long))
    return flat.view(batch, max_k, *cache.shape[1:])


def _assert_paged_write_in_bounds(
    name: str,
    cache: torch.Tensor,
    page: torch.Tensor,
    within: torch.Tensor,
) -> None:
    """Optional CPU-side bounds check for paged-cache writes.

    The runtime computes per-token slot ids from
    ``KVCacheManagerV2``'s block ids; if the runtime ever produces a
    block id that does not fit the per-layer view's dim-0 the write
    falls into another layer's coalesced memory and corrupts the
    cache, or fires the CUDA ``IndexKernel.cu`` device-side assert
    during fancy indexing. Both are far enough away from the root
    cause to be hard to triage.

    When ``TRTLLM_MINIMAX_M3_DEBUG_BOUNDS`` is set this check runs a
    CPU-side max/min comparison against the cache's dim-0 and dim-2
    bounds, surfacing the misindex with the exact tensor names and
    values instead of a device-side assert spam. It is opt-in
    because the comparison forces a CPU sync.
    """
    if not os.environ.get("TRTLLM_MINIMAX_M3_DEBUG_BOUNDS"):
        return
    num_pages = int(cache.shape[0])
    tokens_per_block = int(cache.shape[1]) if cache.ndim == 4 else int(cache.shape[2])
    page_max = int(page.max().item()) if page.numel() else -1
    page_min = int(page.min().item()) if page.numel() else 0
    within_max = int(within.max().item()) if within.numel() else -1
    within_min = int(within.min().item()) if within.numel() else 0
    assert 0 <= page_min and page_max < num_pages, (
        f"{name}: page index out of bounds — page.min={page_min} "
        f"page.max={page_max} but cache.shape[0]={num_pages} "
        f"(shape={tuple(cache.shape)}). This usually means the "
        f"runtime's get_block_ids_per_seq produced a block id wider "
        f"than the per-layer paged view's dim-0; check that the M3 "
        f"override path returns slot ids in [0, num_slots)."
    )
    assert 0 <= within_min and within_max < tokens_per_block, (
        f"{name}: within-page offset out of bounds — within.min="
        f"{within_min} within.max={within_max} but tokens_per_block="
        f"{tokens_per_block} (shape={tuple(cache.shape)}). This "
        f"usually means out_cache_loc was computed with a different "
        f"tokens_per_block than the cache was allocated with."
    )


def _write_main_kv_slots_to_pool(
    pool: torch.Tensor,
    kv_index: int,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write per-new-token K (``kv_index=0``) or V (``kv_index=1``) into ``pool``.

    ``pool`` is the 5-D main K/V pool returned by
    :meth:`KVCacheManagerV2.get_buffers` with the NHD layout
    ``[num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim]``.
    ``values`` has shape ``[num_new_tokens, num_kv_heads, head_dim]``
    and ``out_cache_loc`` is the 1-D ``[num_new_tokens]`` int tensor of
    flat slot ids the caller wants to update.

    The write decomposes each flat slot id into
    ``(page = s // tokens_per_block, within = s % tokens_per_block)``
    and uses multi-dim fancy-index assignment so the writes propagate
    to the underlying pool storage. The previously used pattern
    ``pool[:, kv_index].reshape(-1, num_kv_heads, head_dim)
    .index_copy_(0, ...)`` instead wrote into a silent copy (see
    :func:`_gather_paged_batched`), so the next forward call read
    zeros for the prefilled positions.

    The optional CPU-side bounds assertion (enabled when the
    ``TRTLLM_MINIMAX_M3_DEBUG_BOUNDS`` env var is set) catches
    block_ids overflowing the pool's dim-0 before the device-side
    ``IndexKernel.cu`` assert fires deep inside the kernel. The
    assertion is a CPU sync, so the env var keeps it opt-in for
    production runs that need a clean fast path.
    """
    tokens_per_block = int(pool.shape[2])
    out_long = out_cache_loc.to(torch.long)
    page = out_long // tokens_per_block
    within = out_long % tokens_per_block
    _assert_paged_write_in_bounds("pool", pool, page, within)
    # KV-cache writes never need to participate in autograd. Wrap the
    # fancy-index assignment in ``torch.no_grad()`` so callers that
    # enter this path with an active grad context (e.g. unit tests
    # exercising :class:`MiniMaxM3Attention` without ``inference_mode``)
    # do not trip the "leaf Variable that requires grad is being used
    # in an in-place operation" autograd guard on the view chain.
    with torch.no_grad():
        # Multi-dim fancy assignment writes into the underlying pool buffer.
        pool[page, kv_index, within] = values.to(pool.dtype)


def _write_main_kv_slots(
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Layout-aware writer for K (or V) caches used by the M3 backend.

    Supports two layouts, mirroring :func:`_gather_paged_batched`:

      * **3-D flat-slot** ``[num_slots, num_kv_heads, head_dim]``: used
        by focused unit tests that allocate the cache as a contiguous
        flat-slot tensor. ``index_copy_(0, ...)`` writes propagate
        because the tensor IS the storage.
      * **4-D multi-dim paged** ``[num_pages, tokens_per_block,
        num_kv_heads, head_dim]``: used when the cache is a view of
        ``kv_pool[:, 0]`` / ``kv_pool[:, 1]``. The view is
        non-contiguous (its dim-0 stride is 2× the contiguous stride
        because dim 1 separates K from V in the pool), so
        ``index_copy_(0, ...)`` would silently fork a copy and the
        write would be lost. Decompose the flat slot id into
        ``(page, within)`` and use multi-dim fancy assignment so the
        write propagates through the view to the underlying pool.
    """
    # KV-cache writes never need to participate in autograd. Wrap both
    # branches in ``torch.no_grad()`` so callers that enter this path
    # with an active grad context (e.g. unit tests exercising
    # :class:`MiniMaxM3Attention` without ``inference_mode``) do not
    # trip the autograd in-place guard on the cache view chain.
    with torch.no_grad():
        if cache.ndim >= 4:
            tokens_per_block = int(cache.shape[1])
            out_long = out_cache_loc.to(torch.long)
            page = out_long // tokens_per_block
            within = out_long % tokens_per_block
            _assert_paged_write_in_bounds("cache", cache, page, within)
            cache[page, within] = values.to(cache.dtype)
        else:
            cache.index_copy_(0, out_cache_loc.to(torch.long), values.to(cache.dtype))


def _scatter_topk_to_block_mask(
    topk_idx_per_head: torch.Tensor,  # [num_idx_heads, total_q, topk] int64, -1 padded
    *,
    num_kv_heads: int,
    num_blocks: int,
) -> torch.Tensor:
    """Build ``[num_kv_heads, total_q, num_blocks]`` selection mask.

    A block is selected for ``(kv_head, q_token)`` iff at least one of
    the ``idx_group_size`` index heads in that GQA group picked it.
    This is the vectorized, sync-free equivalent of ``topk_index_reduce``
    followed by membership check.
    """
    num_idx_heads, total_q, topk = topk_idx_per_head.shape
    idx_group_size = num_idx_heads // num_kv_heads
    # Per (idx_head, q) scatter into a bool [num_idx_heads, total_q, num_blocks].
    base = torch.zeros(
        num_idx_heads,
        total_q,
        num_blocks + 1,  # +1 for the -1 sentinel slot
        dtype=torch.bool,
        device=topk_idx_per_head.device,
    )
    # Map -1 -> num_blocks (the sentinel slot we drop).
    safe_idx = torch.where(
        topk_idx_per_head >= 0,
        topk_idx_per_head,
        torch.full_like(topk_idx_per_head, num_blocks),
    )
    ones = torch.ones_like(safe_idx, dtype=torch.bool)
    base.scatter_(dim=-1, index=safe_idx, src=ones)
    selected = base[..., :num_blocks]  # drop sentinel
    # Reduce idx_group dim with OR.
    selected = selected.view(num_kv_heads, idx_group_size, total_q, num_blocks)
    return selected.any(dim=1)  # [num_kv_heads, total_q, num_blocks]


# Per-chunk FP32 working-set budget for the sparse-GQA and index-attention
# per-Q K/V expansions. Picked so the per-chunk peak stays in low-hundreds
# of MiB at checkpoint-scale (num_kv_heads=4, head_dim=128, max_k≈2.7k).
# Override via the env var ``TRTLLM_MINIMAX_M3_SPARSE_CHUNK_BYTES`` for
# benchmarking.
_DEFAULT_SPARSE_CHUNK_BUDGET_BYTES = 512 * 1024 * 1024


def _sparse_chunk_budget_bytes() -> int:
    raw = os.environ.get("TRTLLM_MINIMAX_M3_SPARSE_CHUNK_BYTES")
    if not raw:
        return _DEFAULT_SPARSE_CHUNK_BUDGET_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_SPARSE_CHUNK_BUDGET_BYTES
    return max(1, value)


def _compute_sparse_gqa_chunk_q(
    total_q: int,
    max_k: int,
    num_kv_heads: int,
    head_dim: int,
    g: int,
    *,
    budget_bytes: Optional[int] = None,
) -> int:
    """Pick a chunk_q so the per-chunk FP32 K/V/QK slabs fit ``budget_bytes``.

    Bounds the peak working set of :func:`_sparse_gqa_masked`. The
    per-Q FP32 expansion of ``k_padded``/``v_padded`` is the dominant
    term — at checkpoint scale (num_kv_heads=4, head_dim=128,
    max_k≈2.7k) it costs ~5.5 MiB per Q row, so an unchunked
    total_q=2688 prefill would materialize ~14 GiB just for ``k_per_q``.
    Chunking caps the peak at ``chunk_q * bytes_per_q``.
    """
    if total_q <= 0 or max_k <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        return max(1, total_q)
    budget = budget_bytes if budget_bytes is not None else _sparse_chunk_budget_bytes()
    bytes_per_q = (
        max_k * num_kv_heads * head_dim * 4 * 2  # k_per_q + v_per_q FP32 slabs
        + num_kv_heads * max(1, g) * max_k * 4 * 2  # qk + attn FP32 slabs
    )
    if bytes_per_q <= 0:
        return total_q
    chunk = max(1, budget // bytes_per_q)
    return min(total_q, int(chunk))


def _compute_index_attn_chunk_q(
    total_q: int,
    max_k: int,
    num_idx_heads: int,
    sparse_index_dim: int,
    *,
    disable_index_value: bool,
    budget_bytes: Optional[int] = None,
) -> int:
    """Pick a chunk_q so the per-chunk FP32 index-K/V slab fits ``budget_bytes``.

    Bounds the peak working set of :func:`_index_attention_and_select`.
    The per-Q FP32 expansion of ``idx_k_padded`` (and ``idx_v_padded``
    when index-value is enabled) is the dominant intermediate.
    """
    if total_q <= 0 or max_k <= 0 or sparse_index_dim <= 0:
        return max(1, total_q)
    budget = budget_bytes if budget_bytes is not None else _sparse_chunk_budget_bytes()
    bytes_per_q = max_k * sparse_index_dim * 4  # idx_k_per_q FP32 slab
    if not disable_index_value:
        bytes_per_q += max_k * sparse_index_dim * 4  # idx_v_per_q FP32 slab
    bytes_per_q += num_idx_heads * max_k * 4  # qk FP32 slab
    if bytes_per_q <= 0:
        return total_q
    chunk = max(1, budget // bytes_per_q)
    return min(total_q, int(chunk))


def _index_attention_and_select(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, sparse_index_dim]
    idx_k_padded: torch.Tensor,  # [batch, max_k, 1, sparse_index_dim]
    idx_v_padded: Optional[torch.Tensor],  # [batch, max_k, 1, sparse_index_dim] or None
    seq_lens: torch.Tensor,
    q_batch_row: torch.Tensor,  # [total_q] int64
    q_positions: Optional[torch.Tensor],  # [total_q] int64, None for decode
    *,
    config: MiniMaxM3SparseConfig,
    max_k: int,
    disable_index_value: bool,
    idx_sm_scale: float,
    causal: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Vectorized index attention + per-block top-k selection.

    Returns ``(idx_o, block_mask)`` where ``block_mask`` has shape
    ``[num_kv_heads, total_q, num_blocks]`` boolean. ``idx_o`` is
    ``None`` when ``disable_index_value=True``.

    The per-Q FP32 expansion of ``idx_k_padded`` is computed in slabs
    of ``chunk_q`` rows so the peak working set scales with
    ``chunk_q * max_k * sparse_index_dim`` rather than
    ``total_q * max_k * sparse_index_dim``.
    """
    total_q = int(idx_q.shape[0])
    num_idx_heads = config.num_index_heads
    sparse_index_dim = config.sparse_index_dim
    num_kv_heads = config.num_kv_heads
    block_size = config.block_size

    # qk output buffer: [total_q, num_idx_heads, max_k] in FP32. This is
    # small compared to the per-Q K slab (e.g. 2688*4*2688*4 = 110 MiB at
    # checkpoint scale) so we allocate once and fill chunk-by-chunk.
    qk = torch.empty((total_q, num_idx_heads, max_k), dtype=torch.float32, device=idx_q.device)

    # Precompute per-Q valid-position masks (small, reused across chunks).
    arange_k = torch.arange(max_k, device=qk.device, dtype=torch.int64)
    seq_lens_per_q = seq_lens.index_select(0, q_batch_row).to(torch.int64)
    valid_mask = arange_k.unsqueeze(0) < seq_lens_per_q.unsqueeze(1)  # [total_q, max_k]
    if causal:
        assert q_positions is not None
        causal_mask = arange_k.unsqueeze(0) <= q_positions.unsqueeze(1)
        valid_mask = valid_mask & causal_mask

    idx_q_fp32_full = idx_q.to(torch.float32)

    if total_q > 0 and max_k > 0:
        chunk_q = _compute_index_attn_chunk_q(
            total_q,
            max_k,
            num_idx_heads,
            sparse_index_dim,
            disable_index_value=disable_index_value,
        )
        for start in range(0, total_q, chunk_q):
            end = min(start + chunk_q, total_q)
            q_batch_row_chunk = q_batch_row[start:end]
            # Per-Q K matrix for the chunk only.
            idx_k_per_q_chunk = (
                idx_k_padded.squeeze(2).to(torch.float32).index_select(0, q_batch_row_chunk)
            )  # [chunk, max_k, sparse_index_dim]
            qk_chunk = (
                torch.einsum(
                    "ihd,iqd->ihq",
                    idx_q_fp32_full[start:end],
                    idx_k_per_q_chunk,
                )
                * idx_sm_scale
            )
            qk[start:end] = qk_chunk
    # else: empty total_q or max_k — qk is the empty tensor and the
    # downstream code paths handle that as the degenerate case.

    qk = qk.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))

    # Per-block reduction via the OpenAI Triton kernel
    # ``_block_max_score_kernel``. The kernel operates on the unpadded
    # ``qk`` of shape ``[total_q, num_idx_heads, max_k]`` directly,
    # reading out-of-range positions as ``-inf`` so they cannot win the
    # max. Size-zero inputs are handled by the kernel wrapper itself.
    scores = triton_block_max_score(qk, block_size)
    n_blocks = scores.shape[-1]

    # Effective K length per Q token in tokens, in blocks.
    if causal:
        assert q_positions is not None
        eff_k = (q_positions + 1).clamp_max(seq_lens_per_q)
    else:
        eff_k = seq_lens_per_q
    n_valid_blocks = (eff_k + block_size - 1) // block_size  # [total_q]

    # Init / local priority via vectorized override.
    block_ids = torch.arange(n_blocks, device=qk.device, dtype=torch.int64)
    if config.init_blocks > 0:
        init_mask = block_ids < config.init_blocks
        # Broadcast: [n_blocks] -> [1, 1, n_blocks]
        scores = torch.where(
            init_mask.view(1, 1, -1),
            torch.full_like(scores, _INIT_SCORE),
            scores,
        )
    if config.local_blocks > 0:
        local_start = (n_valid_blocks - config.local_blocks).clamp_min(0)  # [total_q]
        # local region: local_start[i] <= b < n_valid_blocks[i]
        local_mask = (block_ids.view(1, -1) >= local_start.view(-1, 1)) & (
            block_ids.view(1, -1) < n_valid_blocks.view(-1, 1)
        )  # [total_q, n_blocks]
        scores = torch.where(
            local_mask.unsqueeze(1),  # broadcast over num_idx_heads
            torch.full_like(scores, _LOCAL_SCORE),
            scores,
        )

    # Mask invalid blocks (past n_valid_blocks) to -inf so they cannot be
    # selected even if init/local left them alone.
    block_valid = block_ids.view(1, -1) < n_valid_blocks.view(-1, 1)  # [total_q, n_blocks]
    scores = scores.masked_fill(~block_valid.unsqueeze(1), float("-inf"))

    # Top-k per (idx_head, q). When n_valid < topk the bottom slots are
    # -inf-scored and we mark them -1 in the output.
    k = min(config.topk, n_blocks)
    if k == 0:
        topk_idx_per_head = torch.full(
            (num_idx_heads, total_q, config.topk),
            -1,
            device=qk.device,
            dtype=torch.int64,
        )
    else:
        # scores: [total_q, num_idx_heads, n_blocks] -> permute -> [num_idx_heads, total_q, n_blocks]
        s = scores.permute(1, 0, 2)
        vals, idx = s.topk(k=k, dim=-1)
        # Any -inf-scored slot is invalid (no real block to select).
        idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
        if k < config.topk:
            pad_idx = torch.full(
                (num_idx_heads, total_q, config.topk - k),
                -1,
                device=qk.device,
                dtype=torch.int64,
            )
            topk_idx_per_head = torch.cat([idx, pad_idx], dim=-1)
        else:
            topk_idx_per_head = idx

    block_mask = _scatter_topk_to_block_mask(
        topk_idx_per_head,
        num_kv_heads=num_kv_heads,
        num_blocks=n_blocks,
    )

    # Optional index-attention output.
    if disable_index_value:
        idx_o: Optional[torch.Tensor] = None
    else:
        if idx_v_padded is None:
            raise RuntimeError("index V cache missing but disable_index_value=False")
        attn_full = qk.softmax(dim=-1, dtype=torch.float32)
        # attn was padded with -inf, so softmax weight there is 0; but
        # idx_v_per_q is only [..., max_k], so we trim attn back.
        attn_full = attn_full[..., :max_k]
        idx_o = torch.empty(
            (total_q, num_idx_heads * sparse_index_dim),
            dtype=idx_q.dtype,
            device=idx_q.device,
        )
        if total_q > 0 and max_k > 0:
            chunk_q = _compute_index_attn_chunk_q(
                total_q,
                max_k,
                num_idx_heads,
                sparse_index_dim,
                disable_index_value=False,
            )
            for start in range(0, total_q, chunk_q):
                end = min(start + chunk_q, total_q)
                q_batch_row_chunk = q_batch_row[start:end]
                idx_v_per_q_chunk = (
                    idx_v_padded.squeeze(2).to(torch.float32).index_select(0, q_batch_row_chunk)
                )  # [chunk, max_k, sparse_index_dim]
                idx_o_chunk = torch.einsum("ihq,iqd->ihd", attn_full[start:end], idx_v_per_q_chunk)
                idx_o[start:end] = idx_o_chunk.reshape(
                    end - start, num_idx_heads * sparse_index_dim
                ).to(idx_q.dtype)
    return idx_o, block_mask


def _sparse_gqa_masked(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_padded: torch.Tensor,  # [batch, max_k, num_kv_heads, head_dim]
    v_padded: torch.Tensor,  # [batch, max_k, num_kv_heads, head_dim]
    block_mask: torch.Tensor,  # [num_kv_heads, total_q, n_blocks] bool
    seq_lens: torch.Tensor,
    q_batch_row: torch.Tensor,  # [total_q] int64
    q_positions: Optional[torch.Tensor],
    *,
    config: MiniMaxM3SparseConfig,
    max_k: int,
    sm_scale: float,
    causal: bool,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized sparse GQA: mask non-selected blocks to -inf in QK.

    Functionally equivalent to gathering selected blocks and running
    standard attention over them, but uses static-shape tensor ops so
    the algorithm is CUDA-graph safe. ``num_q_heads / num_kv_heads`` Q
    heads share the same block_mask within a GQA group.

    Returns ``[total_q, num_q_heads * head_dim]``. The attention result
    accumulates in FP32; when ``output`` is supplied, the final dtype
    conversion writes directly into that tensor.

    The per-Q FP32 expansion of ``k_padded``/``v_padded`` is computed
    in slabs of ``chunk_q`` rows so the peak working set scales with
    ``chunk_q * max_k * num_kv_heads * head_dim`` rather than
    ``total_q * max_k * num_kv_heads * head_dim``. For decode
    (``total_q == batch``, typically 1) the loop runs once and the
    behavior is identical to the unchunked path.
    """
    total_q = int(q.shape[0])
    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    block_size = config.block_size
    g = num_q_heads // num_kv_heads

    # Reshape Q into (kv_head, group, head_dim) so we can broadcast against KV heads.
    q_grp = q.to(torch.float32).view(total_q, num_kv_heads, g, head_dim)

    # Build per-position mask from block mask (chunk-invariant; small).
    arange_k = torch.arange(max_k, device=q.device, dtype=torch.int64)
    pos_block = arange_k // block_size  # [max_k]
    block_mask_per_pos = block_mask.index_select(-1, pos_block)  # [num_kv_heads, total_q, max_k]
    block_mask_per_pos = block_mask_per_pos.permute(1, 0, 2)  # [total_q, num_kv_heads, max_k]

    seq_lens_per_q = seq_lens.index_select(0, q_batch_row).to(torch.int64)
    if causal:
        assert q_positions is not None
        eff_k = (q_positions + 1).clamp_max(seq_lens_per_q)
    else:
        eff_k = seq_lens_per_q
    valid_pos = arange_k.unsqueeze(0) < eff_k.unsqueeze(1)  # [total_q, max_k]
    valid_pos_per_kv = valid_pos.unsqueeze(1)  # [total_q, 1, max_k]
    attended = block_mask_per_pos & valid_pos_per_kv  # [total_q, num_kv_heads, max_k]
    row_has_any = attended.any(dim=-1, keepdim=True)  # [total_q, num_kv_heads, 1]

    # Preserve FP32 accumulation until the final cast. When a preallocated
    # output is supplied this avoids allocating a q.dtype intermediate and
    # then copying that intermediate into the custom op's output tensor.
    o = torch.empty((total_q, num_kv_heads, g, head_dim), dtype=torch.float32, device=q.device)

    if total_q > 0 and max_k > 0:
        chunk_q = _compute_sparse_gqa_chunk_q(total_q, max_k, num_kv_heads, head_dim, g)

        for start in range(0, total_q, chunk_q):
            end = min(start + chunk_q, total_q)
            q_batch_row_chunk = q_batch_row[start:end]
            # Per-Q K/V matrices for the chunk only — the dominant FP32 slab.
            k_per_q_chunk = k_padded.to(torch.float32).index_select(
                0, q_batch_row_chunk
            )  # [chunk, max_k, num_kv_heads, head_dim]
            v_per_q_chunk = v_padded.to(torch.float32).index_select(0, q_batch_row_chunk)
            q_grp_chunk = q_grp[start:end]
            attended_chunk = attended[start:end]
            row_has_any_chunk = row_has_any[start:end]

            # qk: [chunk, num_kv_heads, g, max_k]
            qk_chunk = torch.einsum("ihgd,iqhd->ihgq", q_grp_chunk, k_per_q_chunk) * sm_scale
            qk_chunk = qk_chunk.masked_fill(~attended_chunk.unsqueeze(2), float("-inf"))

            # Apply the mask + softmax via the OpenAI Triton kernel
            # ``_sparse_softmax_kernel`` which honors the per-position
            # ``attended`` mask, computes softmax in fp32, and folds in the
            # all-False-row fix so captured graphs never produce NaN.
            attn_chunk = triton_sparse_softmax(qk_chunk, attended_chunk)

            # o_chunk: [chunk, num_kv_heads, g, head_dim]
            o_chunk = torch.einsum("ihgq,iqhd->ihgd", attn_chunk, v_per_q_chunk)
            # Zero out rows that had no valid positions.
            keep_chunk = row_has_any_chunk.squeeze(-1)  # [chunk, num_kv_heads]
            o_chunk = o_chunk * keep_chunk.unsqueeze(-1).unsqueeze(-1)
            o[start:end] = o_chunk

    o_flat = o.view(total_q, num_q_heads * head_dim)
    if output is None:
        return o_flat.to(q.dtype)
    expected_shape = (total_q, num_q_heads * head_dim)
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"output must have shape {expected_shape}, got {tuple(output.shape)}")
    output.copy_(o_flat)
    return output


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def minimax_m3_sparse_decode(
    q: torch.Tensor,
    idx_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    metadata: MiniMaxM3SparseAttentionMetadata,
    config: MiniMaxM3SparseConfig,
    *,
    disable_index_value: bool,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """MiniMax-M3 sparse decode (CUDA-graph safe).

    Inputs:
        q:           ``[batch_size, num_q_heads, head_dim]``
        idx_q:       ``[batch_size, num_idx_heads, sparse_index_dim]``
        k_cache:     ``[num_slots, num_kv_heads, head_dim]`` (already
                     contains the newly-decoded token at the appropriate
                     slot)
        v_cache:     ``[num_slots, num_kv_heads, head_dim]``
        idx_k_cache: ``[num_slots, 1, sparse_index_dim]``
        idx_v_cache: ``[num_slots, 1, sparse_index_dim]`` or ``None``
                     (must be ``None`` when ``disable_index_value=True``)
        metadata:    forward metadata; ``is_prefill`` must be ``False``
        config:      layer-invariant sparse configuration

    Returns ``(idx_o, o)``:
        idx_o: ``None`` when ``disable_index_value=True``; otherwise
               ``[batch_size, num_idx_heads * sparse_index_dim]``.
        o:     ``[batch_size, num_q_heads * head_dim]``.
    """
    if metadata.is_prefill:
        raise ValueError("decode entry called with prefill metadata")
    if disable_index_value and idx_v_cache is not None:
        raise ValueError("disable_index_value=True must be paired with idx_v_cache=None")
    if not disable_index_value and idx_v_cache is None:
        raise ValueError("disable_index_value=False requires idx_v_cache to be allocated")

    max_k = int(metadata.max_seqlen_k)
    sm_scale = sm_scale if sm_scale is not None else config.head_dim**-0.5
    idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5

    # Gather per-batch padded caches once. Out-of-range slots read slot 0
    # and are masked out by the seq_lens-derived valid mask.
    idx_k_padded = _gather_paged_batched(
        idx_k_cache, metadata.req_to_token, metadata.slot_ids, max_k
    )  # [batch, max_k, 1, sparse_index_dim]
    idx_v_padded: Optional[torch.Tensor] = None
    if not disable_index_value:
        idx_v_padded = _gather_paged_batched(
            idx_v_cache, metadata.req_to_token, metadata.slot_ids, max_k
        )
    k_padded = _gather_paged_batched(
        k_cache, metadata.req_to_token, metadata.slot_ids, max_k
    )  # [batch, max_k, num_kv_heads, head_dim]
    v_padded = _gather_paged_batched(v_cache, metadata.req_to_token, metadata.slot_ids, max_k)

    # Decode: total_q == batch, q_batch_row is identity, no q_positions.
    batch = int(metadata.slot_ids.shape[0])
    q_batch_row = torch.arange(batch, device=q.device, dtype=torch.int64)

    idx_o, block_mask = _index_attention_and_select(
        idx_q,
        idx_k_padded,
        idx_v_padded,
        metadata.seq_lens,
        q_batch_row,
        None,
        config=config,
        max_k=max_k,
        disable_index_value=disable_index_value,
        idx_sm_scale=idx_sm_scale,
        causal=False,
    )
    o = _sparse_gqa_masked(
        q,
        k_padded,
        v_padded,
        block_mask,
        metadata.seq_lens,
        q_batch_row,
        None,
        config=config,
        max_k=max_k,
        sm_scale=sm_scale,
        causal=False,
        output=output,
    )
    return idx_o, o


def minimax_m3_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    metadata: MiniMaxM3SparseAttentionMetadata,
    config: MiniMaxM3SparseConfig,
    *,
    disable_index_value: bool,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """MiniMax-M3 sparse prefill / chunked-extend (CUDA-graph safe).

    Inputs:
        q:           ``[total_q_tokens, num_q_heads, head_dim]``
        idx_q:       ``[total_q_tokens, num_idx_heads, sparse_index_dim]``
        k_cache:     paged main cache (slots already populated for the
                     prefix plus current chunk)
        v_cache:     same
        idx_k_cache: paged index-K cache (slots populated)
        idx_v_cache: paged index-V cache (None when disable_index_value)
        metadata:    prefill metadata; must have ``cu_seqlens_q``,
                     ``prefix_lens``, ``extend_seq_lens_cpu``, and the
                     ``prepare``-populated ``q_batch_row`` /
                     ``q_positions`` fields.
        config:      layer-invariant sparse configuration

    Returns ``(idx_o, o)``:
        idx_o: ``None`` when ``disable_index_value=True``; otherwise
               ``[total_q_tokens, num_idx_heads * sparse_index_dim]``.
        o:     ``[total_q_tokens, num_q_heads * head_dim]``.
    """
    if not metadata.is_prefill:
        raise ValueError("prefill entry called with decode metadata")
    if metadata.cu_seqlens_q is None or metadata.prefix_lens is None:
        raise ValueError("prefill metadata requires cu_seqlens_q and prefix_lens")
    if metadata.q_batch_row is None or metadata.q_positions is None:
        raise ValueError(
            "prefill metadata is missing q_batch_row / q_positions; "
            "did you call metadata.prepare()?"
        )
    if disable_index_value and idx_v_cache is not None:
        raise ValueError("disable_index_value=True must be paired with idx_v_cache=None")
    if not disable_index_value and idx_v_cache is None:
        raise ValueError("disable_index_value=False requires idx_v_cache to be allocated")

    max_k = int(metadata.max_seqlen_k)
    sm_scale = sm_scale if sm_scale is not None else config.head_dim**-0.5
    idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5

    idx_k_padded = _gather_paged_batched(
        idx_k_cache, metadata.req_to_token, metadata.slot_ids, max_k
    )
    idx_v_padded: Optional[torch.Tensor] = None
    if not disable_index_value:
        idx_v_padded = _gather_paged_batched(
            idx_v_cache, metadata.req_to_token, metadata.slot_ids, max_k
        )
    k_padded = _gather_paged_batched(k_cache, metadata.req_to_token, metadata.slot_ids, max_k)
    v_padded = _gather_paged_batched(v_cache, metadata.req_to_token, metadata.slot_ids, max_k)

    q_batch_row = metadata.q_batch_row.to(torch.int64)
    q_positions = metadata.q_positions.to(torch.int64)

    idx_o, block_mask = _index_attention_and_select(
        idx_q,
        idx_k_padded,
        idx_v_padded,
        metadata.seq_lens,
        q_batch_row,
        q_positions,
        config=config,
        max_k=max_k,
        disable_index_value=disable_index_value,
        idx_sm_scale=idx_sm_scale,
        causal=True,
    )
    o = _sparse_gqa_masked(
        q,
        k_padded,
        v_padded,
        block_mask,
        metadata.seq_lens,
        q_batch_row,
        q_positions,
        config=config,
        max_k=max_k,
        sm_scale=sm_scale,
        causal=True,
        output=output,
    )
    return idx_o, o


# ---------------------------------------------------------------------------
# Attention algorithm wrapper
# ---------------------------------------------------------------------------


# Lazy import alias to avoid a circular import at module load — the side
# cache class lives in ``cache_manager`` which imports from this module
# is fine (no cycle), but keeping the indirection explicit makes the
# dependency direction in this file clearer.
def _import_index_cache_cls():
    from .cache_manager import MiniMaxM3SparseIndexCache

    return MiniMaxM3SparseIndexCache


@dataclass
class MiniMaxM3SparseAttention:
    """Thin orchestrator for :func:`minimax_m3_sparse_prefill` and
    :func:`minimax_m3_sparse_decode`.

    Owns the sparse configuration plus an optional reference to a
    :class:`MiniMaxM3SparseIndexCache`. The caller is responsible for
    routing the projected Q, K, V, ``idx_q``, ``idx_k`` (and optional
    ``idx_v``) tensors plus the populated
    :class:`MiniMaxM3SparseAttentionMetadata`.
    """

    config: MiniMaxM3SparseConfig
    index_cache: Optional["MiniMaxM3SparseIndexCache"] = None

    def write_caches(
        self,
        layer_idx: int,
        out_cache_loc: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ) -> None:
        """Append newly-computed K/V (and index K/V) to the paged caches."""
        if self.index_cache is None:
            raise RuntimeError("index_cache is None; cannot write index branch")
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} vs {v.shape}")
        if k.dim() != 3:
            raise ValueError(f"K/V must be [num_tokens, num_kv_heads, head_dim], got {k.shape}")
        k_cache.index_copy_(0, out_cache_loc.to(torch.long), k.to(k_cache.dtype))
        v_cache.index_copy_(0, out_cache_loc.to(torch.long), v.to(v_cache.dtype))
        self.index_cache.set_index_k(layer_idx, out_cache_loc, idx_k)
        if idx_v is not None:
            self.index_cache.set_index_v(layer_idx, out_cache_loc, idx_v)

    def forward(
        self,
        layer_idx: int,
        q: torch.Tensor,
        idx_q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: MiniMaxM3SparseAttentionMetadata,
        *,
        disable_index_value: bool,
        sm_scale: Optional[float] = None,
        idx_sm_scale: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Dispatch to :func:`minimax_m3_sparse_prefill` or
        :func:`minimax_m3_sparse_decode` based on metadata."""
        if self.index_cache is None:
            raise RuntimeError("index_cache must be allocated before forward()")
        idx_k_cache = self.index_cache.get_index_k_buffer(layer_idx)
        idx_v_cache = self.index_cache.get_index_v_buffer(layer_idx)
        if disable_index_value and idx_v_cache is not None:
            raise RuntimeError(
                f"layer {layer_idx} disable_index_value=True but index V "
                "cache is allocated; cache configuration is inconsistent"
            )
        if not disable_index_value and idx_v_cache is None:
            raise RuntimeError(
                f"layer {layer_idx} disable_index_value=False but index V cache is not allocated"
            )
        if metadata.is_prefill:
            return minimax_m3_sparse_prefill(
                q,
                k_cache,
                v_cache,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                metadata,
                self.config,
                disable_index_value=disable_index_value,
                sm_scale=sm_scale,
                idx_sm_scale=idx_sm_scale,
            )
        return minimax_m3_sparse_decode(
            q,
            idx_q,
            k_cache,
            v_cache,
            idx_k_cache,
            idx_v_cache,
            metadata,
            self.config,
            disable_index_value=disable_index_value,
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
        )


# ---------------------------------------------------------------------------
# Runtime integration: AttentionBackend wrapper
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_minimax_m3_attention_backend_cls():
    """Return :class:`MiniMaxM3SparseRuntimeBackend` (lazy import).

    Deferring the :class:`AttentionBackend` import keeps the algorithm
    module usable from test paths that do not need the runtime backend.
    """
    from ...interface import (
        AttentionBackend,
        AttentionForwardArgs,
        AttentionMetadata,
        merge_attention_forward_args,
    )

    metadata_cls = get_minimax_m3_attention_metadata_cls()

    class MiniMaxM3SparseRuntimeBackend(AttentionBackend[AttentionMetadata]):
        """:class:`AttentionBackend` for MiniMax-M3 sparse layers.

        Constructed under the standard ``create_attention(...)`` dispatch
        when ``SparseAttentionConfig(algorithm='minimax_m3', ...)`` is
        configured. Drives the MiniMax-M3 sparse algorithm directly via
        :func:`minimax_m3_sparse_prefill` and
        :func:`minimax_m3_sparse_decode`.

        The standard :class:`AttentionForwardArgs` surface does not
        carry ``idx_q`` / ``idx_k`` slots, so the model layer threads
        the index branch through ``**kwargs`` of :meth:`forward`. When
        ``forward`` is called without ``idx_q`` it raises
        :class:`NotImplementedError` with a pointer at the model layer
        — the backend's ``forward`` is **executable**, but it is not a
        substitute for the MiniMax-specific projection / norm / RoPE
        steps the model layer is responsible for.

        The backend exposes
        :meth:`forward_sparse` for callers that want a name-explicit
        entry point (the model layer calls it directly); :meth:`forward`
        is the standard contract entry point and routes to
        :meth:`forward_sparse` when ``idx_q`` is supplied.
        """

        Metadata = metadata_cls

        def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config=None,
            sparse_params: Optional["MiniMaxM3SparseParams"] = None,
            **kwargs,
        ):
            if sparse_params is None:
                raise ValueError("sparse_params is required for MiniMaxM3SparseRuntimeBackend")
            super().__init__(
                layer_idx,
                num_heads,
                head_dim,
                num_kv_heads=num_kv_heads,
                quant_config=quant_config,
                sparse_params=sparse_params,
                **kwargs,
            )
            self.m3_config = MiniMaxM3SparseConfig.from_sparse_params(
                sparse_params,
                num_q_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                head_dim=head_dim,
            )
            self.disable_index_value = bool(sparse_params.disable_index_value)

        @staticmethod
        def support_fused_rope() -> bool:
            # The MiniMax-M3 model layer applies RoPE explicitly because
            # both the main and the index branches need partial RoPE,
            # and the standard fused-RoPE attention op does not have a
            # hook for the index branch.
            return False

        def forward_sparse(
            self,
            *,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            idx_q: torch.Tensor,
            idx_k: torch.Tensor,
            idx_v: Optional[torch.Tensor],
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            idx_k_cache: torch.Tensor,
            idx_v_cache: Optional[torch.Tensor],
            out_cache_loc: torch.Tensor,
            m3_metadata: "MiniMaxM3SparseAttentionMetadata",
            sm_scale: Optional[float] = None,
            idx_sm_scale: Optional[float] = None,
            output: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Execute the MiniMax-M3 sparse path end-to-end.

            Inputs:
                ``q``, ``k``, ``v``     : new-token projections, already
                                          per-head norm + RoPE applied.
                ``idx_q``, ``idx_k``    : index-branch projections, already
                                          per-head norm + RoPE applied.
                ``idx_v``               : index-V projection (only when
                                          ``disable_index_value=False``).
                ``k_cache``, ``v_cache``: flat-slot view of the paged
                                          main K/V cache,
                                          ``[num_slots, num_kv_heads, head_dim]``.
                ``idx_k_cache``         : side index-K cache,
                                          ``[num_slots, 1, sparse_index_dim]``.
                ``idx_v_cache``         : side index-V cache (or ``None``).
                ``out_cache_loc``       : ``[num_new_tokens]`` int slot
                                          indices to write the new
                                          token's K/V/idx_K to.
                ``m3_metadata``         : populated
                                          :class:`MiniMaxM3SparseAttentionMetadata`.
                ``output``              : optional preallocated final output,
                                          ``[num_tokens, num_q_heads * head_dim]``.

            Returns ``[num_tokens, num_q_heads * head_dim]``.
            """
            num_kv_heads = self.m3_config.num_kv_heads
            head_dim = self.m3_config.head_dim
            sparse_index_dim = self.m3_config.sparse_index_dim
            num_idx_heads = self.m3_config.num_index_heads

            num_tokens = int(q.shape[0])
            q_view = q.view(num_tokens, self.num_heads, head_dim)
            k_view = k.view(num_tokens, num_kv_heads, head_dim)
            v_view = v.view(num_tokens, num_kv_heads, head_dim)
            idx_q_view = idx_q.view(num_tokens, num_idx_heads, sparse_index_dim)
            idx_k_view = idx_k.view(num_tokens, 1, sparse_index_dim)

            # Production paths build the M3 metadata on the cache
            # device in ``MiniMaxM3AttentionMetadata.prepare`` (called
            # outside any CUDA-graph capture window), and test paths
            # construct it directly on the desired device. So all
            # metadata tensors should already live on ``k_cache.device``
            # by this point. We keep a same-device pass for resilience
            # against legacy test callers that produce metadata on a
            # different device, but it must not introduce CPU->GPU
            # copies inside the capture window. ``ensure_metadata_on_device``
            # is a no-op when each tensor is already on the target
            # device; under capture that no-op path is the contract.
            cache_device = k_cache.device
            if any(
                t is not None and t.device != cache_device
                for t in (
                    m3_metadata.req_to_token,
                    m3_metadata.slot_ids,
                    m3_metadata.seq_lens,
                    m3_metadata.prefix_lens,
                    m3_metadata.cu_seqlens_q,
                    m3_metadata.q_batch_row,
                    m3_metadata.q_positions,
                )
            ):
                m3_metadata = ensure_metadata_on_device(m3_metadata, cache_device)

            # Write new K/V/idx_K to the configured slots.
            # ``out_cache_loc`` comes from the pre-built attachment so
            # it already lives on the cache device. The write goes
            # through :func:`_write_main_kv_slots`, which is layout-
            # aware:
            #
            #   * 4-D multi-dim paged caches (the production V2 path:
            #     main K/V is ``kv_pool[:, 0]`` / ``kv_pool[:, 1]``, and
            #     ``idx_k_cache`` is the V2 4-D paged view ``[num_pages,
            #     tokens_per_block, 1, sparse_index_dim]``): decomposes
            #     each slot id into
            #     ``(page, within)`` and uses multi-dim fancy
            #     assignment so the write propagates to the underlying
            #     pool. A plain ``index_copy_(0, ...)`` would either
            #     silently fork a copy (non-contiguous main K/V view)
            #     or raise a shape mismatch (4-D index-K view), but
            #     the layout-aware helper sidesteps both failure
            #     modes.
            #   * 3-D flat-slot caches (focused unit tests that
            #     allocate plain ``torch.zeros((num_slots, num_heads,
            #     channel))`` tensors): falls back to
            #     ``index_copy_(0, ...)`` because the tensor IS the
            #     storage.
            _write_main_kv_slots(k_cache, out_cache_loc, k_view)
            _write_main_kv_slots(v_cache, out_cache_loc, v_view)
            _write_main_kv_slots(idx_k_cache, out_cache_loc, idx_k_view)
            if idx_v is not None and idx_v_cache is not None:
                idx_v_view = idx_v.view(num_tokens, 1, sparse_index_dim)
                _write_main_kv_slots(idx_v_cache, out_cache_loc, idx_v_view)

            if m3_metadata.is_prefill:
                _, o = minimax_m3_sparse_prefill(
                    q_view,
                    k_cache,
                    v_cache,
                    idx_q_view,
                    idx_k_cache,
                    None if self.disable_index_value else idx_v_cache,
                    m3_metadata,
                    self.m3_config,
                    disable_index_value=self.disable_index_value,
                    sm_scale=sm_scale,
                    idx_sm_scale=idx_sm_scale,
                    output=output,
                )
            else:
                _, o = minimax_m3_sparse_decode(
                    q_view,
                    idx_q_view,
                    k_cache,
                    v_cache,
                    idx_k_cache,
                    None if self.disable_index_value else idx_v_cache,
                    m3_metadata,
                    self.m3_config,
                    disable_index_value=self.disable_index_value,
                    sm_scale=sm_scale,
                    idx_sm_scale=idx_sm_scale,
                    output=output,
                )
            return o

        def forward(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            v: Optional[torch.Tensor],
            metadata=None,
            forward_args: Optional[AttentionForwardArgs] = None,
            *,
            output: Optional[torch.Tensor] = None,
            idx_q: Optional[torch.Tensor] = None,
            idx_k: Optional[torch.Tensor] = None,
            idx_v: Optional[torch.Tensor] = None,
            k_cache: Optional[torch.Tensor] = None,
            v_cache: Optional[torch.Tensor] = None,
            idx_k_cache: Optional[torch.Tensor] = None,
            idx_v_cache: Optional[torch.Tensor] = None,
            out_cache_loc: Optional[torch.Tensor] = None,
            m3_metadata: Optional["MiniMaxM3SparseAttentionMetadata"] = None,
            sm_scale: Optional[float] = None,
            idx_sm_scale: Optional[float] = None,
            **kwargs,
        ) -> torch.Tensor:
            """Standard ``AttentionBackend.forward`` entry point.

            The MiniMax-M3 sparse path needs the index branch projection
            and the M3-shaped metadata; both arrive through keyword
            arguments because the standard
            :class:`AttentionForwardArgs` surface does not carry them.

            When ``idx_q`` is omitted, this method raises
            :class:`NotImplementedError` to make the misuse loud — a
            generic AttentionBackend dispatch site cannot drive this
            backend without supplying the index branch.
            """
            forward_args = merge_attention_forward_args(forward_args, kwargs)
            if (
                output is not None
                and forward_args.output is not None
                and output is not forward_args.output
            ):
                raise ValueError("output was supplied both directly and through forward_args")
            if output is None:
                output = forward_args.output
            if idx_q is None or idx_k is None or m3_metadata is None:
                raise NotImplementedError(
                    f"MiniMaxM3SparseRuntimeBackend.forward (layer "
                    f"{self.layer_idx}) requires the M3 index branch and "
                    "metadata to be passed as keyword arguments "
                    "(`idx_q`, `idx_k`, `m3_metadata`, "
                    "`out_cache_loc`, `k_cache`, `v_cache`, "
                    "`idx_k_cache`). The standard AttentionForwardArgs "
                    "surface does not carry them; the model layer "
                    "(`MiniMaxM3Attention.forward`) supplies them when "
                    "calling this backend."
                )
            if (
                k is None
                or v is None
                or k_cache is None
                or v_cache is None
                or idx_k_cache is None
                or out_cache_loc is None
            ):
                raise ValueError(
                    "MiniMaxM3SparseRuntimeBackend.forward requires k, v, "
                    "k_cache, v_cache, idx_k_cache, and out_cache_loc to "
                    "be supplied alongside idx_q / idx_k / m3_metadata."
                )
            return self.forward_sparse(
                q=q,
                k=k,
                v=v,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
                k_cache=k_cache,
                v_cache=v_cache,
                idx_k_cache=idx_k_cache,
                idx_v_cache=idx_v_cache,
                out_cache_loc=out_cache_loc,
                m3_metadata=m3_metadata,
                sm_scale=sm_scale,
                idx_sm_scale=idx_sm_scale,
                output=output,
            )

    return MiniMaxM3SparseRuntimeBackend


__all__ = [
    "MiniMaxM3SparseAttention",
    "get_minimax_m3_attention_backend_cls",
    "minimax_m3_sparse_decode",
    "minimax_m3_sparse_prefill",
]
