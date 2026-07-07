# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, backend-neutral helpers for MiniMax-M3 sparse attention.

Hosts the pure-torch primitives that both the Triton reference backend
and the MSA (`fmha_sm100`) backend rely on, so the MSA path does not
import Triton-backend internals:

  * KV-slot writers (paged and flat-slot aware).
  * Init/local block-priority sentinels.
  * MSA cache-layout adapters (pool NHD to `fmha_sm100` HND).
  * Per-query valid-block counting and torch top-k block selection.
  * The lazy `fmha_sm100` import guard and kernel precondition constants.

None of these functions require CUDA graph capture windows to be closed:
they either operate on device tensors with static shapes on the forward
hot path, or run once per scheduler step during `prepare()`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .metadata import MiniMaxM3SparseAttentionMetadata


# Sentinel block score for blocks that init / local priority forces into
# the top-k regardless of their numerical score.
_INIT_SCORE = 1e30
_LOCAL_SCORE = 1e29

# fmha_sm100 only ships head_dim=128 variants, and the in-tree
# block-selection/driver geometry is validated for topk=16 (the
# MiniMax-M3 checkpoint value). Callers enforce these preconditions early
# so layer construction fails with a clear message rather than a cryptic
# shape error from inside the MSA JIT.
_MSA_REQUIRED_TOPK = 16
_MSA_REQUIRED_HEAD_DIM = 128


def require_msa_module():
    """Lazy-import `fmha_sm100` and raise a clear error on failure.

    Returns the imported module. The import is guarded so the MSA backend
    can be advertised in the config schema even where MSA is not
    installed; the error only fires when a sparse layer actually
    dispatches the MSA path.
    """
    try:
        import fmha_sm100  # noqa: F401
    except ImportError as exc:  # pragma: no cover - install-time error
        raise RuntimeError(
            "MiniMax-M3 MSA backend requires the external `fmha_sm100` "
            "package (MSA: https://github.com/MiniMax-AI/MSA; not on "
            "PyPI). Install it with `pip install "
            "'git+https://github.com/MiniMax-AI/MSA.git'`, or unset "
            "`sparse_use_msa` in the sparse attention config to fall "
            "back to the Triton reference path."
        ) from exc
    return fmha_sm100


# ---------------------------------------------------------------------------
# Paged / flat-slot KV writers
# ---------------------------------------------------------------------------


def _assert_paged_write_in_bounds(
    name: str,
    cache: torch.Tensor,
    page: torch.Tensor,
    within: torch.Tensor,
) -> None:
    """Optional CPU-side bounds check for paged-cache writes.

    The runtime computes per-token slot ids from `KVCacheManagerV2`'s
    block ids. If it ever produces a block id that does not fit the
    per-layer view's dim-0, the write falls into another layer's memory
    and corrupts the cache, or fires the CUDA device-side assert during
    fancy indexing.

    When `TRTLLM_MINIMAX_M3_DEBUG_BOUNDS` is set, this runs a CPU-side
    max/min comparison that surfaces the misindex with exact tensor names
    and values instead of a device-side assert. It is opt-in because the
    comparison forces a CPU sync.
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
        f"{name}: page index out of bounds: page.min={page_min} "
        f"page.max={page_max} but cache.shape[0]={num_pages} "
        f"(shape={tuple(cache.shape)})."
    )
    assert 0 <= within_min and within_max < tokens_per_block, (
        f"{name}: within-page offset out of bounds: within.min="
        f"{within_min} within.max={within_max} but tokens_per_block="
        f"{tokens_per_block} (shape={tuple(cache.shape)})."
    )


def write_main_kv_slots(
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Layout-aware writer for K (or V / index-K) caches.

    Supports two layouts:

      * 3-D flat-slot `[num_slots, num_heads, channel]`: index_copy_
        writes propagate because the tensor is the storage (unit tests).
      * 4-D paged `[num_pages, tokens_per_block, num_heads, channel]`: a
        view of `kv_pool[:, 0]` / `kv_pool[:, 1]` (or the paged index-K
        view). The view is non-contiguous, so index_copy_ would silently
        fork a copy and lose the write; instead decompose the flat slot id
        into (page, within) and use multi-dim fancy assignment so the
        write propagates to the pool.
    """
    # KV-cache writes never need autograd; wrap in no_grad so callers with
    # an active grad context (unit tests without inference_mode) do not
    # trip the in-place autograd guard on the cache view chain.
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


# ---------------------------------------------------------------------------
# MSA cache-layout adapters (pool NHD -> fmha_sm100 HND)
# ---------------------------------------------------------------------------
#
# TRT-LLM's KVCacheManagerV2 stores the main K/V cache as a 5-D pool
# [num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim] (NHD).
# fmha_sm100 expects paged K/V tensors with layout
# [num_pages, num_kv_heads, page_size, head_dim] (HND). The side index-K
# cache is [num_slots, 1, sparse_index_dim] (flat) or the 4-D paged
# variant [num_pages, tokens_per_block, 1, sparse_index_dim].
#
# The helpers below produce MSA-compatible views without mutating the
# pool's storage; they permute and call .contiguous() once per call.


def cache_view_to_msa_paged(cache_view: torch.Tensor) -> torch.Tensor:
    """Convert `[num_pages, page_size, num_kv_heads, head_dim]` to HND.

    A flat-slot 3-D cache (unit tests) is treated as a single virtual
    page of size `num_slots`, giving `[1, num_kv_heads, num_slots,
    head_dim]`.
    """
    if cache_view.dim() == 4:
        return cache_view.permute(0, 2, 1, 3).contiguous()
    if cache_view.dim() == 3:
        return cache_view.permute(1, 0, 2).unsqueeze(0).contiguous()
    raise ValueError(
        f"Unsupported cache view rank {cache_view.dim()} for MSA paged conversion; "
        f"expected 3 (flat-slot) or 4 (paged multi-dim)."
    )


def idx_cache_to_msa_paged(idx_cache: torch.Tensor) -> torch.Tensor:
    """Convert the side index-K cache to MSA's HND paged layout.

    Accepts:
      * 3-D `[num_slots, 1, sparse_index_dim]` to
        `[1, 1, num_slots, sparse_index_dim]`
      * 4-D `[num_pages, tokens_per_block, 1, sparse_index_dim]` to
        `[num_pages, 1, tokens_per_block, sparse_index_dim]`
    """
    if idx_cache.dim() == 4:
        return idx_cache.permute(0, 2, 1, 3).contiguous()
    if idx_cache.dim() == 3:
        return idx_cache.permute(1, 0, 2).unsqueeze(0).contiguous()
    raise ValueError(f"Unsupported index cache rank {idx_cache.dim()} for MSA paged conversion.")


def page_size_from_view(cache_view: torch.Tensor) -> int:
    if cache_view.dim() == 4:
        return int(cache_view.shape[1])
    if cache_view.dim() == 3:
        return int(cache_view.shape[0])
    raise ValueError(f"Unsupported cache view rank {cache_view.dim()} for page-size lookup.")


def build_kv_indices_and_lens(
    metadata: "MiniMaxM3SparseAttentionMetadata",
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build `kv_indices` and `kv_segment_lens` for `fmha_sm100`.

    `kv_indices` is the flattened per-request page table
    `concat([pages_of(seq_0), pages_of(seq_1), ...])` int32; a sequence's
    pages come from `req_to_token[slot, ::page_size] // page_size`.
    `kv_segment_lens` is the per-request effective KV length (already on
    the cache device inside `metadata`).
    """
    device = metadata.req_to_token.device
    slot_ids_long = metadata.slot_ids.to(torch.long)
    req_rows = metadata.req_to_token.index_select(0, slot_ids_long).to(torch.long)
    batch = int(req_rows.shape[0])
    seq_lens_cpu = metadata.seq_lens_cpu.to(torch.long).tolist()

    page_lists = []
    for b in range(batch):
        kv_len = int(seq_lens_cpu[b])
        if kv_len <= 0:
            continue
        num_pages = (kv_len + page_size - 1) // page_size
        # First slot of each page gives the global page id into the paged
        # cache. Do NOT clamp to a per-request bound: block ids are
        # global and non-contiguous in production, so clamping collapses
        # the page table and corrupts the proxy FMHA reads.
        page_starts = torch.arange(num_pages, device=device, dtype=torch.long) * page_size
        page_ids = req_rows[b].gather(0, page_starts) // page_size
        page_lists.append(page_ids.to(torch.int32))

    if page_lists:
        kv_indices = torch.cat(page_lists, dim=0)
    else:
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)

    return kv_indices, metadata.seq_lens.to(torch.int32)


# ---------------------------------------------------------------------------
# Block selection (per-query valid-block masking + torch top-k)
# ---------------------------------------------------------------------------


def per_token_valid_blocks(
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    *,
    causal: bool,
    block_size: int,
) -> torch.Tensor:
    """Per-query number of valid KV blocks (causal-aware), on CPU.

    Expands per-request lens/offsets to a per-token `[total_q]` tensor so
    block selection can honour each query token's own causal extent.
    """
    qo = qo_lens_cpu.to(torch.long)
    kv = kv_lens_cpu.to(torch.long)
    batch = int(qo.shape[0])
    total = int(qo.sum().item())
    if total == 0:
        return torch.zeros(0, dtype=torch.long)
    batch_row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), qo)
    starts = torch.zeros(batch, dtype=torch.long)
    if batch > 1:
        starts[1:] = torch.cumsum(qo, 0)[:-1]
    intra = torch.arange(total, dtype=torch.long) - starts[batch_row]
    kv_per = kv[batch_row]
    if causal:
        if qo_offset_cpu is not None:
            off = qo_offset_cpu.to(torch.long)[batch_row]
        else:
            off = (kv - qo)[batch_row]
        eff = torch.minimum(off + intra + 1, kv_per)
    else:
        eff = kv_per
    return (eff + block_size - 1) // block_size


def select_blocks_from_maxscore(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Per-query block selection from per-KV-head block scores, in torch.

    Mirrors the reference selection (init/local forced blocks, per-query
    valid-block masking, top-k) on the amax-reduced per-KV-head scores
    `[num_kv_heads, n_blocks, total_q]`. Returns `[total_q, num_kv_heads,
    topk]` int32: ascending block indices with -1 tail padding.
    """
    num_kv_heads, n_blocks, total_q = max_score_kv.shape
    device = max_score_kv.device
    scores = max_score_kv.permute(2, 0, 1).to(torch.float32).clone()  # [q, kv, blk]
    block_ids = torch.arange(n_blocks, device=device, dtype=torch.long)
    nvb = n_valid_blocks.to(device=device, dtype=torch.long)  # [total_q]

    if init_blocks > 0:
        init_mask = block_ids.view(1, 1, -1) < init_blocks
        scores = torch.where(init_mask, torch.full_like(scores, _INIT_SCORE), scores)
    if local_blocks > 0:
        local_start = (nvb - local_blocks).clamp_min(0)  # [total_q]
        local_mask = (block_ids.view(1, -1) >= local_start.view(-1, 1)) & (
            block_ids.view(1, -1) < nvb.view(-1, 1)
        )  # [total_q, n_blocks]
        scores = torch.where(local_mask.unsqueeze(1), torch.full_like(scores, _LOCAL_SCORE), scores)
    # Per-query replacement for the kernel's scalar num_valid_pages clamp.
    block_valid = block_ids.view(1, -1) < nvb.view(-1, 1)  # [total_q, n_blocks]
    scores = scores.masked_fill(~block_valid.unsqueeze(1), float("-inf"))

    k = min(topk, n_blocks)
    vals, idx = scores.topk(k=k, dim=-1)  # [total_q, kv, k]
    idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
    sort_key = torch.where(idx < 0, torch.full_like(idx, n_blocks), idx)
    sort_key, _ = torch.sort(sort_key, dim=-1)
    idx = torch.where(sort_key >= n_blocks, torch.full_like(sort_key, -1), sort_key)
    if k < topk:
        pad = torch.full((total_q, num_kv_heads, topk - k), -1, dtype=idx.dtype, device=device)
        idx = torch.cat([idx, pad], dim=-1)
    return idx.to(torch.int32)


__all__ = [
    "_INIT_SCORE",
    "_LOCAL_SCORE",
    "_MSA_REQUIRED_HEAD_DIM",
    "_MSA_REQUIRED_TOPK",
    "build_kv_indices_and_lens",
    "cache_view_to_msa_paged",
    "idx_cache_to_msa_paged",
    "page_size_from_view",
    "per_token_valid_blocks",
    "require_msa_module",
    "select_blocks_from_maxscore",
    "write_main_kv_slots",
]
