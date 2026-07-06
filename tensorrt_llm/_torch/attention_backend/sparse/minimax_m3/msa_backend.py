# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed FMHA runtime for MiniMax-M3 sparse attention.

Implements the same two-step sparse attention as
:mod:`tensorrt_llm._torch.attention_backend.sparse.minimax_m3.backend`,
but lowers the per-block max score, top-k selection, and sparse GQA to
the external ``fmha_sm100`` kernels (a.k.a. MSA, MiniMax Sparse
Attention; see https://github.com/MiniMax-AI/MSA) instead of the
in-tree Triton + SDPA reference path.

The flow per forward call is:

  1. **Index proxy pass.** Run a dense MQA FMHA over the index branch
     with ``output_maxscore=True`` and ``num_kv_heads=1``. ``fmha_sm100``
     returns ``(None, max_score)`` where ``max_score`` has shape
     ``[num_index_heads, max_k_tiles, total_qo_len]``.
  2. **Block selection.** Reduce ``max_score`` to KV-head granularity
     and select top-k blocks per query (ascending indices, ``-1``
     padded) with per-query valid-block masking and forced init/local
     blocks.
  3. **Sparse GQA.** Run a second ``fmha_sm100`` over the main K/V
     branch, passing the block indices via ``kv_block_indexes``.

Prefill runs this flow eagerly through the MSA plan/run API; pure
decode runs it through the CUDA-graph-safe in-tree driver
(:mod:`.decode_wrapper`).

This module deliberately keeps the cache-layout adapter explicit so
the MSA backend is selectable per-layer without disturbing the
existing Triton path. The MSA package is imported lazily; if it is
absent the backend raises a descriptive error at construction time.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .backend import (
    _INIT_SCORE,
    _LOCAL_SCORE,
    _write_main_kv_slots,
    get_minimax_m3_attention_backend_cls,
)
from .metadata import (
    MiniMaxM3SparseAttentionMetadata,
    MiniMaxM3SparseConfig,
    ensure_metadata_on_device,
)

if TYPE_CHECKING:
    from .metadata import MiniMaxM3SparseParams


# ``fmha_sm100`` only ships head_dim=128 variants, and the in-tree
# block-selection/driver geometry is validated for topk=16 (the
# MiniMax-M3 checkpoint value). Enforce these preconditions early so
# layer construction fails with a clear message rather than a cryptic
# shape error from inside the MSA JIT.
_MSA_REQUIRED_TOPK = 16
_MSA_REQUIRED_HEAD_DIM = 128


def _require_msa_module():
    """Lazy-import ``fmha_sm100`` and raise a clear error on failure.

    Returns the imported module. The import is guarded so that the
    MSA backend can be advertised in the config schema even on hosts
    where MSA is not installed; the error only fires when a sparse
    layer actually attempts to dispatch the MSA path.
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
# Cache layout adapters
# ---------------------------------------------------------------------------
#
# TRT-LLM's :class:`KVCacheManagerV2` stores the main K/V cache as a 5-D
# pool ``[num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim]``
# with NHD ordering. ``fmha_sm100`` expects paged K/V tensors with
# layout ``[num_pages, num_kv_heads, page_size, head_dim]`` (HND).
# Sparse layers also carry a side index-K cache whose flat-slot layout
# is ``[num_slots, 1, sparse_index_dim]`` or the 4-D paged variant
# ``[num_pages, tokens_per_block, 1, sparse_index_dim]``.
#
# The helpers below produce MSA-compatible views without mutating the
# pool's underlying storage. They permute and call ``.contiguous()``
# once per call; that is the per-call cost we accept until the cache
# manager grows a native HND view.


def _cache_view_to_msa_paged(cache_view: torch.Tensor) -> torch.Tensor:
    """Convert ``[num_pages, page_size, num_kv_heads, head_dim]`` -> HND.

    The 4-D ``cache_view`` is the non-contiguous slice ``kv_pool[:, k]``
    (``k=0`` for K, ``k=1`` for V) returned by ``KVCacheManagerV2``.
    MSA's ``fmha_sm100`` expects ``[num_pages, num_kv_heads, page_size,
    head_dim]``; we permute dims 1 and 2 and force a contiguous copy
    so the kernel reads sequential memory.

    For flat-slot 3-D caches (focused unit tests) the function
    interprets the cache as a single virtual page of size ``num_slots``
    and returns ``[1, num_kv_heads, num_slots, head_dim]``. That keeps
    the test surface working without forcing every test to provide a
    real paged cache.
    """
    if cache_view.dim() == 4:
        return cache_view.permute(0, 2, 1, 3).contiguous()
    if cache_view.dim() == 3:
        # Flat-slot fallback. Treat the cache as a single page so the
        # MSA kernel's page-table indexing still resolves correctly.
        return cache_view.permute(1, 0, 2).unsqueeze(0).contiguous()
    raise ValueError(
        f"Unsupported cache view rank {cache_view.dim()} for MSA paged conversion; "
        f"expected 3 (flat-slot) or 4 (paged multi-dim)."
    )


def _idx_cache_to_msa_paged(idx_cache: torch.Tensor) -> torch.Tensor:
    """Convert the side index-K cache to MSA's HND paged layout.

    Accepts:
      * 3-D ``[num_slots, 1, sparse_index_dim]``  -> ``[1, 1, num_slots, sparse_index_dim]``
      * 4-D ``[num_pages, tokens_per_block, 1, sparse_index_dim]``
        -> ``[num_pages, 1, tokens_per_block, sparse_index_dim]``
    """
    if idx_cache.dim() == 4:
        return idx_cache.permute(0, 2, 1, 3).contiguous()
    if idx_cache.dim() == 3:
        return idx_cache.permute(1, 0, 2).unsqueeze(0).contiguous()
    raise ValueError(f"Unsupported index cache rank {idx_cache.dim()} for MSA paged conversion.")


def _page_size_from_view(cache_view: torch.Tensor) -> int:
    if cache_view.dim() == 4:
        return int(cache_view.shape[1])
    if cache_view.dim() == 3:
        return int(cache_view.shape[0])
    raise ValueError(f"Unsupported cache view rank {cache_view.dim()} for page-size lookup.")


def _build_kv_indices_and_lens(
    metadata: MiniMaxM3SparseAttentionMetadata,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``kv_indices`` and ``kv_segment_lens`` for ``fmha_sm100``.

    ``kv_indices`` is the flattened per-request page table:
    ``concat([pages_of(seq_0), pages_of(seq_1), ...])`` with dtype
    int32. The pages of a sequence come from
    ``metadata.req_to_token[slot_id, ::page_size] // page_size`` — i.e.
    the page index of every block start. ``kv_segment_lens`` carries
    the per-request effective KV length (already on the cache device
    inside ``metadata``).

    The helper assumes ``page_size == metadata.req_to_token.stride[-1]``
    block geometry, which is the contract :class:`MiniMaxM3KVCacheManagerV2`
    enforces (``tokens_per_block == sparse_block_size``).
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
        # First slot of each page gives the *global* page id into the paged
        # cache (each block is a contiguous run of page_size slots, see
        # KVCacheManagerV2). ``req_rows[b]`` already holds valid slot ids,
        # so ``// page_size`` yields valid global page ids by construction.
        #
        # Do NOT clamp these to a per-request bound: ``max_kv_len //
        # page_size`` is the per-request page count, not a global page-id
        # bound. Clamping page ids to ``max_page - 1`` collapses the page
        # table for every request whose pages exceed that count (i.e. every
        # request after the first in a contiguous layout, and virtually all
        # requests in production where block ids are global and
        # non-contiguous), making the proxy FMHA read the wrong K/V and
        # corrupting the block scores.  (Ported from
        # brb/feat/minimax_m3_mxfp8_msa commit 677bcb45e5.)
        page_starts = torch.arange(num_pages, device=device, dtype=torch.long) * page_size
        # ``page_starts`` is bounded by ``(num_pages - 1) * page_size < kv_len``
        # so it never over-reads; no clamp needed on the read index either.
        page_ids = req_rows[b].gather(0, page_starts) // page_size
        page_lists.append(page_ids.to(torch.int32))

    if page_lists:
        kv_indices = torch.cat(page_lists, dim=0)
    else:
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)

    return kv_indices, metadata.seq_lens.to(torch.int32)


# ---------------------------------------------------------------------------
# Forward primitives
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _select_proxy_fmha_class():
    """Pick the first available :class:`IndexerProxyFmha` from the registry.

    Walks :func:`get_enabled_fmha_lib_classes` and returns the first
    class that (a) is a subclass of :class:`IndexerProxyFmha` and (b)
    reports :meth:`is_available` ``True``. Returns ``None`` if no proxy
    backend is available; callers raise a descriptive error pointing
    at the ``TLLM_FMHA_LIBS`` env var in that case.

    The result is cached because the registry contents only change at
    process start (driven by the ``TLLM_FMHA_LIBS`` env var, also
    cached).
    """
    from ...fmha import IndexerProxyFmha, get_enabled_fmha_lib_classes

    for cls in get_enabled_fmha_lib_classes():
        if not issubclass(cls, IndexerProxyFmha):
            continue
        if cls.is_available():
            return cls
    return None


def _per_token_valid_blocks(
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    *,
    causal: bool,
    block_size: int,
) -> torch.Tensor:
    """Per-query number of valid KV blocks (causal-aware), on CPU.

    Ported from brb/feat/minimax_m3_mxfp8_msa commit 92d8405af5.
    Expands per-request lens/offsets to a per-*token* ``[total_q]``
    tensor so block selection can honour each query token's own causal
    extent — which ``sparse_topk_select``'s scalar ``num_valid_pages`` /
    ``force_end_blocks`` cannot express.
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


def _select_blocks_from_maxscore(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Per-query block selection from per-KV-head block scores, in torch.

    Ported from brb/feat/minimax_m3_mxfp8_msa commit 92d8405af5.
    Mirrors the reference ``backend._index_attention_and_select``
    selection (init/local forced blocks + per-query valid-block masking
    + top-k) on the ``amax``-reduced per-KV-head scores
    (``[num_kv_heads, n_blocks, total_q]``).  Replaces
    ``fmha_sm100.sparse_topk_select``, whose scalar ``num_valid_pages``
    / forced-block windows are batch-wide and therefore wrong for every
    query shorter than the batch-longest.

    Returns ``[total_q, num_kv_heads, topk]`` int32, ascending block
    indices with ``-1`` tail padding (unchanged downstream contract).
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


def _msa_index_proxy_and_topk(
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    *,
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    config: MiniMaxM3SparseConfig,
    idx_sm_scale: float,
    causal: bool,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Run a proxy FMHA + top-k block selection.

    The proxy attention is delegated to an :class:`IndexerProxyFmha`
    implementation looked up via the standard FMHA registry (defaults
    to :class:`MsaProxyMqaFmha` when MSA is installed). The
    group-reduction and ``sparse_topk_select`` step remain here because
    they are MiniMax-M3-specific algorithmic choices, not generic
    proxy concerns.

    Inputs
    ------
    idx_q : ``[total_q, num_index_heads, sparse_index_dim]``, bf16/fp16.
    idx_k_paged : ``[num_pages, 1, page_size, sparse_index_dim]``
        Index K cache in MSA's HND paged layout. The dense proxy pass
        runs MQA (``num_kv_heads_dense=1``) so the single replicated
        index-K head is consumed directly.
    qo_lens_cpu, kv_lens_cpu : ``[batch]``, int32, on CPU.
    qo_offset_cpu : optional ``[batch]`` int32 on CPU. Causal offset
        per request; ``None`` defaults to ``kv_lens - qo_lens``.
    kv_indices : ``[sum_pages]`` int32 on the cache device.
    config : layer-invariant sparse config.
    idx_sm_scale : softmax scale for the index attention.
    causal : whether to apply causal masking (True for prefill, False
        for pure-decode batches).
    init_blocks, local_blocks : forced-include block counts.

    Returns
    -------
    torch.Tensor
        ``[total_q, num_kv_heads, topk]`` int32 ascending top-k block
        indices with ``-1`` padding (MSA kernel contract). When
        ``num_index_heads > num_kv_heads`` the per-block max score is
        reduced across each KV head's index-head group before
        ``sparse_topk_select``, mirroring the
        ``score_type='max'`` reduction the reference path performs.
    """
    proxy_cls = _select_proxy_fmha_class()
    if proxy_cls is None:
        raise RuntimeError(
            "No IndexerProxyFmha backend is available. The MiniMax-M3 MSA "
            "backend needs an indexer-style proxy FMHA enabled via "
            "TLLM_FMHA_LIBS (e.g. 'msa_proxy_mqa') and the corresponding "
            "external dependency installed (e.g. the `fmha_sm100` package "
            "for `msa_proxy_mqa`). Set TLLM_FMHA_LIBS=msa_proxy_mqa or "
            "leave TLLM_FMHA_LIBS at its default to enable all registered "
            "backends."
        )

    proxy = proxy_cls()
    max_score = proxy.forward_proxy(
        idx_q,
        idx_k_paged,
        qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu,
        kv_indices=kv_indices,
        sm_scale=idx_sm_scale,
        causal=causal,
    )

    # ``max_score`` has shape ``[num_index_heads, max_k_tiles, total_q]``.
    # Reduce across each KV head's index-head group via amax so the
    # downstream ``sparse_topk_select`` picks blocks per KV head, which
    # is exactly the granularity ``fmha_sm100``'s sparse FMHA expects.
    if config.num_index_heads % config.num_kv_heads != 0:
        raise ValueError(
            f"num_index_heads ({config.num_index_heads}) must be divisible by "
            f"num_kv_heads ({config.num_kv_heads}) for MSA group-max reduction."
        )
    group = config.num_index_heads // config.num_kv_heads
    if group > 1:
        max_score_kv = max_score.view(
            config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
        ).amax(dim=1)
    else:
        max_score_kv = max_score

    # Per-query valid-block counts + torch selection (replaces
    # ``fmha_sm100.sparse_topk_select``, whose scalar num_valid_pages /
    # forced windows are batch-wide — wrong for heterogeneous batches
    # and for prefill where each token has its own causal extent).
    page_size = int(idx_k_paged.shape[2])
    n_valid_blocks = _per_token_valid_blocks(
        qo_lens_cpu,
        kv_lens_cpu,
        qo_offset_cpu,
        causal=causal,
        block_size=page_size,
    )
    if n_valid_blocks.numel() == 0 or int(n_valid_blocks.max().item()) <= 0:
        # Degenerate batch (no KV) — return all-padded indices.
        return torch.full(
            (idx_q.shape[0], config.num_kv_heads, _MSA_REQUIRED_TOPK),
            -1,
            dtype=torch.int32,
            device=idx_q.device,
        )

    return _select_blocks_from_maxscore(
        max_score_kv,
        topk=_MSA_REQUIRED_TOPK,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


@functools.lru_cache(maxsize=1)
def _select_block_sparse_fmha_class():
    """Pick the first available :class:`BlockSparseFmha` from the registry.

    Mirrors :func:`_select_proxy_fmha_class` for the main attention
    pass: walks :func:`get_enabled_fmha_lib_classes` looking for any
    class that (a) is a subclass of :class:`BlockSparseFmha` and (b)
    reports :meth:`is_available` ``True``. Returns ``None`` if no
    block-sparse backend is available; callers raise a descriptive
    error pointing at the ``TLLM_FMHA_LIBS`` env var.
    """
    from ...fmha import BlockSparseFmha, get_enabled_fmha_lib_classes

    for cls in get_enabled_fmha_lib_classes():
        if not issubclass(cls, BlockSparseFmha):
            continue
        if cls.is_available():
            return cls
    return None


def _msa_sparse_attention(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    *,
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run block-sparse paged GQA over the selected blocks.

    The kernel call is delegated to a :class:`BlockSparseFmha`
    implementation looked up via the standard FMHA registry (defaults
    to :class:`MsaSparseGqaFmha` when MSA is installed).
    ``num_q_heads`` / ``num_kv_heads`` / ``page_size`` are derived from
    the tensor shapes inside the backend so callers do not need to
    pass them explicitly.

    Returns ``[total_q, num_q_heads, head_dim]`` bfloat16.
    """
    sparse_cls = _select_block_sparse_fmha_class()
    if sparse_cls is None:
        raise RuntimeError(
            "No BlockSparseFmha backend is available. The MiniMax-M3 MSA "
            "backend needs a block-sparse FMHA enabled via TLLM_FMHA_LIBS "
            "(e.g. 'msa_sparse_gqa') and the corresponding external "
            "dependency installed (e.g. the `fmha_sm100` package for "
            "`msa_sparse_gqa`). Set TLLM_FMHA_LIBS=msa_sparse_gqa or "
            "leave TLLM_FMHA_LIBS at its default to enable all "
            "registered backends."
        )

    sparse = sparse_cls()
    return sparse.forward_block_sparse(
        q,
        k_paged,
        v_paged,
        kv_block_indexes,
        qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        causal=causal,
    )


# ---------------------------------------------------------------------------
# In-tree graph-safe decode driver
# ---------------------------------------------------------------------------


def _intree_sparse_decode(
    q: torch.Tensor,
    idx_q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    idx_k_paged: torch.Tensor,
    metadata: MiniMaxM3SparseAttentionMetadata,
    config: MiniMaxM3SparseConfig,
    *,
    sm_scale: float,
    idx_sm_scale: float,
    page_size: int,
) -> torch.Tensor:
    """Pure-decode path through the in-tree graph-safe driver.

    Replaces the MSA ``fmha_sm100_plan`` / ``fmha_sm100`` /
    ``sparse_topk_select`` host driver with
    :mod:`.decode_wrapper.dispatch` while running the same
    JIT-compiled kernel binaries.  Everything per-step-varying is read
    from device tensors, so this function is CUDA-graph-capturable and
    every replay tracks the current ``seq_lens`` / page tables (the
    exact property the MSA host driver lacks).
    """
    from .decode_wrapper.dispatch import M3DecodeGeometry, get_decode_driver

    batch = int(q.shape[0])
    seq_lens = metadata.seq_lens.to(torch.int32)

    msa_plans = getattr(metadata, "msa_plans", None)
    if msa_plans is not None:
        kv_indices = msa_plans["kv_indices"]
        kv_page_indptr = msa_plans["kv_page_indptr"]
        max_batch = int(msa_plans.get("max_batch") or 0)
        max_kv_len = int(msa_plans.get("max_kv_len") or 0)
    else:
        # Eager fallback when prepare() did not pre-stage the page
        # table (e.g. focused unit tests). Host-side work is fine here,
        # but it must never run inside a capture — fail loudly instead
        # of letting the unpinned H2D copy below produce a cryptic
        # capture error (or worse, silently freeze stale values).
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MiniMax-M3 in-tree decode reached the eager fallback during "
                "CUDA graph capture: metadata.msa_plans was not pre-staged by "
                "prepare(). This means the MSA geometry was not registered "
                "before capture (see msa_plan_cache.set_global_msa_geometry)."
            )
        kv_indices, _ = _build_kv_indices_and_lens(metadata, page_size)
        num_pages_cpu = (metadata.seq_lens_cpu.to(torch.long) + page_size - 1) // page_size
        kv_page_indptr = torch.zeros(batch + 1, dtype=torch.int32)
        kv_page_indptr[1:] = num_pages_cpu.to(torch.int32).cumsum(0)
        kv_page_indptr = kv_page_indptr.to(q.device, non_blocking=True)
        max_batch = 0
        max_kv_len = 0

    if max_batch <= 0:
        # Stable power-of-two capacity so the driver cache key does not
        # churn as eager batch sizes vary.
        max_batch = max(64, 1 << (batch - 1).bit_length())
    if max_kv_len <= 0:
        max_kv_len = int(metadata.req_to_token.shape[1])

    geometry = M3DecodeGeometry(
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        num_index_heads=config.num_index_heads,
        head_dim=config.head_dim,
        page_size=page_size,
        topk=config.topk,
        init_blocks=config.init_blocks,
        local_blocks=config.local_blocks,
        max_batch=max_batch,
        max_kv_len=max_kv_len,
    )
    driver = get_decode_driver(geometry, q.device)

    max_score = driver.proxy_max_score(
        idx_q,
        idx_k_paged,
        seq_lens=seq_lens,
        kv_page_indptr=kv_page_indptr,
        kv_indices=kv_indices,
        sm_scale=idx_sm_scale,
    )
    kv_block_indexes = driver.select_blocks(max_score, seq_lens=seq_lens)
    out = driver.sparse_attention(
        q,
        k_paged,
        v_paged,
        kv_block_indexes,
        seq_lens=seq_lens,
        kv_page_indptr=kv_page_indptr,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
    )
    # ``out`` aliases the driver's persistent buffer; the caller
    # consumes it immediately (o_proj input) before the next layer's
    # dispatch overwrites it, which is stream-ordered and safe.
    return out.reshape(batch, config.num_q_heads * config.head_dim)


# ---------------------------------------------------------------------------
# Public forward entry points
# ---------------------------------------------------------------------------


def _qo_lens_offsets_from_metadata(
    metadata: MiniMaxM3SparseAttentionMetadata,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract CPU qo_lens / kv_lens / qo_offset tensors.

    Returns ``(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu)`` all on CPU
    with dtype int32. ``qo_offset_cpu`` is ``None`` for pure-decode
    batches (causal=False) and is the per-request prefix length for
    prefill batches.
    """
    seq_lens_cpu = metadata.seq_lens_cpu.to(torch.int32)
    if metadata.is_prefill:
        if metadata.extend_seq_lens_cpu is None:
            raise RuntimeError("prefill metadata requires extend_seq_lens_cpu")
        qo_lens_cpu = torch.tensor(metadata.extend_seq_lens_cpu, dtype=torch.int32)
        # ``prefix_lens`` is the causal offset.
        if metadata.prefix_lens is None:
            raise RuntimeError("prefill metadata requires prefix_lens")
        qo_offset_cpu = metadata.prefix_lens.detach().to(device="cpu", dtype=torch.int32)
    else:
        batch = int(metadata.slot_ids.shape[0])
        qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
        qo_offset_cpu = (seq_lens_cpu - 1).to(torch.int32)
    return qo_lens_cpu, seq_lens_cpu, qo_offset_cpu


def minimax_m3_msa_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    metadata: MiniMaxM3SparseAttentionMetadata,
    config: MiniMaxM3SparseConfig,
    *,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """MiniMax-M3 sparse prefill backed by MSA's ``fmha_sm100``.

    Inputs follow the same conventions as
    :func:`minimax_m3_sparse_prefill`. The main differences are:

      * The K/V caches are accepted in TRT-LLM's pool layout and
        permuted to MSA's HND paged layout internally.
      * The index branch's ``idx_v_cache`` is unused: MSA's proxy pass
        only consumes ``max_score``, so the M3 default
        ``disable_index_value=True`` is the only supported mode.
      * ``num_index_heads`` must be a multiple of ``num_kv_heads``;
        the per-block max score is reduced with ``amax`` over each KV
        head's index-head group before ``sparse_topk_select`` runs.
    """
    if not metadata.is_prefill:
        raise ValueError("MSA prefill entry called with decode metadata")
    if metadata.q_batch_row is None or metadata.q_positions is None:
        raise ValueError("prefill metadata requires q_batch_row / q_positions")
    if config.head_dim != _MSA_REQUIRED_HEAD_DIM:
        raise NotImplementedError(
            f"MSA backend currently supports head_dim={_MSA_REQUIRED_HEAD_DIM}; "
            f"got {config.head_dim}."
        )
    if config.sparse_index_dim != _MSA_REQUIRED_HEAD_DIM:
        raise NotImplementedError(
            f"MSA backend requires sparse_index_dim={_MSA_REQUIRED_HEAD_DIM}; "
            f"got {config.sparse_index_dim}."
        )
    if config.topk != _MSA_REQUIRED_TOPK:
        raise NotImplementedError(
            f"MSA backend currently supports topk={_MSA_REQUIRED_TOPK}; got {config.topk}."
        )

    sm_scale = sm_scale if sm_scale is not None else config.head_dim**-0.5
    idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5

    k_paged = _cache_view_to_msa_paged(k_cache)
    v_paged = _cache_view_to_msa_paged(v_cache)
    idx_k_paged = _idx_cache_to_msa_paged(idx_k_cache)
    page_size = _page_size_from_view(k_cache)
    if page_size != config.block_size:
        raise ValueError(
            f"MSA backend requires page_size == sparse_block_size; "
            f"got page_size={page_size}, sparse_block_size={config.block_size}."
        )

    msa_plans = getattr(metadata, "msa_plans", None)
    if msa_plans is not None:
        qo_lens_cpu = msa_plans["qo_lens_cpu"]
        kv_lens_cpu = msa_plans["kv_lens_cpu"]
        qo_offset_cpu = msa_plans["qo_offset_cpu"]
        kv_indices = msa_plans["kv_indices"]
    else:
        qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = _qo_lens_offsets_from_metadata(metadata)
        kv_indices, _ = _build_kv_indices_and_lens(metadata, page_size)

    kv_block_indexes = _msa_index_proxy_and_topk(
        idx_q,
        idx_k_paged,
        qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu,
        kv_indices=kv_indices,
        config=config,
        idx_sm_scale=idx_sm_scale,
        causal=True,
        init_blocks=config.init_blocks,
        local_blocks=config.local_blocks,
    )

    out = _msa_sparse_attention(
        q,
        k_paged,
        v_paged,
        kv_block_indexes,
        qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        causal=True,
    )

    total_q = int(q.shape[0])
    return out.reshape(total_q, config.num_q_heads * config.head_dim).contiguous()


def minimax_m3_msa_sparse_decode(
    q: torch.Tensor,
    idx_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    metadata: MiniMaxM3SparseAttentionMetadata,
    config: MiniMaxM3SparseConfig,
    *,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure-decode path: always the in-tree graph-safe driver.

    Decode is the ``qo_len == 1`` specialization of
    :func:`minimax_m3_msa_sparse_prefill`.  It runs the same
    ``fmha_sm100`` kernel binaries as the prefill path but through
    :mod:`.decode_wrapper.dispatch` — device-tensor launch
    arguments, device-side top-k, CUDA-graph-capturable end to end.
    The legacy MSA host-driver decode (``fmha_sm100_plan`` +
    ``sparse_topk_select``) was removed after the in-tree driver
    reached bit-parity in eager and passed GSM8K under CUDA graphs.
    """
    if metadata.is_prefill:
        raise ValueError("MSA decode entry called with prefill metadata")
    if config.head_dim != _MSA_REQUIRED_HEAD_DIM:
        raise NotImplementedError(
            f"MSA backend currently supports head_dim={_MSA_REQUIRED_HEAD_DIM}; "
            f"got {config.head_dim}."
        )
    if config.sparse_index_dim != _MSA_REQUIRED_HEAD_DIM:
        raise NotImplementedError(
            f"MSA backend requires sparse_index_dim={_MSA_REQUIRED_HEAD_DIM}; "
            f"got {config.sparse_index_dim}."
        )
    if config.topk != _MSA_REQUIRED_TOPK:
        raise NotImplementedError(
            f"MSA backend currently supports topk={_MSA_REQUIRED_TOPK}; got {config.topk}."
        )

    sm_scale = sm_scale if sm_scale is not None else config.head_dim**-0.5
    idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5

    k_paged = _cache_view_to_msa_paged(k_cache)
    v_paged = _cache_view_to_msa_paged(v_cache)
    idx_k_paged = _idx_cache_to_msa_paged(idx_k_cache)
    page_size = _page_size_from_view(k_cache)
    if page_size != config.block_size:
        raise ValueError(
            f"MSA backend requires page_size == sparse_block_size; "
            f"got page_size={page_size}, sparse_block_size={config.block_size}."
        )

    return _intree_sparse_decode(
        q,
        idx_q,
        k_paged,
        v_paged,
        idx_k_paged,
        metadata,
        config,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# AttentionBackend wrapper
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_minimax_m3_msa_attention_backend_cls():
    """Return :class:`MiniMaxM3MSARuntimeBackend` (lazy import).

    Subclasses the canonical Triton-backed
    :class:`MiniMaxM3SparseRuntimeBackend` and overrides
    :meth:`forward_sparse` so the per-layer dispatch goes through the
    MSA prefill / decode primitives above. Construction validates that
    the layer's config is compatible with the MSA kernel surface
    (head_dim, sparse_index_dim, topk, disable_index_value) before any
    forward call.
    """
    base_cls = get_minimax_m3_attention_backend_cls()

    class MiniMaxM3MSARuntimeBackend(base_cls):  # type: ignore[misc]
        """MSA-backed ``MiniMaxM3SparseRuntimeBackend``.

        See module docstring for the two-step dispatch flow. The
        backend reuses every parent-class lifecycle method
        (``support_fused_rope``, ``__init__`` validation, the standard
        ``forward`` keyword-routing) and only swaps the prefill /
        decode primitives.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.disable_index_value:
                raise NotImplementedError(
                    "MSA backend currently requires disable_index_value=True "
                    "(MSA's proxy pass only consumes max_score; an index-V "
                    "path is not implemented yet)."
                )
            # Validate kernel preconditions up-front so layer
            # construction fails with a clear message rather than a
            # cryptic shape mismatch on the first forward call.
            if self.m3_config.head_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires head_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.head_dim}."
                )
            if self.m3_config.sparse_index_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires sparse_index_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.sparse_index_dim}."
                )
            if self.m3_config.topk != _MSA_REQUIRED_TOPK:
                raise NotImplementedError(
                    f"MSA backend requires topk={_MSA_REQUIRED_TOPK}, got {self.m3_config.topk}."
                )
            # Register the per-rank sparse geometry process-wide at
            # construction time — before any forward, hence before any
            # CUDA graph capture — so every metadata instance's
            # ``prepare()`` (including the CUDA graph runner's separate
            # instances) can pre-build the MSA plans / kv-indices
            # staging. See ``msa_plan_cache.set_global_msa_geometry``.
            from .msa_plan_cache import MsaPlanCacheGeometry, set_global_msa_geometry

            set_global_msa_geometry(
                MsaPlanCacheGeometry(
                    num_q_heads=int(self.m3_config.num_q_heads),
                    num_kv_heads=int(self.m3_config.num_kv_heads),
                    num_index_heads=int(self.m3_config.num_index_heads),
                    head_dim=int(self.m3_config.head_dim),
                    block_size=int(self.m3_config.block_size),
                    topk=int(self.m3_config.topk),
                    init_blocks=int(self.m3_config.init_blocks),
                    local_blocks=int(self.m3_config.local_blocks),
                )
            )

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
        ) -> torch.Tensor:
            """MSA-backed sparse forward.

            Mirrors the parent class's prologue (cache writes, metadata
            device migration) and then dispatches to the MSA prefill /
            decode primitives instead of the Triton path. The MSA
            kernels read directly from the paged caches we just
            populated; we do **not** gather K/V into per-batch padded
            slabs.
            """
            if idx_v is not None:
                raise NotImplementedError(
                    "MSA backend does not consume idx_v (disable_index_value=True is required)."
                )
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

            # KV writes use the same layout-aware helper as the
            # Triton backend; the MSA kernels read from the same paged
            # pool afterwards.
            _write_main_kv_slots(k_cache, out_cache_loc, k_view)
            _write_main_kv_slots(v_cache, out_cache_loc, v_view)
            _write_main_kv_slots(idx_k_cache, out_cache_loc, idx_k_view)

            if m3_metadata.is_prefill:
                return minimax_m3_msa_sparse_prefill(
                    q_view,
                    k_cache,
                    v_cache,
                    idx_q_view,
                    idx_k_cache,
                    m3_metadata,
                    self.m3_config,
                    sm_scale=sm_scale,
                    idx_sm_scale=idx_sm_scale,
                )
            return minimax_m3_msa_sparse_decode(
                q_view,
                idx_q_view,
                k_cache,
                v_cache,
                idx_k_cache,
                m3_metadata,
                self.m3_config,
                sm_scale=sm_scale,
                idx_sm_scale=idx_sm_scale,
            )

    return MiniMaxM3MSARuntimeBackend


def get_minimax_m3_attention_backend_cls_with_msa(
    sparse_params: "MiniMaxM3SparseParams",
):
    """Return the MSA backend class when ``sparse_params.use_msa`` is set.

    Falls back to the Triton-backed
    :class:`MiniMaxM3SparseRuntimeBackend` otherwise. Centralises the
    Triton-vs-MSA dispatch decision so :mod:`...sparse.utils` and the
    model layer do not duplicate the predicate.
    """
    if getattr(sparse_params, "use_msa", False):
        return get_minimax_m3_msa_attention_backend_cls()
    return get_minimax_m3_attention_backend_cls()


__all__ = [
    "get_minimax_m3_attention_backend_cls_with_msa",
    "get_minimax_m3_msa_attention_backend_cls",
    "minimax_m3_msa_sparse_decode",
    "minimax_m3_msa_sparse_prefill",
]
