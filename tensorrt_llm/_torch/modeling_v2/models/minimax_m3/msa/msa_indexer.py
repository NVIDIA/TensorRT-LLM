# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA sparse-attention indexer.

Mirrors the DSA indexer pattern: a submodule owned by the sparse backend
that runs the predictor pass and returns the per-query selected KV block
indices the main attention consumes. It calls fmha_sm100 directly in
output_maxscore mode, reduces the per-index-head max score to KV-head
granularity, and selects the top-k blocks per query.

Results are [total_q, num_kv_heads, topk] int32, ascending with -1 padding.
"""

from __future__ import annotations

import functools
import inspect
import os
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.fmha.utils import get_multi_processor_count_for_device
from tensorrt_llm._utils import async_tensor_h2d
from tensorrt_llm.logger import logger

from .kernels.msa_utils import (
    MSA_REQUIRED_HEAD_DIM,
    MSA_REQUIRED_TOPK,
    per_token_valid_blocks,
    require_msa_module,
    select_blocks_from_maxscore,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
        MiniMaxM3SparseConfig,
    )


@functools.lru_cache(maxsize=1)
def cutedsl_score_runner():
    """Return the CuTe DSL indexer scoring runner, or None if unavailable.

    The CuTe DSL ops are registered only when the nvidia-cutlass-dsl package is
    importable, so this stays a soft dependency.

    Resolved once for the process: package availability cannot change under a
    running model, and every sparse layer of every step scores through here.
    """
    try:
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
    except ImportError:
        return None
    return getattr(cute_dsl_custom_ops, "CuteDSLMiniMaxM3IndexDecodeScoreRunner", None)


def _cutedsl_score(
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    max_score: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    decode_query_len: int,
) -> bool:
    """Try to fill `max_score` with the CuTe DSL scorer; report whether it ran.

    `max_score` is the [num_index_heads, max_k_tiles, total_q] buffer the block
    selector consumes. The kernel writes [head, token, block], so it is handed
    the transposed view: same backing store, no copy, and the stores end up
    coalesced across tokens rather than strided by max_k_tiles.

    The buffer is deliberately not pre-filled with -inf. The kernel writes
    blocks [0, ceil(seq_len / page_size)) for every token of a request, and the
    selector reads only [0, n_valid_blocks[token])), which is bounded by that
    same count for every token including the shorter ones in a multi-token
    speculative step. So every entry the selector reads has just been written.
    """
    runner = cutedsl_score_runner()
    if runner is None:
        return False

    total_q, num_index_heads, head_dim = idx_q.shape
    page_size = int(idx_k_paged.shape[2])
    if not runner.is_supported(
        q_dtype=idx_q.dtype,
        num_heads=num_index_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_decode_query_len=decode_query_len,
    ):
        return False
    if idx_k_paged.dtype != idx_q.dtype or max_score.shape[2] != total_q:
        return False

    # The kernel wants MQA index-K as [num_pages, page_size, head_dim]; the
    # squeeze is zero-copy and keeps the pool's real per-page stride, which the
    # TMA descriptor reads at runtime.
    # Outside torch.compile the runner is called directly: the registered op
    # launches the same compiled kernel, behind a Python custom-op dispatch and
    # an input-description string it formats on every call, once per sparse
    # layer of every eager step.
    score_op = (
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score
        if torch.compiler.is_compiling()
        else runner.forward
    )
    score_op(
        idx_q,
        idx_k_paged.squeeze(1),
        block_table,
        seq_lens_cuda,
        max_score.transpose(1, 2),
        decode_query_len,
    )
    return True


_PLAN_TAKES_MAXSCORE_TILE: dict = {}


def _plan_takes_maxscore_tile(plan) -> bool:
    """Whether the loaded MSA fmha_sm100_plan accepts ``maxscore_qo_tile`` (decided once per function).

    The public planner forwards ``**kwargs`` to its module's ``_fmha_sm100_plan_impl``, which is where the argument
    is declared, so that signature is read too.
    """
    takes = _PLAN_TAKES_MAXSCORE_TILE.get(plan)
    if takes is None:
        takes = False
        for fn in (plan, getattr(inspect.getmodule(plan), "_fmha_sm100_plan_impl", None)):
            try:
                if fn is not None and "maxscore_qo_tile" in inspect.signature(fn).parameters:
                    takes = True
                    break
            except (TypeError, ValueError):
                continue
        _PLAN_TAKES_MAXSCORE_TILE[plan] = takes
    return takes


def proxy_plan_kwargs(
    policy,
    fmha_sm100,
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    rows: Optional[Tuple[int, int]],
    page_size: int,
) -> dict:
    """Extra fmha_sm100_plan arguments for a prefill proxy plan over batch rows ``rows`` under
    MsaRuntimePolicy.prefill_proxy_* (or {} for the plain plan). Shared by prepare()'s plan and the inline plan.

    A span longer than 128 rows takes maxscore_qo_tile (and its splits) only where the loaded planner accepts it:
    without the tile, splits alone leave the 160..15K-row proxy as slow as one split. A span of
    at most 128 rows keeps the planner's own tile and takes only the split rule. Every choice gives the same score
    table bits as the plain plan (0 differing scores in every config).
    """
    if policy is None or rows is None:
        return {}
    first, last = rows
    if last <= first:
        return {}
    max_rows = int(qo_lens_cpu[first:last].max())
    # The policy fields come with RuntimeHooks' MsaRuntimePolicy; an engine without them plans the plain proxy.
    tile = getattr(policy, "prefill_proxy_qo_tile", None)
    kwargs: dict = {}
    if max_rows > 128:
        if tile is None or not _plan_takes_maxscore_tile(fmha_sm100.fmha_sm100_plan):
            return {}
        kwargs["maxscore_qo_tile"] = tile
        splits = getattr(policy, "prefill_proxy_kv_splits", None) or 1
    else:
        splits = 1
    table = getattr(policy, "prefill_proxy_kv_splits_by_rows", None)
    bucket = next((entry for entry in table if max_rows <= entry[0]), None) if table else None
    if bucket is not None:
        # The step's longest KV sets the floor; multi-request steps land within 3% of their best split this way.
        _, top_split, min_pages = bucket
        kv_pages = -(-int(kv_lens_cpu[first:last].max()) // page_size)
        splits = top_split
        while splits > 1 and kv_pages < min_pages * splits:
            splits //= 2
        logger.info_once(
            f"AGG_PROXY_SPLITS taken: prefill proxy num_kv_splits {splits} (rows {max_rows}, kv pages {kv_pages})",
            key=f"msa_agg_proxy_splits_{splits}",
        )
    if not kwargs and splits == 1:
        return {}
    kwargs["num_kv_splits"] = splits
    if "maxscore_qo_tile" in kwargs:
        logger.info_once(
            f"AGG_PROXY_Q128 taken: prefill proxy plan maxscore_qo_tile {tile}, num_kv_splits {splits}",
            key="msa_agg_proxy_qo_tile",
        )
    return kwargs


def _proxy_max_score(
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    *,
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    sm_scale: float,
    causal: bool,
    policy=None,
) -> torch.Tensor:
    """Run the fmha_sm100 MQA proxy pass and return the per-block max score.

    ``policy`` (MsaRuntimePolicy) plans it under the same prefill proxy rule as prepare()'s plan (proxy_plan_kwargs).

    Follows MSA's two-call pattern: fmha_sm100_plan builds the plan with
    output_maxscore and num_kv_heads 1, then fmha_sm100 runs with output_o
    disabled so only the per-block max score is produced. Returns
    [num_index_heads, max_k_tiles, total_q] float32.
    """
    fmha_sm100 = require_msa_module()

    if idx_q.dim() != 3:
        raise ValueError(
            "MsaIndexer expects idx_q [total_q, num_index_heads, head_dim]; "
            f"got {tuple(idx_q.shape)}."
        )
    if idx_k_paged.dim() != 4 or idx_k_paged.shape[1] != 1:
        raise ValueError(
            "MsaIndexer expects MQA paged index-K [num_pages, 1, page_size, head_dim]; "
            f"got {tuple(idx_k_paged.shape)}."
        )

    page_size = int(idx_k_paged.shape[2])
    proxy_plan = fmha_sm100.fmha_sm100_plan(
        qo_lens_cpu,
        kv_lens_cpu,
        idx_q.shape[1],
        num_kv_heads=1,
        qo_offset=qo_offset_cpu,
        page_size=page_size,
        output_maxscore=True,
        causal=causal,
        **{
            "num_kv_splits": 1,
            **proxy_plan_kwargs(
                policy,
                fmha_sm100,
                qo_lens_cpu,
                kv_lens_cpu,
                (0, int(qo_lens_cpu.shape[0])),
                page_size,
            ),
        },
    )
    _, max_score = fmha_sm100.fmha_sm100(
        idx_q,
        idx_k_paged,
        idx_k_paged,
        proxy_plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        sm_scale=sm_scale,
    )
    return max_score


# Context proxy plans split along KV into segments (segmented_proxy_plan) where that fills the GPU.
PREFILL_PROXY_SPLIT = True
# Longest segment of a split proxy plan: fmha_sm100 splits KV only on its 128-query tile.
_PROXY_SEGMENT_LEN = 128
# Below this cached prefix the split plan's longer GPU planner pass eats most of the saving.
_PROXY_SPLIT_MIN_PREFIX = 4096
# fmha_sm100 allocates about 2.6 MB of plan workspace per split.
_PROXY_MAX_KV_SPLITS = 16


def _proxy_kv_splits(qo_lens: list[int], qo_offsets: list[int], num_index_heads: int) -> int:
    """KV splits filling about one SM wave with these rows' segments, or 0 to plan them unsplit."""
    # If every row has <= 64 queries, segments select variants prewarm_split_proxy_variants skips.
    if max(qo_offsets) < _PROXY_SPLIT_MIN_PREFIX or max(qo_lens) <= 64:
        return 0
    num_segments = sum((n + _PROXY_SEGMENT_LEN - 1) // _PROXY_SEGMENT_LEN for n in qo_lens)
    num_sms = get_multi_processor_count_for_device(torch.cuda.current_device())
    return min(_PROXY_MAX_KV_SPLITS, num_sms // (num_segments * num_index_heads))


def _segmented_proxy_plan(
    fmha_sm100: ModuleType,
    qo_lens: list[int],
    qo_offsets: list[int],
    *,
    num_index_heads: int,
    page_size: int,
    num_kv_splits: int,
    kv_indices: torch.Tensor,
) -> tuple[tuple, torch.Tensor]:
    """Plan rows as segments of at most _PROXY_SEGMENT_LEN queries, split along KV.

    Each segment keeps its row's causal window and writes each block max once, from the same
    Q.K products: the scores are bitwise those of the unsplit plan.
    """
    seg_qo_lens, seg_kv_lens, seg_offsets = [], [], []
    # seg_shifts: the row's first page in kv_indices minus the segment's first gathered slot.
    seg_pages, seg_shifts = [], []
    row_first_page = num_gathered = 0
    for qo_len, prefix in zip(qo_lens, qo_offsets):
        for first in range(0, qo_len, _PROXY_SEGMENT_LEN):
            n = min(_PROXY_SEGMENT_LEN, qo_len - first)
            kv_len = prefix + first + n
            pages = (kv_len + page_size - 1) // page_size
            seg_qo_lens.append(n)
            seg_kv_lens.append(kv_len)
            seg_offsets.append(prefix + first)
            seg_pages.append(pages)
            seg_shifts.append(row_first_page - num_gathered)
            num_gathered += pages
        row_first_page += (prefix + qo_len + page_size - 1) // page_size
    plan = fmha_sm100.fmha_sm100_plan(
        torch.tensor(seg_qo_lens, dtype=torch.int32),
        torch.tensor(seg_kv_lens, dtype=torch.int32),
        qo_offset=torch.tensor(seg_offsets, dtype=torch.int32),
        page_size=page_size,
        num_kv_splits=num_kv_splits,
        causal=True,
        num_qo_heads=num_index_heads,
        num_kv_heads=1,
        output_maxscore=True,
    )
    # fmha_sm100 sizes a request's page span by its kv_len, so each segment gets its own page copy.
    staged = async_tensor_h2d([seg_shifts, seg_pages], torch.int64, kv_indices.device)
    gather = torch.arange(num_gathered, device=kv_indices.device)
    gather += torch.repeat_interleave(staged[0], staged[1], output_size=num_gathered)
    return plan, kv_indices.index_select(0, gather)


def segmented_proxy_plan(
    fmha_sm100: ModuleType,
    qo_lens_cpu: torch.Tensor,
    qo_offset_cpu: torch.Tensor,
    *,
    num_index_heads: int,
    page_size: int,
    kv_indices: torch.Tensor,
) -> Optional[tuple[tuple, torch.Tensor]]:
    """KV-split proxy plan for these rows and the page table it reads, or None to plan them unsplit.

    Run it over exactly these rows' query tokens: its per-split workspace is strided by that count.

    Args:
        fmha_sm100: The MSA kernel module.
        qo_lens_cpu: [rows] host int32 query lengths.
        qo_offset_cpu: [rows] host int32 cached prefix lengths.
        num_index_heads: Index heads on this rank.
        page_size: Tokens per page.
        kv_indices: Device int32 flattened page table whose prefix holds these rows' pages.
    """
    if not PREFILL_PROXY_SPLIT:
        return None
    qo_lens, qo_offsets = qo_lens_cpu.tolist(), qo_offset_cpu.tolist()
    num_kv_splits = _proxy_kv_splits(qo_lens, qo_offsets, num_index_heads)
    if num_kv_splits == 0:
        return None
    return _segmented_proxy_plan(
        fmha_sm100,
        qo_lens,
        qo_offsets,
        num_index_heads=num_index_heads,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
        kv_indices=kv_indices,
    )


@functools.lru_cache(maxsize=None)
def prewarm_split_proxy_variants(dtype: torch.dtype, num_index_heads: int, page_size: int) -> None:
    """Load, compiling where missing, the fmha_sm100 variants segmented proxy plans run.

    Warmup never segments, and compiling a variant inline stalls every rank for over a minute.

    Args:
        dtype: Index-K cache dtype, which index Q is cast to.
        num_index_heads: Index heads on this rank.
        page_size: Tokens per page.
    """
    if not PREFILL_PROXY_SPLIT:
        return
    fmha_sm100 = require_msa_module()
    idx_q = torch.zeros(
        _PROXY_SEGMENT_LEN, num_index_heads, MSA_REQUIRED_HEAD_DIM, dtype=dtype, device="cuda"
    )
    num_pages = (_PROXY_SEGMENT_LEN + page_size - 1) // page_size
    idx_k = torch.zeros(num_pages, 1, page_size, MSA_REQUIRED_HEAD_DIM, dtype=dtype, device="cuda")
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    for num_kv_splits in (1, 2):
        plan, plan_kv_indices = _segmented_proxy_plan(
            fmha_sm100,
            [_PROXY_SEGMENT_LEN],
            [0],
            num_index_heads=num_index_heads,
            page_size=page_size,
            num_kv_splits=num_kv_splits,
            kv_indices=kv_indices,
        )
        fmha_sm100.fmha_sm100(
            idx_q,
            idx_k,
            idx_k,
            plan,
            kv_indices=plan_kv_indices,
            output_o=False,
            output_maxscore=True,
        )


def _empty_topk_table(
    total_q: int, num_kv_heads: int, topk: int, device: torch.device, *, head_major: bool
) -> torch.Tensor:
    """An uninitialized [total_q, num_kv_heads, topk] int32 table, head-major backed if asked."""
    shape = (num_kv_heads, total_q, topk) if head_major else (total_q, num_kv_heads, topk)
    # Only a mixed step splits the table and a mixed step is never captured, so
    # this allocation cannot land in a graph's memory pool.
    out = torch.empty(shape, dtype=torch.int32, device=device)
    return out.transpose(0, 1) if head_major else out


def _combined_topk_table(
    ctx_table: torch.Tensor,
    gen_table: torch.Tensor,
    *,
    head_major: bool,
) -> torch.Tensor:
    """Concatenate the context and generation top-k tables along the token axis.

    Both halves are [tokens, num_kv_heads, topk]. `head_major` backs the result
    the way select_blocks_from_maxscore backs its own output, so the combined
    table permutes to a contiguous [num_kv_heads, total_q, topk] as an unsplit
    one does.
    """
    ctx_tokens = int(ctx_table.shape[0])
    total_q = ctx_tokens + int(gen_table.shape[0])
    num_kv_heads, topk = int(ctx_table.shape[1]), int(ctx_table.shape[2])
    shape = (num_kv_heads, total_q, topk) if head_major else (total_q, num_kv_heads, topk)
    # Only a mixed step splits the table and a mixed step is never captured, so
    # this allocation cannot land in a graph's memory pool.
    out = torch.empty(shape, dtype=ctx_table.dtype, device=ctx_table.device)
    if head_major:
        out = out.transpose(0, 1)
    out[:ctx_tokens].copy_(ctx_table)
    out[ctx_tokens:].copy_(gen_table)
    return out


# Context-row top-16 select through minimax_m31_select_prefill_tile on sm_103: one CTA per (KV head,
# 8 queries) reads each 32-byte score sector once. Same table as minimax_m3_select_blocks, bit for bit.
PREFILL_SELECT_TILE = True
# Below this many context tokens the stock select is as fast.
PREFILL_SELECT_TILE_MIN_ROWS = 256
# At or below 128 blocks the stock select runs register-resident bitonic sorts and is faster.
PREFILL_SELECT_TILE_MIN_BLOCKS = 129
# The op's shared-memory bound (kMinimaxM3SelectPrefillTileMaxBlocks).
PREFILL_SELECT_TILE_MAX_BLOCKS = 5856
# Context proxy scores go into an unfilled buffer, since the proxy writes every entry the select reads.
PREFILL_HOST_TRIMS = True


def _family_op(name: str) -> Optional[Callable]:
    """The family's native op ``name`` (``torch.ops.mv2_minimax_m3``), or None when this installation lacks it."""
    from ..native import family_ops

    try:
        return getattr(family_ops(), name, None)
    except Exception:  # noqa: BLE001  (no family library: the Python select runs)
        return None


@functools.lru_cache(maxsize=None)
def _prefill_tile_op(device_index: int) -> Optional[Callable]:
    """The family native op ``minimax_m31_select_prefill_tile`` on an sm_103 device, or None."""
    if torch.cuda.get_device_capability(device_index) != (10, 3):
        return None
    op = _family_op("minimax_m31_select_prefill_tile")
    logger.info(
        "MSA prefill select: minimax_m31_select_prefill_tile "
        f"{'engaged' if op is not None else 'not built in this library'} "
        f"(rows >= {PREFILL_SELECT_TILE_MIN_ROWS}, blocks <= {PREFILL_SELECT_TILE_MAX_BLOCKS})"
    )
    return op


def _prefill_tile_select(
    max_score_kv: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Top-16 table from minimax_m31_select_prefill_tile, or None outside its gate or shapes.

    Returns `out` when given (any [total_q, num_kv_heads, 16] int32 view with unit last-dim stride),
    else a new head-major backed table, as select_blocks_from_maxscore(..., head_major_output=True).
    """
    if (
        not PREFILL_SELECT_TILE
        or max_score_kv.dtype != torch.float32
        or max_score_kv.dim() != 3
        or not max_score_kv.is_cuda
        or max_score_kv.stride(2) != 1
        or init_blocks < 0
        or local_blocks < 0
    ):
        return None
    num_kv_heads, num_blocks, total_q = (int(x) for x in max_score_kv.shape)
    if (
        total_q < max(PREFILL_SELECT_TILE_MIN_ROWS, 1)
        or not max(PREFILL_SELECT_TILE_MIN_BLOCKS, 1)
        <= num_blocks
        <= PREFILL_SELECT_TILE_MAX_BLOCKS
        or torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    op = _prefill_tile_op(max_score_kv.device.index)
    if op is None:
        return None
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    if nvb.numel() < total_q:
        return None
    if out is None:
        out = torch.empty(
            (num_kv_heads, total_q, MSA_REQUIRED_TOPK),
            dtype=torch.int32,
            device=max_score_kv.device,
        ).permute(1, 0, 2)
    op(max_score_kv, nvb, out, init_blocks, local_blocks)
    return out


# Decode-row top-16 select through vLLM's MiniMax-M3.1 ranker (vllm-project/MSA indexer_topk_select, vendored in
# kernels/minimaxM31/vllmTopk) on the NVFP4 index-K reader's native scores: the select vLLM serves after the same
# reader. Not exact: it ranks on a 16-bit key quantized over each row's [min, max] (ties to the lower block), so blocks
# within one key step of the 16th can be picked differently. The select is measured faster per layer on SM107.
# Default ON; TRTLLM_M31_DISABLE_DEP_VLLM_TOPK=1 restores minimax_m3_select_blocks.
DEP_VLLM_TOPK = os.environ.get("TRTLLM_M31_DISABLE_DEP_VLLM_TOPK", "0") != "1"
# The op's bounds (kDepVllmTopkMaxColumns = the vendored select's m3_max_n() in the DEP build, kDepVllmTopkMaxRows).
DEP_VLLM_TOPK_MAX_BLOCKS = 9217
DEP_VLLM_TOPK_MAX_ROWS = 2047


def _dep_vllm_topk_decline(reason: str, detail: str = "") -> None:
    """Log why a decode select stays on minimax_m3_select_blocks, once per reason."""
    logger.info_once(
        f"DEP_VLLM_TOPK declined: {reason}{f' ({detail})' if detail else ''}",
        key=f"dep_vllm_topk_declined_{reason}",
    )


@functools.lru_cache(maxsize=None)
def _dep_vllm_topk_op(device_index: int) -> Optional[Callable]:
    """The family native op ``minimax_m3_sm107_dep_vllm_topk`` on an sm_107 device, or None."""
    if torch.cuda.get_device_capability(device_index) != (10, 7):
        return None
    op = _family_op("minimax_m3_sm107_dep_vllm_topk")
    logger.info(
        "MSA decode select: DEP_VLLM_TOPK vLLM 16-bit-key top-k "
        f"{'engaged' if op is not None else 'not built in this library'} "
        f"(native-layout scores, heads x rows <= {DEP_VLLM_TOPK_MAX_ROWS}, blocks <= {DEP_VLLM_TOPK_MAX_BLOCKS})"
    )
    return op


def _dep_vllm_topk_select(
    max_score_kv: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    max_row_blocks: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Head-major [total_q, num_kv_heads, 16] table from vLLM's ranker, or None to keep the built-in select.

    Takes only scores whose pages are contiguous and whose rows do not overlap (the NVFP4 reader's native layout,
    nvfp4_index_k.score_decode_rows), with no init block and one forced local block, as vLLM selects. Every decline
    logs its reason once ("DEP_VLLM_TOPK declined: ...").

    ``max_row_blocks`` bounds every row's candidate count (the KV manager's max_blocks_per_seq). The scores are shaped
    to the proxy plan's worst-case tile width, which rounds that up to a multiple of 128 (8320 columns for the 8196
    blocks of a 1M max_seq_len with DSpark K7: 8193 padded to a multiple of 4 by the KV manager), so the op ranks the
    first max_row_blocks columns: the same candidates, since no row has more.
    """
    if not DEP_VLLM_TOPK:
        return None
    if init_blocks != 0 or local_blocks != 1:
        _dep_vllm_topk_decline(
            "init / local blocks", f"init {init_blocks}, local {local_blocks}; the op takes 0 / 1"
        )
        return None
    if (
        max_score_kv.dtype != torch.float32
        or max_score_kv.dim() != 3
        or not max_score_kv.is_cuda
        or max_score_kv.stride(1) != 1
    ):
        _dep_vllm_topk_decline(
            "not the native layout",
            f"{max_score_kv.dtype}, shape {tuple(max_score_kv.shape)}, strides {tuple(max_score_kv.stride())}",
        )
        return None
    num_kv_heads, num_blocks, total_q = (int(x) for x in max_score_kv.shape)
    if max_row_blocks is not None and 1 <= max_row_blocks < num_blocks:
        max_score_kv = max_score_kv.narrow(1, 0, max_row_blocks)
        num_blocks = max_row_blocks
    if total_q < 1 or max_score_kv.stride(2) < num_blocks:
        _dep_vllm_topk_decline(
            "overlapping rows", f"{total_q} rows, row stride {max_score_kv.stride(2)}"
        )
        return None
    if not 1 <= num_blocks <= DEP_VLLM_TOPK_MAX_BLOCKS:
        _dep_vllm_topk_decline(
            "too many blocks",
            f"{num_blocks} > {DEP_VLLM_TOPK_MAX_BLOCKS} (max_row_blocks {max_row_blocks})",
        )
        return None
    if num_kv_heads * total_q > DEP_VLLM_TOPK_MAX_ROWS:
        _dep_vllm_topk_decline(
            "too many rows", f"{num_kv_heads} heads x {total_q} rows > {DEP_VLLM_TOPK_MAX_ROWS}"
        )
        return None
    op = _dep_vllm_topk_op(max_score_kv.device.index)
    if op is None:
        _dep_vllm_topk_decline("op not available")
        return None
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    if nvb.numel() < total_q:
        _dep_vllm_topk_decline("short n_valid_blocks", f"{nvb.numel()} < {total_q} rows")
        return None
    out = torch.empty(
        (num_kv_heads, total_q, MSA_REQUIRED_TOPK), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    op(max_score_kv, nvb, out, True)
    if not torch.cuda.is_current_stream_capturing():
        logger.info_once(
            f"DEP_VLLM_TOPK taken: {num_kv_heads} heads x {total_q} rows, {num_blocks} blocks",
            key=f"dep_vllm_topk_taken_{num_kv_heads}_{total_q}",
        )
    return out


def _plan_max_k_tiles(proxy_plan: tuple) -> Optional[Tuple[int, int]]:
    """(index heads, max_k_tiles) of the max-score block an unsplit fmha_sm100 plan writes, or None."""
    if not isinstance(proxy_plan, tuple) or len(proxy_plan) != 5 or proxy_plan[0]:
        return None
    info = proxy_plan[3]
    if not isinstance(info, dict):
        return None
    heads, k_tiles = info.get("orig_num_qo_heads"), info.get("max_k_tiles")
    if not isinstance(heads, int) or not isinstance(k_tiles, int) or heads <= 0 or k_tiles <= 0:
        return None
    return heads, k_tiles


def _unfilled_proxy_max_score(
    proxy_plan: tuple, num_tokens: int, device: torch.device
) -> Optional[torch.Tensor]:
    """An uninitialized [heads, max_k_tiles, tokens] score block for a context proxy pass, or None.

    Left to itself fmha_sm100 allocates this block filled with -inf. The fill is dead: the proxy
    writes every block of [0, ceil(causal extent / page)) for every query, the exact range the
    select reads through n_valid_blocks, and the padded tail past it is never read.
    """
    if not PREFILL_HOST_TRIMS or num_tokens <= 0:
        return None
    geometry = _plan_max_k_tiles(proxy_plan)
    if geometry is None:
        return None
    heads, k_tiles = geometry
    return torch.empty((heads, k_tiles, num_tokens), dtype=torch.float32, device=device)


def _group_max_reduce(
    max_score: torch.Tensor,
    config: "MiniMaxM3SparseConfig",
) -> torch.Tensor:
    """Reduce per-index-head max score to per-KV-head granularity by amax.

    Index heads are assumed to be grouped contiguously per KV head, so head h
    maps to KV group h // group.
    """
    group, rem = divmod(config.num_index_heads, config.num_kv_heads)
    if rem != 0:
        raise ValueError(
            "num_index_heads must be divisible by num_kv_heads for group max "
            f"reduce; got num_index_heads={config.num_index_heads}, "
            f"num_kv_heads={config.num_kv_heads}."
        )
    if group > 1:
        # unflatten splits the head dim of any layout (the NVFP4 reader's scores are a strided view).
        return max_score.unflatten(0, (config.num_kv_heads, group)).amax(dim=1)
    return max_score


# The proxy writes every valid block and selection masks the rest. Reusing the
# step buffer avoids allocating and filling it at each sparse layer. A model
# opt-in (RuntimeHooks.proxy_ms_reuse), armed by the executor
# (arm_proxy_ms_reuse); kill switch TRTLLM_M31_DISABLE_PROXY_MS_REUSE=1.
_PROXY_MS_REUSE = False


def arm_proxy_ms_reuse() -> None:
    """Reuse the step's proxy max-score buffer in this process (RuntimeHooks.proxy_ms_reuse), unless killed."""
    global _PROXY_MS_REUSE
    _PROXY_MS_REUSE = os.environ.get("TRTLLM_M31_DISABLE_PROXY_MS_REUSE", "0") != "1"


def arm_sm107_msa_paths() -> None:
    """Keep this process on the MSA paths the MiniMax-M3.1 sm_107 targets run.

    Context proxy plans stay unsplit (or split by MsaRuntimePolicy), proxy scores go into the
    -inf-filled buffer fmha_sm100 allocates, context rows take the stock select, the decoder
    warmup runs no MSA prewarm, and a hybrid NVFP4 KV cache keeps its attention-op pools.
    """
    global PREFILL_PROXY_SPLIT, PREFILL_HOST_TRIMS, PREFILL_SELECT_TILE
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import cache_manager

    from .prewarm import hook as msa_prewarm_hook

    PREFILL_PROXY_SPLIT = PREFILL_HOST_TRIMS = PREFILL_SELECT_TILE = False
    msa_prewarm_hook.ENABLED = False
    cache_manager.NVFP4_POOLLESS = False


_proxy_ms_step: Optional[tuple] = None  # (proxy_plan, tokens, max_score)

# Prefill context scorer (a model target's MsaIndexer.prefill_scorer_hook, e.g. the dep2ctx exact Q8KV8 scorer): the
# number of layers its register() armed (0: none). prepare() builds the step's proxy plan lazily, on its first read,
# only while it is armed and accepts the step's lengths (msa_backend, MSA_LAZY_PROXY_PLAN), so a step every layer's
# hook takes builds none and a step it declines on its lengths is still planned in prepare().
_PREFILL_SCORER_LAYERS = 0
# The armed scorer's host-length predicate (None: every step) and its kernel warm-up (None: nothing to compile).
_PREFILL_SCORER_ACCEPTS: Optional[
    Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], bool]
] = None
_PREFILL_SCORER_PREWARM: Optional[Callable[[int], None]] = None
# Correctness runs only: also score every hooked step with the proxy, inline, and raise on any table difference.
# Calls the hook took / declined in this process (engagement; a decline runs the proxy over the step's plan).
_PREFILL_SCORER_COUNTS = {"taken": 0, "declined": 0, "checked": 0}


def arm_prefill_context_scorer(
    layers: int,
    *,
    accepts: Optional[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], bool]] = None,
    prewarm: Optional[Callable[[int], None]] = None,
) -> None:
    """Record that a model target set ``prefill_scorer_hook`` on ``layers`` MSA indexers (call after every hook is
    set; 0 disarms). ``accepts(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu)`` is False for a step the hook will decline on
    its host lengths alone (prepare() then plans the proxy eagerly, off the forward's critical path); ``prewarm
    (num_index_heads)`` compiles or loads the hook's kernels, which the warm-up's short prefills never reach
    (prewarm_prefill_context_scorer)."""
    global _PREFILL_SCORER_LAYERS, _PREFILL_SCORER_ACCEPTS, _PREFILL_SCORER_PREWARM
    _PREFILL_SCORER_LAYERS = max(int(layers), 0)
    armed = _PREFILL_SCORER_LAYERS > 0
    _PREFILL_SCORER_ACCEPTS = accepts if armed else None
    _PREFILL_SCORER_PREWARM = prewarm if armed else None


def prefill_context_scorer_armed() -> bool:
    """Whether a model target's prefill context scorer is registered in this process."""
    return _PREFILL_SCORER_LAYERS > 0


def prefill_context_scorer_accepts(
    qo_lens_cpu: torch.Tensor, kv_lens_cpu: torch.Tensor, qo_offset_cpu: Optional[torch.Tensor]
) -> bool:
    """Whether the armed prefill context scorer may take a pure-prefill step with these host lengths (False when none
    is armed)."""
    if _PREFILL_SCORER_LAYERS <= 0:
        return False
    accepts = _PREFILL_SCORER_ACCEPTS
    return accepts is None or bool(accepts(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu))


def prewarm_prefill_context_scorer(num_index_heads: int) -> bool:
    """Compile or load the armed prefill context scorer's kernels before the server is ready (the executor's MSA
    indexer prewarm calls it); whether a warm-up ran."""
    prewarm = _PREFILL_SCORER_PREWARM
    if _PREFILL_SCORER_LAYERS <= 0 or prewarm is None:
        return False
    prewarm(int(num_index_heads))
    return True


def _count_prefill_scorer(taken: bool, rows: int) -> None:
    """Engagement counters and lines: ``MSA_PREFILL_SCORER taken`` once, a WARNING at the 1st, 10th, 100th, ... decline
    (the hook declined, so the proxy scores the step over the plan built for it)."""
    key = "taken" if taken else "declined"
    _PREFILL_SCORER_COUNTS[key] += 1
    n = _PREFILL_SCORER_COUNTS[key]
    if taken:
        if n == 1:
            logger.info(
                f"MSA_PREFILL_SCORER taken: a model prefill scorer scores context rows ({rows} rows, "
                f"{_PREFILL_SCORER_LAYERS} layers armed); the fmha_sm100 proxy pass does not run"
            )
    elif str(n).rstrip("0") == "1":  # 1, 10, 100, ...
        logger.warning(
            f"MSA_PREFILL_SCORER declined: {n} calls so far (this one {rows} rows); the fmha_sm100 proxy scores them"
        )


class MsaIndexer:
    """Predictor submodule: proxy MQA scoring and top-k block selection.

    Owned by the MSA attention layer. Stateless in eager mode: it reads the
    per-forward page table and lengths from the attention metadata and calls
    the kernel directly.
    """

    def __init__(self, config: "MiniMaxM3SparseConfig"):
        self.config = config
        # An optional replacement for the top-k select, set by a model that
        # carries a faster kernel for some shapes. It is called as
        # hook(max_score_kv, n_valid_blocks, init_blocks, local_blocks) on the
        # KV-head-reduced scores and returns either the table
        # select_blocks_from_maxscore would (same values, shape, dtype and
        # head-major backing) or None to leave the step to that select.
        self.selector_hook: Optional[
            Callable[[torch.Tensor, torch.Tensor, int, int], Optional[torch.Tensor]]
        ] = None
        # The most candidate blocks any row can have (the KV manager's max_blocks_per_seq), set by the backend; the
        # decode-row DEP_VLLM_TOPK select ranks only that many of the worst-case-wide score columns.
        self.max_row_blocks: Optional[int] = None
        # An optional prefill scorer a model target may set in place of the fmha_sm100 proxy pass on pure-prefill
        # steps (select_prefill_hooked, which the backend calls ahead of reading any proxy plan): called as
        # hook(idx_q, idx_k_paged, qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices), it returns the per-KV-head
        # block scores [num_kv_heads, n_blocks, total_q] fp32, any strides, equal bit for bit to
        # _group_max_reduce(proxy max score) on every cell the select reads (init_blocks <= block < n_valid -
        # local_blocks; the others are forced), or None to keep the proxy. Unset, nothing changes.
        self.prefill_scorer_hook: Optional[Callable[..., Optional[torch.Tensor]]] = None
        # An optional select for prefill_scorer_hook's scores only (selector_hook's signature and contract), tried
        # first by select_prefill_hooked: a layout only the hook produces (e.g. unit block stride) gets its kernel
        # there, while every other select (decode rows included) keeps selector_hook and the built-in ones.
        self.prefill_scorer_selector: Optional[
            Callable[[torch.Tensor, torch.Tensor, int, int], Optional[torch.Tensor]]
        ] = None

    def select_blocks(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        *,
        idx_sm_scale: float,
        kv_indices: torch.Tensor,
        qo_lens_cpu: Optional[torch.Tensor] = None,
        kv_lens_cpu: Optional[torch.Tensor] = None,
        qo_offset_cpu: Optional[torch.Tensor] = None,
        proxy_plan: Optional[tuple] = None,
        proxy_policy=None,
        max_score: Optional[torch.Tensor] = None,
        n_valid_blocks: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens_cuda: Optional[torch.Tensor] = None,
        decode_query_len: Optional[int] = None,
        require_cutedsl: bool = False,
        gen_token_first: int = 0,
        ctx_rows: int = 0,
        gen_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return [total_q, num_kv_heads, topk] selected block indices.

        Plan/run split, mirroring the sparse GQA: production passes a
        precomputed device `n_valid_blocks`, a prebuilt `proxy_plan` wherever
        any row is left to the proxy, and, on a pure-decode step, the
        preallocated `max_score` the captured region runs into.

        `block_table`, `seq_lens_cuda` and `decode_query_len` put the CuTe DSL
        scorer on this step's generation span in place of the fmha_sm100 proxy
        pass; leaving them unset (the standalone kernel tests) runs the proxy
        over the whole batch. `require_cutedsl` says prepare() narrowed the
        proxy plan to the context prefix, so a decline has no fallback.

        `gen_token_first` and `ctx_rows` mark where that span starts: the
        scorer takes query tokens [gen_token_first, total_q) and rows
        [ctx_rows, batch), the proxy the context prefix ahead of both, and both
        are 0 on a pure-decode step. The halves score into separate buffers,
        since fmha_sm100 writes a contiguous [heads, k_tiles, tokens] block and
        cannot fill a slice of the scorer's, then their tables are joined.
        """
        page_size = int(idx_k_paged.shape[2])
        gen_first = int(gen_token_first)

        scored = False
        if gen_scores is not None:
            # MIX_NVFP4_READER: the backend already scored the generation span from the NVFP4 pages.
            max_score = gen_scores
            scored = True
        elif (
            max_score is not None
            and block_table is not None
            and seq_lens_cuda is not None
            and decode_query_len is not None
        ):
            # The scorer emits raw Q.K rather than idx_sm_scale * Q.K, as the
            # fmha_sm100 proxy does; ranking and the +inf init/local forcing in
            # select_blocks_from_maxscore are invariant under a positive scale.
            scored = _cutedsl_score(
                idx_q[gen_first:],
                idx_k_paged,
                max_score,
                block_table=block_table,
                seq_lens_cuda=seq_lens_cuda,
                decode_query_len=decode_query_len,
            )

        if require_cutedsl and not scored:
            raise RuntimeError(
                "MiniMax-M3 prepare() narrowed the fmha_sm100 proxy plan to this "
                "step's context prefix, but the CuTe DSL indexer scorer declined "
                "its generation span. There is no proxy pass left to score it; "
                "the scorer's geometry is required up front by "
                "_validate_decode_kernel_support, so this should not happen."
            )

        # The scorer took nothing, so the proxy runs every token under the
        # whole-batch plan it was handed.
        if not scored:
            gen_first = 0
            max_score = self._proxy_scores(
                idx_q,
                idx_k_paged,
                proxy_plan=proxy_plan,
                max_score=max_score,
                qo_lens_cpu=qo_lens_cpu,
                kv_lens_cpu=kv_lens_cpu,
                qo_offset_cpu=qo_offset_cpu,
                kv_indices=kv_indices,
                idx_sm_scale=idx_sm_scale,
                policy=proxy_policy,
            )

        if n_valid_blocks is None:
            n_valid_blocks = per_token_valid_blocks(
                qo_lens_cpu,
                kv_lens_cpu,
                qo_offset_cpu,
                causal=True,
                block_size=page_size,
            )

        # Rows the proxy scored are context rows: the scorer owns every
        # generation row of a step that has any.
        gen_table = self._select(max_score, n_valid_blocks[gen_first:], prefill=not scored)
        if gen_first == 0:
            return gen_table
        # The proxy scores the context prefix into its own buffer, under the
        # plan prepare() built over rows [0, ctx_rows). Context pages are the
        # prefix of the flattened page table, so kv_indices needs no slice.
        ctx_lens = (
            None if qo_lens_cpu is None else qo_lens_cpu[:ctx_rows],
            None if kv_lens_cpu is None else kv_lens_cpu[:ctx_rows],
            None if qo_offset_cpu is None else qo_offset_cpu[:ctx_rows],
        )
        ctx_scores = self._proxy_scores(
            idx_q[:gen_first],
            idx_k_paged,
            proxy_plan=proxy_plan,
            max_score=None,
            qo_lens_cpu=ctx_lens[0],
            kv_lens_cpu=ctx_lens[1],
            qo_offset_cpu=ctx_lens[2],
            kv_indices=kv_indices,
            idx_sm_scale=idx_sm_scale,
            policy=proxy_policy,
        )
        if PREFILL_SELECT_TILE:
            # The combined table is head-major like an unsplit one (see
            # _combined_topk_table); the prefill tile select writes the context
            # rows into it in place, so only the generation rows are copied.
            combined = _empty_topk_table(
                int(idx_q.shape[0]),
                int(gen_table.shape[1]),
                int(gen_table.shape[2]),
                gen_table.device,
                head_major=True,
            )
            ctx_out = combined[:gen_first]
            ctx_table = self._select(
                ctx_scores, n_valid_blocks[:gen_first], prefill=True, out=ctx_out
            )
            if ctx_table is not ctx_out:
                ctx_out.copy_(ctx_table)
            combined[gen_first:].copy_(gen_table)
            return combined
        ctx_table = self._select(ctx_scores, n_valid_blocks[:gen_first])
        return _combined_topk_table(ctx_table, gen_table, head_major=True)

    def select_prefill_hooked(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        qo_lens_cpu: Optional[torch.Tensor],
        kv_lens_cpu: Optional[torch.Tensor],
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        n_valid_blocks: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """The [total_q, num_kv_heads, topk] table of a pure-prefill step scored by ``prefill_scorer_hook``, or None
        (no hook, or it declined) to leave the step to select_blocks and the proxy.

        `kv_indices` is the step's flattened page table (each request's ceil(kv_len / page) pages in batch order), the
        one the hook plans its block table from. The table is ``prefill_scorer_selector``'s on the hook's scores when
        it is set and takes them, else _select_kv's, as select_blocks' prefill select on the proxy's (``prefill``).
        """
        hook = self.prefill_scorer_hook
        if hook is None or qo_lens_cpu is None or kv_lens_cpu is None:
            return None
        kv_scores = hook(idx_q, idx_k_paged, qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices)
        rows = int(idx_q.shape[0])
        if kv_scores is None:
            _count_prefill_scorer(False, rows)
            return None
        _count_prefill_scorer(True, rows)
        # The proxy did not run: drop the step's reused proxy buffer (several hundred MB at served shapes) so a
        # decline earlier in the process does not keep it alive (PROXY_MS_REUSE).
        global _proxy_ms_step
        _proxy_ms_step = None
        if n_valid_blocks is None:
            n_valid_blocks = per_token_valid_blocks(
                qo_lens_cpu,
                kv_lens_cpu,
                qo_offset_cpu,
                causal=True,
                block_size=int(idx_k_paged.shape[2]),
            )
        table = None
        if self.prefill_scorer_selector is not None:
            table = self.prefill_scorer_selector(
                kv_scores, n_valid_blocks, self.config.init_blocks, self.config.local_blocks
            )
        if table is None:
            table = self._select_kv(kv_scores, n_valid_blocks, prefill=True)
        return table

    def _proxy_scores(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        *,
        proxy_plan: Optional[tuple],
        max_score: Optional[torch.Tensor],
        qo_lens_cpu: Optional[torch.Tensor],
        kv_lens_cpu: Optional[torch.Tensor],
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        idx_sm_scale: float,
        policy=None,
    ) -> torch.Tensor:
        """Run the fmha_sm100 proxy pass over `idx_q` and return its max score.

        Uses the prebuilt plan when prepare() supplied one, and plans inline
        from the host lengths otherwise (standalone callers that skip prepare).
        """
        if proxy_plan is None:
            return _proxy_max_score(
                idx_q,
                idx_k_paged,
                qo_lens_cpu=qo_lens_cpu,
                kv_lens_cpu=kv_lens_cpu,
                qo_offset_cpu=qo_offset_cpu,
                kv_indices=kv_indices,
                sm_scale=idx_sm_scale,
                causal=True,
                policy=policy,
            )
        global _proxy_ms_step
        fmha_sm100 = require_msa_module()
        tokens = int(idx_q.shape[0])
        reuse = _PROXY_MS_REUSE and max_score is None
        cached = _proxy_ms_step
        if reuse and cached is not None and cached[0] is proxy_plan and cached[1] == tokens:
            max_score = cached[2]
        if max_score is None:
            max_score = _unfilled_proxy_max_score(proxy_plan, tokens, idx_q.device)
        _, scores = fmha_sm100.fmha_sm100(
            idx_q,
            idx_k_paged,
            idx_k_paged,
            proxy_plan,
            kv_indices=kv_indices,
            output_o=False,
            output_maxscore=True,
            max_score=max_score,
            sm_scale=idx_sm_scale,
        )
        if reuse and scores is not None and scores.is_contiguous():
            _proxy_ms_step = (proxy_plan, tokens, scores)
        return scores

    def _select(
        self,
        max_score: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        *,
        prefill: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reduce scores to KV-head granularity and take the top-k blocks.

        Always into a head-major backing, whatever the step: the Triton sparse
        decode kernel reads the table head-major and takes every generation
        row, while fmha_sm100 reads whatever strides it is handed.

        `prefill` marks context rows, which may take the prefill tile select;
        it writes into `out` when given, and the caller copies any other
        returned table there.
        """
        return self._select_kv(
            _group_max_reduce(max_score, self.config), n_valid_blocks, prefill=prefill, out=out
        )

    def _select_kv(
        self,
        max_score_kv: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        *,
        prefill: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The top-k blocks from per-KV-head scores [num_kv_heads, n_blocks, total_q] (any strides): the registered
        selector first, as _select."""
        if self.selector_hook is not None:
            table = self.selector_hook(
                max_score_kv,
                n_valid_blocks,
                self.config.init_blocks,
                self.config.local_blocks,
            )
            if table is not None:
                return table
        if prefill:
            table = _prefill_tile_select(
                max_score_kv,
                n_valid_blocks,
                self.config.init_blocks,
                self.config.local_blocks,
                out=out,
            )
            if table is not None:
                return table
        else:
            table = _dep_vllm_topk_select(
                max_score_kv,
                n_valid_blocks,
                self.config.init_blocks,
                self.config.local_blocks,
                self.max_row_blocks,
            )
            if table is not None:
                return table
        return select_blocks_from_maxscore(
            max_score_kv,
            topk=MSA_REQUIRED_TOPK,
            n_valid_blocks=n_valid_blocks,
            init_blocks=self.config.init_blocks,
            local_blocks=self.config.local_blocks,
            head_major_output=True,
        )


__all__ = [
    "MsaIndexer",
    "arm_prefill_context_scorer",
    "cutedsl_score_runner",
    "prefill_context_scorer_accepts",
    "prefill_context_scorer_armed",
    "prewarm_prefill_context_scorer",
    "prewarm_split_proxy_variants",
    "segmented_proxy_plan",
]
