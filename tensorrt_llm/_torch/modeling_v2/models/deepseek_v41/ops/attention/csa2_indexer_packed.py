# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-chunk indexer as stock runs it (`CSA2Indexer._run_packed_group`).

Per packed group of query rows, each request's compressed index-key prefix is gathered into one buffer
(`gather_prefix_keys_`), scored with DeepGEMM's non-paged FP4 MQA logits (each row bounded to its own request's
visible keys, so logits column `j` is compressed position `j`), keys without a page masked (`mask_packed_logits_`),
the row's Top-K taken by stock's prefill selector (the self-sampling GVR engine from 8192 columns, the exact CuTe
radix below) and the picks finalized ascending (`finalize_selection_`). The candidate source publishes its block
candidates from the same packed logits (`candidate_block_max_` -> radix block Top-K -> `expand_candidates_`).

`csa2_indexer_plan` is the host plan of a step (stock's `CSA2TrtllmMetadata._prepare_indexer_plans` for context
rows): request pieces of at most `budget // columns(width)` rows, merged while a group stays within `chunk` rows
and the logits budget, all descriptors in one upload.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._utils import maybe_pin_memory

from .csa2_indexer_decode import _is_valid_queries, dequantize_queries, pack_queries
from .csa2_pack_rows import csa2_unpack_rows
from .csa2_rows_io import csa2_gather_rows_footer, csa2_row_addr

_SPARSE_BLOCK = 8  # the sparse-logits kernel's block: candidate ids are 8-key blocks
_SELF_SAMPLING_MIN_COLUMNS = (
    8192  # stock's `_SELF_SAMPLING_PREFILL_MIN_COLUMNS`: narrower rows keep the exact radix
)
_SELECTORS: dict = {}


def _columns(width: int) -> int:
    return -(-width // 256) * 256  # DeepGEMM pads each compressed logits row to 256 columns


def csa2_indexer_plan(
    segs, ratio: int, max_positions: int, budget: int, device, chunk: int = 32768
):
    """Stock's packed indexer groups of a context step (a setup helper: host planning, one upload).

    `segs` are the step's `(first row, one past the last, request row, span)` in row order (rows hold positions
    `span - (last - first) .. span - 1`). Returns `(groups, descriptors)`: `groups` a list of `(first row, one past
    the last, width, keys, first piece, one past the last piece)`, `descriptors` int32 `[pieces, 5]` on `device`,
    a piece being `(request row, key width, key offset, first query row, row in group)`."""
    raw: List[list] = []
    for t0, t1, r, span in segs:
        if t1 <= t0:
            continue
        prefix = span - (t1 - t0)
        widest = min(max_positions, span // ratio)
        step = max(1, min(chunk, budget // max(1, _columns(widest))))
        for start in range(t0, t1, step):
            stop = min(start + step, t1)
            width = min(max_positions, (prefix + stop - t0) // ratio)
            piece = (start, stop, width, r)
            if raw and width > 0 and raw[-1][-1][2] > 0:
                group = raw[-1]
                merged = max(width, *(p[2] for p in group))
                if (
                    stop - group[0][0] <= chunk
                    and (stop - group[0][0]) * _columns(merged) <= budget
                ):
                    group.append(piece)
                    continue
            raw.append([piece])
    groups, rows = [], []
    for pieces in raw:
        first, last = pieces[0][0], pieces[-1][1]
        lo, keys, width = len(rows), 0, 0
        if pieces[0][2] > 0:
            for start, stop, w, r in pieces:
                rows.append((r, w, keys, start, start - first))
                keys += w
                width = max(width, w)
        groups.append((first, last, width, keys, lo, len(rows)))
    desc = torch.from_numpy(np.asarray(rows, dtype=np.int32).reshape(-1, 5))
    return groups, maybe_pin_memory(desc).to(device, non_blocking=True)


def _selector(k: int, width: int, heuristic: bool):
    """Stock's prefill Top-K (`CSA2Indexer.top_k` / `narrow_prefill_top_k`), one module per configuration."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

    exact = (
        TopKImplementation.CUTE_DSL_RADIX
        if IS_CUTLASS_DSL_AVAILABLE
        else TopKImplementation.CUDA_RADIX
    )
    gvr = (
        heuristic
        and IS_CUTLASS_DSL_AVAILABLE
        and k in (512, 1024, 2048)
        and width >= _SELF_SAMPLING_MIN_COLUMNS
    )
    impl = TopKImplementation.CUTE_DSL_GVR if gvr else exact
    if (k, impl) not in _SELECTORS:
        _SELECTORS[k, impl] = TopK(k, prefill_implementation=impl)
    return _SELECTORS[k, impl]


def _run_group(
    group,
    desc,
    data,
    scale,
    ww,
    exact_heads,
    vis,
    pages,
    table,
    ratio,
    topk,
    max_positions,
    out,
    cands,
    want_cand,
    heuristic,
    scheds,
) -> int:
    """One packed group (stock's `_run_packed_group`): its picks into `out` (and candidates into `cands`) rows. A
    function of its own, as stock's, so a group's gathered keys and logits are freed before the next group allocates
    (a loop variable would keep up to a logits budget alive across the next group's allocation). Returns the group's
    logits width."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
        candidate_block_max_,
        expand_candidates_,
        finalize_selection_,
        gather_prefix_keys_,
        mask_packed_logits_,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _mqa_logits_pool
    from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits, get_mqa_logits_metadata

    first, last, width, keys, p0, p1 = group
    rows, dev = last - first, ww.device
    kdata = torch.empty(keys, 64, dtype=torch.int8, device=dev)
    kscale = torch.empty(keys, dtype=torch.int32, device=dev)
    valid = torch.empty(keys, dtype=torch.bool, device=dev)
    bounds = torch.empty(5, rows, dtype=torch.int32, device=dev)
    gather_prefix_keys_(
        pages, table, vis, desc[p0:p1], 128 // ratio, max_positions, kdata, kscale, valid, bounds
    )
    row_starts, row_ends, row_lengths, row_widths, zeros = bounds.unbind(0)
    sched = None
    if (
        p1 - p0 > 1 and exact_heads
    ):  # DeepGEMM's per-SM split of a multi-request group (`_packed_group_schedule`)
        sched = None if scheds is None else scheds.get(first)
        if sched is None:
            sched = get_mqa_logits_metadata(row_starts, row_ends, keys, data.shape[1])
            if scheds is not None:
                scheds[first] = sched
    max_seqlen_k = -(-width // 8) * 8
    with _mqa_logits_pool(dev, max_seqlen_k):
        logits = fp8_fp4_mqa_logits(
            (data[first:last], scale[first:last]),
            (kdata, kscale),
            ww[first:last],
            row_starts,
            row_ends,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
            schedule_meta=sched,
        )
    mask_packed_logits_(logits, valid, row_starts, row_lengths)
    cols = logits.shape[1]
    if cands is not None:
        block, nblocks = want_cand
        blocks = torch.empty(rows, -(-cols // block), dtype=torch.float32, device=dev)
        counts = torch.empty(rows, dtype=torch.int32, device=dev)
        candidate_block_max_(
            logits, valid, 0, vis, first, row_starts, row_ends, row_widths, blocks, counts, block, 1
        )
        selected = torch.full((rows, nblocks), -1, dtype=torch.int32, device=dev)
        torch.ops.trtllm.indexer_topk_prefill(blocks, zeros, counts, selected, nblocks)
        expand_candidates_(
            selected,
            blocks,
            vis,
            first,
            row_widths,
            cands[0][first:last],
            block,
            cands[1][first:last],
            cands[2][first:last].view(-1),
            _SPARSE_BLOCK,
        )
    scores = logits
    # stock pads the CuTe selectors' scores to 32-byte rows (`select_prepared_scores`)
    if scores.stride(1) != 1 or scores.stride(0) % 8 or scores.data_ptr() % 32 or cols % 8:
        scores = torch.nn.functional.pad(scores.contiguous(), (0, -cols % 8), value=-torch.inf)
    picks = out[first:last]
    _selector(topk, cols, heuristic)(
        scores, picks, is_prefill=True, row_starts=zeros, row_ends=row_lengths
    )
    finalize_selection_(picks, cols, valid=valid, row_starts=row_starts, row_lengths=row_lengths)
    return cols


class Csa2IndexerTopkPacked(OpWrapper):
    """Per context row the `topk` best compressed positions (ascending, `-1` past the row's picks), scored group by
    group over packed prefixes; the same picks as `csa2_indexer_topk_prefill` up to ties at the k-th score.

    `q` is the index query pair `csa2_store_rows` quantized (data int8 `[n, h, 64]`, scales uint8 `[n, h, 4]`) or
    bf16 `[n, h, 128]`, `w` `[n, h]` fp32 (already scaled head weights), `visible` int `[n]` each row's visible
    compressed positions (`(pos + 1) // ratio`), `index_pool` `[pages, 128 // ratio, 68]` uint8 (the engine's native
    footer pages), `table` the owner's GLOBAL page table, `plan` the step's `csa2_indexer_plan`, `max_positions` the
    pool's position bound. `want_cand=(block, blocks)` (the candidate source) also returns `(positions [n, cw],
    block_ids [n, cw // 8], counts [n, 1])` int32 as `csa2_publish_candidates`. `scheds` (optional dict) keeps the
    multi-request groups' DeepGEMM schedules for the step's other layers over the same plan and `visible`. Returns
    `(picks [n, min(topk, columns)], candidates or None)`, `columns` the widest group's logits width; eager only.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="ratio 1: one 600-row chunk after 900 tokens of history",
            spec=dict(ratio=1, reqs=[(900, 600)]),
        ),
        Cell(
            why="ratio 2: three requests (long and short histories) in one group",
            spec=dict(ratio=2, reqs=[(3000, 300), (200, 120), (0, 50)]),
        ),
        Cell(
            why="ratio 1: a 2500-row chunk split into pieces by a small logits budget",
            spec=dict(ratio=1, reqs=[(1000, 2500)], budget=1 << 20),
        ),
        Cell(
            why="ratio 1, the candidate source: two requests of 400 and 90 rows",
            spec=dict(ratio=1, reqs=[(1200, 400), (60, 90)], cand=True),
        ),
        Cell(
            why="ratio 1, 9000+ visible keys: the self-sampling GVR selector",
            spec=dict(ratio=1, reqs=[(9500, 200)]),
        ),
    )
    note = """
    Picks agree with the dense reference up to ties at the k-th score (the test gates the share of agreeing picks),
    and the unreachable (-1) picks agree exactly; the candidate positions are gated as `csa2_publish_candidates`.
    """

    def __call__(
        self,
        q,
        w: torch.Tensor,
        visible: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        plan,
        max_positions: int,
        want_cand: Optional[Tuple[int, int]] = None,
        heuristic: bool = True,
        scheds: Optional[dict] = None,
    ) -> tuple:
        groups, desc = plan
        data, scale = pack_queries(q)
        n, heads = data.shape[:2]
        dev = w.device
        hp = next(h for h in (8, 16, 32, 64) if h >= heads)
        ww = w.float().contiguous()
        if hp != heads:  # zero weights make the padded heads exact no-ops
            data = torch.nn.functional.pad(data, (0, 0, 0, hp - heads))
            scale = torch.nn.functional.pad(scale, (0, hp - heads))
            ww = torch.nn.functional.pad(ww, (0, hp - heads))
        pages = index_pool.view(-1, 64, 1, 68)
        if table.dtype != torch.int32 or table.stride(-1) != 1:
            table = table.to(torch.int32).contiguous()
        vis = visible.contiguous()
        out = torch.full((n, topk), -1, dtype=torch.int32, device=dev)
        cands = None
        if want_cand is not None:
            block, nblocks = want_cand
            cw = -(-(nblocks * block) // (4 * _SPARSE_BLOCK)) * 4 * _SPARSE_BLOCK
            cands = (
                torch.full((n, cw), -1, dtype=torch.int32, device=dev),
                torch.zeros(n, cw // _SPARSE_BLOCK, dtype=torch.int32, device=dev),
                torch.zeros(n, 1, dtype=torch.int32, device=dev),
            )
        cols_out = 0
        for group in groups:
            if group[2] > 0 and group[3] > 0:  # rows without a visible key keep -1
                cols_out = max(
                    cols_out,
                    _run_group(
                        group,
                        desc,
                        data,
                        scale,
                        ww,
                        heads == hp,
                        vis,
                        pages,
                        table,
                        ratio,
                        topk,
                        max_positions,
                        out,
                        cands,
                        want_cand,
                        heuristic,
                        scheds,
                    ),
                )
        k = min(topk, max(cols_out, 8))
        return (out if k == topk else out[:, :k].contiguous()), cands

    def reference(
        self,
        q,
        w,
        visible,
        index_pool,
        table,
        ratio,
        topk,
        plan,
        max_positions,
        want_cand=None,
        heuristic=True,
        scheds=None,
    ) -> torch.Tensor:
        """Dense fp32 scores of every row over its request's visible positions (unmapped pages -inf), the best
        `topk` ascending, -1 padded to the op's width. The plan's descriptors give each row's request."""
        groups, desc = plan
        qf = dequantize_queries(q).float()
        n, dev = qf.shape[0], qf.device
        rid = torch.full((n,), -1, dtype=torch.long)
        cols_out = 0
        d = desc.cpu().long()
        for first, last, width, keys, p0, p1 in groups:
            if width <= 0 or keys <= 0:
                continue
            cols_out = max(cols_out, -(-width // 8) * 8)
            starts = [int(x) for x in d[p0:p1, 3]] + [last]
            for i, p in enumerate(range(p0, p1)):
                rid[starts[i] : starts[i + 1]] = int(d[p, 0])
        k = min(topk, max(cols_out, 8))
        out = torch.full((n, k), -1, dtype=torch.int32, device=dev)
        live = (rid >= 0).nonzero().view(-1).to(dev)
        if cols_out == 0 or live.numel() == 0:
            return out
        r = rid.to(dev)[live]
        cols = torch.arange(cols_out, device=dev)
        addr = csa2_row_addr(
            table, r.unsqueeze(-1).expand(-1, cols_out), cols.expand(len(r), -1), 128 // ratio
        )
        kf = csa2_unpack_rows(csa2_gather_rows_footer(index_pool, addr), "index", torch.float32)
        s = (torch.einsum("nhd,nkd->nhk", qf[live], kf).relu() * w[live].float().unsqueeze(-1)).sum(
            1
        )
        ok = (cols[None] < visible.long()[live, None].clamp(max=max_positions)) & (addr >= 0)
        s = s.masked_fill(~ok, float("-inf"))
        top = s.topk(min(k, cols_out), dim=-1)
        idx = torch.where(
            top.values > float("-inf"), top.indices, torch.full_like(top.indices, 1 << 30)
        )
        idx = idx.sort(dim=-1).values
        out[live, : idx.shape[1]] = torch.where(idx == 1 << 30, -1, idx).int()
        return out

    def is_valid(
        self,
        q,
        w,
        visible,
        index_pool,
        table,
        ratio,
        topk,
        plan,
        max_positions,
        want_cand=None,
        heuristic=True,
        scheds=None,
    ) -> None:
        _is_valid_queries(q, w, visible, visible, index_pool, table, ratio, topk)
        groups, desc = plan
        assert (
            desc.dtype == torch.int32
            and desc.dim() == 2
            and desc.shape[1] == 5
            and desc.is_contiguous()
        ), "plan: the groups and int32 [pieces, 5] descriptors of `csa2_indexer_plan`"
        assert all(g[1] <= w.shape[0] and g[5] <= desc.shape[0] for g in groups), (
            "the plan covers this step's rows"
        )
        assert visible.dtype in (torch.int32, torch.int64) and max_positions > 0
        assert index_pool.data_ptr() % 16 == 0, "the prefix gather reads 16-byte aligned pages"


csa2_indexer_topk_packed = Csa2IndexerTopkPacked()
