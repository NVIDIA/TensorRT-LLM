# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexer top-k at scale: paged FP4 MQA logits + device-length radix top-k (+ candidate blocks).

Wraps DeepGEMM `fp8_fp4_paged_mqa_logits` and the engine's `torch.ops.trtllm.cute_dsl_indexer_topk_decode`
(the candidate-block kernels are the entries of `csa2_indexer_candidates`). Every length is device
data and every shape depends on `max_positions` only, so `csa2_indexer_topk_decode` is safe under
CUDA-graph capture after one eager warm-up step. `csa2_indexer_topk_prefill` is its eager,
row-tiled form for context chunks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_indexer import csa2_indexer_topk
from .csa2_pack_rows import csa2_pack_rows, csa2_unpack_rows

_INT_MAX = 2**31 - 1
_GVR_ROW_CTAS = (
    4  # the engine's `indexer._GVR_ROW_CTAS`: CTAs per row of the attention-TP decode GVR
)


def _pdl() -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return get_env_enable_pdl()


def pack_queries(q):
    """`q` bf16 `[n, h, 128]` -> `(data int8 [n, h, 64], scale int32 [n, h])`; a `(data, scales uint8 [n, h, 4])` pair
    (already quantized by `csa2_store_rows`) is passed through."""
    if isinstance(q, tuple):
        data, scales = q
        return data, scales.contiguous().view(torch.int32).squeeze(-1)
    n, heads, _ = q.shape
    packed = csa2_pack_rows(q.reshape(n * heads, -1), "index")
    return (
        packed[:, :64].contiguous().view(torch.int8).view(n, heads, 64),
        packed[:, 64:].contiguous().view(torch.int32).view(n, heads),
    )


def dequantize_queries(q) -> torch.Tensor:
    """Queries as bf16 `[n, h, 128]` on the fp4 grid the kernels score (a pair is decoded, bf16 passes through)."""
    if not isinstance(q, tuple):
        return q
    data, scales = q
    n, heads = data.shape[:2]
    rows = torch.cat(
        [data.contiguous().view(torch.uint8).view(n * heads, 64), scales.reshape(n * heads, 4)],
        dim=-1,
    )
    return csa2_unpack_rows(rows, "index", torch.bfloat16).view(n, heads, 128)


def _split_row_heads(rows_per_request: int, heads: int, requests: int) -> int:
    """The engine's `split_row_logits_heads` for six-query requests: heads per query row when a decode batch's paged MQA
    logits run split rows (six 32-head queries of a request as three 64-head rows of the CuTe DSL kernel, one pass over
    its keys), else 0. The engine also splits four-query requests, but only where it then selects from the kernel's
    block maxima (`block_select_batch`), a selection this op does not run, so those keep DeepGEMM's passes here."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import split_row_logits_heads

    if rows_per_request != 6:
        return 0
    # Six-query requests never reach the block-select test, so its width / Top-K / publication arguments are unused.
    return split_row_logits_heads(rows_per_request, heads, requests, 0, 0, 0)


def _logits(
    q,
    w,
    clen_i32,
    index_pool,
    table,
    rid,
    ratio,
    width,
    blocks=None,
    sched=None,
    rows_per_request=1,
    split_rows=False,
):
    """Paged FP4 MQA logits `[n, width]` fp32. `blocks` is the row's native block table (`csa2_indexer_descriptors`);
    without it the table is derived from `table` / `rid` here. `rows_per_request = L > 1`: the rows are requests of `L`
    consecutive rows each (a decode step's token and drafts); as the engine's indexer, the rows of one request share its
    block-table row, so the kernel reads each request's keys once for its `L` queries instead of once per row.

    `split_rows` (with `L > 1`): where the engine's decode indexer runs split rows (`split_row_logits_heads`: six
    32-head queries per request, at least 4 requests), the logits are its CuTe DSL kernel over the request's
    `L * h / 64` 64-head rows and the request's last (longest) visible length, one pass over its keys where DeepGEMM
    makes two; the same bits on every row's visible columns (a row's columns past its own length, which every consumer
    bounds by the row's length, are scored too). `sched` is then ignored:
    the schedule is the engine's, over the requests' lengths."""
    from tensorrt_llm import deep_gemm

    data, scale = pack_queries(q)
    n, heads = data.shape[:2]
    L = rows_per_request
    if split_rows and L > 1 and blocks is not None and _split_row_heads(L, heads, n // L) == heads:
        b, r = n // L, L * heads // 64
        request_context = clen_i32.view(b, L)[
            :, -1
        ].contiguous()  # a request's last row sees its whole context
        sched = deep_gemm.get_paged_mqa_logits_metadata(
            request_context.view(-1, 1),
            64,
            torch.cuda.get_device_properties(data.device).multi_processor_count,
        )
        return torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
            data.view(torch.uint8).view(b, r, 64, -1),
            scale.contiguous().view(b, r, 64),
            index_pool.view(-1, 64, 1, 68),
            w.float().contiguous().view(b * r, 64),
            request_context,
            blocks.view(b, L, -1)[:, 0],
            sched,
            width,
            row_heads=heads,
        )
    if blocks is None:
        sub = (128 // ratio) // 64  # native 64-row pages per GLOBAL page
        pages = table[rid.long()].long()
        native = torch.where(pages >= 0, pages, 0).unsqueeze(-1) * sub + torch.arange(
            sub, device=data.device
        )
        blocks = native.reshape(n, -1).to(torch.int32).contiguous()
    q_data, q_scale = data.view(n, 1, heads, 64), scale.view(n, 1, heads)
    weights = w
    hp = 64 if heads > 32 else 32
    if hp != heads:  # zero weights make the padded heads exact no-ops
        q_data = torch.nn.functional.pad(q_data, (0, 0, 0, hp - heads))
        q_scale = torch.nn.functional.pad(q_scale, (0, hp - heads))
        weights = torch.nn.functional.pad(weights, (0, hp - heads))
    if L > 1:
        b = n // L
        q_data, q_scale = q_data.view(b, L, *q_data.shape[2:]), q_scale.view(b, L, -1)
        blocks = blocks.view(b, L, -1)[:, 0].contiguous()
    if sched is None:  # else the caller's (`csa2_logits_schedule`), built off the critical path
        sched = deep_gemm.get_paged_mqa_logits_metadata(
            clen_i32.view(-1, L),
            64,
            torch.cuda.get_device_properties(data.device).multi_processor_count,
        )
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_data, q_scale),
        index_pool.view(-1, 64, 1, 68),
        weights.float().contiguous(),
        clen_i32.view(-1, L),
        blocks,
        sched,
        width,
    )


def _selfsampling_fits(logits: torch.Tensor, k: int) -> bool:
    """The self-sampling engine's hardware-format gate: fp32 rows,
    float4-aligned stride and base, K in 512/1024/2048."""
    return (
        logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
        and logits.data_ptr() % 16 == 0
        and (logits.shape[0] > 1 or logits.shape[1] % 4 == 0)
        and k in (512, 1024, 2048)
    )


_SELF_SAMPLING_PREFILL_MIN_COLUMNS = (
    8192  # the engine's: narrower prefill scores take its exact radix TopK
)
_PREFILL_TOPK: dict = {}  # (top_k, heuristic) -> (the engine's context TopK, its narrow exact TopK or None)


def _engine_prefill_topk(
    logits: torch.Tensor, c_len: torch.Tensor, top_k: int, heuristic: bool = True
) -> torch.Tensor:
    """The engine's context-row selection (`CSA2Indexer.select_prepared_scores(is_prefill=True)` with the TopK modules
    its indexer configures): scores narrower than 8192 columns on the exact CuTe-DSL radix prefill kernel, wider ones on
    the self-sampling GVR prefill engine (`use_self_sampling_gvr`: a 512/1024/2048-pick indexer on SM100/103/107) when
    `heuristic` (`enable_heuristic_topk`), else on the exact radix kernel too.
    Row `r` selects among its leading `c_len[r]` columns. Returns
    `[rows, top_k]` column picks (unordered, -1 padded)."""
    from tensorrt_llm._torch.attention.backends.sparse.params import use_self_sampling_gvr
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
    from tensorrt_llm._utils import get_sm_version

    sel = _PREFILL_TOPK.get((top_k, heuristic))
    if sel is None:
        sm = get_sm_version()
        exact = (
            TopKImplementation.CUTE_DSL_RADIX
            if IS_CUTLASS_DSL_AVAILABLE and 100 <= sm < 110
            else TopKImplementation.CUDA_RADIX
        )
        gvr = use_self_sampling_gvr(
            enable_heuristic_topk=heuristic,
            use_self_sampling_topk=True,
            index_topk=top_k,
            compress_ratio=1,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=sm,
            is_prefill=True,
        )
        main = TopK(top_k, prefill_implementation=TopKImplementation.CUTE_DSL_GVR if gvr else exact)
        sel = _PREFILL_TOPK[(top_k, heuristic)] = (
            main,
            TopK(top_k, prefill_implementation=exact) if gvr else None,
        )
    main, narrow = sel
    selector = (
        narrow
        if narrow is not None and logits.shape[1] < _SELF_SAMPLING_PREFILL_MIN_COLUMNS
        else main
    )
    scores = logits
    vector = (
        32 // scores.element_size()
    )  # the engine pads storage for the CuTe-DSL kernels' 32-byte copies
    if (
        scores.stride(1) != 1
        or scores.stride(0) % vector
        or scores.data_ptr() % 32
        or scores.shape[1] % vector
    ):
        scores = torch.nn.functional.pad(
            scores.contiguous(), (0, -scores.shape[1] % vector), value=-torch.inf
        )
    out = torch.full((scores.shape[0], top_k), -1, dtype=torch.int32, device=scores.device)
    zeros = torch.zeros(scores.shape[0], dtype=torch.int32, device=scores.device)
    selector(
        scores, out, is_prefill=True, row_starts=zeros, row_ends=c_len.to(torch.int32).contiguous()
    )
    return out


def _sorted_valid(idx: torch.Tensor, c_len: torch.Tensor) -> torch.Tensor:
    """Picks `idx` -> ascending, `-1` where unreachable (`idx` outside `[0, c_len)`)."""
    ok = (idx >= 0) & (idx < c_len.unsqueeze(-1))
    s = torch.where(ok, idx, torch.full_like(idx, _INT_MAX)).sort(dim=-1).values
    return torch.where(s == _INT_MAX, torch.full_like(s, -1), s)


def _segments(pos: torch.Tensor, rid: torch.Tensor):
    """`(first row, one past the last, request row, upper bound of position + 1)` per run of equal `rid`."""
    rids, poss = rid.tolist(), pos.tolist()
    segs, a = [], 0
    for b in range(1, len(rids) + 1):
        if b == len(rids) or rids[b] != rids[a]:
            segs.append((a, b, rids[a], max(poss[a:b]) + 1))
            a = b
    return segs


def _is_valid_queries(q, w, pos, rid, index_pool, table, ratio, topk, bf16_weights=False):
    assert ratio in (1, 2), (
        "the indexer layers have compression ratio 1 or 2 (128 // ratio rows per page)"
    )
    data = q[0] if isinstance(q, tuple) else q
    if isinstance(q, tuple):
        assert (
            data.dtype == torch.int8
            and q[1].dtype == torch.uint8
            and q[1].shape == (*data.shape[:2], 4)
        ), "a quantized q is (data int8 [n, h, 64], scales uint8 [n, h, 4])"
        assert data.shape[-1] == 64
    else:
        assert q.dtype == torch.bfloat16 and q.shape[-1] == 128, "q is bf16 [n, h, 128]"
    assert data.dim() == 3 and data.shape[1] in (32, 64), (
        "the logits kernel is certified at 32 or 64 index heads"
    )
    assert w.shape == data.shape[:2] and (
        w.dtype == torch.float32 or (bf16_weights and w.dtype == torch.bfloat16)
    ), "w is the fp32 (a candidate consumer: or bf16), already scaled head weights"
    assert pos.shape == rid.shape == (data.shape[0],)
    assert (
        index_pool.dtype == torch.uint8
        and index_pool.shape[1:] == (128 // ratio, 68)
        and index_pool.is_contiguous()
    ), "the index pool is [pages, 128 // ratio, 68] uint8"
    assert table.dim() == 2 and topk > 0


class Csa2IndexerTopkDecode(OpWrapper):
    """Per query row the `topk` best compressed positions (ascending, `-1` = unreachable) for any row count.

    Same picks as `csa2_indexer_topk`. `q` is `[n, h, 128]` bf16 (rotated, quantize-dequantized) or the
    pair `csa2_store_rows` quantized, `w` `[n, h]` fp32 (already scaled head weights; a consumer may hand them as
    bf16, the sparse logits kernel's own dtype, cast where they are produced), `pos` `[n]` int32,
    `rid` `[n]` the request's table row, `index_pool` `[pages, 128 // ratio, 68]` uint8 (the engine's native
    footer pages), `table` the page table. Score of key `j`: `sum_h relu(q_h . k_j) * w_h`, reachable when
    `j < (pos + 1) // ratio`. `max_positions` bounds the scored width (a host constant, so shapes stay static).

    `want_cand=(block, blocks)` (candidate-source layer): also returns `(positions [n, cw], block_ids
    [n, cw // 8], counts [n, 1])` int32, the `blocks` best blocks of `block` keys per query expanded to ascending
    positions, the newest reachable block pinned. `cand_in` (a consumer layer) is such a triple of its source:
    only those positions are scored and the top-k is taken among them. Returns `(picks, candidates or None)`.

    `split_rows` (a decode step with `desc` and `rows_per_request`): where the engine runs split-row paged logits (six
    32-head rows per request, at least 4 requests), the logits are its CuTe DSL kernel (one pass over a request's keys)
    instead of DeepGEMM's; the same bits on every row's visible columns, so the same picks.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="ratio 1, three requests of four draft rows",
            spec=dict(ratio=1, nreq=3, rows=4, cand=False),
        ),
        Cell(
            why="ratio 2, three requests of four draft rows",
            spec=dict(ratio=2, nreq=3, rows=4, cand=False),
        ),
        Cell(
            why="ratio 1, 200 requests (a dep8 batch), long histories",
            spec=dict(ratio=1, nreq=200, rows=4, cand=False),
        ),
        Cell(
            why="ratio 2, 200 requests (a dep8 batch), long histories",
            spec=dict(ratio=2, nreq=200, rows=4, cand=False),
        ),
        Cell(
            why="candidate source then consumer: 24 requests of two rows, 40 blocks of 8 keys",
            spec=dict(ratio=1, nreq=24, rows=2, cand=True),
        ),
    )
    note = """
    Picks agree with the dense reference up to ties at the k-th score; unreachable picks (-1) agree
    exactly. The consumer path scores in bf16 (the sparse logits kernel takes bf16 head weights), so its
    test states a score band instead of an overlap. The reference returns the candidate set as per-request
    position masks, not as the kernel's expanded triple.
    """

    def __call__(
        self,
        q,
        w: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        max_positions: Optional[int] = None,
        cand_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        want_cand: Optional[Tuple[int, int]] = None,
        desc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        heuristic: bool = False,
        cand_sched: Optional[torch.Tensor] = None,
        publish_stream: Optional[torch.cuda.Stream] = None,
        publish_done: Optional[torch.cuda.Event] = None,
        logits_sched: Optional[torch.Tensor] = None,
        rows_per_request: int = 1,
        mask_unmapped: bool = False,
        cand_decode_topk: bool = False,
        split_rows: bool = False,
        prefill_select: bool = False,
        tp_chain: bool = False,
        masked_blocks: Optional[torch.Tensor] = None,
    ) -> tuple:
        from tensorrt_llm import deep_gemm  # noqa: F401  (availability)

        n = w.shape[0]
        rpp = 128 // ratio
        width = table.shape[1] * rpp
        if max_positions is not None:
            width = min(width, -(-max(1, max_positions) // 8) * 8)
        if (
            desc is not None
        ):  # the owner's per-row descriptors, built once per step (`csa2_indexer_descriptors`)
            blocks, context, clen_i32, valid = desc
        else:
            blocks, context, valid = None, None, None
            clen_i32 = ((pos.long() + 1) // ratio).clamp(min=0).to(torch.int32).contiguous()
            context = clen_i32.clamp(min=1)
        cands = None
        if cand_in is not None:
            from .csa2_indexer_candidates import consume_candidates

            return consume_candidates(
                q, w, rid, index_pool, table, ratio, topk, cand_in, clen_i32, blocks, cand_sched
            ), None
        rpr = (
            rows_per_request
            if rows_per_request > 1 and n % rows_per_request == 0 and blocks is not None
            else 1
        )
        logits = _logits(
            q,
            w,
            context,
            index_pool,
            table,
            rid,
            ratio,
            width,
            blocks if masked_blocks is None else masked_blocks,
            logits_sched,
            rpr,
            split_rows,
        )
        if mask_unmapped and masked_blocks is None:
            # as the engine's decode selection (`_select_paged_logits`): the columns of unallocated pages (scored on
            # page 0 by the paged kernel) are masked in place among each row's visible ones before any top-k reads them
            from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import mask_logits_

            mask_logits_(logits, valid, row_lengths=clen_i32)
        if want_cand is not None:
            from .csa2_indexer_candidates import publish_candidates

            if publish_stream is None:
                cands = publish_candidates(
                    logits, table, rid, ratio, clen_i32, *want_cand, valid, cand_decode_topk
                )
            else:
                # Only later consumer layers read the candidates (as the engine's source layer): publish them on
                # `publish_stream` next to this layer's selection; `publish_done` marks them ready for the caller.
                ready = torch.cuda.Event()
                ready.record()
                with torch.cuda.stream(publish_stream):
                    publish_stream.wait_event(ready)
                    cands = publish_candidates(
                        logits, table, rid, ratio, clen_i32, *want_cand, valid, cand_decode_topk
                    )
                    publish_done.record()
                for t in (logits, clen_i32, *((valid,) if valid is not None else ())):
                    t.record_stream(publish_stream)
                for t in cands:
                    t.record_stream(torch.cuda.current_stream())
        k = min(topk, width)
        if prefill_select:
            # context rows select as the engine's context indexer (`_run_packed_group` -> `select_prepared_scores` with
            # `is_prefill`): its kernels, its picks on tied scores
            picks = _engine_prefill_topk(logits, clen_i32, topk, heuristic)
            if valid is None:
                return _sorted_valid(picks, clen_i32.long())[:, :k].contiguous(), cands
            from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
                finalize_selection_,
            )

            return finalize_selection_(picks, width, valid=valid, row_lengths=clen_i32)[
                :, :k
            ].contiguous(), cands
        gvr = heuristic and desc is not None and _selfsampling_fits(logits, k)
        # the self-sampling engine writes every entry of a row (a row of n_r <= k visible keys: identity, then -1), as
        # the engine's preallocated selection buffer relies on; the radix kernel's rows start from -1
        if gvr:
            out = torch.empty((n, k), dtype=torch.int32, device=w.device)
        else:
            out = torch.full((n, k), -1, dtype=torch.int32, device=w.device)
        if gvr:
            # the engine's hint-free self-sampling GVR top-k (`enable_heuristic_topk`): exact up to ties, one launch,
            # no cross-step state; per-row lengths are the visible lengths, the order of the picks is unspecified
            from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
                paged_finalize_sorts,
            )
            from tensorrt_llm._torch.kernels.blackwell.top_k import selfsampling_topk_run_varlen

            launch = {}
            fold = None
            if tp_chain:
                # Attention TP (the built-in `CSA2Indexer._select_paged_logits` TP selection chain, #22366): per row, a
                # programmatic dependent of the logits, at most four CTAs per row (the cluster family), whose rank-0 CTA
                # also runs the paged finalize below when this width sorts past the finalize's bitmap.
                launch = dict(pdl=_pdl(), r_cap=_GVR_ROW_CTAS)
                if valid is not None and paged_finalize_sorts(width) and k % 2 == 0 and k <= 1024:
                    fold = dict(valid=valid, visible=clen_i32, width=width, page=64, done=False)
                    launch["finalize"] = fold
            if rpr > 1 and ratio in (1, 4) and not tp_chain:
                # the self-sampling engine's compression ratios (ratio-2 layers: per row); as the engine's decode
                # top-k: the rows of one request in one group (`next_n`), each request's KV length in token space
                # (row j of a request sees `(kv - L + j + 1) // ratio` compressed positions)
                kv = (pos.view(-1, rpr)[:, -1].to(torch.int32) + 1).contiguous()
                selfsampling_topk_run_varlen(
                    logits,
                    kv,
                    out,
                    next_n=rpr,
                    compress_ratio=ratio,
                    max_seq_len=width * ratio,
                    wide_cta_rows=True,
                )
            else:
                selfsampling_topk_run_varlen(
                    logits,
                    clen_i32,
                    out,
                    next_n=1,
                    compress_ratio=1,
                    max_seq_len=width,
                    wide_cta_rows=True,
                    **launch,
                )
            if fold is not None and fold["done"]:
                return out, cands
        else:
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                logits, clen_i32, out, k, 1, wide_fp32_rows=True, split_long_rows=True
            )
        if valid is None:
            return _sorted_valid(out, clen_i32.long()), cands
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import finalize_selection_

        # one launch: drops picks past the row's length or on an unmapped page, sorts ascending, pads with -1
        return finalize_selection_(
            out, width, valid=valid, row_lengths=clen_i32, shuffle=tp_chain
        ), cands

    def reference(
        self,
        q,
        w: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        max_positions: Optional[int] = None,
        cand_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        want_cand: Optional[Tuple[int, int]] = None,
        desc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        heuristic: bool = False,
        cand_sched: Optional[torch.Tensor] = None,
        publish_stream: Optional[torch.cuda.Stream] = None,
        publish_done: Optional[torch.cuda.Event] = None,
        logits_sched: Optional[torch.Tensor] = None,
        rows_per_request: int = 1,
        mask_unmapped: bool = False,
        cand_decode_topk: bool = False,
        split_rows: bool = False,
        prefill_select: bool = False,
        tp_chain: bool = False,
        masked_blocks: Optional[torch.Tensor] = None,
    ):
        """The dense per-request oracle `csa2_indexer_topk` (segments rebuilt from `rid` on the host).

        With `cand_in`, the candidate consumer's reference. With `want_cand` the second result is the
        oracle's per-request candidate masks, not the kernel's `(positions, ids, counts)`. `max_positions`
        is ignored. Picks are `[n, min(topk, widest span)]`.
        """
        if cand_in is not None:
            from .csa2_indexer_candidates import consume_candidates

            return consume_candidates.reference(
                q,
                w,
                rid,
                index_pool,
                table,
                ratio,
                topk,
                cand_in,
                ((pos.long() + 1) // ratio).clamp(min=0),
            ), None
        block, blocks = want_cand if want_cand is not None else (0, 0)
        return csa2_indexer_topk(
            dequantize_queries(q), w, pos, _segments(pos, rid), index_pool, table, ratio, topk,
            block, blocks, None, want_cand is not None,
        )  # fmt: skip

    def is_valid(
        self,
        q,
        w: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        max_positions: Optional[int] = None,
        cand_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        want_cand: Optional[Tuple[int, int]] = None,
        desc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        heuristic: bool = False,
        cand_sched: Optional[torch.Tensor] = None,
        publish_stream: Optional[torch.cuda.Stream] = None,
        publish_done: Optional[torch.cuda.Event] = None,
        logits_sched: Optional[torch.Tensor] = None,
        rows_per_request: int = 1,
        mask_unmapped: bool = False,
        cand_decode_topk: bool = False,
        split_rows: bool = False,
        prefill_select: bool = False,
        tp_chain: bool = False,
        masked_blocks: Optional[torch.Tensor] = None,
    ) -> None:
        _is_valid_queries(
            q, w, pos, rid, index_pool, table, ratio, topk, bf16_weights=cand_in is not None
        )
        if desc is not None:
            blocks, context, vis, valid = desc
            n = w.shape[0]
            assert (
                blocks.dtype == torch.int32 and blocks.shape[0] == n and context.shape == (n, 1)
            ), "desc is the `csa2_indexer_descriptors` tuple of this step's rows"
            assert vis.dtype == torch.int32 and vis.shape == (n,) and valid.dtype == torch.bool
        assert not mask_unmapped or (desc is not None and cand_in is None), (
            "mask_unmapped masks the scored columns with the descriptors' page validity (not the consumer path)"
        )
        if masked_blocks is not None:
            assert mask_unmapped and not split_rows, (
                "masked_blocks replaces mask_unmapped's launch (DeepGEMM paged logits only)"
            )
            assert masked_blocks.dtype == torch.int32 and masked_blocks.shape == desc[0].shape, (
                "masked_blocks is the descriptors' DeepGEMM-masked table, laid out as their blocks"
            )
        if want_cand is not None:
            assert want_cand[0] == 8, (
                "candidate blocks are 8 keys (the sparse logits kernel's block)"
            )
        if cand_in is not None:
            assert (
                len(cand_in) == 3
                and cand_in[0].dtype == torch.int32
                and cand_in[0].shape[0] == w.shape[0]
            ), (
                "cand_in is the (positions, block_ids, counts) triple of a candidate-source layer, one row per query"
            )


class Csa2IndexerTopkPrefill(OpWrapper):
    """`csa2_indexer_topk_decode` for context chunks: rows are scored in tiles of `tile` rows.

    Tiling bounds the logits buffer to `tile x max_positions` fp32, with each row's own causal length;
    per-tile results are concatenated. `max_positions` is a host upper bound of the compressed positions any
    row reaches (`ceil(span / ratio)`, rounded up to 8); it fixes the logits and candidate widths, so tiles
    agree. Eager only (a tile loop and host sizes: no CUDA graph). `desc` (optional) is the rows' descriptors
    (`csa2_indexer_descriptors`, built once per owner and step): the page gathers are not redone per layer and the
    picks are finalized in one launch. `heuristic` is the engine's `enable_heuristic_topk`: wide rows select on the
    self-sampling GVR engine when set, on the exact radix kernel otherwise (exact either way, up to ties). A caller
    should hand only the table columns the step's spans reach (the engine's tables are `max_seq_len / 128` columns
    wide; every per-row gather is that wide).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="ratio 1: a 330-row chunk after 700 tokens of history, tiles of 100 rows",
            spec=dict(ratio=1, hist=700, n=330, tile=100),
        ),
        Cell(
            why="ratio 2: a 330-row chunk after 1400 tokens of history, tiles of 100 rows",
            spec=dict(ratio=2, hist=1400, n=330, tile=100),
        ),
        Cell(
            why="dep2 context step: the step's descriptors, a table narrowed to the columns the span reaches, ratio 1",
            spec=dict(ratio=1, hist=900, n=600, tile=256, desc=True),
        ),
        Cell(
            why="dep2 context step: the step's descriptors, ratio 2",
            spec=dict(ratio=2, hist=1500, n=400, tile=256, desc=True),
        ),
    )
    note = "Same bands as `csa2_indexer_topk_decode`, candidate source and consumer included."

    def __call__(
        self,
        q,
        w: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        max_positions: int,
        cand_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        want_cand: Optional[Tuple[int, int]] = None,
        tile: int = 1024,
        desc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        heuristic: bool = True,
    ) -> tuple:
        n = w.shape[0]
        outs, cands = [], []
        for a in range(0, n, tile):
            b = min(a + tile, n)
            qa = (q[0][a:b], q[1][a:b]) if isinstance(q, tuple) else q[a:b]
            ci = None if cand_in is None else tuple(t[a:b] for t in cand_in)
            da = None if desc is None else tuple(t[a:b] for t in desc)
            o, c = csa2_indexer_topk_decode(
                qa,
                w[a:b],
                pos[a:b],
                rid[a:b],
                index_pool,
                table,
                ratio,
                topk,
                max_positions,
                ci,
                want_cand,
                da,
                heuristic,
                prefill_select=True,
            )
            outs.append(o)
            cands.append(c)
        cand = (
            None if want_cand is None else tuple(torch.cat([c[i] for c in cands]) for i in range(3))
        )
        return torch.cat(outs), cand

    def reference(
        self,
        q,
        w: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        max_positions: int,
        cand_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        want_cand: Optional[Tuple[int, int]] = None,
        tile: int = 1024,
        desc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        heuristic: bool = True,
    ) -> torch.Tensor:
        """`csa2_indexer_topk_decode.reference` over the whole chunk (tiling and `desc` do not change the result)."""
        return csa2_indexer_topk_decode.reference(
            q, w, pos, rid, index_pool, table, ratio, topk, max_positions, cand_in, want_cand
        )

    def is_valid(
        self,
        q,
        w: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        max_positions: int,
        cand_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        want_cand: Optional[Tuple[int, int]] = None,
        tile: int = 1024,
        desc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        heuristic: bool = True,
    ) -> None:
        # a candidate consumer takes bf16 head weights, as the decode form
        _is_valid_queries(
            q, w, pos, rid, index_pool, table, ratio, topk, bf16_weights=cand_in is not None
        )
        assert tile > 0 and max_positions % 8 == 0, (
            "max_positions is the span rounded up to a multiple of 8"
        )
        if (
            desc is not None
        ):  # `csa2_indexer_descriptors(table, rid, (pos + 1) // ratio, ratio)` of these rows
            assert len(desc) == 4 and desc[0].shape == (
                w.shape[0],
                table.shape[1] * (128 // ratio // 64),
            )


csa2_indexer_topk_decode = Csa2IndexerTopkDecode()
csa2_indexer_topk_prefill = Csa2IndexerTopkPrefill()
