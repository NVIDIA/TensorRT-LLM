# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-block publication and consumption for the generation indexer (device-side, graph safe).

Two entries wrapping the engine's CuTe glue kernels `candidate_block_max_` / `expand_candidates_`, the
radix top-k `torch.ops.trtllm.indexer_topk_prefill` over block maxima, and DeepGEMM
`fp8_fp4_paged_sparse_mqa_logits`:

- `publish_candidates`: the candidate-source layer's second result. The `nblocks` best blocks of
  8 keys per query by block-max logit (the newest reachable block pinned), expanded to positions.
- `consume_candidates`: a consumer layer scores only those positions and takes the top-k among them.

Both are driven through `csa2_indexer_topk_decode` / `csa2_indexer_topk_prefill` by the targets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_indexer import csa2_candidate_blocks
from .csa2_pack_rows import csa2_unpack_rows
from .csa2_rows_io import csa2_gather_rows_footer, csa2_row_addr

SPARSE_BLOCK = 8  # keys per block of the sparse logits kernel (the candidate block size)
_DECODE_TOPK_MAX_ROWS = 64  # the engine's `_CANDIDATE_CLUSTER_TOPK_MAX_ROWS`
_NATIVE = 64
_CELLS_SPEC = (
    Cell(
        why="ratio 1, 24 requests of two draft rows (the generation shape), 40 blocks of 8 keys",
        spec=dict(ratio=1, nreq=24, rows=2, block=8, blocks=40),
    ),
    Cell(
        why="ratio 2, a 330-row context chunk after 1400 tokens, 50 blocks of 8 keys",
        spec=dict(ratio=2, hist=1400, n=330, block=8, blocks=50),
    ),
)


def _valid_pages(table, rid, ratio):
    sub = (128 // ratio) // _NATIVE
    return (table[rid.long()] >= 0).repeat_interleave(sub, dim=1).contiguous()


class Csa2PublishCandidates(OpWrapper):
    """Top `nblocks` blocks of `block` keys per query by block-max logit (newest reachable block pinned).

    `logits` `[n, width]` fp32 (`fp8_fp4_paged_mqa_logits` of the layer), `table` / `rid` the page table and each
    row's request, `c_len` `[n]` int32 compressed length of each row (`(pos + 1) // ratio`), `block` 8.
    Returns `(positions [n, cw] int32 ascending and -1 padded, block_ids [n, cw // 8] int32, counts [n, 1]
    int32)`; `cw` is `nblocks * block` rounded up to a multiple of 32, whatever `width` is (positions past a row's
    kept blocks are -1, so a consumer sizes its picks by `cw`).

    `decode_topk` (a decode tile of at most 64 rows): the block top-k is the engine's candidate selector at decode
    (`CSA2Indexer.candidate_top_k`, CuTe-DSL radix, `cute_dsl_indexer_topk_decode`: several CTAs per wide row) instead
    of the one-CTA-per-row prefill radix, which grows with the history; the block maxima are padded with -inf to the
    kernel's 8-float vectors as the engine pads them. Same selection up to ties at the k-th block maximum.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _CELLS_SPEC
    note = """
    `reference` returns the kept positions as a bool mask `[n, width]` (the same selection in the
    plain-torch `csa2_candidate_blocks` form), not the expanded triple; ties between blocks of equal
    maximum may be resolved differently, so the test gates the share of agreeing positions.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        table: torch.Tensor,
        rid: torch.Tensor,
        ratio: int,
        c_len: torch.Tensor,
        block: int,
        nblocks: int,
        valid: Optional[torch.Tensor] = None,
        decode_topk: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
            candidate_block_max_,
            expand_candidates_,
        )

        n, width = logits.shape
        dev = logits.device
        if valid is None:  # the step's descriptors (`csa2_indexer_descriptors`) already hold it
            valid = _valid_pages(table, rid, ratio)
        nb = -(-width // block)
        # the engine's candidate selector at decode (`select_prepared_scores`): the multi-CTA block top-k
        decode = decode_topk and n <= _DECODE_TOPK_MAX_ROWS and nb > nblocks
        # As the engine's source layer (`_publish_block_candidates`, `_aligned_candidate_blocks`,
        # `_candidate_fill_on_publish`): at decode the block maxima fill whole 8-float vectors (the kernel writes -inf
        # past each row's blocks), the zero row starts come from a buffer filled once, and the selection and the
        # publication are written whole by their kernels, so a decode step launches no fill or pad.
        blocks = torch.empty(n, -(-nb // 8) * 8 if decode else nb, dtype=torch.float32, device=dev)
        counts_b = torch.empty(n, dtype=torch.int32, device=dev)
        zeros = _zero_rows(n, dev)
        candidate_block_max_(
            logits,
            valid,
            valid.stride(0),
            c_len,
            0,
            zeros,
            c_len,
            c_len,
            blocks,
            counts_b,
            block,
            _NATIVE,
        )
        # Always the full candidate width: the sparse-logits kernels are JIT-compiled per candidate width, and a width
        # that followed a short context's `nb` would compile (about 2 s) once per prompt length. Unused columns are
        # published -1 by the expansion.
        cw = nblocks * block
        cw = -(-cw // (4 * SPARSE_BLOCK)) * 4 * SPARSE_BLOCK
        if decode:
            selected = torch.empty((n, nblocks), dtype=torch.int32, device=dev)
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                blocks, counts_b, selected, nblocks, 1, wide_fp32_rows=True, split_long_rows=True
            )
        else:
            selected = torch.full((n, nblocks), -1, dtype=torch.int32, device=dev)
            torch.ops.trtllm.indexer_topk_prefill(blocks, zeros, counts_b, selected, nblocks)
        positions = torch.empty((n, cw), dtype=torch.int32, device=dev)
        ids = torch.empty(n, cw // SPARSE_BLOCK, dtype=torch.int32, device=dev)
        cnt = torch.empty(n, 1, dtype=torch.int32, device=dev)
        expand_candidates_(
            selected, blocks, c_len, 0, c_len, positions, block, ids, cnt.view(-1), SPARSE_BLOCK
        )
        return positions, ids, cnt

    def reference(
        self,
        logits: torch.Tensor,
        table: torch.Tensor,
        rid: torch.Tensor,
        ratio: int,
        c_len: torch.Tensor,
        block: int,
        nblocks: int,
        valid: Optional[torch.Tensor] = None,
        decode_topk: bool = False,
    ) -> torch.Tensor:
        """Bool mask `[n, width]` of the kept positions: unreachable positions (past `c_len`, or on an
        unmapped page) score -inf, then `csa2_candidate_blocks`."""
        n, width = logits.shape
        cols = torch.arange(width, device=logits.device)
        pages = table[rid.long()]
        mapped = pages[:, (cols // (128 // ratio)).clamp(max=pages.shape[1] - 1)] >= 0
        reach = (cols < c_len.long().unsqueeze(-1)) & mapped
        score = logits.masked_fill(~reach, float("-inf"))
        return csa2_candidate_blocks(score, c_len.long().unsqueeze(-1), nblocks, block)

    def is_valid(
        self,
        logits: torch.Tensor,
        table: torch.Tensor,
        rid: torch.Tensor,
        ratio: int,
        c_len: torch.Tensor,
        block: int,
        nblocks: int,
        valid: Optional[torch.Tensor] = None,
        decode_topk: bool = False,
    ) -> None:
        assert valid is None or (valid.dtype == torch.bool and valid.shape[0] == logits.shape[0]), (
            "valid is the per-native-page bool of `csa2_indexer_descriptors`"
        )
        assert block == SPARSE_BLOCK, (
            "candidate blocks are 8 keys, the block of the sparse logits kernel"
        )
        assert ratio in (1, 2) and logits.dtype == torch.float32 and logits.dim() == 2
        assert c_len.dtype == torch.int32 and c_len.shape == (logits.shape[0],)
        assert nblocks > 0


_ROW_IDS: dict = {}
# device index -> every zero buffer handed out, newest last: an outgrown buffer stays alive, since CUDA graphs captured
# while it was the newest read it by address on every replay.
_ZERO_ROWS: dict = {}


def _zero_rows(n: int, device) -> torch.Tensor:
    """`n` int32 zeros from a per-device buffer filled once outside graph capture and never written (the engine's
    `CSA2Indexer._zero_rows`: no fill launch in a decode graph). A larger `n` (an eager context step past every captured
    decode batch) adds a larger buffer and keeps the outgrown ones, which captured graphs still read."""
    buffers = _ZERO_ROWS.setdefault(torch.device(device).index, [])
    if not buffers or buffers[-1].numel() < n:
        if torch.cuda.is_current_stream_capturing():
            return torch.zeros(n, dtype=torch.int32, device=device)
        size = max(n, 256, 2 * buffers[-1].numel() if buffers else 0)
        buffers.append(torch.zeros(size, dtype=torch.int32, device=device))
    return buffers[-1][:n]


def _row_ids(n: int, device) -> torch.Tensor:
    """`arange(n)` int32, made once per (device, n) outside graph capture (one launch under capture otherwise)."""
    key = (torch.device(device).index, n)
    ids = _ROW_IDS.get(key)
    if ids is None:
        ids = torch.arange(n, dtype=torch.int32, device=device)
        if not torch.cuda.is_current_stream_capturing():
            _ROW_IDS[key] = ids
    return ids


def consumer_select_fused(rows: int, cand_in, topk: int) -> bool:
    """Whether a consumer's decode rows over the candidate triple `cand_in` take the one-launch select
    (`_reindex_select`), which needs no paged sparse-logits schedule (the shape rule of `reindex_select_applies`)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.reindex_select import (
        reindex_select_applies,
    )

    positions, ids = cand_in[0], cand_in[1]
    return reindex_select_applies(rows, ids.shape[1] * SPARSE_BLOCK, min(topk, positions.shape[1]))


def _reindex_select(
    data, scale, weights, k_pages, native, ids, counts, positions, k
) -> Optional[torch.Tensor]:
    """The candidate logits and the mapped top-k of decode rows in one launch, one cluster per row (the engine's
    `reindex_select_`, the built-in Reindex consumers), the two launches' selection; None where it does not apply
    (shape, row count, the two-launch path not on its fused selection) or would compile under capture."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.reindex_select import (
        reindex_select_,
        reindex_select_applies,
    )

    n, heads = data.shape[:2]
    if not (
        ids.stride(1) == 1
        and positions.is_contiguous()
        and positions.shape[1] >= ids.shape[1] * SPARSE_BLOCK
        and positions.data_ptr() % 16 == 0
        and positions.stride(0) % 4 == 0
        and native.dtype == torch.int32
        and native.stride(1) == 1
        and reindex_select_applies(n, ids.shape[1] * SPARSE_BLOCK, k)
    ):
        return None
    out = torch.empty(n, k, dtype=torch.int32, device=data.device)
    rows = _row_ids(n, data.device)  # the block table holds one row per query row
    ran = reindex_select_(
        data.contiguous(),
        scale.contiguous().view(torch.int32).reshape(n, heads),
        weights.contiguous(),
        k_pages,
        native,
        rows,
        ids,
        counts,
        positions,
        out,
    )
    return out if ran else None


class Csa2ConsumeCandidates(OpWrapper):
    """Score only the source layer's candidate keys (paged sparse FP4 MQA logits) and take the top-k among them.

    `q` is the layer's queries (bf16 `[n, h, 128]` on the fp4 grid, or the pair `csa2_store_rows` quantized),
    `w` `[n, h]` fp32 head weights (the kernel takes them as bf16; bf16 weights, cast where they are produced, are
    taken as they are), `cand_in` the triple of
    `publish_candidates`, `c_len` `[n]` int32. Returns `[n, min(topk, cw)]` int32 key positions, ascending, `-1`
    padded. Decode rows of the checkpoint's shape (2048 blocks of 8 keys, top-512, at most one wave of clusters) run
    the scoring and the selection in one launch (`_reindex_select`, the engine's `reindex_select_`), the two launches'
    selection; `fused=False` keeps the two launches.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _CELLS_SPEC + (
        Cell(
            why="tep8 Reindex decode: six rows, 2048 candidate blocks of 8 keys, top-512 (the one-launch select)",
            spec=dict(ratio=1, nreq=1, rows=6, block=8, blocks=2048, hist=20000, topk=512),
        ),
    )
    note = """
    The kernel returns bf16 scores and takes bf16 head weights, so a pick may rank below the exact k-th best
    candidate only by scores within about 2^-7 of the row's largest; the test states that band instead of
    an overlap with the reference. `reference` scores every candidate exactly (bf16-rounded weights) and
    takes the top-k of them.
    """

    def __call__(
        self,
        q,
        w: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        cand_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        c_len: torch.Tensor,
        blocks: Optional[torch.Tensor] = None,
        sched: Optional[torch.Tensor] = None,
        fused: bool = True,
    ) -> torch.Tensor:
        from tensorrt_llm.deep_gemm import (
            fp8_fp4_paged_sparse_mqa_logits,
            get_paged_sparse_mqa_logits_metadata,
        )

        from .csa2_indexer_decode import pack_queries

        positions, ids, cnt = cand_in
        data, scale = pack_queries(q)
        n, heads = data.shape[:2]
        dev = data.device
        if blocks is None:
            sub = (128 // ratio) // _NATIVE
            pages = table[rid.long()].long()
            native = torch.where(pages >= 0, pages, 0).unsqueeze(-1) * sub + torch.arange(
                sub, device=dev
            )
            native = native.reshape(n, -1).to(torch.int32).contiguous()
        else:
            native = blocks
        q_data, q_scale = data.view(n, 1, heads, 64), scale.view(n, 1, heads)
        weights = w.to(torch.bfloat16)
        counts = cnt.reshape(-1)
        k_pages = index_pool.view(-1, _NATIVE, 1, 68)
        if fused:
            k_fused = min(topk, positions.shape[1])
            out = _reindex_select(
                data, scale, weights, k_pages, native, ids, counts, positions, k_fused
            )
            if out is not None:
                return out
        if (
            sched is None
        ):  # else the step's schedule, shared by its consumer layers (`csa2_consumer_schedule`)
            rows = torch.arange(n, dtype=torch.int32, device=dev)
            sched = get_paged_sparse_mqa_logits_metadata(
                counts, native, rows, k_pages.shape[1], ids, torch.int8, SPARSE_BLOCK
            )
        scores = fp8_fp4_paged_sparse_mqa_logits(
            (q_data, q_scale), k_pages, weights, sched, ids.shape[1], SPARSE_BLOCK
        )
        cw = positions.shape[1]
        k = min(topk, cw)
        from tensorrt_llm._torch.attention.backends.sparse.csa2.decode_select import (
            fused_consumer_topk_supported,
            select_mapped_consumer_topk_,
        )

        if (
            scores.dtype == torch.bfloat16
            and scores.shape[1] <= cw
            and fused_consumer_topk_supported(scores, k)
        ):
            # one launch: top-k of each row's leading `counts` scores, mapped to positions, ascending, -1 padded
            # (the published positions are all reachable: both layers have the row's visible length)
            return select_mapped_consumer_topk_(
                scores, counts, positions, torch.empty(n, k, dtype=torch.int32, device=dev)
            )
        scores = scores.to(torch.float32, memory_format=torch.contiguous_format)
        sel = torch.full((n, k), -1, dtype=torch.int32, device=dev)
        zeros = torch.zeros(n, dtype=torch.int32, device=dev)
        ncol = counts.clamp(max=scores.shape[1]).to(torch.int32)  # counts are valid columns
        torch.ops.trtllm.indexer_topk_prefill(scores, zeros, ncol, sel, k)
        live = (sel >= 0) & (
            sel < ncol.unsqueeze(-1)
        )  # the selector leaves unwritten slots on short rows
        picked = torch.gather(positions, 1, sel.long().clamp(0, cw - 1))
        picked = torch.where(live & (picked >= 0) & (picked < c_len.unsqueeze(-1)), picked, -1)
        big = torch.iinfo(torch.int32).max
        s = torch.where(picked >= 0, picked, big).sort(dim=-1).values
        return torch.where(s == big, -1, s)

    def reference(
        self,
        q,
        w: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        cand_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        c_len: torch.Tensor,
        blocks: Optional[torch.Tensor] = None,
        sched: Optional[torch.Tensor] = None,
        fused: bool = True,
    ) -> torch.Tensor:
        from .csa2_indexer_decode import dequantize_queries

        positions = cand_in[0]
        qb = dequantize_queries(q).float()
        pos_c = positions.long().clamp(min=0)
        addr = csa2_row_addr(table, rid.long().unsqueeze(-1).expand_as(pos_c), pos_c, 128 // ratio)
        keys = csa2_unpack_rows(csa2_gather_rows_footer(index_pool, addr), "index", torch.float32)
        score = torch.einsum("nhd,ncd->nhc", qb, keys).relu() * w.to(
            torch.bfloat16
        ).float().unsqueeze(-1)
        score = score.sum(1)
        ok = (positions >= 0) & (positions.long() < c_len.long().unsqueeze(-1)) & (addr >= 0)
        score = score.masked_fill(~ok, float("-inf"))
        top = score.topk(min(topk, positions.shape[1]), dim=-1)
        picked = torch.gather(positions, 1, top.indices)
        picked = torch.where(torch.isfinite(top.values), picked, torch.full_like(picked, -1))
        big = torch.iinfo(torch.int32).max
        s = torch.where(picked >= 0, picked, torch.full_like(picked, big)).sort(dim=-1).values
        return torch.where(s == big, torch.full_like(s, -1), s)

    def is_valid(
        self,
        q,
        w: torch.Tensor,
        rid: torch.Tensor,
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        cand_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        c_len: torch.Tensor,
        blocks: Optional[torch.Tensor] = None,
        sched: Optional[torch.Tensor] = None,
        fused: bool = True,
    ) -> None:
        assert blocks is None or (blocks.dtype == torch.int32 and blocks.shape[0] == w.shape[0]), (
            "blocks is the int32 native block table of `csa2_indexer_descriptors`"
        )
        assert ratio in (1, 2) and topk > 0
        assert len(cand_in) == 3 and cand_in[0].dtype == torch.int32 and cand_in[0].dim() == 2, (
            "cand_in is the (positions, block_ids, counts) triple of publish_candidates"
        )
        assert cand_in[0].shape[1] % (4 * SPARSE_BLOCK) == 0 and cand_in[0].shape[0] == w.shape[0]
        assert index_pool.dtype == torch.uint8 and index_pool.shape[1:] == (128 // ratio, 68)
        assert c_len.dtype == torch.int32 and c_len.shape == (w.shape[0],)
        assert w.dtype in (torch.float32, torch.bfloat16), (
            "w is the fp32 head weights (or their bf16 cast)"
        )


class Csa2PrefetchCandidates(OpWrapper):
    """Start reading into L2 the native index rows (data and scales) of each decode row's published candidate blocks
    (`cand_in`, the source layer's `publish_candidates` triple: block ids `[n, cw / 8]`, counts `[n, 1]`), the rows a
    consumer layer is about to score: the engine's `prefetch_candidate_blocks_` (`CandidateBlockPrefetchKernel`), which
    the built-in Reindex consumer issues on a side stream while the layer projects its queries (#22465). `index_pool`
    is the owner's index pool `[pages, 128 // ratio, 68]` uint8, `blocks` the rows' native block table
    (`csa2_indexer_descriptors`). Reads only: nothing is written."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 Reindex decode: six rows, 2048 candidate blocks of 8 keys",
            spec=dict(ratio=1, nreq=1, rows=6, block=8, blocks=2048, hist=20000),
        ),
        Cell(
            why="ratio 2, 24 requests of two rows, 40 blocks of 8 keys",
            spec=dict(ratio=2, nreq=24, rows=2, block=8, blocks=40),
        ),
    )
    note = "An L2 prefetch: the reference is a no-op and the test checks that every operand is left unchanged."

    def __call__(
        self,
        index_pool: torch.Tensor,
        blocks: torch.Tensor,
        cand_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
            prefetch_candidate_blocks_,
        )

        prefetch_candidate_blocks_(
            index_pool.view(-1, _NATIVE, 1, 68),
            blocks,
            cand_in[1],
            cand_in[2].reshape(-1),
            SPARSE_BLOCK,
        )

    def reference(
        self,
        index_pool: torch.Tensor,
        blocks: torch.Tensor,
        cand_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """No result: the launch only warms L2."""
        return None

    def is_valid(
        self,
        index_pool: torch.Tensor,
        blocks: torch.Tensor,
        cand_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        assert (
            index_pool.dtype == torch.uint8
            and index_pool.shape[-1] == 68
            and index_pool.is_contiguous()
        )
        assert blocks.dtype == torch.int32 and blocks.dim() == 2 and blocks.stride(1) == 1, (
            "blocks is the int32 native block table of `csa2_indexer_descriptors`"
        )
        assert len(cand_in) == 3 and cand_in[1].dtype == torch.int32 and cand_in[1].stride(1) == 1
        assert (
            cand_in[2].dtype == torch.int32
            and cand_in[2].numel() == blocks.shape[0] == cand_in[1].shape[0]
        )


publish_candidates = Csa2PublishCandidates()
prefetch_candidates = Csa2PrefetchCandidates()
consume_candidates = Csa2ConsumeCandidates()
