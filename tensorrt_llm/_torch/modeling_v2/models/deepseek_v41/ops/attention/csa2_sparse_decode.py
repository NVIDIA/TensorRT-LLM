# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused latent sparse attention that reads the packed SWA / compressed-KV pools in place.

Two entries, both the math of `csa2_sparse_attn` (window rows + selected compressed rows + attention
sink, fp32 softmax, keys dequantized exactly from the packed rows):

- `csa2_sparse_decode` wraps `torch.ops.trtllm.csa2_sparse_decode` (64 heads, any row count, also
  context chunks); the index arguments the kernel takes are derived here from the step's own tensors
  (`pos`, `rid`, page tables). Static shapes and no host sync: safe under CUDA-graph capture once the
  shared workspace exists (the target allocates it with `csa2_sparse_decode_workspace` on an eager step).
- `csa2_direct_decode` wraps the engine's CuTe-DSL `direct_sparse_decode` (8 / 16 / 32 / 64 heads, at most
  16 query rows, generation only), the path of the 8 heads per rank of TP8.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_chunk_window import csa2_chunk_window
from .csa2_pack_rows import csa2_unpack_rows
from .csa2_rows_io import csa2_gather_rows, csa2_row_addr
from .csa2_sparse_attn import csa2_sparse_attn
from .csa2_store_rows import csa2_rope_inplace

HEADS, HEAD_DIM, SWA_ROW, MAIN_ROW = 64, 512, 528, 288


def csa2_sparse_decode_supported(heads: int, device) -> bool:
    """True when the fused kernel serves this layer: 64 query heads on an SM100-family GPU with the op built."""
    if heads != HEADS or not hasattr(
        torch.ops.trtllm, "csa2_sparse_decode"
    ):  # as the engine's probe
        return False
    with torch.cuda.device(device):
        return bool(torch.ops.trtllm.csa2_sparse_decode_supported())


def csa2_sparse_decode_workspace(device) -> torch.Tensor:
    """Scratch the kernel needs (any contents); the target allocates it once per device, eagerly, and passes it in."""
    return torch.empty(
        int(torch.ops.trtllm.csa2_sparse_decode_workspace_size()), dtype=torch.uint8, device=device
    )


def csa2_direct_decode_supported(heads: int, rows: int, device, main_width: int = 512) -> bool:
    """True when the engine's CuTe-DSL direct decode serves `heads` heads
    (8/16/32/64); `rows` beyond its 16-row limit run in chunks."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.direct_decode import (
        direct_decode_device_supported,
        direct_decode_fits,
    )

    return (
        rows > 0  # more than MAX_ROWS rows run as several launches
        and direct_decode_device_supported(torch.device(device))
        and direct_decode_fits(heads, 128, main_width)
    )


def _pad_topk(topk_idx):
    """The selected rows as int32 `[n, k]` with `k` a multiple of 64 (-1 padded); no copy when they already are."""
    k = topk_idx.shape[1]
    pad = -k % 64 + (64 if k == 0 else 0)
    idx = topk_idx if topk_idx.dtype == torch.int32 else topk_idx.int()
    return idx if pad == 0 and idx.is_contiguous() else F.pad(idx, (0, pad), value=-1).contiguous()


def _window_rows(pos, rid, swa_table, window):
    """Flat SWA pool row of each of the `window` positions ending at `pos` (`[n, window]` int64, -1 = none)."""
    qpos = pos.long().unsqueeze(-1) - (window - 1) + torch.arange(window, device=pos.device)
    return csa2_row_addr(
        swa_table, rid.long().unsqueeze(-1).expand_as(qpos), qpos, 128
    ).contiguous()


def _visible_rows(pos, ratio):
    """Compressed rows visible to each query row: `(pos + 1) // ratio`."""
    return ((pos.long() + 1) // ratio).clamp(min=0).contiguous()


def _check_window_inputs(q, window, swa_rows, visible):
    if swa_rows is not None:
        assert (
            swa_rows.dtype == torch.int64
            and swa_rows.shape == (q.shape[0], window)
            and swa_rows.is_contiguous()
        ), "swa_rows are the int64 [n, window] pool rows of `csa2_swa_slots` (reads[layer])"
    if visible is not None:
        assert (
            visible.dtype == torch.int64
            and visible.shape == (q.shape[0],)
            and visible.is_contiguous()
        ), "visible is the int64 [n] row of `csa2_swa_slots` (visible[layer])"


def _check_pools(q, swa_pool, window, main_pool, main_table, topk_idx, ratio):
    assert swa_pool.dim() == 3 and swa_pool.shape[1:] == (128, SWA_ROW), (
        "the SWA pool is [pages, 128, 528]; the row addressing is hard-wired to 128-row pages"
    )
    assert 0 < window <= 128, "the SWA pool holds the last 128 positions only"
    if topk_idx is None:
        assert main_pool is None, "main_pool without topk_idx is silently ignored"
    else:
        assert ratio in (1, 2) and main_pool is not None and main_table is not None
        assert main_pool.shape[1:] == (128 // ratio, MAIN_ROW), (
            "the main pool is [pages, 128 // ratio, 288]"
        )
        assert topk_idx.shape[0] == q.shape[0]


_O_GROUPS = 8  # output groups of a 64-head layer: the MXFP8 output's layout


class Csa2SparseDecode(OpWrapper):
    """Latent sparse attention of 64-head layers on the packed pools, one fused launch.

    `q` `[n, 64, 512]` bf16, `sink` `[64]` fp32, `pos` `[n]` int32 position of each query row, `rid` `[n]`
    its page-table row, `swa_pool` `[pages, 128, 528]` / `main_pool` `[pages, 128 // ratio, 288]` the
    packed pools with their page tables, `topk_idx` `[n, k]` the selected compressed rows (-1 = none).

    `step_rows is None` (generation): every window row, this step's included, is already in `swa_pool`.
    Otherwise (context chunks, which can be longer than the pool's window) `step_rows` `[n, 528]` uint8 are
    this step's packed window rows in token order (`csa2_pack_rows(kv, "swa")`) and `cs` `[n]` the first
    position of each token's request in the step; rows before `cs` come from `swa_pool`.
    `floor` `[n]` int (optional) is the first window position each row may read: earlier window rows are
    masked, as in the truncated window of a bounded decoder replay (the step's own rows are never below it).
    `req_cs` `[requests]` int64 (optional, context chunks) is `cs` per request, built once per step by the caller.
    `workspace` is the scratch of `csa2_sparse_decode_workspace`. Returns bf16 `[n, 64, 512]`.
    Generation steps may pass `q` as unit-scale E4M3 (stock's FP8 decode query, `fp8_query_gemm`), which the kernel
    reads as is (stock's fused decode with `projected_quant_q`).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="generation, no compression: two requests with history 300 and 5, 1 and 4 rows",
            spec=dict(reqs=[(300, 1), (5, 4)], ratio=0, topk=0, ctx=False),
        ),
        Cell(
            why="generation, ratio 2: three requests (one without history), 64 selected rows",
            spec=dict(reqs=[(300, 2), (1000, 1), (0, 1)], ratio=2, topk=64, ctx=False),
        ),
        Cell(
            why="generation, ratio 1: 2000 tokens of history, four draft rows, 96 selected rows",
            spec=dict(reqs=[(2000, 4)], ratio=1, topk=96, ctx=False),
        ),
        Cell(
            why="context chunk from scratch, no compression: 700 rows, longer than the window",
            spec=dict(reqs=[(0, 700)], ratio=0, topk=0, ctx=True),
        ),
        Cell(
            why="context chunks of two requests, ratio 2: history 300 + 520 rows and a fresh 200",
            spec=dict(reqs=[(300, 520), (0, 200)], ratio=2, topk=128, ctx=True),
        ),
        Cell(
            why="context, ratio 1: a long history (5000), a short one (130) and a one-row request",
            spec=dict(reqs=[(5000, 300), (130, 40), (0, 1)], ratio=1, topk=256, ctx=True),
        ),
        Cell(
            why="bounded decoder replay: history rows below the floor (history 800, floor 740) are masked",
            spec=dict(reqs=[(800, 100), (300, 60)], ratio=1, topk=128, ctx=True, floor=[740, 290]),
        ),
        Cell(
            why="bounded decoder replay, generation form: history rows below the floor are masked",
            spec=dict(reqs=[(800, 4), (300, 4)], ratio=1, topk=128, ctx=False, floor=[790, 290]),
        ),
    )
    note = """
    Probabilities are rounded to bf16 in the kernel, so the agreement with the plain-torch reference is
    bf16-level (the test states the band). The workspace is the caller's: the op keeps nothing.
    """

    def __call__(
        self,
        workspace: torch.Tensor,
        q: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        step_rows: Optional[torch.Tensor] = None,
        cs: Optional[torch.Tensor] = None,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        floor: Optional[torch.Tensor] = None,
        req_cs: Optional[torch.Tensor] = None,
        mxfp8_out: Optional[tuple] = None,
    ) -> "torch.Tensor | tuple":
        """With `mxfp8_out = (pos int32 [n], cos_sin fp32 [P, 64])` the launch writes the grouped O-A projection's MXFP8
        input instead of the bf16 output: E4M3 `[8, n, 4096]` and per-group swizzled UE8M0 scales, rotated back at `pos`
        (what the engine's `inverse_rope_mxfp8` makes of the bf16 output; the native op's own option)."""
        n, h, d = q.shape
        assert h == HEADS and d == HEAD_DIM and window % 2 == 0, (h, d, window)
        dev = q.device
        flat_swa = swa_pool.view(-1, SWA_ROW)
        if step_rows is None:
            local = flat_swa
            swa_idx = (
                swa_rows if swa_rows is not None else _window_rows(pos, rid, swa_table, window)
            )
        elif (
            window == 128
            and pos.dtype == torch.int32
            and pos.is_contiguous()
            and rid.dtype == torch.int64
        ):
            # context chunk: the history rows of every request and each slot's
            # row in `cat([step_rows, old_rows])`, two launches
            req_cs = (
                req_cs
                if req_cs is not None
                else torch.zeros(swa_table.shape[0], dtype=torch.int64, device=dev).index_put_(
                    (rid,), cs.long()
                )
            )
            old_rows, swa_idx = csa2_chunk_window(
                pos, rid, cs, req_cs, swa_pool, swa_table, window, floor
            )
            local = torch.cat([step_rows, old_rows], dim=0)
            floor = None  # applied by the op
        else:
            p = pos.long().unsqueeze(-1)  # [n, 1]
            qpos = p - (window - 1) + torch.arange(window, device=dev)  # [n, window]
            rid2 = rid.long().unsqueeze(-1).expand_as(qpos)
            nreq = swa_table.shape[0]
            cs_t = cs.long()
            cs_req = torch.zeros(nreq, dtype=torch.long, device=dev).index_put_((rid.long(),), cs_t)
            oldpos = cs_req.unsqueeze(-1) - (window - 1) + torch.arange(window - 1, device=dev)
            oaddr = csa2_row_addr(
                swa_table,
                torch.arange(nreq, device=dev).unsqueeze(-1).expand_as(oldpos),
                oldpos,
                128,
            )
            local = torch.cat(
                [step_rows, csa2_gather_rows(swa_pool, oaddr).view(-1, SWA_ROW)], dim=0
            )
            from_step = qpos >= cs_t.unsqueeze(-1)
            t = torch.arange(n, device=dev).unsqueeze(-1)
            old_slot = (qpos - (cs_t.unsqueeze(-1) - (window - 1))).clamp(0, window - 2)
            old_off = rid2 * (window - 1) + old_slot
            old_ok = (oaddr >= 0).view(-1)[old_off.reshape(-1)].view_as(old_off)
            swa_idx = torch.where(from_step, t - (p - qpos), torch.where(old_ok, n + old_off, -1))
            swa_idx = torch.where(qpos >= 0, swa_idx, -1)
        if floor is not None:
            if step_rows is None:  # the generation form has not derived the window positions yet
                qpos = pos.long().unsqueeze(-1) - (window - 1) + torch.arange(window, device=dev)
            swa_idx = torch.where(qpos >= floor.long().unsqueeze(-1), swa_idx, -1)
        swa_idx = swa_idx.contiguous()
        main_idx = table = requests = None
        mpool = None
        rpp = max_positions = 0
        if topk_idx is not None:
            main_idx = _pad_topk(topk_idx)
            mpool, table = main_pool.view(-1, MAIN_ROW), main_table
            rpp = 128 // ratio
            max_positions = main_table.shape[1] * rpp
            visible = (
                visible if visible is not None else _visible_rows(pos, ratio)
            )  # the step plan's, else derived here
            requests = (
                rid if rid.dtype == torch.int64 and rid.is_contiguous() else rid.long().contiguous()
            )
        else:
            visible = None
        extra = ()
        if mxfp8_out is not None:
            out = torch.empty(
                (_O_GROUPS, n, HEADS // _O_GROUPS * HEAD_DIM), dtype=torch.float8_e4m3fn, device=dev
            )
            scales = torch.empty(
                _O_GROUPS * (-(-n // 128) * 128) * (HEADS // _O_GROUPS * HEAD_DIM // 32),
                dtype=torch.uint8,
                device=dev,
            )
            extra = (scales, mxfp8_out[0], mxfp8_out[1])
        else:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=dev)
        torch.ops.trtllm.csa2_sparse_decode(
            q,
            local,
            swa_idx,
            mpool,
            main_idx,
            table,
            requests,
            visible,
            rpp,
            max_positions,
            sink.float().contiguous(),
            float(scale),
            out,
            workspace,
            *extra,
        )
        return out if mxfp8_out is None else (out, extra[0])

    def reference(
        self,
        workspace: torch.Tensor,
        q: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        step_rows: Optional[torch.Tensor] = None,
        cs: Optional[torch.Tensor] = None,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        floor: Optional[torch.Tensor] = None,
        req_cs: Optional[torch.Tensor] = None,
        mxfp8_out: Optional[tuple] = None,
    ) -> torch.Tensor:
        """`csa2_sparse_attn` (the plain-torch op) on the same packed pools; `workspace` is ignored.

        Generation: every window row is in the pool, so no row is taken from the step (`cs = pos + 1`). The window rows
        and visible counts are derived from `pos` and the tables: supplied `swa_rows` / `visible` must be the derived
        ones (`test_precomputed_window_slots_match_the_derived_ones` checks the op gives the same result with both).
        Context: the step's rows are `step_rows` decoded with `csa2_unpack_rows`. With `mxfp8_out` the result is that
        output after the engine's `inverse_rope_mxfp8` (the O-A input format).
        """
        q = q.bfloat16()  # an E4M3 query is exact in bf16
        if step_rows is None:
            kv = torch.zeros(q.shape[0], HEAD_DIM, dtype=torch.bfloat16, device=q.device)
            cs = pos.long() + 1
        else:
            kv = csa2_unpack_rows(step_rows, "swa")
        out = csa2_sparse_attn(
            q, kv, sink, pos, cs, rid, swa_pool, swa_table, window, scale,
            main_pool, main_table, topk_idx, ratio, True, floor,
        )  # fmt: skip
        if mxfp8_out is None:
            return out
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import inverse_rope_mxfp8

        return inverse_rope_mxfp8(
            out.reshape(out.shape[0], -1), mxfp8_out[0], mxfp8_out[1], _O_GROUPS
        )

    def is_valid(
        self,
        workspace: torch.Tensor,
        q: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        step_rows: Optional[torch.Tensor] = None,
        cs: Optional[torch.Tensor] = None,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        floor: Optional[torch.Tensor] = None,
        req_cs: Optional[torch.Tensor] = None,
        mxfp8_out: Optional[tuple] = None,
    ) -> None:
        assert (
            q.dim() == 3
            and q.shape[1:] == (HEADS, HEAD_DIM)
            and q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        )
        assert q.dtype == torch.bfloat16 or step_rows is None, (
            "an E4M3 query is the generation form's only"
        )
        assert q.is_contiguous()
        assert workspace.dtype == torch.uint8, (
            "workspace is the uint8 scratch of csa2_sparse_decode_workspace"
        )
        assert sink.shape == (HEADS,) and window % 2 == 0
        assert pos.shape == rid.shape == (q.shape[0],)
        assert floor is None or floor.shape == pos.shape, (
            "floor is the per-token first readable window position [n]"
        )
        _check_pools(q, swa_pool, window, main_pool, main_table, topk_idx, ratio)
        assert (step_rows is None) == (cs is None), (
            "step_rows and cs go together: without cs the kernel reads the step's own rows from the pool, "
            "with step_rows it needs each request's first position to tell them from history"
        )
        if step_rows is not None:
            assert step_rows.dtype == torch.uint8 and step_rows.shape == (q.shape[0], SWA_ROW)
            assert swa_rows is None, "swa_rows describe the generation form (step_rows is None)"
        _check_window_inputs(q, window, swa_rows, visible)


class Csa2DirectDecode(OpWrapper):
    """`csa2_sparse_decode` (generation form) for small batches of any supported head count.

    Wraps the engine's `direct_sparse_decode` (CuTe DSL, 16 query rows per launch, chunked above that), e.g. the 8 heads
    per rank of TP8. The step's own window rows must already be in `swa_pool`. With `q_rope = (pos, table)` `q` is the
    projected query before its RoPE and the launch rotates it (the table is the engine's float32 `[P, 64]` layout). The
    first call of a geometry compiles: run one eager step before graph capture.

    As the engine's index-source / Reuse layers (`_exported_slots`): `slots_out` (int64 `[n, k64]`, `k64` = `topk_idx`'s
    width padded to 64) also stores the pool row every selection resolves to (-1 when invalid); a later layer over the
    same selections, owner and compression ratio passes them as `main_slots` and the launch reads those rows unmapped
    (`topk_idx` then only feeds the reference).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tp8: 8 heads, six rows, no compression",
            spec=dict(heads=8, reqs=[(300, 6)], ratio=0, topk=0),
        ),
        Cell(
            why="tp8: 8 heads, ratio 2, 2000 tokens of history, 64 selected rows",
            spec=dict(heads=8, reqs=[(2000, 6)], ratio=2, topk=64),
        ),
        Cell(
            why="tp8: 8 heads, ratio 1, two requests of four rows with different history",
            spec=dict(heads=8, reqs=[(1500, 4), (40, 4)], ratio=1, topk=96),
        ),
        Cell(
            why="64 heads at four rows, ratio 2",
            spec=dict(heads=64, reqs=[(900, 4)], ratio=2, topk=128),
        ),
        Cell(
            why="tp8: 8 heads, 16 requests of six rows (96 rows, six launches), ratio 2",
            spec=dict(heads=8, reqs=[(400 + 90 * i, 6) for i in range(16)], ratio=2, topk=64),
        ),
        Cell(
            why="tp8: 8 heads, three requests of six rows (18 rows, a two-row tail launch), ratio 1",
            spec=dict(heads=8, reqs=[(700, 6), (1500, 6), (90, 6)], ratio=1, topk=96),
        ),
    )
    note = "The kernel serves 16 query rows per launch; more rows run as several launches (static chunks)."

    def __call__(
        self,
        q: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        q_rope: Optional[tuple] = None,
        slots_out: Optional[torch.Tensor] = None,
        main_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.direct_decode import (
            MAX_ROWS,
            direct_sparse_decode,
        )

        swa_idx = swa_rows if swa_rows is not None else _window_rows(pos, rid, swa_table, window)
        main_idx = mapping = None
        mpool = None
        if (
            main_slots is not None
        ):  # rows resolved by an earlier launch's `slots_out` (same selections, owner, visible lengths)
            main_idx, mpool = main_slots, main_pool.view(-1, MAIN_ROW)
        elif topk_idx is not None:
            main_idx = _pad_topk(topk_idx)
            mpool = main_pool.view(-1, MAIN_ROW)
            rpp = 128 // ratio
            mapping = (
                main_table,
                rpp,
                main_table.shape[1] * rpp,
                rid
                if rid.dtype == torch.int64 and rid.is_contiguous()
                else rid.long().contiguous(),
                visible if visible is not None else _visible_rows(pos, ratio),
            )
        q = q.contiguous()
        rope = None if q_rope is None else q_rope[1]
        swa_flat, sink32 = swa_pool.view(-1, SWA_ROW), sink.float().contiguous()
        n = q.shape[0]
        if n <= MAX_ROWS:
            return direct_sparse_decode(
                q, swa_flat, swa_idx, mpool, main_idx, sink32, float(scale), main_mapping=mapping,
                query_rope=None if rope is None else (q_rope[0], rope), slots_out=slots_out,
            )  # fmt: skip
        # More rows than one launch serves: static chunks of MAX_ROWS (the row count is a graph constant).
        out = torch.empty_like(q)
        for lo in range(0, n, MAX_ROWS):
            hi = min(lo + MAX_ROWS, n)
            chunk_mapping = (
                None if mapping is None else mapping[:3] + (mapping[3][lo:hi], mapping[4][lo:hi])
            )
            direct_sparse_decode(
                q[lo:hi],
                swa_flat,
                swa_idx[lo:hi],
                mpool,
                None if main_idx is None else main_idx[lo:hi],
                sink32,
                float(scale),
                main_mapping=chunk_mapping,
                output=out[lo:hi],
                query_rope=None if rope is None else (q_rope[0][lo:hi], rope),
                slots_out=None if slots_out is None else slots_out[lo:hi],
            )
        return out

    def reference(
        self,
        q: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        q_rope: Optional[tuple] = None,
        slots_out: Optional[torch.Tensor] = None,
        main_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`csa2_sparse_attn` (the plain-torch op); every window row is read from the pool. With `q_rope` the query is
        rotated first by `csa2_rope_inplace.reference`. The window rows and visible counts are derived from `pos`
        and the tables: supplied `swa_rows` / `visible` must be the derived ones."""
        if q_rope is not None:
            q = csa2_rope_inplace.reference(q, q_rope[0], q_rope[1])
        kv = torch.zeros(q.shape[0], HEAD_DIM, dtype=torch.bfloat16, device=q.device)
        return csa2_sparse_attn(
            q, kv, sink, pos, pos.long() + 1, rid, swa_pool, swa_table, window, scale,
            main_pool, main_table, topk_idx, ratio, True,
        )  # fmt: skip

    def is_valid(
        self,
        q: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        q_rope: Optional[tuple] = None,
        slots_out: Optional[torch.Tensor] = None,
        main_slots: Optional[torch.Tensor] = None,
    ) -> None:
        assert q.dim() == 3 and q.shape[2] == HEAD_DIM and q.dtype == torch.bfloat16
        assert window == 128, "the direct kernel reads a 128-row window"
        _check_window_inputs(q, window, swa_rows, visible)
        assert q_rope is None or (
            q_rope[0].dtype == torch.int32
            and q_rope[1].dtype == torch.float32
            and q_rope[1].shape[-1] == 64
        ), (
            "q_rope is (positions int32 [n], the engine's float32 [P, 64] table of `csa2_stock_rope_table`)"
        )
        assert sink.shape == (q.shape[1],) and pos.shape == rid.shape == (q.shape[0],)
        assert csa2_direct_decode_supported(q.shape[1], q.shape[0], q.device), (
            "the direct decode serves 8/16/32/64 heads on a supported GPU"
        )
        _check_pools(q, swa_pool, window, main_pool, main_table, topk_idx, ratio)
        k64 = None if topk_idx is None else -(-max(topk_idx.shape[1], 1) // 64) * 64
        for t, name in ((slots_out, "slots_out"), (main_slots, "main_slots")):
            assert t is None or (
                topk_idx is not None
                and t.dtype == torch.int64
                and t.shape == (q.shape[0], k64)
                and t.is_contiguous()
            ), f"{name} is int64 [n, k64] over the step's selections (topk_idx, padded to 64)"
        assert slots_out is None or main_slots is None, (
            "a launch exports the rows it maps; resolved rows are not mapped"
        )


csa2_sparse_decode = Csa2SparseDecode()
csa2_direct_decode = Csa2DirectDecode()
