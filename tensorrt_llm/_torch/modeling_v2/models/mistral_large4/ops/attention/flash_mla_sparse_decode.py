# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MQA of one query row per sequence over an fp8_ds_mla pool: FlashMLA's sm100f
sparse fp8 decode kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm.flash_mla import flash_mla_with_kvcache as _flash_mla_with_kvcache

ROW_BYTES = 656
D_NOPE = 512
D_ROPE = 64
TILE = 128
#: The query head counts FlashMLA's sm100 sparse decode runs (64, or 64 twice).
KERNEL_HEADS = (64, 128)
#: The most rows (one-row sequences) one call takes: the largest the cells drive. FlashMLA's
#: scheduler exits the process past 11622 on sm_103 (the note's Hard limits).
MAX_ROWS = 8192

#: The tp4 deployment at the 1M envelope: 16 live heads of the 64-head scratch, sequences to
#: 1048576 keys, the fp8_ds_mla pool's 32-token pages of 63 KV-owner rows per block.
_ML4_1M = dict(h_q=64, h_out=16, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: The tep4 deployment at the 1M envelope: 16 live heads of its 64-head scratch, top-2048
#: selections, sequences to 1048576 keys, 32-token pages of 63 KV-owner rows per block.
_TEP4_1M = dict(h_q=64, h_out=16, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: The dep4 deployment at the 1M envelope: all 64 heads of a rank (its scratch is the kernel's
#: 64 heads, no padding), top-2048 selections, sequences to 1048576 keys, 32-token pages of
#: 63 KV-owner rows per block.
_DEP4_1M = dict(h_q=64, h_out=64, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: `pages` (blocks x 63) of a 74183-block pool (the delivered 1M tp4 boot's, above the
#: 72757 blocks a 1M tep4 rank's boot measured and the 64292 the 1M dep4 boot measured per
#: rank at fraction 0.9, on fp8_ds_mla), of the 107917-block far end (the far cells of the
#: 1M attention entries), of the 1100-block pool the dep4 pre-merge eager and context cells
#: spread 1M sequences over, and of the 2048-block pool the tep4, dep4 and tep8 pre-merge
#: cells read past 2**31 bytes.
_BOOT_PAGES = 74183 * 63
_FAR_PAGES = 107917 * 63
_SMALL_PAGES = 1100 * 63
_PRE_PAGES = 2048 * 63
#: The batches the engine captures (padded): an attention-DP rank replays the largest rank's.
_CAPTURED = (1, 2, 4, 8, 16)
#: The pre-merge share of the ragged-tail sweep's row counts: 1..65, either side of every
#: power of two up to 4096, and the top 33.
_TAIL_SHARE = (
    tuple(range(1, 66))
    + tuple(r for k in range(7, 13) for r in (2**k - 1, 2**k, 2**k + 1))
    + tuple(range(8160, 8193))
)
#: The ragged-tail sweep's pre-merge share of the last sub-batch's row counts r: 1..65, 96,
#: 2**k - 1 / 2**k / 2**k + 1 up to 4097, 8160 and 8175..8192.
_RAGGED_SHARE = (
    tuple(range(1, 66))
    + (96,)
    + tuple(r for k in range(7, 14) for r in (2**k - 1, 2**k, 2**k + 1) if r < 8175)
    + (8160,)
    + tuple(range(8175, 8193))
)
#: The dep4 ragged-tail sweep's pre-merge share of the last sub-batch's row counts r: 1..65,
#: 2**k - 1 / 2**k / 2**k + 1 to 4097, r017-land2's 6445, the witness's 8096 and 8175..8192.
_DEP4_TAIL_SHARE = (
    tuple(range(1, 66))
    + tuple(r for k in range(7, 13) for r in (2**k - 1, 2**k, 2**k + 1))
    + (6445, 8096)
    + tuple(range(8175, 8193))
)
#: The tep8 deployment at the 1M envelope: 8 live heads of its 64-head scratch, top-2048
#: selections, sequences to 1048576 keys, 32-token pages of 63 KV-owner rows per block.
_TEP8_1M = dict(h_q=64, h_out=8, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: `pages` (blocks x 63) of the 147456-block pool bound, the "1M tep8" cells' far pool (a 1M
#: tep8 rank holds 114326 blocks on fp8_ds_mla at fraction 0.9); their pre-merge cells read
#: the 2048-block `_PRE_PAGES` past 2**31 bytes and run the `_RAGGED_SHARE` row counts.
_BOUND_PAGES = 147456 * 63


class FlashMlaSparseDecode(OpWrapper):
    """For every query row r and head h < `out.shape[1]`::

        K[j]   = [e4m3(row_j[:512]) * tile_scale_j (per 128) | bf16(row_j[528:656])]
        p      = softmax_j(softmax_scale * q[r, h] . K[j]),  j in indices[r] (-1 skipped)
        out[r, h] = sum_j p[j] * K[j][:512]

    `kv_cache`: uint8 `[pages, page_size, 1, 656]`, a page's rows contiguous; a slot is
    `page * page_size + offset`. `indices`: int32 `[rows, topk]` slots, -1 padded.
    `q`: bf16 `[rows, h_q, 576]` with h_q 64 or 128 -- the kernel's head counts; a rank
    with fewer heads passes them first and zeros after (the caller's scratch), and
    names how many it keeps through `out`. `sched_meta`: a FlashMLA scheduler object
    (`tensorrt_llm.flash_mla.get_mla_metadata()[0]`), filled on its first call and
    shared by every call of one step with the same rows, h_q and topk. Writes `out`
    (bf16 `[rows, h_out, 512]`, any strides) and returns it; with `out` None, returns the
    kernel's own `[rows, h_q, 512]` output (a view, no copy), whose first heads a caller
    reads in place. At most `MAX_ROWS` rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `rows` sequences of one query row, `h_q` kernel heads of which `h_out` are
    # real, `topk` slots over `kv_len`-token sequences in a pool of `pages` pages of
    # `page_size` x `owners` rows; `layers` calls share one sched meta; `captured`
    # replays from a CUDA graph; `holes` scatters -1 through short rows; `view` passes
    # `out` None and gates the returned kernel output.
    # The "1M tp4" cells: `kv_len` is the longest sequence, the others spread below it;
    # `graph_batches` the captured batches (each replay on new slots bitwise equal to an
    # eager call); `far` builds the pool at `pages` / `owners` blocks exactly with the
    # requests' pages its top ones and the first request's last page its top block;
    # `q_offset` makes q and out tail views from that row of a 32768-row step scratch;
    # `context` makes the rows a step's context rows (`chunk_rows` behind `cached`, or the
    # packed `contexts` of (cached, new)), each a one-row sequence, driven in sub-batches of
    # `sub_batch` rows with one sched meta per sub-batch size (`rows`: the largest).
    # `ragged_tail` drives a chunk's last sub-batch at every row count r in 1..`sub_batch`
    # (`tail_rows`: those r only): rows `lead_rows` .. `lead_rows` + r - 1 of the chunk behind
    # `cached`, q and out views from that row of the step's scratch, a sched meta per r.
    # The "1M tep4", "1M dep4" and "1M tep8" cells build the pool on the device, every byte
    # 0xFF (e4m3, fp32 and bf16 NaN) but one owner row of the pages the requests hold.
    # `kv_len` is the longest decode sequence, the others spread below it; `graph_batches`
    # the batches captured (each replay on new queries and new slots in the same pages,
    # bitwise an eager call); `live: 1` with `boot_capture`: each batch captured on the boot
    # warm-up's rows (its longest dummy at kv_len - 1 keys beside one-key fillers), then
    # replayed as attention DP replays it, one live row at kv_len keys behind one-key padding
    # dummies that all read one slot, and an idle rank's one-key dummy behind the same
    # padding;
    # `row_counts`: eager decode steps at each count, a fresh sched meta per step; `far`: the
    # pool is `pages` / `owners` blocks, the requests' pages distinct and its top ones, the
    # first request's last page on the top block; `top_blocks`: the requests' pages drawn
    # from the pool's top that many blocks; `q_offset`: q and out tail views from that row
    # of a 32768-row step scratch; `context`: the rows are a step's context rows
    # (`chunk_rows` behind `cached`, or the packed `contexts` of (cached, new)), each a
    # one-row sequence, in sub-batches of `sub_batch` rows with one sched meta per sub-batch
    # size shared by every layer (`rows`: the largest sub-batch); `ragged_tail`: only a
    # chunk's last sub-batch, each r one call on a sched meta of its own -- a "1M dep4"
    # cell's rows `lead_rows` + [0, r) of the chunk behind `cached`, at every r in
    # 1..`sub_batch` (`tail_rows`: at those r only), a "1M tep4" or "1M tep8" cell's the
    # last sub-batch of a 32768-row chunk behind `cached`, its first r rows at each r in
    # `tails` (every r in 1..`sub_batch` without it; a "1M tep8" cell's the largest first).
    # The test drives the "1M tp4", the "1M tep4", the "1M dep4" and the "1M tep8" cells each
    # with the harness that certified them.
    CELLS: tuple[Cell, ...] = (
        *tuple(
            Cell(
                why=f"tep4/tp4 decode batch {b}: 16 real heads padded to 64, graph replay, "
                "3 layers share a sched meta",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * b // 16 + 64,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16)
        ),
        *tuple(
            Cell(
                why=f"dep4 decode batch {b}: 64 heads, graph replay",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=64,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * max(1, b // 16) + 64,
                    layers=2,
                    captured=True,
                ),
            )
            for b in (1, 16, 64)
        ),
        Cell(
            why="short sequences: -1 tails and scattered holes",
            spec=dict(
                rows=16,
                h_q=64,
                h_out=16,
                topk=2048,
                kv_len=700,
                page_size=32,
                owners=63,
                pages=512,
                layers=1,
                holes=True,
            ),
        ),
        Cell(
            why="NoPE layer: zero query and key rope halves",
            spec=dict(
                rows=16,
                h_q=64,
                h_out=16,
                topk=2048,
                kv_len=9000,
                page_size=32,
                owners=63,
                pages=512,
                layers=1,
                nope=True,
            ),
        ),
        Cell(
            why="tp8-style 8 real heads padded to 64",
            spec=dict(
                rows=8,
                h_q=64,
                h_out=8,
                topk=2048,
                kv_len=4096,
                page_size=32,
                owners=63,
                pages=512,
                layers=1,
            ),
        ),
        *tuple(
            Cell(
                why=f"64-token pages: tep4/tp4 decode batch {b}: 16 real heads padded to 64, "
                "graph replay, 3 layers share a sched meta",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=64,
                    owners=63,
                    pages=550 * b // 16 + 32,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16)
        ),
        Cell(
            why="1M tep8 pre-merge context sub-batches past 2**31 pool bytes: a chunk's last "
            "12000 rows behind 1036576 cached keys, 8192 + 3808 on two sched metas, every "
            "slot above 3273604, 2 layers",
            spec=dict(
                _TEP8_1M,
                rows=8192,
                pages=_PRE_PAGES,
                top_blocks=400,
                layers=2,
                context=True,
                chunk_rows=12000,
                cached=1036576,
                sub_batch=8192,
            ),
        ),
        Cell(
            why="1M tep8 pre-merge generation tail past 2**31 pool bytes: 13 rows of keys to "
            "1048576, q and out tail views from row 32755 of a 32768-row scratch, 3 layers",
            spec=dict(
                _TEP8_1M,
                rows=13,
                pages=_PRE_PAGES,
                top_blocks=400,
                layers=3,
                q_offset=32755,
            ),
        ),
    )
    # `out` None at the 1M config, the decode call the targets read in place (dep4's always,
    # tp4 / tep4's under `_GEN_ATTN_FLASH_MLA` with the fused glue): the "1M tep4" / "1M dep4"
    # harness's pools and rows, every layer's call returning the kernel's 64-head output, its
    # live heads bitwise the `out` call's copy on the same inputs and gated to the reference.
    CELLS += (
        Cell(
            why="1M view tp4/tep4 decode graphs at every batch 1..16, boot pool (large): out "
            "None, the 16 live heads of the 64-head output read in place, 3 layers on one "
            "sched meta, rows of 1048576 keys down to 1, the pool's last slot read",
            spec=dict(
                _TEP4_1M,
                rows=16,
                pages=_BOOT_PAGES,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
                far=True,
                view=True,
                large=True,
            ),
        ),
        Cell(
            why="1M view tp4/tep4 decode graph 16, far pool (large): 107917 blocks (142.7 GB), "
            "out None, the top slot read at byte 1.427e11",
            spec=dict(
                _TEP4_1M,
                rows=16,
                pages=_FAR_PAGES,
                layers=3,
                captured=True,
                far=True,
                view=True,
                large=True,
            ),
        ),
        Cell(
            why="1M view dep4 decode graphs at every batch 1..16, boot pool (large): out None, "
            "all 64 heads as returned, 3 layers on one sched meta",
            spec=dict(
                _DEP4_1M,
                rows=16,
                pages=_BOOT_PAGES,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
                far=True,
                view=True,
                large=True,
            ),
        ),
        Cell(
            why="1M view dep4 NoPE decode graph 16 (large): owner row 35, zero rope halves, out "
            "None, boot pool",
            spec=dict(
                _DEP4_1M,
                rows=16,
                pages=_BOOT_PAGES,
                layers=3,
                captured=True,
                nope=True,
                far=True,
                view=True,
                large=True,
            ),
        ),
    )

    note = """
    FlashMLA sparse FP8 decode over paged fp8_ds_mla pools (64 kernel heads; real heads in
    [0, h_out)). Covers 8/16/64 live heads, graph replay, -1 holes, NoPE owner row, 64-token
    pages, and 1M context/generation tail sub-batches.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        sched_meta,
        softmax_scale: float,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        rows, h_q = q.shape[0], q.shape[1]
        o, _ = _flash_mla_with_kvcache(
            q.view(rows, 1, h_q, q.shape[-1]),
            kv_cache,
            None,
            None,
            D_NOPE,
            sched_meta,
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=indices.view(rows, 1, indices.shape[-1]),
        )
        if out is None:
            return o[:, 0]
        out.copy_(o[:, 0, : out.shape[1]])
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        sched_meta,
        softmax_scale: float,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """fp32, exact dequantization, softmax over the valid slots; a new tensor shaped
        like `out` (`[rows, h_q, 512]` when `out` is None), rounded to bf16 once. Ignores
        `sched_meta`."""
        shape = (q.shape[0], q.shape[1], D_NOPE) if out is None else tuple(out.shape)
        h_out = shape[1]
        flat = kv_cache.view(torch.uint8).reshape(-1, ROW_BYTES)
        res = torch.zeros(shape, dtype=torch.float32, device=q.device)
        with true_fp32_matmul():
            for r in range(q.shape[0]):
                idx = indices[r]
                idx = idx[idx >= 0].long()
                if idx.numel() == 0:
                    continue
                rows = flat[idx]
                codes = rows[:, :D_NOPE].contiguous().view(torch.float8_e4m3fn).float()
                scale = rows[:, D_NOPE : D_NOPE + 16].contiguous().view(torch.float32)
                nope = (codes.view(-1, D_NOPE // TILE, TILE) * scale.unsqueeze(-1)).view(-1, D_NOPE)
                rope = rows[:, D_NOPE + 16 :].contiguous().view(torch.bfloat16).float()
                k = torch.cat([nope, rope], 1)
                s = (q[r, :h_out].float() @ k.T) * softmax_scale
                res[r] = torch.softmax(s, -1) @ nope
        return res.to(torch.bfloat16 if out is None else out.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        sched_meta,
        softmax_scale: float,
        out: torch.Tensor | None,
    ) -> None:
        if q.shape[1] not in KERNEL_HEADS:
            raise ValueError(f"h_q {q.shape[1]} is not one of the kernel's {KERNEL_HEADS}")
        if out is not None and out.shape[1] > q.shape[1]:
            raise ValueError("out names more heads than q carries")
        if indices.dtype != torch.int32:
            raise ValueError("indices must be int32 slots")
        if kv_cache.shape[-1] != ROW_BYTES or kv_cache.dim() != 4 or kv_cache.shape[2] != 1:
            raise ValueError("kv_cache must be [pages, page_size, 1, 656]")
        if q.shape[0] > MAX_ROWS:
            raise ValueError(
                f"{q.shape[0]} rows in one call, more than {MAX_ROWS}: FlashMLA's decode "
                "scheduler exits the process past its shared-memory row limit (11622 rows on "
                "sm_103); split the rows into sub-batches"
            )


flash_mla_sparse_decode = FlashMlaSparseDecode()
