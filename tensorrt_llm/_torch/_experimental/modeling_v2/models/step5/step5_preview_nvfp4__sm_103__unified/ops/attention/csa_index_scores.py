# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA indexer scores: weighted-ReLU dots of e4m3 query heads against paged region summaries."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Step-5's sparse indexer: 4 heads per provider group, proxy dim 256, 8-token
#: regions, 32-token pages (4 regions per page), and 1152 regions at
#: `max_seq_len` 9216.
_CHECKPOINT = dict(
    heads_per_group=4,
    proxy_dim=256,
    region_size=8,
    tokens_per_block=32,
    q_dtype=torch.bfloat16,
    weights_dtype=torch.float32,
    positions_dtype=torch.int32,
    table_dtype=torch.int32,
)

#: The decode site's fixed width: `max_seq_len // region_size`.
_DECODE_COLUMNS = 1152
#: Pages per table row at `max_seq_len` 9216 and 32-token pages.
_TABLE_PAGES = 288


def _spec(**overrides) -> dict:
    """A cell spec: this checkpoint's geometry unless the cell says otherwise."""
    return {**_CHECKPOINT, **overrides}


#: Rows gathered per chunk on the per-row-table path: bounds the gathered keys to
#: 2^26 elements (128 MiB in bf16) whatever the row count.
_GATHER_ELEMENTS = 1 << 26


class CsaIndexScores(OpWrapper):
    """Score every complete region a query row can see, per provider group.

    Tier 4: composed from torch ops in `__call__` (an e4m3 round trip, an index
    gather, `torch.mm` / `torch.bmm` with bf16 operands and `out_dtype=float32`,
    relu, a weighted head sum, a mask). For query row `t`, group `g` and region
    `r`::

        heads     = g * Hg .. g * Hg + Hg - 1          (Hg = H // num_groups)
        s_r       = summary_cache[block_table[row, r // R], r % R]
                    (row = request_indices[t], or row 0 when the table has one row;
                     R = summary_cache.shape[1], the regions per page)
        out[g * T + t, r] = sum_h relu(e4m3(q[t, h]) . e4m3(s_r)) * weights[t, h]

    for `r < positions[t] // region_size` (the row's complete history regions;
    its current region is excluded), and `-inf` at every other column. Rows are
    group-major: all `T` rows of group 0, then group 1, ... -- the order the
    region top-k consumes.

    `e4m3(x)` is `x.to(torch.float8_e4m3fn)` read back: round to nearest even,
    saturating at +-448 on this build (measured, see `note`). Every e4m3 value is
    exact in bf16, so the dots run on bf16 operands with fp32 accumulation and
    every product is exact; only summation order is left to the GEMM.

    The summary plane is any `[pages, R, D]` view: the region's last-token slot
    of a paged per-token plane is `plane[:, region_size - 1 :: region_size, :D]`
    (R = tokens_per_block // region_size), a dense per-region plane is itself.
    Its dtype is bf16 holding e4m3 values (re-rounding is the identity) or
    `float8_e4m3fn`. The target passes the dense `[pages, 4, 256]` e4m3 plane
    (`Step5CacheManager.get_index_summary_buffer`) at every call site, and the
    cells drive it at groups 1 (tp4/tep4) and 4 (dep4) on both table paths:
    per-row decode at 1, 24 and 128 rows on 1152 columns, shared-table prefill
    at 512 to 8192 rows on 63 to 1152 columns, and a decode row on the
    measured 84979-page pool. The anchor view (`plane[:, 7::8, :256]` of a bf16
    k|z plane) is driven at groups 1 and 4 as well.

    Two shape-selected paths, no data-dependent branch: a one-row `block_table`
    is shared by every query row (one gather, one `mm` per head; the prefill
    site, one request per call) -- `request_indices` is not read then; otherwise
    each row gathers its own table row's summaries (`bmm` per head, rows in
    chunks of at most 2^26 gathered elements; the decode site).

    Rejected at cheaper rungs: `torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits`
    (DeepSeek DSA indexer) computes `sum_h relu(q_h . k) * w_h` but only at
    head dim 128 (`SUPPORTED_HEAD_DIM = 128`), over per-token keys with a
    per-token scale; qwen4_exp's `qsa_paged_index_scores` has no per-head
    weight, a fixed `head_dim ** -0.5` scale and reads one key per compressed
    block at the group's anchor with compress ratio 4; FlashInfer's
    `msa_ops.msa_proxy_score` is SM120/121 only.

    Fusion boundary: the e4m3 rounding of q and of the summaries, the table walk,
    the dots, relu, weighting, head sum and causal mask are inside. Producing q
    (projection, norm, RoPE), the `x 0.5` on the weights, compressing and storing
    the summaries, and the top-k are not. Returns a fresh fp32
    `[num_groups * T, num_columns]`; nothing is written in place.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: groups, rows (query rows), layout ("anchor": the last-token slot of a
    # bf16 [pages, 32, 512] k|z plane; "dense": a [pages, 4, 256] e4m3 region
    # plane), table ("per_row" with `slots` table rows > requests, or "shared":
    # one row), requests as (first position, rows) segments, num_columns,
    # pool_pages_spare (unmapped pool pages beyond those the requests map; the
    # pool's highest page is always one a request maps).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode dp: 128 rows x 4 groups, 1152 columns, positions 0..9215",
            spec=_spec(
                groups=4,
                layout="anchor",
                table="per_row",
                slots=160,
                decode_positions="ragged",
                rows=128,
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=512,
            ),
        ),
        Cell(
            why="decode tp4: 128 rows x 1 group, 1152 columns",
            spec=_spec(
                groups=1,
                layout="anchor",
                table="per_row",
                slots=160,
                decode_positions="ragged",
                rows=128,
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=512,
            ),
        ),
        Cell(
            why="decode dp: 1 row at position 9215, all 1152 regions visible",
            spec=_spec(
                groups=4,
                layout="anchor",
                table="per_row",
                slots=8,
                segments=((9215, 1),),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="decode dp on a dense e4m3 region plane, 64 rows",
            spec=_spec(
                groups=4,
                layout="dense",
                table="per_row",
                slots=96,
                decode_positions="ragged",
                rows=64,
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=256,
            ),
        ),
        Cell(
            why="prefill dp: 8192-row chunk at past 1024, shared one-row table",
            spec=_spec(
                groups=4,
                layout="anchor",
                table="shared",
                segments=((1024, 8192),),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="prefill tp4: 8192 rows from 0, 1023 columns (last position // 8)",
            spec=_spec(
                groups=1,
                layout="anchor",
                table="shared",
                segments=((0, 8192),),
                num_columns=1023,
                table_pages=256,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="chunked prefill dp: 512-row chunk at past 8192, 1087 columns",
            spec=_spec(
                groups=4,
                layout="anchor",
                table="shared",
                segments=((8192, 512),),
                num_columns=1087,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="per-row tables, 3 requests x 200 rows: 600 rows in 3 gather chunks",
            spec=_spec(
                groups=4,
                layout="anchor",
                table="per_row",
                slots=12,
                segments=((3000, 200), (7, 200), (8900, 200)),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense tp4 decode: 1 row at 9215, 1 group, 1152 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="per_row",
                slots=8,
                segments=((9215, 1),),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense tp4 decode: 24 ragged rows, 1 group, 1152 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="per_row",
                slots=40,
                decode_positions="ragged",
                rows=24,
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=256,
            ),
        ),
        Cell(
            why="dense tp4 decode: 128 ragged rows, 1 group, 1152 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="per_row",
                slots=160,
                decode_positions="ragged",
                rows=128,
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=512,
            ),
        ),
        Cell(
            why="dense dp decode: 1 row at 9215, 4 groups, 1152 columns",
            spec=_spec(
                groups=4,
                layout="dense",
                table="per_row",
                slots=8,
                segments=((9215, 1),),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense dp decode: 128 ragged rows, 4 groups, 1152 columns",
            spec=_spec(
                groups=4,
                layout="dense",
                table="per_row",
                slots=160,
                decode_positions="ragged",
                rows=128,
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=512,
            ),
        ),
        Cell(
            why="dense tp4 prefill: 512 rows from 0, shared table, 63 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="shared",
                segments=((0, 512),),
                num_columns=63,
                table_pages=16,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense tp4 prefill: 8192 rows from 0, shared table, 1023 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="shared",
                segments=((0, 8192),),
                num_columns=1023,
                table_pages=256,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense tp4 chunked prefill: 512 rows at past 8192, 1087 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="shared",
                segments=((8192, 512),),
                num_columns=1087,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense tp4 prefill: 4966 rows at past 1135, shared table, 762 columns",
            spec=_spec(
                groups=1,
                layout="dense",
                table="shared",
                segments=((1135, 4966),),
                num_columns=762,
                table_pages=191,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense dp prefill: 8192 rows at past 1024, shared table, 1152 columns",
            spec=_spec(
                groups=4,
                layout="dense",
                table="shared",
                segments=((1024, 8192),),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense dp prefill: 512 rows from 0, shared table, 63 columns",
            spec=_spec(
                groups=4,
                layout="dense",
                table="shared",
                segments=((0, 512),),
                num_columns=63,
                table_pages=16,
                pool_pages_spare=64,
            ),
        ),
        Cell(
            why="dense tp4 decode on the measured pool: 84979 pages, last page mapped",
            spec=_spec(
                groups=1,
                layout="dense",
                table="per_row",
                slots=8,
                segments=((9215, 1),),
                num_columns=_DECODE_COLUMNS,
                table_pages=_TABLE_PAGES,
                pool_pages_spare=84691,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. What a caller passes:
    `q` the indexer's normed and roped `[T, H, 256]` query heads (bf16; any
    strides, e.g. a head view of the q|k|z|w projection), `weights` `[T, H]` the
    per-head weights already multiplied by 0.5 (fp32), `positions` each row's
    absolute position (the token being processed), `block_table` the index
    plane's page table (int32, -1 past each sequence) and `request_indices` the
    table row of each query row. Heads `g*Hg .. g*Hg+Hg-1` form group `g`.

    Preconditions (not checked, data-dependent): every table entry a row's
    visible regions reach is a valid page of `summary_cache` (an entry past the
    pool is an out-of-range gather and a device-side assert; -1 entries are
    clamped to page 0 and only ever feed masked columns); the summaries of every
    visible region were stored; `positions[t] // region_size <= num_columns`
    -- a row whose history is wider than the output loses its regions past
    `num_columns` silently (`test_history_wider_than_the_output_is_truncated`
    pins it). The op raises `ValueError` on every shape mismatch it can see
    (heads not divisible by `num_groups`, weights/positions/request_indices not
    one entry per row, summary width != q's, a per-row table with fewer than
    `num_columns / R` pages), so `is_valid` adds nothing.

    e4m3 rounding: `.to(torch.float8_e4m3fn)` on this build saturates -- 455,
    464, 500, 1e4 and +inf all read back as 448, -500 as -448 (measured on
    bf16 and fp32 inputs). The reference uses the same cast, so a saturated
    operand is certified as torch rounds it, not as a non-saturating cast
    (NaN) would.

    Accuracy: the test's band is the worst-case summation bound
    (D + Hg + 2) * 2^-24 * sum_h |w_h| sum_d |q_hd s_rd| per element (every
    e4m3 product is exact in fp32). Measured worst over all cells: 0.0032 of
    that band. Wrong readings land outside it by 401x (q not e4m3-rounded),
    1.5e4x (relu after the head sum), 4.2e4x (heads interleaved across groups),
    4.4e4x (query-major rows) and 2.0e6x (summary read from the region's first
    token slot).

    Columns at or past a row's bound are exactly `-inf`, never garbage, so a
    consumer may scan the full width; `-1` table entries are never read
    unmasked. A row with position < region_size has no visible region and is
    all `-inf`. `T == 0` returns an empty `[0, num_columns]` without a launch.

    CUDA graphs: the per-row-table path is capturable (no host sync, all shapes
    from host ints); `test_cuda_graph_capture_and_replay` captures 8 decode rows
    at 1152 columns on the anchor view at 4 groups, and
    `test_cuda_graph_capture_and_replay_dense_one_group` captures 8 and 128
    rows on the dense e4m3 plane at 1 group (the tp4/tep4 decode site); each
    replays after rewriting q, weights, positions and summaries in place, and
    every replay matched an eager call bitwise. The shared path is the same op
    sequence but is driven eager only.

    Memory: the shared path holds three `[T, G, C]` fp32 temporaries (453 MiB
    at 8192 x 4 x 1152); the per-row path gathers at most 2^26 bf16 elements
    per chunk. The measured tep4 pool (2719328 KV tokens, 84979 pages of 32
    tokens) is one dense `[84979, 4, 256]` e4m3 plane per layer, 83 MiB; a cell
    drives it with its highest page (84978) mapped.

    Hard limits: output `G * T * C` elements reach 2^31 at T = 466033 rows for
    G 4 x C 1152 -- not reached (8192 rows max). Gather indices are int64 inside
    torch; no index passes through fp32. The dense plane's element offset
    peaks at 84979 * 1024 = 8.7e7, far below 2^31. Not probed.

    Not certified: other head counts per group, proxy dims, region sizes or
    regions per page (R 4 only, from 32-token pages), q in fp16/fp32, bf16
    weights, int64 positions or tables, more than 8192 rows, a shared table
    with more than 1 row reused across requests, positions past 9215 and
    widths other than 1152 / 1087 / 1023 / 762 / 63; dense-plane decode
    batches other than 1, 8, 24, 64 and 128 rows; pools larger than 84979
    pages.
    """

    def __call__(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        summary_cache: torch.Tensor,
        block_table: torch.Tensor,
        request_indices: torch.Tensor,
        positions: torch.Tensor,
        num_groups: int,
        region_size: int,
        num_columns: int,
    ) -> torch.Tensor:
        rows, heads, dim = q.shape
        pages, regions_per_page, summary_dim = summary_cache.shape
        if heads % num_groups:
            raise ValueError(f"{heads} q heads do not split into {num_groups} groups")
        if weights.shape != (rows, heads):
            raise ValueError(f"weights must be [{rows}, {heads}], got {tuple(weights.shape)}")
        if positions.shape != (rows,) or request_indices.shape != (rows,):
            raise ValueError("positions and request_indices need one entry per row")
        if summary_dim != dim:
            raise ValueError(f"summary width {summary_dim} != q head dim {dim}")
        table_pages = -(-num_columns // regions_per_page)
        if block_table.dim() != 2 or block_table.shape[1] < table_pages:
            raise ValueError(
                f"block_table must be [rows, >= {table_pages}] for {num_columns} columns "
                f"at {regions_per_page} regions per page, got {tuple(block_table.shape)}"
            )
        per_group = heads // num_groups
        out = torch.zeros(num_groups, rows, num_columns, dtype=torch.float32, device=q.device)
        if rows == 0 or num_columns == 0:
            return out.view(num_groups * rows, num_columns)

        columns = torch.arange(num_columns, device=q.device)
        physical = block_table[:, :table_pages].long()[:, columns // regions_per_page]
        physical = physical.clamp_min(0)
        in_page = columns % regions_per_page
        q8 = q.to(torch.float8_e4m3fn).to(torch.bfloat16).reshape(rows, num_groups, per_group, dim)
        w = weights.float().reshape(rows, num_groups, per_group)

        if block_table.shape[0] == 1:
            keys = summary_cache[physical[0], in_page]
            keys = keys.to(torch.float8_e4m3fn).to(torch.bfloat16).t()
            for h in range(per_group):
                dots = torch.mm(
                    q8[:, :, h].reshape(rows * num_groups, dim), keys, out_dtype=torch.float32
                ).view(rows, num_groups, num_columns)
                out += (torch.relu(dots) * w[:, :, h, None]).transpose(0, 1)
        else:
            chunk = max(1, _GATHER_ELEMENTS // (num_columns * dim))
            requests = request_indices.long()
            for lo in range(0, rows, chunk):
                hi = min(lo + chunk, rows)
                keys = summary_cache[physical[requests[lo:hi]], in_page]
                keys = keys.to(torch.float8_e4m3fn).to(torch.bfloat16).transpose(1, 2)
                for h in range(per_group):
                    dots = torch.bmm(q8[lo:hi, :, h], keys, out_dtype=torch.float32)
                    out[:, lo:hi] += (torch.relu(dots) * w[lo:hi, :, h, None]).transpose(0, 1)

        bound = torch.div(positions.long(), region_size, rounding_mode="floor")
        hidden = columns[None, :] >= bound[:, None]
        out.masked_fill_(hidden[None], float("-inf"))
        return out.view(num_groups * rows, num_columns)

    def reference(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        summary_cache: torch.Tensor,
        block_table: torch.Tensor,
        request_indices: torch.Tensor,
        positions: torch.Tensor,
        num_groups: int,
        region_size: int,
        num_columns: int,
    ) -> torch.Tensor:
        """fp32 scores accumulated in float64 and rounded once; `-inf` past each
        row's bound.

        Walks the table page by page (a page's `R` summaries are its regions
        `p*R .. p*R+R-1`) rather than by column, and scores one table row's
        query rows at a time, so neither the column-to-page arithmetic nor the
        per-row gather of `__call__` is shared. Float64 makes every product and
        every partial sum exact or within 2^-53 relative, so this side's error is
        far below one fp32 ulp of the result.
        """
        rows, heads, dim = q.shape
        per_group = heads // num_groups
        regions_per_page = summary_cache.shape[1]
        table_pages = -(-num_columns // regions_per_page)
        shared = block_table.shape[0] == 1
        table_rows = (
            torch.zeros(rows, dtype=torch.long, device=q.device)
            if shared
            else request_indices.long()
        )
        q8 = q.to(torch.float8_e4m3fn).double().reshape(rows, num_groups, per_group, dim)
        w = weights.double().reshape(rows, num_groups, per_group)
        scores = torch.zeros(rows, num_groups, num_columns, dtype=torch.float64, device=q.device)
        for row in torch.unique(table_rows).tolist():
            members = (table_rows == row).nonzero().squeeze(1)
            page_ids = block_table[row, :table_pages].long().clamp_min(0)
            logical = summary_cache[page_ids].reshape(table_pages * regions_per_page, dim)
            keys = logical[:num_columns].to(torch.float8_e4m3fn).double()
            dots = torch.einsum("tghd,cd->tghc", q8[members], keys)
            scores[members] = (torch.relu(dots) * w[members][..., None]).sum(dim=2)
        bound = positions.long() // region_size
        visible = torch.arange(num_columns, device=q.device)[None, :] < bound[:, None]
        scores = scores.float().masked_fill(~visible[:, None, :], float("-inf"))
        return scores.transpose(0, 1).reshape(num_groups * rows, num_columns)


csa_index_scores = CsaIndexScores()
