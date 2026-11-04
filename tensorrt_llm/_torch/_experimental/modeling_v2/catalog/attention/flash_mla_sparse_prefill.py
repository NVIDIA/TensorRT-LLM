# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MQA of context rows over a bf16 latent workspace: FlashMLA's sm100f sparse prefill."""

from __future__ import annotations

import torch

from tensorrt_llm.flash_mla import flash_mla_sparse_fwd as _flash_mla_sparse_fwd

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

D_NOPE = 512
D_QK = 576
KERNEL_HEADS = (64, 128)
#: The tep4, dep4 and tep8 targets' bounded context workspace at max_seq_len 1048576:
#: B = 2**19 bf16 rows.
_WS_ROWS = 2**19
#: The row sweep's pre-merge share of the 32768 query row counts: 1..65, 2**k - 1, 2**k and
#: 2**k + 1 up to 32768, every 4093rd count, and 32736 (a chunk with generation rows behind).
_ROW_SAMPLE = tuple(
    sorted(
        [
            *range(1, 66),
            *[2**k + d for k in range(7, 16) for d in (-1, 0, 1) if 2**k + d <= 32768],
            *range(4093, 32768, 4093),
            32736,
        ]
    )
)
#: The dep4 row sweep's pre-merge share of the 32768 query row counts: 1..65, 2**k - 1,
#: 2**k and 2**k + 1 up to 32768, every 4093rd count, 29127 and 29128 (q's byte offset
#: passes 2**31 inside row 29127) and 32767.
_DEP4_ROW_SAMPLE = tuple(
    sorted(
        [
            *range(1, 66),
            *[2**k + d for k in range(7, 16) for d in (-1, 0, 1) if 2**k + d <= 32768],
            *range(4093, 32768, 4093),
            29127,
            29128,
        ]
    )
)
#: The tep8 row sweep's pre-merge share of the 32768 query row counts: 1..65, 2**k - 1, 2**k
#: and 2**k + 1 up to 32768, every 4093rd count, 29127 and 29128 (q's byte offset passes
#: 2**31 inside row 29127) and 32736 (a chunk with a generation row behind it).
_TEP8_ROW_SAMPLE = tuple(
    sorted(
        [
            *range(1, 66),
            *[2**k + d for k in range(7, 16) for d in (-1, 0, 1) if 2**k + d <= 32768],
            *range(4093, 32768, 4093),
            29127,
            29128,
            32736,
        ]
    )
)


class FlashMlaSparsePrefill(OpWrapper):
    """For every query row r and head h < `out.shape[1]`::

        p = softmax_j(softmax_scale * q[r, h] . kv[j]),  j in indices[r] (-1 skipped)
        out[r, h] = sum_j p[j] * kv[j][:512]

    `kv`: bf16 `[N, 576]` (the dequantized fp8_ds_mla rows, `ds_mla_dequant_gather`).
    `indices`: int32 `[rows, topk]` rows of `kv`, -1 padded. `q`: bf16 `[rows, h_q, 576]`,
    h_q 64 or 128 (fewer real heads first, zeros after; `out` names how many are kept).
    Writes `out` (bf16 `[rows, h_out, 512]`) and returns it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `rows` query rows of one or more requests of `kv_len` keys (the last `new` positions
    # of each, all of them without `new`), `h_q` kernel heads with `h_out` real, `topk` causal
    # selections; `holes` scatters -1. The "1M tp4" cells add `kv_rows`: `kv` is the first
    # N = sum(kv_len) rows of a workspace buffer of that many, its later rows stale (NaN), and
    # `out` leads a longer buffer whose tail stays untouched; `selection="topk"`: each row's
    # request-local selection as the indexer's top-k returns it (positions 0..p in order then
    # -1 while p + 1 <= topk, else topk distinct positions spread over [0, p], 0 and p among
    # them, in shuffled order); `dense`: the engine's dense causal index for a skipped indexer
    # (0..p, then -1). Both are offset to the request's run in the workspace, -1 kept. With
    # `kv_rows`, a "tep4 1M", "dep4 1M" or "tep8 1M" cell is the tep4, dep4 or tep8 target's
    # bounded context workspace: kv is the first sum(kv_len) rows of a `kv_rows`-row buffer
    # whose later rows are stale, each request's run at its cumulative offset, and the
    # selections are the indexer's top-`topk` shape (`selection="topk"`, which the "dep4 1M"
    # cells name: distinct, spread over [0, p], 0 and p among them; a row that sees fewer
    # keys takes all of them, then -1) or with `dense` DSA's dense causal index.
    # `row_sweep` also calls q[:m], indices[:m] and out[:m] for every m in 1..rows, and
    # `row_counts` for the m it lists: each m's rows bitwise the full call's leading rows, and
    # out's rows past m unwritten.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tep4/tp4: 32768 rows of one request, 16 heads padded to 64",
            spec=dict(rows=32768, kv_len=[32768], h_q=64, h_out=16, topk=2048),
        ),
        Cell(
            why="dep4: 8192 rows, 64 heads",
            spec=dict(rows=8192, kv_len=[8192], h_q=64, h_out=64, topk=2048),
        ),
        Cell(
            why="three requests with cached prefixes, 16 of 64 heads",
            spec=dict(
                rows=2348, kv_len=[300, 3076, 5186], new=[300, 2047, 1], h_q=64, h_out=16, topk=2048
            ),
        ),
        Cell(
            why="short rows with scattered -1 holes",
            spec=dict(rows=700, kv_len=[700], h_q=64, h_out=16, topk=2048, holes=True),
        ),
        Cell(
            why="tep8: 32768 rows of one request, 8 heads padded to 64",
            spec=dict(rows=32768, kv_len=[32768], h_q=64, h_out=8, topk=2048),
        ),
        # The tp4 target at the 1M envelope, workspace path (PrefillTarget._attn_ctx, every
        # layer, eager): a step whose context KV (cached + new over its context requests) is
        # at most B = 2**19 rows attends over kv = buf[:N] of one bf16 [B, 576] buffer; q is
        # the context rows of the [T, 64, 576] fused_q scratch, this rank's 16 heads first.
        Cell(
            why="1M tp4 index mode at N = B = 2**19: a 32768-row chunk behind a 491520-token "
            "prefix, each row's 2048 distinct selections spread over its whole prefix, row 0 "
            "and its own among them (workspace rows to 524287)",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=524288,
                selection="topk",
            ),
        ),
        Cell(
            why="1M tp4 index mode, packed chunk step: four requests cached 262144/131072/0/"
            "2048 + 16384/8192/4096/4096, runs at workspace offsets 0/278528/417792/421888 "
            "(N = 428032 of the 2**19-row buffer), short rows 0..p then -1",
            spec=dict(
                rows=32768,
                kv_len=[278528, 139264, 4096, 6144],
                new=[16384, 8192, 4096, 4096],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=524288,
                selection="topk",
            ),
        ),
        Cell(
            why="1M tp4 index mode, a 16-row chunk tail behind a 524272-token prefix (the "
            "workspace path's largest), selections to workspace row 524287",
            spec=dict(
                rows=16,
                kv_len=[524288],
                new=[16],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=524288,
                selection="topk",
            ),
        ),
        Cell(
            why="1M tp4 short step in the stale 2**19-row buffer: kv = buf[:8192] of one "
            "8192-token prompt, every row past it NaN",
            spec=dict(
                rows=8192,
                kv_len=[8192],
                new=[8192],
                kv_rows=524288,
                h_q=64,
                h_out=16,
                topk=2048,
                selection="topk",
            ),
        ),
        Cell(
            why="1M tp4 dense mode (the engine skipped the context indexer): three requests "
            "2048/1777/700 cached 1748/1700/0, the dense causal index 0..p then -1 at "
            "workspace offsets 0/2048/3825",
            spec=dict(
                rows=1077,
                kv_len=[2048, 1777, 700],
                new=[300, 77, 700],
                h_q=64,
                h_out=16,
                topk=2048,
                dense=True,
                kv_rows=524288,
            ),
        ),
        Cell(
            why="1M tp4 dense mode: sixteen 2048-token prompts in one 32768-row step, the "
            "dense causal index over 16 runs of the workspace",
            spec=dict(
                rows=32768,
                kv_len=[2048] * 16,
                new=[2048] * 16,
                h_q=64,
                h_out=16,
                topk=2048,
                dense=True,
                kv_rows=524288,
            ),
        ),
        # The workspace path's steps of a 1M prompt: chunk k is 32768 rows behind a
        # 32768 * (k - 1)-token prefix, N = 32768 * k, for k = 2..15 (chunk 1 is the shared
        # 32768-row cell's N, chunk 16 the N = B cell above; past it the decode path).
        *(
            Cell(
                why=f"1M tp4 index mode at N = {n}: a 1M prompt's chunk {n // 32768}, 32768 "
                f"rows behind a {n - 32768}-token prefix, each row's 2048 distinct selections "
                "spread over its whole prefix, row 0 and its own among them",
                spec=dict(
                    rows=32768,
                    kv_len=[n],
                    new=[32768],
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_rows=524288,
                    selection="topk",
                ),
            )
            for n in range(65536, 524288, 32768)
        ),
        # A fresh request's dense step under 16 rows: the engine's warmup steps (1 and 2
        # rows) and a short request (7), the request's own new tokens its whole KV.
        *(
            Cell(
                why=f"1M tp4 dense mode, a fresh {n}-row step (the engine's warmup steps, a "
                f"short request): the dense causal index 0..p over N = {n} workspace rows",
                spec=dict(
                    rows=n,
                    kv_len=[n],
                    new=[n],
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    dense=True,
                    kv_rows=524288,
                ),
            )
            for n in (1, 2, 7)
        ),
        Cell(
            why="tep4 1M workspace: a 32768-row chunk at cached 491520 selects over all 2**19 rows",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep4 1M workspace: a packed step, four runs, the last at offsets 421888..428031",
            spec=dict(
                rows=32768,
                kv_len=[278528, 139264, 4096, 6144],
                new=[16384, 8192, 4096, 4096],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep4 1M workspace: a 16-row chunk tail over the whole 2**19-row prefix",
            spec=dict(
                rows=16,
                kv_len=[524288],
                new=[16],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep4 1M workspace: an 8192-row first chunk, the buffer's later rows stale",
            spec=dict(
                rows=8192,
                kv_len=[8192],
                new=[8192],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep4 1M workspace dense: three requests' causal index over cached prefixes",
            spec=dict(
                rows=1077,
                kv_len=[2048, 1777, 700],
                new=[300, 77, 700],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        Cell(
            why="tep4 1M workspace dense: sixteen 2048-token prompts in one 32768-row step",
            spec=dict(
                rows=32768,
                kv_len=[2048] * 16,
                new=[2048] * 16,
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        # The 1M landing's workspace steps outside the cells above: chunks 1..15 of a 1M
        # prompt cut to 32736 rows by the generation row behind each (the last at N 491040),
        # and the dense fresh steps of 1, 2 and 7 rows (the engine's warmup, a short prompt).
        # Any other count in 1..32768 is a prompt's last chunk: the row sweep.
        Cell(
            why="tep4 1M workspace: a 32736-row chunk at cached 458304, N 491040, a generation row "
            "behind it",
            spec=dict(
                rows=32736,
                kv_len=[491040],
                new=[32736],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep4 1M workspace dense: a fresh 1-row step, N 1 (the engine's warmup)",
            spec=dict(
                rows=1,
                kv_len=[1],
                new=[1],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        Cell(
            why="tep4 1M workspace dense: a fresh 2-row step, N 2 (the engine's warmup)",
            spec=dict(
                rows=2,
                kv_len=[2],
                new=[2],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        Cell(
            why="tep4 1M workspace dense: a fresh 7-row step, N 7 (a short prompt)",
            spec=dict(
                rows=7,
                kv_len=[7],
                new=[7],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        Cell(
            why="tep4 1M workspace row sweep, large: every m 1..32768 of a 32768-row chunk over "
            "all 2**19 rows, each m bitwise the full call's leading rows (195 s)",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                row_sweep=True,
                large=True,
            ),
        ),
        Cell(
            why="tep4 1M workspace row sample: the row sweep's pre-merge share, m 1..65, "
            "2**k +- 1, every 4093rd, 32736 and 32767 (0.3 s)",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=_WS_ROWS,
                row_counts=_ROW_SAMPLE,
            ),
        ),
        Cell(
            why="dep4 1M workspace, 64 live heads: a fresh prompt's first 32768-row chunk "
            "(cached 0, the KV-capacity estimate's step)",
            spec=dict(
                rows=32768,
                kv_len=[32768],
                new=[32768],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
            ),
        ),
        Cell(
            why="dep4 1M workspace, 64 live heads: a 32768-row chunk at cached 491520 selects "
            "over all 2**19 rows",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
            ),
        ),
        Cell(
            why="dep4 1M workspace, 64 live heads: a packed step of four of the rank's requests, "
            "the last run at offsets 421888..428031",
            spec=dict(
                rows=32768,
                kv_len=[278528, 139264, 4096, 6144],
                new=[16384, 8192, 4096, 4096],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
            ),
        ),
        Cell(
            why="dep4 1M workspace, 64 live heads: a 16-row chunk tail over the whole 2**19-row "
            "prefix",
            spec=dict(
                rows=16,
                kv_len=[524288],
                new=[16],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
            ),
        ),
        Cell(
            why="dep4 1M workspace, 64 live heads: an 8192-row first chunk, the buffer's later "
            "rows stale",
            spec=dict(
                rows=8192,
                kv_len=[8192],
                new=[8192],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
            ),
        ),
        Cell(
            why="dep4 1M workspace dense, 64 live heads: three requests' causal index over "
            "cached prefixes",
            spec=dict(
                rows=1077,
                kv_len=[2048, 1777, 700],
                new=[300, 77, 700],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        Cell(
            why="dep4 1M workspace dense, 64 live heads: sixteen 2048-token prompts of the rank "
            "in one 32768-row step",
            spec=dict(
                rows=32768,
                kv_len=[2048] * 16,
                new=[2048] * 16,
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        # The 1M landing's workspace steps outside the cells above: chunk k of a 1M prompt is
        # 32768 rows behind 32768 * (k - 1) cached keys over N = 32768 * k rows, k = 2..15
        # (chunk 1 and chunk 16 at N = 2**19 are cells above; past 2**19 the decode path).
        *(
            Cell(
                why=f"dep4 1M workspace, 64 live heads: chunk {n // 32768} of a 1M prompt, 32768 "
                f"rows behind {n - 32768} cached keys over N = {n} workspace rows",
                spec=dict(
                    rows=32768,
                    kv_len=[n],
                    new=[32768],
                    h_q=64,
                    h_out=64,
                    topk=2048,
                    kv_rows=_WS_ROWS,
                    selection="topk",
                ),
            )
            for n in range(65536, 524288, 32768)
        ),
        # Fresh dense steps of every rank: the engine's warm-up steps of 1 and 2 rows (and the
        # mixed warm-up step's one context row beside a generation row) and a 7-token prompt,
        # the request's own new tokens its whole KV.
        *(
            Cell(
                why=f"dep4 1M workspace dense, 64 live heads: {n}-row fresh step over N = {n} "
                "workspace rows (the warm-up steps, a short prompt)",
                spec=dict(
                    rows=n,
                    kv_len=[n],
                    new=[n],
                    h_q=64,
                    h_out=64,
                    topk=2048,
                    kv_rows=_WS_ROWS,
                    dense=True,
                ),
            )
            for n in (1, 2, 7)
        ),
        # Every other row count in 1..32768 is a prompt's last chunk, a short prompt, or a chunk
        # cut short by the rank's generation rows behind it: the row sweep, at N = 2**19.
        Cell(
            why="dep4 1M workspace, 64 live heads, row sweep (large): every m 1..32768 of a "
            "32768-row chunk over all 2**19 rows, each m bitwise the full call's first m rows",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
                row_sweep=True,
                large=True,
            ),
        ),
        Cell(
            why="dep4 1M workspace, 64 live heads, row sample: the row sweep's pre-merge share, "
            "m 1..65, 2**k - 1 / 2**k / 2**k + 1, every 4093rd, 29127, 29128 and 32767",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=64,
                topk=2048,
                kv_rows=_WS_ROWS,
                selection="topk",
                row_counts=_DEP4_ROW_SAMPLE,
            ),
        ),
        Cell(
            why="tep8 1M workspace: a 32768-row chunk at cached 491520 selects over all 2**19 rows",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep8 1M workspace: a packed step, four runs, the last at offsets 421888..428031",
            spec=dict(
                rows=32768,
                kv_len=[278528, 139264, 4096, 6144],
                new=[16384, 8192, 4096, 4096],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep8 1M workspace: a 16-row chunk tail over the whole 2**19-row prefix",
            spec=dict(
                rows=16,
                kv_len=[524288],
                new=[16],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep8 1M workspace: an 8192-row first chunk, the buffer's later rows stale",
            spec=dict(
                rows=8192,
                kv_len=[8192],
                new=[8192],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        Cell(
            why="tep8 1M workspace dense: three requests' causal index over cached prefixes",
            spec=dict(
                rows=1077,
                kv_len=[2048, 1777, 700],
                new=[300, 77, 700],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        Cell(
            why="tep8 1M workspace dense: sixteen 2048-token prompts in one 32768-row step",
            spec=dict(
                rows=32768,
                kv_len=[2048] * 16,
                new=[2048] * 16,
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
                dense=True,
            ),
        ),
        # The 1M landing's workspace steps outside the cells above, as they ran: chunk k of a
        # 1,040,000-token prompt is 32736 rows (cut from 32768 by a 7-token prompt's generation
        # row riding behind it) behind 32736 * (k - 1) cached keys over N = 32736 * k rows,
        # k = 1..15; chunk 16 is 32768 rows behind 491040 over N = 523808; past 2**19 the
        # decode path.
        *(
            Cell(
                why=f"tep8 1M workspace: chunk {n // 32736} of a 1M prompt, 32736 rows (a "
                f"generation row behind) after {n - 32736} cached keys over N = {n} rows",
                spec=dict(
                    rows=32736,
                    kv_len=[n],
                    new=[32736],
                    h_q=64,
                    h_out=8,
                    topk=2048,
                    kv_rows=_WS_ROWS,
                ),
            )
            for n in range(32736, 491041, 32736)
        ),
        Cell(
            why="tep8 1M workspace: chunk 16 of a 1M prompt, 32768 rows after 491040 cached keys "
            "over N = 523808 rows",
            spec=dict(
                rows=32768,
                kv_len=[523808],
                new=[32768],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
            ),
        ),
        # Fresh dense steps (the engine skipped the indexer): the warm-up's 1-row and 2-row
        # steps, the mixed warm-up step's one context row beside a generation row, and a 7-token
        # prompt, the request's own new tokens its whole KV.
        *(
            Cell(
                why=f"tep8 1M workspace dense: a fresh {n}-row step, N = {n} (the engine's "
                "warm-up steps, a short prompt)",
                spec=dict(
                    rows=n,
                    kv_len=[n],
                    new=[n],
                    h_q=64,
                    h_out=8,
                    topk=2048,
                    kv_rows=_WS_ROWS,
                    dense=True,
                ),
            )
            for n in (1, 2, 7)
        ),
        # Every other row count in 1..32768 is a prompt's last chunk, a short prompt, or a chunk
        # cut short by the generation rows behind it: the row sweep, at N = 2**19.
        Cell(
            why="tep8 1M workspace row sweep (large): every m 1..32768 of a 32768-row chunk over "
            "all 2**19 rows, each m bitwise the full call's first m rows",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
                row_sweep=True,
                large=True,
            ),
        ),
        Cell(
            why="tep8 1M workspace row sample: the row sweep's pre-merge share, m 1..65, "
            "2**k - 1 / 2**k / 2**k + 1, every 4093rd, 29127, 29128 and 32736",
            spec=dict(
                rows=32768,
                kv_len=[524288],
                new=[32768],
                h_q=64,
                h_out=8,
                topk=2048,
                kv_rows=_WS_ROWS,
                row_counts=_TEP8_ROW_SAMPLE,
            ),
        ),
    )

    note = """
    Call sites: mistral_large4 targets' context rows, every layer. Kernel (sm_103):
    FlashMLA sm100 sparse prefill (`sparse_prefill_fwd`).

    Mistral Large 4 NVFP4 tp4 at 1M (PrefillTarget._attn_ctx, every layer, eager, on a step
    whose context KV, cached + new over its context requests, is at most B = 2**19 rows):
    q = the context rows of the [T, 64, 576] fused_q scratch (the rank's 16 heads first,
    the rest zero), kv = buf[:N] of the bf16 [B, 576] workspace `ds_mla_dequant_gather`
    fills (each request's run contiguous, the buffer's later rows stale), indices = the
    indexer's request-local top-2048, or the engine's dense causal index when it skipped
    the indexer, plus the request's run offset (-1 kept), softmax_scale 128**-0.5, out =
    lat_out[:tc] [tc, 16, 512]. h_q 64 runs FlashMLA's sm100 head64 sparse prefill kernel.
    A 1M prompt's 32768-row chunks on this path, N = 32768 k for k = 2..16, are each a
    cell; a chunk's time rises with N. A row's answer is bitwise the same wherever it sits
    in the call (each 4096-row window re-run).
    Memory: every call allocates the kernel's own [rows, h_q, 512] bf16 output and two
    [rows, h_q] fp32 (max_logits, lse), dropped on return: measured as the call's peak
    over its inputs, 2,164,260,864 B (2.02 GiB) at 32768 rows x 64 heads, 541,065,216 B at
    8192. A caller budgeting a 1M step counts it beside the workspace and fused_q.
    An index at or past kv.shape[0] is skipped as -1 is (FlashMLA's contract for its
    indices) and the row it names is not read: bitwise the answer with -1 in its place,
    with NaN rows behind the view. So a kv view shorter than the step's workspace drops
    the selections past it without an error; the test's gate sees that defect (the
    step's KV counted as its new tokens only) at a whole-call relative L2 of 1.46 on the
    N = B cell, against a band of 4e-3.

    tep4 at max_seq_len 1048576 (the `kv_rows` cells): `PrefillTarget._attn_ctx` calls it once
    per layer, eager, on a step whose context KV fits the bounded workspace of 2**19 rows. q is
    the step's context rows of the [T, 64, 576] fused_q scratch (this rank's 16 heads first,
    zeros after), out `lat_out[:tc]` [tc, 16, 512], kv the workspace's first N = cached + new
    rows of the step's context requests (`ds_mla_dequant_gather`), and indices the
    request-local selections plus each request's offset in the workspace, -1 kept: the
    indexer's top-2048, or on a step whose context indexer the engine skipped DSA metadata's
    dense causal index (`_get_dense_topk_indices`). On sm_103 the call runs FlashMLA's
    `sm100::fwd::head64::sparse_attn_fwd_kernel` (h_q 64 selects it in
    `sparse_attn_prefill_interface`, FlashMLA's `api/sparse_fwd.h`; seen in torch.profiler)
    and a copy into out. A row's answer is bitwise the same in any call that holds it
    (windows of 4093 rows re-run alone), and every leading row count m in 1..32768 of a
    32768-row call over N = 2**19 answers bitwise as that call's first m rows, writing no row
    of out past m (the `row_sweep` cell, large, 195 s; pre-merge runs `_ROW_SAMPLE`'s 100
    counts). The kernel is launched one CTA per query row (FlashMLA's sm100 head64 phase1), so
    the row count moves only the grid and q's and out's tensor-map extents: a wave of one row
    per SM, the next row starting a second. Besides the cells' row counts the 1M
    landing (r019-land1) made 32736-row chunks, a generation row behind each, at N = 32736 x j
    to 491040, and dense fresh steps of 1, 2 and 7 rows; a prompt's last chunk is any count.
    Every call allocates the kernel's own [rows, h_q, 512] bf16 output and fp32 max_logits
    and lse [rows, h_q]: 2164260864 B (2.02 GiB) at 32768 x 64, 541065216 B at 8192 x 64; a
    caller budgets it beside the workspace and the scratch.
    The kernel takes s_kv from kv.shape[0], and an index at or past it answers bitwise as -1
    does: the row is skipped and not read (measured at N, at the buffer's last row and at
    2**31 - 1, with NaN rows past N). A kv view cut short of the step's KV (N from the new
    tokens only, say) therefore drops selections with no error and no NaN; the call cannot
    refuse it without a device sync, so `is_valid` does not.

    dep4 at max_seq_len 1048576 (the `kv_rows` cells): attention DP, so a rank runs all 64 heads
    on its own requests. `PrefillTarget._attn_ctx` calls it once per layer, eager, on a step
    whose context KV on the rank fits the bounded workspace of 2**19 rows. q is the step's
    context rows of the rank's [T, 64, 576] fused_q scratch (every head live, no padding), out
    `lat_out[:tc]` [tc, 64, 512], kv the workspace's first N = cached + new rows of the rank's
    context requests (`ds_mla_dequant_gather`, up to 16 of them), and indices the request-local
    selections plus each request's offset in the workspace, -1 kept: the indexer's top-2048, or
    on a step whose context indexer the engine skipped (every context within 2048 keys) DSA
    metadata's dense causal index (`_get_dense_topk_indices`). On sm_103 the call runs FlashMLA's
    `sm100::fwd::head64::sparse_attn_fwd_kernel` (h_q 64 selects it in
    `sparse_attn_prefill_interface`, FlashMLA's `api/sparse_fwd.h`; seen in torch.profiler)
    and a device-to-device copy into out. A row's answer is bitwise the same in any call that
    holds it (windows of 4093 rows re-run alone) and from call to call. Heads are not bitwise
    independent: heads 0..15 of a 64-live-head call differ from the same call with heads 16..63
    zero (the tep4 / tp4 shape) in 11-14% of elements, relative L2 1.3e-3 to 1.5e-3 between
    the two, both equally close to the reference; the 16-live-head cells do not stand in for
    this shape.
    Besides those cells' row counts the 1M landing (r017-land2) made on this path chunks 2..15
    of a 1,038,637-token prompt (32768 rows behind 32768 x (k - 1) cached keys over N = 32768 k,
    to 491520) and every rank's fresh dense steps of 1, 2 and 7 rows (the warm-up's 1-row and
    2-row steps, the mixed warm-up step's one context row, a 7-token prompt), each now a cell.
    A prompt's last chunk at or under 2**19, a short prompt and a chunk cut short by the rank's
    generation rows make any other count in 1..32768: every leading row count m of a 32768-row
    call over N = 2**19 answers bitwise as that call's first m rows and writes no row of out
    past m (the `row_sweep` cell, large, 211 s; pre-merge runs `_DEP4_ROW_SAMPLE`'s 101 counts,
    0.16 s). The kernel is launched one CTA of 384 threads per query row (grid = rows, 123
    registers a thread; kineto trace), so the row count moves only the grid and q's and out's
    extents: a wave is one row per SM, the next row starting a second, and the op alone
    over every m 1..32768 takes 196 s.
    Every call allocates the kernel's own [rows, h_q, 512] bf16 output and fp32 max_logits
    and lse [rows, h_q]: 2164260864 B (2.02 GiB) at 32768 x 64, 541065216 B at 8192 x 64; a
    caller budgets it beside the workspace, the scratch and its own out (another 2 GiB at
    32768 x 64 x 512).
    The kernel takes s_kv from kv.shape[0], and an index at or past it answers bitwise as -1
    does: the row is skipped and not read (measured at N, at the buffer's last row and at
    2**31 - 1, with NaN rows past N). A kv view cut short of the step's KV (N from the new
    tokens only, say) therefore drops selections with no error and no NaN; the call cannot
    refuse it without a device sync, so `is_valid` does not.

    tep8 at max_seq_len 1048576 (the `kv_rows` cells): `PrefillTarget._attn_ctx` calls it
    once per layer, eager, on a step whose context KV fits the bounded workspace of 2**19
    rows. q is the step's context rows of the [T, 64, 576] fused_q scratch (this rank's 8
    heads first, zeros after), out `lat_out[:tc]` [tc, 8, 512], kv the workspace's first N =
    cached + new rows of the step's context requests (`ds_mla_dequant_gather`), and indices
    the request-local selections plus each request's offset in the workspace, -1 kept: the
    indexer's top-2048 (every context row's, after the q-split all-gather), or on a step
    whose context indexer the engine skipped DSA metadata's dense causal index
    (`_get_dense_topk_indices`). On sm_103 the call runs FlashMLA's
    `sm100::fwd::head64::sparse_attn_fwd_kernel` (h_q 64 selects it in
    `sparse_attn_prefill_interface`, FlashMLA's `api/sparse_fwd.h`; seen in torch.profiler)
    and a copy of heads 0..7 into out. A row's answer is bitwise the same in any call that
    holds it (windows of 4093 rows re-run alone), and every leading row count m in 1..32768
    of a 32768-row call over N = 2**19 answers bitwise as that call's first m rows and writes
    no row of out past m (the `row_sweep` cell, large, 190 s; pre-merge runs
    `_TEP8_ROW_SAMPLE`'s 102 counts, 0.15 s). The kernel is launched one CTA of 384 threads
    per query row with all 64 heads in it (grid = rows, 123 registers a thread; kineto
    trace), and takes s_kv only as the index bound and the extent of kv's tensor map
    (`sparse_attn_fwd_kernel`, FlashMLA's `sm100/prefill/sparse/fwd/head64/phase1.cuh`), so
    the row count moves only the grid and q's and out's extents: a wave is 152 rows on 152
    SMs, the 153rd row starting a second.
    Besides those cells' row counts the 1M landing (r018-land1) made on this path chunks
    1..15 of a 1,040,000-token prompt at 32736 rows (a 7-token prompt's generation row
    behind each) behind 32736 x (k - 1) cached keys over N = 32736 k, to 491040, chunk 16 at
    32768 rows over N = 523808, and dense fresh steps of 1, 2 and 7 rows (the engine's
    warm-up steps, the mixed warm-up step's one context row, the 7-token prompt), each now a
    cell. A prompt's last chunk at or under 2**19, a short prompt and a chunk cut short by
    the generation rows behind it make any other count in 1..32768: the sweep.
    Every call allocates the kernel's own [rows, h_q, 512] bf16 output and fp32 max_logits
    and lse [rows, h_q], whatever h_out is: 2164260864 B (2.02 GiB) at 32768 x 64,
    541065216 B at 8192 x 64; a caller budgets it beside the workspace and the scratch.
    The kernel takes s_kv from kv.shape[0], and an index at or past it answers bitwise as -1
    does: the row is skipped and not read (measured at N, at the buffer's last row and at
    2**31 - 1, with NaN rows past N). A kv view cut short of the step's KV (N from the new
    tokens only, say) therefore drops selections with no error and no NaN; the call cannot
    refuse it without a device sync, so `is_valid` does not.

    Hard limits (sm_103)
      q           rows x 64 x 576 x 2 bytes     2**31 bytes inside row 29127   probed at
                  (TMA coordinates)                                            32768 rows:
                                                rows 29119..29134 gated, every row
                                                window-checked, and every m 29128..32768 of
                                                the tep4, dep4 and tep8 row sweeps (16, 64
                                                and 8 live heads); passed
      q           rows x h_q x 576 elements     2**31 at 58255 rows (h_q 64)   not reached
                                                (1.21e9 at 32768)
      kernel out  rows x 64 x 512 x 2 bytes     2**31 bytes at 32768 rows      reached at
                                                32768 rows (its last byte), passed
      indices     rows x topk int32 offset      2**31 at 1048576 rows          not reached
      kv          index x 1152 bytes            2**31 at index 1864136         not reached:
                  index x 576 elements          2**31 at index 3728271         the workspace
                                                path stops at N = 2**19, indices to 524287
                                                read and passed
      indices     through fp32                  2**24 + 1                      not reached
                                                (largest 524287)
    Not certified: h_q 128 at these shapes, kv past 2**19 rows (the target takes the
    decode path there), topk other than 2048, the kernel's topk_length and attn_sink
    arguments (the call passes neither), CUDA-graph capture (the call site is eager),
    selections from the live indexer (the cells' have its output's shape), a row count no
    cell names at an N other than 2**19 (the sweep's; s_kv enters the kernel only as the
    index bound and the kv tensor map's extent), a packed step of several requests at a
    row count other than 32768.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        rows = q.shape[0]
        o = _flash_mla_sparse_fwd(
            q, kv.view(-1, 1, D_QK), indices.view(rows, 1, indices.shape[-1]), softmax_scale, D_NOPE
        )[0]
        out.copy_(o[:, : out.shape[1]])
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """fp32 softmax over the valid rows; a new tensor shaped like `out`."""
        h_out = out.shape[1]
        res = torch.zeros(out.shape, dtype=torch.float32, device=q.device)
        k_all = kv.float()
        with true_fp32_matmul():
            for r in range(q.shape[0]):
                idx = indices[r]
                idx = idx[idx >= 0].long()
                if idx.numel() == 0:
                    continue
                k = k_all[idx]
                s = (q[r, :h_out].float() @ k.T) * softmax_scale
                res[r] = torch.softmax(s, -1) @ k[:, :D_NOPE]
        return res.to(out.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor,
    ) -> None:
        if q.shape[1] not in KERNEL_HEADS or out.shape[1] > q.shape[1]:
            raise ValueError(f"h_q must be one of {KERNEL_HEADS} and cover out's heads")
        if indices.dtype != torch.int32:
            raise ValueError("indices must be int32")
        if kv.dtype != torch.bfloat16 or kv.shape[-1] != D_QK:
            raise ValueError("kv must be bf16 [N, 576]")


flash_mla_sparse_prefill = FlashMlaSparsePrefill()
