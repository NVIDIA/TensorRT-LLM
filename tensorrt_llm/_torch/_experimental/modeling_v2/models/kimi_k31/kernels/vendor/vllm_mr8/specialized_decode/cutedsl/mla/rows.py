# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Verify rows: a spec verify step of B requests x q tokens as B * q single-query
rows, so the q = 1 MLA machinery (mla.query POLLED_X, mla.dcp LSE_FOLD / POLLED_TS /
WIDE_MERGE / POLLED_OUTPUT) runs it. Verify token t of request i (global length T_i,
its q newest tokens being verified) attends causally to the first T_i - q + 1 + t
tokens. Under DCP the local tokens are stored in global order, so that causal prefix is
a prefix of the rank's local tokens: row (i, t) is a q = 1 query of request i's block
table with the prefix's local length, the same attention as the masked verify query.

``VerifyRows`` writes, once per step, every row's block table (its request's row, up to
the pages its prefix can reach), local length and global length (0 for rows beyond the
step's requests: empty). ``causal`` False: every row of a request attends to the whole
request (the DSpark drafter's non-causal draft block)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, fake
from ..common.policy import batch_table
from ..common.primitives import launch

# Rubin-class devices under DCP, spec verify: token-row batches (B * q) whose MLA layers
# run as single-query rows on the q = 1 path. B1 x q 8, 4 ranks (24 gathered heads),
# in-proj -> DcpValue chain, us/layer at local 550 / 4096: verify path 22.0 / 22.6,
# rows 18.0 / 21.1.
VERIFY_ROWS = batch_table("8:1")
MIN_PAGE = 16  # the smallest KV page (tokens) the rows' table bound assumes
TILE_PAGES = 8  # entries past the prefix's last page that a 128-token k-tile may read


class VerifyRows:
    """One CTA per row: (table, totals, out_table, out_local, out_total, requests);
    ``table``, ``totals`` and ``out_table`` are ``dynamic_view``s."""

    def __init__(self, q, rows, world, rank, interleave, causal=True):
        self.q, self.rows = q, rows
        self.world, self.rank, self.interleave = world, rank, interleave
        self.causal = causal

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.rows, 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        table, totals, out_table, out_local, out_total, requests = args
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        q, world, interleave = self.q, self.world, self.interleave
        request = row // q
        live = request < requests
        total = Int32(0)
        if live:
            total = totals[request]
        prefix = Int32(0)
        if cutlass.const_expr(self.causal):
            if total > 0:
                prefix = max(total - (q - 1) + row % q, 0)
        else:
            prefix = total
        # vllm.v1.attention.backends.utils.get_dcp_local_seq_lens of the prefix
        base = prefix // interleave // world * interleave
        rest = prefix - base * world - self.rank * interleave
        local = base + min(max(rest, 0), interleave)
        if tid == 0:
            out_local[row] = local
            out_total[row] = prefix
        # Only the table entries the row's local prefix can reach (pages of >= 16
        # tokens, plus the partial last 128-token tile's), not the whole row: the
        # serving tables are max_model_len wide. The rest keeps earlier valid pages
        # (the buffer starts zeroed).
        width = out_table.shape[1]
        need = min(width, (local + MIN_PAGE - 1) // MIN_PAGE + TILE_PAGES)
        for i in cutlass.range((need + 127) // 128):
            j = tid + 128 * i
            if j < need:
                page = Int32(0)
                if live:
                    page = table[request, j]
                out_table[row, j] = page


@cache
def compile_rows(q, rows, world, rank, interleave, causal=True):
    i32 = torch.int32
    # 2-D tables and totals as dynamic layouts (``dynamic_view`` at the call; the
    # accesses are scalar, so the stride divisibility hint of ``dynamic_fake`` does
    # not constrain the row pitch), the per-row outputs static.
    table, out_table = dynamic_fake(2, i32), dynamic_fake(2, i32)
    totals = dynamic_fake(1, i32)
    out = fake((rows,), None, i32, 4)
    kernel = VerifyRows(q, rows, world, rank, interleave, causal)
    args = (table, totals, out_table, out, out, Int32(0))
    return cute.compile(kernel, *args, make_fake_stream())
