# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MQA of context rows over a bf16 latent workspace: FlashMLA's sm100f sparse prefill."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm.flash_mla import flash_mla_sparse_fwd as _flash_mla_sparse_fwd

D_NOPE = 512
D_QK = 576
KERNEL_HEADS = (64, 128)
#: The tep4, dep4 and tep8 deployments' bounded context workspace at max_seq_len 1048576:
#: B = 2**19 bf16 rows.
_WS_ROWS = 2**19


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
    # `kv_rows`, a "tep4 1M", "dep4 1M" or "tep8 1M" cell is the tep4, dep4 or tep8 deployment's
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
            why="tep8: 32768 rows of one request, 8 heads padded to 64",
            spec=dict(rows=32768, kv_len=[32768], h_q=64, h_out=8, topk=2048),
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
            why="1M workspace index mode at N = B = 2**19: 32768-row chunk behind cached prefix",
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
            why="1M workspace packed chunk step with 4 cached requests",
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
            why="1M workspace dense mode (skipped indexer): three requests with cached prefixes",
            spec=dict(
                rows=1077,
                kv_len=[2048, 1777, 700],
                new=[300, 77, 700],
                h_q=64,
                h_out=16,
                topk=2048,
                kv_rows=524288,
                dense=True,
            ),
        ),
    )

    note = """
    FlashMLA sm100f sparse prefill over a contiguous bf16 [N, 576] latent workspace.
    Indices >= kv.shape[0] or -1 are skipped; real heads occupy [0, h_out) of the 64-head query.
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
