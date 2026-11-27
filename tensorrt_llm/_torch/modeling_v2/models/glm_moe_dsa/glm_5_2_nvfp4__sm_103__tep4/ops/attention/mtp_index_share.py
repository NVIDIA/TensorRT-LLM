# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP draft-loop index share: the engine's DSA indexer semantics (`Indexer.sparse_attn_indexer`
with `mtp_index_share`), for the MTP layer's indexer, the only one a draft step runs.

The engine's MTP-Eagle linear draft loop marks the metadata `in_mtp_draft_loop`, hands it the
trunk's accepted counts (`mtp_num_accepted`) and sets `indexer_skip_topk` from the second draft
step on (`eagle3._forward_linear_draft_loop`); it does so for every DSA metadata, and each
indexer's own `mtp_index_share` decides whether to act. With it on, draft step 0 stashes one
selection row per request in `shared_topk_indices` (`mtp_share_store`: a context request's last
row, a generation request's last accepted row), and the later steps copy those rows instead of
scoring and selecting (`mtp_share_load`). After step 0 the loop rewrites the metadata in place
to one row per request in request order (`num_contexts = 0`), so stash row `r` is request `r`'s:
contexts first, then generations. The keys still project and append on every step: the
engine's `pre_indexer_proj` and `_update_k_cache` run unconditionally.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count


def mtp_share_reuses(md) -> bool:
    """The engine's `reuse_topk` without the flag (the caller's): a draft step after the first,
    with a stash to read."""
    return bool(
        getattr(md, "in_mtp_draft_loop", False)
        and getattr(md, "indexer_skip_topk", False)
        and getattr(md, "shared_topk_indices", None) is not None
    )


class MtpShareLoad(OpWrapper):
    """A reusing draft step's selection (counted `mtp_ix_share`): stash rows `[0, seqs)` into
    its `seqs` one-row requests' rows of `dst`, as the engine copies
    `shared_topk_indices[:num_generations]`."""

    def __call__(self, md, dst: torch.Tensor, seqs: int, count: Count) -> None:
        count("mtp_ix_share", True)
        dst.copy_(md.shared_topk_indices[:seqs])

    def reference(self, md, dst: torch.Tensor, seqs: int, count: Count) -> torch.Tensor:
        """The stash's first `seqs` rows (`count` ignored)."""
        return md.shared_topk_indices[:seqs].clone()


class MtpShareStore(OpWrapper):
    """Draft step 0's stash, as the engine writes it after the indexer: each context request's
    last row of `topk_buf[:tc]`, then each generation request's last accepted row of
    `topk_buf[tc:]` (row `gen_p - 1` without accepted counts), from stash row 0. A dense
    (skipped) phase's rows are stashed too, as the engine does."""

    def __call__(
        self, md, topk_buf: torch.Tensor, ctx: int, tc: int, seqs: int, gen_p: int
    ) -> None:
        rows = _stash_rows(md, topk_buf, ctx, tc, seqs, gen_p)
        if rows is None:
            return
        shared = md.shared_topk_indices
        if shared is None:
            md.shared_topk_indices = rows.contiguous()
        else:
            shared[: rows.shape[0], : rows.shape[1]].copy_(rows)

    def reference(
        self, md, topk_buf: torch.Tensor, ctx: int, tc: int, seqs: int, gen_p: int
    ) -> torch.Tensor | None:
        """The rows the stash receives."""
        return _stash_rows(md, topk_buf, ctx, tc, seqs, gen_p)


def _stash_rows(md, topk_buf: torch.Tensor, ctx: int, tc: int, seqs: int, gen_p: int):
    rows = None
    if seqs:
        gen_topk = topk_buf[tc : tc + seqs * gen_p]
        num_accepted = getattr(md, "mtp_num_accepted", None)
        if num_accepted is None:
            rows = gen_topk[gen_p - 1 :: gen_p]
        else:
            base = torch.arange(seqs, device=gen_topk.device, dtype=torch.long) * gen_p
            offset = (num_accepted[ctx : ctx + seqs] - 1).clamp(0, gen_p - 1)
            rows = gen_topk[base + offset]
    if ctx:
        ctx_last = torch.cumsum(md.seq_lens_cuda[:ctx].to(torch.long), dim=0) - 1
        ctx_rows = topk_buf[ctx_last]
        rows = ctx_rows if rows is None else torch.cat([ctx_rows, rows])
    return rows


mtp_share_load = MtpShareLoad()
mtp_share_store = MtpShareStore()
