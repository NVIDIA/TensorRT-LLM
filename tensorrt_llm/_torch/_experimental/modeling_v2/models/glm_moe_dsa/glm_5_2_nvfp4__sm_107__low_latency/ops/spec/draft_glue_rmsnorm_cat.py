# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's MTP eh_proj input window -- embedding gather, enorm, hnorm, concat, column split -- in one kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._draft_glue_rmsnorm import draft_glue_rmsnorm_cat as _draft_glue_rmsnorm_cat

#: GLM-5.2's hidden size and the eh_proj input width (the concat of two hidden rows).
_HIDDEN = 6144
#: Columns of the eh_proj input each tp16 rank holds (row-parallel eh_proj).
_TP16_WINDOW = 2 * _HIDDEN // 16


def _tp16_window(rank: int) -> tuple[int, int]:
    return rank * _TP16_WINDOW, (rank + 1) * _TP16_WINDOW


class GlmDraftGlueRmsnormCat(OpWrapper):
    """Columns `[col_lo, col_hi)` of `cat([rmsnorm(embed_table[input_ids]), rmsnorm(hidden_states)], -1)`.

    The embedding rows are normalized with `enorm_weight`, the hidden rows with `hnorm_weight`, both with `eps`:
    `bf16((x * rsqrt(mean(x^2) + eps)) * w)` in FP32. An id outside `[0, embed_table.shape[0])` gives a zero
    embedding row, as the Embedding module does. Only the halves the window overlaps are computed and only the
    window is stored. Returns a contiguous BF16 `[M, col_hi - col_lo]` tensor.

    With `enable_pdl` the kernel waits for its producer before any load and lets its dependent (eh_proj) launch
    right after that wait.
    """

    ARCHS = frozenset({Arch.SM_107})

    # Draft step 0 at c1: the 8 verified rows of the one request (accepted-token ids, the target's last hidden
    # rows). eh_proj is row-parallel over the 12288-column concat, so rank r feeds it columns
    # [768 r, 768 (r + 1)): ranks 0-7 normalize only the embedding half, ranks 8-15 only the hidden half. The
    # spec's `vocab` is the table's row count; the gather does not depend on it.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 step 0, tp16 rank 0: the first embedding-half window",
            spec=dict(rows=8, vocab=4096, window=_tp16_window(0), ids="valid"),
        ),
        Cell(
            why="c1 step 0, tp16 rank 7: the last embedding-half window",
            spec=dict(rows=8, vocab=4096, window=_tp16_window(7), ids="valid"),
        ),
        Cell(
            why="c1 step 0, tp16 rank 8: the first hidden-half window",
            spec=dict(rows=8, vocab=4096, window=_tp16_window(8), ids="valid"),
        ),
        Cell(
            why="c1 step 0, tp16 rank 15: the last hidden-half window",
            spec=dict(rows=8, vocab=4096, window=_tp16_window(15), ids="valid"),
        ),
        Cell(
            why="ids outside the table (-1 padding, past the end) give zero rows",
            spec=dict(rows=8, vocab=4096, window=_tp16_window(3), ids="outside"),
        ),
        Cell(
            why="one row: a draft step that feeds the glue a single token",
            spec=dict(rows=1, vocab=4096, window=_tp16_window(5), ids="valid"),
        ),
    )

    note = """
    The kernel covers hidden sizes whose FlashInfer RMSNormKernel runs one CTA per row (6144 here); a window over
    both halves (tp below 2, or attention DP) is supported but no target of this family reaches it. Each row is
    normalized with FlashInfer's CuTe RMSNorm thread geometry, reduction and arithmetic statements; the cells
    certify it against the FP32 reference.
    """

    def __call__(
        self,
        embed_table: torch.Tensor,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        eps: float,
        col_lo: int,
        col_hi: int,
        enable_pdl: bool,
    ) -> torch.Tensor:
        return _draft_glue_rmsnorm_cat(
            embed_table,
            input_ids,
            hidden_states,
            enorm_weight,
            hnorm_weight,
            eps,
            col_lo,
            col_hi,
            enable_pdl,
        )

    def reference(
        self,
        embed_table: torch.Tensor,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        eps: float,
        col_lo: int,
        col_hi: int,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """The unfused chain in FP32: gather with the zero-row guard, both RMSNorms, concat, column slice."""
        vocab = embed_table.shape[0]
        ids = input_ids.long()
        valid = (ids >= 0) & (ids < vocab)
        embeds = embed_table[ids.clamp(0, vocab - 1)].float() * valid.unsqueeze(1)

        def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            return ((x * rstd) * weight.float()).to(torch.bfloat16)

        concat = torch.cat(
            [rmsnorm(embeds, enorm_weight), rmsnorm(hidden_states.float(), hnorm_weight)], dim=-1
        )
        return concat[:, col_lo:col_hi].contiguous()

    def is_valid(
        self,
        embed_table: torch.Tensor,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        eps: float,
        col_lo: int,
        col_hi: int,
        enable_pdl: bool,
    ) -> None:
        hidden = hidden_states.shape[1]
        # The predicate stores whole 8-column vectors by their first column: a window edge inside a vector moves
        # columns into or out of the window.
        assert 0 <= col_lo < col_hi <= 2 * hidden and col_lo % 8 == 0 and col_hi % 8 == 0, (
            "the window must be 8-column aligned inside [0, 2 * hidden)"
        )
        # The kernel is compiled for 16-byte aligned rows whose stride is a multiple of 8 elements.
        rows = [("hidden_states", hidden_states)]
        if col_lo < hidden:
            rows.append(("embed_table", embed_table))
            assert input_ids.dim() == 1 and input_ids.is_contiguous(), (
                "input_ids must be a contiguous vector"
            )
            assert input_ids.shape[0] == hidden_states.shape[0], "one id per hidden row"
        for name, t in rows:
            assert t.dim() == 2 and t.shape[1] == hidden and t.stride(1) == 1, (
                f"{name} must be [rows, hidden] with unit column stride"
            )
            assert t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0, (
                f"{name} rows must be 16-byte aligned"
            )
        for name, w in (("enorm_weight", enorm_weight), ("hnorm_weight", hnorm_weight)):
            assert w.shape == (hidden,) and w.is_contiguous() and w.data_ptr() % 16 == 0, (
                f"{name} must be a contiguous, 16-byte aligned [hidden] vector"
            )


glm_draft_glue_rmsnorm_cat = GlmDraftGlueRmsnormCat()
