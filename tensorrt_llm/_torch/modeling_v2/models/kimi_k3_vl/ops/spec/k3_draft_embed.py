# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's vocab-parallel embedding as one owner-multicast CuTe DSL launch: the
owner rank of each token stores its row into every rank's one-shot all-reduce buffer."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ...kernels.k3r_spec.dspark_draft_ar import ELTS, EMBED_THREADS, run_draft_embed


class K3DraftEmbed(OpWrapper):
    """``out[t]`` (BF16 ``[M, hidden]``, contiguous) = the full-vocab row of ``ids[t]``, zero for
    an id outside ``[0, vocab)``: the all-reduce of the vocab-sharded lookup. This rank holds
    rows ``[rank * slice_width, ...)`` of the table in ``table`` (BF16, contiguous); ``ids`` is
    contiguous int32 / int64 ``[M]``. Returns None.
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Through the one-shot Lamport buffer of ``workspace`` (the drafter embedding all-reduce's
    MNNVL workspace): the owner stores its row, ``-0`` sent as ``+0``, into slot (token, owner) of
    every rank and every rank polls that one slot, so the result is the all-reduce's, bit for
    bit, in one launch and one hop. ``hidden`` must be a multiple of ELTS * EMBED_THREADS. A
    process-wide compile cache holds one kernel per (world, hidden, id dtype, PDL). Every rank
    must make the same call.
    """

    def __call__(
        self,
        workspace,
        ids: torch.Tensor,
        table: torch.Tensor,
        slice_width: int,
        vocab: int,
        out: torch.Tensor,
        use_pdl: bool = True,
    ) -> None:
        run_draft_embed(workspace, ids, table, slice_width, vocab, out, use_pdl=use_pdl)

    def reference(
        self,
        workspace,
        ids: torch.Tensor,
        tables: Sequence[torch.Tensor],
        slice_width: int,
        vocab: int,
        out: torch.Tensor,
        use_pdl: bool = True,
    ) -> torch.Tensor:
        """``out`` after the call (new). A collective: ``tables`` (in place of ``table``) is the
        group's table slices in rank order; ``workspace`` and ``out`` are not read. Every row is
        a copy, ``-0`` becoming ``+0`` as in the sum."""
        full = torch.cat([t[:slice_width] for t in tables])[:vocab]
        ids = ids.to(torch.int64)
        valid = (ids >= 0) & (ids < vocab)
        rows = full[ids.clamp(0, vocab - 1)]
        rows = torch.where(valid[:, None], rows, torch.zeros_like(rows))
        return rows + 0.0

    def is_valid(
        self,
        workspace,
        ids: torch.Tensor,
        table: torch.Tensor,
        slice_width: int,
        vocab: int,
        out: torch.Tensor,
        use_pdl: bool = True,
    ) -> None:
        assert table.dtype == out.dtype == torch.bfloat16 and table.is_contiguous()
        assert out.is_contiguous() and ids.is_contiguous() and ids.dim() == 1
        assert ids.dtype in (torch.int32, torch.int64) and ids.shape[0] == out.shape[0]
        assert out.shape[1] % (ELTS * EMBED_THREADS) == 0, "hidden not a whole number of chunks"
        assert table.shape[0] >= slice_width or vocab <= slice_width * (workspace.rank + 1)


k3_draft_embed = K3DraftEmbed()
