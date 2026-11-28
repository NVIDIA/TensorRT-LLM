# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the rows of a CuteDSL MoE output that the FC2 finalize scatter-adds into, sized to the
routing: one Triton launch over the rows' routing, ahead of ``rubin_cutedsl_fc2_finalize``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

from ...kernels.moe.rubin_moe_zero_rows import zero_local_rows

#: The routed experts' latent width (the output's row), experts per token, and experts.
_HIDDEN, _TOP_K, _EXPERTS = 3584, 16, 896
#: The INFO line of the first call in a process (``logger.info_once`` under ``LOG_KEY``): a
#: serving run's log carries it once per rank whose MoE zeroes only the received rows.
LOG_LINE = "Kimi K3.1 Rubin MoE zero: received rows only (rubin_moe_zero_local_rows)"
LOG_KEY = "kimi_k31_rubin_moe_zero_local_rows"


def _cell(counts: tuple, slots: int, local: int, offset: int) -> dict:
    """``counts``: the tokens each source sends this rank, packed at the front of its ``slots``."""
    return dict(
        counts=counts,
        slots=slots,
        hidden=_HIDDEN,
        top_k=_TOP_K,
        num_experts=_EXPERTS,
        local_num_experts=local,
        local_expert_offset=offset,
    )


class RubinMoeZeroLocalRows(OpWrapper):
    """Zero ``input`` (``[T, hidden]``, dense) where the finalize will add into it: each row whose
    ``token_selected_experts`` row (int32 ``[T, top_k]``, dense) names an expert of
    ``[local_expert_offset, local_expert_offset + local_num_experts)``, the experts
    ``rubin_moe_sort`` gives that window. Every other row keeps its contents. In place; returns
    nothing.

    On a DEP16 context server ``input`` is the all-to-all's combine payload, ``[ep_size *
    runtime_max, latent]``. The dispatch packs each source rank's tokens at the front of the
    source's slot block, each token with at least one expert of this rank's window, and the slots
    past them route nowhere (every id -1). So the zeroed rows are the received rows: the rows the
    finalize adds into and the only rows the combine reads. ``rubin_moe_output_memset`` at EP no
    larger than top-k zeroes every slot instead.

    After its launch the first call in a process logs ``LOG_LINE`` at INFO, once: a DEP16 context
    server's log then shows, per rank, that its MoE zeroes only the received rows.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 context: 16 sources of 1024 tokens, the received rows packed in 1024 slots",
            spec=_cell((1024,) * 16, 1024, 56, 728),
        ),
        Cell(
            why="uneven sources, two of them empty: slots past each count route nowhere",
            spec=_cell(
                (300, 0, 1024, 17, 512, 1, 0, 999, 64, 1023, 5, 700, 256, 33, 2, 128), 1024, 56, 0
            ),
        ),
        Cell(
            why="a decode step of the context server: one slot per source, the last window",
            spec=_cell((1,) * 16, 1, 56, 840),
        ),
        Cell(
            why="every slot received: the most rows this zeroes",
            spec=_cell((512,) * 16, 512, 56, 56),
        ),
    )

    note = """
    The rows it skips keep their old contents: the finalize never adds into them and the
    all-to-all combine (prepare and pull) reads only the slots below each source's received count,
    which are the zeroed rows. The zero is +0 in every byte, as the memset's. Run it on the memset
    stream beside FC1 and join before FC2, as the DEP16 context does (``record_stream`` on eager
    steps).
    """

    def __call__(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        local_expert_offset: int,
        local_num_experts: int,
    ) -> None:
        zero_local_rows(input, token_selected_experts, local_expert_offset, local_num_experts)
        logger.info_once(LOG_LINE, key=LOG_KEY)

    def reference(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        local_expert_offset: int,
        local_num_experts: int,
    ) -> torch.Tensor:
        """A copy of ``input`` with every row that has an expert of the window zero."""
        out = input.clone()
        ids = token_selected_experts
        hi = local_expert_offset + local_num_experts
        live = ((ids >= local_expert_offset) & (ids < hi)).any(dim=1)
        out[live] = 0
        return out

    def is_valid(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        local_expert_offset: int,
        local_num_experts: int,
    ) -> None:
        # The launcher refuses a layout it would read wrongly. An empty window zeroes nothing,
        # and a window reaching below 0 takes the slots that route nowhere (id -1) for received
        # rows; the finalize would then add into stale rows, or the zero would grow past them.
        assert local_num_experts > 0, "an empty window zeroes no row the finalize adds into"
        assert local_expert_offset >= 0, "a window below 0 holds the id of a slot routing nowhere"


rubin_moe_zero_local_rows = RubinMoeZeroLocalRows()
