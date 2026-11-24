# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The next step's input tokens with the drafter's ``kv_lens`` rewind in the same launch:
``kernels/spec/spec_tail_kernels.py``'s ``fused_next_new_tokens_rewind`` (one Triton program)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec import rubin_ull_spec_tail_kernels as _tail


def _cell(why: str, **spec) -> Cell:
    base = dict(contexts=0, gens=1, draft_len=7, dtypes=("int32", "int32"))
    base.update(spec)
    return Cell(why=why, spec=base)


class DraftNextNewTokensRewind(OpWrapper):
    """``out[i] = [accepted_tokens[i, num_accepted_tokens[i] - 1], *next_draft_tokens[i]]`` for
    every row ``i`` of ``next_draft_tokens`` ``[batch, K]`` (``SpecWorker._prepare_next_new_tokens``
    with ``batch_indices = arange``), in the promoted dtype of the two token tensors, and in place
    ``kv_lens[num_contexts + g] = max(kv_lens[num_contexts + g] - rewind[g], 0)`` for each of the
    ``rewind.shape[0]`` generation requests (``DFlashWorker._apply_kv_rewind_after_draft``).
    Returns ``out`` ``[batch, K + 1]`` (allocated by the call).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1: one generation request, K 7"),
        _cell("served b4: 4 generation requests, accepted counts 1..8", gens=4),
        _cell("a context request first: its row is not rewound", contexts=1, gens=2),
        _cell(
            "int64 accepted tokens and int32 drafts: an int64 output",
            gens=3,
            dtypes=("int64", "int32"),
        ),
        _cell("16 requests: the largest fused select-accept batch", gens=16),
        _cell("rewinds past zero clamp at 0", gens=2, clamp=True),
    )

    note = """
    The helix16 target's DFlash2 worker calls it after the draft forward of a step (not a graph
    warmup) whose rewind is pending (``DFlashWorker._fused_kv_rewind_operands``), and so does the
    helix8 target's under its draft_tail_rewind constant; the rewind
    amounts are the ones the select-accept launch (or ``_prepare_kv_for_draft_forward``) wrote.
    One program, ``BLOCK_B`` the next power of two of the
    larger of the batch and the generation count. Integer copies and arithmetic: exact.
    """

    def __call__(
        self,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kv_lens: torch.Tensor,
        rewind: torch.Tensor,
        num_contexts: int,
    ) -> torch.Tensor:
        return _tail.fused_next_new_tokens_rewind(
            accepted_tokens, next_draft_tokens, num_accepted_tokens, kv_lens, rewind, num_contexts
        )

    def reference(
        self,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kv_lens: torch.Tensor,
        rewind: torch.Tensor,
        num_contexts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(out, kv_lens after)``; ``kv_lens`` itself is not modified."""
        batch, draft_len = next_draft_tokens.shape
        dtype = torch.promote_types(accepted_tokens.dtype, next_draft_tokens.dtype)
        rows = torch.arange(batch, device=accepted_tokens.device)
        last = accepted_tokens[rows, num_accepted_tokens[:batch].long() - 1]
        out = torch.cat([last[:, None].to(dtype), next_draft_tokens.to(dtype)], dim=1)
        kv_after = kv_lens.clone()
        g = rewind.shape[0]
        span = kv_after[num_contexts : num_contexts + g]
        span.copy_((span - rewind.to(span.dtype)).clamp(min=0))
        return out, kv_after

    def is_valid(
        self,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kv_lens: torch.Tensor,
        rewind: torch.Tensor,
        num_contexts: int,
    ) -> None:
        """Rows are read and written unguarded: raise where an accepted count leaves its row or
        the rewound rows run past ``kv_lens``."""
        batch = next_draft_tokens.shape[0]
        if accepted_tokens.shape[0] < batch or num_accepted_tokens.shape[0] < batch:
            raise ValueError("draft_next_new_tokens_rewind: fewer accepted rows than the batch")
        if batch:
            n = num_accepted_tokens[:batch]
            if bool((n < 1).any()) or int(n.max()) > accepted_tokens.shape[1]:
                raise ValueError("draft_next_new_tokens_rewind: an accepted count outside its row")
        if kv_lens.shape[0] < num_contexts + rewind.shape[0]:
            raise ValueError("draft_next_new_tokens_rewind: kv_lens shorter than the rewound rows")


draft_next_new_tokens_rewind = DraftNextNewTokensRewind()
