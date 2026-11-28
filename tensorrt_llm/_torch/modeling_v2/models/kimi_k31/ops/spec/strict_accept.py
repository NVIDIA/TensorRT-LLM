# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict acceptance of a uniform-width speculative draft, in one Triton launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.speculative.spec_tail_kernels import strict_accept as _strict_accept


class StrictAccept(OpWrapper):
    """Return `(accepted_tokens [B, K + 1], num_accepted_tokens [B])`, both new int32 tensors.

    `target_tokens` holds the sampled target tokens of the step: one per context request, then K + 1
    per generation request (`num_contexts + num_gens * (K + 1)` entries, any integer dtype);
    `draft_tokens` is the [num_gens, K] draft verified this step (any strides); `batch_size =
    num_contexts + num_gens`. A context row accepts its one token (count 1, column 0 defined only);
    a generation row copies its K + 1 target tokens and counts 1 + the number of leading draft
    tokens equal to them.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 DFlash2 w{draft_len + 1} verify, {num_gens} generation rows",
                spec=dict(num_contexts=0, num_gens=num_gens, draft_len=draft_len),
            )
            for draft_len in (4, 7)
            for num_gens in (1, 8, 64, 200)
        ),
        *(
            Cell(
                why=f"K3.1 mixed step: {num_contexts} context + {num_gens} generation rows, w{draft_len + 1}",
                spec=dict(num_contexts=num_contexts, num_gens=num_gens, draft_len=draft_len),
            )
            for num_contexts, num_gens, draft_len in ((1, 0, 4), (3, 5, 4), (2, 130, 7))
        ),
        Cell(
            why="drafts handed in as a column slice of a wider draft buffer (strided rows)",
            spec=dict(num_contexts=1, num_gens=9, draft_len=4, draft_buffer_width=7),
        ),
        Cell(
            why="sm_107 DEP16 generation: 12 generation rows a rank, w8",
            spec=dict(num_contexts=0, num_gens=12, draft_len=7),
        ),
        Cell(
            why="sm_107 Helix CP 8 generation: 16 generation rows, w6",
            spec=dict(num_contexts=0, num_gens=16, draft_len=5),
        ),
    )
    note = """
    The Kimi K3.1 target's DFlash worker calls it for every step that is not a
    DFlash2 sampling step (greedy and rejection-off verification; the worker's
    `_sample_and_accept_draft_tokens_base`, after `_sample_tokens_for_batch`), with
    K = max_draft_len (4 for DFlash2 w5, 7 for w8) and the target tokens the
    engine sampled for the step; ragged verification takes the engine's ragged
    helpers instead. The worker then applies its forced-acceptance override to
    the counts.

    Pure function of its inputs: no metadata, workspace or state; the outputs
    are allocated per call. One launch with a fixed 128-row block, so one
    compiled variant per draft length serves every batch composition; it
    compiles on first use (warm it outside CUDA-graph capture). Integer
    compares only: exact and deterministic. Rows past `batch_size` are not
    written.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _strict_accept(target_tokens, draft_tokens, num_contexts, batch_size)

    def reference(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plain torch: context rows take their token, generation rows the cumulative-match length.

        Context rows' columns past 0 are left 0 (the op leaves them undefined; compare column 0).
        """
        num_gens = batch_size - num_contexts
        draft_len = draft_tokens.shape[-1]
        device = target_tokens.device
        accepted = torch.zeros((batch_size, draft_len + 1), dtype=torch.int32, device=device)
        num_accepted = torch.ones((batch_size,), dtype=torch.int32, device=device)
        accepted[:num_contexts, 0] = target_tokens[:num_contexts].to(torch.int32)
        if num_gens:
            gen = target_tokens[num_contexts : num_contexts + num_gens * (draft_len + 1)]
            gen = gen.reshape(num_gens, draft_len + 1).to(torch.int32)
            accepted[num_contexts:] = gen
            matches = (draft_tokens.to(torch.int32) == gen[:, :draft_len]).to(torch.int32)
            num_accepted[num_contexts:] += torch.cumprod(matches, dim=1).sum(dim=1).to(torch.int32)
        return accepted, num_accepted

    def is_valid(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ) -> None:
        # The kernel indexes both inputs from the request counts with masks built from those
        # counts alone: a short input is read past its end instead of raising.
        num_gens = batch_size - num_contexts
        width = draft_tokens.shape[-1] + 1
        assert target_tokens.dim() == 1 and target_tokens.is_contiguous(), (
            "target_tokens must be one contiguous [num_contexts + num_gens * (K + 1)] vector"
        )
        assert target_tokens.numel() >= num_contexts + num_gens * width, (
            f"target_tokens holds {target_tokens.numel()} entries, the step needs "
            f"{num_contexts + num_gens * width}; the kernel reads the rest out of bounds"
        )
        assert draft_tokens.dim() == 2 and draft_tokens.shape[0] >= num_gens, (
            f"draft_tokens must be [>= num_gens ({num_gens}), K]; got {tuple(draft_tokens.shape)}"
        )


strict_accept = StrictAccept()
