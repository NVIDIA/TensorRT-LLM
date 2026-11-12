# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unfused DFlash generation-step bookkeeping of a paged single-latent context (Triton,
one program): the preparation a generation profile without the tail fusion takes."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import PagedLatentPrepare, dflash_paged_latent_prepare
from .k3_dflash_gen_prepare import gen_prepare_reference


class K3DflashPagedLatentPrepare(OpWrapper):
    """``PagedLatentPrepare`` of new tensors for the ``len(slots)`` generation requests at batch
    rows ``first_row + g``, plus ``ctx_len`` advanced in place (the dummy slot zeroed) and
    ``seq_lens_cuda`` of those rows set to ``width``: ``k3_dflash_gen_prepare`` with one rank
    and without the embedding ids."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383) shape: 16 requests of DSpark 5 on one rank (the unfused preparation "
            "no shipped profile takes)",
            spec=dict(num_gens=16, draft_len=5, first_row=0, dummies=1),
        ),
        Cell(
            why="the same unfused preparation with 2 context rows ahead of 2 DSpark 7 requests: "
            "the generation rows start at batch row 2",
            spec=dict(num_gens=2, draft_len=7, first_row=2, dummies=0),
        ),
    )

    note = """
    ``K3DFlashWorker.prepare_1st_drafter_inputs`` and ``_build_mla_block_fixup`` in one program;
    every lane reads its old context length before any lane writes one. No shipped K3 generation
    profile reaches it (each runs with the tail fusion, which takes ``k3_dflash_gen_prepare``).
    """

    def __call__(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        first_row: int,
        dummy_slot: int,
        max_ctx: int,
        page_size: int,
        block_size: int,
        width: int,
    ) -> PagedLatentPrepare:
        return dflash_paged_latent_prepare(
            num_accepted_tokens,
            accepted_tokens,
            slots,
            ctx_len,
            block_counts,
            block_tables,
            seq_lens_cuda,
            first_row,
            dummy_slot,
            max_ctx,
            page_size,
            block_size,
            width,
        )

    def reference(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        first_row: int,
        dummy_slot: int,
        max_ctx: int,
        page_size: int,
        block_size: int,
        width: int,
    ) -> tuple[PagedLatentPrepare, torch.Tensor, torch.Tensor]:
        """``(the op's PagedLatentPrepare, ctx_len after, seq_lens_cuda after)``; nothing is
        written."""
        out, ctx_out, seq_out = gen_prepare_reference(
            num_accepted_tokens,
            accepted_tokens,
            slots,
            ctx_len,
            block_counts,
            block_tables,
            seq_lens_cuda,
            first_row,
            dummy_slot,
            max_ctx,
            page_size,
            block_size,
            width,
        )
        prep = PagedLatentPrepare(**{name: out[name] for name in PagedLatentPrepare._fields})
        if slots.shape[0] == 0:
            # Nothing is launched without a generation request: the dummy slot keeps its length.
            ctx_out = ctx_len.clone()
        return prep, ctx_out, seq_out

    def is_valid(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        first_row: int,
        dummy_slot: int,
        max_ctx: int,
        page_size: int,
        block_size: int,
        width: int,
    ) -> None:
        assert ctx_len.dtype == torch.int64, "ctx_len must be int64"
        live = slots[slots != dummy_slot]
        assert live.unique().numel() == live.numel(), "generation requests share a slot"


k3_dflash_paged_latent_prepare = K3DflashPagedLatentPrepare()
