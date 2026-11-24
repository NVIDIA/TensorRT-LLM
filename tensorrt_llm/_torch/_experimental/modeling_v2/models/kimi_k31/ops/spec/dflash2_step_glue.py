# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter step's glue around the block decode as single launches
(``kernels/spec/dflash2_step_glue.py``).

* ``k31_dflash2_noise_block``: ``dflash_noise_block_embedding`` (the engine's input block of the
  drafter) for a drafter that embeds with a plain table: one copy kernel in place of the id fill,
  the id concatenation, the embedding gather, the mask-row broadcast and the bonus-row copy.
* ``k31_draft_agreed_tokens``: the end of the helix16 worker's draft-token agreement, the agreed
  drafts and the next step's input tokens from the all-reduced FP32 row, in place of a conversion
  and a concatenation.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_step_glue import agreed_tokens, noise_block

#: K3.1's vocabulary, hidden width and the DFlash2 block (one bonus slot and seven mask slots).
VOCAB, HIDDEN, BLOCK = 163840, 7168, 8


class K31DFlash2NoiseBlock(OpWrapper):
    """``[n, block, hidden]``: request ``i``'s slot 0 is ``table[bonus[i]]`` and every other slot
    ``table[mask_token_id]`` (``dflash_noise_block_embedding(F.embedding over table, bonus,
    mask_token_id, block)``: no trained mask row).

    Args:
        table: ``[vocab, hidden]`` embedding table, unit column stride.
        bonus: ``[n]`` integer token ids (each request's last accepted token).
        mask_token_id: the mask token.
        block: slots per request.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="served b1: one request, block 8", spec=dict(n=1, block=BLOCK)),
        Cell(why="two requests, block 8", spec=dict(n=2, block=BLOCK)),
        Cell(why="16 requests, the vocabulary's first and last ids", spec=dict(n=16, block=BLOCK)),
        Cell(
            why="a bonus id equal to the mask id", spec=dict(n=3, block=BLOCK, bonus_is_mask=True)
        ),
        Cell(why="block 1: the bonus rows only", spec=dict(n=4, block=1)),
        *(
            Cell(why=f"sm_107 helix8 generation: {n} requests, block {b}", spec=dict(n=n, block=b))
            for n, b in ((2, 6), (4, 6), (9, 6), (11, 6), (13, 6), (16, 6), (9, 2), (9, 4), (9, 9))
        ),
    )

    note = """
    The helix16 target's DFlash2 worker calls it in the generation prepare under the context
    overlap (``DFlashWorker._prepare_paged_latent_gen_inputs``), and the helix8 target's under its
    draft_noise_block constant, with the target's embedding table
    (``DraftCore.w["embed"]``, the full vocabulary on every rank, read by ``F.embedding``) and the
    prepare kernel's bonus ids. Grid (n * block, hidden / 1024), 16-byte vectors. Row copies: the
    reference's bits.
    """

    def __call__(
        self, table: torch.Tensor, bonus: torch.Tensor, mask_token_id: int, block: int
    ) -> torch.Tensor:
        return noise_block(table, bonus, table[mask_token_id], block)

    def reference(
        self, table: torch.Tensor, bonus: torch.Tensor, mask_token_id: int, block: int
    ) -> torch.Tensor:
        """The engine's ``dflash_noise_block_embedding`` over ``F.embedding(·, table)``, restated
        (the helper is the DFlash worker's; the test checks this against it)."""
        mask = torch.full((1,), int(mask_token_id), dtype=torch.long, device=bonus.device)
        rows = torch.nn.functional.embedding(torch.cat([bonus, mask], dim=0), table)
        out = rows[bonus.shape[0]].expand(bonus.shape[0], block, -1).clone()
        out[:, 0, :] = rows[: bonus.shape[0]]
        return out

    def is_valid(
        self, table: torch.Tensor, bonus: torch.Tensor, mask_token_id: int, block: int
    ) -> None:
        assert table.dim() == 2 and table.stride(-1) == 1, "table rows with unit column stride"
        assert bonus.dim() == 1 and not bonus.is_floating_point(), "1-D integer bonus ids"
        assert 0 <= mask_token_id < table.shape[0], "the mask id indexes the table"
        assert block >= 1, "at least the bonus slot"
        # Ids are read unguarded (F.embedding raises on an id outside the table).
        if bonus.numel():
            assert int(bonus.min()) >= 0 and int(bonus.max()) < table.shape[0], "ids in the table"


class K31DraftAgreedTokens(OpWrapper):
    """``(drafts, next_new)`` from the agreement's all-reduced FP32 rows ``reduced [rows, width]``:
    ``drafts = reduced[:, :draft_len].to(draft_dtype)`` and ``next_new = cat([new_tokens[:, :1],
    drafts.to(new_tokens.dtype)], 1)`` (``DFlashWorker._agree_draft_tokens`` after the
    all-reduce), both new tensors.

    Args:
        reduced: FP32 ``[rows, width >= draft_len]`` with unit column stride; the draft columns
            hold integers (token ids below 2**24, exact in FP32).
        draft_len: K.
        draft_dtype: the drafts' integer dtype.
        new_tokens: ``[rows, >= 1]`` integer, unit column stride: column 0 is kept.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="served b1: one request, K 7, the 2048-wide row", spec=dict(rows=1, width=2048)),
        Cell(why="two requests", spec=dict(rows=2, width=2048)),
        Cell(why="32 requests, the vocabulary's last id", spec=dict(rows=32, width=2048)),
        Cell(
            why="int64 next tokens, int32 drafts",
            spec=dict(rows=3, width=2048, new_dtype=torch.int64),
        ),
    )

    note = """
    The helix16 target's DFlash2 worker calls it after the agreement's MNNVL all-reduce
    (``ZeroPaddedReduce.reduce``) with ``new_tokens`` the draft forward's next-step tokens. One
    program (rows <= the agreement buffer's rows). FP32 -> integer conversion of exact integers
    and copies: the reference's values.
    """

    def __call__(
        self,
        reduced: torch.Tensor,
        draft_len: int,
        draft_dtype: torch.dtype,
        new_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return agreed_tokens(reduced, draft_len, draft_dtype, new_tokens)

    def reference(
        self,
        reduced: torch.Tensor,
        draft_len: int,
        draft_dtype: torch.dtype,
        new_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        agreed = reduced[:, :draft_len].to(draft_dtype)
        return agreed, torch.cat([new_tokens[:, :1], agreed.to(new_tokens.dtype)], dim=1)

    def is_valid(
        self,
        reduced: torch.Tensor,
        draft_len: int,
        draft_dtype: torch.dtype,
        new_tokens: torch.Tensor,
    ) -> None:
        assert reduced.dtype == torch.float32 and reduced.stride(-1) == 1, "FP32 rows"
        assert reduced.shape[1] >= draft_len >= 1, "the row holds the drafts"
        assert new_tokens.shape[0] == reduced.shape[0], "one next-token row per agreement row"
        assert new_tokens.stride(-1) == 1, "new_tokens with unit column stride"
        assert not draft_dtype.is_floating_point, "integer drafts"
        drafts = reduced[:, :draft_len]
        assert bool((drafts == drafts.round()).all()), "the draft columns hold integers"
        assert bool((drafts.abs() < 2**24).all()), "ids below 2**24 (exact in FP32)"


k31_dflash2_noise_block = K31DFlash2NoiseBlock()
k31_draft_agreed_tokens = K31DraftAgreedTokens()

__all__ = [
    "K31DFlash2NoiseBlock",
    "K31DraftAgreedTokens",
    "k31_dflash2_noise_block",
    "k31_draft_agreed_tokens",
]
