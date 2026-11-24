# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The flat chain's step begin in one launch: the target's embedding rows and the KDA checkpoint
indices (``kernels/embedding/rubin_ull_flat_begin.py``'s ``begin_fold``, one Triton kernel)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.embedding import rubin_ull_flat_begin as _flat_begin


def _cell(why: str, batch: int, width: int, units: tuple, ids_dtype: str = "int32") -> Cell:
    return Cell(why=why, spec=dict(batch=batch, width=width, units=units, ids_dtype=ids_dtype))


_DENSE = (3 * 8 + 2, 3, 1)
_TRUNCATING = ((1 << 28) + 5, (1 << 26) + 1, 7)


class BeginFold(OpWrapper):
    """``(slots, accepted, rows)`` = ``replay.checkpoint_indices(state_indices)`` and the embedding
    rows of ``ids`` from ``table``, from one launch:

    * ``slots`` [B, 1 + K] int32: request ``b``'s checkpoint slot of token ``j``,
      ``header + state_indices[b] * slot_units + j * state_units`` truncated to int32;
    * ``accepted`` [B] int32: ``prev_num_accepted_tokens[state_indices[b]] + 1`` (any stride);
    * ``rows`` [len(ids), hidden]: ``table[ids[r]]`` where ``0 <= ids[r] < vocab``, zeros elsewhere
      (the row copy ``layers.embedding`` takes for a contiguous table).

    ``replay`` is the cache manager's ``KDAReplayState`` in checkpoint mode; ``state_indices`` [B] of
    any integer dtype and stride; ``ids`` 1-D of ``>= B`` rows (int32 or int64); ``table`` a contiguous
    [vocab, hidden] table. Allocates and returns the three tensors.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1 q8 verify (8 rows), dense checkpoint units", 1, 8, _DENSE),
        _cell("served b4 q8 verify (32 rows), padding rows on the dummy slot", 4, 8, _DENSE),
        _cell("b2 q8 with int64 ids", 2, 8, _DENSE, "int64"),
        _cell("plain q1 decode, 4 rows", 4, 1, _DENSE),
        _cell("slot strides whose int64 products truncate in the int32 cast", 4, 8, _TRUNCATING),
    )

    note = """
    The head of the ULL generation target's decode graph (Helix CP16 GEN, DFlash2 q8 or plain q1):
    the embedding gather and the ten integer ops of KDAReplayState.checkpoint_indices folded into
    one launch. Row copies and integer index arithmetic only, so every output is bitwise the
    unfolded ops'. Out-of-table ids (graph padding rows) give zero rows. Callers check
    ``kernels.flat_begin.folds_embedding`` first and take the unfolded path where it declines.
    """

    def __call__(
        self, replay, state_indices: torch.Tensor, ids: torch.Tensor, table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _flat_begin.begin_fold(replay, state_indices, ids, table)

    def reference(
        self, replay, state_indices: torch.Tensor, ids: torch.Tensor, table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The unfolded ops in plain torch: int64 index arithmetic, then the int32 cast."""
        idx = state_indices.long()
        tokens = 1 + replay.num_speculative_tokens
        j = torch.arange(tokens, device=idx.device, dtype=torch.int64)
        slots = (
            replay.checkpoint_header_units
            + idx[:, None] * replay.checkpoint_slot_units
            + j[None, :] * replay.checkpoint_state_units
        ).to(torch.int32)
        accepted = (replay.prev_num_accepted_tokens[idx] + 1).to(torch.int32)
        vocab = table.shape[0]
        valid = (ids >= 0) & (ids < vocab)
        rows = torch.zeros((ids.shape[0], table.shape[1]), dtype=table.dtype, device=table.device)
        rows[valid] = table[ids[valid].long()]
        return slots, accepted, rows

    def is_valid(
        self, replay, state_indices: torch.Tensor, ids: torch.Tensor, table: torch.Tensor
    ) -> None:
        """Raise where the kernel would refuse or read past its inputs."""
        if not _flat_begin.folds_embedding(ids, table):
            raise ValueError("begin_fold: ids must be 1-D CUDA and the table contiguous, 16-B rows")
        if not replay.checkpoints or not replay.checkpoint_slot_units:
            raise ValueError("begin_fold: the KDA replay state is not in checkpoint mode")
        if state_indices.dim() != 1 or state_indices.shape[0] > ids.shape[0]:
            raise ValueError("begin_fold: 1-D state indices, at most one request per row")


begin_fold = BeginFold()
