# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode step's begin in one launch (``kernels/embedding/rubin_ull_step_begin.py``): the
engine's speculative-decoding head prep, the Helix DSA decode metadata and the flat chain's begin
(embedding rows and KDA checkpoint indices)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.attention.backends.sparse.dsa.kernels import (
    fused_dsa_helix_decode_metadata,
)
from tensorrt_llm._torch.speculative.spec_head_kernels import fused_spec_head_prep

from ...kernels.embedding import rubin_ull_flat_begin as _flat_begin
from ...kernels.embedding import rubin_ull_step_begin as _step_begin
from .rubin_ull_begin_fold import begin_fold


def _cell(why: str, requests: int, overlap: bool = True, schedule: bool = True) -> Cell:
    return Cell(why=why, spec=dict(requests=requests, overlap=overlap, schedule=schedule))


class StepBegin(OpWrapper):
    """``fused_spec_head_prep(**prep)``, ``fused_dsa_helix_decode_metadata(*meta_args,
    **meta_kwargs)`` and ``begin_fold(replay, state_indices, ids, table)`` from one launch.

    The first two write their outputs in place, as the engine's launches do (the prep: position
    ids, KV lengths and the Helix slots and bounds; the metadata: request indices, index-key slots,
    indptrs, per-token lengths and the DeepGEMM schedules); the begin's ``(slots, accepted, rows)``
    are allocated and returned. ``prep`` and ``meta_*`` are those functions' own arguments, the
    metadata reading what the prep writes (the engine's ``on_update_kv_lens`` after its
    ``preprocess_inputs``).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1 q8 verify (8 rows), overlap scheduler, schedules in the kernel", 1),
        _cell("b2 q8 (16 rows)", 2),
        _cell("b4 q8 (32 rows)", 4),
        _cell("b8 q8 (64 rows)", 8),
        _cell("b1 without the overlap correction", 1, overlap=False),
        _cell("b2 with the schedules left to the caller", 2, schedule=False),
    )

    note = """
    The head of the ULL generation target's decode graph (Helix CP16 GEN, DFlash2 q8): three
    serial launches folded into one. The prep and the metadata run in one CTA in their own order
    (a barrier between them); the begin's programs run beside it and touch none of their
    buffers. Integer arithmetic and row copies only: every output is bitwise the three launches'.
    """

    def __call__(
        self,
        prep: dict,
        meta_args: tuple,
        meta_kwargs: dict,
        replay,
        state_indices: torch.Tensor,
        ids: torch.Tensor,
        table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _step_begin.step_begin(
            prep, meta_args, meta_kwargs, replay, state_indices, ids, table
        )

    def reference(
        self,
        prep: dict,
        meta_args: tuple,
        meta_kwargs: dict,
        replay,
        state_indices: torch.Tensor,
        ids: torch.Tensor,
        table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three launches unfolded, in the engine's order: the prep and the metadata are the
        engine's own entries (each bitwise its eager torch chain in the engine's tests), the
        begin is ``begin_fold``'s plain-torch reference."""
        fused_spec_head_prep(**prep)
        fused_dsa_helix_decode_metadata(*meta_args, **meta_kwargs)
        return begin_fold.reference(replay, state_indices, ids, table)

    def is_valid(
        self,
        prep: dict,
        meta_args: tuple,
        meta_kwargs: dict,
        replay,
        state_indices: torch.Tensor,
        ids: torch.Tensor,
        table: torch.Tensor,
    ) -> None:
        """Raise where the fold would read past its inputs: the begin's (``begin_fold``)."""
        if not _flat_begin.folds_embedding(ids, table):
            raise ValueError("step_begin: ids must be 1-D CUDA and the table contiguous, 16-B rows")
        begin_fold.is_valid(replay, state_indices, ids, table)


step_begin = StepBegin()


def prep_and_metadata(prep: dict, meta_args: tuple, meta_kwargs: dict) -> None:
    """The first two of :class:`StepBegin`'s launches unfolded, in order: for a step begin that
    does not fold the embedding rows (``begin_fold``'s unfolded path)."""
    fused_spec_head_prep(**prep)
    fused_dsa_helix_decode_metadata(*meta_args, **meta_kwargs)
