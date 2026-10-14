# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's speculative step-input staging (the in-tree ``SpecStepFold`` protocol).

Between two decode graphs of a one-model Eagle3 step, the spec sampler stores the step's outputs into its per-slot
store and the engine gathers them back into its flat input buffers for the next step. This object does both in one
launch each (``m3_spec_tail.slot_scatter`` / ``step_gather``): integer copies only, the same values the engine's and
the sampler's own copies write. The forward never stores the sampler's outputs itself, so ``begin_step`` and
``sampler_rows_stored`` always report that nothing was staged.

* ``store``       -- the spec sampler's per-slot store (``bind_sampler_store``).
* ``step_inputs`` -- the engine's (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets) device buffers
  (``bind_engine_buffers``).
"""

from __future__ import annotations

from typing import Optional

import torch

from .kernels import m3_spec_tail


class M3SpecStepFold:
    """``SpecStepFold`` implementation of the MiniMax-M3 tp8 target (see the module docstring).

    One instance per target model; the target passes it to ``RuntimeHooks.spec_step_fold``.
    """

    def __init__(self) -> None:
        self.store: Optional[object] = None
        self.step_inputs: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

    def bind_engine_buffers(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        position_id_offsets: torch.Tensor,
        kv_lens_offsets: torch.Tensor,
        max_batch_size: int,
    ) -> None:
        self.step_inputs = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        return False

    def gather(
        self,
        new_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        previous_slots: torch.Tensor,
        previous_pos_indices: torch.Tensor,
        previous_batch_len: int,
        tokens_per_gen_step: int,
        draft_token_buffer_width: int,
        num_tokens: int,
        num_draft_tokens: int,
        first_previous_row: int,
    ) -> bool:
        if self.step_inputs is None:
            return False
        input_ids, draft_tokens, position_id_offsets, kv_lens_offsets = self.step_inputs
        return m3_spec_tail.step_gather(
            new_tokens,
            next_draft_tokens,
            new_tokens_lens,
            previous_slots,
            previous_pos_indices,
            previous_batch_len,
            tokens_per_gen_step,
            draft_token_buffer_width,
            input_ids,
            num_tokens,
            draft_tokens,
            num_draft_tokens,
            position_id_offsets,
            kv_lens_offsets,
            first_previous_row,
        )

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list[int]) -> bool:
        return False

    def scatter(
        self,
        seq_slots: torch.Tensor,
        new_tokens: torch.Tensor,
        next_new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
    ) -> bool:
        store = self.store
        if store is None:
            return False
        return m3_spec_tail.slot_scatter(
            seq_slots,
            new_tokens,
            store.new_tokens.squeeze(-1).T,
            next_new_tokens,
            store.next_new_tokens.squeeze(-1).T,
            new_tokens_lens,
            store.new_tokens_lens,
            next_draft_tokens,
            store.next_draft_tokens,
        )
