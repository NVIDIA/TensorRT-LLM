# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec step fold of this target (``M3SpecStepFold``): the spec sampler's per-slot stores and the engine's
next-step input gathers between two decode graphs of a one-model Eagle3 step."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.pyexecutor.h2d_cache import copy_h2d_if_changed
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from .kernels import m3_fold_tail, m3_spec_tail


class M3SpecStepFold:
    """``SpecStepFold`` of this target.

    Between two decode graphs of a one-model Eagle3 step, the spec sampler stores the step's outputs into its per-slot
    store and the engine gathers them back into its flat input buffers for the next step. This object does both in one
    launch each (``m3_spec_tail.slot_scatter`` / ``step_gather``): integer copies only, the same values the engine's and
    the sampler's own copies write.

    With ``in_forward`` (``STEP_SPEC_FOLD_IN_FORWARD``) the forward of a decode-only step also does both at its tail
    (``store_in_forward``, one ``m3_fold_tail`` launch inside the step graph): it stores the step's outputs for the rows
    ``begin_step`` armed and stages the next step's inputs for those rows in that order. The arming is a device-side row
    count and slot list staged before the replay, so a captured graph whose step was not armed writes nothing. The
    sampler then skips its store for exactly the armed rows (``sampler_rows_stored``), and the next step skips its
    gathers when its rows are those rows (``begin_step``).

    One instance per target model; the target passes it to ``RuntimeHooks.spec_step_fold``.
    """

    def __init__(
        self, in_forward: bool = False, store_copy_wait: bool = False, cpp_tail: bool = False
    ) -> None:
        self.store: Optional[object] = None
        self.step_inputs: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.in_forward = in_forward
        # The forward's tail as this target's C++ op instead of the Triton kernel (m3_fold_tail.fold_tail(cpp=)).
        self.cpp_tail = in_forward and cpp_tail
        # Device arming read by the forward's tail: ctrl[0] = armed rows (0 = write nothing), slots[:rows].
        self._ctrl: Optional[torch.Tensor] = None
        self._slots: Optional[torch.Tensor] = None
        # The rows (sequence slots) the next forward stores and stages for, as begin_step armed them.
        self._armed: Optional[tuple[int, ...]] = None
        # The rows the previous forward staged the next step's inputs for (confirmed by sampler_rows_stored).
        self._staged_for: Optional[tuple[int, ...]] = None
        # Output buffers the tail ran on: the captured graphs' static outputs (by address) and the last eager step's.
        self._graph_outputs: set[int] = set()
        self._eager_outputs: Optional[torch.Tensor] = None
        self._logged = False
        # store_copy_wait: the sampler records this (external) event on its D2H side stream after copying the store
        # to the host, and the forward's tail waits on it before it rewrites the store; the sampler can then copy
        # on the side stream on the steps this forward stored the rows (see SpecSamplerBase._copy_outputs_to_host).
        self._store_copy_wait = in_forward and store_copy_wait
        self.store_copies_done: Optional[torch.cuda.Event] = None

    def bind_engine_buffers(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        position_id_offsets: torch.Tensor,
        kv_lens_offsets: torch.Tensor,
        max_batch_size: int,
    ) -> None:
        self.step_inputs = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)
        if self._store_copy_wait:
            # Recorded once now so a captured wait has a completed record to wait on before the first copy.
            self.store_copies_done = torch.cuda.Event(external=True)
            self.store_copies_done.record()
            logger.info(
                "[M3 host] spec store host copies on the D2H side stream, awaited by the forward's tail"
            )
        if self.cpp_tail:
            logger.info(
                "[M3 host] spec fold tail: one C++ launch (no Triton) at the forward's tail"
            )
        if self.in_forward:
            self._ctrl = torch.zeros(1, dtype=torch.int32, device=input_ids.device)
            self._slots = torch.zeros(
                max(max_batch_size, 1), dtype=torch.long, device=input_ids.device
            )

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        if not self.in_forward or self._ctrl is None:
            return False
        rows = tuple(seq_slots) if seq_slots is not None else None
        staged = rows is not None and rows == self._staged_for
        self._staged_for = None
        armed = rows if rows is not None and 0 < len(rows) <= self._slots.numel() else None
        # Read only by this step's forward, so both uploads may wait for the step-input pack's flush.
        if armed is not None:
            slots = torch.tensor(armed, dtype=torch.long, pin_memory=prefer_pinned())
            copy_h2d_if_changed(self._slots[: len(armed)], slots, pack=True)
        ctrl = torch.tensor(
            [len(armed) if armed is not None else 0], dtype=torch.int32, pin_memory=prefer_pinned()
        )
        copy_h2d_if_changed(self._ctrl, ctrl, pack=True)
        self._armed = armed
        return staged

    # The launches the store wait may precede. store_copies_done is an external event: under graph capture its wait
    # is a node of its own, so the kernel launched right after it loses its programmatic (PDL) edge and with it the
    # overlap of its pre-wait part. Only a launch that waits for its predecessor's completion anyway may follow it:
    # the fold tail (a plain launch, no griddepcontrol).
    STORE_WAIT_BEFORE = frozenset({"fold_tail"})

    def _wait_store_copies(self, before: str) -> None:
        """Order the store's rewrite after the sampler's host copies of it; ``before`` names the launch that follows.
        Under graph capture a launch outside STORE_WAIT_BEFORE fails the capture."""
        if self.store_copies_done is None:
            return
        if before not in self.STORE_WAIT_BEFORE and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"M3SpecStepFold: the store wait would precede {before!r}, a launch outside "
                f"{sorted(self.STORE_WAIT_BEFORE)}; an external-event wait node drops the next kernel's PDL edge"
            )
        torch.cuda.current_stream().wait_event(self.store_copies_done)

    def store_in_forward(self, outputs: dict) -> None:
        """At the forward's tail: store the armed rows' outputs and stage the next step's inputs (one launch)."""
        if self._ctrl is None or self.store is None or self.step_inputs is None:
            return
        store = self.store
        input_ids, draft_tokens, position_id_offsets, kv_lens_offsets = self.step_inputs
        new_tokens = outputs.get("new_tokens")
        next_new_tokens = outputs.get("next_new_tokens")
        new_tokens_lens = outputs.get("new_tokens_lens")
        next_draft_tokens = outputs.get("next_draft_tokens")
        if any(
            t is None or not isinstance(t, torch.Tensor)
            for t in (new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens)
        ):
            return
        rows = new_tokens.shape[0]
        # The previous step's host copies of the store run on the sampler's side stream.
        self._wait_store_copies(before="fold_tail")
        if not m3_fold_tail.fold_tail(
            self._ctrl,
            self._slots,
            min(rows, self._slots.numel()),
            new_tokens,
            store.new_tokens.squeeze(-1).T,
            next_new_tokens,
            store.next_new_tokens.squeeze(-1).T,
            next_draft_tokens,
            store.next_draft_tokens,
            new_tokens_lens,
            store.new_tokens_lens,
            input_ids,
            draft_tokens,
            position_id_offsets,
            kv_lens_offsets,
            next_new_tokens.shape[1],
            cpp=self.cpp_tail,
        ):
            return
        self._note_tail_ran(new_tokens)

    def _note_tail_ran(self, new_tokens: torch.Tensor) -> None:
        """Record that this forward's tail stored the step's outputs (``sampler_rows_stored`` reads it)."""
        if torch.cuda.is_current_stream_capturing():
            self._graph_outputs.add(new_tokens.data_ptr())
        else:
            self._eager_outputs = new_tokens

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
        armed, self._armed = self._armed, None
        new_tokens = outputs.get("new_tokens")
        ran = isinstance(new_tokens, torch.Tensor) and (
            new_tokens is self._eager_outputs or new_tokens.data_ptr() in self._graph_outputs
        )
        self._eager_outputs = None
        stored = (
            ran
            and armed is not None
            and num_skip == 0
            and tuple(seq_slots) == armed
            and new_tokens.shape[0] == len(armed)
        )
        self._staged_for = armed if stored else None
        if stored and not self._logged:
            self._logged = True
            logger.info("[M3 host] spec step stored and next step staged in the forward tail")
        return stored

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
