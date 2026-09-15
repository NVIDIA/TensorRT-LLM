# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Occurrence penalties (repetition / presence / frequency).

Two implementations of the same three penalties live here, differing only in the
execution contract they must satisfy:

* :class:`PenaltyHandler` (+ :class:`PenaltyStore`) -- the eager path.
  ``TorchSampler`` owns one instance and drives it through request validation,
  admission, the per-step apply, and the post-processing commit of finalized
  tokens. Carries the per-beam bookkeeping beam search needs.
* :class:`PenaltyState` and the ``apply_penalties`` / ``update_penalty_counts``
  pair -- the CUDA-graph-safe path, used by one-model speculative decoding.
  Buffers are preallocated at stable addresses and every branch that matters is
  taken on device, because a captured graph freezes host control flow.

Both compute the same formula, so a request decodes the same way whichever path
serves it.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from tensorrt_llm._utils import nvtx_range, prefer_pinned

from ..llm_request import LlmRequest
from .ops.vanilla import Fusions, occurrence_penalized_logits
from .sampler_common import _get_max_beam_width

__all__ = [
    "PenaltyHandler",
    "PenaltyState",
    "PenaltyStore",
    "apply_penalties",
    "build_row_mapping",
    "has_occurrence_penalty",
    "seed_prompt",
    "update_penalty_counts",
]


def has_occurrence_penalty(request: LlmRequest) -> bool:
    sampling_config = request.sampling_config
    repetition = sampling_config.repetition_penalty
    presence = sampling_config.presence_penalty
    frequency = sampling_config.frequency_penalty
    return (
        (repetition is not None and repetition != 1.0)
        or (presence is not None and presence != 0.0)
        or (frequency is not None and frequency != 0.0)
    )


@dataclass(kw_only=True)
class PenaltyStore:
    """Persistent device state: penalty-parameter buffers + occurrence workspace.

    This is the torch counterpart of the tensors ``PenaltyLayer`` allocates, and
    the anchor for the workspace semantics the ops and the handler rely on:

    * The **parameter buffers** (``repetition_cuda`` / ``presence_cuda`` /
      ``frequency_cuda``, plus the ``active_cuda`` gate) are the counterpart of
      ``allocateBuffer`` + ``fillBuffers``: one entry per sequence slot, written
      once per request and gathered every step, never rebuilt on the host.
    * The **occurrence workspace** (``counts_cuda`` and ``presence_prefix_cuda``)
      is the counterpart of ``allocateWorkspace`` / ``mPenaltyWorkspaceDevice``,
      updated incrementally each step. A token in the ignored prompt prefix
      ``[0, prompt_ignore_length)`` only sets ``presence_prefix_cuda``, so it
      contributes to the repetition penalty but not to presence/frequency; every
      other token (the rest of the prompt plus each generated token) increments
      ``counts_cuda``, which drives presence/frequency and -- via ``counts > 0`` --
      repetition as well.

    ``counts_cuda`` carries one row per *beam*: beam ``b`` of slot ``s`` owns row
    ``s * max_beam_width + b`` (see ``counts_rows``), because beams diverge and each
    needs its own history. ``presence_prefix_cuda`` stays one row per slot -- the
    ignored prompt prefix is shared by every beam of a request. With
    ``max_beam_width == 1`` the beam axis vanishes and both are plain slot-indexed.
    """

    max_num_sequences: int
    max_beam_width: int
    device: torch.device

    # --- Penalty parameters (allocateBuffer counterpart), shape [max_num_sequences] ---
    repetition_cuda: torch.Tensor
    """float32; per-slot repetition penalty (default 1.0)."""
    presence_cuda: torch.Tensor
    """float32; per-slot presence penalty (default 0.0)."""
    frequency_cuda: torch.Tensor
    """float32; per-slot frequency penalty (default 0.0)."""
    active_cuda: torch.Tensor
    """bool[slots]; whether a slot has an active occurrence penalty."""
    has_previous_token_cuda: torch.Tensor
    """bool[slots]; whether ``new_tokens`` contains a token to accumulate."""
    beam_slot_cuda: torch.Tensor
    """bool[slots]; whether the slot's counts must be re-parented each step.

    Beam width is a per-request property, so a beam engine may host single-beam requests
    too (see ``py_executor._validate_request`` / TRTLLM-14792). Only true beam slots have
    a meaningful ``predecessor_beams`` row; a single-beam slot's is never written and must
    not be believed. Stays all-False on a single-beam engine."""

    # --- Occurrence workspace (allocateWorkspace counterpart), allocated lazily ---
    counts_cuda: torch.Tensor | None = None
    """int32[slots * max_beam_width, vocab_size] or None; occurrence counts
    (see class docstring)."""
    presence_prefix_cuda: torch.Tensor | None = None
    """bool[slots, vocab_size] or None; ignored-prompt-prefix presence mask."""

    # Per-step request metadata, staged into persistent device buffers by
    # ``stage_request_metadata`` so the hot path does not allocate per step.
    request_offsets_cuda: torch.Tensor | None = None
    request_num_steps_cuda: torch.Tensor | None = None
    request_num_beams_cuda: torch.Tensor | None = None
    """Stays None on a single-beam engine, which never stages a beam width."""

    @classmethod
    def create(
        cls, *, max_num_sequences: int, max_beam_width: int, device: torch.device
    ) -> "PenaltyStore":
        """Allocate the vocab-independent buffers with their no-op defaults.

        ``inference_mode(False)`` guards every allocation in this class: the
        buffers persist across sampler steps and are mutated in place later, which
        inference-mode tensors forbid.
        """
        with torch.inference_mode(False):
            return cls(
                max_num_sequences=max_num_sequences,
                max_beam_width=max_beam_width,
                device=device,
                repetition_cuda=torch.ones(max_num_sequences, dtype=torch.float32, device=device),
                presence_cuda=torch.zeros(max_num_sequences, dtype=torch.float32, device=device),
                frequency_cuda=torch.zeros(max_num_sequences, dtype=torch.float32, device=device),
                active_cuda=torch.zeros(max_num_sequences, dtype=torch.bool, device=device),
                has_previous_token_cuda=torch.zeros(
                    max_num_sequences, dtype=torch.bool, device=device
                ),
                beam_slot_cuda=torch.zeros(max_num_sequences, dtype=torch.bool, device=device),
            )

    def counts_rows(self, slots_cuda: torch.Tensor) -> torch.Tensor:
        """Map slot indices to the ``counts_cuda`` rows they own.

        One row per beam, so slot ``s`` owns ``s * max_beam_width + b``. Returns
        ``slots_cuda`` unchanged in the single-beam case, where the two coincide.
        """
        if self.max_beam_width == 1:
            return slots_cuda
        beams = torch.arange(self.max_beam_width, device=slots_cuda.device)
        return (slots_cuda.unsqueeze(1) * self.max_beam_width + beams).reshape(-1)

    def ensure_workspace(self, *, vocab_size: int, needs_prefix: bool) -> None:
        """Allocate the vocab-sized workspace on first use.

        Deferred because ``vocab_size`` is only known once logits arrive, mirroring
        ``PenaltyLayer::allocateWorkspace`` being gated on penalty usage. The prefix
        mask is allocated only if some request has used ``prompt_ignore_length``.
        """
        with torch.inference_mode(False):
            if self.counts_cuda is None:
                self.counts_cuda = torch.zeros(
                    (self.max_num_sequences * self.max_beam_width, vocab_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            if needs_prefix and self.presence_prefix_cuda is None:
                self.presence_prefix_cuda = torch.zeros(
                    (self.max_num_sequences, vocab_size),
                    dtype=torch.bool,
                    device=self.device,
                )

    def stage_request_metadata(
        self,
        request_offsets_host: torch.Tensor,
        request_num_steps_host: torch.Tensor,
        request_num_beams_host: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Copy this step's ``[R]`` request metadata into persistent device buffers.

        The host tensors are already pinned by the caller, so each step costs a couple
        of small async H2D copies into a reused allocation rather than fresh device
        tensors. Returned views are only valid until the next call.

        All three buffers are ``[R]`` and grow together under one capacity check.
        ``request_num_beams_host`` is omitted on a single-beam engine, which then gets
        ``None`` back and allocates no third buffer; a beam engine must pass it on every
        call, including the first, or the buffer is never allocated.
        """
        num_requests = request_offsets_host.numel()
        with torch.inference_mode(False):
            if (
                self.request_offsets_cuda is None
                or self.request_offsets_cuda.numel() < num_requests
            ):
                capacity = max(num_requests, self.max_num_sequences)
                self.request_offsets_cuda = torch.empty(
                    capacity, dtype=request_offsets_host.dtype, device=self.device
                )
                self.request_num_steps_cuda = torch.empty(
                    capacity, dtype=request_num_steps_host.dtype, device=self.device
                )
                self.request_num_beams_cuda = (
                    torch.empty(capacity, dtype=request_num_beams_host.dtype, device=self.device)
                    if request_num_beams_host is not None
                    else None
                )
        assert self.request_num_steps_cuda is not None
        offsets = self.request_offsets_cuda[:num_requests]
        num_steps = self.request_num_steps_cuda[:num_requests]
        offsets.copy_(request_offsets_host, non_blocking=True)
        num_steps.copy_(request_num_steps_host, non_blocking=True)
        if request_num_beams_host is None:
            return offsets, num_steps, None
        assert self.request_num_beams_cuda is not None, (
            "a beam engine must stage request_num_beams from its first call onwards"
        )
        num_beams = self.request_num_beams_cuda[:num_requests]
        num_beams.copy_(request_num_beams_host, non_blocking=True)
        return offsets, num_steps, num_beams


class PenaltyHandler:
    """Applies the occurrence penalties: repetition, presence and frequency.

    These rescale or subtract from a token's logit based on how often it has already
    occurred, and run before the sampling strategy divides by temperature. Bans that
    force a logit to -inf (min_length, bad words, no-repeat-ngram) are a different
    kind of transform and live in ``TokenBanHandler``.

    The implementation follows the semantics of the former C++
    ``batchApplyPenalty`` kernel as driven by ``PenaltyLayer``.
    Its persistent device state lives in :class:`PenaltyStore`, which documents the
    workspace semantics. Per-slot parameter buffers are filled once per request,
    batched across all requests admitted in a step (``prepare_for_new_request``
    accumulates on the host, ``update_for_new_requests`` issues the device updates).
    Vocab-sized workspaces are allocated lazily and skipped entirely when no matching
    request uses an occurrence penalty.
    """

    @dataclass(kw_only=True)
    class _SlotState:
        """Per-slot host-only bookkeeping (never read by the ops)."""

        prompt_ignore_length: int
        uses_beam_search: bool = False
        initialized: bool = False

    def __init__(
        self,
        *,
        max_num_sequences: int,
        max_beam_width: int,
        device: torch.device | str,
    ):
        self._max_num_sequences = max_num_sequences
        self._max_beam_width = max_beam_width
        self._device = torch.device(device)
        # Whether any (past or current) active request uses prompt_ignore_length > 0,
        # which requires allocating the presence-prefix mask.
        self._needs_prefix = False
        self._num_active_slots = 0
        # Per-slot state; None marks a slot without active occurrence penalties.
        self._slots: list[PenaltyHandler._SlotState | None] = [None] * max_num_sequences
        # Slots admitted this step that carry an occurrence penalty, with their
        # parameters; drained by ``update_for_new_requests``.
        self._new_slots: list[int] = []
        self._new_repetition: list[float] = []
        self._new_presence: list[float] = []
        self._new_frequency: list[float] = []
        self._new_beam_slot: list[bool] = []
        self.store = PenaltyStore.create(
            max_num_sequences=max_num_sequences,
            max_beam_width=max_beam_width,
            device=self._device,
        )

    def _to_device(self, values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, pin_memory=prefer_pinned()).to(
            self._device, non_blocking=True
        )

    def prepare_for_new_request(self, request: LlmRequest, slot: int) -> None:
        """Record the slot's penalty parameters for this step's batched flush.

        Called from ``TorchSampler.setup_sampler_step`` for each new request, mirroring
        ``PenaltyLayer::setup`` (``fillBuffers`` + per-``batchSlot`` ``setZero``). This
        only touches host state; ``update_for_new_requests`` issues the device updates
        for all requests admitted in the step at once. Inactive slots are never
        gathered, so their stale parameters/counts are left untouched.
        """
        was_active = self._slots[slot] is not None
        if not has_occurrence_penalty(request):
            self._slots[slot] = None
            if was_active:
                self._num_active_slots -= 1
            return

        sampling_config = request.sampling_config
        repetition = sampling_config.repetition_penalty
        presence = sampling_config.presence_penalty
        frequency = sampling_config.frequency_penalty
        prompt_ignore_length = sampling_config.prompt_ignore_length
        # min(prompt_ignore_length, inputLen), matching the C++ kernel.
        prompt_ignore_length = min(
            prompt_ignore_length if prompt_ignore_length is not None else 0,
            request.py_orig_prompt_len,
        )
        if prompt_ignore_length > 0:
            self._needs_prefix = True

        uses_beam_search = self._max_beam_width > 1 and _get_max_beam_width(request) > 1
        self._slots[slot] = self._SlotState(
            prompt_ignore_length=prompt_ignore_length,
            uses_beam_search=uses_beam_search,
        )
        if not was_active:
            self._num_active_slots += 1

        self._new_slots.append(slot)
        self._new_repetition.append(repetition if repetition is not None else 1.0)
        self._new_presence.append(presence if presence is not None else 0.0)
        self._new_frequency.append(frequency if frequency is not None else 0.0)
        self._new_beam_slot.append(uses_beam_search)

    def update_for_new_requests(self, *, new_seq_slots_cuda_long: torch.Tensor) -> None:
        """Flush this step's admissions to the device in a handful of batched updates.

        ``new_seq_slots_cuda_long`` holds *every* slot admitted this step. Clearing the
        active gate and the pending-token flag across all of them also covers slot
        reuse: a slot whose prior occupant was penalized but whose new occupant is not
        must read False.
        """
        store = self.store
        store.active_cuda.index_fill_(0, new_seq_slots_cuda_long, False)
        store.has_previous_token_cuda.index_fill_(0, new_seq_slots_cuda_long, False)
        if self._max_beam_width > 1:
            # Stays all-False on a single-beam engine, so clearing it there would be a
            # kernel launch per step for nothing.
            store.beam_slot_cuda.index_fill_(0, new_seq_slots_cuda_long, False)

        if not self._new_slots:
            return

        slots_cuda = self._to_device(self._new_slots, torch.int64)
        # One [3, N] host tensor -> one H2D for all three parameter buffers.
        params_cuda = torch.tensor(
            [self._new_repetition, self._new_presence, self._new_frequency],
            dtype=torch.float32,
            pin_memory=prefer_pinned(),
        ).to(self._device, non_blocking=True)
        store.repetition_cuda.index_copy_(0, slots_cuda, params_cuda[0])
        store.presence_cuda.index_copy_(0, slots_cuda, params_cuda[1])
        store.frequency_cuda.index_copy_(0, slots_cuda, params_cuda[2])
        store.active_cuda.index_fill_(0, slots_cuda, True)
        if self._max_beam_width > 1:
            store.beam_slot_cuda.index_copy_(
                0,
                slots_cuda,
                torch.tensor(self._new_beam_slot, dtype=torch.bool, pin_memory=prefer_pinned()).to(
                    self._device, non_blocking=True
                ),
            )

        # Re-zero the workspace rows so a prior occupant's counts do not leak in.
        # counts_cuda holds one row per beam, so every beam of the slot must be cleared.
        if store.counts_cuda is not None:
            store.counts_cuda.index_fill_(0, store.counts_rows(slots_cuda), 0)
        if store.presence_prefix_cuda is not None:
            store.presence_prefix_cuda.index_fill_(0, slots_cuda, False)

        self._new_slots.clear()
        self._new_repetition.clear()
        self._new_presence.clear()
        self._new_frequency.clear()
        self._new_beam_slot.clear()

    def _initialize_workspace(
        self,
        request: LlmRequest,
        state: "PenaltyHandler._SlotState",
        vocab_size: int,
    ) -> None:
        """Initialize one regular slot from its prompt exactly once."""
        if state.initialized:
            return

        slot = request.py_seq_slot
        assert slot is not None
        counts_cuda = self.store.counts_cuda
        assert counts_cuda is not None

        prompt = request.get_tokens(0)[: request.py_orig_prompt_len]
        state.initialized = True
        if not prompt:
            return

        # One conversion for the whole prompt; the split point is just
        # prompt_ignore_length, so the two groups are plain slices.
        base_row = slot * self._max_beam_width
        tokens = self._to_device(prompt, torch.int64)
        prefix_tokens = tokens[: state.prompt_ignore_length]
        counted_tokens = tokens[state.prompt_ignore_length :]

        # Multimodal models place placeholder ids >= vocab_size in the prompt (see
        # _torch/models/modeling_multimodal_utils.py), so out-of-range ids reach us
        # here and must be dropped before they index the workspace.
        counted_tokens = counted_tokens[(counted_tokens >= 0) & (counted_tokens < vocab_size)]
        prefix_tokens = prefix_tokens[(prefix_tokens >= 0) & (prefix_tokens < vocab_size)]

        Fusions.update_occurrence_workspace(
            counts_cuda,
            self.store.presence_prefix_cuda,
            torch.full_like(counted_tokens, base_row),
            counted_tokens,
            # The prefix mask is per slot: every beam shares the prompt.
            torch.full_like(prefix_tokens, slot),
            prefix_tokens,
        )
        if state.uses_beam_search:
            # Every beam starts from the same prompt. Seeding all of them, rather than
            # only beam 0, also covers a generation-only (disaggregated decode) request
            # whose first penalized step already has several beams and hence no
            # re-parenting to broadcast beam 0's counts for it.
            counts_cuda[base_row + 1 : base_row + self._max_beam_width].copy_(counts_cuda[base_row])

    def update_token_counts(
        self,
        updates: list[tuple[int, list[int]]],
    ) -> None:
        """Commit finalized sampled tokens that replaced the device pending token.

        This is used after sampler-side postprocessing has finalized a multi-token
        result. The complete confirmed sequence is counted here, then the raw first
        token left in ``new_tokens`` is marked consumed so the next kernel cannot count
        it again. Regular one-token sampling never calls this method and keeps its
        fused device-pending fast path.
        """
        if not updates or self._num_active_slots == 0:
            return

        counts_cuda = self.store.counts_cuda
        assert counts_cuda is not None
        # Only speculative decoding reaches here, and it is rejected together with
        # beam search (TorchSampler.__init__), so beam 0 is the only live beam.
        assert self._max_beam_width == 1, "speculative token commit is single-beam only"
        vocab_size = counts_cuda.size(-1)
        consumed_slots: list[int] = []
        counted_slots: list[int] = []
        counted_tokens: list[int] = []

        for slot, tokens in updates:
            if self._slots[slot] is None:
                continue
            consumed_slots.append(slot)
            for token in tokens:
                if 0 <= token < vocab_size:
                    counted_slots.append(slot)
                    counted_tokens.append(token)

        if consumed_slots:
            self.store.has_previous_token_cuda.index_fill_(
                0, self._to_device(consumed_slots, torch.int64), False
            )

        if not counted_tokens:
            return

        Fusions.update_occurrence_workspace(
            counts_cuda,
            self.store.presence_prefix_cuda,
            self._to_device(counted_slots, torch.int64),
            self._to_device(counted_tokens, torch.int64),
        )

    @nvtx_range("apply_penalties")
    @torch.inference_mode()
    def apply(
        self,
        logits: torch.Tensor,
        requests: list[LlmRequest],
        *,
        new_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        request_offsets: torch.Tensor,
        request_num_steps: torch.Tensor,
        request_num_beams: torch.Tensor | None = None,
        predecessor_beams: torch.Tensor | None = None,
        is_draft_batch: bool = False,
    ) -> None:
        """Advance the occurrence state for this step and apply the penalties to ``logits``.

        ``logits`` is the packed generated-token logits ``[sum(num_steps * num_beams),
        vocab_size]``; request ``r`` owns ``request_num_steps[r] * request_num_beams[r]``
        consecutive rows starting at ``request_offsets[r]``, in beam-major / step-minor
        order. ``request_offsets`` / ``request_num_steps`` are the caller's pinned host
        tensors and are staged to the device here.

        Beam search changes only where the counts come from, never where the penalty is
        applied: every row is rewritten here, in place, on raw logits and before
        temperature -- the same position in the pipeline as the single-beam path, so both
        keep the ordering ``bias -> penalty -> bans -> temperature -> sampling``. What
        beam search adds is a per-beam counts row, re-parented each step, and a row ->
        (slot, beam) mapping so each beam is penalized against its own history.

        With ``max_beam_width == 1`` the pending-token fold stays fused into the same
        graph, so the whole step is one kernel.

        Args:
            request_num_beams / predecessor_beams: required when ``max_beam_width > 1``.
                ``request_num_beams`` is the row-layout width the caller packed ``logits``
                at (the static admission width, not the per-iteration one), so that the
                row -> beam mapping matches ``request_offsets``. ``predecessor_beams``
                must still hold the *previous* step's parent map, i.e. this must run
                before the step's beam sampling overwrites it.
            is_draft_batch: draft batches share this sampler but draw ``py_seq_slot``
                from a separate numbering space that collides with target slots, so
                penalizing them would read/write an unrelated target request's
                occurrence state; skip them like the pending-steps tracking.
        """
        if is_draft_batch or not requests or self._num_active_slots == 0:
            return

        # Cheap per-batch scan so the vocab-sized workspace is only allocated when this
        # batch actually contains a penalized request.
        active_requests: list[tuple[LlmRequest, "PenaltyHandler._SlotState"]] = []
        for request in requests:
            slot = request.py_seq_slot
            assert slot is not None
            state = self._slots[slot]
            if state is not None:
                active_requests.append((request, state))
        if not active_requests:
            return

        store = self.store
        store.ensure_workspace(vocab_size=logits.size(-1), needs_prefix=self._needs_prefix)
        counts_cuda = store.counts_cuda
        assert counts_cuda is not None
        for request, state in active_requests:
            self._initialize_workspace(request, state, logits.size(-1))

        if self._max_beam_width > 1:
            assert request_num_beams is not None and predecessor_beams is not None, (
                "beam search requires request_num_beams and predecessor_beams"
            )
            num_beams_host = request_num_beams
        else:
            # Left None so the packed pass specializes to its single-beam graph.
            num_beams_host = None

        # Staged ahead of both consumers, since the views last only until the next call.
        request_offsets_cuda, request_num_steps_cuda, num_beams_cuda = store.stage_request_metadata(
            request_offsets, request_num_steps, num_beams_host
        )

        if self._max_beam_width > 1:
            # Both were established by the branch above -- a beam engine always stages a
            # beam width -- but the narrowing does not survive stage_request_metadata, so
            # restate it for the type checker.
            assert predecessor_beams is not None and num_beams_cuda is not None
            # Re-parent and fold up front. On a single-beam engine the fold stays fused
            # into the packed graph below; here it cannot, because re-parenting has to
            # happen first and it is not expressible inside that graph.
            Fusions.update_beam_occurrence_counts(
                counts_cuda,
                store.active_cuda,
                store.has_previous_token_cuda,
                store.beam_slot_cuda,
                new_tokens,
                predecessor_beams,
                seq_slots,
                num_beams_cuda,
                self._max_beam_width,
            )
        Fusions.apply_batched_occurrence_penalties(
            logits,
            counts_cuda,
            store.presence_prefix_cuda,
            store.active_cuda,
            store.has_previous_token_cuda,
            new_tokens,
            seq_slots,
            request_offsets_cuda,
            request_num_steps_cuda,
            store.repetition_cuda,
            store.presence_cuda,
            store.frequency_cuda,
            # None / 1 / True on a single-beam engine: no beam axis in the row mapping and
            # the fold stays fused in.
            num_beams_cuda,
            self._max_beam_width,
            self._max_beam_width == 1,
        )
        self._arm_pending_tokens(requests, request_num_steps)

    def _arm_pending_tokens(
        self, requests: list[LlmRequest], request_num_steps: torch.Tensor
    ) -> None:
        """Arm has_previous_token for the slots this step advanced (active, num_steps > 0).

        The next call then folds their sampled ``new_tokens``. Done on the host rather than
        in the compiled op because the fold reads the flag for every request row; flipping
        it in the same graph would make the result depend on execution order within the
        kernel.

        The scan is kept on the host deliberately. The same thing can be expressed on
        device as active_cuda[seq_slots] & (num_steps > 0), avoiding this loop and the
        H2D, but that costs several extra kernel launches and measured 5-7us slower for
        batches up to 32 and no better at 64-256: the loop overlaps with the model
        forward, the launches do not.
        """
        pending_token_slots: list[int] = []
        for request, num_steps in zip(requests, request_num_steps.tolist()):
            slot = request.py_seq_slot
            if slot is None:
                continue
            if self._slots[slot] is not None and num_steps > 0:
                pending_token_slots.append(slot)
        if pending_token_slots:
            self.store.has_previous_token_cuda.index_fill_(
                0, self._to_device(pending_token_slots, torch.int64), True
            )


# ---------------------------------------------------------------------------
# CUDA-graph-safe occurrence penalties
#
# A second implementation of the same three penalties, for callers that run
# inside a captured CUDA graph -- today the one-model speculative decoding
# worker, and any future graph-capturing TorchSampler path.
#
# It exists alongside PenaltyHandler above rather than replacing it because the
# execution contracts differ: PenaltyHandler may allocate lazily and skip work
# with a host-side branch, whereas everything below must preallocate at stable
# addresses and gate on device tensors, since a captured graph freezes host
# control flow and buffer pointers. PenaltyHandler additionally carries the
# per-beam bookkeeping that speculative decoding never needs (beam width is
# always 1 there).
#
# The arithmetic itself is shared: both halves call
# ``ops.vanilla.occurrence_penalized_logits``, so only the operand gathering
# differs between them.
# ---------------------------------------------------------------------------


class SpecMetadataLike(Protocol):
    """What the CUDA-graph penalty entry points need from the caller's metadata.

    A Protocol rather than an import of ``SpecMetadata``: that type lives in the
    speculative package, which already imports this module, so naming it directly
    would close an import cycle.
    """

    enable_penalty: bool
    vocab_size: int
    penalty_state: Optional["PenaltyState"]
    batch_slot_ids: Optional[torch.Tensor]


@dataclass(kw_only=True)
class PenaltyState:
    """Device state backing the occurrence penalties, one row per sequence slot.

    Split by where a token came from, because the three penalties do not agree on
    what the prompt means:

    * ``counts`` -- int32 [rows, vocab_size]. How often the MODEL produced each
      token, plus any prompt tokens past ``prompt_ignore_length``. Read as
      ``count > 0`` by repetition and presence, and as the count itself by
      frequency.
    * ``prompt_mask`` -- int32 [rows, ceil(vocab_size / 32)], one bit per token id.
      The ignored prompt prefix, which drives repetition ONLY: the user's own text
      is not charged presence/frequency. Repetition only asks "seen or not", so a
      bit carries everything a count would, at 1/32 the memory.

    The parameter vectors hold their no-op defaults (1.0 / 0.0 / 0.0), so a slot
    that was never filled leaves logits untouched, and ``active`` gates each row on
    device -- which is what lets a captured CUDA graph decide per replay whether a
    row is penalized.

    ``rows`` is the slot pool plus one scratch row: CUDA-graph dummy/padding
    requests (``py_seq_slot is None``) are routed to ``dummy_row`` so they can never
    disturb a live request's history.
    """

    counts: torch.Tensor
    prompt_mask: torch.Tensor
    repetition: torch.Tensor
    presence: torch.Tensor
    frequency: torch.Tensor
    active: torch.Tensor
    dummy_row: int

    @classmethod
    def create(
        cls, *, slot_capacity: int, vocab_size: int, device: torch.device | str = "cuda"
    ) -> "PenaltyState":
        """Allocate every buffer up front, at addresses that stay put.

        Deliberately not lazy (unlike ``TorchSampler``'s
        ``PenaltyStore.ensure_workspace``): a captured CUDA graph records fixed
        pointers, so a first allocation after capture would leave the replayed
        kernel reading the wrong memory.
        """
        rows = slot_capacity + 1
        return cls(
            counts=torch.zeros((rows, vocab_size), dtype=torch.int32, device=device),
            # 32 token ids per int32 word.
            prompt_mask=torch.zeros(
                (rows, (vocab_size + 31) // 32), dtype=torch.int32, device=device
            ),
            repetition=torch.ones((rows,), dtype=torch.float32, device=device),
            presence=torch.zeros((rows,), dtype=torch.float32, device=device),
            frequency=torch.zeros((rows,), dtype=torch.float32, device=device),
            active=torch.zeros((rows,), dtype=torch.bool, device=device),
            dummy_row=slot_capacity,
        )


def _unpack_prompt_mask(
    prompt_mask: torch.Tensor, row_slots: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Expand the packed prompt bitmask to a ``bool[T, vocab_size]`` view.

    ``prompt_mask`` stores one bit per token id (32 ids per int32 word), which is
    what makes it 32x smaller than a count tensor. Only presence is representable,
    which is all repetition needs.
    """
    words = prompt_mask[row_slots]  # [T, ceil(vocab/32)]
    bits = torch.arange(32, device=prompt_mask.device, dtype=torch.int32)
    unpacked = (words.unsqueeze(-1) >> bits) & 1  # [T, words, 32]
    return unpacked.reshape(words.size(0), -1)[:, :vocab_size].to(torch.bool)


@torch.compile(dynamic=None, fullgraph=True)
def _apply_penalties_impl(
    logits: torch.Tensor,
    counts: torch.Tensor,
    prompt_mask: torch.Tensor,
    row_slots: torch.Tensor,
    intra_step_tokens: torch.Tensor,
    intra_step_valid: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Rewrite ``logits`` in place with the three occurrence penalties.

    NB: traced by ``torch.compile(fullgraph=True)`` and replayed inside a captured
    CUDA graph. That imposes constraints invisible from the body:

    * no data-dependent shapes and no host syncs -- every extent must come from the
      argument shapes, never from a value read off a device tensor;
    * ``logits`` is the ONLY tensor written; the workspaces are read-only here, so
      the same graph can be replayed for any batch whose state has changed;
    * dims the caller marked dynamic must not be read back as concrete sizes, or
      dynamo specializes them and later batch shapes fall out of the graph.

    Callers pass a private copy of ``logits``: see ``_apply_occurrence_penalties``.
    """
    # Fold this step's earlier speculative positions into the counts each row sees.
    # Position k must be penalized against positions 0..k-1 of the SAME step, which
    # are not in ``counts`` yet (they are only committed once acceptance is known).
    # ``intra_step_tokens`` is [T, draft_len]: entry [r, j] is the token row r's own
    # request drafted at position j, and ``intra_step_valid`` keeps only the
    # positions strictly earlier than r's -- a position never penalizes itself.
    # Because a row only holds its own request's drafts, requests cannot leak into
    # each other regardless of how the batch is packed.
    #
    # One-hot accumulate rather than scatter_add_, because rows sharing a slot would
    # otherwise race on the same counts entry -- and a captured graph cannot rely on
    # scatter ordering.
    # Bound the ids on BOTH sides before they index the workspace. A draft slot can
    # legitimately hold an id outside [0, vocab_size) -- an unfilled/padding entry, or
    # a reduced-vocab draft head -- and an unbounded id would scatter out of bounds.
    vocab = logits.size(-1)
    in_range = (intra_step_tokens >= 0) & (intra_step_tokens < vocab)
    safe_tokens = intra_step_tokens.masked_fill(~in_range, 0)
    intra_counts = torch.zeros_like(logits, dtype=torch.int32)
    intra_counts.scatter_add_(
        1, safe_tokens.to(torch.int64), (intra_step_valid & in_range).to(torch.int32)
    )

    # Output-side occurrences: drive all three penalties.
    count = counts[row_slots] + intra_counts

    rep = repetition[row_slots].unsqueeze(1)
    pre = presence[row_slots].unsqueeze(1)
    freq = frequency[row_slots].unsqueeze(1)

    output_seen = count > 0
    # Prompt-side occurrences drive repetition ONLY -- a token the user wrote should
    # not be charged presence/frequency, which describe what the model itself emitted.
    # Storing them as a bitmask costs 1 bit instead of the 32 an int32 count would.
    seen_for_repetition = output_seen | _unpack_prompt_mask(prompt_mask, row_slots, logits.size(-1))

    penalized = occurrence_penalized_logits(
        logits,
        count=count,
        seen_for_repetition=seen_for_repetition,
        seen_for_presence=output_seen,
        repetition=rep,
        presence=pre,
        frequency=freq,
    )
    # Cast before the select so untouched rows stay bit-identical.
    row_active = active[row_slots].unsqueeze(1)
    logits.copy_(torch.where(row_active, penalized, logits))


def apply_penalties(
    logits: torch.Tensor,
    spec_metadata: "SpecMetadataLike",
    row_slots: torch.Tensor,
    intra_step_tokens: torch.Tensor,
    intra_step_valid: torch.Tensor,
    any_active: bool = True,
) -> None:
    """Apply the occurrence penalties to ``logits`` in place, before sampling.

    Args:
        logits: ``[T, vocab_size]`` target logits. Modified in place.
        spec_metadata: carries the buffers from ``prepare_penalty_buffers``.
        row_slots: ``int64[T]`` slot row owning each logits row. Rows belonging to
            no live request must point at ``penalty_dummy_row``, whose parameters
            keep their no-op defaults.
        intra_step_tokens: ``int64[T, draft_len]`` -- column j is the token this
            row's own request drafted at position j. Only ``intra_step_valid``
            decides which of them count, so entries are never sentinels.
        intra_step_valid: ``bool[T, draft_len]`` mask matching ``intra_step_tokens``;
            true only for positions strictly earlier than the row's own.
        any_active: whether any request in this batch actually has a penalty. False
            skips the launch entirely -- the whole pass is a vocab-sized read/write
            that would otherwise be paid to compute a no-op. The caller knows this
            from host state (``SpecMetadata.batch_uses_penalty``), so testing it
            here costs no device sync.

    No-op unless the penalty buffers exist (i.e. ``enable_penalty``).
    """
    if not getattr(spec_metadata, "enable_penalty", False) or not any_active:
        return
    state = spec_metadata.penalty_state
    if state is None or logits.numel() == 0:
        return
    counts = state.counts
    # A vocab-sharded logits view cannot be penalized against full-vocab counts.
    if logits.size(-1) != counts.size(-1):
        return

    _apply_penalties_impl(
        logits,
        counts,
        state.prompt_mask,
        row_slots,
        intra_step_tokens,
        intra_step_valid,
        state.repetition,
        state.presence,
        state.frequency,
        state.active,
    )


def build_row_mapping(
    spec_metadata: "SpecMetadataLike",
    num_contexts: int,
    batch_size: int,
    draft_len: int,
    draft_tokens: torch.Tensor,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Derive the per-row slot map and intra-step prefix for a packed logits batch.

    Assumes the layout every linear one-model mode produces once
    ``_reshape_logits_for_accept`` has run: ``[ctx (1 row each), gen (draft_len + 1
    rows each)]``, request-major. PARD emits ``2 * draft_len`` rows per gen request
    but that hook already narrows them to ``draft_len + 1``, so both share this code.

    Tree modes do NOT fit: their rows are tree nodes, so a row's prefix is its
    root path rather than the rows before it. They are rejected at admission.

    Returns ``(row_slots, intra_tokens, intra_valid)`` shaped for
    :func:`apply_penalties`, or ``None`` when there is nothing to penalize. The two
    intra-step tensors are ``[total_rows, draft_len]``: a row only ever reads its own
    request's draft positions, so the width is the draft length rather than the batch
    row count. (The wide ``[T, T]`` form would reach hundreds of MiB per step at
    serving batch sizes while leaving all but ``draft_len`` columns per row unused.)
    """
    slot_ids = spec_metadata.batch_slot_ids
    if slot_ids is None:
        return None

    num_gens = batch_size - num_contexts
    rows_per_gen = draft_len + 1
    total_rows = num_contexts + num_gens * rows_per_gen

    batch_slots = slot_ids[:batch_size].to(torch.int64)
    # Context requests own one row; gen requests own rows_per_gen consecutive rows.
    row_slots = (
        torch.cat(
            [
                batch_slots[:num_contexts],
                batch_slots[num_contexts:batch_size].repeat_interleave(rows_per_gen),
            ]
        )
        if num_gens > 0
        else batch_slots[:num_contexts]
    )

    # Intra-step prefix: within one gen request, row k must see the draft tokens at
    # positions 0..k-1 -- they are this step's earlier speculative positions, not yet
    # committed to the counts. Strictly earlier: a position never penalizes itself.
    #
    # Column j of a row means "this request's draft position j", so requests cannot
    # see each other's tokens by construction -- no cross-request masking needed.
    intra_tokens = torch.zeros((total_rows, max(draft_len, 0)), dtype=torch.int64, device=device)
    intra_valid = torch.zeros((total_rows, max(draft_len, 0)), dtype=torch.bool, device=device)
    if num_gens > 0 and draft_len > 0 and draft_tokens.numel() > 0:
        drafts = draft_tokens.reshape(num_gens, -1)[:, :draft_len].to(torch.int64)
        gen_rows = torch.arange(num_gens * rows_per_gen, device=device)
        g_of_row = gen_rows // rows_per_gen  # which request owns the row
        k_of_row = gen_rows % rows_per_gen  # the row's speculative position
        # Every gen row carries its own request's drafts; validity alone selects the
        # strictly-earlier prefix. Column j counts for row k iff j < k.
        intra_tokens[num_contexts:] = drafts[g_of_row]
        intra_valid[num_contexts:] = torch.arange(draft_len, device=device).unsqueeze(
            0
        ) < k_of_row.unsqueeze(1)
    return row_slots, intra_tokens, intra_valid


def seed_prompt(
    spec_metadata: "SpecMetadataLike",
    slot: int,
    prompt_tokens: torch.Tensor,
    prompt_ignore_length: int = 0,
) -> None:
    """Seed a request's prompt into its occurrence state, once per sequence.

    ``prompt_ignore_length`` splits the prompt the same way the C++ penalty kernel
    does: the first N tokens are recorded in the packed bitmask, so they drive
    repetition only, while the remainder is counted like generated text and drives
    all three penalties. N is clamped to the prompt length, and values <= 0 mean the
    whole prompt is counted.

    ``prompt_tokens`` may legitimately contain ids outside ``[0, vocab_size)`` --
    multimodal models place placeholders above the vocab -- so they are dropped
    rather than allowed to index the workspace.

    No-op unless the penalty buffers exist (i.e. ``enable_penalty``).
    """
    if not getattr(spec_metadata, "enable_penalty", False):
        return
    state = spec_metadata.penalty_state
    if state is None or prompt_tokens.numel() == 0:
        return
    prompt_mask, counts = state.prompt_mask, state.counts
    # Live rows only. A negative index would wrap onto another request's row, and
    # the scratch row at dummy_row belongs to CUDA-graph padding, not to a sequence.
    if not 0 <= slot < state.dummy_row:
        return

    vocab_size = spec_metadata.vocab_size
    tokens = prompt_tokens.to(device=prompt_mask.device, dtype=torch.int64)
    ignore = max(0, min(prompt_ignore_length, tokens.numel()))
    ignored, counted = tokens[:ignore], tokens[ignore:]

    ignored = ignored[(ignored >= 0) & (ignored < vocab_size)]
    if ignored.numel():
        # Deduplicate first: two ids in the same 32-id word must contribute both
        # bits, so the reduction has to be a bitwise OR. scatter_reduce_ has no OR
        # mode and 'amax' would keep only the larger bit; over unique ids the bits
        # within a word are disjoint, so a plain sum IS their bitwise OR.
        unique = torch.unique(ignored)
        row = torch.zeros_like(prompt_mask[slot])
        row.scatter_add_(
            0,
            unique // 32,
            torch.ones_like(unique, dtype=torch.int32) << (unique % 32).to(torch.int32),
        )
        prompt_mask[slot] = prompt_mask[slot].bitwise_or(row)

    counted = counted[(counted >= 0) & (counted < vocab_size)]
    if counted.numel():
        # Counted like generated tokens: repeats must accumulate, so this is a real
        # count rather than a bitmask.
        counts[slot].scatter_add_(0, counted, torch.ones_like(counted, dtype=counts.dtype))


@torch.compile(dynamic=None, fullgraph=True)
def _update_penalty_counts_impl(
    counts: torch.Tensor,
    slot_rows: torch.Tensor,
    tokens: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Accumulate ``tokens`` into ``counts`` in place, one row per request.

    NB: traced by ``torch.compile(fullgraph=True)``. Keep it free of
    data-dependent shapes and host syncs -- ``valid`` masks the entries to skip
    rather than filtering them out, precisely so the shapes stay static.
    """
    vocab = counts.size(-1)
    safe_tokens = tokens.masked_fill(~valid, 0)
    flat = slot_rows.unsqueeze(1) * vocab + safe_tokens
    counts.view(-1).scatter_add_(0, flat.reshape(-1), valid.to(counts.dtype).reshape(-1))


def update_penalty_counts(
    spec_metadata: "SpecMetadataLike",
    slot_rows: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Commit the tokens this step accepted into the persistent counts.

    Only accepted tokens are counted: a rejected speculative token was never part
    of the sequence and must not penalize anything.

    Args:
        spec_metadata: carries the buffers from ``prepare_penalty_buffers``.
        slot_rows: ``int64[B]`` slot row per request; dummy/padding requests must
            point at ``penalty_dummy_row``.
        accepted_tokens: ``int32[B, max_accepted]`` accepted token ids.
        num_accepted_tokens: ``int32[B]`` how many of each row's entries are real.

    No-op unless the penalty buffers exist (i.e. ``enable_penalty``).
    """
    if not getattr(spec_metadata, "enable_penalty", False):
        return
    state = spec_metadata.penalty_state
    if state is None or accepted_tokens.numel() == 0:
        return
    counts = state.counts

    vocab = counts.size(-1)
    positions = torch.arange(accepted_tokens.size(1), device=accepted_tokens.device)
    tokens = accepted_tokens.to(torch.int64)
    # Drop padding columns past num_accepted, and ids outside the vocab
    # (multimodal placeholders and the untouched tail of the buffer).
    valid = (
        (positions.unsqueeze(0) < num_accepted_tokens.unsqueeze(1).to(positions.dtype))
        & (tokens >= 0)
        & (tokens < vocab)
    )
    # Dummy rows keep active=False, so their counts never reach a real request,
    # but they still must not corrupt a shared row: they are masked out here too.
    valid = valid & state.active[slot_rows].unsqueeze(1)

    _update_penalty_counts_impl(counts, slot_rows, tokens, valid)
