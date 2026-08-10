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

"""Occurrence penalties (repetition / presence / frequency) for ``TorchSampler``.

The feature's persistent device state lives in :class:`PenaltyStore` (which
documents the workspace semantics) and its whole lifecycle in
:class:`PenaltyHandler`; ``TorchSampler`` owns one instance and drives it
through request validation, admission, the per-step apply, and the
post-processing commit of finalized tokens.
"""

from dataclasses import dataclass

import torch

from tensorrt_llm._utils import nvtx_range, prefer_pinned

from ..llm_request import LlmRequest
from .ops.vanilla import Fusions
from .sampler_common import _get_max_beam_width, _unwrap_singleton

__all__ = ["PenaltyHandler", "PenaltyStore", "has_occurrence_penalty"]


def has_occurrence_penalty(request: LlmRequest) -> bool:
    sampling_config = request.sampling_config
    repetition = _unwrap_singleton(sampling_config.repetition_penalty)
    presence = _unwrap_singleton(sampling_config.presence_penalty)
    frequency = _unwrap_singleton(sampling_config.frequency_penalty)
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
    """

    max_num_sequences: int
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

    # --- Occurrence workspace (allocateWorkspace counterpart), allocated lazily ---
    counts_cuda: torch.Tensor | None = None
    """int32[slots, vocab_size] or None; occurrence counts (see class docstring)."""
    presence_prefix_cuda: torch.Tensor | None = None
    """bool[slots, vocab_size] or None; ignored-prompt-prefix presence mask."""

    # Per-step request metadata, staged into persistent device buffers by
    # ``stage_request_metadata`` so the hot path does not allocate per step.
    request_offsets_cuda: torch.Tensor | None = None
    request_num_steps_cuda: torch.Tensor | None = None

    @classmethod
    def create(cls, *, max_num_sequences: int, device: torch.device) -> "PenaltyStore":
        """Allocate the vocab-independent buffers with their no-op defaults.

        ``inference_mode(False)`` guards every allocation in this class: the
        buffers persist across sampler steps and are mutated in place later, which
        inference-mode tensors forbid.
        """
        with torch.inference_mode(False):
            return cls(
                max_num_sequences=max_num_sequences,
                device=device,
                repetition_cuda=torch.ones(max_num_sequences, dtype=torch.float32, device=device),
                presence_cuda=torch.zeros(max_num_sequences, dtype=torch.float32, device=device),
                frequency_cuda=torch.zeros(max_num_sequences, dtype=torch.float32, device=device),
                active_cuda=torch.zeros(max_num_sequences, dtype=torch.bool, device=device),
                has_previous_token_cuda=torch.zeros(
                    max_num_sequences, dtype=torch.bool, device=device
                ),
            )

    def ensure_workspace(self, *, vocab_size: int, needs_prefix: bool) -> None:
        """Allocate the vocab-sized workspace on first use.

        Deferred because ``vocab_size`` is only known once logits arrive, mirroring
        ``PenaltyLayer::allocateWorkspace`` being gated on penalty usage. The prefix
        mask is allocated only if some request has used ``prompt_ignore_length``.
        """
        with torch.inference_mode(False):
            if self.counts_cuda is None:
                self.counts_cuda = torch.zeros(
                    (self.max_num_sequences, vocab_size),
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
        self, request_offsets_host: torch.Tensor, request_num_steps_host: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy this step's ``[R]`` request metadata into persistent device buffers.

        The host tensors are already pinned by the caller, so each step costs two
        small async H2D copies into a reused allocation rather than two fresh
        device tensors. Returned views are only valid until the next call.
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
        assert self.request_num_steps_cuda is not None
        offsets = self.request_offsets_cuda[:num_requests]
        num_steps = self.request_num_steps_cuda[:num_requests]
        offsets.copy_(request_offsets_host, non_blocking=True)
        num_steps.copy_(request_num_steps_host, non_blocking=True)
        return offsets, num_steps


class PenaltyHandler:
    """Applies the occurrence penalties: repetition, presence and frequency.

    These rescale or subtract from a token's logit based on how often it has already
    occurred, and run before the sampling strategy divides by temperature. Bans that
    force a logit to -inf (min_length, bad words, no-repeat-ngram) are a different
    kind of transform and live in ``TokenBanHandler``.

    The implementation follows the C++ ``batchApplyPenalty`` kernel
    (``cpp/tensorrt_llm/kernels/penaltyKernels.cu``) as driven by ``PenaltyLayer``.
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
        initialized: bool = False

    def __init__(
        self,
        *,
        max_num_sequences: int,
        device: torch.device | str,
    ):
        self._max_num_sequences = max_num_sequences
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
        self.store = PenaltyStore.create(max_num_sequences=max_num_sequences, device=self._device)

    @staticmethod
    def validate_request(request: LlmRequest) -> None:
        """Reject unsupported combinations for a penalized request.

        Called from ``TorchSampler.validate_request`` (request admission), so a
        violating request is failed individually instead of aborting the whole batch.
        """
        if _get_max_beam_width(request) > 1 and has_occurrence_penalty(request):
            raise ValueError(
                "TorchSampler does not support repetition, presence, or frequency "
                "penalties with beam search."
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
        if not (_get_max_beam_width(request) == 1 and has_occurrence_penalty(request)):
            self._slots[slot] = None
            if was_active:
                self._num_active_slots -= 1
            return

        sampling_config = request.sampling_config
        repetition = _unwrap_singleton(sampling_config.repetition_penalty)
        presence = _unwrap_singleton(sampling_config.presence_penalty)
        frequency = _unwrap_singleton(sampling_config.frequency_penalty)
        prompt_ignore_length = _unwrap_singleton(sampling_config.prompt_ignore_length)
        # min(prompt_ignore_length, inputLen), matching the C++ kernel.
        prompt_ignore_length = min(
            prompt_ignore_length if prompt_ignore_length is not None else 0,
            request.py_orig_prompt_len,
        )
        if prompt_ignore_length > 0:
            self._needs_prefix = True

        self._slots[slot] = self._SlotState(prompt_ignore_length=prompt_ignore_length)
        if not was_active:
            self._num_active_slots += 1

        self._new_slots.append(slot)
        self._new_repetition.append(repetition if repetition is not None else 1.0)
        self._new_presence.append(presence if presence is not None else 0.0)
        self._new_frequency.append(frequency if frequency is not None else 0.0)

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

        # Re-zero the workspace rows so a prior occupant's counts do not leak in.
        if store.counts_cuda is not None:
            store.counts_cuda.index_fill_(0, slots_cuda, 0)
        if store.presence_prefix_cuda is not None:
            store.presence_prefix_cuda.index_fill_(0, slots_cuda, False)

        self._new_slots.clear()
        self._new_repetition.clear()
        self._new_presence.clear()
        self._new_frequency.clear()

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
            torch.full_like(counted_tokens, slot),
            counted_tokens,
            torch.full_like(prefix_tokens, slot),
            prefix_tokens,
        )

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
        is_draft_batch: bool = False,
    ) -> None:
        """Apply the occurrence penalties to ``logits`` in place.

        ``logits`` is the packed generated-token logits ``[sum(num_steps * num_beams),
        vocab_size]``; request ``r`` owns ``request_num_steps[r]`` consecutive rows
        starting at ``request_offsets[r]``, in beam-major / step-minor order.
        ``request_offsets`` / ``request_num_steps`` are the caller's pinned host
        tensors and are staged to the device here.

        Args:
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

        request_offsets_cuda, request_num_steps_cuda = store.stage_request_metadata(
            request_offsets, request_num_steps
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
        )
        # Arm has_previous_token for the slots this call penalized (active, num_steps > 0)
        # so the next apply folds their sampled new_tokens. Done here rather than in the
        # compiled op because the op's fold reads the flag for every request row; flipping
        # it in the same graph would make the result depend on execution order within the
        # kernel.
        #
        # The scan is kept on the host deliberately. The same thing can be expressed on
        # device as active_cuda[seq_slots] & (num_steps > 0), avoiding this loop and the
        # H2D, but that costs several extra kernel launches and measured 5-7us slower for
        # batches up to 32 and no better at 64-256: the loop overlaps with the model
        # forward, the launches do not.
        pending_token_slots: list[int] = []
        for request, num_steps in zip(requests, request_num_steps.tolist()):
            slot = request.py_seq_slot
            if slot is None:
                continue
            if self._slots[slot] is not None and num_steps > 0:
                pending_token_slots.append(slot)
        if pending_token_slots:
            store.has_previous_token_cuda.index_fill_(
                0, self._to_device(pending_token_slots, torch.int64), True
            )
