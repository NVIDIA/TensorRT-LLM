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

"""Top-P Decay support for ``TorchSampler``.

The feature's per-slot runtime state and its whole lifecycle live in
:class:`TopPDecayHandler`; ``TorchSampler`` owns one instance and drives it
through the four lifecycle hooks documented on :class:`TopPDecayStore`.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._utils import prefer_pinned

from ..llm_request import LlmRequest, get_draft_token_length
from .ops.vanilla import Fusions, StrategyMetadata
from .sampler_common import _request_get_sampling_params, top_p_decay_active

__all__ = ["TopPDecayHandler", "TopPDecayStore"]


@dataclass(kw_only=True)
class TopPDecayMetadata(StrategyMetadata):
    """Per-group runtime top-p override for Top-P Decay (attached to the
    top-p-carrying groups -- top_p, top_k_top_p and min_p -- via the
    ``StrategyMetadata`` mechanism).

    ``slots`` maps each per-step group row to its sequence slot; the decayed
    per-row top-p is gathered on-device from the per-slot ``runtime_top_p``
    store, gated by ``is_decay_slot`` (non-decay rows keep their static top-p).
    Consumed by the TopP*/TopKTopP*/MinP* strategy impls in ``sample()``. See
    ``top_p_decay.TopPDecayStore`` for the feature-level semantics.
    """

    slots: torch.Tensor
    """Per-step group rows' sequence slots (int64, device)."""
    runtime_top_p: torch.Tensor
    """Per-slot runtime (decayed) top-p store (float32, device)."""
    is_decay_slot: torch.Tensor
    """Per-slot decay-active gate (bool, device)."""


@dataclass(kw_only=True)
class TopPDecayStore:
    """Per-slot runtime state for Top-P Decay -- the single source of truth
    for the feature's semantics on the torch path.

    Semantics (matching the legacy C++ ``computeToppDecay`` kernel): after
    every sampled token of a decay-active request::

        runtime_top_p = initial_top_p                                if token == reset_id
                      = max(runtime_top_p * top_p_decay, top_p_min)  otherwise

    A negative ``reset_ids`` sentinel (-1, "reset disabled") never matches,
    since sampled token ids are non-negative. Decay is active iff
    ``top_p_decay`` is set and < 1.0 (``SamplingParams.
    params_imply_top_p_decay_active``); an active decay forces a
    top-p-capable strategy even for an otherwise implicitly-greedy request
    (initial top-p defaults to 1.0), while explicit greedy controls win.
    Beam search and speculative draft tokens are rejected at admission
    (``TopPDecayHandler.validate_request``); parameter ranges are enforced by
    ``SamplingParams._validate`` / the executor::SamplingConfig constructor.

    Lifecycle per slot (each tensor has shape ``(max_num_sequences,)``):

    1. Admission (``TopPDecayHandler.setup_for_new_requests``): membership is
       cleared then re-set for the newly-admitted slots -- both the host-side
       ``TopPDecayHandler._slots`` set (an O(1) hot-path early-out)
       and its device mirror ``is_top_p_decay_slot_cuda`` (the gate the
       fused ops use, so the hot path needs no host-side filtering) --
       and the per-slot buffers are initialized. This clear-then-set also
       covers slot reuse: stale entries from a prior occupant are never
       consumed.
    2. Pre-sample (``TopPDecayHandler.build_metadata`` -> ``TopPDecayMetadata``
       -> ``TopPDecayMixin``): the per-row top-p fed to top_p /
       top_k_top_p / min_p sampling is overridden with the decayed runtime value
       for decay-active rows (fused gather, ``top_p_decay_gather``).
    3. Post-sample (``TopPDecayHandler.update_after_sample``): the recurrence
       above is applied in place for the sampled decay-active slots (fused
       update, ``top_p_decay_update``).
    4. Finish (``TopPDecayHandler.retire_slot``): the slot leaves the
       membership set so the early-outs re-arm; the device buffers need no
       cleanup (a freed slot is never sampled, reuse re-initializes it).
    """

    runtime_top_p_decay_cuda: torch.Tensor
    """The current (decaying) top-p per slot; mutated post-sample each step."""
    initial_top_p_decay_cuda: torch.Tensor
    """The initial top-p per slot; used to reset on a reset-id match."""
    top_p_decay_cuda: torch.Tensor
    """Per-slot multiplicative decay factor."""
    top_p_decay_min_cuda: torch.Tensor
    """Per-slot lower bound for the decayed top-p."""
    top_p_decay_reset_ids_cuda: torch.Tensor
    """Per-slot reset token id (< 0 never matches a sampled token)."""
    is_top_p_decay_slot_cuda: torch.Tensor
    """Per-slot bool gate (device mirror of ``TopPDecayHandler._slots``). Lets the
    fused post-sample update op filter decay-active slots on the GPU,
    avoiding a host-side ``.tolist()`` / set intersection each step."""

    @classmethod
    def create(cls, max_num_sequences: int) -> "TopPDecayStore":
        n = (max_num_sequences,)
        return cls(
            runtime_top_p_decay_cuda=torch.empty(n, dtype=torch.float32, device="cuda"),
            initial_top_p_decay_cuda=torch.empty(n, dtype=torch.float32, device="cuda"),
            top_p_decay_cuda=torch.empty(n, dtype=torch.float32, device="cuda"),
            top_p_decay_min_cuda=torch.empty(n, dtype=torch.float32, device="cuda"),
            top_p_decay_reset_ids_cuda=torch.empty(n, dtype=torch.int, device="cuda"),
            # The gate buffer IS the gate, so (unlike the others) it must start
            # False: a slot that was never admitted as decay-active must read
            # False.
            is_top_p_decay_slot_cuda=torch.zeros(n, dtype=torch.bool, device="cuda"),
        )


class TopPDecayHandler:
    """Owns the Top-P Decay store and drives the feature's lifecycle.

    The host-side membership set is the sole gate for reading/updating the
    per-slot :class:`TopPDecayStore` buffers, and doubles as an O(1) early-out
    on the hot path: with no decay-active request the pre/post-sample hooks
    return without touching the device.
    """

    def __init__(self, max_num_sequences: int):
        # Allocated for all sampler instances; only slots in self._slots are
        # ever read.
        self.store = TopPDecayStore.create(max_num_sequences)
        # Slots with an active top-p-decay request. Discarded on slot reuse so
        # stale buffer entries are never consumed.
        self._slots: set[int] = set()

    @property
    def active(self) -> bool:
        """Whether any resident request currently uses top-p decay."""
        return bool(self._slots)

    @staticmethod
    def validate_request(request: LlmRequest) -> None:
        """Reject unsupported combinations for a top-p-decay-active request.

        Top-p decay is supported only for single-token decode steps without beam
        search. Called from validate_request (request admission), so a violating
        request is failed individually instead of aborting the whole batch.
        """
        params = _request_get_sampling_params(request)
        # NB: value ranges need no re-check here. Every request enters through
        # the executor::SamplingConfig constructor, which hard-validates
        # top_p_decay in (0, 1], top_p_min in (0, 1] and top_p_reset_ids >= 0
        # (samplingConfig.cpp check* helpers) for all frontends. A reset id
        # >= vocab_size is not checked anywhere but is semantically inert: it
        # can never match a sampled token, i.e. it behaves as "reset disabled".
        if not top_p_decay_active(params):
            return
        if params.use_beam_search:
            raise ValueError("top_p_decay is not supported with beam search.")
        # A non-zero draft length means the request carries speculative draft
        # tokens and produces multiple tokens per step (req_num_steps =
        # 1 + draft_token_length). One-model speculation (vanilla MTP, one-model
        # Eagle3 / MTP-Eagle, SA, draft-target-one-model) uses its own
        # SpecSampler and never reaches TorchSampler; the
        # drafter-based modes that DO flow draft tokens through TorchSampler
        # (two-model draft-target, NGram, user-provided, two-model Eagle3 /
        # MTP-Eagle) are what can make this length non-zero. top-p decay does not
        # support these multi-token steps.
        # NB: at admission time the draft tokens of drafter-based modes are
        # usually not attached yet, so this check is best-effort. Two-model
        # speculation (the only source of such requests in TorchSampler) is
        # slated for removal, so in practice no speculative request reaches the
        # decay path; a debug assert in build_metadata guards the
        # invariant at sample time.
        if get_draft_token_length(request) > 0:
            raise ValueError(
                "top_p_decay is not supported for requests carrying speculative "
                "draft tokens (req_num_steps > 1). This covers the drafter-based "
                "modes routed through TorchSampler (two-model draft-target, NGram, "
                "user-provided, two-model Eagle3 / MTP-Eagle); one-model "
                "speculation uses its own sampler and is unaffected."
            )

    def setup_for_new_requests(
        self,
        new_requests: list[LlmRequest],
        *,
        new_seq_slots_cuda_long: torch.Tensor,
    ) -> None:
        """Refresh top-p-decay membership and per-slot buffers for admitted requests
        (lifecycle step 1, see :class:`TopPDecayStore`).

        Drops stale membership from prior occupants of the newly-admitted slots
        (host set and device gate), then re-admits the decay-active requests and
        initializes their per-slot store entries. Unsupported decay combinations
        were already rejected per-request in validate_request at admission.
        """
        # Clear the device decay gate for every newly-admitted slot (covers slot
        # reuse: a slot previously decay-active but reused by a non-decay request
        # must read False). Decay-active slots are then set True below.
        decay_gate = self.store.is_top_p_decay_slot_cuda
        decay_gate.index_fill_(0, new_seq_slots_cuda_long, False)

        decay_seq_slots: list[int] = []
        initial_top_p: list[float] = []
        top_p_decay: list[float] = []
        top_p_min: list[float] = []
        top_p_reset_ids: list[int] = []
        for request in new_requests:
            slot = request.py_seq_slot
            assert slot is not None
            self._slots.discard(slot)
            sampling_params = _request_get_sampling_params(request)
            if not top_p_decay_active(sampling_params):
                continue
            self._slots.add(slot)
            decay_seq_slots.append(slot)
            # Initial runtime top-p defaults to 1.0 when top_p is unset.
            initial_top_p.append(
                sampling_params.top_p if sampling_params.top_p is not None else 1.0
            )
            # decay is guaranteed non-None and < 1.0 here (top_p_decay_active);
            # min/reset fall back to the C++ runtime defaults when unset.
            assert sampling_params.top_p_decay is not None
            top_p_decay.append(sampling_params.top_p_decay)
            top_p_min.append(
                sampling_params.top_p_min if sampling_params.top_p_min is not None else 1e-6
            )
            top_p_reset_ids.append(
                sampling_params.top_p_reset_ids
                if sampling_params.top_p_reset_ids is not None
                else -1
            )

        if decay_seq_slots:
            self._update_store_for_new_requests(
                decay_seq_slots=decay_seq_slots,
                initial_top_p=initial_top_p,
                top_p_decay=top_p_decay,
                top_p_min=top_p_min,
                top_p_reset_ids=top_p_reset_ids,
            )

    def _update_store_for_new_requests(
        self,
        *,
        decay_seq_slots: list[int],
        initial_top_p: list[float],
        top_p_decay: list[float],
        top_p_min: list[float],
        top_p_reset_ids: list[int],
    ) -> None:
        """Initialize per-slot Top-P Decay buffers for newly admitted decay requests.

        runtime_top_p and initial_top_p both start at the effective initial top-p;
        the runtime value is decayed post-sample each step.
        """
        store = self.store
        device = store.runtime_top_p_decay_cuda.device
        slots_cuda = torch.tensor(
            decay_seq_slots, device="cpu", dtype=torch.int64, pin_memory=prefer_pinned()
        ).to(device, non_blocking=True)
        floats_host = torch.tensor(
            [initial_top_p, top_p_decay, top_p_min],
            device="cpu",
            dtype=torch.float32,
            pin_memory=prefer_pinned(),
        )
        floats_cuda = floats_host.to(device, non_blocking=True)
        initial_cuda = floats_cuda[0]
        reset_ids_cuda = torch.tensor(
            top_p_reset_ids, device="cpu", dtype=torch.int32, pin_memory=prefer_pinned()
        ).to(device, non_blocking=True)

        store.runtime_top_p_decay_cuda.index_copy_(0, slots_cuda, initial_cuda)
        store.initial_top_p_decay_cuda.index_copy_(0, slots_cuda, initial_cuda)
        store.top_p_decay_cuda.index_copy_(0, slots_cuda, floats_cuda[1])
        store.top_p_decay_min_cuda.index_copy_(0, slots_cuda, floats_cuda[2])
        store.top_p_decay_reset_ids_cuda.index_copy_(0, slots_cuda, reset_ids_cuda)
        # Enable the device gate for these decay-active slots (cleared for all new
        # slots in setup_sampler_step just before this call).
        store.is_top_p_decay_slot_cuda.index_fill_(0, slots_cuda, True)

    def build_metadata(
        self,
        *,
        group_req_indices: torch.Tensor,
        req_num_steps: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_slots_cuda: torch.Tensor,
    ) -> Optional[TopPDecayMetadata]:
        """Build the Top-P Decay metadata for a top_p / top_k_top_p / min_p group.

        Lifecycle step 2, see :class:`TopPDecayStore`. Returns None when no request
        currently uses decay. The metadata's ``slots`` tensor is aligned to the
        group's per-STEP row order (matching group_strategies_per_step);
        non-decay rows (possibly multi-step draft rows) are gated out on-device
        by ``is_decay_slot``, so decay presence in the group is not checked
        host-side: a group without decay rows samples every row with its static
        top-p -- same result as returning None.
        """
        if not self._slots:
            return None
        store = self.store
        # Fast path (steady-state decoding): if every row in the group is
        # single-token, the per-STEP row order equals the per-request order, and
        # (group_req_indices being sorted ascending) a contiguous group's slots
        # are just a slice of seq_slots_cuda -- no host layout build and no H2D
        # copy.
        first_req = int(group_req_indices[0].item())
        last_req = int(group_req_indices[-1].item())
        group_steps = req_num_steps[group_req_indices]
        if last_req - first_req + 1 == group_req_indices.size(0) and (
            group_steps.max().item() == 1
        ):
            per_step_slots_cuda = seq_slots_cuda[first_req : last_req + 1]
        else:
            # Build the per-STEP slot layout (each request contributes
            # req_num_steps rows).
            group_seq_slots = seq_slots[group_req_indices]
            if __debug__:
                # Internal invariant (stripped under python -O): a decay-active
                # row is always single-token -- the only source of multi-step
                # rows in TorchSampler is two-model speculation (slated for
                # removal), and decay + draft tokens is rejected per-request at
                # admission in validate_request.
                decay_row_steps = group_steps[
                    torch.isin(group_seq_slots, torch.tensor(list(self._slots)))
                ]
                assert decay_row_steps.numel() == 0 or decay_row_steps.max().item() == 1, (
                    "top_p_decay row with req_num_steps != 1; decay + draft tokens "
                    "should have been rejected at admission"
                )
            per_step_slots_cuda = torch.repeat_interleave(group_seq_slots.long(), group_steps).to(
                seq_slots_cuda.device, non_blocking=True
            )
        return TopPDecayMetadata(
            slots=per_step_slots_cuda,
            runtime_top_p=store.runtime_top_p_decay_cuda,
            is_decay_slot=store.is_top_p_decay_slot_cuda,
        )

    def update_after_sample(
        self,
        *,
        step_tokens: torch.Tensor,
        sampled_slots_cuda: torch.Tensor,
    ) -> None:
        """Apply the post-sample decay recurrence for sampled decay-active slots.

        See :class:`TopPDecayStore` for the feature-level semantics (lifecycle
        step 3). Restricting to the sampled slots avoids reading stale
        new_tokens_cuda for slots that were not scheduled this iteration.

        ``step_tokens`` holds the token sampled this step per slot; decay is
        single-token-only, so the caller passes local step 0, beam 0.
        """
        # Host-side O(1) early-out: skip the kernel launch when no request uses
        # decay; otherwise a single fused (torch.compile) op gates on
        # is_decay_slot on-device and gathers the sampled token in place.
        if not self._slots:
            return
        store = self.store
        Fusions.top_p_decay_update(
            runtime_top_p=store.runtime_top_p_decay_cuda,
            initial_top_p=store.initial_top_p_decay_cuda,
            top_p_decay=store.top_p_decay_cuda,
            top_p_min=store.top_p_decay_min_cuda,
            reset_ids=store.top_p_decay_reset_ids_cuda,
            is_decay_slot=store.is_top_p_decay_slot_cuda,
            step_tokens=step_tokens,
            sampled_slots=sampled_slots_cuda,
        )

    def retire_slot(self, req: LlmRequest) -> None:
        """Retire a finished request's slot from the top-p-decay membership set
        (lifecycle step 4, see :class:`TopPDecayStore`), so the O(1) hot-path
        early-outs re-arm once decay traffic drains.

        Callers ensure ``req`` has finished. Requests that finish outside the
        sampler (e.g. cancellation) are covered by the slot-reuse cleanup at
        admission instead.
        """
        if self._slots and req.py_seq_slot is not None:
            self._slots.discard(req.py_seq_slot)
