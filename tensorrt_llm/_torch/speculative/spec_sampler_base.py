# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base class for speculative decoding samplers.

This module provides a common base class for MTPSampler, SASampler, and
Eagle3OneModelSampler.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    AsyncWorkerMixin,
    Sampler,
    SampleState,
    SampleStateTensors,
    TorchSampler,
    add_token,
    int_tensor,
)
from ..pyexecutor.scheduler import ScheduledRequests


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor
    #: cap-accept only; None on every other path.
    cap_trim_lens: Optional[torch.Tensor] = None
    #: ragged verification only; None on every other path. The TOKEN windows
    #: the step actually verified (bonus included), slot-indexed. Under
    #: device-window selection this D2H is the host's only source of the true
    #: windows; zero marks "no window this step".
    verify_lens: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class SampleStateSpec(SampleState):
    """Sample state for speculative decoding."""

    device: SampleStateTensorsSpec
    host: SampleStateTensorsSpec
    #: This step's per-request verify windows, keyed by request id. Snapshotted
    #: because ``update_requests`` for step N-1 runs *after* the scheduler has
    #: already stamped step N's windows onto the same LlmRequest objects, so
    #: reading ``request.py_verify_len`` live rewinds by the wrong amount under
    #: the overlap scheduler. None on every uniform path.
    verify_lens_snapshot: Optional[dict] = None
    #: Same idea, for ``cap-accept``: the window the scheduler chose, which
    #: bounds *commitment* only -- the target still scored the full block, so
    #: the KV rewind must use the full block and not this. Kept in a separate
    #: field from ``verify_lens_snapshot`` for exactly that reason: the two
    #: never coexist, and conflating them would rewind by the window after
    #: verifying the block, silently leaking KV.
    verify_caps_snapshot: Optional[dict] = None
    #: The draft pass that produced the block THIS state's tokens verified;
    #: the STS recorder joins its acceptance label to a logits snapshot by
    #: this key. Snapshotted at sampling time for the same reason
    #: ``verify_lens_snapshot`` is: by the time ``update_requests`` runs, the
    #: live counter already belongs to a later pass. None unless STS
    #: collection is on.
    sts_target_seq: Optional[int] = None


class SpecSamplerBase(Sampler[SampleStateSpec], AsyncWorkerMixin):
    """
    Base class for speculative decoding samplers (MTP, NGram, Eagle3, SA).

    Provides common functionality:
    - Pre-allocated GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    Subclasses can customize behavior by overriding:
    - _get_max_tokens(): How to calculate max_tokens for storage
    - _get_draft_tokens_storage_size(): Size of next_draft_tokens tensor
    - _add_dummy_draft_tokens(): Whether to add dummy drafts for context requests
    """

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors."""

        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor
        #: cap-accept: per-request positions the verify window discarded.
        #: Slot-indexed like new_tokens_lens, and rewritten for every slot in
        #: the batch each step -- a slot left alone would hand the previous
        #: occupant's loss to whoever holds it now.
        cap_trim_lens: torch.Tensor
        #: ragged verification: the TOKEN windows the step actually verified,
        #: slot-indexed; zeroed for every batch slot on non-ragged steps (the
        #: same stale-slot discipline as cap_trim_lens).
        verify_lens: torch.Tensor

    def __init__(self, args: TorchSampler.Args, *, draft_len: int):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            draft_len: Maximum number of draft tokens per iteration.
        """
        self._async_worker_init(args.enable_async_worker)
        self.mapping = None
        self.draft_len = draft_len
        self.max_seq_len = args.max_seq_len

        seq_slots = args.max_num_sequences
        max_tokens = self._get_max_tokens(args, draft_len)
        max_new_tokens = self._get_max_new_tokens(args, draft_len)
        draft_tokens_size = self._get_draft_tokens_storage_size(args, draft_len)
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"

        self.store = self.Store(
            new_tokens=int_tensor((max_new_tokens, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_draft_tokens=int_tensor((seq_slots, draft_tokens_size)),
            new_tokens_lens=int_tensor((seq_slots,)),
            cap_trim_lens=int_tensor((seq_slots,)),
            verify_lens=int_tensor((seq_slots,)),
        )

    def _get_max_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate max_tokens for storage allocation.

        Override in subclasses if needed. Default: draft_len + 1.
        MTP uses args.max_total_draft_tokens + 1 for tree-based speculation.
        """
        return draft_len + 1

    def _get_max_new_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """Max depth of accepted token path for new_tokens buffer.

        Defaults to _get_max_tokens (same size as next_new_tokens).
        Override when accepted path depth differs from total draft tokens,
        e.g. dynamic tree where max_draft_len < max_total_draft_tokens.
        """
        return self._get_max_tokens(args, draft_len)

    def _get_draft_tokens_storage_size(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate storage size for next_draft_tokens tensor.

        Override in subclasses if needed. Default: draft_len.
        MTP uses args.max_total_draft_tokens for tree-based speculation.
        """
        return draft_len

    def _add_dummy_draft_tokens(self) -> bool:
        """
        Whether to add dummy draft tokens for context requests.

        Override in subclasses. Default: True (needed for KV cache preparation).
        """
        return True

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        runtime_draft_len: Optional[int],
    ) -> None:
        """Common handling for both context and generation requests."""
        if request.py_return_context_logits:
            logger.warning(
                "return_context_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_generation_logits:
            logger.warning(
                "return_generation_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_log_probs:
            logger.warning(
                "return_log_probs not supported with speculative decoding, skipping for request %s",
                request.py_request_id,
            )
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

    #: Optional observability sink, attached by the DSpark worker. Left None
    #: everywhere else so this costs one attribute read per request.
    acceptance_stats = None
    #: STS calibration collection; attached alongside `acceptance_stats` and
    #: None everywhere else. `sts_row_for` resolves a py_request_id to its row
    #: in the worker's confidence buffer through the worker's OWN allocator --
    #: `py_seq_slot` comes from a different allocator whose numbering only
    #: coincides until the first request completes. `sts_seq_provider` names
    #: the draft pass the current step's verification targets.
    sts_recorder = None
    sts_row_for = None
    sts_seq_provider = None

    @staticmethod
    def _verified_len(request: LlmRequest, runtime_draft_len: int,
                      verify_lens_snapshot: Optional[dict] = None,
                      ridden_verify_lens: Optional[list] = None) -> int:
        """How many draft positions THIS request was given to verify.

        Uniform scheduling gives every request the batch-wide
        ``runtime_draft_len``; ragged (per-request) verification does not, and
        the KV rewind below is computed per request:

            rewind = verified_positions - accepted_positions

        Using the batch-wide value under ragged verification rewinds the wrong
        amount for every request that got a shorter window -- silently dropping
        or keeping KV entries, i.e. wrong output with no error anywhere.

        ``py_verify_len`` is set only by the ragged path, so this is exactly
        today's behavior for every other speculation mode.
        """
        # First preference: the windows the device layout actually verified,
        # ridden back on this step's sampler D2H (slot-indexed TOKEN windows;
        # zero = no window). Under device-window selection the host attribute
        # holds only a shape split, so this is the only correct source; under
        # host windows the two agree.
        if ridden_verify_lens is not None:
            token_window = ridden_verify_lens[request.py_seq_slot]
            if token_window > 0:
                return int(token_window) - 1
        # Second: the snapshot taken when this step's tokens were sampled. The
        # live attribute already belongs to the NEXT step by the time the
        # overlap scheduler rewinds this one.
        if verify_lens_snapshot is not None:
            per_request = verify_lens_snapshot.get(request.py_request_id)
        else:
            per_request = getattr(request, "py_verify_len", None)
        return runtime_draft_len if per_request is None else int(per_request)

    def update_requests(
        self,
        state: SampleStateSpec,
        resource_manager: Optional[BaseResourceManager] = None,
    ) -> None:
        """
        CPU-side request updates after GPU->CPU sync.

        Waits for async copy to complete, then updates request state with:
        - Accepted tokens
        - Stop criteria checks
        - Next iteration draft tokens
        """
        assert isinstance(state, SampleStateSpec)

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        cap_trim_list = (state.host.cap_trim_lens.tolist()
                         if state.host.cap_trim_lens is not None else None)
        ridden_verify_lens = (state.host.verify_lens.tolist()
                              if state.host.verify_lens is not None else None)
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        caps = state.verify_caps_snapshot
        for req in state.requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            # cap-accept: `num_new_tokens` has ALREADY been capped, on device,
            # inside acceptance (`interface.apply_accept_caps`) -- it has to be,
            # because the drafter reads the same count later in that forward to
            # advance its own state. The cap is read here only to report the
            # window the request was actually given.
            cap = None if caps is None else caps.get(req.py_request_id)
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            verified_len = self._verified_len(req, runtime_draft_len,
                                              state.verify_lens_snapshot,
                                              ridden_verify_lens)
            req.py_rewind_len = verified_len - req.py_num_accepted_draft_tokens
            # Acceptance against the window the request was actually given.
            # Recorded here because this is the only place both numbers are
            # host-side and belong to the same step -- the live py_verify_len
            # has already moved on by now, which is why the snapshot exists.
            # `acceptance_stats` is attached by the DSpark worker and is None
            # for every other path.
            #
            # Under cap-accept the window is the cap, NOT `verified_len`: the
            # target verified the whole block (which is what makes the rewind
            # above correct), but the scheduler only granted `cap`. Reporting
            # the block here would count every request as untrimmed and hide
            # the mode's entire measurement.
            if self.acceptance_stats is not None:
                self.acceptance_stats.record_acceptance(
                    accepted=req.py_num_accepted_draft_tokens,
                    window=verified_len if cap is None else int(cap),
                    cap_trim=(0 if cap_trim_list is None else
                              cap_trim_list[req.py_seq_slot]))
            # STS calibration pairs this request's drafted-block logits with
            # how much of that block was accepted. The logits come from the
            # recorder's snapshot ring, selected by the draft pass snapshotted
            # into this state at sampling time and checked against the row's
            # own stamp -- reading the live buffer here paired the label with
            # whatever a LATER pass had written over it.
            if self.sts_recorder is not None:
                self.sts_recorder.record(
                    row=(self.sts_row_for(req.py_request_id)
                         if self.sts_row_for is not None else None),
                    accepted=req.py_num_accepted_draft_tokens,
                    target_seq=state.sts_target_seq)
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)

        if self.sts_recorder is not None:
            # Periodic flush, so a run killed mid-collection still leaves usable
            # shards rather than everything living in a list that dies with it.
            self.sts_recorder.end_step()

    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict[str, torch.Tensor],
        num_context_logits_prefix_sum: list[int],
    ) -> SampleStateSpec:
        """
        Async sampling - schedules GPU->CPU copy.
        Called after CUDA graph replay.

        Args:
            scheduled_requests: Batch of scheduled requests
            outputs: Dict from worker forward() containing:
                - new_tokens: [batch, max_draft_len + 1] accepted tokens
                - new_tokens_lens: [batch] number of accepted tokens
                - next_draft_tokens: [batch, max_draft_len] draft tokens for next iter
                - next_new_tokens: [batch, max_draft_len + 1] input for next iter
            num_context_logits_prefix_sum: Prefix sum of context logits (unused)

        Returns:
            SampleStateSpec with device and host tensors
        """
        num_skip = len(scheduled_requests.context_requests_chunking)
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests
        num_sampling_requests = len(sampling_requests)

        slots = torch.as_tensor([r.py_seq_slot for r in sampling_requests], dtype=torch.long)
        slots = slots.to(device="cuda", non_blocking=True)

        o_new_tokens = outputs["new_tokens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = outputs["new_tokens_lens"][num_skip : num_skip + num_sampling_requests]
        o_next_draft_tokens = outputs["next_draft_tokens"][
            num_skip : num_skip + num_sampling_requests
        ]
        o_next_new_tokens = outputs["next_new_tokens"][num_skip : num_skip + num_sampling_requests]
        runtime_draft_len = o_next_draft_tokens.shape[1]

        # Pad or truncate to match fixed-size store buffers for index_copy_.
        # Use actual store buffer dimensions (which may differ from draft_len
        # when _get_max_new_tokens is overridden, e.g. dynamic tree mode).
        new_tokens_width = self.store.new_tokens.shape[0]
        next_new_tokens_width = self.store.next_new_tokens.shape[0]
        draft_tokens_width = self.store.next_draft_tokens.shape[1]
        if o_new_tokens.shape[1] < new_tokens_width:
            o_new_tokens = torch.nn.functional.pad(
                o_new_tokens, (0, new_tokens_width - o_new_tokens.shape[1])
            )
        elif o_new_tokens.shape[1] > new_tokens_width:
            o_new_tokens = o_new_tokens[:, :new_tokens_width]
        if o_next_draft_tokens.shape[1] < draft_tokens_width:
            o_next_draft_tokens = torch.nn.functional.pad(
                o_next_draft_tokens, (0, draft_tokens_width - o_next_draft_tokens.shape[1])
            )
        elif o_next_draft_tokens.shape[1] > draft_tokens_width:
            o_next_draft_tokens = o_next_draft_tokens[:, :draft_tokens_width]
        if o_next_new_tokens.shape[1] < next_new_tokens_width:
            o_next_new_tokens = torch.nn.functional.pad(
                o_next_new_tokens, (0, next_new_tokens_width - o_next_new_tokens.shape[1])
            )
        elif o_next_new_tokens.shape[1] > next_new_tokens_width:
            o_next_new_tokens = o_next_new_tokens[:, :next_new_tokens_width]

        # Use index_copy_ for efficient copying (slots are unique)
        self.store.new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new_tokens)
        self.store.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next_new_tokens)
        self.store.new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
        self.store.next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)
        # cap-accept: absent on every other path, and written as ZEROS then
        # rather than skipped. The buffer is persistent and slot-indexed, so a
        # slot left untouched would keep whatever the previous occupant lost
        # and hand it to the current one.
        o_cap_trim = outputs.get("cap_trim_lens")
        if o_cap_trim is None:
            o_cap_trim = torch.zeros_like(o_new_tokens_lens)
        else:
            o_cap_trim = o_cap_trim[num_skip:num_skip + num_sampling_requests]
        self.store.cap_trim_lens.index_copy_(
            0, slots, o_cap_trim.to(self.store.cap_trim_lens.dtype))
        # Ragged verification: the true token windows, same zero-then-write
        # slot discipline as cap_trim_lens. Under device-window selection the
        # host py_verify_len holds only a shape split, so the rewind
        # arithmetic in update_requests must read THIS.
        o_verify_lens = outputs.get("verify_lens")
        if o_verify_lens is None:
            o_verify_lens = torch.zeros_like(o_new_tokens_lens)
        else:
            o_verify_lens = o_verify_lens[num_skip:num_skip +
                                          num_sampling_requests]
        self.store.verify_lens.index_copy_(
            0, slots, o_verify_lens.to(self.store.verify_lens.dtype))

        # Create sample state with async D2H copy
        device_tensors = SampleStateTensorsSpec(
            new_tokens=self.store.next_new_tokens,
            new_tokens_lens=self.store.new_tokens_lens,
            next_draft_tokens=self.store.next_draft_tokens,
        )

        host_tensors = SampleStateTensorsSpec(
            new_tokens=self._copy_to_host(self.store.new_tokens),
            new_tokens_lens=self._copy_to_host(self.store.new_tokens_lens),
            next_draft_tokens=self._copy_to_host(self.store.next_draft_tokens),
            # [max_num_sequences] int32 -- kilobytes next to new_tokens above,
            # and it rides the same event, so no extra synchronization.
            cap_trim_lens=self._copy_to_host(self.store.cap_trim_lens),
            verify_lens=self._copy_to_host(self.store.verify_lens),
        )
        sampler_event = self._record_sampler_event()

        # Add dummy draft tokens to context requests for KV cache preparation
        if self._add_dummy_draft_tokens():
            for request in finished_context_requests:
                request.py_draft_tokens = [1] * self.draft_len

        # Cheap (a few ints) and only populated when the ragged path is live.
        verify_lens_snapshot = None
        verify_caps_snapshot = None
        for request in sampling_requests:
            window = getattr(request, "py_verify_len", None)
            if window is not None:
                if verify_lens_snapshot is None:
                    verify_lens_snapshot = {}
                verify_lens_snapshot[request.py_request_id] = int(window)
            cap = getattr(request, "py_verify_cap", None)
            if cap is not None:
                if verify_caps_snapshot is None:
                    verify_caps_snapshot = {}
                verify_caps_snapshot[request.py_request_id] = int(cap)

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
            verify_lens_snapshot=verify_lens_snapshot,
            verify_caps_snapshot=verify_caps_snapshot,
            sts_target_seq=(self.sts_seq_provider()
                            if self.sts_seq_provider is not None else None),
        )
