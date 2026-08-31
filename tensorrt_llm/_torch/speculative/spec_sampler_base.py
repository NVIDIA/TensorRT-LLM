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
Sampler for one-model speculative decoding.

Every one-model speculative mode (MTP, Eagle3, SA, DraftTarget, PARD, DFlash,
DSpark) shares a single sampler: the worker's fused kernel already performs
drafting, target verification and acceptance, so the sampler only scatters that
output into slot-indexed buffers, starts the async D2H copy, and updates
requests host-side. Buffer shapes derive entirely from ``TorchSampler.Args``.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState, get_draft_token_length
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
from ..pyexecutor.sampler.penalties import has_occurrence_penalty
from ..pyexecutor.sampler.sampler_common import _request_get_sampling_params, top_p_decay_active
from ..pyexecutor.sampler.sampler_features import handle_stop_criteria
from ..pyexecutor.scheduler import ScheduledRequests


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor


@dataclass(kw_only=True)
class SampleStateSpec(SampleState):
    """Sample state for speculative decoding."""

    device: SampleStateTensorsSpec
    host: SampleStateTensorsSpec
    # Per-request draft-token counts of the step this state samples, captured
    # in sample_async before dummy draft tokens are added (index-aligned with
    # `requests`; 0 for finished-context requests). update_requests pairs
    # them with py_num_accepted_draft_tokens: reading py_draft_tokens there
    # instead would see the NEXT step's buffer, which update_requests itself
    # installs.
    draft_lens: Optional[list[int]] = None
    # Per-request verify windows for this sampler step, keyed by request id.
    # The overlap scheduler can stamp the next step on the live request before
    # this state is consumed, so rewind accounting must use this snapshot.
    verify_lens_snapshot: Optional[dict[int, int]] = None


class SpecSampler(Sampler[SampleStateSpec], AsyncWorkerMixin):
    """
    Sampler for all one-model speculative decoding modes.

    Provides:
    - Pre-allocated, slot-indexed GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    This class carries no per-mode behavior. ``args.max_total_draft_tokens`` is
    ``spec_config.tokens_per_gen_step - 1`` (see ``create_torch_sampler_args``),
    i.e. the target's per-step input width minus one, which is exactly the
    draft length every mode used to compute for itself.
    """

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    def validate_request(self, request: LlmRequest) -> None:
        """Reject sampling parameters the one-model speculative path cannot honor.

        The one-model sampling kernels take only temperature/top_k/top_p (see
        SpecMetadata.populate_sampling_params_for_one_model); min_p has no
        buffer there, so it would be silently dropped and the request would
        decode from a different distribution than the user asked for. Threading
        it through costs measurable throughput on the rejection path, so reject
        instead. This sampler also does not return context logits, generation
        logits, or log probabilities. Raised from validate_request (request
        admission), so only the offending request fails rather than the whole
        executor step.
        """
        requested_outputs = (
            ("return_context_logits / prompt_logprobs", request.py_return_context_logits),
            ("return_generation_logits", request.py_return_generation_logits),
            ("logprobs", request.py_return_log_probs),
        )
        unsupported_outputs = [name for name, requested in requested_outputs if requested]
        if unsupported_outputs:
            raise ValueError(
                "The following output options are not supported with speculative decoding: "
                f"{', '.join(unsupported_outputs)}. Drop these options from "
                "the request, or disable speculative decoding."
            )

        sampling_config = request.sampling_config
        if sampling_config is None:
            return
        # min_p lives on the C++ SamplingConfig as an optional scalar.
        min_p = sampling_config.min_p
        if min_p and min_p > 0.0:
            raise ValueError(
                "min_p is not supported with one-model speculative decoding. "
                "Drop min_p from the request, or disable speculative decoding."
            )
        self._validate_unsupported_logits_processors(request)
        # The occurrence penalties need a [slots, vocab_size] workspace that is only
        # allocated when the deploy opted in, so a request asking for them while the
        # flag is off cannot be honored. Reject instead of silently decoding from an
        # unpenalized distribution -- same reasoning as min_p above.
        if not has_occurrence_penalty(request):
            return
        if not self._enable_penalty:
            raise ValueError(
                "repetition_penalty / presence_penalty / frequency_penalty require "
                "'enable_penalty: true' in the speculative decoding config when using "
                "one-model speculative decoding. Enable that flag, drop the penalties "
                "from the request, or disable speculative decoding."
            )
        # Tree speculation lays each request's logits out as tree nodes, where a
        # row's history is its root path rather than the rows before it. Applying
        # the linear mapping there would let sibling branches penalize each other,
        # so reject until tree-aware prefixes are implemented.
        if not self._penalty_supported:
            raise ValueError(
                "repetition_penalty / presence_penalty / frequency_penalty are not "
                "supported with tree speculative decoding (eagle_choices / dynamic "
                "tree) yet. Drop the penalties, use a linear speculation mode, or "
                "disable speculative decoding."
            )

    @staticmethod
    def _validate_unsupported_logits_processors(request: LlmRequest) -> None:
        """Reject logits-side sampling features the one-model path cannot apply.

        TorchSampler implements min_length, bad_words, no_repeat_ngram_size,
        embedding_bias and top_p_decay by editing the target logits before
        sampling. The one-model path has no logits-editing hook, so each is
        rejected here instead of being silently dropped.

        Each check gates on a non-neutral value rather than on presence, since a
        frontend may forward a default explicitly.
        """
        # py_min_length mirrors the C++ SamplingConfig field, i.e. an optional
        # scalar. The OpenAI frontend always forwards min_tokens (default 0), so
        # it is routinely present and holds the neutral value -- gate on the
        # value, not on its presence.
        min_length = getattr(request, "py_min_length", None)
        if min_length and min_length > 0:
            raise ValueError(
                "min_length is not supported with one-model speculative decoding. "
                "Drop min_length from the request, or disable speculative decoding."
            )
        if getattr(request, "py_bad_words", None):
            raise ValueError(
                "bad_words is not supported with one-model speculative decoding. "
                "Drop bad_words from the request, or disable speculative decoding."
            )
        if getattr(request, "py_no_repeat_ngram_size", None):
            raise ValueError(
                "no_repeat_ngram_size is not supported with one-model speculative "
                "decoding. Drop no_repeat_ngram_size from the request, or disable "
                "speculative decoding."
            )
        if getattr(request, "py_embedding_bias", None) is not None:
            raise ValueError(
                "embedding_bias is not supported with one-model speculative decoding. "
                "Drop embedding_bias from the request, or disable speculative decoding."
            )
        # Reuse the handler's own predicate so "active" cannot drift between paths.
        if top_p_decay_active(_request_get_sampling_params(request)):
            raise ValueError(
                "top_p_decay is not supported with one-model speculative decoding. "
                "Drop top_p_decay / top_p_min from the request, or disable "
                "speculative decoding."
            )

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors."""

        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor

    def __init__(
        self,
        args: TorchSampler.Args,
        *,
        accepted_path_len: Optional[int] = None,
        enable_penalty: bool = False,
        penalty_supported: bool = True,
    ):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            accepted_path_len: Upper bound on the number of tokens a single step
                can accept, used to size new_tokens. Defaults to
                ``args.max_draft_len + 1``; see the store comment below for the
                one mode that has to override it.
            enable_penalty: whether the deploy enabled the occurrence penalties.
                Only used to decide whether a request asking for them is admitted;
                the penalties themselves are applied inside the worker.
            penalty_supported: whether this speculation mode's row layout is one
                the penalties can map (linear modes yes, tree modes not yet).
        """
        self._enable_penalty = enable_penalty
        self._penalty_supported = penalty_supported
        self._async_worker_init(args.enable_async_worker)
        self.mapping = None
        self.max_seq_len = args.max_seq_len
        # Wire width minus one: the number of draft slots the target verifies
        # per step. Linear modes set max_total_draft_tokens == max_draft_len;
        # tree modes set it to the total node count; PARD sets it to 2K-1
        # because it also feeds mask tokens through the target.
        self.draft_len = args.max_total_draft_tokens

        seq_slots = args.max_num_sequences
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"

        # new_tokens holds the accepted tokens only, so it is sized to how many
        # a step can accept rather than to the wire width. Normally that is
        # max_draft_len + 1: the drafter advances max_draft_len times, and the
        # golden token the target always accepts adds one. Verified against
        # Eagle3 dynamic tree (K=6, T=60), MTP dynamic tree, PARD (T=2K-1) and
        # the linear modes -- none exceed it.
        #
        # The exception is the deprecated eagle_choices static tree. There the
        # one-model drafter ignores the tree and runs _forward_draft_loop, a
        # linear loop over runtime_draft_len == max_total_draft_tokens, so a
        # step can accept up to max_total_draft_tokens + 1 tokens even though
        # max_draft_len only describes the depth of a tree that is never built.
        # (Tree-aware acceptance lives in TorchSampler, i.e. the two-model
        # path.) get_spec_decoder passes the wire width for that mode; both it
        # and this workaround go away with the feature in release 1.4.
        self.max_accepted_path_len = (
            accepted_path_len if accepted_path_len is not None else args.max_draft_len + 1
        )
        self.store = self.Store(
            new_tokens=int_tensor((self.max_accepted_path_len, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor(
                (args.max_total_draft_tokens + 1, seq_slots, self.max_beam_width)
            ),
            next_draft_tokens=int_tensor((seq_slots, args.max_total_draft_tokens)),
            new_tokens_lens=int_tensor((seq_slots,)),
        )

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        runtime_draft_len: Optional[int],
    ) -> None:
        """Common handling for both context and generation requests."""
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

    @staticmethod
    def _verified_len(
        request: LlmRequest,
        runtime_draft_len: int,
        verify_lens_snapshot: Optional[dict[int, int]],
    ) -> int:
        """Return the number of draft positions verified for one request."""
        if verify_lens_snapshot is not None:
            verify_len = verify_lens_snapshot.get(request.py_request_id)
        else:
            verify_len = getattr(request, "py_verify_len", None)
        return runtime_draft_len if verify_len is None else int(verify_len)

    @staticmethod
    def _snapshot_verify_lens(
        requests: list[LlmRequest],
    ) -> Optional[dict[int, int]]:
        """Capture overlap-sensitive ragged windows for one sampler step."""
        snapshot = {
            request.py_request_id: int(verify_len)
            for request in requests
            if (verify_len := getattr(request, "py_verify_len", None)) is not None
        }
        return snapshot or None

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
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            # new_tokens is sized to this bound, and add_token indexes a plain
            # host-side list, so a violation would otherwise surface as an
            # opaque IndexError.
            assert num_new_tokens <= self.max_accepted_path_len, (
                f"accepted {num_new_tokens} tokens in one step, but new_tokens is "
                f"sized for {self.max_accepted_path_len}"
            )
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            # Pair the acceptance count with the draft count of the SAME step,
            # snapshotted at sample_async time: _request_common_handling below
            # replaces py_draft_tokens with the next step's buffer, so its
            # length must not be used as the denominator (0 for the request's
            # prefill step, where nothing was verified).
            verified_len = self._verified_len(req, runtime_draft_len, state.verify_lens_snapshot)
            drafted_len = state.draft_lens[req_idx] if state.draft_lens is not None else 0
            req.py_num_draft_tokens_verified = min(drafted_len, verified_len)
            req.py_rewind_len = verified_len - req.py_num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)

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

        # Snapshot each request's draft count for THIS step before
        # _add_dummy_draft_tokens below installs placeholder drafts on
        # finished-context requests; update_requests pairs these with the
        # acceptance counts (see SampleStateSpec.draft_lens). Drafter-fed
        # flows (NGram, SA) pad py_draft_tokens to the static max for CUDA
        # graphs before the forward, so prefer the pre-padding count the
        # drafter recorded; min() guards against a stale count when the
        # buffer was since cleared (e.g. speculation dynamically disabled).
        draft_lens = [
            min(r.py_draft_tokens_effective_len, get_draft_token_length(r))
            if r.py_draft_tokens_effective_len is not None
            else get_draft_token_length(r)
            for r in sampling_requests
        ]

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
        # The worker output width tracks runtime_draft_len, which dynamic draft
        # length shrinks below the statically allocated store width.
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
        )
        sampler_event = self._record_sampler_event()

        # Add dummy draft tokens to context requests for KV cache preparation
        for request in finished_context_requests:
            request.py_draft_tokens = [1] * self.draft_len

        verify_lens_snapshot = self._snapshot_verify_lens(sampling_requests)

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
            draft_lens=draft_lens,
            verify_lens_snapshot=verify_lens_snapshot,
        )
