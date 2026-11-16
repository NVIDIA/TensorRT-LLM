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

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..._device_copy import DeviceCopyBuffer
from ..._utils import prefer_pinned
from ...bindings.executor import FinishReason
from ...sampling_params import SamplingParams
from ..kernels.spec_step_copies import op as spec_step_copies
from ..pyexecutor.llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.runtime_hooks import SpecStepFold
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    AsyncWorkerMixin,
    Sampler,
    SampleState,
    SampleStateTensors,
    TorchSampler,
    int_tensor,
)
from ..pyexecutor.sampler.penalties import has_occurrence_penalty
from ..pyexecutor.sampler.sampler_common import _request_get_sampling_params, top_p_decay_active
from ..pyexecutor.sampler.sampler_features import max_token_room, meet_stop_token_criteria
from ..pyexecutor.sampler.watermark import request_carries_watermark
from ..pyexecutor.scheduler import ScheduledRequests
from .spec_store_scatter import spec_store_scatter

# Rejection messages for the sampling features the one-model speculative path
# cannot honor. They are module constants because two places raise them: the
# executor-side SpecSampler.validate_request below, and the admission-time
# check LLM._check_arguments runs before the response stream opens. Sharing the
# text is what keeps the two from drifting apart.
UNSUPPORTED_MIN_LENGTH_MSG = (
    "min_length is not supported with one-model speculative decoding. "
    "Drop min_length from the request, or disable speculative decoding."
)
UNSUPPORTED_BAD_WORDS_MSG = (
    "bad_words is not supported with one-model speculative decoding. "
    "Drop bad_words from the request, or disable speculative decoding."
)
UNSUPPORTED_NO_REPEAT_NGRAM_MSG = (
    "no_repeat_ngram_size is not supported with one-model speculative "
    "decoding. Drop no_repeat_ngram_size from the request, or disable "
    "speculative decoding."
)
UNSUPPORTED_EMBEDDING_BIAS_MSG = (
    "embedding_bias is not supported with one-model speculative decoding. "
    "Drop embedding_bias from the request, or disable speculative decoding."
)
UNSUPPORTED_TOP_P_DECAY_MSG = (
    "top_p_decay is not supported with one-model speculative decoding. "
    "Drop top_p_decay / top_p_min from the request, or disable "
    "speculative decoding."
)
UNSUPPORTED_LOGITS_POST_PROCESSOR_MSG = (
    "logits_processor (a logits post-processor callback) is not supported with one-model "
    "speculative decoding. Drop logits_processor from the request, or disable speculative "
    "decoding."
)
UNSUPPORTED_MIN_P_MSG = (
    "min_p requires 'advanced_sampling_mode: full' in the speculative decoding "
    "config when using one-model speculative decoding. Restore that default, "
    "drop min_p from the request, or disable speculative decoding."
)


def one_model_sampling_rejection_reason(
    sampling_params: Optional[SamplingParams], *, fused_sampling: bool = False
) -> Optional[str]:
    """Why ``sampling_params`` cannot be served by the one-model speculative path.

    ``SpecSampler.validate_request`` is the authoritative check, but it runs on
    the executor's admission path. By then the OpenAI frontend has already
    answered 200 and opened the response stream, so the rejection reaches the
    client as an aborted stream (``ClientPayloadError`` /
    ``TransferEncodingError``) that it cannot tell apart from a network fault.
    This mirrors the subset of those rules ``SamplingParams`` alone can decide,
    so the request can be rejected before the stream opens; the executor-side
    check stays as the backstop for submission paths that bypass the LLM API.

    Returns ``None`` when nothing in ``sampling_params`` is unsupported.
    """
    if sampling_params is None:
        return None
    # min_tokens is what SamplingParams calls the field LlmRequest carries as
    # py_min_length. The OpenAI frontend always forwards it, so gate on the
    # value rather than on its presence -- same reasoning as the LlmRequest
    # check below.
    min_tokens = sampling_params.min_tokens
    if min_tokens and min_tokens > 0:
        return UNSUPPORTED_MIN_LENGTH_MSG
    # Either form populates py_bad_words; the string form needs a tokenizer to
    # resolve, so test the raw fields rather than _get_bad_words().
    if sampling_params.bad or sampling_params.bad_token_ids:
        return UNSUPPORTED_BAD_WORDS_MSG
    if sampling_params.no_repeat_ngram_size:
        return UNSUPPORTED_NO_REPEAT_NGRAM_MSG
    if sampling_params.embedding_bias is not None:
        return UNSUPPORTED_EMBEDDING_BIAS_MSG
    # A per-request callback becomes py_logits_post_processors; the one-model path never
    # calls it (a fused verify may not even materialize the full logits).
    if sampling_params.logits_processor:
        return UNSUPPORTED_LOGITS_POST_PROCESSOR_MSG
    # The same predicate top_p_decay_active() delegates to, so "active" cannot
    # mean one thing here and another in the sampler.
    if SamplingParams.params_imply_top_p_decay_active(sampling_params.top_p_decay):
        return UNSUPPORTED_TOP_P_DECAY_MSG
    # The one filter whose support depends on the deploy config rather than on
    # sampling_params, so the caller has to supply it.
    min_p = sampling_params.min_p
    if min_p and min_p > 0.0 and not fused_sampling:
        return UNSUPPORTED_MIN_P_MSG
    return None


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor
    # Actual token windows, aligned with SampleStateSpec.requests, when the
    # worker selects verification lengths on device.
    verify_lens: Optional[torch.Tensor] = None


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

    # The model's speculative step-input staging, when it opted into one (bind_spec_step_fold).
    spec_step_fold: Optional[SpecStepFold] = None
    # Copy the store on the D2H side stream also on the steps a step fold's forward stored the rows, when the fold
    # awaits the copies through its store_copies_done (RuntimeHooks.sampler_d2h_side_stream).
    d2h_side_stream_copies: bool = False
    # Keep the stored rows' slot ids in a device buffer re-uploaded only when they change
    # (RuntimeHooks.spec_small_h2d).
    static_slot_ids: bool = False
    _slot_ids_host: Optional[list[int]] = None
    _slot_ids_cuda: Optional[torch.Tensor] = None

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    def validate_request(self, request: LlmRequest) -> None:
        """Reject sampling parameters the one-model speculative path cannot honor.

        The fused backend supports min_p; other one-model sampling backends do
        not. This sampler also does not return context logits, generation logits,
        or log probabilities. Raised during request admission, so only the
        offending request fails rather than the whole executor step.
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

        # A greedy request cannot carry a watermark, so it is served normally.
        reason = getattr(self, "_watermark_unsupported_reason", None)
        if reason is not None and request_carries_watermark(self._watermark_config, request):
            raise ValueError(reason)

        sampling_config = request.sampling_config
        if sampling_config is None:
            return
        min_p = sampling_config.min_p
        if min_p and min_p > 0.0 and not self._fused_sampling:
            raise ValueError(UNSUPPORTED_MIN_P_MSG)
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
                "supported with tree speculative decoding (dynamic tree) yet. Drop "
                "the penalties, use a linear speculation mode, or disable "
                "speculative decoding."
            )

    @staticmethod
    def _validate_unsupported_logits_processors(request: LlmRequest) -> None:
        """Reject logits-side sampling features the one-model path cannot apply.

        TorchSampler implements min_length, bad_words, no_repeat_ngram_size,
        embedding_bias, top_p_decay and logits post-processor callbacks by editing
        the target logits before sampling. The one-model path has no logits-editing hook, so each is
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
            raise ValueError(UNSUPPORTED_MIN_LENGTH_MSG)
        if getattr(request, "py_bad_words", None):
            raise ValueError(UNSUPPORTED_BAD_WORDS_MSG)
        if getattr(request, "py_no_repeat_ngram_size", None):
            raise ValueError(UNSUPPORTED_NO_REPEAT_NGRAM_MSG)
        if getattr(request, "py_embedding_bias", None) is not None:
            raise ValueError(UNSUPPORTED_EMBEDDING_BIAS_MSG)
        # The one-model sampler never calls a request's logits post-processors, and a fused
        # verify (e.g. the MiniMax-M3 shard verify) does not even gather the full logits, so
        # the callback would be skipped silently.
        if getattr(request, "py_logits_post_processors", None):
            raise ValueError(UNSUPPORTED_LOGITS_POST_PROCESSOR_MSG)
        # Reuse the handler's own predicate so "active" cannot drift between paths.
        if top_p_decay_active(_request_get_sampling_params(request)):
            raise ValueError(UNSUPPORTED_TOP_P_DECAY_MSG)

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors.

        The host-read tensors are views of ``host_read``, so one D2H copy snapshots them.
        """

        host_read: torch.Tensor
        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor

        @classmethod
        def allocate(
            cls,
            *,
            num_slots: int,
            accepted_len: int,
            draft_len: int,
            beam_width: int,
            device: str = "cuda",
        ) -> "SpecSampler.Store":
            """A store for ``num_slots`` slots."""
            new_tokens_shape = (accepted_len, num_slots, beam_width)
            host_read = int_tensor(
                (math.prod(new_tokens_shape) + num_slots * (1 + draft_len),), device=device
            )
            new_tokens, new_tokens_lens, next_draft_tokens = cls._host_read_views(
                host_read, new_tokens_shape, (num_slots, draft_len)
            )
            return cls(
                host_read=host_read,
                new_tokens=new_tokens,
                next_new_tokens=int_tensor((draft_len + 1, num_slots, beam_width), device=device),
                next_draft_tokens=next_draft_tokens,
                new_tokens_lens=new_tokens_lens,
            )

        @staticmethod
        def _host_read_views(
            host_read: torch.Tensor,
            new_tokens_shape: tuple[int, ...],
            next_draft_tokens_shape: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            lens_start = math.prod(new_tokens_shape)
            drafts_start = lens_start + next_draft_tokens_shape[0]
            return (
                host_read[:lens_start].view(new_tokens_shape),
                host_read[lens_start:drafts_start],
                host_read[drafts_start:].view(next_draft_tokens_shape),
            )

        def views_of(self, host_read: torch.Tensor) -> SampleStateTensorsSpec:
            """The host-read tensors of a copy of ``host_read``."""
            new_tokens, new_tokens_lens, next_draft_tokens = self._host_read_views(
                host_read, tuple(self.new_tokens.shape), tuple(self.next_draft_tokens.shape)
            )
            return SampleStateTensorsSpec(
                new_tokens=new_tokens,
                new_tokens_lens=new_tokens_lens,
                next_draft_tokens=next_draft_tokens,
            )

    def __init__(
        self,
        args: TorchSampler.Args,
        *,
        enable_penalty: bool = False,
        penalty_supported: bool = True,
        fused_sampling: bool = False,
        watermark_unsupported_reason: Optional[str] = None,
    ):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            enable_penalty: whether the deploy enabled the occurrence penalties.
                Only used to decide whether a request asking for them is admitted;
                the penalties themselves are applied inside the worker.
            penalty_supported: whether this speculation mode's row layout is one
                the penalties can map (linear modes yes, tree modes not yet).
            fused_sampling: whether the configured sampler supports min_p.
            watermark_unsupported_reason: why this speculation mode cannot serve watermarked
                requests (None when it can). Such requests are rejected at admission.
        """
        self._enable_penalty = enable_penalty
        self._penalty_supported = penalty_supported
        self._watermark_config = args.watermark_config
        self._watermark_unsupported_reason = watermark_unsupported_reason
        self._fused_sampling = fused_sampling
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
        # a step can accept rather than to the wire width: max_draft_len + 1,
        # because the drafter advances max_draft_len times and the golden token
        # the target always accepts adds one. Verified against Eagle3 dynamic
        # tree (K=6, T=60), MTP dynamic tree, PARD (T=2K-1) and the linear
        # modes -- none exceed it.
        self.max_accepted_path_len = args.max_draft_len + 1
        self._store_scatter: Optional[spec_step_copies.SlotScatter] = None
        self._slot_table_buffer: Optional[DeviceCopyBuffer] = None
        if spec_step_copies.is_supported():
            self._store_scatter = spec_step_copies.SlotScatter()
            self._slot_table_buffer = DeviceCopyBuffer(
                torch.zeros((seq_slots,), dtype=torch.int32, device="cuda"),
                pin_memory=prefer_pinned(),
            )
        # Recorded after the last step's host snapshot of the stores, which reads
        # them on the D2H side stream.
        self._store_copies_done: Optional[torch.cuda.Event] = None
        self.store = self.Store.allocate(
            num_slots=seq_slots,
            accepted_len=self.max_accepted_path_len,
            draft_len=args.max_total_draft_tokens,
            beam_width=self.max_beam_width,
        )
        # The last batch's slots and their device tensor.
        self._last_slot_index: Optional[tuple[tuple[int, ...], torch.Tensor]] = None

    def _slot_index(self, slot_ids: list[int]) -> torch.Tensor:
        """The batch's slots as a device tensor, reused while they do not change."""
        key = tuple(slot_ids)
        if self._last_slot_index is None or self._last_slot_index[0] != key:
            slots = torch.tensor(key, dtype=torch.long, pin_memory=prefer_pinned()).to(
                device="cuda", non_blocking=True
            )
            self._last_slot_index = (key, slots)
        return self._last_slot_index[1]

    def _device_slot_ids(self, seq_slots: list[int]) -> torch.Tensor:
        """The slot ids on the device, uploaded only when they differ from the last upload.

        The buffer is this sampler's own: only this method writes it, so the last uploaded
        list describes its contents, and the stream orders each upload after the scatter that
        read the previous ids.
        """
        n = len(seq_slots)
        if self._slot_ids_cuda is None or self._slot_ids_cuda.numel() < n:
            self._slot_ids_cuda = torch.empty(max(n, 8), dtype=torch.long, device="cuda")
            self._slot_ids_host = None
        if self._slot_ids_host != list(seq_slots):
            self._slot_ids_cuda[:n].copy_(
                torch.tensor(seq_slots, dtype=torch.long, pin_memory=prefer_pinned()),
                non_blocking=True,
            )
            self._slot_ids_host = list(seq_slots)
        return self._slot_ids_cuda[:n]

    def _copy_outputs_to_host(
        self,
        o_verify_lens: Optional[torch.Tensor],
        forward_stored_rows: bool,
        store_copies_done: Optional[torch.cuda.Event] = None,
    ) -> tuple[SampleStateTensorsSpec, Optional[torch.cuda.Event]]:
        """Host copies of the step's stored outputs, and the side-stream event to await them by (or None).

        The per-slot store is copied on the D2H side stream, so the copies overlap the next step's input staging
        instead of preceding it on the main stream. That is sound while nothing rewrites the store before the next
        ``sample_async`` waits on ``_store_copies_done``, which holds when this sampler stored the rows itself.
        ``verify_lens`` is a forward output buffer the next replay rewrites, so it is always copied on the main
        stream.

        When the forward stored the rows (a step fold), the next replay rewrites the store, so the copies stay on
        the main stream unless the fold awaits them: with ``d2h_side_stream_copies`` and the fold's
        ``store_copies_done``, that event is recorded on the side stream after the copies and the forward waits
        on it before it writes the store.
        """
        verify_lens = self._copy_to_host(o_verify_lens) if o_verify_lens is not None else None
        fold_awaits_copies = self.d2h_side_stream_copies and store_copies_done is not None
        if self._async_worker_active() or (forward_stored_rows and not fold_awaits_copies):
            host = self.store.views_of(self._copy_to_host(self.store.host_read))
            host.verify_lens = verify_lens
            return host, None
        with self._make_side_stream_copier() as copier:
            host_read = copier.stage_copy_to_host(self.store.host_read)
        host = self.store.views_of(host_read)
        host.verify_lens = verify_lens
        self._store_copies_done = copier.event
        if store_copies_done is not None and copier.event is not None:
            # The fold's forward waits on it before every store write, whoever stored this step's rows.
            store_copies_done.record(self._d2h_side_stream)
        return host, copier.event

    def bind_spec_step_fold(self, spec_step_fold: SpecStepFold) -> None:
        """Let the model's forward store into this sampler's per-slot store (see SpecStepFold)."""
        self.spec_step_fold = spec_step_fold
        spec_step_fold.bind_sampler_store(self.store)

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        runtime_draft_len: Optional[int],
    ) -> None:
        """Common handling for both context and generation requests."""
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

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
        token_windows = (
            state.host.verify_lens.tolist() if state.host.verify_lens is not None else None
        )
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        if token_windows is not None:
            for request_index, request in enumerate(state.requests):
                if request.state == LlmRequestState.GENERATION_COMPLETE:
                    continue
                token_window = token_windows[request_index]
                num_new_tokens = new_tokens_lens_list[request.py_seq_slot]
                max_new_tokens = min(token_window, self.max_accepted_path_len)
                if state.draft_lens is not None:
                    max_new_tokens = min(max_new_tokens, state.draft_lens[request_index] + 1)
                if not 1 <= token_window <= runtime_draft_len + 1 or not (
                    1 <= num_new_tokens <= max_new_tokens
                ):
                    raise ValueError(
                        "executed verify_lens must be positive and new_tokens_lens must fit "
                        "the executed window, original proposals, and token storage"
                    )

        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            seq_slot = req.py_seq_slot
            assert seq_slot is not None
            num_new_tokens = new_tokens_lens_list[seq_slot]
            # new_tokens is sized to this bound; fail clearly, not with an IndexError.
            assert num_new_tokens <= self.max_accepted_path_len, (
                f"accepted {num_new_tokens} tokens in one step, but new_tokens is "
                f"sized for {self.max_accepted_path_len}"
            )
            # handle_stop_criteria inlined, with the per-request reads hoisted.
            end_id = req.py_end_id
            room = max_token_room(req, self.max_seq_len, beam_idx)
            stop_words = req.py_stop_words_list
            for i in range(num_new_tokens):
                new_token = new_tokens[i][seq_slot][beam_idx]
                req.add_new_token(new_token, beam_idx)
                if new_token == end_id:
                    req.finish_by(FinishReason.END_ID, beam_idx)
                    break
                if i + 1 >= room:
                    req.finish_by(FinishReason.LENGTH, beam_idx)
                    break
                if stop_words and meet_stop_token_criteria(req, new_token, beam_idx):
                    req.finish_by(FinishReason.STOP_WORDS, beam_idx)
                    break
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            # Pair the acceptance count with the draft count of the SAME step,
            # snapshotted at sample_async time: _request_common_handling below
            # replaces py_draft_tokens with the next step's buffer, so its
            # length must not be used as the denominator (0 for the request's
            # prefill step, where nothing was verified).
            verified_len = (
                token_windows[req_idx] - 1 if token_windows is not None else runtime_draft_len
            )
            drafted_len = state.draft_lens[req_idx] if state.draft_lens is not None else 0
            req.py_num_draft_tokens_verified = min(drafted_len, verified_len)
            req.py_rewind_len = verified_len - req.py_num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)

    def _scatter_to_stores(
        self, outputs: dict[str, torch.Tensor], num_skip: int, slots: list[int]
    ) -> None:
        """Move rows ``num_skip`` onward of the forward's outputs into the stores at ``slots``."""
        num_sampling_requests = len(slots)

        o_new_tokens = outputs["new_tokens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = outputs["new_tokens_lens"][num_skip : num_skip + num_sampling_requests]
        o_next_draft_tokens = outputs["next_draft_tokens"][
            num_skip : num_skip + num_sampling_requests
        ]
        o_next_new_tokens = outputs["next_new_tokens"][num_skip : num_skip + num_sampling_requests]

        slots_device = (
            self._device_slot_ids(slots) if self.static_slot_ids else self._slot_index(slots)
        )
        step_fold = self.spec_step_fold
        if step_fold is not None and step_fold.scatter(
            slots_device, o_new_tokens, o_next_new_tokens, o_new_tokens_lens, o_next_draft_tokens
        ):
            return

        # Pad or truncate to match fixed-size store buffers for the scatter.
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

        # The scatter reads one output row and writes one store slot per request, unchecked.
        num_slots = self.store.new_tokens_lens.shape[0]
        if any(
            rows.shape[0] != num_sampling_requests
            for rows in (o_new_tokens, o_new_tokens_lens, o_next_draft_tokens, o_next_new_tokens)
        ) or (slots and not (min(slots) >= 0 and max(slots) < num_slots)):
            raise ValueError("speculative outputs need one row and an in-range slot per request")

        spec_store_scatter(
            slots_device,
            o_new_tokens,
            o_next_new_tokens,
            o_new_tokens_lens,
            o_next_draft_tokens,
            self.store.new_tokens.squeeze(-1),
            self.store.next_new_tokens.squeeze(-1),
            self.store.new_tokens_lens,
            self.store.next_draft_tokens,
        )

    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict[str, torch.Tensor | bool],
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
                - verify_lens: optional int32 [output rows] executed token windows
                - verify_lens_in_output_order: required True with verify_lens
                - native_uniform_verify: optional boolean declaring uniform windows
                - host_policy_windows_snapshot: legacy True is rejected
            num_context_logits_prefix_sum: Prefix sum of context logits (unused)

        Returns:
            SampleStateSpec with device and host tensors
        """
        num_skip = len(scheduled_requests.context_requests_chunking)
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests

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

        num_sampling_requests = len(sampling_requests)
        o_verify_lens = outputs.get("verify_lens")
        native_uniform = outputs.get("native_uniform_verify", False)
        host_snapshot = outputs.get("host_policy_windows_snapshot", False)
        output_order = outputs.get("verify_lens_in_output_order", False)
        if any(
            type(marker) is not bool for marker in (native_uniform, host_snapshot, output_order)
        ):
            raise ValueError("verification-window markers must be boolean")
        if host_snapshot:
            raise ValueError("legacy ragged producers must publish executed output-row verify_lens")
        if native_uniform and (o_verify_lens is not None or output_order):
            raise ValueError("native-uniform verification cannot also publish ragged windows")
        if output_order != (o_verify_lens is not None):
            raise ValueError("verify_lens requires an explicit output-row layout declaration")
        if o_verify_lens is None:
            if not native_uniform and any(
                getattr(r, "py_verify_len", None) is not None for r in sampling_requests
            ):
                raise ValueError("ragged producers must publish executed output-row verify_lens")
        else:
            output_rows = outputs["new_tokens_lens"].shape[0]
            if (
                not isinstance(o_verify_lens, torch.Tensor)
                or o_verify_lens.dim() != 1
                or o_verify_lens.dtype != torch.int32
                or o_verify_lens.device != outputs["new_tokens_lens"].device
                or o_verify_lens.shape[0] != output_rows
                or output_rows < num_skip + num_sampling_requests
                or any(
                    outputs[name].shape[0] != output_rows
                    for name in ("new_tokens", "next_draft_tokens", "next_new_tokens")
                )
            ):
                raise ValueError("verify_lens must contain one int32 token window per output row")
            o_verify_lens = o_verify_lens[num_skip : num_skip + num_sampling_requests]

        runtime_draft_len = outputs["next_draft_tokens"].shape[1]
        if self._store_copies_done is not None:
            # The previous step's host copies may still be reading the stores.
            torch.cuda.current_stream().wait_event(self._store_copies_done)
        slots = [r.py_seq_slot for r in sampling_requests]
        step_fold = self.spec_step_fold
        forward_stored_rows = step_fold is not None and step_fold.sampler_rows_stored(
            outputs, num_skip, slots
        )
        scattered = forward_stored_rows
        # A model's step fold scatters the stores itself, in _scatter_to_stores.
        if not scattered and step_fold is None and self._store_scatter is not None:
            # A step whose batch is unchanged leaves the slot table as it is.
            self._slot_table_buffer.update(torch.tensor(slots, dtype=torch.int32))
            scattered = self._store_scatter.scatter(
                outputs,
                num_skip,
                len(slots),
                self._slot_table_buffer.tensor,
                self.store.new_tokens,
                self.store.next_new_tokens,
                self.store.new_tokens_lens,
                self.store.next_draft_tokens,
            )
        if not scattered:
            self._scatter_to_stores(outputs, num_skip, slots)

        # Create sample state with async D2H copy
        device_tensors = SampleStateTensorsSpec(
            new_tokens=self.store.next_new_tokens,
            new_tokens_lens=self.store.new_tokens_lens,
            next_draft_tokens=self.store.next_draft_tokens,
        )

        store_copies_done = getattr(step_fold, "store_copies_done", None) if step_fold else None
        host_tensors, side_stream_event = self._copy_outputs_to_host(
            o_verify_lens, forward_stored_rows, store_copies_done
        )
        sampler_event = self._record_sampler_event(side_stream_event=side_stream_event)

        # Add dummy draft tokens to context requests for KV cache preparation
        for request in finished_context_requests:
            request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
            draft_lens=draft_lens,
        )
