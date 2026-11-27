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
from ..._utils import current_cuda_stream, prefer_pinned
from ...bindings.executor import FinishReason
from ...sampling_params import LogprobMode, SamplingParams
from ..kernels.spec_step_copies import op as spec_step_copies
from ..modules.multi_stream_utils import _record_stream
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
from . import logit_edits
from .interface import OneModelSamplingParams
from .spec_store_scatter import spec_store_scatter
from .target_logprobs import OUTPUT_KEY as TARGET_LOGPROBS_KEY

# Rejection messages for the sampling features the one-model speculative path
# cannot honor. They are module constants because two places raise them: the
# executor-side SpecSampler.validate_request below, and the admission-time
# check LLM._check_arguments runs before the response stream opens. Sharing the
# text is what keeps the two from drifting apart.
UNSUPPORTED_MIN_LENGTH_MSG = (
    "min_length is not supported with one-model speculative decoding. "
    "Drop min_length from the request, or disable speculative decoding."
)
UNSUPPORTED_NO_REPEAT_NGRAM_MSG = (
    "no_repeat_ngram_size is not supported with one-model speculative "
    "decoding. Drop no_repeat_ngram_size from the request, or disable "
    "speculative decoding."
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
UNSUPPORTED_LOGPROBS_FORMAT_MSG = (
    "One-model speculative decoding returns only the sampled token's logprob. Request it with "
    "logprobs=0 (or True) and logprobs_simple_format=True, or disable speculative decoding."
)
UNSUPPORTED_PROCESSED_LOGPROBS_MSG = (
    "logprobs_mode='processed' with one-model speculative decoding supports temperature only: it needs "
    "a sampled request (temperature > 0) without top_k, top_p, min_p, penalties, embedding_bias or "
    "bad_words. Use logprobs_mode='raw', simplify the request, or disable speculative decoding."
)
UNSUPPORTED_LOGPROBS_TREE_MSG = (
    "logprobs are not supported with tree speculative decoding (dynamic tree). Drop logprobs, use a "
    "linear speculation mode, or disable speculative decoding."
)
UNSUPPORTED_LOGPROBS_RELAXED_ACCEPTANCE_MSG = (
    "logprobs are not supported with relaxed acceptance for thinking: it emits draft tokens the target "
    "did not sample, so their logprobs would not be the target's. Drop logprobs, disable "
    "use_relaxed_acceptance_for_thinking, or disable speculative decoding."
)
UNSUPPORTED_LOGPROBS_MODE_MSG = (
    "logprobs are not supported with this speculative decoding mode (SA). Drop logprobs, or disable "
    "speculative decoding."
)
_TARGET_LOGPROBS_LIMIT_MSGS = {
    "tree": UNSUPPORTED_LOGPROBS_TREE_MSG,
    "relaxed_acceptance": UNSUPPORTED_LOGPROBS_RELAXED_ACCEPTANCE_MSG,
    "mode": UNSUPPORTED_LOGPROBS_MODE_MSG,
}
# embedding_bias and bad_words are served only where the deploy opted into rewriting the
# target logits ahead of acceptance, and by the workers that verify through that pass.
LOGIT_EDITS_DISABLED_MSG = (
    "embedding_bias (logit_bias) and bad_words require 'enable_penalty: true' in the "
    "speculative decoding config when using one-model speculative decoding. Enable that "
    "flag, drop them from the request, or disable speculative decoding."
)
LOGIT_EDITS_TREE_MSG = (
    "embedding_bias (logit_bias) and bad_words are not supported with tree speculative "
    "decoding (dynamic tree) yet. Drop them from the request, use a linear speculation "
    "mode, or disable speculative decoding."
)
LOGIT_EDITS_RELAXED_ACCEPTANCE_MSG = (
    "embedding_bias (logit_bias) and bad_words are not supported with relaxed acceptance "
    "for thinking. Drop them from the request, disable "
    "use_relaxed_acceptance_for_thinking, or disable speculative decoding."
)
LOGIT_EDITS_MODE_MSG = (
    "embedding_bias (logit_bias) and bad_words are not supported with this speculative "
    "decoding mode (SA). Drop them from the request, or disable speculative decoding."
)
# The occurrence penalties are applied by the same pass, so the same deploys cannot serve them.
PENALTY_TREE_MSG = (
    "repetition_penalty / presence_penalty / frequency_penalty are not "
    "supported with tree speculative decoding (dynamic tree) yet. Drop "
    "the penalties, use a linear speculation mode, or disable "
    "speculative decoding."
)
PENALTY_RELAXED_ACCEPTANCE_MSG = (
    "repetition_penalty / presence_penalty / frequency_penalty are not supported with "
    "relaxed acceptance for thinking. Drop the penalties, disable "
    "use_relaxed_acceptance_for_thinking, or disable speculative decoding."
)
PENALTY_MODE_MSG = (
    "repetition_penalty / presence_penalty / frequency_penalty are not supported with this "
    "speculative decoding mode (SA). Drop the penalties, or disable speculative decoding."
)

# What keeps a deploy's worker from verifying through the logits-edit pass, mapped to the
# (embedding_bias and bad_words, occurrence penalties) rejection messages.
_EDIT_PASS_LIMIT_MSGS = {
    "tree": (LOGIT_EDITS_TREE_MSG, PENALTY_TREE_MSG),
    "relaxed_acceptance": (LOGIT_EDITS_RELAXED_ACCEPTANCE_MSG, PENALTY_RELAXED_ACCEPTANCE_MSG),
    "mode": (LOGIT_EDITS_MODE_MSG, PENALTY_MODE_MSG),
}


def _edit_pass_limit(spec_config) -> Optional[str]:
    """What keeps this one-model deploy's worker from verifying through the logits-edit
    pass (``SpecWorkerBase._edit_target_logits``): a key of ``_EDIT_PASS_LIMIT_MSGS``, or
    None when it does.

    The pass maps one logits row to each verified position of a linear draft, and strict
    or rejection acceptance reads its output. Tree rows, relaxed acceptance and SA, which
    verifies outside it, are not covered. This follows ``get_spec_worker``: Eagle3 serves
    a tree of top-K 1 with its linear worker, while MTP picks its tree worker on the flag
    alone. Only fields the user sets are read, so the LLM API and the executor reach the
    same answer before and after the checkpoint resolves an MTP config's mode.
    """
    spec_dec_mode = getattr(spec_config, "spec_dec_mode", None)
    if getattr(spec_config, "use_dynamic_tree", False):
        linear_eagle3 = (
            spec_dec_mode is not None
            and spec_dec_mode.is_eagle3_one_model()
            and (getattr(spec_config, "dynamic_tree_max_topK", None) or 0) <= 1
        )
        if not linear_eagle3:
            return "tree"
    if getattr(spec_config, "use_relaxed_acceptance_for_thinking", False):
        return "relaxed_acceptance"
    if spec_dec_mode is not None and spec_dec_mode.is_sa():
        return "mode"
    return None


def verifies_through_edit_pass(spec_config) -> bool:
    """Whether this one-model deploy's worker verifies against the logits the edit pass
    returns, on every step. The occurrence penalties, embedding_bias and bad_words are
    served only where it does."""
    return _edit_pass_limit(spec_config) is None


def logit_edits_unsupported_reason(spec_config) -> Optional[str]:
    """Why this one-model deploy cannot serve embedding_bias and bad_words, or None.

    Both are applied to the target logits a worker verifies against, in the pass
    ``enable_penalty`` turns on, so the deploy must enable it and verify through it
    (``verifies_through_edit_pass``).
    """
    if not getattr(spec_config, "enable_penalty", False):
        return LOGIT_EDITS_DISABLED_MSG
    limit = _edit_pass_limit(spec_config)
    return None if limit is None else _EDIT_PASS_LIMIT_MSGS[limit][0]


def penalty_unsupported_reason(spec_config) -> Optional[str]:
    """Why this one-model deploy cannot serve the occurrence penalties even with
    ``enable_penalty`` on, or None when it can."""
    limit = _edit_pass_limit(spec_config)
    return None if limit is None else _EDIT_PASS_LIMIT_MSGS[limit][1]


def target_logprobs_unsupported_reason(spec_config) -> Optional[str]:
    """Why this one-model deploy cannot return target logprobs, or None when it can.

    The scores are read from one target logits row per verified position of a linear draft, so the deploys whose
    rows are tree nodes, or whose emitted tokens are not draws from the target row, cannot serve them.
    """
    limit = _edit_pass_limit(spec_config)
    return None if limit is None else _TARGET_LOGPROBS_LIMIT_MSGS[limit]


def _logprobs_rejection_reason(
    *,
    num_logprobs: Optional[int],
    simple_format: bool,
    processed: bool,
    sampled_temperature_only: bool,
    has_penalty_or_edits: bool,
    deploy_reason: Optional[str],
) -> Optional[str]:
    """Why a request for generation logprobs cannot be served, or None. Shared by the early admission check
    (from ``SamplingParams``) and ``SpecSampler.validate_request`` (from the ``LlmRequest``)."""
    if num_logprobs is None:
        return None
    if deploy_reason is not None:
        return deploy_reason
    if num_logprobs != 0 or not simple_format:
        return UNSUPPORTED_LOGPROBS_FORMAT_MSG
    if processed and not (sampled_temperature_only and not has_penalty_or_edits):
        return UNSUPPORTED_PROCESSED_LOGPROBS_MSG
    return None


def one_model_sampling_rejection_reason(
    sampling_params: Optional[SamplingParams],
    *,
    fused_sampling: bool = False,
    logit_edits_unsupported: Optional[str] = LOGIT_EDITS_DISABLED_MSG,
    target_logprobs_unsupported: Optional[str] = None,
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

    ``logit_edits_unsupported`` is ``logit_edits_unsupported_reason`` of the deploy: why it
    cannot serve embedding_bias and bad_words, or None when it can. ``target_logprobs_unsupported``
    is ``target_logprobs_unsupported_reason`` of the deploy.

    Returns ``None`` when nothing in ``sampling_params`` is unsupported.
    """
    if sampling_params is None:
        return None
    processed = (
        sampling_params.logprobs is not None
        and sampling_params.logprobs_mode == LogprobMode.PROCESSED
    )
    reason = _logprobs_rejection_reason(
        num_logprobs=sampling_params.logprobs,
        simple_format=sampling_params.logprobs_simple_format,
        processed=processed,
        sampled_temperature_only=bool(sampling_params.temperature)
        and not sampling_params.top_k
        and sampling_params.top_p in (None, 1.0)
        and not sampling_params.min_p,
        has_penalty_or_edits=(
            sampling_params.repetition_penalty not in (None, 1.0)
            or bool(sampling_params.presence_penalty)
            or bool(sampling_params.frequency_penalty)
            or bool(sampling_params.bad or sampling_params.bad_token_ids)
            or sampling_params.embedding_bias is not None
        ),
        deploy_reason=target_logprobs_unsupported,
    )
    if reason is not None:
        return reason
    # min_tokens is what SamplingParams calls the field LlmRequest carries as
    # py_min_length. The OpenAI frontend always forwards it, so gate on the
    # value rather than on its presence -- same reasoning as the LlmRequest
    # check below.
    min_tokens = sampling_params.min_tokens
    if min_tokens and min_tokens > 0:
        return UNSUPPORTED_MIN_LENGTH_MSG
    # Either form populates py_bad_words; the string form needs a tokenizer to
    # resolve, so test the raw fields rather than _get_bad_words().
    has_bad_words = bool(sampling_params.bad or sampling_params.bad_token_ids)
    if has_bad_words or sampling_params.embedding_bias is not None:
        if logit_edits_unsupported is not None:
            return logit_edits_unsupported
        # The string form counts once it is tokenized; the executor-side check sees
        # every word either way.
        bad_words = [[token] for token in sampling_params.bad_token_ids or ()]
        bad_words += sampling_params._bad_word_ids or []
        reason = logit_edits.rejection_reason(sampling_params.embedding_bias, bad_words)
        if reason is not None:
            return reason
    if sampling_params.no_repeat_ngram_size:
        return UNSUPPORTED_NO_REPEAT_NGRAM_MSG
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
class StepOutputRows:
    """A decode-only step's forward outputs in request order, with the requests' slots."""

    next_new_tokens: torch.Tensor  # int32 [rows, >= tokens per step]
    new_tokens_lens: torch.Tensor  # int32 [rows]
    next_draft_tokens: torch.Tensor  # int32 [rows, >= draft length]
    slots: tuple[int, ...]


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor
    # Actual token windows, aligned with SampleStateSpec.requests, when the
    # worker selects verification lengths on device.
    verify_lens: Optional[torch.Tensor] = None
    # Recorded on the sampler's side stream once it has written this step into the stores,
    # when that update runs there: a reader of the stores on another stream waits on it.
    stores_ready: Optional[torch.cuda.Event] = None
    # The forward's outputs the stores are updated from (a decode-only step's, with stores_ready).
    rows: Optional[StepOutputRows] = None


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
    # [rows, width] host copy of the target logprobs of the emitted tokens, when a request of the batch asked.
    target_logprobs: Optional[torch.Tensor] = None


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
    # Set where the engine's steady MTP prepare reads a step's output rows: the store update then
    # runs on the D2H side stream.
    stores_on_side_stream = False

    SampleState = SampleStateSpec

    def validate_request(self, request: LlmRequest) -> None:
        """Reject sampling parameters the one-model speculative path cannot honor.

        The fused backend supports min_p; other one-model sampling backends do
        not. This sampler returns the target logprob of each emitted token, but not
        context logits, generation logits or top-k logprobs. Raised during request
        admission, so only the offending request fails rather than the whole executor
        step.
        """
        requested_outputs = (
            ("return_context_logits / prompt_logprobs", request.py_return_context_logits),
            ("return_generation_logits", request.py_return_generation_logits),
        )
        unsupported_outputs = [name for name, requested in requested_outputs if requested]
        if unsupported_outputs:
            raise ValueError(
                "The following output options are not supported with speculative decoding: "
                f"{', '.join(unsupported_outputs)}. Drop these options from "
                "the request, or disable speculative decoding."
            )

        if request.py_return_log_probs:
            params = OneModelSamplingParams.of_request(request)
            reason = _logprobs_rejection_reason(
                num_logprobs=request.py_num_logprobs,
                simple_format=request.py_logprobs_simple_format,
                processed=request.py_logprobs_mode == LogprobMode.PROCESSED,
                sampled_temperature_only=params.is_temperature_only,
                has_penalty_or_edits=has_occurrence_penalty(request)
                or logit_edits.request_has_logit_edits(request),
                deploy_reason=self._target_logprobs_unsupported,
            )
            if reason is not None:
                raise ValueError(reason)

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
        self._validate_unsupported_logits_processors(
            request,
            logit_edits_unsupported=getattr(
                self, "_logit_edits_unsupported", LOGIT_EDITS_DISABLED_MSG
            ),
        )
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
        # so reject until tree-aware prefixes are implemented. Relaxed acceptance
        # and SA verify outside the pass that applies the penalties.
        if self._penalty_unsupported_reason is not None:
            raise ValueError(self._penalty_unsupported_reason)

    @staticmethod
    def _validate_unsupported_logits_processors(
        request: LlmRequest,
        *,
        logit_edits_unsupported: Optional[str] = LOGIT_EDITS_DISABLED_MSG,
    ) -> None:
        """Reject logits-side sampling features the one-model path cannot apply.

        TorchSampler implements min_length, bad_words, no_repeat_ngram_size,
        embedding_bias, top_p_decay and logits post-processor callbacks by editing
        the target logits before sampling. The one-model path applies embedding_bias and
        bad_words inside the worker where the deploy supports it (``logit_edits_unsupported``
        is None) and the request fits the per-slot tables; everything else is
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
        if logit_edits.request_has_logit_edits(request):
            reason = logit_edits_unsupported or logit_edits.rejection_reason(
                getattr(request, "py_embedding_bias", None),
                getattr(request, "py_bad_words", None),
            )
            if reason is not None:
                raise ValueError(reason)
        if getattr(request, "py_no_repeat_ngram_size", None):
            raise ValueError(UNSUPPORTED_NO_REPEAT_NGRAM_MSG)
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
        penalty_unsupported_reason: Optional[str] = None,
        fused_sampling: bool = False,
        watermark_unsupported_reason: Optional[str] = None,
        logit_edits_unsupported: Optional[str] = LOGIT_EDITS_DISABLED_MSG,
        target_logprobs_unsupported: Optional[str] = None,
    ):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            enable_penalty: whether the deploy enabled the occurrence penalties.
                Only used to decide whether a request asking for them is admitted;
                the penalties themselves are applied inside the worker.
            penalty_unsupported_reason: why this deploy cannot serve the occurrence
                penalties even when it enabled them (None when it can).
            fused_sampling: whether the configured sampler supports min_p.
            watermark_unsupported_reason: why this speculation mode cannot serve watermarked
                requests (None when it can). Such requests are rejected at admission.
            logit_edits_unsupported: why this deploy cannot serve embedding_bias and
                bad_words (None when it can). Such requests are rejected at admission;
                the edits themselves are applied inside the worker.
            target_logprobs_unsupported: why this deploy cannot return target logprobs
                (None when it can). Such requests are rejected at admission.
        """
        self._enable_penalty = enable_penalty
        self._penalty_unsupported_reason = penalty_unsupported_reason
        self._watermark_config = args.watermark_config
        self._watermark_unsupported_reason = watermark_unsupported_reason
        self._fused_sampling = fused_sampling
        self._logit_edits_unsupported = logit_edits_unsupported
        self._target_logprobs_unsupported = target_logprobs_unsupported
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
        self._slot_table_slots: Optional[list[int]] = None
        if spec_step_copies.slot_scatter_supported():
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

    def _copy_target_logprobs_to_host(
        self,
        outputs: dict[str, torch.Tensor | bool],
        sampling_requests: list[LlmRequest],
        num_skip: int,
    ) -> Optional[torch.Tensor]:
        """Host copy of the target logprobs of the sampling requests, or None when no request asked.

        The scores are a forward output the next replay rewrites, so they are copied on the main stream like
        ``verify_lens``. A batch without a logprob request copies nothing.
        """
        if not any(request.py_return_log_probs for request in sampling_requests):
            return None
        scores = outputs.get(TARGET_LOGPROBS_KEY)
        if not isinstance(scores, torch.Tensor):
            raise ValueError(
                "this step's forward did not score the emitted tokens: logprobs need dense, full-vocabulary "
                "target logits and a linear, non-ragged verification. A model that verifies on vocabulary "
                "shards (for example with a fused sharded verify), ragged verification windows and tree "
                "speculation are not supported yet"
            )
        return self._copy_to_host(scores[num_skip : num_skip + len(sampling_requests)])

    def bind_spec_step_fold(self, spec_step_fold: SpecStepFold) -> None:
        """Let the model's forward store into this sampler's per-slot store (see SpecStepFold)."""
        self.spec_step_fold = spec_step_fold
        spec_step_fold.bind_sampler_store(self.store)

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

        max_accepted_path_len = self.max_accepted_path_len
        max_seq_len = self.max_seq_len
        draft_lens = state.draft_lens
        target_logprobs = (
            state.target_logprobs.tolist() if state.target_logprobs is not None else None
        )
        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            seq_slot = req.py_seq_slot
            assert seq_slot is not None
            num_new_tokens = new_tokens_lens_list[seq_slot]
            # new_tokens is sized to this bound; fail clearly, not with an IndexError.
            assert num_new_tokens <= max_accepted_path_len, (
                f"accepted {num_new_tokens} tokens in one step, but new_tokens is "
                f"sized for {max_accepted_path_len}"
            )
            # handle_stop_criteria inlined, with the per-request reads hoisted.
            end_id = req.py_end_id
            room = max_token_room(req, max_seq_len, beam_idx)
            stop_words = req.py_stop_words_list
            appended = 0
            for i in range(num_new_tokens):
                new_token = new_tokens[i][seq_slot][beam_idx]
                req.add_new_token(new_token, beam_idx)
                appended = i + 1
                if new_token == end_id:
                    req.finish_by(FinishReason.END_ID, beam_idx)
                    break
                if i + 1 >= room:
                    req.finish_by(FinishReason.LENGTH, beam_idx)
                    break
                if stop_words and meet_stop_token_criteria(req, new_token, beam_idx):
                    req.finish_by(FinishReason.STOP_WORDS, beam_idx)
                    break
            if target_logprobs is not None and req.py_return_log_probs:
                # One score per token appended: a stop token ends the loop above before every accepted token is.
                req.py_result.append_log_probs([target_logprobs[req_idx][:appended]])
            num_accepted = num_new_tokens - 1
            req.py_num_accepted_draft_tokens = num_accepted
            # Pair acceptance with the SAME step's draft count, snapshotted at sample_async time
            # (0 at prefill): py_draft_tokens is replaced below with the next step's buffer.
            verified_len = (
                token_windows[req_idx] - 1 if token_windows is not None else runtime_draft_len
            )
            drafted_len = draft_lens[req_idx] if draft_lens is not None else 0
            req.py_num_draft_tokens_verified = min(drafted_len, verified_len)
            req.py_rewind_len = verified_len - num_accepted
            req.py_draft_tokens = next_draft_tokens_list[seq_slot][:runtime_draft_len]
            req.py_decoding_iter += 1

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

    def _scatter_on_side_stream(
        self, outputs: dict[str, torch.Tensor], num_skip: int, slots: list[int]
    ) -> bool:
        """Update the stores from the outputs on the D2H side stream; False, writing nothing, if
        the store update kernel does not take the step."""
        side = self._d2h_side_stream
        # The update reads the outputs this step's forward wrote on the current stream.
        side.wait_stream(current_cuda_stream())
        with self._d2h_side_stream_ctx:
            # A step whose batch is unchanged leaves the slot table as it is.
            if slots != self._slot_table_slots:
                self._slot_table_buffer.update(torch.tensor(slots, dtype=torch.int32))
                self._slot_table_slots = slots
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
        if scattered:
            for name in ("new_tokens", "next_new_tokens", "new_tokens_lens", "next_draft_tokens"):
                # Keep the caching allocator from reusing the outputs before the side stream read them.
                _record_stream(outputs[name], side)
        return scattered

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
        slots = [r.py_seq_slot for r in sampling_requests]
        step_fold = self.spec_step_fold
        stores_ready, rows = None, None
        scattered = False
        if (
            self.stores_on_side_stream
            and step_fold is None
            and self._store_scatter is not None
            and o_verify_lens is None
            and not self._async_worker_active()
        ):
            scattered = self._scatter_on_side_stream(outputs, num_skip, slots)
            if scattered:
                stores_ready = torch.cuda.Event()
                stores_ready.record(self._d2h_side_stream)
                if num_skip == 0 and not finished_context_requests:
                    n = len(slots)
                    rows = StepOutputRows(
                        next_new_tokens=outputs["next_new_tokens"][:n],
                        new_tokens_lens=outputs["new_tokens_lens"][:n],
                        next_draft_tokens=outputs["next_draft_tokens"][:n],
                        slots=tuple(slots),
                    )
        if not scattered and self._store_copies_done is not None:
            # The previous step's host copies may still be reading the stores.
            current_cuda_stream().wait_event(self._store_copies_done)
        forward_stored_rows = False
        if not scattered and step_fold is not None:
            scattered = forward_stored_rows = step_fold.sampler_rows_stored(
                outputs, num_skip, slots
            )
        # A model's step fold scatters the stores itself, in _scatter_to_stores.
        if not scattered and step_fold is None and self._store_scatter is not None:
            # A step whose batch is unchanged leaves the slot table as it is.
            if slots != self._slot_table_slots:
                self._slot_table_buffer.update(torch.tensor(slots, dtype=torch.int32))
                self._slot_table_slots = slots
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
            stores_ready=stores_ready,
            rows=rows,
        )

        store_copies_done = getattr(step_fold, "store_copies_done", None) if step_fold else None
        host_tensors, side_stream_event = self._copy_outputs_to_host(
            o_verify_lens, forward_stored_rows, store_copies_done
        )
        host_target_logprobs = self._copy_target_logprobs_to_host(
            outputs, sampling_requests, num_skip
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
            target_logprobs=host_target_logprobs,
        )
