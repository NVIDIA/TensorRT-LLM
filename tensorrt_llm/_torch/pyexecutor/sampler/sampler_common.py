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

"""Shared infrastructure for the sampler package.

The package's base layer: it imports nothing from its siblings, so anything
here is safe for every other module to depend on. Holds what the feature
modules (token bans, top-p decay, finish reasons, beam search, penalties,
log-probs) build on:

* the per-request queries that read an ``LlmRequest``'s sampling config into
  :class:`UtilsSamplingParams`, plus the predicates over it
  (``top_p_decay_active``, ``_request_sampling_params_cachable``);
* the shared step/beam index constants and beam-width accessors;
* tensor helpers (``int_tensor``, ``add_token``);
* plain data types passed *between* modules that no single feature owns --
  :class:`RequestSeeds` (seed manager -> strategy impls) and
  :class:`_BatchedSamplingResult` (sampler -> log-probs).

That last group follows the package's dependency rule: a type shared across
features belongs here, so that a lower-layer module never has to import a
higher-layer one to name it. A type owned by a single feature belongs with that
feature instead -- the ``*Store`` classes all live in their own modules.

Resolving a request's ``Strategy`` lives in ``sampler_strategy``.
"""

from dataclasses import dataclass, field
from typing import Optional, TypeAlias, cast

import torch

from tensorrt_llm.sampling_params import SamplingParams

from ..llm_request import LlmRequest

# Beam index to use when no beam search is used but a beam index is required
DEFAULT_BEAM_IDX = 0
# Step index to use when no speculative decoding is used but a step index is required
DEFAULT_STEP_IDX = 0

FinishReasonsList: TypeAlias = list[list[list[int]]]


@dataclass(frozen=True, kw_only=True)
class UtilsSamplingParams:
    """Subset of tensorrt_llm::runtime::SamplingConfig supported by the torch sampler.

    Args:
        temperature: The temperature to use for sampling.
        top_p: The top-p to use for sampling.
        top_k: The top-k to use for sampling.
        min_p: The min-p to use for sampling.
        use_beam_search: Whether to use beam search.
        beam_width_in: The beam_width of a request before the sampling step.
        beam_width_out: The beam_width of a request after the sampling step.
        top_p_decay: Per-step multiplicative decay applied to the runtime top-p.
        top_p_min: Lower bound for the decayed runtime top-p.
        top_p_reset_ids: Token id which, when sampled, resets the runtime top-p to
            its initial value. A value < 0 never matches a token.
        length_penalty: Beam-search length penalty exponent; scores are
            normalized as cum_log_prob / length**length_penalty. 0 disables.
        beam_search_diversity_rate: Beam-search diversity adjustment; adds
            rate * source_beam_index to the candidate ranking score. 0 disables.
        early_stopping: Beam-search stopping mode; see ``BeamSearchEarlyStop``.
            ``TRUE`` (1, default) stops as soon as beam_width finished
            candidates exist; ``FALSE`` (0) and ``NEVER`` (2) are the
            exhaustive modes backed by the candidate-beams array. ``FALSE``
            bounds a beam's best attainable score by its current score (assume
            scores decrease monotonically with sequence length); ``NEVER``
            places no upper bound on attainability for unfinished beams (assume
            scores can increase with length, e.g. when length_penalty > 0).
    """

    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    use_beam_search: Optional[bool]
    min_p: Optional[float] = None
    beam_width_in: Optional[int] = None
    beam_width_out: Optional[int] = None
    # Rows the forward path allocated per request (static admission width).
    # Equals beam_width_in unless the request uses a variable beam width array.
    row_stride: Optional[int] = None
    top_p_decay: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    length_penalty: Optional[float] = None
    beam_search_diversity_rate: Optional[float] = None
    early_stopping: Optional[int] = None


def int_tensor(shape: tuple[int, ...], device: str = "cuda") -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int, device=device)


def add_token(
    request: LlmRequest, new_tokens: list[list[list[int]]], *, beam_idx: int, step: int = 0
) -> int:
    # NB: Accessing nested lists faster than torch.Tensor or numpy.ndarray
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    new_token = new_tokens[step][seq_slot][beam_idx]
    request.add_new_token(new_token, beam_idx)
    return new_token


def _get_beam_width_in(request: LlmRequest) -> int:
    return (
        1
        if request.is_context_init_state
        else request.get_beam_width_by_iter(for_next_iteration=False)
    )


def _get_beam_width_out(request: LlmRequest) -> int:
    return request.get_beam_width_by_iter(for_next_iteration=True)


def _get_max_beam_width(request: LlmRequest) -> int:
    sampling_config = request.sampling_config
    max_beam_width = cast(int, sampling_config.beam_width)
    # The array holds at most kMaxBeamWidthArrayLength (8) entries, so reduce it
    # on the host.
    #
    # An empty array means "no schedule" -- checkBeamWidthArray bounds only the
    # array's length, so an empty one passes admission -- and must fall through
    # to beam_width rather than reduce over nothing. Hence truthiness rather
    # than `is not None`, the same check LlmRequest.get_beam_width_by_iter and
    # PyExecutor._validate_request make.
    beam_width_array = sampling_config.beam_width_array
    if beam_width_array:
        max_beam_width = max(max_beam_width, *map(int, beam_width_array))
    return max_beam_width


def request_random_seed(request: LlmRequest) -> Optional[int]:
    """The request's user-specified ``SamplingParams.seed``, if any.

    Deliberately kept out of ``UtilsSamplingParams``: that struct backs the
    sampling-strategy cache key, and a per-request seed there would make every
    seeded request its own strategy group, defeating batched sampling. The seed
    does not change *which* distribution is sampled, only the RNG stream, so it
    is read separately and applied as per-row RNG state.

    The C++ ``SamplingConfig.randomSeed`` is ``uint64``, so a user seed in
    ``[2**63, 2**64)`` (e.g. from ``random.getrandbits(64)``) does not fit the
    int64 tensors the torch sampler stores it in. Reinterpret the bit pattern as
    signed rather than rejecting or clamping: the RNG consumes the seed as an
    opaque bit pattern, so this keeps distinct seeds distinct, and it avoids an
    overflow that would abort the whole sampling step rather than one request.
    """
    seed = request.sampling_config.seed
    if seed is None:
        return None
    return seed - (1 << 64) if seed >= (1 << 63) else seed


def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
    sampling_config = request.sampling_config
    temperature = sampling_config.temperature
    top_p = sampling_config.top_p
    top_k = sampling_config.top_k
    min_p = sampling_config.min_p
    top_p_decay = sampling_config.top_p_decay
    top_p_min = sampling_config.top_p_min
    top_p_reset_ids = sampling_config.top_p_reset_ids
    beam_width_out = _get_beam_width_out(request)
    beam_width_in = _get_beam_width_in(request)
    # ModelEngine lays generation rows out at the static admission width; see
    # the row_stride note in _beam_step_preprocess.
    row_stride = 1 if request.is_context_init_state else request.py_beam_width
    use_beam_search = _get_max_beam_width(request) > 1
    length_penalty = sampling_config.length_penalty
    beam_search_diversity_rate = sampling_config.beam_search_diversity_rate
    early_stopping = sampling_config.early_stopping

    return UtilsSamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        beam_width_in=beam_width_in,
        beam_width_out=beam_width_out,
        row_stride=row_stride,
        use_beam_search=use_beam_search,
        top_p_decay=top_p_decay,
        top_p_min=top_p_min,
        top_p_reset_ids=top_p_reset_ids,
        length_penalty=length_penalty,
        beam_search_diversity_rate=beam_search_diversity_rate,
        early_stopping=early_stopping,
    )


def _request_sampling_params_cachable(params: UtilsSamplingParams) -> bool:
    return not params.use_beam_search


@dataclass(kw_only=True)
class RequestSeeds:
    """Per-request RNG state for user-specified ``SamplingParams.seed``.

    Threaded alongside ``generator`` through the strategy impls and handed to
    the flashinfer sampling ops as their stateless ``seed``/``offset`` pair.
    Both tensors are int64 and 1-D with one entry per group row, matching the
    per-row shape flashinfer documents; a row whose request did not specify a
    seed carries the sampler's global seed, so unseeded requests keep their
    previous behavior only in distribution, not token-for-token (see
    ``_SeedManager``).

    NB: the pinned flashinfer (0.6.15) accepts these per-row tensors but reads
    only element 0 of each, separating rows by ``blockIdx.x``. The per-row
    values below are therefore carried end-to-end but not yet honored for
    batched requests; see the warning on ``_SeedManager`` and the upstream fix
    at https://github.com/flashinfer-ai/flashinfer/pull/2345.

    ``offset`` advances per request per sampling step, which is what makes a
    seeded request's stream depend on how many tokens it has drawn rather than
    on which batch it happened to land in.
    """

    seed: torch.Tensor
    """Per-row Philox seed (int64, device)."""
    offset: torch.Tensor
    """Per-row Philox offset (int64, device)."""

    def index_select(self, indices: torch.Tensor) -> "RequestSeeds":
        """Narrow to a subset of rows, mirroring ``group_logit_indices``."""
        return RequestSeeds(
            seed=self.seed.index_select(0, indices),
            offset=self.offset.index_select(0, indices),
        )


def top_p_decay_active(params: UtilsSamplingParams) -> bool:
    """Whether dynamic top-p decay is active for a request.

    Delegates to the single-source predicate on SamplingParams; note that
    ``top_p_min`` / ``top_p_reset_ids`` alone do not activate dynamic behavior.
    """
    return SamplingParams.params_imply_top_p_decay_active(params.top_p_decay)


@dataclass(kw_only=True, frozen=True)
class _BatchedSamplingResult:
    # Original request indices for all requests (permuted due to batching by strategy):
    req_indices: torch.Tensor
    # Next tokens for all requests:
    next_tokens_cuda_int: torch.Tensor

    # Processed and raw logprobs buffer. The tensor is sized to accommodate logprobs for all requests currently being
    # processed by the sampler and slice(0, processed_logprobs_end) contains processed logprobs, ordered consistently
    # with processed_logprobs_reqs_indices. Excludes beam search requests, which have a separate path for logprobs
    # handling.
    logprobs_cuda: torch.Tensor | None = None

    # Requests requesting processed logprobs (incl. beam-search requests), same ordering as req_indices.
    processed_logprobs_reqs_indices: list[int] = field(default_factory=list)
    # Index of first unused row of logprobs_cuda
    processed_logprobs_end: int = 0

    # Requests requesting raw logprobs (incl. beam-search requests), ordered consistently with original
    # (unpermuted) requests
    raw_logprobs_reqs_indices: list[int] = field(default_factory=list)
    # Indices into logits tensor, ordered consistently with raw_logprobs_reqs_indices.
    # Excludes beam search requests, which have a separate path for logprobs handling.
    raw_logprobs_logit_indices_cuda: torch.Tensor | None = None
