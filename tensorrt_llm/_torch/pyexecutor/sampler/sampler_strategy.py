# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

"""Sampling strategies: what to draw, and how to get there from a request.

Holds the :data:`Strategy` types, their implementations and the grouped
samplers, which operate on logits and probs; ``_request_strategy`` maps an
``LlmRequest`` onto a strategy, reading its sampling config via
``sampler_common``.
"""

import abc
import sys
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, Type, TypeAlias, TypeVar, cast

import torch

from tensorrt_llm._torch.pyexecutor.sampler.beam_search import (
    BEAM_SEARCH_PAD_TOKEN,
    BeamHistory,
    BeamSearchEarlyStop,
    BeamSearchMetadata,
    BeamSearchStore,
    CBAState,
    beam_search_sampling_batch_cba,
)

# These op wrappers are safe to import without flashinfer installed; each one
# resolves the flashinfer symbol only when called. Most are reached only on the
# flashinfer sampler / speculative-worker paths, but radix_topk_op also backs
# beam search's wide-row top-k, which guards on IS_FLASHINFER_AVAILABLE and
# falls back to torch.topk (see beam_search._beam_topk).
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    sampling_from_probs_op,
    sanitize_top_k,
    softmax_op,
    top_k_mask_logits_op,
    top_k_renorm_probs_op,
    top_k_sampling_from_probs_op,
    top_k_top_p_sampling_from_logits_op,
    top_k_top_p_sampling_from_probs_op,
    top_p_renorm_probs_op,
    top_p_sampling_from_probs_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    GREEDY_TEMPERATURE_THRESHOLD,
    Fusions,
    StrategyMetadata,
    get_rejected_indices,
    greedy_search_sampling_batch,
    min_p_renorm_probs,
    sample_rejected,
    top_k_top_p_sampling_batch,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.sampling_params import SamplingParams

from ..llm_request import LlmRequest
from .sampler_common import (
    UtilsSamplingParams,
    _request_get_sampling_params,
    _request_sampling_params_cachable,
)

# Ops imported above are re-exported for dependent modules (sampler, drafting
# loops, tests). mypy runs in strict mode (no implicit re-export), so they must
# be listed here.
__all__ = [
    "BEAM_SEARCH_PAD_TOKEN",
    "GREEDY_TEMPERATURE_THRESHOLD",
    "BeamHistory",
    "BeamSearchEarlyStop",
    "BeamSearchMetadata",
    "BeamSearchStore",
    "CBAState",
    "Fusions",
    "StrategyMetadata",
    "beam_search_sampling_batch_cba",
    "get_rejected_indices",
    "greedy_search_sampling_batch",
    "sample_rejected",
    "sampling_from_probs_op",
    "softmax_op",
    "top_k_mask_logits_op",
    "top_k_sampling_from_probs_op",
    "top_k_top_p_sampling_batch",
    "top_k_top_p_sampling_from_logits_op",
    "top_k_top_p_sampling_from_probs_op",
    "top_p_renorm_probs_op",
    "top_p_sampling_from_probs_op",
]

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


TemperatureOnly: TypeAlias = tuple[Literal["temperature"], float]
TopK: TypeAlias = tuple[Literal["top_k"], int, float]
TopP: TypeAlias = tuple[Literal["top_p"], float, float]
TopKTopP: TypeAlias = tuple[Literal["top_k_top_p"], int, float, float]
# (tag, top_k, top_p, min_p, temperature)
MinP: TypeAlias = tuple[Literal["min_p"], int, float, float, float]
Greedy: TypeAlias = tuple[Literal["greedy"], None]


class BeamSearch(NamedTuple):
    """Beam-search strategy tuple. A NamedTuple (not a bare tuple alias) so the
    six numeric fields are self-documenting; it still matches ``case
    ("beam_search", ...)`` sequence patterns and indexes like the other
    strategy tuples."""

    tag: Literal["beam_search"]
    beam_width_in: int
    beam_width_out: int
    temperature: float
    length_penalty: float
    diversity_rate: float
    early_stopping: BeamSearchEarlyStop
    # Appended last on purpose: _common_fields() reads the fields above by
    # position, so inserting earlier would shift those indices.
    row_stride: int = 0


GREEDY: Greedy = ("greedy", None)

Strategy: TypeAlias = TopK | TopP | Greedy | TopKTopP | TemperatureOnly | MinP | BeamSearch


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


def top_p_decay_active(params: UtilsSamplingParams) -> bool:
    """Whether dynamic top-p decay is active for a request.

    Delegates to the single-source predicate on SamplingParams; note that
    ``top_p_min`` / ``top_p_reset_ids`` alone do not activate dynamic behavior.
    """
    return SamplingParams.params_imply_top_p_decay_active(params.top_p_decay)


def resolve_sampling_strategy(params: UtilsSamplingParams, *, vocab_size: int) -> Strategy:
    # The semantics are specified in the doc-string of SamplingParams

    use_beam_search = params.use_beam_search
    temperature = params.temperature
    top_p = params.top_p
    top_k = params.top_k
    min_p = params.min_p

    # The greedy verdict (including the top-p-decay override of the implicit
    # all-unset greedy default, and explicit greedy controls winning over decay)
    # is single-sourced in SamplingParams.params_imply_greedy_decoding.
    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_beam_search=use_beam_search,
        min_p=min_p,
        top_p_decay=params.top_p_decay,
    ):
        return GREEDY

    # --- resolving default values
    # NB: not greedy, hence temperature != 0 if specified
    temperature = temperature or 1.0

    # Beam search does not rely on top_p, top_k or min_p, so we can return the strategy here
    if use_beam_search:
        assert params.beam_width_in is not None and params.beam_width_out is not None, (
            "beam_width_in and beam_width_out must be specified for beam search"
        )
        return BeamSearch(
            tag="beam_search",
            beam_width_in=params.beam_width_in,
            beam_width_out=params.beam_width_out,
            temperature=temperature,
            length_penalty=params.length_penalty or 0.0,
            diversity_rate=params.beam_search_diversity_rate or 0.0,
            early_stopping=BeamSearchEarlyStop.from_raw(params.early_stopping),
            row_stride=params.row_stride or params.beam_width_in,
        )

    # NB: not greedy, hence top_p != 0 if specified
    top_p = top_p or 1.0
    # NB: not greedy, hence top_k != 1 if specified
    #     (0 and vocab_size are equivalent)
    top_k = top_k or vocab_size

    assert top_k > 1, "non-greedy sampling requires valid top_k"
    need_top_k = top_k < vocab_size
    assert top_p > 0, "non-greedy sampling requires valid top_p"
    # A decay-active request must go through a top-p-capable path even when its
    # initial top_p is 1.0, so the runtime top-p (sourced per-row at sample time)
    # can shrink the nucleus on later steps.
    need_top_p = top_p < 1 or top_p_decay_active(params)

    # Disabled top_k is 0 ("keep all"), not vocab_size, which can be the
    # fast-greedy probe (2**31) and overflow the int32 tensor; _compute_probs
    # sanitizes it.
    if min_p is not None and min_p > 0.0:
        return ("min_p", top_k if need_top_k else 0, top_p, min_p, temperature)

    if need_top_p:
        if need_top_k:
            return ("top_k_top_p", top_k, top_p, temperature)
        return ("top_p", top_p, temperature)
    if need_top_k:
        return ("top_k", top_k, temperature)
    return ("temperature", temperature)


def sample(
    strategy: Strategy,
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    group_metadata: StrategyMetadata | None = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    softmax: torch.Tensor | None
    # 'cast' needed b/c of https://github.com/python/mypy/issues/19081
    match strategy:
        case ("top_k", top_k, temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                top_k=cast(int, top_k),
                temperature=cast(float, temperature),
                generator=generator,
            )
        case ("top_p", top_p, temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                top_p=cast(float, top_p),
                generator=generator,
                temperature=cast(float, temperature),
            )
        case ("top_k_top_p", top_k, top_p, temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                top_k=cast(int, top_k),
                top_p=cast(float, top_p),
                temperature=cast(float, temperature),
                generator=generator,
            )
        case ("temperature", temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                temperature=cast(float, temperature),
                generator=generator,
            )
        case ("min_p", top_k, top_p, min_p, temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                top_k=cast(int, top_k),
                top_p=cast(float, top_p),
                min_p=cast(float, min_p),
                temperature=cast(float, temperature),
                generator=generator,
            )
        case ("greedy", None):
            tokens, softmax = greedy_search_sampling_batch(logits, return_probs=return_probs)
            # Returns instead of falling through: the other patterns bind
            # `temperature` as `float`, so assigning None here does not type check.
            return tokens, softmax, None
        case (
            "beam_search",
            beam_width_in,
            beam_width_out,
            temperature,
            length_penalty,
            beam_search_diversity_rate,
            early_stopping,
            *_,
        ):
            row_stride = cast(BeamSearch, strategy).row_stride
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata), (
                "BeamSearchMetadata is required for beam search"
            )
            # Every early_stopping mode goes through the candidate-beams-array
            # path: TRUE differs only in the done verdict (pool full, without
            # weighing attainability), matching the C++ decoder, which keeps
            # the pool for all modes.
            tokens, softmax = beam_search_sampling_batch_cba(
                logits,
                beam_width_in=cast(int, beam_width_in),
                beam_width_out=cast(int, beam_width_out),
                row_stride=row_stride,
                beam_search_args=group_metadata,
                temperature=cast(float, temperature),
                early_stopping=cast(int, early_stopping),
                length_penalty=cast(float, length_penalty),
                diversity_rate=cast(float, beam_search_diversity_rate),
                return_probs=return_probs,
            )
    return tokens, softmax, cast(float, temperature)


GenericStrategyKeyType = TypeVar("GenericStrategyKeyType", bound=Hashable)


class _StrategyImpls:
    class StrategyImpl(abc.ABC):
        @classmethod
        @abc.abstractmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.StrategyImpl":
            pass

        @classmethod
        @abc.abstractmethod
        def computes_probs(cls) -> bool:
            pass

        def get_temperature(self) -> torch.Tensor | None:
            return getattr(self, "_temperature", None)

        @abc.abstractmethod
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            pass

        # TODO: Revisit this after determining performance impact
        #
        # NB: NaN logits can lead to crashes, see
        #     https://github.com/flashinfer-ai/flashinfer/issues/1575
        #
        @staticmethod
        def _flashinfer_check_nans(inputs: torch.Tensor) -> bool:
            # Deliberately returns False to keep FlashInfer's own 'check_nan' path
            # disabled: that path is a host-side `if torch.any(torch.isnan(...))`,
            # which forces a device sync on every call. The explicit async
            # device-side assert below provides the same protection without
            # stalling the pipeline.
            # https://github.com/pytorch/pytorch/issues/36853
            torch._assert_async(~torch.any(torch.isnan(inputs)))
            return False

        @staticmethod
        def _make_tensor(data: list[Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            return torch.tensor(data, dtype=dtype, pin_memory=prefer_pinned()).to(
                device=device, non_blocking=True
            )

        @staticmethod
        def _prepare_logits_with_temperature(
            logits: torch.Tensor,
            group_logit_indices: Optional[torch.Tensor],
            temperature: torch.Tensor,
        ) -> torch.Tensor:
            temperature = temperature.unsqueeze(-1)
            if group_logit_indices is not None:
                logits = torch.index_select(logits, 0, group_logit_indices)
                logits /= temperature
            else:
                logits = logits / temperature
            return logits

        @staticmethod
        def _prepare_probs_with_temperature(
            logits: torch.Tensor,
            group_logit_indices: Optional[torch.Tensor],
            temperature: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if group_logit_indices is not None:
                logits = logits[group_logit_indices]
            return softmax_op(logits, temperature)

        @classmethod
        def _sample_from_probs(
            cls,
            probs: torch.Tensor,
            generator: Optional[torch.Generator],
            seeds: Optional["RequestSeeds"] = None,
        ) -> torch.Tensor:
            # Explicit seed/offset take precedence over generator in the op
            # layer, so passing both is safe and keeps the generator as the
            # fallback when no request asked for a seed.
            return sampling_from_probs_op(
                probs,
                generator=generator,
                seed=seeds.seed if seeds is not None else None,
                offset=seeds.offset if seeds is not None else None,
                check_nan=cls._flashinfer_check_nans(probs),
            )

        def _sample_greedy_with_probs(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if group_logit_indices is not None:
                logits = torch.index_select(logits, 0, group_logit_indices)
            tokens = torch.argmax(logits, dim=-1)
            probs = torch.zeros_like(logits)
            probs.scatter_(1, tokens.unsqueeze(-1), 1.0)
            return tokens, probs

        @classmethod
        def _compute_probs(
            cls,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor],
            top_k: Optional[torch.Tensor],
            top_p: Optional[torch.Tensor],
            min_p: Optional[torch.Tensor],
            temperature: torch.Tensor,
        ) -> torch.Tensor:
            """Temperature + softmax + optional min-p / top-k / top-p renorm.

            min_p runs first (vLLM semantics): its threshold is relative to the
            max probability of the unfiltered row, and top_k/top_p renormalize,
            which inflates that max and would make a later min_p stricter.
            """
            probs = cls._prepare_probs_with_temperature(logits, group_logit_indices, temperature)
            if min_p is not None:
                probs = min_p_renorm_probs(probs, min_p)

            if top_k is not None:
                top_k = sanitize_top_k(top_k, probs.shape[-1])
                probs = top_k_renorm_probs_op(probs, top_k)

            if top_p is not None:
                probs = top_p_renorm_probs_op(probs, top_p)

            return probs

        @classmethod
        def _sample_with_probs(
            cls,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor],
            top_k: Optional[torch.Tensor],
            top_p: Optional[torch.Tensor],
            min_p: Optional[torch.Tensor],
            temperature: torch.Tensor,
            generator: Optional[torch.Generator],
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            probs = cls._compute_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
            )
            new_tokens = cls._sample_from_probs(probs, generator=generator, seeds=seeds)
            return new_tokens, probs

    class TopPDecayMixin:
        """Mixed into the TopP*/TopKTopP*/MinP* impls (the owners of a per-row
        ``_top_p`` tensor) to consume ``TopPDecayMetadata``."""

        _top_p: torch.Tensor

        def _maybe_apply_top_p_decay(self, group_metadata: Optional[StrategyMetadata]) -> None:
            """Override the per-row static top-p with the decayed runtime top-p.

            Only decay-active rows (per the on-device ``is_decay_slot`` gate) are
            overridden, so a group mixing top-p-decay and plain top-p requests
            keeps each row's correct value. The overridden ``self._top_p`` tensor
            then feeds both sampling and ``top_p_renorm_probs_op`` (so processed
            logprobs match). Fused via torch.compile (gather + gate + select).
            """
            if not isinstance(group_metadata, TopPDecayMetadata):
                return
            assert self._top_p.shape == group_metadata.slots.shape, (
                self._top_p.shape,
                group_metadata.slots.shape,
            )
            self._top_p = Fusions.top_p_decay_gather(
                runtime_top_p=group_metadata.runtime_top_p,
                is_decay_slot=group_metadata.is_decay_slot,
                static_top_p=self._top_p,
                slots=group_metadata.slots,
            )

    class StrategyImplWithProbs(StrategyImpl):
        @override
        @classmethod
        def computes_probs(cls) -> bool:
            return True

    class GreedyWithProbs(StrategyImplWithProbs):
        def __init__(self) -> None:
            self._temperature = None

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.GreedyWithProbs":
            return cls()

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self._sample_greedy_with_probs(logits, group_logit_indices=group_logit_indices)

    class TopKTopPWithProbs(TopPDecayMixin, StrategyImplWithProbs):
        def __init__(self, top_k: torch.Tensor, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKTopPWithProbs":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.int32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[3] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=self._top_p,
                min_p=None,
                temperature=self._temperature,
                generator=generator,
                seeds=seeds,
            )

    class TopKWithProbs(StrategyImplWithProbs):
        def __init__(self, top_k: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKWithProbs":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.int32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=None,
                min_p=None,
                temperature=self._temperature,
                generator=generator,
                seeds=seeds,
            )

    class TopPWithProbs(TopPDecayMixin, StrategyImplWithProbs):
        def __init__(self, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TopPWithProbs":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=self._top_p,
                min_p=None,
                temperature=self._temperature,
                generator=generator,
                seeds=seeds,
            )

    class TemperatureOnlyWithProbs(StrategyImplWithProbs):
        def __init__(self, temperature: torch.Tensor):
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TemperatureOnlyWithProbs":
            return cls(cls._make_tensor([s[1] for s in strategies], torch.float32, cuda_device))

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                min_p=None,
                temperature=self._temperature,
                generator=generator,
                seeds=seeds,
            )

    class MinPWithProbs(TopPDecayMixin, StrategyImplWithProbs):
        def __init__(
            self,
            top_k: torch.Tensor,
            top_p: torch.Tensor,
            min_p: torch.Tensor,
            temperature: torch.Tensor,
        ):
            self._top_k = top_k
            self._top_p = top_p
            self._min_p = min_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.MinPWithProbs":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.int32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[3] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[4] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=self._top_p,
                min_p=self._min_p,
                temperature=self._temperature,
                generator=generator,
                seeds=seeds,
            )

    class StrategyImplSampleOnly(StrategyImpl):
        @override
        @classmethod
        def computes_probs(cls) -> bool:
            return False

    class GreedySampleOnly(StrategyImplSampleOnly):
        def __init__(self) -> None:
            self._temperature = None

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.GreedySampleOnly":
            return cls()

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if group_logit_indices is not None:
                logits = logits[group_logit_indices]
            return torch.argmax(logits, dim=-1), None

    class TopKTopPSampleOnly(TopPDecayMixin, StrategyImplSampleOnly):
        def __init__(self, top_k: torch.Tensor, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKTopPSampleOnly":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.int32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[3] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            logits = self._prepare_logits_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_k_top_p_sampling_from_logits_op(
                logits,
                self._top_k,
                self._top_p,
                generator=generator,
                seed=seeds.seed if seeds is not None else None,
                offset=seeds.offset if seeds is not None else None,
                check_nan=self._flashinfer_check_nans(logits),
            ), None

    class TopKSampleOnly(StrategyImplSampleOnly):
        def __init__(self, top_k: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKSampleOnly":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.int32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_k_sampling_from_probs_op(
                probs,
                self._top_k,
                generator=generator,
                seed=seeds.seed if seeds is not None else None,
                offset=seeds.offset if seeds is not None else None,
                check_nan=self._flashinfer_check_nans(probs),
            ), None

    class TopPSampleOnly(TopPDecayMixin, StrategyImplSampleOnly):
        def __init__(self, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TopPSampleOnly":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_p_sampling_from_probs_op(
                probs,
                self._top_p,
                generator=generator,
                seed=seeds.seed if seeds is not None else None,
                offset=seeds.offset if seeds is not None else None,
                check_nan=self._flashinfer_check_nans(probs),
            ), None

    class TemperatureOnlySampleOnly(StrategyImplSampleOnly):
        def __init__(self, temperature: torch.Tensor):
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.TemperatureOnlySampleOnly":
            return cls(cls._make_tensor([s[1] for s in strategies], torch.float32, cuda_device))

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, _ = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                min_p=None,
                temperature=self._temperature,
                generator=generator,
                seeds=seeds,
            )
            return new_tokens, None

    class MinPSampleOnly(TopPDecayMixin, StrategyImplSampleOnly):
        def __init__(
            self,
            top_k: torch.Tensor,
            top_p: torch.Tensor,
            min_p: torch.Tensor,
            temperature: torch.Tensor,
        ):
            self._top_k = top_k
            self._top_p = top_p
            self._min_p = min_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.MinPSampleOnly":
            return cls(
                cls._make_tensor([s[1] for s in strategies], torch.int32, cuda_device),
                cls._make_tensor([s[2] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[3] for s in strategies], torch.float32, cuda_device),
                cls._make_tensor([s[4] for s in strategies], torch.float32, cuda_device),
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            # With min_p applied first, nothing has to run after top_k/top_p, so
            # the fused kernel can filter and sample in one pass instead of two
            # renorms plus a separate sampling step.
            probs = self._compute_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                min_p=self._min_p,
                temperature=self._temperature,
            )
            return top_k_top_p_sampling_from_probs_op(
                probs,
                sanitize_top_k(self._top_k, probs.shape[-1]),
                self._top_p,
                generator=generator,
                seed=seeds.seed if seeds is not None else None,
                offset=seeds.offset if seeds is not None else None,
                check_nan=self._flashinfer_check_nans(probs),
            ), None

    class BeamSearchStep(StrategyImpl):
        """Base for the beam-search step strategies.

        ``sample`` applies the shared temperature preprocessing and delegates to
        the ``_select_and_update`` hook, implemented per stopping mode by
        ``CBABeamSearchStep``, which every stopping mode uses -- the mode only
        selects the done verdict computed inside it. With-probs is a constructor flag
        (``computes_probs``), not a subclass.
        """

        @dataclass(frozen=True, kw_only=True)
        class CommonFields:
            """Constructor arguments shared by all beam-search step strategies."""

            beam_width_in: int
            beam_width_out: int
            row_stride: int
            temperature: torch.Tensor
            length_penalty: Optional[torch.Tensor]
            diversity_rate: Optional[torch.Tensor]

        def __init__(
            self,
            beam_width_in: int,
            beam_width_out: int,
            row_stride: int,
            temperature: torch.Tensor,
            length_penalty: Optional[torch.Tensor],
            diversity_rate: Optional[torch.Tensor],
            *,
            computes_probs: bool = False,
        ):
            self._beam_width_in = beam_width_in
            self._beam_width_out = beam_width_out
            self._row_stride = row_stride
            self._temperature = temperature
            self._length_penalty = length_penalty
            self._diversity_rate = diversity_rate
            self._computes_probs = computes_probs

        @override
        def computes_probs(self) -> bool:  # type: ignore[override]
            # Instance flag, not a per-subclass constant: beam search has no
            # separate with-probs sampling path.
            return self._computes_probs

        def with_computes_probs(self, computes_probs: bool) -> "_StrategyImpls.BeamSearchStep":
            """Set the return-probs flag after construction and return self."""
            self._computes_probs = computes_probs
            return self

        @staticmethod
        def _common_fields(
            strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.BeamSearchStep.CommonFields":
            """Extract the fields shared by every beam-search step strategy.

            Separate from ``from_strategies`` so subclasses can reuse the
            extraction without instantiating this class, which is abstract
            (``_select_and_update``).
            """
            assert all(strat[0] == "beam_search" for strat in strategies)
            narrowed_strats = cast(list[BeamSearch], strategies)
            (beam_width_in,) = set(strat[1] for strat in narrowed_strats)
            (beam_width_out,) = set(strat[2] for strat in narrowed_strats)
            # row_stride is deliberately NOT part of the grouping key (see
            # strategy_grouping_key), so nothing in the grouping guarantees a
            # single value here. It holds because admission pins every request
            # to max_beam_width, making row_stride == py_beam_width identical
            # across a group. Unpack rather than pick one, so that the day
            # narrower requests are admitted this fails loudly instead of
            # silently strideing one request's logits by another's width.
            (row_stride,) = set(strat.row_stride or beam_width_in for strat in narrowed_strats)
            temperature = _StrategyImpls.BeamSearchStep._make_tensor(
                [strat[3] or 1.0 for strat in narrowed_strats], torch.float32, cuda_device
            )
            length_penalties = [strat[4] or 0.0 for strat in narrowed_strats]
            length_penalty: Optional[torch.Tensor] = None
            if any(lp != 0.0 for lp in length_penalties):
                length_penalty = _StrategyImpls.BeamSearchStep._make_tensor(
                    length_penalties, torch.float32, cuda_device
                )
            diversity_rates = [strat[5] or 0.0 for strat in narrowed_strats]
            diversity_rate: Optional[torch.Tensor] = None
            if any(dr != 0.0 for dr in diversity_rates):
                diversity_rate = _StrategyImpls.BeamSearchStep._make_tensor(
                    diversity_rates, torch.float32, cuda_device
                )
            return _StrategyImpls.BeamSearchStep.CommonFields(
                beam_width_in=beam_width_in,
                beam_width_out=beam_width_out,
                row_stride=row_stride,
                temperature=temperature,
                length_penalty=length_penalty,
                diversity_rate=diversity_rate,
            )

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.BeamSearchStep":
            fields = _StrategyImpls.BeamSearchStep._common_fields(strategies, cuda_device)
            return cls(
                fields.beam_width_in,
                fields.beam_width_out,
                fields.row_stride,
                fields.temperature,
                fields.length_penalty,
                fields.diversity_rate,
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
            seeds: Optional["RequestSeeds"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata)
            # Temperature is applied before the op slices the padding rows off,
            # so it must cover every row the forward path laid out: the static
            # admission width, which exceeds beam_width_in while a variable
            # beam width array is still widening.
            temperature = self._temperature.repeat_interleave(self._row_stride)
            logits = self._prepare_logits_with_temperature(logits, group_logit_indices, temperature)
            return self._select_and_update(logits, group_metadata)

        @abc.abstractmethod
        def _select_and_update(
            self, logits: torch.Tensor, group_metadata: BeamSearchMetadata
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Mode-specific candidate selection and state update."""

    class CBABeamSearchStep(BeamSearchStep):
        """The candidate-beams-array step, used by every early_stopping mode.

        The mode only selects the done verdict computed inside the op: TRUE
        stops as soon as the pool is full, FALSE and NEVER additionally weigh
        what is still attainable.
        """

        def __init__(
            self,
            beam_width_in: int,
            beam_width_out: int,
            row_stride: int,
            temperature: torch.Tensor,
            length_penalty: Optional[torch.Tensor],
            diversity_rate: Optional[torch.Tensor],
            early_stopping: BeamSearchEarlyStop,
            *,
            computes_probs: bool = False,
        ):
            super().__init__(
                beam_width_in,
                beam_width_out,
                row_stride,
                temperature,
                length_penalty,
                diversity_rate,
                computes_probs=computes_probs,
            )
            self._early_stopping = early_stopping

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.CBABeamSearchStep":
            fields = _StrategyImpls.BeamSearchStep._common_fields(strategies, cuda_device)
            narrowed_strats = cast(list[BeamSearch], strategies)
            # early_stopping is part of the grouping key, hence unique per group.
            (early_stopping,) = set(strat[6] for strat in narrowed_strats)
            return cls(
                fields.beam_width_in,
                fields.beam_width_out,
                fields.row_stride,
                fields.temperature,
                fields.length_penalty,
                fields.diversity_rate,
                early_stopping,
            )

        @override
        def _select_and_update(
            self, logits: torch.Tensor, group_metadata: BeamSearchMetadata
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return beam_search_sampling_batch_cba(
                logits,
                beam_width_in=self._beam_width_in,
                beam_width_out=self._beam_width_out,
                row_stride=self._row_stride,
                beam_search_args=group_metadata,
                temperature=None,
                early_stopping=self._early_stopping,
                length_penalty=self._length_penalty,
                diversity_rate=self._diversity_rate,
                return_probs=self.computes_probs(),
            )


_STRATEGY_KEY_TYPE: TypeAlias = (
    Literal["temperature"]
    | Literal["top_k"]
    | Literal["top_p"]
    | Literal["top_k_top_p"]
    | Literal["min_p"]
    | Literal["greedy"]
    | tuple[Literal["beam_search"], int, int, int]
)


class FlashInferGroupedStrategySampler:
    """Implements batched sampling with FlashInfer.sampling kernels."""

    STRATEGY_KEY_TYPE: TypeAlias = _STRATEGY_KEY_TYPE

    @staticmethod
    def strategy_grouping_key(strategy: Strategy) -> _STRATEGY_KEY_TYPE:
        match strategy:
            case (
                ("top_k", _, _)
                | ("top_p", _, _)
                | ("top_k_top_p", _, _, _)
                | ("temperature", _)
                | ("min_p", _, _, _, _)
                | ("greedy", None)
            ):
                return cast(_STRATEGY_KEY_TYPE, strategy[0])
            # Trailing wildcard: row_stride is appended after early_stopping.
            case (
                "beam_search",
                beam_width_in,
                beam_width_out,
                _,
                _,
                _,
                early_stopping,
                *_,
            ):
                return cast(
                    _STRATEGY_KEY_TYPE,
                    (strategy[0], beam_width_in, beam_width_out, early_stopping),
                )
            case _:
                raise NotImplementedError("Unsupported strategy encountered")

    @staticmethod
    def get_metadata_type_for_group(
        strategy_key: _STRATEGY_KEY_TYPE,
    ) -> Type[StrategyMetadata] | None:
        match strategy_key:
            case ("beam_search", _, _, _):
                return BeamSearchMetadata
            case "top_p" | "top_k_top_p" | "min_p":
                return TopPDecayMetadata
            case _:
                return None

    @staticmethod
    def sample_grouped_strategies(
        group_key: _STRATEGY_KEY_TYPE,
        strategies: list[Strategy],
        logits: torch.Tensor,
        *,
        group_logit_indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_probs: bool,
        group_metadata: StrategyMetadata | None = None,
        seeds: Optional[RequestSeeds] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample grouped strategies.

        Returns:
          - Sampled tokens
          - Processed probs (whenever return_probs=True)
          - Temperature (used to compute processed _log_ probs)
        """
        beam_width_in = 1
        strategy_impl_cls: Type[_StrategyImpls.StrategyImpl]
        if return_probs:
            match group_key:
                case "top_k":
                    strategy_impl_cls = _StrategyImpls.TopKWithProbs
                case "top_p":
                    strategy_impl_cls = _StrategyImpls.TopPWithProbs
                case "top_k_top_p":
                    strategy_impl_cls = _StrategyImpls.TopKTopPWithProbs
                case "temperature":
                    strategy_impl_cls = _StrategyImpls.TemperatureOnlyWithProbs
                case "min_p":
                    strategy_impl_cls = _StrategyImpls.MinPWithProbs
                case "greedy":
                    strategy_impl_cls = _StrategyImpls.GreedyWithProbs
                case ("beam_search", beam_width_in_key, _, _):
                    beam_width_in = beam_width_in_key
                    # Beam search encodes with-probs as a constructor flag, not
                    # a subclass. Every stopping mode uses the CBA step; the
                    # mode only changes the done verdict inside it.
                    strategy_impl_cls = _StrategyImpls.CBABeamSearchStep
                case _:
                    raise NotImplementedError("Unsupported strategy key encountered")
        else:
            match group_key:
                case "top_p":
                    strategy_impl_cls = _StrategyImpls.TopPSampleOnly
                case "top_k":
                    strategy_impl_cls = _StrategyImpls.TopKSampleOnly
                case "top_k_top_p":
                    strategy_impl_cls = _StrategyImpls.TopKTopPSampleOnly
                case "temperature":
                    strategy_impl_cls = _StrategyImpls.TemperatureOnlySampleOnly
                case "min_p":
                    strategy_impl_cls = _StrategyImpls.MinPSampleOnly
                case "greedy":
                    strategy_impl_cls = _StrategyImpls.GreedySampleOnly
                case ("beam_search", beam_width_in_key, _, _):
                    beam_width_in = beam_width_in_key
                    strategy_impl_cls = _StrategyImpls.CBABeamSearchStep
                case _:
                    raise NotImplementedError("Unsupported strategy key encountered")
        if group_logit_indices is None:
            # Beam-search rows are laid out at the static admission width
            # (row_stride), which exceeds beam_width_in on a widening
            # variable-beam-width step; the op slices down to the live beams.
            rows_per_request = beam_width_in
            if strategies and strategies[0][0] == "beam_search":
                rows_per_request = strategies[0].row_stride
            assert logits.size(0) == rows_per_request * len(strategies)
        else:
            assert group_logit_indices.size(0) == beam_width_in * len(strategies)
        strategy_impl = strategy_impl_cls.from_strategies(strategies, cuda_device=logits.device)
        # Beam search carries with-probs as a flag rather than a subclass, so
        # inject return_probs here (the other strategies encode it in the class).
        if isinstance(strategy_impl, _StrategyImpls.BeamSearchStep):
            strategy_impl.with_computes_probs(return_probs)
        next_tokens, softmax = strategy_impl.sample(
            logits,
            group_logit_indices=group_logit_indices,
            generator=generator,
            group_metadata=group_metadata,
            seeds=seeds,
        )
        return next_tokens, softmax, strategy_impl.get_temperature()


def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
    # We try to cache the resolved strategy on the request object, as it's not cheap enough to
    # resolve it on every iteration.
    cached_sampling_strategy = request.py_sampling_strategy
    if cached_sampling_strategy is not None:
        return cached_sampling_strategy

    params = _request_get_sampling_params(request)
    sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
    if _request_sampling_params_cachable(params):
        request.py_sampling_strategy = sampling_strategy
    return sampling_strategy
