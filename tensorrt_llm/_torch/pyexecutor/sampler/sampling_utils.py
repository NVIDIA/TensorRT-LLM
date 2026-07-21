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

"""Helper functions for sampling.

Code in this module should operate on logits and probs, without
referring to types like LlmRequest.
"""

import abc
import sys
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type, TypeAlias, TypeVar, cast

import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer, vanilla

# These op wrappers are safe to import without flashinfer installed; they are
# only called on the flashinfer sampler / speculative-worker paths.
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    sampling_from_probs_op,
    softmax_op,
    top_k_mask_logits_op,
    top_k_sampling_from_probs_op,
    top_k_top_p_sampling_from_logits_op,
    top_p_renorm_probs_op,
    top_p_sampling_from_probs_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    GREEDY_TEMPERATURE_THRESHOLD,
    BeamSearchMetadata,
    Fusions,
    StrategyMetadata,
    beam_search_sampling_batch,
    beam_search_sampling_batch_cba,
    get_rejected_indices,
    greedy_search_sampling_batch,
    sample_rejected,
    top_k_top_p_sampling_batch,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.sampling_params import SamplingParams

# Ops imported above are re-exported for dependent modules (sampler, drafting
# loops, tests). mypy runs in strict mode (no implicit re-export), so they must
# be listed here.
__all__ = [
    "GREEDY_TEMPERATURE_THRESHOLD",
    "BeamSearchMetadata",
    "Fusions",
    "StrategyMetadata",
    "beam_search_sampling_batch",
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
Greedy: TypeAlias = tuple[Literal["greedy"], None]
BeamSearch: TypeAlias = tuple[Literal["beam_search"], int, int, float, float, float, int]
GREEDY: Greedy = ("greedy", None)

Strategy: TypeAlias = TopK | TopP | Greedy | TopKTopP | TemperatureOnly | BeamSearch

# Re-exported from the beam-search op implementation (single source of truth).
BEAM_SEARCH_PAD_TOKEN = vanilla.BEAM_SEARCH_PAD_TOKEN


@dataclass(frozen=True, kw_only=True)
class UtilsSamplingParams:
    """Subset of tensorrt_llm::runtime::SamplingConfig supported by sampling_utils.

    Args:
        temperature: The temperature to use for sampling.
        top_p: The top-p to use for sampling.
        top_k: The top-k to use for sampling.
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
        early_stopping: Beam-search stopping mode: 1 stops as soon as
            beam_width finished candidates exist (default); 0 and other values
            are the exhaustive modes backed by the candidate-beams array.
    """

    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    use_beam_search: Optional[bool]
    beam_width_in: Optional[int] = None
    beam_width_out: Optional[int] = None
    top_p_decay: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    length_penalty: Optional[float] = None
    beam_search_diversity_rate: Optional[float] = None
    early_stopping: Optional[int] = None


@dataclass(kw_only=True)
class TopPDecayMetadata(StrategyMetadata):
    """Per-group runtime top-p override for Top-P Decay (attached to top_p /
    top_k_top_p groups via the ``StrategyMetadata`` mechanism).

    ``slots`` maps each per-step group row to its sequence slot; the decayed
    per-row top-p is gathered on-device from the per-slot ``runtime_top_p``
    store, gated by ``is_decay_slot`` (non-decay rows keep their static top-p).
    Consumed by the TopP*/TopKTopP* strategy impls in ``sample()``. See
    ``TorchSampler.TopPDecayStore`` for the feature-level semantics.
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

    # The greedy verdict (including the top-p-decay override of the implicit
    # all-unset greedy default, and explicit greedy controls winning over decay)
    # is single-sourced in SamplingParams.params_imply_greedy_decoding.
    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_beam_search=use_beam_search,
        top_p_decay=params.top_p_decay,
    ):
        return GREEDY

    # --- resolving default values
    # NB: not greedy, hence temperature != 0 if specified
    temperature = temperature or 1.0

    # Beam search does not rely on top_p or top_k, so we can return the strategy here
    if use_beam_search:
        assert params.beam_width_in is not None and params.beam_width_out is not None, (
            "beam_width_in and beam_width_out must be specified for beam search"
        )
        return (
            "beam_search",
            params.beam_width_in,
            params.beam_width_out,
            temperature,
            params.length_penalty or 0.0,
            params.beam_search_diversity_rate or 0.0,
            1 if params.early_stopping is None else params.early_stopping,
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
        case ("greedy", None):
            tokens, softmax = greedy_search_sampling_batch(logits, return_probs=return_probs)
            temperature = None
        case (
            "beam_search",
            beam_width_in,
            beam_width_out,
            temperature,
            length_penalty,
            beam_search_diversity_rate,
            early_stopping,
        ):
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata), (
                "BeamSearchMetadata is required for beam_search_sampling_batch"
            )
            if cast(int, early_stopping) != 1:
                tokens, softmax = beam_search_sampling_batch_cba(
                    logits,
                    beam_width_in=cast(int, beam_width_in),
                    beam_width_out=cast(int, beam_width_out),
                    beam_search_args=group_metadata,
                    temperature=cast(float, temperature),
                    early_stopping=cast(int, early_stopping),
                    length_penalty=cast(float, length_penalty),
                    diversity_rate=cast(float, beam_search_diversity_rate),
                    return_probs=return_probs,
                )
            else:
                tokens, softmax = beam_search_sampling_batch(
                    logits,
                    beam_width_in=cast(int, beam_width_in),
                    beam_width_out=cast(int, beam_width_out),
                    beam_search_args=group_metadata,
                    temperature=cast(float, temperature),
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
        ) -> torch.Tensor:
            return sampling_from_probs_op(
                probs, generator=generator, check_nan=cls._flashinfer_check_nans(probs)
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
        def _sample_with_probs(
            cls,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor],
            top_k: Optional[torch.Tensor],
            top_p: Optional[torch.Tensor],
            temperature: torch.Tensor,
            generator: Optional[torch.Generator],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if top_k is not None:
                logits = cls._prepare_logits_with_temperature(
                    logits, group_logit_indices, temperature
                )
                logits = top_k_mask_logits_op(logits, top_k)
                probs = cls._prepare_probs_with_temperature(logits, None, None)
            else:
                probs = cls._prepare_probs_with_temperature(
                    logits, group_logit_indices, temperature
                )
            if top_p is not None:
                probs = top_p_renorm_probs_op(probs, top_p)
            new_tokens = cls._sample_from_probs(probs, generator=generator)
            return new_tokens, probs

    class TopPDecayMixin:
        """Mixed into the TopP*/TopKTopP* impls (the owners of a per-row
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=self._top_p,
                temperature=self._temperature,
                generator=generator,
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=None,
                temperature=self._temperature,
                generator=generator,
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=self._top_p,
                temperature=self._temperature,
                generator=generator,
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                temperature=self._temperature,
                generator=generator,
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_k_sampling_from_probs_op(
                probs,
                self._top_k,
                generator=generator,
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            self._maybe_apply_top_p_decay(group_metadata)
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_p_sampling_from_probs_op(
                probs,
                self._top_p,
                generator=generator,
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
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, _ = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                temperature=self._temperature,
                generator=generator,
            )
            return new_tokens, None

    class BeamSearchMixin(StrategyImpl):
        def __init__(
            self,
            beam_width_in: int,
            beam_width_out: int,
            temperature: torch.Tensor,
            length_penalty: Optional[torch.Tensor],
            diversity_rate: Optional[torch.Tensor],
            early_stopping: int,
        ):
            self._beam_width_in = beam_width_in
            self._beam_width_out = beam_width_out
            self._temperature = temperature
            self._length_penalty = length_penalty
            self._diversity_rate = diversity_rate
            self._early_stopping = early_stopping

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Any], cuda_device: torch.device
        ) -> "_StrategyImpls.BeamSearchMixin":
            assert all(strat[0] == "beam_search" for strat in strategies)
            narrowed_strats = cast(list[BeamSearch], strategies)
            (beam_width_in,) = set(strat[1] for strat in narrowed_strats)
            (beam_width_out,) = set(strat[2] for strat in narrowed_strats)
            temperature = cls._make_tensor(
                [strat[3] or 1.0 for strat in narrowed_strats], torch.float32, cuda_device
            )
            length_penalties = [strat[4] or 0.0 for strat in narrowed_strats]
            length_penalty: Optional[torch.Tensor] = None
            if any(lp != 0.0 for lp in length_penalties):
                length_penalty = cls._make_tensor(length_penalties, torch.float32, cuda_device)
            diversity_rates = [strat[5] or 0.0 for strat in narrowed_strats]
            diversity_rate: Optional[torch.Tensor] = None
            if any(dr != 0.0 for dr in diversity_rates):
                diversity_rate = cls._make_tensor(diversity_rates, torch.float32, cuda_device)
            # early_stopping is part of the grouping key, hence unique per group.
            (early_stopping,) = set(strat[6] for strat in narrowed_strats)
            return cls(
                beam_width_in,
                beam_width_out,
                temperature,
                length_penalty,
                diversity_rate,
                early_stopping,
            )

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: Optional[StrategyMetadata] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata)
            temperature = self._temperature.repeat_interleave(self._beam_width_in)
            logits = self._prepare_logits_with_temperature(logits, group_logit_indices, temperature)
            if self._early_stopping != 1:
                return beam_search_sampling_batch_cba(
                    logits,
                    beam_width_in=self._beam_width_in,
                    beam_width_out=self._beam_width_out,
                    beam_search_args=group_metadata,
                    temperature=None,
                    early_stopping=self._early_stopping,
                    length_penalty=self._length_penalty,
                    diversity_rate=self._diversity_rate,
                    topk_fn=flashinfer.topk_op,
                    return_probs=self.computes_probs(),
                )
            return beam_search_sampling_batch(
                logits,
                beam_width_in=self._beam_width_in,
                beam_width_out=self._beam_width_out,
                beam_search_args=group_metadata,
                temperature=None,
                length_penalty=self._length_penalty,
                diversity_rate=self._diversity_rate,
                # TorchSampler hard-depends on flashinfer (enforced in its
                # constructor); topk_op picks flashinfer's radix top-k for
                # vocab-sized rows (~5x faster than torch.topk).
                topk_fn=flashinfer.topk_op,
                return_probs=self.computes_probs(),
            )

    class BeamSearchWithProbs(BeamSearchMixin, StrategyImplWithProbs):
        pass

    class BeamSearchSampleOnly(BeamSearchMixin, StrategyImplSampleOnly):
        pass


_STRATEGY_KEY_TYPE: TypeAlias = (
    Literal["temperature"]
    | Literal["top_k"]
    | Literal["top_p"]
    | Literal["top_k_top_p"]
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
                | ("greedy", None)
            ):
                return cast(_STRATEGY_KEY_TYPE, strategy[0])
            case ("beam_search", beam_width_in, beam_width_out, _, _, _, early_stopping):
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
            case "top_p" | "top_k_top_p":
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
                case "greedy":
                    strategy_impl_cls = _StrategyImpls.GreedyWithProbs
                case ("beam_search", beam_width_in_key, _, _):
                    beam_width_in = beam_width_in_key
                    strategy_impl_cls = _StrategyImpls.BeamSearchWithProbs
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
                case "greedy":
                    strategy_impl_cls = _StrategyImpls.GreedySampleOnly
                case ("beam_search", beam_width_in_key, _, _):
                    beam_width_in = beam_width_in_key
                    strategy_impl_cls = _StrategyImpls.BeamSearchSampleOnly
                case _:
                    raise NotImplementedError("Unsupported strategy key encountered")
        if group_logit_indices is None:
            assert logits.size(0) == beam_width_in * len(strategies)
        else:
            assert group_logit_indices.size(0) == beam_width_in * len(strategies)
        strategy_impl = strategy_impl_cls.from_strategies(strategies, cuda_device=logits.device)
        next_tokens, softmax = strategy_impl.sample(
            logits,
            group_logit_indices=group_logit_indices,
            generator=generator,
            group_metadata=group_metadata,
        )
        return next_tokens, softmax, strategy_impl.get_temperature()


# ---------------------------------------------------------------------------
# Spec-decoding interface: compute_probs_from_logits (per-request tensor params)
# ---------------------------------------------------------------------------


def sanitize_top_k(top_k: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map ``top_k`` into a backend-safe range before top-k filtering.

    Per ``SamplingParams``, ``top_k == 0`` means "all logits" (top-k disabled),
    but the flashinfer top-k kernels (``top_k_mask_logits``) break on a literal
    0 — they mask the entire row (all-zero probs). Map any non-positive value
    (and any oversized disable sentinel such as ``INT32_MAX``) to
    ``vocab_size`` (== keep all tokens), leaving genuine top_k values
    untouched.
    """
    return top_k.clamp(max=vocab_size).masked_fill_(top_k <= 0, vocab_size)


@torch.compile(options={"max-autotune": True})
def compute_probs_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute filtered+normalized probs via flashinfer (hard dependency).

    ``temperatures``, ``top_k``, ``top_p`` are per-request tensors matching the
    spec-decoding call site in interface.py.
    """
    if top_k is not None:
        top_k = sanitize_top_k(top_k, logits.shape[-1])

    return flashinfer.compute_probs_from_logits_op(logits, temperatures, top_k, top_p)


@torch.compile(options={"max-autotune": True})
def sampling_batch_spec_dec_one_model(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA-graph compatible sampling; supports mixed sampling params. Returns sampled tokens."""
    top_k = sanitize_top_k(top_k, logits.shape[-1])
    # Greedy rows (temperature <= threshold) reduce to top_k=1 sampling: with the
    # divisor clamped to 1.0 by safely_apply_temperature_inplace (order-preserving
    # for those rows), flashinfer deterministically returns the max-probability
    # token, i.e. the argmax of the original logits. All ops remain branch-free
    # (no data-dependent control flow), so this stays CUDA-graph safe.
    is_greedy = temperatures <= vanilla.GREEDY_TEMPERATURE_THRESHOLD
    top_k = torch.where(is_greedy, torch.ones_like(top_k), top_k)
    top_p = torch.where(is_greedy, torch.ones_like(top_p), top_p)
    logits = vanilla.safely_apply_temperature_inplace(logits, temperatures)
    return flashinfer.top_k_top_p_sampling_from_logits_op(
        logits, top_k, top_p, seed=seed, offset=offset
    )


@torch.compile(options={"max-autotune": True})
def sampling_batch_spec_dec_one_model_for_rejection(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft sampler returning tokens AND probs for the downstream rejection-sampling path."""
    # Rejection sampling relies on flashinfer's seed/offset support for
    # determinism and cross-rank consistency.
    probs = compute_probs_from_logits(logits, temperatures, top_k, top_p)
    tokens = flashinfer.sampling_from_probs_op(probs, seed=seed, offset=offset)
    return tokens, probs
