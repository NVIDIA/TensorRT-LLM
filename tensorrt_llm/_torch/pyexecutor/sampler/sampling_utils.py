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
    sampling_from_probs_generator_op as sampling_from_probs_generator_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import softmax_op as softmax_op
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    top_k_mask_logits_op as top_k_mask_logits_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    top_k_sampling_from_probs_generator_op as top_k_sampling_from_probs_generator_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    top_k_top_p_sampling_from_logits_with_generator_op as top_k_top_p_sampling_from_logits_with_generator_op,  # noqa: E501
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    top_p_renorm_probs_op as top_p_renorm_probs_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    top_p_sampling_from_probs_generator_op as top_p_sampling_from_probs_generator_op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    BeamSearchMetadata as BeamSearchMetadata,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import StrategyMetadata as StrategyMetadata
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import _Fusions as _Fusions
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    beam_search_sampling_batch as beam_search_sampling_batch,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    get_rejected_indices as get_rejected_indices,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import greedy as _torch_greedy
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    greedy_search_sampling_batch as greedy_search_sampling_batch,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import sample_rejected as sample_rejected
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    temperature_sampling_batch as temperature_sampling_batch,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    top_k_sampling_batch as top_k_sampling_batch,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    top_k_top_p_sampling_batch as top_k_top_p_sampling_batch,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    top_p_sampling_batch as top_p_sampling_batch,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.sampling_params import SamplingParams

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


TemperatureOnly: TypeAlias = tuple[Literal["temperature"], float]
TopK: TypeAlias = tuple[Literal["top_k"], int, float]
TopP: TypeAlias = tuple[Literal["top_p"], float, float]
TopKTopP: TypeAlias = tuple[Literal["top_k_top_p"], int, float, float]
Greedy: TypeAlias = tuple[Literal["greedy"], None]
BeamSearch: TypeAlias = tuple[Literal["beam_search"], int, int, float]
GREEDY: Greedy = ("greedy", None)

Strategy: TypeAlias = TopK | TopP | Greedy | TopKTopP | TemperatureOnly | BeamSearch

BEAM_SEARCH_PAD_TOKEN = -1


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
    """

    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    use_beam_search: Optional[bool]
    beam_width_in: Optional[int] = None
    beam_width_out: Optional[int] = None


def resolve_sampling_strategy(params: UtilsSamplingParams, *, vocab_size: int) -> Strategy:
    # The semantics are specified in the doc-string of SamplingParams

    use_beam_search = params.use_beam_search
    temperature = params.temperature
    top_p = params.top_p
    top_k = params.top_k

    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_beam_search=use_beam_search,
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
        return ("beam_search", params.beam_width_in, params.beam_width_out, temperature)

    # NB: not greedy, hence top_p != 0 if specified
    top_p = top_p or 1.0
    # NB: not greedy, hence top_k != 1 if specified
    #     (0 and vocab_size are equivalent)
    top_k = top_k or vocab_size

    assert top_k > 1, "non-greedy sampling requires valid top_k"
    need_top_k = top_k < vocab_size
    assert top_p > 0, "non-greedy sampling requires valid top_p"
    need_top_p = top_p < 1

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
            tokens, softmax = top_k_sampling_batch(
                logits,
                top_k=cast(int, top_k),
                temperature=cast(float, temperature),
                generator=generator,
            )
        case ("top_p", top_p, temperature):
            tokens, softmax = top_p_sampling_batch(
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
            tokens, softmax = temperature_sampling_batch(
                logits,
                temperature=cast(float, temperature),
                generator=generator,
            )
        case ("greedy", None):
            tokens, softmax = greedy_search_sampling_batch(logits, return_probs=return_probs)
            temperature = None
        case ("beam_search", beam_width_in, beam_width_out, temperature):
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata), (
                "BeamSearchMetadata is required for beam_search_sampling_batch"
            )
            tokens, softmax = beam_search_sampling_batch(
                logits,
                beam_width_in=cast(int, beam_width_in),
                beam_width_out=cast(int, beam_width_out),
                beam_search_args=group_metadata,
                temperature=cast(float, temperature),
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

        @staticmethod
        def _flashinfer_check_nans(inputs: torch.Tensor) -> bool:
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
            return sampling_from_probs_generator_op(
                probs, generator, check_nan=cls._flashinfer_check_nans(probs)
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

    class TopKTopPWithProbs(StrategyImplWithProbs):
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

    class TopPWithProbs(StrategyImplWithProbs):
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

    class TopKTopPSampleOnly(StrategyImplSampleOnly):
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
            logits = self._prepare_logits_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_k_top_p_sampling_from_logits_with_generator_op(
                logits,
                self._top_k,
                self._top_p,
                generator,
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
            return top_k_sampling_from_probs_generator_op(
                probs, self._top_k, generator, check_nan=self._flashinfer_check_nans(probs)
            ), None

    class TopPSampleOnly(StrategyImplSampleOnly):
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
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return top_p_sampling_from_probs_generator_op(
                probs, self._top_p, generator, check_nan=self._flashinfer_check_nans(probs)
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
        def __init__(self, beam_width_in: int, beam_width_out: int, temperature: torch.Tensor):
            self._beam_width_in = beam_width_in
            self._beam_width_out = beam_width_out
            self._temperature = temperature

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
            return cls(beam_width_in, beam_width_out, temperature)

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
            return beam_search_sampling_batch(
                logits,
                beam_width_in=self._beam_width_in,
                beam_width_out=self._beam_width_out,
                beam_search_args=group_metadata,
                temperature=None,
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
    | tuple[Literal["beam_search"], int, int]
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
            case ("beam_search", beam_width_in, beam_width_out, _):
                return cast(_STRATEGY_KEY_TYPE, (strategy[0], beam_width_in, beam_width_out))
            case _:
                raise NotImplementedError("Unsupported strategy encountered")

    @staticmethod
    def get_metadata_type_for_group(
        strategy_key: _STRATEGY_KEY_TYPE,
    ) -> Type[StrategyMetadata] | None:
        match strategy_key:
            case ("beam_search", _, _):
                return BeamSearchMetadata
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
                case ("beam_search", beam_width_in_key, _):
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
                case ("beam_search", beam_width_in_key, _):
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


# Re-export the torch greedy op (used by drafting_loops and speculative/interface).
greedy = _torch_greedy

# ---------------------------------------------------------------------------
# Spec-decoding interface: compute_probs_from_logits (per-request tensor params)
# ---------------------------------------------------------------------------


def sanitize_top_k(top_k: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map ``top_k`` into a backend-safe range before top-k filtering.

    Per ``SamplingParams``, ``top_k == 0`` means "all logits" (top-k disabled),
    but the flashinfer (``top_k_mask_logits``) and PyTorch-native top-k paths
    break on a literal 0 — they mask the entire row (all-zero probs) or gather
    out of bounds. Mirror the C++ op (``dynamicTreeKernels.cu``): map any
    non-positive value (and any oversized disable sentinel such as
    ``INT32_MAX``) to ``vocab_size`` (== keep all tokens), leaving genuine
    top_k values untouched.
    """
    return torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab_size)).clamp(max=vocab_size)


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
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """CUDA-graph compatible sampling; supports mixed sampling params. Returns sampled tokens."""
    top_k = sanitize_top_k(top_k, logits.shape[-1])
    # Greedy rows (temperature <= threshold) must return the argmax token, not a
    # sample from the temperature-scaled distribution. Capture the argmax from the
    # *original* logits up front; _safely_apply_temperature_inplace then guards the division
    # against the greedy sentinel, and torch.where restores the greedy rows below.
    # All ops are branch-free (no data-dependent control flow), so this stays
    # CUDA-graph safe.
    is_greedy = temperatures <= vanilla._GREEDY_TEMPERATURE_THRESHOLD
    greedy_tokens = logits.argmax(dim=-1)
    logits = vanilla._safely_apply_temperature_inplace(logits, temperatures)
    sampled = flashinfer.top_k_top_p_sampling_from_logits_op(
        logits, top_k, top_p, seed=seed, offset=offset
    )
    # argmax yields int64; cast so torch.where preserves the sampler's dtype
    # (flashinfer returns int32) instead of promoting the result to int64.
    return torch.where(is_greedy, greedy_tokens.to(sampled.dtype), sampled)


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
