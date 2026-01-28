# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Helper functions for using FlashInfer.sampling.

Code in this module should operate on logits and probs, without
referring to types like LlmRequest.
"""

import abc
import sys
from typing import Optional, Type, TypeAlias, cast

import flashinfer.sampling
import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from ..flashinfer_utils import get_env_enable_pdl
from .sampling_utils import (
    GREEDY,
    BeamSearch,
    BeamSearchMetadata,
    GroupedStrategySampler,
    Strategy,
    StrategyMetadata,
    TemperatureOnly,
    TopK,
    TopKTopP,
    TopP,
    beam_search_sampling_batch,
    greedy_search_sampling_batch,
)


class _StrategyImpls:
    class StrategyImpl(abc.ABC):
        @classmethod
        @abc.abstractmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.StrategyImpl":
            pass

        @classmethod
        @abc.abstractmethod
        def computes_probs(cls) -> bool:
            pass

        @abc.abstractmethod
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            pass

        # TODO: Revisit this after determining performance impact
        #
        # NB: NaN logits can lead to crashes, see
        #     https://github.com/flashinfer-ai/flashinfer/issues/1575
        #
        @staticmethod
        def _flashinfer_check_nans(inputs: torch.Tensor) -> bool:
            # Using explicit async NaN check because FlashInfer.sampling 'nan_check' syncs

            # https://github.com/pytorch/pytorch/issues/36853
            torch._assert_async(~torch.any(torch.isnan(inputs)))

            return False

        @staticmethod
        def _make_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            return torch.tensor(data, dtype=dtype, pin_memory=True).to(
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
                logits = torch.index_select(logits, 0, group_logit_indices)  # ensures copy
                logits /= temperature
            else:
                logits = logits / temperature  # not inplace
            return logits

        @staticmethod
        def _prepare_probs_with_temperature(
            logits: torch.Tensor,
            group_logit_indices: Optional[torch.Tensor],
            temperature: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if group_logit_indices is not None:
                logits = logits[group_logit_indices]
            probs = flashinfer.sampling.softmax(
                logits,
                temperature,
                enable_pdl=get_env_enable_pdl(),
            )
            return probs

        @classmethod
        def _sample_from_probs(
            cls,
            probs: torch.Tensor,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            new_tokens = flashinfer.sampling.sampling_from_probs(
                probs,
                deterministic=True,
                generator=generator,
                check_nan=cls._flashinfer_check_nans(probs),
            )
            return new_tokens

        def _sample_greedy_with_probs(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if group_logit_indices is not None:
                logits = torch.index_select(logits, 0, group_logit_indices)  # ensures copy
            new_tokens, probs = greedy_search_sampling_batch(logits, return_probs=True)
            return new_tokens, probs

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
                logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
                probs = cls._prepare_probs_with_temperature(logits, None, None)  # plain softmax
            else:
                probs = cls._prepare_probs_with_temperature(
                    logits, group_logit_indices, temperature
                )

            if top_p is not None:
                probs = flashinfer.sampling.top_p_renorm_probs(probs, top_p)

            new_tokens = cls._sample_from_probs(probs, generator=generator)
            return new_tokens, probs

    class BeamSearchMixin(StrategyImpl):
        def __init__(
            self,
            beam_width_in: torch.Tensor,
            beam_width_out: torch.Tensor,
            temperature: torch.Tensor,
        ):
            self._beam_width_in = beam_width_in
            self._beam_width_out = beam_width_out
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.BeamSearchMixin":
            assert all(strat[0] == "beam_search" for strat in strategies)
            narrowed_strats = cast(list[BeamSearch], strategies)
            beam_width_in = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.int32, cuda_device
            )
            beam_width_out = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.int32, cuda_device
            )
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
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata), (
                "BeamSearchMetadata is required for beam_search_sampling_batch"
            )
            assert torch.unique(self._beam_width_in).numel() == 1, (
                "beam_width_in must be the same for all strategies"
            )
            assert torch.unique(self._beam_width_out).numel() == 1, (
                "beam_width_out must be the same for all strategies"
            )
            logits = self._prepare_logits_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return beam_search_sampling_batch(
                logits,
                beam_width_in=self._beam_width_in[0],
                beam_width_out=self._beam_width_out[0],
                beam_search_args=group_metadata,
                temperature=None,
                generator=generator,
                return_probs=self.computes_probs(),
            )

    class StrategyImplWithProbs(StrategyImpl):
        @override
        @classmethod
        def computes_probs(cls) -> bool:
            return True

    class GreedyWithProbs(StrategyImplWithProbs):
        def __init__(self):
            self._temperature = None

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.GreedyWithProbs":
            assert all(strat == GREEDY for strat in strategies)
            return cls()

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
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
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKTopPWithProbs":
            assert all(strat[0] == "top_k_top_p" for strat in strategies)
            narrowed_strats = cast(list[TopKTopP], strategies)
            top_k = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.int32, cuda_device
            )
            top_p = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.float32, cuda_device
            )
            temperature = cls._make_tensor(
                [strat[3] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(top_k, top_p, temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, probs = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=self._top_p,
                temperature=self._temperature,
                generator=generator,
            )
            return new_tokens, probs

    class TopKWithProbs(StrategyImplWithProbs):
        def __init__(self, top_k: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKWithProbs":
            assert all(strat[0] == "top_k" for strat in strategies)
            narrowed_strats = cast(list[TopK], strategies)
            top_k = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.int32, cuda_device
            )
            temperature = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(top_k, temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, probs = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=self._top_k,
                top_p=None,
                temperature=self._temperature,
                generator=generator,
            )
            return new_tokens, probs

    class TopPWithProbs(StrategyImplWithProbs):
        def __init__(self, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TopPWithProbs":
            assert all(strat[0] == "top_p" for strat in strategies)
            narrowed_strats = cast(list[TopP], strategies)
            top_p = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.float32, cuda_device
            )
            temperature = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(top_p, temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, probs = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=self._top_p,
                temperature=self._temperature,
                generator=generator,
            )
            return new_tokens, probs

    class TemperatureOnlyWithProbs(StrategyImplWithProbs):
        def __init__(self, temperature: torch.Tensor):
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TemperatureOnlyWithProbs":
            assert all(strat[0] == "temperature" for strat in strategies)
            narrowed_strats = cast(list[TemperatureOnly], strategies)
            temperature = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, probs = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                temperature=self._temperature,
                generator=generator,
            )
            return new_tokens, probs

    class BeamSearchWithProbs(BeamSearchMixin, StrategyImplWithProbs):
        pass

    class StrategyImplSampleOnly(StrategyImpl):
        @override
        @classmethod
        def computes_probs(cls) -> bool:
            return False

    class GreedySampleOnly(StrategyImplSampleOnly):
        def __init__(self):
            self._temperature = None

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.GreedySampleOnly":
            assert all(strat == GREEDY for strat in strategies)
            return cls()

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if group_logit_indices is not None:
                logits = logits[group_logit_indices]
            return greedy_search_sampling_batch(logits, return_probs=False)

    class TopKTopPSampleOnly(StrategyImplSampleOnly):
        def __init__(self, top_k: torch.Tensor, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKTopPSampleOnly":
            assert all(strat[0] == "top_k_top_p" for strat in strategies)
            narrowed_strats = cast(list[TopKTopP], strategies)
            top_k = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.int32, cuda_device
            )
            top_p = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.float32, cuda_device
            )
            temperature = cls._make_tensor(
                [strat[3] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(top_k, top_p, temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            logits = self._prepare_logits_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return flashinfer.sampling.top_k_top_p_sampling_from_logits(
                logits,
                top_k=self._top_k,
                top_p=self._top_p,
                # NB: Leveraging 'indices' would require applying temperature+softmax before batching,
                #     because 'flashinfer.sampling.softmax' has no 'indices' argument; but that would
                #     compute unnecessarily softmax also for situations allowing
                #     flashinfer.sampling...._sampling_from_logits.
                # indices=group_logit_indices,
                filter_apply_order="top_k_first",
                deterministic=True,
                check_nan=self._flashinfer_check_nans(logits),
                generator=generator,
            ), None

    class TopKSampleOnly(StrategyImplSampleOnly):
        def __init__(self, top_k: torch.Tensor, temperature: torch.Tensor):
            self._top_k = top_k
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TopKSampleOnly":
            assert all(strat[0] == "top_k" for strat in strategies)
            narrowed_strats = cast(list[TopK], strategies)
            top_k = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.int32, cuda_device
            )
            temperature = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(top_k, temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return flashinfer.sampling.top_k_sampling_from_probs(
                probs,
                top_k=self._top_k,
                # NB: Leveraging 'indices' would require applying temperature+softmax before batching,
                #     because 'flashinfer.sampling.softmax' has no 'indices' argument; but that would
                #     compute unnecessarily softmax also for situations allowing
                #     flashinfer.sampling...._sampling_from_logits.
                # indices=group_logit_indices,
                deterministic=True,
                check_nan=self._flashinfer_check_nans(probs),
                generator=generator,
            ), None

    class TopPSampleOnly(StrategyImplSampleOnly):
        def __init__(self, top_p: torch.Tensor, temperature: torch.Tensor):
            self._top_p = top_p
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TopPSampleOnly":
            assert all(strat[0] == "top_p" for strat in strategies)
            narrowed_strats = cast(list[TopP], strategies)
            top_p = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.float32, cuda_device
            )
            temperature = cls._make_tensor(
                [strat[2] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(top_p, temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            probs = self._prepare_probs_with_temperature(
                logits, group_logit_indices, self._temperature
            )
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs,
                top_p=self._top_p,
                # NB: Leveraging 'indices' would require applying temperature+softmax before batching,
                #     because 'flashinfer.sampling.softmax' has no 'indices' argument; but that would
                #     compute unnecessarily softmax also for situations allowing
                #     flashinfer.sampling...._sampling_from_logits.
                # indices=group_logit_indices,
                deterministic=True,
                check_nan=self._flashinfer_check_nans(probs),
                generator=generator,
            ), None

    class TemperatureOnlySampleOnly(StrategyImplSampleOnly):
        def __init__(self, temperature: torch.Tensor):
            self._temperature = temperature

        @override
        @classmethod
        def from_strategies(
            cls, strategies: list[Strategy], cuda_device: torch.device
        ) -> "_StrategyImpls.TemperatureOnlySampleOnly":
            assert all(strat[0] == "temperature" for strat in strategies)
            narrowed_strats = cast(list[TemperatureOnly], strategies)
            temperature = cls._make_tensor(
                [strat[1] for strat in narrowed_strats], torch.float32, cuda_device
            )
            return cls(temperature)

        @override
        def sample(
            self,
            logits: torch.Tensor,
            *,
            group_logit_indices: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            group_metadata: StrategyMetadata | None = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            new_tokens, _ = self._sample_with_probs(
                logits,
                group_logit_indices=group_logit_indices,
                top_k=None,
                top_p=None,
                temperature=self._temperature,
                generator=generator,
            )
            # FIXME: https://nvbugs/5791242
            # logits = self._prepare_logits_with_temperature(
            #    logits, group_logit_indices, self._temperature
            # )
            # new_tokens = flashinfer.sampling.sampling_from_logits(
            #    logits,
            #    # NB: Leveraging 'indices' would require applying temperature+softmax before batching,
            #    #     because 'flashinfer.sampling.softmax' has no 'indices' argument; but that would
            #    #     compute unnecessarily softmax also for situations allowing
            #    #     flashinfer.sampling...._sampling_from_logits.
            #    # indices=group_logit_indices,
            #    deterministic=True,
            #    generator=generator,
            #    check_nan=self._flashinfer_check_nans(logits),
            # )
            return new_tokens, None

    class BeamSearchSampleOnly(BeamSearchMixin, StrategyImplSampleOnly):
        pass


def _create_beam_search_specialized_cls(
    beam_width_in: torch.Tensor,
    beam_width_out: torch.Tensor,
    return_probs: bool,
) -> Type[_StrategyImpls.BeamSearchMixin]:
    """Create a class that implements BeamSearchMixin with static parameters for grouping."""

    class BeamSearchSpecialized(
        _StrategyImpls.BeamSearchWithProbs if return_probs else _StrategyImpls.BeamSearchSampleOnly
    ):
        static_beam_width_in = beam_width_in
        static_beam_width_out = beam_width_out

        @override
        def __hash__(self) -> int:
            return hash((super(), self.static_beam_width_in, self.static_beam_width_out))

        @override
        def __eq__(self, other: object) -> bool:
            return (
                super().__eq__(other)
                and self.static_beam_width_in == other.static_beam_width_in
                and self.static_beam_width_out == other.static_beam_width_out
            )

    return BeamSearchSpecialized


class FlashInferGroupedStrategySampler(GroupedStrategySampler[Type[_StrategyImpls.StrategyImpl]]):
    """Implements batched sampling with FlashInfer.sampling kernels.

    Note: Currently, FlashInfer.sampling appears to have limited CUDA graph
          support, see https://github.com/flashinfer-ai/flashinfer/issues/978.
    """

    STRATEGY_KEY_TYPE: TypeAlias = Type[_StrategyImpls.StrategyImpl]

    @override
    @staticmethod
    def strategy_grouping_key(strategy: Strategy, return_probs: bool) -> STRATEGY_KEY_TYPE:
        if return_probs:
            match strategy:
                case ("top_k", _, _):
                    return _StrategyImpls.TopKWithProbs
                case ("top_p", _, _):
                    return _StrategyImpls.TopPWithProbs
                case ("top_k_top_p", _, _, _):
                    return _StrategyImpls.TopKTopPWithProbs
                case ("temperature", _):
                    return _StrategyImpls.TemperatureOnlyWithProbs
                case ("greedy", None):
                    return _StrategyImpls.GreedyWithProbs
                case ("beam_search", beam_width_in, beam_width_out, _):
                    return _create_beam_search_specialized_cls(beam_width_in, beam_width_out, True)
        else:
            match strategy:
                case ("top_p", _, _):
                    return _StrategyImpls.TopPSampleOnly
                case ("top_k", _, _):
                    return _StrategyImpls.TopKSampleOnly
                case ("top_k_top_p", _, _, _):
                    return _StrategyImpls.TopKTopPSampleOnly
                case ("temperature", _):
                    return _StrategyImpls.TemperatureOnlySampleOnly
                case ("greedy", None):
                    return _StrategyImpls.GreedySampleOnly
                case ("beam_search", beam_width_in, beam_width_out, _):
                    return _create_beam_search_specialized_cls(beam_width_in, beam_width_out, False)

    @override
    @staticmethod
    def get_metadata_type_for_group(
        strategy_key: STRATEGY_KEY_TYPE,
    ) -> Type[StrategyMetadata] | None:
        if issubclass(strategy_key, _StrategyImpls.BeamSearchMixin):
            return BeamSearchMetadata
        else:
            return None

    @override
    @staticmethod
    def sample_grouped_strategies(
        group_key: STRATEGY_KEY_TYPE,
        strategies: list[Strategy],
        logits: torch.Tensor,
        *,
        group_logit_indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_probs: bool,
        group_metadata: StrategyMetadata | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if hasattr(group_key, "static_beam_width_in"):
            beam_width_in = group_key.static_beam_width_in
        else:
            beam_width_in = 1
        if group_logit_indices is None:
            assert logits.size(0) == beam_width_in * len(strategies)
        else:
            assert group_logit_indices.size(0) == beam_width_in * len(strategies)

        assert return_probs == group_key.computes_probs()

        strategy_impl_cls = group_key
        sampling_object = strategy_impl_cls.from_strategies(strategies, cuda_device=logits.device)
        next_tokens, softmax = sampling_object.sample(
            logits,
            group_logit_indices=group_logit_indices,
            generator=generator,
            group_metadata=group_metadata,
        )
        temperature = (
            sampling_object._temperature.unsqueeze(-1)
            if sampling_object._temperature is not None
            else None
        )
        return next_tokens, softmax, temperature
