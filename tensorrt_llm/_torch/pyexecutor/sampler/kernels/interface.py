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

"""Sampling strategy-grouping selection and binding.

Holds the small config/contract objects (``SamplerConfig``,
``BoundSamplingBackend``) and the ``resolve_sampling_backend`` factory that
picks the strategy-grouping sampler (FlashInfer vs simple/torch) and binds its
grouping callables once at init time, so no per-call dispatch happens inside
CUDA graph capture.
"""

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.pyexecutor.sampler.sampling_utils import (
    FlashInferGroupedStrategySampler,
    SimpleGroupedStrategySampler,
)


@dataclass(frozen=True, kw_only=True)
class SamplerConfig:
    """Configuration used to resolve and bind a sampling backend at init time."""

    use_flashinfer: bool = False


class BoundSamplingBackend:
    """Holds the strategy-grouping callables bound at initialisation.

    Resolved once (in resolve_sampling_backend) so that no per-call dispatch
    occurs inside CUDA graph capture, and so TorchSampler needs no separate
    _grouped_sampler_cls attribute.
    """

    def __init__(
        self,
        strategy_grouping_key: Any,
        get_metadata_type_for_group: Any,
        sample_grouped_strategies: Any,
    ) -> None:
        self.strategy_grouping_key = strategy_grouping_key
        self.get_metadata_type_for_group = get_metadata_type_for_group
        self.sample_grouped_strategies = sample_grouped_strategies


def resolve_sampling_backend(
    device: torch.device,
    config: SamplerConfig,
) -> BoundSamplingBackend:
    """Bind the strategy-grouping callables at init time; CUDA-graph safe.

    Selection order:
      1. FlashInfer  — CUDA device AND IS_FLASHINFER_AVAILABLE AND config.use_flashinfer
      2. Torch       — everything else (including CPU)
    """
    if device.type == "cuda" and IS_FLASHINFER_AVAILABLE and config.use_flashinfer:
        return BoundSamplingBackend(
            strategy_grouping_key=FlashInferGroupedStrategySampler.strategy_grouping_key,
            get_metadata_type_for_group=FlashInferGroupedStrategySampler.get_metadata_type_for_group,
            sample_grouped_strategies=FlashInferGroupedStrategySampler.sample_grouped_strategies,
        )
    return BoundSamplingBackend(
        strategy_grouping_key=SimpleGroupedStrategySampler.strategy_grouping_key,
        get_metadata_type_for_group=SimpleGroupedStrategySampler.get_metadata_type_for_group,
        sample_grouped_strategies=SimpleGroupedStrategySampler.sample_grouped_strategies,
    )
