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

"""Sampling strategy-grouping backend selection.

Holds ``SamplerConfig`` and the ``resolve_sampling_backend`` factory that picks
the strategy-grouping sampler (FlashInfer vs simple/torch) once at init time, so
no per-call dispatch happens inside CUDA graph capture. The three grouping
callables are tightly coupled (they share the same key type), so the whole
sampler class is bound as a unit rather than field by field.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Type, cast

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.pyexecutor.sampler.sampling_utils import (
    FlashInferGroupedStrategySampler,
    GroupedStrategySampler,
    SimpleGroupedStrategySampler,
)


@dataclass(frozen=True, kw_only=True)
class SamplerConfig:
    """Configuration used to resolve the sampling backend at init time."""

    use_flashinfer: bool = False


def resolve_sampling_backend(
    is_cuda: bool,
    config: SamplerConfig,
) -> Type[GroupedStrategySampler[Hashable]]:
    """Pick the strategy-grouping sampler class at init time; CUDA-graph safe.

    Selection order:
      1. FlashInfer  — is_cuda AND IS_FLASHINFER_AVAILABLE AND config.use_flashinfer
      2. Torch       — everything else (including CPU)
    """
    # The two samplers use different key types (Strategy vs a narrower FlashInfer
    # key); the caller only forwards the grouping callables, so erase to Hashable.
    if is_cuda and IS_FLASHINFER_AVAILABLE and config.use_flashinfer:
        return cast(Type[GroupedStrategySampler[Hashable]], FlashInferGroupedStrategySampler)
    return cast(Type[GroupedStrategySampler[Hashable]], SimpleGroupedStrategySampler)
