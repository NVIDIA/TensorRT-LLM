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

"""Sampling backend selection and binding.

Holds the small config/contract objects (``SamplerConfig``,
``BoundSamplingBackend``) and the ``resolve_sampling_backend`` factory that
picks one of the leaf backend implementations (vanilla / flashinfer / trtllm)
and binds its kernel functions once at init time, so no per-call dispatch
happens inside CUDA graph capture.
"""

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.pyexecutor.sampling_backend.vanilla import (
    compute_probs as _torch_compute_probs,
)
from tensorrt_llm._torch.pyexecutor.sampling_backend.vanilla import greedy as _torch_greedy
from tensorrt_llm._torch.pyexecutor.sampling_backend.vanilla import (
    sample_from_logits as _torch_sample_from_logits,
)
from tensorrt_llm._torch.pyexecutor.sampling_backend.vanilla import (
    sample_from_probs as _torch_sample_from_probs,
)
from tensorrt_llm._torch.pyexecutor.sampling_utils import (
    FlashInferGroupedStrategySampler,
    SimpleGroupedStrategySampler,
)


@dataclass(frozen=True, kw_only=True)
class SamplerConfig:
    """Configuration used to resolve and bind a sampling backend at init time."""

    use_flashinfer: bool = False


class BoundSamplingBackend:
    """Holds kernel function pointers bound at initialisation; CUDA-graph safe.

    All callables are resolved once (in resolve_sampling_backend) so that no
    per-call dispatch occurs inside CUDA graph capture.  Includes the
    strategy-grouping helpers so TorchSampler needs no separate
    _grouped_sampler_cls attribute.
    """

    def __init__(
        self,
        compute_probs: Any,
        sample_from_probs: Any,
        sample_from_logits: Any,
        greedy: Any,
        strategy_grouping_key: Any,
        get_metadata_type_for_group: Any,
        sample_grouped_strategies: Any,
    ) -> None:
        self.compute_probs = compute_probs
        self.sample_from_probs = sample_from_probs
        self.sample_from_logits = sample_from_logits
        self.greedy = greedy
        self.strategy_grouping_key = strategy_grouping_key
        self.get_metadata_type_for_group = get_metadata_type_for_group
        self.sample_grouped_strategies = sample_grouped_strategies


def resolve_sampling_backend(
    device: torch.device,
    config: SamplerConfig,
) -> BoundSamplingBackend:
    """Bind kernel functions at init time; returns a CUDA-graph-safe backend.

    Selection order:
      1. FlashInfer  — CUDA device AND IS_FLASHINFER_AVAILABLE AND config.use_flashinfer
      2. Torch       — everything else (including CPU)
    """
    if device.type == "cuda" and IS_FLASHINFER_AVAILABLE and config.use_flashinfer:
        from tensorrt_llm._torch.pyexecutor.sampling_backend.flashinfer import (
            compute_probs as _fi_compute_probs,
        )
        from tensorrt_llm._torch.pyexecutor.sampling_backend.flashinfer import greedy as _fi_greedy
        from tensorrt_llm._torch.pyexecutor.sampling_backend.flashinfer import (
            sample_from_logits as _fi_sample_from_logits,
        )
        from tensorrt_llm._torch.pyexecutor.sampling_backend.flashinfer import (
            sample_from_probs as _fi_sample_from_probs,
        )

        return BoundSamplingBackend(
            compute_probs=_fi_compute_probs,
            sample_from_probs=_fi_sample_from_probs,
            sample_from_logits=_fi_sample_from_logits,
            greedy=_fi_greedy,
            strategy_grouping_key=FlashInferGroupedStrategySampler.strategy_grouping_key,
            get_metadata_type_for_group=FlashInferGroupedStrategySampler.get_metadata_type_for_group,
            sample_grouped_strategies=FlashInferGroupedStrategySampler.sample_grouped_strategies,
        )
    return BoundSamplingBackend(
        compute_probs=_torch_compute_probs,
        sample_from_probs=_torch_sample_from_probs,
        sample_from_logits=_torch_sample_from_logits,
        greedy=_torch_greedy,
        strategy_grouping_key=SimpleGroupedStrategySampler.strategy_grouping_key,
        get_metadata_type_for_group=SimpleGroupedStrategySampler.get_metadata_type_for_group,
        sample_grouped_strategies=SimpleGroupedStrategySampler.sample_grouped_strategies,
    )
