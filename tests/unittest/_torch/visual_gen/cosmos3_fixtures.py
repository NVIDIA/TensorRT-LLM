# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Cosmos3 checkpoint settings built by hand, for tests that run a pipeline or a
resolver without a checkpoint on disk."""

from collections.abc import Mapping
from typing import Any, Optional

from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    DEFAULT,
    Cosmos3ActionSettings,
    Cosmos3ArchitectureSettings,
    Cosmos3AudioSettings,
    Cosmos3Config,
    Cosmos3PromptSettings,
    Cosmos3SamplingSettings,
    Cosmos3VAESettings,
    ParameterResolver,
    Resolved,
)


def bare_config(
    *,
    fixed_sigmas: Optional[tuple[float, ...]] = None,
    policy: Optional[Mapping[str, Any]] = None,
    use_system_prompt: bool = False,
    native_flow_schedule: bool = False,
    flow_shift: Optional[float] = None,
) -> Cosmos3Config:
    """A base-checkpoint :class:`Cosmos3Config` with the given facts overridden."""
    settings = {
        "architecture.position_embedding_type": "unified_3d_mrope",
        "architecture.max_position_embeddings": 262144,
        "vae.temporal_compression": 4,
        "vae.spatial_compression": 16,
        "audio.temporal_compression": 1,
        "sampling.fixed_sigmas": fixed_sigmas,
        "sampling.native_flow_schedule": native_flow_schedule,
        "sampling.flow_shift": flow_shift,
        "prompt.use_system_prompt": use_system_prompt,
        "action.policy": policy,
    }
    return Cosmos3Config(
        architecture=Cosmos3ArchitectureSettings("unified_3d_mrope", 262144),
        vae=Cosmos3VAESettings(temporal_compression=4, spatial_compression=16),
        audio=Cosmos3AudioSettings(temporal_compression=1),
        sampling=Cosmos3SamplingSettings(
            fixed_sigmas=fixed_sigmas,
            native_flow_schedule=native_flow_schedule,
            flow_shift=flow_shift,
        ),
        prompt=Cosmos3PromptSettings(use_system_prompt=use_system_prompt),
        action=Cosmos3ActionSettings(policy=policy),
        resolved={k: Resolved(v, DEFAULT) for k, v in settings.items() if v is not None},
    )


def bare_resolver(family: str, **facts: Any) -> ParameterResolver:
    """A silent :class:`ParameterResolver` over :func:`bare_config`'s settings."""
    return ParameterResolver(family, bare_config(**facts), log=False)
