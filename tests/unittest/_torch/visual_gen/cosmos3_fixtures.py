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
resolver without a real checkpoint: either a :class:`Cosmos3CheckpointSettings` built in memory,
or the checkpoint's JSON files written into a temporary directory."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    DEFAULT,
    VAE_FILE,
    Cosmos3ActionSettings,
    Cosmos3CheckpointSettings,
    Cosmos3PromptSettings,
    Cosmos3SamplingSettings,
    Cosmos3VAESettings,
    ParameterResolver,
    Resolved,
)

# The VAE config of a test checkpoint; the resolver requires both factors of a shipped VAE.
VAE_CONFIG = {"scale_factor_temporal": 4, "scale_factor_spatial": 16}


def write_checkpoint_files(
    checkpoint_dir: Path, files: Mapping[str, Any], *, vae: bool = False
) -> Path:
    """Write each ``{path under the checkpoint: JSON document}`` of ``files`` into
    ``checkpoint_dir``, plus :data:`VAE_CONFIG` as the VAE config when ``vae`` is set.
    Returns ``checkpoint_dir``."""
    if vae:
        files = {VAE_FILE: VAE_CONFIG, **files}
    for name, doc in files.items():
        path = checkpoint_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return checkpoint_dir


def bare_config(
    *,
    fixed_sigmas: Optional[tuple[float, ...]] = None,
    policy: Optional[Mapping[str, Any]] = None,
    use_system_prompt: bool = False,
    native_flow_schedule: bool = False,
    flow_shift: Optional[float] = None,
) -> Cosmos3CheckpointSettings:
    """A base-checkpoint :class:`Cosmos3CheckpointSettings` with the given facts overridden."""
    settings = {
        "vae.temporal_compression": 4,
        "vae.spatial_compression": 16,
        "sampling.fixed_sigmas": fixed_sigmas,
        "sampling.native_flow_schedule": native_flow_schedule,
        "sampling.flow_shift": flow_shift,
        "prompt.use_system_prompt": use_system_prompt,
        "action.policy": policy,
    }
    return Cosmos3CheckpointSettings(
        vae=Cosmos3VAESettings(temporal_compression=4, spatial_compression=16),
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
