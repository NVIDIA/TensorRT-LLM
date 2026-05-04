# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Transforms to handle SSM cache insertion."""

from typing import Type

from pydantic import Field

from ..interface import TransformConfig, TransformRegistry
from .kvcache import InsertCachedAttentionConfig, _InsertCachedOperator


class SSMCacheTransformConfig(InsertCachedAttentionConfig):
    """Configuration for insert_cached_ssm_attention.

    Extends the base attention config with SSM-specific options.
    """

    ssm_replay: bool = Field(
        default=False,
        description=(
            "Enable the replay SSM kernel (tl.dot fast-forward) for the MTP extend path. "
            "Requires SM >= 80 (Ampere+). Falls back to FlashInfer when disabled or "
            "when incompatible features are active (block reuse, tree attention)."
        ),
    )


# TODO: think about separating valid attention backends per transform better in the future
@TransformRegistry.register("insert_cached_ssm_attention")
class SSMCacheTransform(_InsertCachedOperator):
    """A transform to handle SSM cache operations."""

    config: SSMCacheTransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return SSMCacheTransformConfig

    @property
    def attn_descriptor(self):
        descriptor = super().attn_descriptor
        if self.config.ssm_replay and hasattr(descriptor, "ssm_replay"):
            # Return a thin subclass with ssm_replay enabled so get_cache_initializers
            # allocates replay buffers instead of intermediate_ssm_state_cache.
            return type(descriptor.__name__ + "_Replay", (descriptor,), {"ssm_replay": True})
        return descriptor


@TransformRegistry.register("insert_cached_causal_conv")
class InitializeCausalConvCache(_InsertCachedOperator):
    """A transform to handle causal conv cache operations."""


@TransformRegistry.register("insert_cached_delta_rule")
class InsertCachedDeltaRule(_InsertCachedOperator):
    """A transform to handle delta rule cache operations."""


@TransformRegistry.register("insert_cached_gated_delta_rule")
class InsertCachedGatedDeltaRule(_InsertCachedOperator):
    """A transform to handle gated delta rule cache operations."""
