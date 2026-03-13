# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Transform to allocate a per-slot mRoPE position-delta cache.

Multimodal models using mRoPE (multi-dimensional rotary position embeddings)
compute a position delta during prefill that must be remembered for subsequent
decode steps.  This transform allocates a tiny ``(max_slots, 1)`` int32 buffer
and registers it as a state resource so that the runtime can write during prefill
and read during decode.

Detection: scans the model source for usage of the
``auto_deploy::mrope_delta_with_cache`` custom op via ``torch.ops``.
"""

import inspect
from typing import Tuple, Type

import torch
import torch.nn as nn

from ...custom_ops.attention_interface import StateResourceHandler
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_MROPE_DELTA_CACHE_OP = "mrope_delta_with_cache"


def _model_uses_mrope_delta_cache(mod: nn.Module) -> bool:
    """Return True if any submodule's forward references the mrope_delta_with_cache op."""
    for m in mod.modules():
        try:
            src = inspect.getsource(type(m).forward)
        except (TypeError, OSError):
            continue
        if _MROPE_DELTA_CACHE_OP in src:
            return True
    return False


@TransformRegistry.register("initialize_qwen_mrope_delta_cache")
class InitializeMropeDeltaCache(BaseTransform):
    """Allocate a per-slot mrope_delta_cache for multimodal mRoPE models.

    Skipped when no submodule forward method references the
    ``qwen3_mrope_delta_with_cache`` custom op.
    """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        if not _model_uses_mrope_delta_cache(mod):
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        cm.add_resource("mrope_delta_cache", StateResourceHandler(1, dtype=torch.int32))

        return mod, TransformInfo(
            skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True
        )
