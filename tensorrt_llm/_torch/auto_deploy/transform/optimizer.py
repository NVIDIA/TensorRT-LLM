# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from .interface import (
    BaseTransform,
    DistConfig,
    InferenceOptimizerConfig,
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)


class InferenceOptimizer:
    def __init__(
        self,
        factory: ModelFactory,
        config: InferenceOptimizerConfig,
        dist_config: Optional[DistConfig] = None,
    ):
        self.factory = factory
        self.config = self._clean_config(config)
        self._cache_key_config = self._copy_config(self.config)
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()

        self.shared_config = SharedConfig(
            local_rank=local_rank,
            world_size=world_size,
            dist_config=dist_config,
        )

    def _clean_config(self, config: InferenceOptimizerConfig) -> StrictInferenceOptimizerConfig:
        """Get a typed checked ("strict") config with sorted keys according to stages."""
        # convert to nested kwargs, no TransformConfig objects allowed
        nested_kwargs = {
            k: v.model_dump() if isinstance(v, TransformConfig) else v for k, v in config.items()
        }
        # sort by stage
        keys_sorted = sorted(nested_kwargs.keys(), key=lambda k: Stages(nested_kwargs[k]["stage"]))
        # create strict config with correct config classes and correct order
        strict_config: StrictInferenceOptimizerConfig = {
            k: TransformRegistry.get_config_class(k)(**nested_kwargs[k]) for k in keys_sorted
        }
        # return strict config
        return strict_config

    def _copy_config(
        self, config: StrictInferenceOptimizerConfig
    ) -> StrictInferenceOptimizerConfig:
        """Return a deep copy used for stable cache keys across mutating transforms."""
        return {k: v.model_copy(deep=True) for k, v in config.items()}

    def __call__(self, cm: CachedSequenceInterface, mod: Optional[nn.Module] = None) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        can_restore_from_prefix = mod is None

        # start with an empty model if not provided
        if mod is None:
            mod = nn.Module()

        start_time = time.time()
        start_idx = 0
        if can_restore_from_prefix:
            restored_mod, start_idx = self._maybe_restore_from_cache(cm)
            if restored_mod is not None:
                mod = restored_mod

        # iterate over all transforms sorted by stage in the config
        for idx, (t_name, t_config) in enumerate(list(self.config.items())[start_idx:], start_idx):
            # instantiate transform
            transform = self._create_transform(t_name, t_config)
            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config, idx)
        total_time = time.time() - start_time
        ad_logger.info(f"Total time for all transforms: {total_time:.2f}s")

        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()
        return mod

    def _maybe_restore_from_cache(
        self, cm: CachedSequenceInterface
    ) -> Tuple[Optional[nn.Module], int]:
        """Ask cache transforms for a restore before running their prefix."""
        for idx, (t_name, t_config) in reversed(list(enumerate(self._cache_key_config.items()))):
            transform_cls = TransformRegistry.get(t_name)
            if not callable(getattr(transform_cls, "maybe_restore", None)):
                continue
            transform = self._create_transform(t_name, t_config)
            restored_mod = transform.maybe_restore(cm, self.factory, self.shared_config, idx)
            if restored_mod is not None:
                return restored_mod, idx + 1
        return None, 0

    def _create_transform(self, t_name: str, t_config: TransformConfig) -> BaseTransform:
        transform = TransformRegistry.get(t_name)(t_config)
        if t_name == "pipeline_cache":
            transform.set_cache_key_config(self._cache_key_config)
        return transform
