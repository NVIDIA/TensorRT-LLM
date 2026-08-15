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
"""A simple wrapper transform to build a model via the model factory."""

from typing import Tuple, Type

import torch.nn as nn
from pydantic import Field

from ...models import ModelFactory, hf
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class BuildModelConfig(TransformConfig):
    """Configuration for the build model transform."""

    device: str = Field(default="meta", description="The device to build the model on.")


@TransformRegistry.register("build_model")
class BuildModel(BaseTransform):
    """A simple wrapper transform to build a model via the model factory build_model method.

    This transform will build the model via the ``build_model`` method of the model factory on the
    meta device (or the set device) and not load the weights.
    """

    config: BuildModelConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return BuildModelConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # build the model
        model = factory.build_model(self.config.device)

        # update the kv cache config
        cm.update_kv_cache_config(**factory.get_cache_config_updates())

        # by convention, we say the model is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return model, info


@TransformRegistry.register("build_and_load_factory_model")
class BuildAndLoadFactoryModel(BuildModel):
    """A simple wrapper transform to build AND load a model via the factory's build_and_load API.

    Under the hood, the factory can use a different way to build and load the model at the same time
    rather than just building the model. For example, the HF factory uses the `.from_pretrained`
    API to directly build and load the model at the same time.

    We also assume that the `build_and_load_model` method will auto-shard the model appropriately.
    """

    config: BuildModelConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # load model with auto sharding
        assert isinstance(factory, hf.AutoModelFactory), "Only HF models are supported."

        # build and load the model
        model = factory.build_and_load_model(cm.device)

        # we set the standard example sequence WITHOUT extra_args to set them to None so that
        # only the text portion of the model gets called.
        cm.info.set_example_sequence()

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return model, info
