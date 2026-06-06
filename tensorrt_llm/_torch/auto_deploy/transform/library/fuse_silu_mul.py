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

"""Compatibility alias for SiLU+Mul fusion through MLIR elementwise fusion."""

from typing import Tuple, Type

from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from ._mlir_elementwise_alias import apply_mlir_elementwise_alias


class FuseSiluMulConfig(TransformConfig):
    """Compatibility configuration for the SiLU+Mul fusion transform."""

    backend: str = Field(
        default="flashinfer",
        description=(
            "Deprecated compatibility field. SiLU+Mul fusion is handled by "
            "mlir_elementwise_fusion regardless of this value."
        ),
    )


@TransformRegistry.register("fuse_silu_mul")
class FuseSiluMul(BaseTransform):
    """Compatibility alias for SiLU+Mul fusion through ``mlir_elementwise_fusion``."""

    config: FuseSiluMulConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseSiluMulConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        return apply_mlir_elementwise_alias(
            "fuse_silu_mul", self.config, gm, cm, factory, shared_config
        )
