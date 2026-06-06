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
"""Compatibility helpers for legacy elementwise transform names."""

from typing import Tuple

from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import SharedConfig, TransformConfig, TransformInfo
from .mlir_elementwise_fusion import MLIRElementwiseFusion, MLIRElementwiseFusionConfig


def apply_mlir_elementwise_alias(
    config: TransformConfig,
    gm: GraphModule,
    cm: CachedSequenceInterface,
    factory: ModelFactory,
    shared_config: SharedConfig,
) -> Tuple[GraphModule, TransformInfo]:
    """Run ``mlir_elementwise_fusion`` for a legacy elementwise transform name."""
    if not config.enabled:
        return gm, TransformInfo(skipped=True, num_matches=0)

    mlir_config = MLIRElementwiseFusionConfig(
        stage=config.stage,
        enabled=True,
        run_graph_cleanup=config.run_graph_cleanup,
        run_shape_prop=config.run_shape_prop,
        requires_clean_graph=config.requires_clean_graph,
        requires_shape_prop=config.requires_shape_prop,
        debug_visualize_dir=config.debug_visualize_dir,
    )
    return MLIRElementwiseFusion(mlir_config)._apply(gm, cm, factory, shared_config)
