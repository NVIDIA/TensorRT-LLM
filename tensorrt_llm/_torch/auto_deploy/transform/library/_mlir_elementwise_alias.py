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

_LEGACY_ALIAS_REWRITES = {
    "fuse_l2norm": ("l2norm",),
    "fuse_relu2_quant_nvfp4": ("relu2_quant_nvfp4",),
    "fuse_rmsnorm_quant_fp8": ("rmsnorm_quant_fp8",),
    "fuse_rmsnorm_quant_nvfp4": ("rmsnorm_quant_nvfp4",),
    "fuse_silu_mul": ("silu_mul",),
}


def _canonical_mlir_enabled(shared_config: SharedConfig | None) -> bool:
    if shared_config is None:
        return False
    transform_config = getattr(shared_config, "transform_config", {}) or {}
    mlir_config = transform_config.get("mlir_elementwise_fusion")
    if mlir_config is None:
        return False
    if isinstance(mlir_config, dict):
        return mlir_config.get("enabled", True)
    return getattr(mlir_config, "enabled", True)


def apply_mlir_elementwise_alias(
    alias_name: str,
    config: TransformConfig,
    gm: GraphModule,
    cm: CachedSequenceInterface,
    factory: ModelFactory,
    shared_config: SharedConfig | None,
) -> Tuple[GraphModule, TransformInfo]:
    """Run ``mlir_elementwise_fusion`` for a legacy elementwise transform name."""
    enabled_rewrites = _LEGACY_ALIAS_REWRITES.get(alias_name)
    if not config.enabled or enabled_rewrites is None or _canonical_mlir_enabled(shared_config):
        return gm, TransformInfo(skipped=True, num_matches=0)

    mlir_config = MLIRElementwiseFusionConfig(
        stage=config.stage,
        enabled=True,
        enabled_rewrites=list(enabled_rewrites),
        run_graph_cleanup=config.run_graph_cleanup,
        run_shape_prop=config.run_shape_prop,
        requires_clean_graph=config.requires_clean_graph,
        requires_shape_prop=config.requires_shape_prop,
        debug_visualize_dir=config.debug_visualize_dir,
    )
    return MLIRElementwiseFusion(mlir_config)._apply(gm, cm, factory, shared_config)
