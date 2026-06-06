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
"""Graph transforms for standardizing L2Norm and routing fusion through MLIR."""

from typing import Literal, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface

# It is important to import ADPatternMatcherPass from pattern_matcher.py, not from torch._inductor.pattern_matcher
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from ._mlir_elementwise_alias import apply_mlir_elementwise_alias


def _l2_norm_pattern(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Implements the L2Norm pattern for pattern matching.

    L2 normalization: x / sqrt(sum(x^2) + eps)

    Args:
        data: Input tensor to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2 normalized tensor.
    """
    input_dtype = data.dtype
    data = data.to(torch.float32)
    sum_sq = (data * data).sum(dim=-1, keepdim=True)
    data = data * torch.rsqrt(sum_sq + eps)
    return data.to(input_dtype)


def _l2_norm_pattern_no_dtype_cast(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Implements the L2Norm pattern without dtype casting for pattern matching.

    Some models may already operate in float32 and skip the dtype cast.

    Args:
        data: Input tensor to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2 normalized tensor.
    """
    sum_sq = (data * data).sum(dim=-1, keepdim=True)
    return data * torch.rsqrt(sum_sq + eps)


def _l2_norm_to_torch_l2norm(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace L2Norm pattern with torch_l2norm op (standardized representation).

    Args:
        data: Input tensor to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2 normalized tensor using torch_l2norm.
    """
    return torch.ops.auto_deploy.torch_l2norm(data, eps)


@TransformRegistry.register("match_l2norm_pattern")
class MatchL2NormPattern(BaseTransform):
    """Matches L2Norm patterns in the graph and replaces them with torch_l2norm op.

    This transform runs in the pattern_matcher stage and standardizes L2Norm patterns
    to use torch_l2norm op, which can later be fused to a specific backend in the
    post_load_fusion stage.

    Args:
        gm: Input graph module to transform.

    Returns:
        Transformed graph module with standardized torch_l2norm operations.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()

        bs = 2
        hidden_size = 512

        def dummy_args(input_dtype: torch.dtype, eps: float = 1e-6):
            return [
                torch.randn(bs, hidden_size, device="cuda", dtype=input_dtype),
                eps,
            ]

        configs = [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]

        search_fns = [
            _l2_norm_pattern,
            _l2_norm_pattern_no_dtype_cast,
        ]
        for search_fn in search_fns:
            for input_dtype in configs:
                register_ad_pattern(
                    search_fn=search_fn,
                    replace_fn=_l2_norm_to_torch_l2norm,
                    patterns=patterns,
                    dummy_args=dummy_args(input_dtype),
                    op_ignore_types={},
                    scalar_workaround={"eps": 1e-6},
                    skip_duplicates=True,
                )

        cnt = patterns.apply(graph)

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info


class FuseL2NormConfig(TransformConfig):
    """Configuration for the L2Norm fusion transform."""

    backend: Literal["torch", "fla"] = Field(
        default="fla",
        description="Backend to use for L2Norm computation ('fla' or 'torch').",
    )


@TransformRegistry.register("fuse_l2norm")
class FuseL2Norm(BaseTransform):
    """Compatibility alias for L2Norm fusion through ``mlir_elementwise_fusion``."""

    config: FuseL2NormConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseL2NormConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        return apply_mlir_elementwise_alias(
            "fuse_l2norm", self.config, gm, cm, factory, shared_config
        )
