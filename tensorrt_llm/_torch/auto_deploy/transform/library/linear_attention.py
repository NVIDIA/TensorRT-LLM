# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pattern matching for linear attention patterns (causal conv)."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.fx import GraphModule

from ...custom_ops.mamba import torch_causal_conv  # noqa: F401 - registers custom op
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("match_causal_conv")
class MatchCausalConv(BaseTransform):
    """Match causal conv pattern and replace with torch_causal_conv1d reference op."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        patterns = ADPatternMatcherPass()

        # Dummy args for tracing the pattern
        batch, seq_len, channels, kernel_size = 2, 16, 64, 4
        dummy_args = [
            torch.randn(batch, seq_len, channels, device="meta", dtype=torch.float16),
            torch.randn(channels, channels, kernel_size, device="meta", dtype=torch.float16),
            torch.randn(channels, device="meta", dtype=torch.float16),
        ]

        def _causal_conv_pattern(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            """Pattern: transpose -> conv1d (with padding) -> slice -> transpose."""
            seq_len = x.shape[1]
            kernel_size = weight.shape[2]
            padding = kernel_size - 1
            x_t = x.transpose(1, 2)
            conv_out = F.conv1d(x_t, weight, bias, padding=padding)
            conv_out = conv_out[:, :, :seq_len]
            return conv_out.transpose(1, 2)

        def _causal_conv_replacement(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            """Replacement: torch_causal_conv1d custom op."""
            kernel_size = weight.shape[2]
            return torch.ops.auto_deploy.torch_causal_conv1d(
                x, weight, bias, 1, kernel_size - 1, 1, 1, "zeros"
            )

        register_ad_pattern(
            search_fn=_causal_conv_pattern,
            replace_fn=_causal_conv_replacement,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types={
                torch.ops.aten.slice.Tensor: (int,),
                torch.ops.aten.conv1d.default: (list, tuple, int),
            },
        )

        num_matches = patterns.apply(gm.graph)

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
