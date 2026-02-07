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

"""Graph transforms for SwiGLU MLP fusion.

This module provides two-stage transformation for SwiGLU MLP:
1. MatchSwiGLUPattern: Detects SwiGLU patterns and replaces with torch_swiglu_mlp
2. FuseSwiGLU: Fuses gate+up weights into a single concatenated matmul

The SwiGLU pattern is: silu(x @ gate.T) * (x @ up.T) @ down.T
"""

from typing import Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

# Import the custom ops to ensure they are registered and for use in replacements
from ...custom_ops.linear.swiglu import torch_swiglu_mlp
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import delete_all_unused_submodules, eliminate_dead_code, get_attr_by_name
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _swiglu_pattern_no_bias(x, gate_weight, up_weight, down_weight):
    """Pattern for SwiGLU MLP without biases.

    Matches: silu(linear(x, gate_weight, None)) * linear(x, up_weight, None) -> linear(down_weight, None)
    """
    gate_out = torch.ops.auto_deploy.torch_linear_simple.default(x, gate_weight, None)
    up_out = torch.ops.auto_deploy.torch_linear_simple.default(x, up_weight, None)
    silu_out = torch.ops.aten.silu.default(gate_out)
    mul_out = torch.ops.aten.mul.Tensor(silu_out, up_out)
    down_out = torch.ops.auto_deploy.torch_linear_simple.default(mul_out, down_weight, None)
    return down_out


def _swiglu_replacement_no_bias(x, gate_weight, up_weight, down_weight):
    """Replacement for SwiGLU pattern without biases."""
    # Call the Python wrapper directly, not via torch.ops.auto_deploy
    # This ensures proper FakeTensor mode handling during tracing
    return torch_swiglu_mlp(x, gate_weight, up_weight, down_weight, None, None, None)


def _swiglu_pattern_with_bias(
    x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias
):
    """Pattern for SwiGLU MLP with biases.

    Matches: silu(linear(x, gate_weight, gate_bias)) * linear(x, up_weight, up_bias) -> linear(down_weight, down_bias)
    """
    gate_out = torch.ops.auto_deploy.torch_linear_simple.default(x, gate_weight, gate_bias)
    up_out = torch.ops.auto_deploy.torch_linear_simple.default(x, up_weight, up_bias)
    silu_out = torch.ops.aten.silu.default(gate_out)
    mul_out = torch.ops.aten.mul.Tensor(silu_out, up_out)
    down_out = torch.ops.auto_deploy.torch_linear_simple.default(mul_out, down_weight, down_bias)
    return down_out


def _swiglu_replacement_with_bias(
    x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias
):
    """Replacement for SwiGLU pattern with biases."""
    # Call the Python wrapper directly, not via torch.ops.auto_deploy
    # This ensures proper FakeTensor mode handling during tracing
    return torch_swiglu_mlp(x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias)


@TransformRegistry.register("match_swiglu_pattern")
class MatchSwiGLUPattern(BaseTransform):
    """Matches SwiGLU MLP patterns and replaces with torch_swiglu_mlp op.

    This transform runs in the pattern_matcher stage and detects the following pattern:
        silu(x @ gate.T) * (x @ up.T) @ down.T

    And replaces it with a single torch_swiglu_mlp op that can be fused later.

    Uses ADPatternMatcherPass for declarative pattern matching.
    """

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        patterns = ADPatternMatcherPass()

        # Dummy shapes for tracing - shapes don't matter for matching
        hidden, intermediate = 128, 256

        # Pattern 1: SwiGLU without biases (most common case)
        dummy_args_no_bias = [
            torch.randn(2, hidden, device="meta", dtype=torch.float16),  # x
            torch.randn(intermediate, hidden, device="meta", dtype=torch.float16),  # gate_weight
            torch.randn(intermediate, hidden, device="meta", dtype=torch.float16),  # up_weight
            torch.randn(hidden, intermediate, device="meta", dtype=torch.float16),  # down_weight
        ]
        register_ad_pattern(
            search_fn=_swiglu_pattern_no_bias,
            replace_fn=_swiglu_replacement_no_bias,
            patterns=patterns,
            dummy_args=dummy_args_no_bias,
        )

        # Pattern 2: SwiGLU with biases
        dummy_args_with_bias = [
            torch.randn(2, hidden, device="meta", dtype=torch.float16),  # x
            torch.randn(intermediate, hidden, device="meta", dtype=torch.float16),  # gate_weight
            torch.randn(intermediate, hidden, device="meta", dtype=torch.float16),  # up_weight
            torch.randn(hidden, intermediate, device="meta", dtype=torch.float16),  # down_weight
            torch.randn(intermediate, device="meta", dtype=torch.float16),  # gate_bias
            torch.randn(intermediate, device="meta", dtype=torch.float16),  # up_bias
            torch.randn(hidden, device="meta", dtype=torch.float16),  # down_bias
        ]
        register_ad_pattern(
            search_fn=_swiglu_pattern_with_bias,
            replace_fn=_swiglu_replacement_with_bias,
            patterns=patterns,
            dummy_args=dummy_args_with_bias,
        )

        num_matches = patterns.apply(gm.graph)

        if num_matches > 0:
            gm.recompile()

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )

        return gm, info


class FuseSwiGLUConfig(TransformConfig):
    """Configuration for the SwiGLU fusion transform."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable SwiGLU fusion.",
    )


@TransformRegistry.register("fuse_swiglu")
class FuseSwiGLU(BaseTransform):
    """Fuses torch_swiglu_mlp ops by concatenating gate and up weights.

    This transform runs in the post_load_fusion stage and replaces torch_swiglu_mlp ops
    with fused_swiglu_mlp ops that use a single concatenated gate+up weight matrix.

    This reduces memory bandwidth by performing a single matmul instead of two
    separate matmuls for gate and up projections.
    """

    config: FuseSwiGLUConfig

    @classmethod
    def get_config_class(cls) -> Type[FuseSwiGLUConfig]:
        return FuseSwiGLUConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled:
            return gm, TransformInfo(skipped=True, num_matches=0)

        graph = gm.graph
        cnt = 0
        fused_weight_idx = 0

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_swiglu_mlp.default):
                continue

            # Extract args: (input, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias)
            input_node = node.args[0]
            gate_weight_node = node.args[1]
            up_weight_node = node.args[2]
            down_weight_node = node.args[3]
            gate_bias_node = node.args[4] if len(node.args) > 4 else None
            up_bias_node = node.args[5] if len(node.args) > 5 else None
            down_bias_node = node.args[6] if len(node.args) > 6 else None

            # Get the actual weight tensors
            gate_weight = get_attr_by_name(gm, gate_weight_node.target)
            up_weight = get_attr_by_name(gm, up_weight_node.target)

            # Concatenate gate and up weights: [intermediate, hidden] -> [2*intermediate, hidden]
            gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

            # Create new attribute for the fused weight
            fused_weight_name = f"fused_swiglu_gate_up_{fused_weight_idx}"
            gm.register_buffer(fused_weight_name, gate_up_weight)

            # Handle biases
            gate_up_bias_node = None
            fused_bias_name = None

            if gate_bias_node is not None and gate_bias_node.op == "get_attr":
                gate_bias = get_attr_by_name(gm, gate_bias_node.target)
                up_bias = get_attr_by_name(gm, up_bias_node.target) if up_bias_node else None

                if up_bias is not None:
                    gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)
                    fused_bias_name = f"fused_swiglu_gate_up_bias_{fused_weight_idx}"
                    gm.register_buffer(fused_bias_name, gate_up_bias)

            # Create get_attr node for the fused weight
            with graph.inserting_before(node):
                fused_weight_node = graph.get_attr(fused_weight_name)

                if fused_bias_name is not None:
                    gate_up_bias_node = graph.get_attr(fused_bias_name)

            # Create the fused_swiglu_mlp node
            with graph.inserting_after(node):
                fused_node: Node = graph.call_function(
                    torch.ops.auto_deploy.fused_swiglu_mlp.default,
                    args=(
                        input_node,
                        fused_weight_node,
                        down_weight_node,
                        gate_up_bias_node,
                        down_bias_node,
                    ),
                )

            # Replace uses and erase old node
            node.replace_all_uses_with(fused_node)
            graph.erase_node(node)

            fused_weight_idx += 1
            cnt += 1

        if cnt > 0:
            gm.recompile()

            # Clean up dead code and free memory from unfused weights
            # The original gate_weight and up_weight tensors are no longer referenced
            # after fusion, so we can delete them to save GPU memory.
            eliminate_dead_code(gm)
            delete_all_unused_submodules(gm)

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info
