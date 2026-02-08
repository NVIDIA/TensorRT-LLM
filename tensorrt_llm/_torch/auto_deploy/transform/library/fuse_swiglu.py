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


# ── NVFP4 quantized SwiGLU pattern matching and fusion ──────────────────────

from ...custom_ops.linear.swiglu import torch_nvfp4_swiglu_mlp  # noqa: E402


def _nvfp4_swiglu_pattern_no_bias(
    x,
    gate_weight,
    gate_input_scale,
    gate_weight_scale,
    gate_alpha,
    up_weight,
    up_input_scale,
    up_weight_scale,
    up_alpha,
    down_weight,
    down_input_scale,
    down_weight_scale,
    down_alpha,
):
    """Pattern for NVFP4 quantized SwiGLU MLP without biases.

    Matches: silu(nvfp4_linear(x, gate)) * nvfp4_linear(x, up) -> nvfp4_linear(down)
    """
    gate_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear.default(
        x,
        gate_weight,
        None,
        input_scale=[gate_input_scale],
        weight_scale=[gate_weight_scale, gate_alpha],
        input_zp=[],
        weight_zp=[],
    )
    up_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear.default(
        x,
        up_weight,
        None,
        input_scale=[up_input_scale],
        weight_scale=[up_weight_scale, up_alpha],
        input_zp=[],
        weight_zp=[],
    )
    silu_out = torch.ops.aten.silu.default(gate_out)
    mul_out = torch.ops.aten.mul.Tensor(silu_out, up_out)
    down_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear.default(
        mul_out,
        down_weight,
        None,
        input_scale=[down_input_scale],
        weight_scale=[down_weight_scale, down_alpha],
        input_zp=[],
        weight_zp=[],
    )
    return down_out


def _nvfp4_swiglu_replacement_no_bias(
    x,
    gate_weight,
    gate_input_scale,
    gate_weight_scale,
    gate_alpha,
    up_weight,
    up_input_scale,
    up_weight_scale,
    up_alpha,
    down_weight,
    down_input_scale,
    down_weight_scale,
    down_alpha,
):
    """Replacement for NVFP4 quantized SwiGLU pattern without biases."""
    return torch_nvfp4_swiglu_mlp(
        x,
        gate_weight,
        up_weight,
        down_weight,
        gate_input_scale,
        gate_weight_scale,
        gate_alpha,
        up_input_scale,
        up_weight_scale,
        up_alpha,
        down_input_scale,
        down_weight_scale,
        down_alpha,
    )


@TransformRegistry.register("match_nvfp4_swiglu_pattern")
class MatchNVFP4SwiGLUPattern(BaseTransform):
    """Matches NVFP4 quantized SwiGLU MLP patterns and replaces with torch_nvfp4_swiglu_mlp.

    This transform runs in the pattern_matcher stage AFTER quantize_nvfp4_linear_from_config
    has converted torch_linear_simple ops to torch_fake_quant_nvfp4_linear ops.

    It detects the following NVFP4 pattern:
        silu(nvfp4_linear(x, gate)) * nvfp4_linear(x, up) -> nvfp4_linear(down)

    And replaces it with a single torch_nvfp4_swiglu_mlp op that can be fused later.
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

        # FP4 shape params for dummy args (shapes don't matter for matching)
        N = 32  # intermediate_size
        K_packed = 32  # hidden_size / 2 (FP4 packing)
        K_eff = 2 * K_packed  # actual hidden_size
        N_down = K_eff  # hidden_size (output of down proj)
        K_down_packed = N // 2  # intermediate_size / 2 (down proj input)

        # Weight scale sizes (per-block scale: N * K / 16)
        gate_cutlass_len = N * (K_eff // 16)
        down_cutlass_len = N_down * (N // 16)

        x = torch.randn(2, K_eff, device="meta", dtype=torch.float16)

        # Gate args
        gate_w = torch.randint(0, 255, (N, K_packed), device="meta", dtype=torch.uint8)
        gate_is = torch.tensor(0.01, device="meta", dtype=torch.float32)
        gate_ws = torch.randint(0, 255, (gate_cutlass_len,), device="meta", dtype=torch.uint8)
        gate_a = torch.tensor(1.2345, device="meta", dtype=torch.float32)

        # Up args (same shapes as gate)
        up_w = torch.randint(0, 255, (N, K_packed), device="meta", dtype=torch.uint8)
        up_is = torch.tensor(0.02, device="meta", dtype=torch.float32)
        up_ws = torch.randint(0, 255, (gate_cutlass_len,), device="meta", dtype=torch.uint8)
        up_a = torch.tensor(2.3456, device="meta", dtype=torch.float32)

        # Down args
        down_w = torch.randint(0, 255, (N_down, K_down_packed), device="meta", dtype=torch.uint8)
        down_is = torch.tensor(0.03, device="meta", dtype=torch.float32)
        down_ws = torch.randint(0, 255, (down_cutlass_len,), device="meta", dtype=torch.uint8)
        down_a = torch.tensor(3.4567, device="meta", dtype=torch.float32)

        dummy_args = [
            x,
            gate_w,
            gate_is,
            gate_ws,
            gate_a,
            up_w,
            up_is,
            up_ws,
            up_a,
            down_w,
            down_is,
            down_ws,
            down_a,
        ]

        register_ad_pattern(
            search_fn=_nvfp4_swiglu_pattern_no_bias,
            replace_fn=_nvfp4_swiglu_replacement_no_bias,
            patterns=patterns,
            dummy_args=dummy_args,
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


@TransformRegistry.register("fuse_nvfp4_swiglu")
class FuseNVFP4SwiGLU(BaseTransform):
    """Fuses torch_nvfp4_swiglu_mlp ops by concatenating gate and up FP4 weights.

    This transform runs in the post_load_fusion stage and replaces torch_nvfp4_swiglu_mlp
    ops with fused_nvfp4_swiglu_mlp ops that use a single concatenated gate+up weight matrix.

    FP4 weight fusion:
    - gate+up packed weights are concatenated along dim=0
    - gate+up per-block weight scales are concatenated along dim=0
    - gate+up input_scale and alpha must match (shared input)
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
        graph = gm.graph
        cnt = 0
        fused_weight_idx = 0

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_nvfp4_swiglu_mlp.default):
                continue

            # Extract args:
            # (input, gate_weight, up_weight, down_weight,
            #  gate_input_scale, gate_weight_scale, gate_alpha,
            #  up_input_scale, up_weight_scale, up_alpha,
            #  down_input_scale, down_weight_scale, down_alpha)
            input_node = node.args[0]
            gate_weight_node = node.args[1]
            up_weight_node = node.args[2]
            down_weight_node = node.args[3]
            gate_input_scale_node = node.args[4]
            gate_weight_scale_node = node.args[5]
            gate_alpha_node = node.args[6]
            up_input_scale_node = node.args[7]  # noqa: F841
            up_weight_scale_node = node.args[8]
            up_alpha_node = node.args[9]  # noqa: F841
            down_input_scale_node = node.args[10]
            down_weight_scale_node = node.args[11]
            down_alpha_node = node.args[12]

            # Get the actual weight tensors
            gate_weight = get_attr_by_name(gm, gate_weight_node.target)
            up_weight = get_attr_by_name(gm, up_weight_node.target)

            # Concatenate gate and up FP4 packed weights along dim=0
            gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

            # Get and concatenate weight scales
            gate_weight_scale = get_attr_by_name(gm, gate_weight_scale_node.target)
            up_weight_scale = get_attr_by_name(gm, up_weight_scale_node.target)
            gate_up_weight_scale = torch.cat([gate_weight_scale, up_weight_scale], dim=0)

            # Register fused buffers
            prefix = f"fused_nvfp4_swiglu_{fused_weight_idx}"
            gm.register_buffer(f"{prefix}_gate_up_weight", gate_up_weight)
            gm.register_buffer(f"{prefix}_gate_up_weight_scale", gate_up_weight_scale)

            # Create get_attr nodes for fused weights/scales
            with graph.inserting_before(node):
                fused_gate_up_weight_node = graph.get_attr(f"{prefix}_gate_up_weight")
                fused_gate_up_weight_scale_node = graph.get_attr(f"{prefix}_gate_up_weight_scale")

            # Create the fused_nvfp4_swiglu_mlp node
            # Use gate's input_scale and alpha (same as up's since they share input)
            with graph.inserting_after(node):
                fused_node: Node = graph.call_function(
                    torch.ops.auto_deploy.fused_nvfp4_swiglu_mlp.default,
                    args=(
                        input_node,
                        fused_gate_up_weight_node,
                        down_weight_node,
                        gate_input_scale_node,  # shared input_scale for gate+up
                        fused_gate_up_weight_scale_node,
                        gate_alpha_node,  # shared alpha for gate+up
                        down_input_scale_node,
                        down_weight_scale_node,
                        down_alpha_node,
                    ),
                )

            # Replace uses and erase old node
            node.replace_all_uses_with(fused_node)
            graph.erase_node(node)

            fused_weight_idx += 1
            cnt += 1

        if cnt > 0:
            gm.recompile()
            eliminate_dead_code(gm)
            delete_all_unused_submodules(gm)

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info
