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

"""Fuse ReLU² activation + NVFP4 quantization into a single kernel.

Matches exported aten patterns where relu²(x) feeds into
`torch_quant_nvfp4_linear` and replaces them with a fused relu²+quantize
kernel followed by a GEMM-only op that takes pre-quantized FP4 input.

Supported patterns:
    relu_out = aten.relu(x)
    relu2_out = aten.square(relu_out)
    relu2_out = aten.pow(relu_out, 2)
    relu2_out = aten.mul(relu_out, relu_out)
    out = torch_quant_nvfp4_linear(relu2_out, weight, bias, ...)

Replaced with:
    fp4_out, sf_out = trtllm_fused_relu2_quant_nvfp4(x, input_scale)
    out = trtllm_nvfp4_prequant_linear(fp4_out, weight, sf_out,
                                       weight_scale, alpha, bias, out_dtype)

This matcher-only transform intentionally does not handle `call_module`
ReLU² variants or patterns with shared intermediate users.
"""

import operator
from typing import Optional, Tuple, Type

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match, register_graph_pattern
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.pattern_matcher import ADPatternMatcherPass
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _get_out_dtype(node: Node) -> torch.dtype:
    val = node.meta.get("val")
    if hasattr(val, "dtype"):
        return val.dtype
    return torch.bfloat16


def _fuse_relu2_quant_handler(
    match: Match,
    x: Node,
    weight_fp4: Node,
    input_scale: Node,
    weight_scale: Node,
    alpha: Node,
    bias: Optional[Node] = None,
) -> None:
    graph = match.graph
    output_node = match.output_node()
    out_dtype = _get_out_dtype(output_node)

    with graph.inserting_before(output_node):
        fused_quant = graph.call_function(
            torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default,
            args=(x, input_scale),
        )
        fp4_out = graph.call_function(operator.getitem, args=(fused_quant, 0))
        sf_out = graph.call_function(operator.getitem, args=(fused_quant, 1))
        fused_linear = graph.call_function(
            torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default,
            args=(fp4_out, weight_fp4, sf_out, weight_scale, alpha),
            kwargs={"bias": bias, "out_dtype": out_dtype},
        )

    output_node.replace_all_uses_with(fused_linear)
    match.erase_nodes()


def _register_relu2_quant_nvfp4_patterns(patterns: ADPatternMatcherPass) -> None:
    def _register(pattern) -> None:
        register_graph_pattern(pattern, pass_dict=patterns)(_fuse_relu2_quant_handler)

    x = KeywordArg("x")
    weight_fp4 = KeywordArg("weight_fp4")
    bias = KeywordArg("bias")
    input_scale = KeywordArg("input_scale")
    weight_scale = KeywordArg("weight_scale")
    alpha = KeywordArg("alpha")

    relu = CallFunction(torch.ops.aten.relu.default, x)
    relu2_patterns = (
        CallFunction(torch.ops.aten.square.default, relu),
        CallFunction(torch.ops.aten.pow.Tensor_Scalar, relu, 2),
        CallFunction(torch.ops.aten.mul.Tensor, relu, relu),
    )

    for relu2 in relu2_patterns:
        _register(
            CallFunction(
                torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
                relu2,
                weight_fp4,
                None,
                input_scale,
                weight_scale,
                alpha,
            )
        )
        _register(
            CallFunction(
                torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
                relu2,
                weight_fp4,
                bias,
                input_scale,
                weight_scale,
                alpha,
            )
        )


@TransformRegistry.register("fuse_relu2_quant_nvfp4")
class FuseRelu2QuantNVFP4(BaseTransform):
    """Fuse matcher-supported ReLU² + NVFP4 quantization patterns."""

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
        _register_relu2_quant_nvfp4_patterns(patterns)
        cnt = patterns.apply(gm.graph)

        if cnt > 0:
            gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
