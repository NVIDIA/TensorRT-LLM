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

Matches patterns where relu²(x) feeds into torch_quant_nvfp4_linear and
replaces them with a fused relu²+quantize kernel followed by a GEMM-only op
that takes pre-quantized FP4 input.

Pattern:
    up_out = some_linear(x, ...)
    relu_out = aten.relu(up_out)
    square_out = aten.pow(relu_out, 2)  OR  aten.mul(relu_out, relu_out)
    down_out = torch_quant_nvfp4_linear(square_out, weight, bias,
                                         input_scale, weight_scale, alpha)

Replaced with:
    fp4_out, sf_out = trtllm_fused_relu2_quant_nvfp4(up_out, input_scale)
    down_out = trtllm_nvfp4_prequant_linear(fp4_out, weight, sf_out,
                                             weight_scale, alpha, bias, dtype)

This eliminates the DRAM round-trip between the activation and quantisation
steps by fusing relu² and FP4 block-scale quantisation in one kernel pass.
"""

import operator
from typing import List, Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _is_relu2_chain(
    input_node: Node, gm: Optional[GraphModule] = None
) -> Optional[Tuple[Node, List[Node]]]:
    """Check if input_node is the output of a relu²(x) = square(relu(x)) chain.

    Handles:
    1. Aten path: aten.relu(x) -> aten.pow(..., 2) or aten.mul(x, x).
    2. Call_module path: call_module(ReLUSquaredActivation, x) (e.g. HuggingFace ACT2FN["relu2"]).

    Returns:
        (relu_input, nodes_to_erase) if matched, else None.
    """
    # Path 1: input is output of call_module(ReLUSquaredActivation)
    if input_node.op == "call_module" and gm is not None:
        try:
            target = input_node.target
            if not isinstance(target, str):
                return None
            submod = gm.get_submodule(target)
            type_name = type(submod).__name__
            # HuggingFace transformers.activations.ReLUSquaredActivation; allow subclass names
            if type_name == "ReLUSquaredActivation" or "ReLUSquared" in type_name:
                if len(input_node.args) >= 1 and isinstance(input_node.args[0], Node):
                    return (input_node.args[0], [input_node])
        except Exception:
            pass
        return None

    # Path 2: input is output of aten.square(relu), aten.pow(relu, 2), or aten.mul(relu, relu)
    square_node = input_node
    relu_node = None

    if is_op(square_node, torch.ops.aten.square.default):
        relu_node = square_node.args[0]
    elif is_op(square_node, torch.ops.aten.pow.Tensor_Scalar):
        if square_node.args[1] == 2:
            relu_node = square_node.args[0]
    elif is_op(square_node, torch.ops.aten.mul.Tensor):
        lhs, rhs = square_node.args[0], square_node.args[1]
        if lhs is rhs:
            relu_node = lhs

    if relu_node is None or not isinstance(relu_node, Node):
        return None

    if not is_op(relu_node, torch.ops.aten.relu.default):
        return None

    relu_input = relu_node.args[0]
    if not isinstance(relu_input, Node):
        return None

    return (relu_input, [relu_node, square_node])


def _get_out_dtype_str(node: Node) -> str:
    if "val" in node.meta:
        val = node.meta["val"]
        if hasattr(val, "dtype"):
            return val.dtype
    return torch.bfloat16


@TransformRegistry.register("fuse_relu2_quant_nvfp4")
class FuseRelu2QuantNVFP4(BaseTransform):
    """Fuse ReLU² + NVFP4 quantization into the trtllm fused kernel.

    Matches two patterns:

    1. Aten: relu(x) -> square(x) or pow(..., 2) or mul(x, x) -> torch_quant_nvfp4_linear(...)
    2. Call_module: call_module(ReLUSquaredActivation, x) -> torch_quant_nvfp4_linear(...)
       (e.g. HuggingFace ACT2FN["relu2"] used by Nemotron)

    Replaces with:
        fp4_out, sf_out = trtllm_fused_relu2_quant_nvfp4(x, input_scale)
        linear_out = trtllm_nvfp4_prequant_linear(fp4_out, w_fp4, sf_out,
                                                    weight_scale, alpha, bias)
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

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear):
                continue

            input_arg, weight_fp4, bias, input_scale, weight_scale, alpha = extract_op_args(
                node, "input", "weight_fp4", "bias", "input_scale", "weight_scale", "alpha"
            )

            if not isinstance(input_arg, Node):
                continue

            chain = _is_relu2_chain(input_arg, gm)
            if chain is None:
                if input_arg.op == "call_module" and isinstance(input_arg.target, str):
                    try:
                        submod = gm.get_submodule(input_arg.target)
                        ad_logger.info(
                            "fuse_relu2_quant_nvfp4: linear input is call_module(%s), "
                            "type=%s (need ReLUSquaredActivation)",
                            input_arg.target,
                            type(submod).__name__,
                        )
                    except Exception as e:
                        ad_logger.info(
                            "fuse_relu2_quant_nvfp4: linear input is call_module(%s), "
                            "get_submodule failed: %s",
                            input_arg.target,
                            e,
                        )
                else:
                    ad_logger.info(
                        "fuse_relu2_quant_nvfp4: linear input op=%s (target=%s), not relu2"
                        % (input_arg.op, getattr(input_arg, "target", None))
                    )
                continue

            relu_input, nodes_to_erase = chain

            out_dtype = _get_out_dtype_str(node)

            with graph.inserting_before(node):
                fused_quant = graph.call_function(
                    torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default,
                    args=(relu_input, input_scale),
                )
                fp4_node = graph.call_function(operator.getitem, args=(fused_quant, 0))
                sf_node = graph.call_function(operator.getitem, args=(fused_quant, 1))

                gemm_node = graph.call_function(
                    torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default,
                    args=(fp4_node, weight_fp4, sf_node, weight_scale, alpha),
                    kwargs={"bias": bias, "out_dtype": out_dtype},
                )

            node.replace_all_uses_with(gemm_node)
            graph.erase_node(node)

            for n in nodes_to_erase:
                if len(n.users) == 0:
                    graph.erase_node(n)

            cnt += 1

        gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
