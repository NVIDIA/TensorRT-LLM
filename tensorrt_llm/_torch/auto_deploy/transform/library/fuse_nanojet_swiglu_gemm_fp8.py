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

"""Collapse the whole SwiGLU MLP front half into one nanojet gated GEMM.

The graph arrives as two FP8 projections off one activation, then silu, then a multiply::

    gate_projection = fp8_linear(x, gate_proj.weight, ..., [in_scale], [gate_scale])
    up_projection = fp8_linear(x, up_proj.weight, ..., [in_scale], [up_scale])
    h = mul(silu(gate_projection), up_projection)

Four kernels once the quantize before ``down_proj`` is counted. nanojet's ``swiglu`` is a
CUTLASS gated GEMM that does all of it and writes e4m3, which is what the native path runs.
Ordered before ``fuse_nanojet_act_and_mul`` so this takes the MLPs it can and that one keeps
whatever is left.
"""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ....nanojet_utils import ensure_tune_configs
from ...custom_ops.linear.nanojet_swiglu_gemm_fp8 import register
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.nanojet_graph import (
    accepts_out_dtype,
    match_fp8_projection,
    per_tensor_scale,
    set_val_meta,
)
from ...utils.node_utils import (
    collect_terminal_users_through_passthrough,
    get_shared_input_scale_for_fp8_linears,
    is_op,
    set_op_args,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class FuseNanojetSwiGLUGemmFP8Config(TransformConfig):
    """Configuration for the nanojet gated SwiGLU GEMM fusion."""


@TransformRegistry.register("fuse_nanojet_swiglu_gemm_fp8")
class FuseNanojetSwiGLUGemmFP8(BaseTransform):
    """Replace two FP8 projections + silu + mul with one nanojet gated GEMM."""

    config: FuseNanojetSwiGLUGemmFP8Config

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNanojetSwiGLUGemmFP8Config

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled or not register():
            return gm, TransformInfo(skipped=True, num_matches=0)

        ensure_tune_configs(factory)

        graph = gm.graph
        num_matches = 0

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.aten.mul.Tensor) or len(node.args) != 2:
                continue
            first, second = node.args
            # The product is symmetric; the activation marks which side is the gate_projection.
            if is_op(first, torch.ops.aten.silu):
                silu_node, up_node = first, second
            elif is_op(second, torch.ops.aten.silu):
                silu_node, up_node = second, first
            else:
                continue
            if not isinstance(up_node, Node) or len(silu_node.users) != 1:
                continue

            gate = match_fp8_projection(gm, silu_node.args[0])
            up = match_fp8_projection(gm, up_node)
            if gate is None or up is None:
                continue
            # One GEMM means one activation and one activation scale for both halves.
            if gate.activation is not up.activation or gate.input_scale != up.input_scale:
                continue
            if gate.weight.shape != up.weight.shape:
                continue
            readers, traversal_ok = collect_terminal_users_through_passthrough(node)
            fp8_readers, scale = get_shared_input_scale_for_fp8_linears(readers)
            if not (traversal_ok and fp8_readers and len(fp8_readers) == len(readers)):
                continue
            output_scale = per_tensor_scale(gm, scale)
            if output_scale is None:
                continue

            # ``[up, gate]`` — up first, which is the order nanojet's kernel indexes.
            stacked = torch.cat([up.weight, gate.weight], dim=0).contiguous()
            # Graph-unique node name, not ``id()``: an address is neither stable across runs
            # nor unique over time, since a collected node's address can be reused.
            weight_name = f"nanojet_gate_up_weight_{node.name}"
            gm.register_buffer(weight_name, stacked)

            with graph.inserting_before(node):
                weight_node = graph.get_attr(weight_name)
                set_val_meta(weight_node, stacked)
                # Fresh get_attr: the projections' scale nodes may sit later in the graph
                # than the node being inserted here.
                gate_scale_node = graph.get_attr(gate.input_scale_node.target)
                fused = graph.call_function(
                    torch.ops.auto_deploy.nanojet_swiglu_gemm_fp8.default,
                    args=(
                        gate.activation,
                        gate_scale_node,
                        weight_node,
                        gate.input_scale,
                        up.weight_scale,
                        gate.weight_scale,
                        1.0 / output_scale,
                    ),
                )
            # An e4m3 activation carries no hint of the output dtype, but only the TensorRT LLM
            # linear takes the hint; the others fix their return dtype in the implementation.
            for consumer in fp8_readers:
                if accepts_out_dtype(consumer):
                    set_op_args(consumer, out_dtype=str(node.meta["val"].dtype).rsplit(".", 1)[-1])
            set_val_meta(fused, node, dtype=torch.float8_e4m3fn)
            node.replace_all_uses_with(fused)
            num_matches += 1

        if num_matches:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(f"fuse_nanojet_swiglu_gemm_fp8: {num_matches} gated SwiGLU GEMMs fused")

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=True,
        )
