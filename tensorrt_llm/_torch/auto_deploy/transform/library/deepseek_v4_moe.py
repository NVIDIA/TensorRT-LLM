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

"""DeepSeek V4 MoE lowering skeleton.

The correct production lowering must preserve three distinct pieces of behavior:

* DeepSeek V4 sqrt-softplus/hash routing.
* Packed MXFP4 routed experts with the DeepSeek V4 SwiGLU limit.
* FineGrained FP8 shared expert linears.

Until the packed routed-expert and shared-expert production paths are wired
together, this transform refuses to replace the canonical op by default. Tests
can opt into a dense reference lowering that keeps the graph export-friendly
without pretending to be the production path.
"""

from __future__ import annotations

import operator
from typing import Any

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.fused_moe.deepseek_v4_moe import (  # noqa: F401
    torch_deepseek_v4_moe,
    torch_deepseek_v4_moe_from_routing,
)
from ...custom_ops.fused_moe.deepseek_v4_router import torch_deepseek_v4_router  # noqa: F401
from ...utils.node_utils import is_op
from ..interface import BaseTransform, TransformConfig, TransformInfo, TransformRegistry

__all__ = [
    "DeepSeekV4MoELowering",
    "DeepSeekV4MoELoweringConfig",
    "DeepSeekV4MoELoweringError",
]


class DeepSeekV4MoELoweringError(RuntimeError):
    """Raised when a DeepSeek V4 MoE graph cannot be lowered safely."""


class DeepSeekV4MoELoweringConfig(TransformConfig):
    allow_reference_lowering: bool = Field(
        default=False,
        description=(
            "Allow lowering torch_deepseek_v4_moe to router + dense reference "
            "torch_deepseek_v4_moe_from_routing. This is intended for unit tests, "
            "not production inference."
        ),
    )


_ARG_NAMES = (
    "hidden_states",
    "input_ids",
    "router_weight",
    "router_bias",
    "tid2eid",
    "routed_w1_weight",
    "routed_w2_weight",
    "routed_w3_weight",
    "shared_w1_weight",
    "shared_w2_weight",
    "shared_w3_weight",
    "top_k",
    "route_scale",
    "swiglu_limit",
    "is_hash_layer",
    "layer_type",
)
_MISSING = object()


def _node_arg(node: Node, name: str, default: Any = _MISSING) -> Any:
    index = _ARG_NAMES.index(name)
    if len(node.args) > index:
        return node.args[index]
    if name in node.kwargs:
        return node.kwargs[name]
    if default is not _MISSING:
        return default
    raise DeepSeekV4MoELoweringError(
        f"Malformed torch_deepseek_v4_moe node is missing required argument {name!r}."
    )


def _unsupported_production_lowering_error() -> DeepSeekV4MoELoweringError:
    return DeepSeekV4MoELoweringError(
        "DeepSeek V4 MoE production lowering is not implemented for "
        "auto_deploy::torch_deepseek_v4_moe yet. A safe production lowering must "
        "lower routed experts to auto_deploy::triton_mxfp4_moe/triton_mxfp4_moe_ep "
        "and lower the shared expert through FineGrained FP8 linear/SwiGLU surfaces "
        "while preserving swiglu_limit. Set allow_reference_lowering=True only for "
        "tests or reference graphs."
    )


def _lower_to_reference_graph(gm: GraphModule, node: Node) -> None:
    graph = gm.graph
    hidden_states = _node_arg(node, "hidden_states")
    input_ids = _node_arg(node, "input_ids")
    router_weight = _node_arg(node, "router_weight")
    router_bias = _node_arg(node, "router_bias")
    tid2eid = _node_arg(node, "tid2eid")
    top_k = _node_arg(node, "top_k")
    route_scale = _node_arg(node, "route_scale")
    is_hash_layer = _node_arg(node, "is_hash_layer")

    with graph.inserting_before(node):
        router_node = graph.call_function(
            torch.ops.auto_deploy.torch_deepseek_v4_router.default,
            args=(
                hidden_states,
                input_ids,
                router_weight,
                router_bias,
                tid2eid,
                top_k,
                route_scale,
                is_hash_layer,
            ),
        )
        selected_experts = graph.call_function(operator.getitem, args=(router_node, 0))
        routing_weights = graph.call_function(operator.getitem, args=(router_node, 1))

    node.target = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default
    node.args = (
        hidden_states,
        selected_experts,
        routing_weights,
        _node_arg(node, "routed_w1_weight"),
        _node_arg(node, "routed_w2_weight"),
        _node_arg(node, "routed_w3_weight"),
        _node_arg(node, "shared_w1_weight"),
        _node_arg(node, "shared_w2_weight"),
        _node_arg(node, "shared_w3_weight"),
        _node_arg(node, "swiglu_limit"),
        _node_arg(node, "layer_type", "moe"),
    )
    node.kwargs = {}


@TransformRegistry.register("lower_deepseek_v4_moe")
class DeepSeekV4MoELowering(BaseTransform):
    """Lower DeepSeek V4 MoE source ops.

    The default behavior is intentionally conservative: matching a canonical
    node raises a clear error until a real MXFP4 + FineGrained FP8 production
    lowering is available. Unit tests may opt into dense reference lowering.
    """

    config: DeepSeekV4MoELoweringConfig

    @classmethod
    def get_config_class(cls) -> type[TransformConfig]:
        return DeepSeekV4MoELoweringConfig

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> tuple[GraphModule, TransformInfo]:
        del cm, factory, shared_config

        num_matches = 0
        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe):
                continue
            num_matches += 1
            if not self.config.allow_reference_lowering:
                raise _unsupported_production_lowering_error()
            _lower_to_reference_graph(gm, node)

        if num_matches:
            gm.graph.lint()
            gm.recompile()

        return gm, TransformInfo(
            skipped=num_matches == 0,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
