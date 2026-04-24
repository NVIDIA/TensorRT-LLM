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

"""DeepSeek V4 MoE lowering.

The production lowering preserves three distinct pieces of behavior:

* DeepSeek V4 sqrt-softplus/hash routing.
* Packed MXFP4 routed experts with the DeepSeek V4 SwiGLU limit.
* FineGrained FP8 shared expert linears.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.fused_moe.deepseek_v4_moe import (  # noqa: F401
    torch_deepseek_v4_moe,
    torch_deepseek_v4_moe_from_routing,
)
from ...custom_ops.fused_moe.deepseek_v4_router import torch_deepseek_v4_router  # noqa: F401
from ...custom_ops.fused_moe.mxfp4_moe import (  # noqa: F401
    triton_deepseek_v4_mxfp4_moe_from_routing,
)
from ...custom_ops.quantization.torch_quant import (  # noqa: F401
    torch_fake_quant_finegrained_fp8_linear,
)
from ...custom_ops.sharding_ops import all_reduce  # noqa: F401
from ...utils.node_utils import invalidate_weight_node_cache, is_op
from ..interface import BaseTransform, TransformConfig, TransformInfo, TransformRegistry
from .deepseek_v4_mxfp4 import load_deepseek_v4_mxfp4_experts

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
    enable_mxfp4_bridge: bool = Field(
        default=True,
        description=(
            "Lower canonical DeepSeek V4 MoE to DeepSeek routing plus packed MXFP4 routed "
            "experts and FineGrained FP8 shared expert linears."
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
_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<ffn_path>layers\.(?P<layer_idx>\d+)\.ffn)\.experts\."
    r"(?P<expert_idx>\d+)\.(?P<proj>w[123])\.weight$"
)


@dataclass(frozen=True)
class _MoEBridge:
    layer_idx: int
    ffn_path: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    gate_up_blocks_name: str
    gate_up_scales_name: str
    down_blocks_name: str
    down_scales_name: str


@dataclass(frozen=True)
class _LayerInfo:
    layer_idx: int
    ffn_path: str
    hidden_size: int
    intermediate_size: int
    num_experts: int


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


def _get_attr_name(node: Any) -> str | None:
    if isinstance(node, Node) and node.op == "get_attr":
        return str(node.target)
    return None


def _stack_get_attr_names(stack_node: Any) -> list[str]:
    if not isinstance(stack_node, Node) or stack_node.op != "call_function":
        return []
    if not stack_node.args:
        return []
    values = stack_node.args[0]
    if not isinstance(values, (list, tuple)):
        return []
    names = [_get_attr_name(value) for value in values]
    if any(name is None for name in names):
        return []
    return [name for name in names if name is not None]


def _infer_layer_info(gm: GraphModule, node: Node) -> _LayerInfo:
    w1_names = _stack_get_attr_names(_node_arg(node, "routed_w1_weight"))
    if not w1_names:
        raise DeepSeekV4MoELoweringError(
            "DeepSeek V4 MXFP4 bridge requires routed w1 weights to be an exported "
            "stack of per-expert get_attr nodes."
        )

    match = _EXPERT_WEIGHT_RE.match(w1_names[0])
    if match is None:
        raise DeepSeekV4MoELoweringError(
            f"Could not infer DeepSeek V4 layer path from routed expert weight {w1_names[0]!r}."
        )

    ffn_path = match.group("ffn_path")
    layer_idx = int(match.group("layer_idx"))
    expected_prefix = f"{ffn_path}.experts."
    for expert_idx, name in enumerate(w1_names):
        expected = f"{expected_prefix}{expert_idx}.w1.weight"
        if name != expected:
            raise DeepSeekV4MoELoweringError(
                "DeepSeek V4 MXFP4 bridge requires routed experts in checkpoint order; "
                f"expected {expected!r}, got {name!r}."
            )

    first_weight = gm.get_parameter(w1_names[0])
    if first_weight.ndim != 2:
        raise DeepSeekV4MoELoweringError(
            f"Expected routed expert weight {w1_names[0]!r} to be rank 2, got "
            f"shape {tuple(first_weight.shape)}."
        )

    return _LayerInfo(
        layer_idx=layer_idx,
        ffn_path=ffn_path,
        hidden_size=int(first_weight.shape[1]),
        intermediate_size=int(first_weight.shape[0]),
        num_experts=len(w1_names),
    )


def _set_parameter(
    gm: GraphModule,
    full_name: str,
    value: torch.Tensor,
    *,
    requires_grad: bool = False,
) -> None:
    module_name, _, attr_name = full_name.rpartition(".")
    module = gm.get_submodule(module_name) if module_name else gm
    setattr(module, attr_name, nn.Parameter(value, requires_grad=requires_grad))


def _register_buffer(gm: GraphModule, full_name: str, value: torch.Tensor) -> None:
    module_name, _, attr_name = full_name.rpartition(".")
    module = gm.get_submodule(module_name) if module_name else gm
    if attr_name in module._buffers:
        module._buffers[attr_name] = value
    else:
        module.register_buffer(attr_name, value)


def _create_get_attr_with_meta(gm: GraphModule, graph: torch.fx.Graph, full_name: str) -> Node:
    node = graph.create_node("get_attr", full_name)
    module_name, _, attr_name = full_name.rpartition(".")
    module = gm.get_submodule(module_name) if module_name else gm
    value = getattr(module, attr_name)
    if isinstance(value, torch.Tensor):
        node.meta["val"] = value.detach()
    return node


def _fp8_scale_shape(weight_shape: torch.Size) -> tuple[int, int]:
    return (math.ceil(int(weight_shape[0]) / 128), math.ceil(int(weight_shape[1]) / 128))


def _ensure_shared_fp8_param_and_scale(gm: GraphModule, weight_name: str) -> str:
    weight = gm.get_parameter(weight_name)
    if weight.dtype != torch.float8_e4m3fn:
        _set_parameter(
            gm,
            weight_name,
            torch.empty(tuple(weight.shape), dtype=torch.float8_e4m3fn, device=weight.device),
        )
    scale_name = weight_name.rsplit(".", 1)[0] + ".weight_scale_inv"
    if scale_name not in dict(gm.named_buffers()):
        _register_buffer(
            gm,
            scale_name,
            torch.ones(_fp8_scale_shape(weight.shape), dtype=torch.float32, device=weight.device),
        )
    return scale_name


def _register_bridge_params(gm: GraphModule, info: _LayerInfo) -> _MoEBridge:
    h_blocks = info.hidden_size // 32
    i_blocks = info.intermediate_size // 32
    if info.hidden_size % 32 != 0 or info.intermediate_size % 32 != 0:
        raise DeepSeekV4MoELoweringError(
            "DeepSeek V4 MXFP4 bridge requires hidden and intermediate sizes to be "
            f"divisible by 32, got H={info.hidden_size}, I={info.intermediate_size}."
        )

    ffn_path = info.ffn_path
    gate_up_blocks_name = f"{ffn_path}.mxfp4_gate_up_blocks"
    gate_up_scales_name = f"{ffn_path}.mxfp4_gate_up_scales"
    down_blocks_name = f"{ffn_path}.mxfp4_down_blocks"
    down_scales_name = f"{ffn_path}.mxfp4_down_scales"

    _set_parameter(
        gm,
        gate_up_blocks_name,
        torch.zeros(
            (info.num_experts, 2 * info.intermediate_size, h_blocks, 16), dtype=torch.uint8
        ),
    )
    _set_parameter(
        gm,
        gate_up_scales_name,
        torch.zeros((info.num_experts, 2 * info.intermediate_size, h_blocks), dtype=torch.uint8),
    )
    _set_parameter(
        gm,
        down_blocks_name,
        torch.zeros((info.num_experts, info.hidden_size, i_blocks, 16), dtype=torch.uint8),
    )
    _set_parameter(
        gm,
        down_scales_name,
        torch.zeros((info.num_experts, info.hidden_size, i_blocks), dtype=torch.uint8),
    )

    return _MoEBridge(
        layer_idx=info.layer_idx,
        ffn_path=ffn_path,
        hidden_size=info.hidden_size,
        intermediate_size=info.intermediate_size,
        num_experts=info.num_experts,
        gate_up_blocks_name=gate_up_blocks_name,
        gate_up_scales_name=gate_up_scales_name,
        down_blocks_name=down_blocks_name,
        down_scales_name=down_scales_name,
    )


def _register_mxfp4_load_hook(gm: GraphModule, bridges: list[_MoEBridge]) -> None:
    if not bridges:
        return
    existing = getattr(gm, "_deepseek_v4_mxfp4_bridges", [])
    setattr(gm, "_deepseek_v4_mxfp4_bridges", [*existing, *bridges])
    if getattr(gm, "_deepseek_v4_mxfp4_load_hook_registered", False):
        return

    def _load_hook(state_dict: dict[str, torch.Tensor], prefix: str, *args) -> None:
        del args
        prefix = prefix or ""
        all_bridges = getattr(gm, "_deepseek_v4_mxfp4_bridges", [])
        load_state = state_dict
        if prefix:
            load_state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }

        for bridge in all_bridges:
            has_any_expert_key = any(
                f"layers.{bridge.layer_idx}.ffn.experts.{expert_idx}.w1.weight" in load_state
                for expert_idx in range(bridge.num_experts)
            )
            if has_any_expert_key:
                layout = load_deepseek_v4_mxfp4_experts(
                    load_state,
                    layer_idx=bridge.layer_idx,
                    hidden_size=bridge.hidden_size,
                    intermediate_size=bridge.intermediate_size,
                    num_experts=bridge.num_experts,
                )
                state_dict[prefix + bridge.gate_up_blocks_name] = layout.gate_up_blocks
                state_dict[prefix + bridge.gate_up_scales_name] = layout.gate_up_scales
                state_dict[prefix + bridge.down_blocks_name] = layout.down_blocks
                state_dict[prefix + bridge.down_scales_name] = layout.down_scales

            for expert_idx in range(bridge.num_experts):
                for proj in ("w1", "w2", "w3"):
                    for suffix in ("weight", "scale"):
                        state_dict.pop(
                            prefix
                            + f"layers.{bridge.layer_idx}.ffn.experts.{expert_idx}."
                            + f"{proj}.{suffix}",
                            None,
                        )

            for proj in ("w1", "w2", "w3"):
                scale_alias = prefix + f"{bridge.ffn_path}.shared_experts.{proj}.scale"
                scale_buffer = prefix + f"{bridge.ffn_path}.shared_experts.{proj}.weight_scale_inv"
                if scale_alias in state_dict:
                    state_dict[scale_buffer] = state_dict.pop(scale_alias)

    gm._register_load_state_dict_pre_hook(_load_hook)
    setattr(gm, "_deepseek_v4_mxfp4_load_hook_registered", True)


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


def _call_shared_fp8_linear(
    graph: torch.fx.Graph,
    hidden_states: Node,
    weight: Any,
    scale: Node,
    tp_mode: str,
) -> Node:
    return graph.call_function(
        torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default,
        args=(
            hidden_states,
            weight,
            None,
            [],
            [scale],
            [],
            [],
            tp_mode,
            None,
            1,
            "moe",
        ),
    )


def _lower_to_mxfp4_bridge(gm: GraphModule, node: Node) -> _MoEBridge:
    graph = gm.graph
    info = _infer_layer_info(gm, node)
    bridge = _register_bridge_params(gm, info)

    hidden_states = _node_arg(node, "hidden_states")
    input_ids = _node_arg(node, "input_ids")
    router_weight = _node_arg(node, "router_weight")
    router_bias = _node_arg(node, "router_bias")
    tid2eid = _node_arg(node, "tid2eid")
    top_k = _node_arg(node, "top_k")
    route_scale = _node_arg(node, "route_scale")
    swiglu_limit = _node_arg(node, "swiglu_limit")
    is_hash_layer = _node_arg(node, "is_hash_layer")
    layer_type = _node_arg(node, "layer_type", "moe")

    shared_w1 = _node_arg(node, "shared_w1_weight")
    shared_w2 = _node_arg(node, "shared_w2_weight")
    shared_w3 = _node_arg(node, "shared_w3_weight")
    shared_w1_name = _get_attr_name(shared_w1)
    shared_w2_name = _get_attr_name(shared_w2)
    shared_w3_name = _get_attr_name(shared_w3)
    if shared_w1_name is None or shared_w2_name is None or shared_w3_name is None:
        raise DeepSeekV4MoELoweringError(
            "DeepSeek V4 MXFP4 bridge requires shared expert weights to be get_attr nodes."
        )
    shared_w1_scale_name = _ensure_shared_fp8_param_and_scale(gm, shared_w1_name)
    shared_w2_scale_name = _ensure_shared_fp8_param_and_scale(gm, shared_w2_name)
    shared_w3_scale_name = _ensure_shared_fp8_param_and_scale(gm, shared_w3_name)

    gate_up_bias_name = f"{info.ffn_path}.mxfp4_gate_up_bias"
    down_bias_name = f"{info.ffn_path}.mxfp4_down_bias"
    _set_parameter(
        gm,
        gate_up_bias_name,
        torch.zeros((info.num_experts, 2 * info.intermediate_size), dtype=torch.float32),
    )
    _set_parameter(
        gm,
        down_bias_name,
        torch.zeros((info.num_experts, info.hidden_size), dtype=torch.float32),
    )

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

        gate_up_blocks = _create_get_attr_with_meta(gm, graph, bridge.gate_up_blocks_name)
        gate_up_bias = _create_get_attr_with_meta(gm, graph, gate_up_bias_name)
        gate_up_scales = _create_get_attr_with_meta(gm, graph, bridge.gate_up_scales_name)
        down_blocks = _create_get_attr_with_meta(gm, graph, bridge.down_blocks_name)
        down_bias = _create_get_attr_with_meta(gm, graph, down_bias_name)
        down_scales = _create_get_attr_with_meta(gm, graph, bridge.down_scales_name)

        routed = graph.call_function(
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default,
            args=(
                hidden_states,
                selected_experts,
                routing_weights,
                gate_up_blocks,
                gate_up_bias,
                gate_up_scales,
                1.0,
                float(swiglu_limit),
                down_blocks,
                down_bias,
                down_scales,
                layer_type,
            ),
        )

        shared_w1_scale = _create_get_attr_with_meta(gm, graph, shared_w1_scale_name)
        shared_w2_scale = _create_get_attr_with_meta(gm, graph, shared_w2_scale_name)
        shared_w3_scale = _create_get_attr_with_meta(gm, graph, shared_w3_scale_name)
        shared_gate = _call_shared_fp8_linear(
            graph, hidden_states, shared_w1, shared_w1_scale, "colwise"
        )
        shared_up = _call_shared_fp8_linear(
            graph, hidden_states, shared_w3, shared_w3_scale, "colwise"
        )
        shared_act = graph.call_function(torch.ops.aten.silu.default, args=(shared_gate,))
        shared_hidden = graph.call_function(torch.ops.aten.mul.Tensor, args=(shared_act, shared_up))
        shared = _call_shared_fp8_linear(
            graph, shared_hidden, shared_w2, shared_w2_scale, "rowwise"
        )
        shared = graph.call_function(
            torch.ops.auto_deploy.all_reduce.default,
            args=(shared,),
            kwargs={"layer_type": "moe"},
        )
        output = graph.call_function(torch.ops.aten.add.Tensor, args=(routed, shared))

    node.replace_all_uses_with(output)
    graph.erase_node(node)
    return bridge


@TransformRegistry.register("lower_deepseek_v4_moe")
class DeepSeekV4MoELowering(BaseTransform):
    """Lower DeepSeek V4 MoE source ops.

    Unit tests may opt into dense reference lowering; production uses packed
    MXFP4 routed experts plus FineGrained FP8 shared experts.
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
        bridges: list[_MoEBridge] = []
        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe):
                continue
            num_matches += 1
            if self.config.allow_reference_lowering:
                _lower_to_reference_graph(gm, node)
                continue
            if not self.config.enable_mxfp4_bridge:
                raise _unsupported_production_lowering_error()
            bridges.append(_lower_to_mxfp4_bridge(gm, node))

        if num_matches:
            _register_mxfp4_load_hook(gm, bridges)
            invalidate_weight_node_cache(gm)
            gm.graph.lint()
            gm.recompile()

        return gm, TransformInfo(
            skipped=num_matches == 0,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
