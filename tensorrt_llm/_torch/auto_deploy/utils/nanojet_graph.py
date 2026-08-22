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

"""Shared graph queries for the nanojet fusions and the nanojet attention backend."""

import math
from typing import NamedTuple, Optional

import torch
from torch.fx import GraphModule, Node

from .node_utils import (
    extract_op_args,
    get_param_or_buffer,
    is_op,
    unwrap_input_through_passthrough,
)


def get_attr_tensor(gm: GraphModule, node) -> Optional[torch.Tensor]:
    """The tensor behind a ``get_attr`` node, or None if it is not one."""
    if not isinstance(node, Node) or node.op != "get_attr":
        return None
    try:
        return get_param_or_buffer(node.target, gm)
    except KeyError:
        return None


def per_tensor_scale(gm: GraphModule, node) -> Optional[float]:
    """A quantization scale as a plain float, or None if it is not one this can use.

    Accepts both spellings a quantized linear uses — a bare tensor, or the single-element
    list ``torch_fake_quant_fp8_linear`` wraps it in. Rejects anything that is not a usable
    divisor: every caller divides by this, so a zero, negative or non-finite value would
    become an infinity that spreads through the model with nothing failing.
    """
    if isinstance(node, (list, tuple)):
        if len(node) != 1:
            return None
        node = node[0]
    tensor = get_attr_tensor(gm, node)
    if tensor is None or tensor.numel() != 1:
        return None
    scale = float(tensor.reshape(-1)[0])
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def scale_node(node):
    """The node holding a quantization scale, unwrapping the single-element list spelling.

    Separate from :func:`per_tensor_scale` on purpose: one answers "what is the value", the
    other "which node carries it". Returning both from one function as a tuple is what let a
    caller keep indexing it after the value-only version replaced it.
    """
    if isinstance(node, (list, tuple)):
        if len(node) != 1:
            return None
        node = node[0]
    return node if isinstance(node, Node) and node.op == "get_attr" else None


_FP8_LINEAR_NAMES = (
    "trtllm_quant_fp8_linear",
    "torch_quant_fp8_linear",
    "torch_fake_quant_fp8_linear",
)


def fp8_linear_ops():
    """auto_deploy's FP8 linears, resolved on use rather than at import.

    This module is imported during the custom-op package scan, before those ops exist.
    """
    namespace = torch.ops.auto_deploy
    return tuple(
        getattr(namespace, name) for name in _FP8_LINEAR_NAMES if hasattr(namespace, name)
    )


def is_fp8_linear(node: Node) -> bool:
    """Any of auto_deploy's FP8 linears.

    They agree on the leading argument names and on scales being amax/448, which is all the
    nanojet fusions read; which one the graph holds depends on ``fuse_fp8_linear.backend``.
    """
    return any(is_op(node, op) for op in fp8_linear_ops())


def accepts_out_dtype(node: Node) -> bool:
    """Whether this linear takes an out_dtype hint; only the TensorRT LLM one does."""
    return any(argument.name == "out_dtype" for argument in node.target._schema.arguments)


class Fp8Projection(NamedTuple):
    """One matched FP8 projection: the node, its weight and its scales."""

    node: Node
    activation: Node
    weight: torch.Tensor
    input_scale: float
    weight_scale: float
    input_scale_node: Optional[Node]


def match_fp8_projection(gm: GraphModule, node: Node) -> Optional[Fp8Projection]:
    """Resolve ``node``, through any views, to the FP8 linear producing it.

    Only bias-free projections match: a bias would have to be folded into the epilogue, and
    nanojet's kernels do not take one.
    """
    source, _ = unwrap_input_through_passthrough(node)
    if not isinstance(source, Node) or not is_fp8_linear(source) or not source.args:
        return None
    if extract_op_args(source, "bias")[0] is not None:
        return None
    # The FP8 linears name their weight differently; everything else they spell the same.
    weight_node = extract_op_args(source, "weight_fp8")[0]
    if weight_node is None:
        weight_node = extract_op_args(source, "weight_quantized")[0]
    weight = get_attr_tensor(gm, weight_node)
    if weight is None or weight.dtype != torch.float8_e4m3fn:
        return None
    input_scale_arg = extract_op_args(source, "input_scale")[0]
    input_scale = per_tensor_scale(gm, input_scale_arg)
    weight_scale = per_tensor_scale(gm, extract_op_args(source, "weight_scale")[0])
    if input_scale is None or weight_scale is None:
        return None
    return Fp8Projection(
        node=source,
        activation=source.args[0],
        weight=weight,
        input_scale=input_scale,
        weight_scale=weight_scale,
        input_scale_node=scale_node(input_scale_arg),
    )


def set_val_meta(node: Node, source, shape=None, dtype=None) -> None:
    """Record what ``node`` produces, taking shape and dtype from ``source`` by default.

    ``source`` is a node, a meta value or a real tensor. ``tensor_meta`` is a second record
    of the same facts and must not outlive them.
    """
    value = source.meta.get("val") if isinstance(source, Node) else source
    if value is None:
        return
    target_shape = value.shape if shape is None else tuple(shape)
    target_dtype = value.dtype if dtype is None else dtype
    if hasattr(value, "new_empty") and value.device.type == "meta":
        node.meta["val"] = value.new_empty(target_shape, dtype=target_dtype)
    else:
        node.meta["val"] = torch.empty(target_shape, dtype=target_dtype, device="meta")
    node.meta.pop("tensor_meta", None)
