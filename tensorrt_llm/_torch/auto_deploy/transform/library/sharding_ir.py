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

"""Hint-driven IR sharding transform for AutoDeploy.

This module implements the ``apply_sharding_hints`` and ``strip_sharding_hints``
transforms, which apply deterministic, node-local sharding based on explicit
hint kwargs on custom ops and a runtime ``DistConfig``.

This is the replacement for the legacy heuristic-based sharding pipeline in
``sharding.py``.  See the design documents in ``sharding_architecture_documents/``
for background.
"""

import operator
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from pydantic import Field, field_validator
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import GraphModule, Node

from ..._compat import AllReduceStrategy
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import del_attr_by_name, eliminate_dead_code
from ...utils.dist_config import DistConfig
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    WeightBiasInfoCache,
    WeightNode,
    _get_op_schema,
    extract_op_args,
    extract_weight_nodes,
    get_param_or_buffer,
    invalidate_weight_node_cache,
    is_any_lin_op,
    is_op,
    set_op_args,
    shape,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

# NOTE: sharding.py module will be deprecated in the future. The following
# imports will move into sharding_ir.py when legacy sharding is removed.
from .sharding import (
    SplitDimension,
    _get_dist_ops,
    _load_hook,
    _shard_fp4_weight_scale,
    shard_weight_tensor,
    validate_allreduce_strategy,
)


def _split_fp8_block_scale(
    scale: torch.Tensor, dim: int, rank: int, world_size: int
) -> torch.Tensor:
    """Split a finegrained FP8 per-block scale tensor along *dim*.

    Handles the edge case where ``scale.shape[dim] < world_size`` (e.g., a
    2-row scale shared across 8 GPUs) by grouping ranks that share a row.
    """
    scale_dim = scale.shape[dim]
    if scale_dim >= world_size:
        return torch.tensor_split(scale, world_size, dim=dim)[rank]
    group = rank // (world_size // scale_dim)
    return torch.tensor_split(scale, scale_dim, dim=dim)[group]


def _shard_scale_and_hook(
    gm: GraphModule,
    sn: WeightNode,
    sharded_scale: torch.Tensor,
    f_split,
) -> None:
    """Register a sharded scale buffer and its corresponding load hook."""
    buf_name = sn.node_key.rsplit(".", 1)[-1]
    sn.submod.register_buffer(buf_name, sharded_scale)
    gm._register_load_state_dict_pre_hook(
        partial(_load_hook, f_split=f_split, param_key=sn.node_key, param_shape=sharded_scale.shape)
    )


def _assert_divisible(value: int, divisor: int, msg: str) -> None:
    assert value % divisor == 0, f"{msg}: {value} is not divisible by {divisor}"


def _contiguous_partition(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    _assert_divisible(total, world_size, "contiguous partition")
    part = total // world_size
    return rank * part, (rank + 1) * part


def _split_flattened_groups(
    tensor: torch.Tensor,
    *,
    num_groups: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Split a row-major ``[num_groups * rows_per_group, ...]`` tensor by group."""
    _assert_divisible(num_groups, world_size, "group count")
    _assert_divisible(int(tensor.shape[0]), num_groups, "flattened grouped rows")
    group_lo, group_hi = _contiguous_partition(num_groups, world_size, rank)
    rows_per_group = int(tensor.shape[0]) // num_groups
    grouped = tensor.reshape(num_groups, rows_per_group, *tensor.shape[1:])
    return grouped[group_lo:group_hi].reshape(-1, *tensor.shape[1:]).contiguous()


def _slice_group_range(tensor: torch.Tensor, group_lo: int, group_hi: int) -> torch.Tensor:
    return tensor[group_lo:group_hi].contiguous()


def _is_tp_local_group_view(node: Node, group_dim: int) -> bool:
    if not isinstance(node, Node) or not is_op(node, torch.ops.auto_deploy.view):
        return False
    [tp_scaled_dim, view_shape] = extract_op_args(node, "tp_scaled_dim", "shape")
    if tp_scaled_dim < 0:
        tp_scaled_dim = len(view_shape) + tp_scaled_dim
    return tp_scaled_dim == group_dim


def _view_base_and_shape(node: Node) -> Tuple[Optional[Node], Optional[List[Any]]]:
    if not isinstance(node, Node):
        return None, None
    if node.op == "call_method" and node.target in {"view", "reshape"}:
        shape_args = node.args[1:]
    elif is_op(node, torch.ops.auto_deploy.view):
        shape_args = (node.args[1],)
    elif is_op(node, (torch.ops.aten.view, torch.ops.aten.reshape)):
        shape_args = node.args[1:]
    else:
        return None, None

    if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)):
        view_shape = list(shape_args[0])
    else:
        view_shape = list(shape_args)
    return node.args[0], view_shape


def _set_view_shape(node: Node, view_shape: List[Any]) -> None:
    if is_op(node, torch.ops.auto_deploy.view):
        set_op_args(node, shape=view_shape)
    elif node.op == "call_method" and node.target in {"view", "reshape"}:
        node.args = (node.args[0], *view_shape)
    else:
        node.args = (node.args[0], view_shape)


def _is_aten_view_or_reshape(node: Node) -> bool:
    return isinstance(node, Node) and (
        (node.op == "call_method" and node.target in {"view", "reshape"})
        or is_op(node, (torch.ops.aten.view, torch.ops.aten.reshape))
    )


def _scale_view_shape_dim(node: Node, dim: int, dc: DistConfig, reason: str) -> bool:
    _, view_shape = _view_base_and_shape(node)
    if view_shape is None:
        return False
    if dim < 0:
        dim = len(view_shape) + dim
    if dim >= len(view_shape) or not isinstance(view_shape[dim], int):
        return False

    scaled_size = view_shape[dim]
    if scaled_size == -1:
        return False
    if -1 in view_shape:
        _assert_divisible(scaled_size, dc.tp_size, reason)
        view_shape[dim] = scaled_size // dc.tp_size
    else:
        view_shape[dim] = -1
    _set_view_shape(node, view_shape)
    ad_logger.debug(f"  recovered DeepSeek V4 view shape at dim {dim}: {view_shape}")
    return True


def _node_depends_on(node: Node, target: Node) -> bool:
    def _child_nodes(value):
        if isinstance(value, Node):
            return [value]
        if isinstance(value, (list, tuple)):
            nodes = []
            for item in value:
                nodes.extend(_child_nodes(item))
            return nodes
        if isinstance(value, dict):
            nodes = []
            for item in value.values():
                nodes.extend(_child_nodes(item))
            return nodes
        return []

    pending = [node]
    seen = set()
    while pending:
        current = pending.pop()
        if current is target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(_child_nodes(current.args))
        pending.extend(_child_nodes(current.kwargs))
    return False


def _is_recovered_tp_local_group_view(
    node: Node,
    group_dim: int,
    *,
    num_groups: int,
    dc: DistConfig,
) -> bool:
    if _is_tp_local_group_view(node, group_dim):
        return True
    if not _is_aten_view_or_reshape(node):
        return False

    _, view_shape = _view_base_and_shape(node)
    if view_shape is None:
        return False
    if group_dim < 0:
        group_dim = len(view_shape) + group_dim
    if group_dim >= len(view_shape):
        return False
    return view_shape[group_dim] == num_groups // dc.tp_size


def _get_attr_tensor(gm: GraphModule, node: Node, arg_name: str) -> Tuple[str, torch.Tensor]:
    assert isinstance(node, Node) and node.op == "get_attr" and isinstance(node.target, str), (
        f"Expected {arg_name} to be a get_attr node, got {node!r}"
    )
    return node.target, get_param_or_buffer(node.target, gm)


def _weight_node_from_attr(gm: GraphModule, node: Node, tensor: torch.Tensor) -> WeightNode:
    node_key = str(node.target)
    return WeightNode(
        node=node,
        node_key=node_key,
        tensor=tensor,
        submod=gm.get_submodule(node_key.rpartition(".")[0]),
    )


_SHARDING_HINT_NAMES = frozenset(
    {
        "tp_mode",
        "output_sizes",
        "tp_min_local_shape",
        "layer_type",
        "enable_sharding",
        "tp_scaled_dim",
    }
)

# =============================================================================
# ShardableNode abstract base class
# =============================================================================


class ShardableNode(ABC):
    """Base class for graph nodes that carry sharding hints.

    Each specialized subclass encapsulates the sharding logic for one category of
    custom op.  Subclasses self-register via the ``@ShardableNode.register``
    decorator, and ``from_node`` dispatches an FX node to the correct subclass.
    """

    _REGISTRY: Dict[OpOverload, Type["ShardableNode"]] = {}

    def __init__(self, node: Node):
        self.node = node

    @classmethod
    def register(cls, *op_targets):
        """Class decorator that registers a ShardableNode subclass for the given op targets."""

        def decorator(subcls):
            for target in op_targets:
                if isinstance(target, OpOverloadPacket):
                    for overload_name in target.overloads():
                        cls._REGISTRY[getattr(target, overload_name)] = subcls
                else:
                    cls._REGISTRY[target] = subcls
            return subcls

        return decorator

    @classmethod
    def _resolve(cls, target) -> Optional[Type["ShardableNode"]]:
        """Look up the registered subclass for an op target.

        Handles both ``OpOverload`` (e.g. ``torch_moe.default``) and
        ``OpOverloadPacket`` (e.g. ``torch_moe``) since FX nodes may
        store either form as their target.
        """
        subcls = cls._REGISTRY.get(target)
        if subcls is None and isinstance(target, OpOverloadPacket):
            subcls = cls._REGISTRY.get(getattr(target, "default", None))
        return subcls

    @staticmethod
    def from_node(node: Node) -> Optional["ShardableNode"]:
        """Return a ShardableNode for *node*, or ``None`` if the op is not enable_sharding."""
        if not isinstance(node, Node) or node.op != "call_function":
            return None
        subcls = ShardableNode._resolve(node.target)
        if subcls is None:
            return None
        return subcls(node)

    @abstractmethod
    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        """Apply sharding to this node.  Returns 1 if modified, 0 otherwise."""
        ...

    @classmethod
    def strip_hints(cls, node: Node) -> bool:
        """Strip sharding hint args/kwargs from *node*.  Returns ``True`` if modified.

        Dispatches to the registered subclass's ``_strip_node_hints``.  The base
        implementation uses schema introspection to strip trailing hint args and
        reset non-trailing ones to defaults.  Subclasses that represent pure
        placeholder ops (view, split_with_sizes, all_reduce) override this to
        lower the node to a zero-copy aten equivalent instead.
        """
        if node.op != "call_function":
            return False
        subcls = cls._resolve(node.target)
        if subcls is None:
            return False
        return subcls._strip_node_hints(node)

    @classmethod
    def _strip_node_hints(cls, node: Node) -> bool:
        """Default: strip hint args/kwargs by schema introspection.

        Trailing hint args are popped entirely (so downstream transforms see
        the canonical short args tuple).  Non-trailing hints are reset to their
        schema defaults via ``set_op_args``.  Hint kwargs are removed.
        """
        schema = _get_op_schema(node)
        hint_defaults: Dict[str, Any] = {}
        hint_positions: set = set()
        for i, a in enumerate(schema.arguments):
            if a.name in _SHARDING_HINT_NAMES:
                hint_defaults[a.name] = a.default_value if a.has_default_value else None
                hint_positions.add(i)

        has_hint_kwargs = bool(node.kwargs and (_SHARDING_HINT_NAMES & node.kwargs.keys()))
        if not hint_defaults and not has_hint_kwargs:
            return False

        modified = False

        # Pop trailing hint args to keep the args tuple minimal
        args = list(node.args)
        while args and (len(args) - 1) in hint_positions:
            args.pop()
            modified = True
        if modified:
            node.args = tuple(args)

        # Reset any non-trailing hints still in the args to their defaults
        non_trailing = {
            schema.arguments[i].name: hint_defaults[schema.arguments[i].name]
            for i in hint_positions
            if i < len(node.args)
        }
        if non_trailing:
            set_op_args(node, **non_trailing)
            modified = True

        # Strip hint kwargs
        if has_hint_kwargs:
            node.kwargs = {k: v for k, v in node.kwargs.items() if k not in _SHARDING_HINT_NAMES}
            modified = True

        return modified


# =============================================================================
# Specialized ShardableNode subclasses
# =============================================================================


@ShardableNode.register(
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.auto_deploy.torch_fake_quant_fp8_linear,
    torch.ops.auto_deploy.torch_quant_fp8_linear,
    torch.ops.auto_deploy.trtllm_quant_fp8_linear,
    torch.ops.auto_deploy.torch_fake_quant_int4_linear,
    torch.ops.auto_deploy.torch_fake_quant_int4_gptq_linear,
)
class LinearShardableNode(ShardableNode):
    """Linear ops: weight + bias sharding. No quantized scale handling.

    Covers BF16, standard FP8 (per-tensor scales via List args), and INT4.
    Quantization variants with per-block scale buffers use subclasses that
    override ``_shard_scales``.
    """

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [tp_mode, output_sizes, tp_min_local_shape] = extract_op_args(
            self.node, "tp_mode", "output_sizes", "tp_min_local_shape"
        )
        if tp_mode == "none":
            return 0

        split_dim = SplitDimension.COLUMN if tp_mode == "colwise" else SplitDimension.ROW
        fused = tuple(output_sizes) if output_sizes else None
        min_shape = tp_min_local_shape if tp_min_local_shape else 1

        weight_nodes = extract_weight_nodes(self.node)

        for wn in weight_nodes.weights:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=split_dim,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                min_local_shape=min_shape,
                fused_weight_dims=fused,
            )

        self._shard_scales(gm, dc, weight_nodes, split_dim, min_shape, fused)

        for bn in weight_nodes.biases:
            if split_dim == SplitDimension.COLUMN:
                shard_weight_tensor(
                    gm=gm,
                    weight_tensor=bn.tensor,
                    param_key=bn.node_key,
                    dim=SplitDimension.COLUMN,
                    rank=dc.tp_rank,
                    world_size=dc.tp_size,
                    fused_weight_dims=fused,
                )

        ad_logger.debug(f"  sharded linear tp_mode={tp_mode}")
        return 1

    def _shard_scales(self, gm, dc, weight_nodes, dim, min_shape=1, fused=None):
        """Override in quantization subclasses to shard per-block scale buffers."""
        pass


@ShardableNode.register(
    torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear,
    torch.ops.auto_deploy.trtllm_finegrained_fp8_linear,
)
class FineGrainedFP8LinearShardableNode(LinearShardableNode):
    """FineGrained FP8 linear: shards per-block ``weight_scale_inv`` buffers."""

    def _shard_scales(self, gm, dc, weight_nodes, dim, min_shape=1, fused=None):
        for sn in weight_nodes.scales:
            f_split = partial(
                _split_fp8_block_scale, dim=dim, rank=dc.tp_rank, world_size=dc.tp_size
            )
            sharded = f_split(sn.tensor)
            _shard_scale_and_hook(gm, sn, sharded, f_split)


@ShardableNode.register(
    torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
)
class DeepSeekV4GroupedWoAShardableNode(ShardableNode):
    """DeepSeek V4 grouped ``wo_a`` FP8 linear: shard by output group boundaries."""

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        if dc.tp_size <= 1:
            return 0

        [input_node, weight_node, bias_node, weight_scale] = extract_op_args(
            self.node, "input", "weight_quantized", "bias", "weight_scale"
        )
        input_shape = shape(input_node)
        output_shape = shape(self.node)
        if input_shape is not None:
            num_groups = int(input_shape[-2])
            group_dim = len(input_shape) - 2
        elif output_shape is not None:
            num_groups = int(output_shape[-2])
            group_dim = len(output_shape) - 2
        else:
            raise AssertionError(
                f"Cannot determine DeepSeek V4 wo_a group count for node {self.node.name}."
            )
        _assert_divisible(num_groups, dc.tp_size, "DeepSeek V4 wo_a output groups")
        group_lo, group_hi = _contiguous_partition(num_groups, dc.tp_size, dc.tp_rank)

        weight_name, weight_tensor = _get_attr_tensor(gm, weight_node, "weight_quantized")
        split_groups = partial(
            _split_flattened_groups,
            num_groups=num_groups,
            rank=dc.tp_rank,
            world_size=dc.tp_size,
        )
        shard_weight_tensor(
            gm=gm,
            weight_tensor=weight_tensor,
            param_key=weight_name,
            dim=SplitDimension.COLUMN,
            rank=dc.tp_rank,
            world_size=dc.tp_size,
            custom_shard_fn=split_groups,
        )

        if isinstance(bias_node, Node):
            bias_name, bias_tensor = _get_attr_tensor(gm, bias_node, "bias")
            if bias_tensor.ndim == 1:
                bias_split = split_groups
            else:
                bias_split = partial(_slice_group_range, group_lo=group_lo, group_hi=group_hi)
            shard_weight_tensor(
                gm=gm,
                weight_tensor=bias_tensor,
                param_key=bias_name,
                dim=SplitDimension.COLUMN,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                custom_shard_fn=bias_split,
            )

        assert isinstance(weight_scale, (list, tuple)) and len(weight_scale) == 1, (
            "DeepSeek V4 wo_a expects a single weight_scale_inv tensor."
        )
        scale_node = weight_scale[0]
        _, scale_tensor = _get_attr_tensor(gm, scale_node, "weight_scale")
        _assert_divisible(
            int(scale_tensor.shape[0]),
            num_groups,
            "DeepSeek V4 wo_a scale rows",
        )
        scale_split = partial(
            _split_flattened_groups,
            num_groups=num_groups,
            rank=dc.tp_rank,
            world_size=dc.tp_size,
        )
        scale_weight_node = _weight_node_from_attr(gm, scale_node, scale_tensor)
        _shard_scale_and_hook(gm, scale_weight_node, scale_split(scale_tensor), scale_split)

        if isinstance(input_node, Node) and not _is_recovered_tp_local_group_view(
            input_node,
            group_dim,
            num_groups=num_groups,
            dc=dc,
        ):
            with gm.graph.inserting_before(self.node):
                local_input = gm.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    args=(input_node, group_dim, group_lo, group_hi, 1),
                )
            set_op_args(self.node, input=local_input)

        ad_logger.debug(f"  sharded DeepSeek V4 wo_a groups [{group_lo}:{group_hi}] / {num_groups}")
        return 1


@ShardableNode.register(
    torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention,
    torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2,
)
class DeepSeekV4SparseAttentionShardableNode(ShardableNode):
    """DeepSeek V4 sparse attention: shard head-owned parameters, not activations."""

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [enable_sharding, attn_sink, q] = extract_op_args(
            self.node,
            "enable_sharding",
            "attn_sink",
            "q",
        )
        if not enable_sharding or dc.tp_size <= 1:
            return 0

        sink_name, sink_tensor = _get_attr_tensor(gm, attn_sink, "attn_sink")
        full_heads = int(sink_tensor.shape[0])
        _assert_divisible(
            full_heads,
            dc.tp_size,
            "DeepSeek V4 attention heads",
        )
        local_heads = full_heads // dc.tp_size
        q_shape = shape(q)
        if q_shape is not None:
            q_heads = int(q_shape[-2])
            assert q_heads in (local_heads, full_heads), (
                "DeepSeek V4 sparse attention q head count must match either local "
                f"heads ({local_heads}) or pre-shard metadata heads ({full_heads}), got {q_heads}"
            )
        shard_weight_tensor(
            gm=gm,
            weight_tensor=sink_tensor,
            param_key=sink_name,
            dim=0,
            rank=dc.tp_rank,
            world_size=dc.tp_size,
        )

        ad_logger.debug(f"  sharded DeepSeek V4 sparse attention attn_sink ({sink_name})")
        return 1


@ShardableNode.register(
    torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear,
    torch.ops.auto_deploy.torch_quant_nvfp4_linear,
)
class FP4LinearShardableNode(LinearShardableNode):
    """NVFP4 linear: shards cutlass-format ``weight_scale`` buffers."""

    def _shard_scales(self, gm, dc, weight_nodes, dim, min_shape=1, fused=None):
        weight_shape = weight_nodes.weights[0].tensor.shape if weight_nodes.weights else None
        if weight_shape is None:
            return
        for sn in weight_nodes.scales:
            f_split = partial(
                _shard_fp4_weight_scale,
                original_uint8_weight_shape=weight_shape,
                dim=dim,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                min_local_shape=min_shape,
                fused_weight_dims=fused,
            )
            sharded = f_split(sn.tensor)
            _shard_scale_and_hook(gm, sn, sharded, f_split)


@ShardableNode.register(torch.ops.auto_deploy.view)
class ViewShardableNode(ShardableNode):
    """View op: scale one shape dimension so the view matches TP-local tensors."""

    @classmethod
    def _strip_node_hints(cls, node: Node) -> bool:
        """Lower to aten.reshape, keeping only (input, shape)."""
        node.target = torch.ops.aten.reshape.default
        node.args = (node.args[0], node.args[1])
        node.kwargs = {}
        return True

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [tp_scaled_dim, view_shape] = extract_op_args(self.node, "tp_scaled_dim", "shape")
        if tp_scaled_dim == -1:
            return 0

        view_shape = list(view_shape)
        if tp_scaled_dim < 0:
            tp_scaled_dim = len(view_shape) + tp_scaled_dim
        if tp_scaled_dim < len(view_shape) and isinstance(view_shape[tp_scaled_dim], int):
            scaled_size = view_shape[tp_scaled_dim]
            if scaled_size == -1:
                return 0
            if -1 in view_shape:
                _assert_divisible(scaled_size, dc.tp_size, "view shape dimension")
                view_shape[tp_scaled_dim] = scaled_size // dc.tp_size
            else:
                view_shape[tp_scaled_dim] = -1
            set_op_args(self.node, shape=view_shape)
            ad_logger.debug(f"  updated view shape at dim {tp_scaled_dim}: {view_shape}")
            return 1
        return 0


@ShardableNode.register(torch.ops.auto_deploy.split_with_sizes)
class SplitShardableNode(ShardableNode):
    """split_with_sizes op: divide all split sizes by tp_size."""

    @classmethod
    def _strip_node_hints(cls, node: Node) -> bool:
        """Lower to aten.split_with_sizes, keeping only (input, sizes, dim)."""
        dim = node.args[2] if len(node.args) > 2 else -1
        node.target = torch.ops.aten.split_with_sizes.default
        node.args = (node.args[0], node.args[1], dim)
        node.kwargs = {}
        return True

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [enable_sharding, split_sizes] = extract_op_args(
            self.node, "enable_sharding", "split_sizes"
        )
        if not enable_sharding:
            return 0

        split_sizes = list(split_sizes)
        for s in split_sizes:
            assert s % dc.tp_size == 0, (
                f"split_with_sizes: size {s} is not divisible by tp_size={dc.tp_size}. "
                f"Full split_sizes={split_sizes}. Ensure the model dimensions are "
                f"compatible with the tensor parallel degree."
            )
        scaled = [s // dc.tp_size for s in split_sizes]
        set_op_args(self.node, split_sizes=scaled)
        ad_logger.debug(f"  updated split_with_sizes: {split_sizes} -> {scaled}")
        return 1


@ShardableNode.register(torch.ops.auto_deploy.all_reduce)
class AllReduceShardableNode(ShardableNode):
    """all_reduce placeholder: replace with real dist.all_reduce or identity."""

    @classmethod
    def _strip_node_hints(cls, node: Node) -> bool:
        """Remove the all_reduce placeholder entirely (passthrough to input)."""
        node.replace_all_uses_with(node.args[0])
        node.graph.erase_node(node)
        return True

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        if dc.tp_size <= 1:
            return 0

        _, all_reduce_op = _get_dist_ops("auto")
        [x] = extract_op_args(self.node, "x")
        self.node.target = all_reduce_op
        self.node.args = (x, dc.allreduce_strategy)
        self.node.kwargs = {}
        ad_logger.debug(f"  inserted real all_reduce ({all_reduce_op.__name__})")
        return 1


@ShardableNode.register(torch.ops.auto_deploy.torch_causal_conv1d)
class Conv1dShardableNode(ShardableNode):
    """Conv1d op: shard weight/bias with fused dims, update groups."""

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [enable_sharding, output_sizes] = extract_op_args(
            self.node, "enable_sharding", "output_sizes"
        )
        if not enable_sharding:
            return 0

        fused = list(output_sizes) if output_sizes else None
        weight_nodes = extract_weight_nodes(self.node)

        for wn in weight_nodes.weights:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=0,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                fused_weight_dims=fused,
            )

        for bn in weight_nodes.biases:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=bn.tensor,
                param_key=bn.node_key,
                dim=0,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                fused_weight_dims=fused,
            )

        # No quantized conv1d variants exist; scales not handled.
        # If a quantized variant is added, add _shard_scales() like LinearShardableNode.

        [groups] = extract_op_args(self.node, "groups")
        assert groups % dc.tp_size == 0, (
            f"conv1d groups ({groups}) must be divisible by tp_size ({dc.tp_size})"
        )
        set_op_args(self.node, groups=groups // dc.tp_size)
        ad_logger.debug(f"  sharded conv1d, groups {groups} -> {groups // dc.tp_size}")
        return 1


@ShardableNode.register(
    torch.ops.auto_deploy.torch_ssm,
    torch.ops.auto_deploy.torch_gated_delta_rule,
    torch.ops.auto_deploy.torch_mla,
)
class WeightedParamShardableNode(ShardableNode):
    """Ops whose weight parameters are sharded along dim 0 (head dimension).

    Covers SSM (A, D, dt_bias), GatedDeltaNet (A_log, dt_bias), and MLA
    (kv_b_proj).  All share identical sharding logic: when ``enable_sharding``
    is ``True``, every discovered weight parameter is split along dim 0.
    """

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [enable_sharding] = extract_op_args(self.node, "enable_sharding")
        if not enable_sharding:
            return 0

        weight_nodes = extract_weight_nodes(self.node)

        # SSM/GDN/MLA ops have only weight parameters (A, D, dt_bias, kv_b_proj);
        # no biases or quantized scales. Assert this assumption explicitly.
        assert not weight_nodes.biases, (
            f"Unexpected biases on {self.node.target}: {weight_nodes.biases}"
        )
        assert not weight_nodes.scales, (
            f"Unexpected scales on {self.node.target}: {weight_nodes.scales}"
        )

        count = 0
        for wn in weight_nodes.weights:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=0,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
            )
            count += 1

        ad_logger.debug(f"  sharded weighted params ({count} tensors)")
        return 1 if count > 0 else 0


@ShardableNode.register(
    torch.ops.auto_deploy.torch_rmsnorm_gated,
    torch.ops.auto_deploy.triton_rmsnorm_gated,
)
class NormShardableNode(ShardableNode):
    """Gated RMSNorm op: shard weight parameter."""

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        [tp_mode] = extract_op_args(self.node, "tp_mode")
        if tp_mode == "none":
            return 0

        weight_nodes = extract_weight_nodes(self.node)
        count = 0
        for wn in weight_nodes.weights:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=0,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
            )
            count += 1

        ad_logger.debug(f"  sharded norm ({count} tensors)")
        return 1 if count > 0 else 0


@ShardableNode.register(torch.ops.auto_deploy.torch_swiglu_mlp)
class SwiGLUShardableNode(ShardableNode):
    """SwiGLU MLP ops: shard gate/up colwise, down rowwise.

    Handles the intermediate (unfused) SwiGLU representation produced by
    ``match_swiglu_pattern``.  The fused variant (``fused_swiglu_mlp``) is
    created AFTER sharding in ``post_load_fusion`` and needs no handling.

    Quantized variants with per-block scale buffers use subclasses that
    override ``_shard_scales``.
    """

    @staticmethod
    def _dim_for_key(node_key: str) -> int:
        """Determine split dimension from the module path: ``down`` → ROW, else COLUMN."""
        return SplitDimension.ROW if "down" in node_key else SplitDimension.COLUMN

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        weight_nodes = extract_weight_nodes(self.node)
        if not weight_nodes.weights:
            return 0

        for wn in weight_nodes.weights:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=self._dim_for_key(wn.node_key),
                rank=dc.tp_rank,
                world_size=dc.tp_size,
            )

        for bn in weight_nodes.biases:
            if self._dim_for_key(bn.node_key) == SplitDimension.COLUMN:
                shard_weight_tensor(
                    gm=gm,
                    weight_tensor=bn.tensor,
                    param_key=bn.node_key,
                    dim=SplitDimension.COLUMN,
                    rank=dc.tp_rank,
                    world_size=dc.tp_size,
                )

        self._shard_scales(gm, dc, weight_nodes)

        ad_logger.debug(
            f"  sharded SwiGLU MLP ({len(weight_nodes.weights)} weights, "
            f"{len(weight_nodes.scales)} scales)"
        )
        return 1

    def _shard_scales(self, gm, dc, weight_nodes):
        """Override in quantization subclasses to shard per-block scale buffers."""
        pass


@ShardableNode.register(torch.ops.auto_deploy.torch_finegrained_fp8_swiglu_mlp)
class FineGrainedFP8SwiGLUShardableNode(SwiGLUShardableNode):
    """FineGrained FP8 SwiGLU: shards per-block ``weight_scale_inv`` buffers."""

    def _shard_scales(self, gm, dc, weight_nodes):
        for sn in weight_nodes.scales:
            dim = self._dim_for_key(sn.node_key)
            f_split = partial(
                _split_fp8_block_scale, dim=dim, rank=dc.tp_rank, world_size=dc.tp_size
            )
            _shard_scale_and_hook(gm, sn, f_split(sn.tensor), f_split)


@ShardableNode.register(torch.ops.auto_deploy.torch_nvfp4_swiglu_mlp)
class FP4SwiGLUShardableNode(SwiGLUShardableNode):
    """NVFP4 SwiGLU: shards cutlass-format ``weight_scale`` buffers."""

    def _shard_scales(self, gm, dc, weight_nodes):
        weight_shape = weight_nodes.weights[0].tensor.shape if weight_nodes.weights else None
        if weight_shape is None:
            return
        for sn in weight_nodes.scales:
            dim = self._dim_for_key(sn.node_key)
            f_split = partial(
                _shard_fp4_weight_scale,
                original_uint8_weight_shape=weight_shape,
                dim=dim,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                min_local_shape=1,
                fused_weight_dims=None,
            )
            _shard_scale_and_hook(gm, sn, f_split(sn.tensor), f_split)


@ShardableNode.register(
    torch.ops.auto_deploy.torch_moe,
    torch.ops.auto_deploy.torch_quant_fp8_moe,
    torch.ops.auto_deploy.torch_quant_nvfp4_moe,
    torch.ops.auto_deploy.torch_quant_finegrained_fp8_moe,
)
class MoEShardableNode(ShardableNode):
    """List-based MoE ops: EP weight partitioning, expert ID localization, mapping injection.

    Handles ops where args[3:6] are ``List[torch.Tensor]`` (per-expert weight
    lists).  Stacked-tensor MoE variants (``torch_moe_fused``,
    ``torch_moe_dense_mlp``, ``triton_mxfp4_moe``) are NOT registered here --
    see ``StackedMoEShardableNode`` for stacked-tensor variants, and the other two are either
    converted to list-based ``torch_moe`` before sharding or left replicated.
    """

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        ep_size = dc.moe_ep_size
        ep_rank = dc.moe_ep_rank
        tp_size = dc.moe_tp_size
        tp_rank = dc.moe_tp_rank
        enable_alltoall = dc.enable_attention_dp and ep_size > 1

        if ep_size <= 1 and tp_size <= 1:
            return 0

        [selected_experts, routing_weights, w1_weight, w2_weight, w3_weight] = extract_op_args(
            self.node, "selected_experts", "routing_weights", "w1_weight", "w2_weight", "w3_weight"
        )
        num_experts = len(w1_weight)
        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )
        experts_per_rank = num_experts // ep_size

        def get_partition(lst, world_size, rank):
            n = len(lst)
            per_part = n // world_size
            start = rank * per_part
            end_idx = n if (rank == world_size - 1) else start + per_part
            return lst[start:end_idx], lst[:start] + lst[end_idx:]

        w1_sharded, w1_removed = get_partition(w1_weight, ep_size, ep_rank)
        w2_sharded, w2_removed = get_partition(w2_weight, ep_size, ep_rank)
        w3_sharded, w3_removed = get_partition(w3_weight, ep_size, ep_rank)
        nodes_to_remove = w1_removed + w2_removed + w3_removed

        if tp_size > 1:
            for w in w1_sharded + w3_sharded:
                shard_weight_tensor(
                    gm=gm,
                    weight_tensor=gm.get_parameter(w.target),
                    param_key=w.target,
                    dim=SplitDimension.COLUMN,
                    rank=tp_rank,
                    world_size=tp_size,
                )
            for w in w2_sharded:
                shard_weight_tensor(
                    gm=gm,
                    weight_tensor=gm.get_parameter(w.target),
                    param_key=w.target,
                    dim=SplitDimension.ROW,
                    rank=tp_rank,
                    world_size=tp_size,
                )

        set_op_args(self.node, w1_weight=w1_sharded, w2_weight=w2_sharded, w3_weight=w3_sharded)

        # Shard scale lists (quantized MoE ops have per-expert scale lists).
        # Unlike Linear/SwiGLU where scales are single buffer tensors handled by
        # _shard_scales(), MoE scales are List[Tensor] (one per expert) -- the same
        # structure as weights. They must be EP-partitioned identically to weights.
        # We use positional args[6:] because scale arg names vary across quantized
        # op variants (w1_weight_scale, input_scale, etc.).
        args = list(self.node.args)
        for i in range(6, len(args)):
            if isinstance(args[i], (list, tuple)) and len(args[i]) == num_experts:
                sharded, removed = get_partition(list(args[i]), ep_size, ep_rank)
                args[i] = sharded
                nodes_to_remove.extend(removed)
        self.node.args = tuple(args)

        if enable_alltoall:
            # mapping and max_num_tokens are needed downstream for MoE all-to-all dispatcher
            mapping_config = dc.serialize()
            set_op_args(self.node, mapping_config=mapping_config, max_num_tokens=max_num_tokens)
        else:
            # with pure EP/TP parallelism, global expert indices must be localized
            self._localize_expert_indices(
                gm, selected_experts, routing_weights, experts_per_rank, ep_rank, ep_size
            )

        ad_logger.debug(
            f"  sharded MoE: {num_experts} experts, ep={ep_size}, ep_rank={ep_rank}, "
            f"tp={tp_size}, tp_rank={tp_rank}, alltoall={enable_alltoall}, "
            f"local_experts={len(w1_sharded)}, mapping_config_keys="
            f"[ep={dc.moe_ep_size},tp={dc.moe_tp_size},attn_dp={dc.enable_attention_dp}]"
        )
        self._pending_dead_nodes = nodes_to_remove
        return 1

    def _localize_expert_indices(
        self,
        gm: GraphModule,
        selected_experts: Node,
        routing_weights: Node,
        experts_per_rank: int,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        """Remap global expert indices to EP-local indices and mask routing weights.

        Inserts graph nodes that (1) subtract the rank offset from
        selected_experts to get local indices, and (2) zero out routing
        weights for experts not assigned to this rank.
        """
        with gm.graph.inserting_before(self.node):
            lower = experts_per_rank * ep_rank
            selected_experts_local = gm.graph.create_node(
                "call_function", operator.sub, args=(selected_experts, lower), kwargs={}
            )
            div_node = gm.graph.create_node(
                "call_function",
                operator.floordiv,
                args=(selected_experts, experts_per_rank),
                kwargs={},
            )
            comp_op = torch.ge if ep_rank == ep_size - 1 else torch.eq
            rank_mask = gm.graph.create_node(
                "call_function", comp_op, args=(div_node, ep_rank), kwargs={}
            )
            routing_weights_local = gm.graph.create_node(
                "call_function", operator.mul, args=(routing_weights, rank_mask), kwargs={}
            )
        set_op_args(
            self.node,
            selected_experts=selected_experts_local,
            routing_weights=routing_weights_local,
        )


class DeepSeekV4MXFP4FromRoutingShardableNode(ShardableNode):
    """DeepSeek V4 pre-routed MXFP4 MoE: EP-slice experts and localize expert ids."""

    _EXPERT_ARG_NAMES = (
        "gate_up_blocks",
        "gate_up_bias",
        "gate_up_scales",
        "down_blocks",
        "down_bias",
        "down_scales",
    )

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        ep_size = dc.moe_ep_size
        ep_rank = dc.moe_ep_rank

        if ep_size <= 1:
            return 0

        op_args = extract_op_args(
            self.node,
            "selected_experts",
            "routing_weights",
            *self._EXPERT_ARG_NAMES,
        )
        selected_experts = op_args[0]
        routing_weights = op_args[1]
        expert_nodes = dict(zip(self._EXPERT_ARG_NAMES, op_args[2:]))

        gate_up_name, gate_up_tensor = _get_attr_tensor(
            gm, expert_nodes["gate_up_blocks"], "gate_up_blocks"
        )
        num_experts = int(gate_up_tensor.shape[0])
        _assert_divisible(
            num_experts,
            ep_size,
            "DeepSeek V4 routed expert count",
        )
        expert_lo, expert_hi = _contiguous_partition(num_experts, ep_size, ep_rank)

        shard_weight_tensor(
            gm=gm,
            weight_tensor=gate_up_tensor,
            param_key=gate_up_name,
            dim=0,
            rank=ep_rank,
            world_size=ep_size,
        )
        for arg_name in self._EXPERT_ARG_NAMES[1:]:
            param_name, tensor = _get_attr_tensor(gm, expert_nodes[arg_name], arg_name)
            shard_weight_tensor(
                gm=gm,
                weight_tensor=tensor,
                param_key=param_name,
                dim=0,
                rank=ep_rank,
                world_size=ep_size,
            )

        self._localize_expert_indices(
            gm,
            selected_experts,
            routing_weights,
            expert_lo,
            expert_hi,
        )
        self._insert_all_reduce(gm, dc)

        ad_logger.debug(
            f"  sharded DeepSeek V4 MXFP4 from routing: {num_experts} experts, "
            f"ep={ep_size}, rank slice [{expert_lo}:{expert_hi}]"
        )
        return 1

    def _localize_expert_indices(
        self,
        gm: GraphModule,
        selected_experts: Node,
        routing_weights: Node,
        expert_lo: int,
        expert_hi: int,
    ) -> None:
        with gm.graph.inserting_before(self.node):
            local_ids = gm.graph.create_node(
                "call_function",
                operator.sub,
                args=(selected_experts, expert_lo),
                kwargs={},
            )
            ge_lower = gm.graph.call_function(torch.ge, args=(selected_experts, expert_lo))
            lt_upper = gm.graph.call_function(torch.lt, args=(selected_experts, expert_hi))
            rank_mask = gm.graph.call_function(torch.logical_and, args=(ge_lower, lt_upper))
            selected_experts_local = gm.graph.create_node(
                "call_function",
                operator.mul,
                args=(local_ids, rank_mask),
                kwargs={},
            )
            routing_weights_local = gm.graph.create_node(
                "call_function",
                operator.mul,
                args=(routing_weights, rank_mask),
                kwargs={},
            )
        set_op_args(
            self.node,
            selected_experts=selected_experts_local,
            routing_weights=routing_weights_local,
        )

    def _insert_all_reduce(self, gm: GraphModule, dc: DistConfig) -> None:
        _, all_reduce_op = _get_dist_ops("auto")
        with gm.graph.inserting_after(self.node):
            red = gm.graph.call_function(
                all_reduce_op,
                args=(self.node, dc.allreduce_strategy),
            )
            self.node.replace_all_uses_with(red)
            red.replace_input_with(red, self.node)


try:
    ShardableNode.register(torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing)(
        DeepSeekV4MXFP4FromRoutingShardableNode
    )
except AttributeError:
    pass


class StackedMoEShardableNode(ShardableNode):
    """Stacked-tensor MoE EP sharding: slice along the expert dimension and rewrite.

    Unlike :class:`MoEShardableNode` which handles list-based expert weights
    (``List[Tensor]``), this class handles MoE ops where expert weights are
    stacked into 3-D tensors (``Tensor[num_experts, ...]``).  Sharding slices
    along dim 0 to select the local expert partition.

    Currently the only registered variant is ``triton_mxfp4_moe`` (MXFP4
    quantized), but the approach generalises to any stacked-tensor MoE op.
    The op is rewritten to ``triton_mxfp4_moe_ep`` with an explicit
    ``all_reduce`` after the node.
    """

    _IDX_GATE_UP_BLOCKS = 4
    _IDX_GATE_UP_BIAS = 5
    _IDX_GATE_UP_SCALES = 6
    _IDX_DOWN_BLOCKS = 9
    _IDX_DOWN_BIAS = 10
    _IDX_DOWN_SCALES = 11

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        ep_size = dc.moe_ep_size
        ep_rank = dc.moe_ep_rank

        if ep_size <= 1:
            return 0

        expert_shape = shape(self.node.args[self._IDX_GATE_UP_BLOCKS])
        assert expert_shape is not None, (
            f"Cannot determine num_experts: gate_up_blocks arg has no shape metadata "
            f"(node: {self.node.name})"
        )
        num_experts = expert_shape[0]
        base = num_experts // ep_size
        lo = base * ep_rank
        hi = num_experts if ep_rank == ep_size - 1 else base * (ep_rank + 1)

        args = list(self.node.args)
        for idx in (
            self._IDX_GATE_UP_BLOCKS,
            self._IDX_GATE_UP_BIAS,
            self._IDX_GATE_UP_SCALES,
            self._IDX_DOWN_BLOCKS,
            self._IDX_DOWN_BIAS,
            self._IDX_DOWN_SCALES,
        ):
            with gm.graph.inserting_after(args[idx]):
                args[idx] = gm.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    args=(args[idx], 0, lo, hi, 1),
                )

        self.node.target = torch.ops.auto_deploy.triton_mxfp4_moe_ep.default
        self.node.args = tuple(args) + (int(ep_size), int(ep_rank))

        _, all_reduce_op = _get_dist_ops("auto")
        with gm.graph.inserting_after(self.node):
            red = gm.graph.call_function(
                all_reduce_op,
                args=(self.node, dc.allreduce_strategy),
            )
            self.node.replace_all_uses_with(red)
            red.replace_input_with(red, self.node)

        ad_logger.debug(
            f"  sharded MXFP4 MoE: {num_experts} experts, ep={ep_size}, rank slice [{lo}:{hi}]"
        )
        return 1


# MXFP4 MoE ops depend on triton_kernels + TRT-LLM internals that aren't in the
# standalone package, so the op may not be registered at import time. Register
# only when the op is actually available.
try:
    ShardableNode.register(torch.ops.auto_deploy.triton_mxfp4_moe)(StackedMoEShardableNode)
except AttributeError:
    pass


# =============================================================================
# IR sharding config
# =============================================================================


class IRShardingConfig(TransformConfig):
    """Minimal configuration for the hint-driven IR sharding transform.

    This replaces the legacy ``ShardingTransformConfig`` for
    ``ApplyShardingHints``, carrying only the fields that the IR path actually
    reads.  When the legacy sharding path is removed, this is the only sharding
    config class.
    """

    allreduce_strategy: AllReduceStrategy = Field(
        default=AllReduceStrategy.AUTO,
        description="AllReduce strategy for distributed operations.",
    )
    simple_shard_only: bool = Field(default=False)
    shard_layers: Optional[List[str]] = Field(
        default=None,
        description="When set, only shard nodes whose layer_type hint is in this list.",
    )
    enable_attention_dp: bool = Field(default=False)
    dist_mapping: dict[str, int] = Field(default_factory=dict)
    dist_config: DistConfig = Field(default_factory=DistConfig)

    @field_validator("allreduce_strategy", mode="before")
    @classmethod
    def _validate_allreduce_strategy(cls, v):
        return validate_allreduce_strategy(v)

    def _init_dist_config(self, rank: int, world_size: int):
        """Initialize DistConfig from dist_mapping config (fallback path).

        Called when ``shared_config.dist_config`` is None (e.g. test suites
        that construct ``InferenceOptimizer`` without a ``Mapping``).
        ``rank`` and ``world_size`` come from ``shared_config``.
        """
        self.dist_config = DistConfig(
            world_size=world_size,
            rank=rank,
            tp_size=self.dist_mapping.get("tp", world_size),
            moe_tp_size=self.dist_mapping.get("moe_tp", 1),
            moe_ep_size=self.dist_mapping.get("moe_ep", world_size),
            moe_cluster_size=self.dist_mapping.get("moe_cluster", 1),
            enable_attention_dp=self.enable_attention_dp,
            allreduce_strategy=self.allreduce_strategy.name,
        )


# =============================================================================
# Standalone helpers
# =============================================================================


def _log_sharding_prelude(dc: DistConfig) -> None:
    """Log the sharding configuration before apply_sharding_hints runs."""
    skip = " (skipping)" if dc.tp_size <= 1 else ""
    ad_logger.info(
        f"apply_sharding_hints{skip}: tp_size={dc.tp_size}, tp_rank={dc.tp_rank}, "
        f"moe grid: [ep x tp] = [{dc.moe_ep_size} x {dc.moe_tp_size}], "
        f"strategy={dc.allreduce_strategy}"
    )


def _log_sharding_result(
    dc: DistConfig,
    num_updates: int,
    num_skipped: int = 0,
    *,
    shard_layers: Optional[List[str]] = None,
) -> None:
    """Log the sharding result after apply_sharding_hints completes."""
    mode = "attention_dp" if dc.enable_attention_dp else "TP + EP"
    parts = [f"apply_sharding_hints ({mode}): {num_updates} nodes processed"]
    if num_skipped:
        parts.append(f"{num_skipped} skipped (shard_layers={shard_layers})")
    ad_logger.info(", ".join(parts))


def _apply_simple_shard(gm: GraphModule, dc: DistConfig) -> int:
    """Simple shard fallback: column-split every linear weight, bias, and scale, then all_gather.

    Uses the polymorphic ``_shard_scales`` method on the resolved
    ``ShardableNode`` subclass for format-aware scale handling (FP4 block
    alignment, fine-grained FP8 per-block splits).
    """
    num_updates = 0
    for node in list(gm.graph.nodes):
        if not is_any_lin_op(node):
            continue
        weight_nodes = extract_weight_nodes(node)
        if not weight_nodes.weights:
            continue
        for wn in weight_nodes.weights:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=SplitDimension.COLUMN,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
            )
        for bn in weight_nodes.biases:
            shard_weight_tensor(
                gm=gm,
                weight_tensor=bn.tensor,
                param_key=bn.node_key,
                dim=SplitDimension.COLUMN,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
            )
        enable_sharding = ShardableNode.from_node(node)
        if isinstance(enable_sharding, LinearShardableNode):
            enable_sharding._shard_scales(
                gm, dc, weight_nodes, dim=SplitDimension.COLUMN, min_shape=1, fused=None
            )
        with gm.graph.inserting_after(node):
            gather_node = gm.graph.call_function(
                torch.ops.auto_deploy.torch_dist_all_gather.default,
                args=(node, -1),
            )
            node.replace_all_uses_with(gather_node)
            gather_node.replace_input_with(gather_node, node)
        num_updates += 1
    return num_updates


def _recover_deepseek_v4_sparse_q_aten_views(gm: GraphModule, dc: DistConfig) -> int:
    num_updates = 0
    sparse_ops = (
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention,
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2,
    )
    for sparse_node in list(gm.graph.nodes):
        if not is_op(sparse_node, sparse_ops):
            continue
        [enable_sharding, attn_sink, q] = extract_op_args(
            sparse_node,
            "enable_sharding",
            "attn_sink",
            "q",
        )
        if not enable_sharding or not isinstance(attn_sink, Node) or not isinstance(q, Node):
            continue

        _, sink_tensor = _get_attr_tensor(gm, attn_sink, "attn_sink")
        full_heads = int(sink_tensor.shape[0])
        _assert_divisible(full_heads, dc.tp_size, "DeepSeek V4 attention heads")

        for view_node in list(gm.graph.nodes):
            if not _is_aten_view_or_reshape(view_node) or not _node_depends_on(q, view_node):
                continue
            _, view_shape = _view_base_and_shape(view_node)
            if (
                view_shape is not None
                and len(view_shape) == 4
                and view_shape[2] == full_heads
                and _scale_view_shape_dim(
                    view_node,
                    2,
                    dc,
                    "DeepSeek V4 sparse attention q heads",
                )
            ):
                num_updates += 1
                break
    return num_updates


def _recover_deepseek_v4_wo_a_group_aten_views(gm: GraphModule, dc: DistConfig) -> int:
    num_updates = 0
    grouped_op = (
        torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear
    )
    for grouped_node in list(gm.graph.nodes):
        if not is_op(grouped_node, grouped_op):
            continue
        [input_node] = extract_op_args(grouped_node, "input")
        if not _is_aten_view_or_reshape(input_node):
            continue

        _, view_shape = _view_base_and_shape(input_node)
        if view_shape is None or len(view_shape) < 2:
            continue
        group_dim = len(view_shape) - 2
        if view_shape[group_dim] != -1 and _scale_view_shape_dim(
            input_node,
            group_dim,
            dc,
            "DeepSeek V4 wo_a output groups",
        ):
            num_updates += 1
    return num_updates


def _recover_deepseek_v4_aten_view_sharding(gm: GraphModule, dc: DistConfig) -> int:
    """Recover DSV4 view hints when export lowered ``auto_deploy.view`` to aten."""
    if dc.tp_size <= 1:
        return 0
    return _recover_deepseek_v4_sparse_q_aten_views(
        gm,
        dc,
    ) + _recover_deepseek_v4_wo_a_group_aten_views(gm, dc)


# =============================================================================
# Transform classes
# =============================================================================


@TransformRegistry.register("strip_sharding_hints")
class StripShardingHints(BaseTransform):
    """Strip sharding hints and lower placeholder ops to zero-copy aten equivalents.

    Placeholder ops (``auto_deploy.view``, ``split_with_sizes``, ``all_reduce``)
    are replaced with native aten ops to eliminate the ``.clone()`` overhead
    required by PyTorch's custom op framework.  Other enable_sharding ops that have no
    aten equivalent get their hint kwargs stripped so downstream transforms see
    canonical op signatures.
    """

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
        count = 0
        for node in list(gm.graph.nodes):
            if ShardableNode.strip_hints(node):
                count += 1
        if count:
            gm.graph.lint()
            gm.recompile()
        return gm, TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )


@TransformRegistry.register("apply_sharding_hints")
class ApplyShardingHints(BaseTransform):
    """Deterministic, node-local sharding transform driven by hint kwargs.

    Iterates graph nodes and applies sharding based on explicit hint arguments
    (tp_mode, tp_scaled_dim, tp_scale_sizes, etc.) together with the runtime
    DistConfig.  No cross-node propagation, no topology inference.
    """

    config: IRShardingConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return IRShardingConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """
        Apply node-local sharding based on hint kwargs and runtime DistConfig.

        Skips when world_size < 2. Supports shard_layers filtering and simple_shard_only mode.
        """
        invalidate_weight_node_cache(gm)

        if shared_config.dist_config is not None:
            # Intentional alias: single shared DistConfig across all transforms
            # so mutations (e.g., allreduce_strategy) propagate to downstream fusions.
            self.config.dist_config = shared_config.dist_config
        else:
            self.config._init_dist_config(shared_config.local_rank, shared_config.world_size)

        dc = self.config.dist_config
        dc.allreduce_strategy = self.config.allreduce_strategy.name
        _log_sharding_prelude(dc)

        if shared_config.world_size < 2:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        max_num_tokens = cm.info.max_num_tokens if (cm and cm.info) else 0

        num_updates = 0
        if self.config.simple_shard_only:
            num_updates = _apply_simple_shard(gm, dc)
            _log_sharding_result(dc, num_updates)
        else:
            shard_layers = self.config.shard_layers
            num_skipped = 0
            all_dead_nodes = []

            with WeightBiasInfoCache():
                if shard_layers is None or "mla" in shard_layers:
                    num_updates += _recover_deepseek_v4_aten_view_sharding(gm, dc)

                for node in list(gm.graph.nodes):
                    shardable_node = ShardableNode.from_node(node)
                    if shardable_node is None:
                        continue
                    if dc.enable_attention_dp and not isinstance(
                        shardable_node, (MoEShardableNode, StackedMoEShardableNode)
                    ):
                        continue
                    if shard_layers is not None:
                        [lt] = extract_op_args(node, "layer_type")
                        if lt is not None and lt not in shard_layers:
                            num_skipped += 1
                            continue

                    num_updates += shardable_node.apply(gm, dc, max_num_tokens)

                    if hasattr(shardable_node, "_pending_dead_nodes"):
                        all_dead_nodes.extend(shardable_node._pending_dead_nodes)

            if all_dead_nodes:
                eliminate_dead_code(gm)
                for dead_node in all_dead_nodes:
                    try:
                        del_attr_by_name(gm, dead_node.target)
                    except AttributeError:
                        pass

            _log_sharding_result(dc, num_updates, num_skipped, shard_layers=shard_layers)

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_updates,
            is_clean=(num_updates == 0),
            has_valid_shapes=True,
        )
