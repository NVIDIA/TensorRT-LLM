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

This is the default AutoDeploy sharding pipeline. ``is_shardingIR_enabled``
auto-detects whether the exported FX graph carries the IR's ``all_reduce``
markers; ``apply_sharding_hints`` consumes them and short-circuits the
heuristic-detection fallback in ``sharding.py``. Both stay registered in
``default.yaml`` so the long-tail of modeling files not yet ported to IR
keeps working transparently. See the design documents in
``sharding_architecture_documents/`` for background.
"""

import operator
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field, field_validator
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import GraphModule, Node

from ..._compat import AllReduceStrategy

try:
    from ...custom_ops.distributed.trtllm_dist import is_trtllm_op_available
except ImportError:

    def is_trtllm_op_available():
        return False


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
    invalidate_weight_node_cache,
    is_any_lin_op,
    is_op,
    set_op_args,
    shape,
)
from ...utils.pipeline_cache_hooks import mark_pipeline_cache_hook
from ...utils.quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    modelopt_fp4_scale_to_cutlass_fp4_scale,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

# ---------------------------------------------------------------------------
# Shared sharding primitives.
#
# These weight-splitting / hook / dist-op helpers are the canonical sharding
# building blocks and live here, in the now-default IR pipeline. The legacy
# heuristic pipeline (``sharding.py``) imports them from this module -- the
# dependency points legacy -> IR, never the reverse, so ``sharding_ir.py`` has
# no import dependency on ``sharding.py``.
# ---------------------------------------------------------------------------


class SplitDimension(IntEnum):
    """Enum for tensor split dimensions in sharding."""

    # NOTE: The names COLUMN/ROW reflect the hugging face
    # base_tp_plan sharding notation, but since we assume Y = W @ X^T,
    # when splitting weight matrix W^T across columns, the actual split
    # is over dimension 0
    COLUMN = 0
    ROW = 1


def validate_allreduce_strategy(v):
    """Convert string names like 'AUTO' to AllReduceStrategy enum.

    This is a shared validator for allreduce_strategy fields across all config classes.

    Args:
        v: Value to validate - can be AllReduceStrategy enum, string name, or integer value

    Returns:
        AllReduceStrategy enum value

    Raises:
        ValueError: If the input is an invalid strategy string
    """
    if isinstance(v, AllReduceStrategy):
        return v
    if isinstance(v, str):
        # Try to get enum by name
        try:
            return AllReduceStrategy[v]
        except KeyError:
            raise ValueError(
                f"Invalid allreduce strategy: {v}. "
                f"Valid options: {', '.join(s.name for s in AllReduceStrategy)}"
            )
    if isinstance(v, int):
        return AllReduceStrategy(v)
    return v  # Let Pydantic handle other types


_LOGGED_DIST_BACKEND_CHOICES: set[tuple[str, str]] = set()


def _log_dist_backend_choice(configured_backend: str, resolved_backend: str):
    key = (configured_backend, resolved_backend)
    if key in _LOGGED_DIST_BACKEND_CHOICES:
        return
    _LOGGED_DIST_BACKEND_CHOICES.add(key)
    ad_logger.info(
        f"AutoDeploy selected distributed backend: {resolved_backend} "
        f"(configured: {configured_backend})"
    )


def _get_dist_ops(backend: str):
    """Get the (all_gather, all_reduce) op pair for *backend*.

    backend may be 'auto', 'trtllm', or 'torch'. 'auto' resolves to TRT-LLM
    ops when available, else PyTorch distributed ops. Strategies (allgather
    SYMM_MEM, allreduce NCCL/etc.) are passed as op arguments at the call
    site, not selected here.
    """
    if hasattr(backend, "value"):
        backend = backend.value
    configured_backend = str(backend)

    if backend == "trtllm":
        _log_dist_backend_choice(configured_backend, "trtllm")
        return (
            torch.ops.auto_deploy.trtllm_dist_all_gather.default,
            torch.ops.auto_deploy.trtllm_dist_all_reduce.default,
        )
    if backend == "torch":
        _log_dist_backend_choice(configured_backend, "torch")
        return (
            torch.ops.auto_deploy.torch_dist_all_gather.default,
            torch.ops.auto_deploy.torch_dist_all_reduce.default,
        )
    if is_trtllm_op_available():
        _log_dist_backend_choice(configured_backend, "trtllm")
        return (
            torch.ops.auto_deploy.trtllm_dist_all_gather.default,
            torch.ops.auto_deploy.trtllm_dist_all_reduce.default,
        )
    _log_dist_backend_choice(configured_backend, "torch")
    return (
        torch.ops.auto_deploy.torch_dist_all_gather.default,
        torch.ops.auto_deploy.torch_dist_all_reduce.default,
    )


def _load_hook(
    state_dict,
    prefix,
    *args,
    f_split: Callable[[torch.Tensor, int], torch.Tensor],
    param_key: str,
    param_shape: torch.Size,
):
    # TODO: we need to support loading either a sharded or unsharded checkpoint.
    # Otherwise, basic workflows like
    # model.load_state_dict(model.state_dict()) will fail.
    # This is quite a hacky solution. A better solution would be to store extra_state in
    # the state_dict to identify whether the state_dict is sharded or not.
    key = prefix + param_key
    if key not in state_dict:
        return
    p_to_load = state_dict[key]
    did_split = param_shape != p_to_load.shape
    p_to_load = p_to_load if not did_split else f_split(p_to_load)
    state_dict[key] = p_to_load


def _split_tensor_for_tp(
    t: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
    min_local_shape: int = 1,
) -> torch.Tensor:
    """Split a tensor for tensor-parallelism, respecting min_local_shape.

    When world_size exceeds the maximum number of even splits (e.g. GQA with
    num_kv_heads < world_size), multiple ranks share the same shard.

    TODO: support num_units % world_size != 0 via GCD-based partial replication.
    When num_heads doesn't divide by world_size (e.g. 28 Q heads at tp_size=8),
    use effective_splits = gcd(num_heads, world_size) to split at head
    boundaries and replicate each shard across world_size // effective_splits
    ranks. To compensate the duplication in all_reduce, scale rowwise weights
    by 1 / replication_factor during sharding (baked into the weight tensor,
    no changes to all_reduce or the graph needed).
    """
    max_split_size = t.shape[dim] // min_local_shape
    if max_split_size == 0:
        # dim is smaller than a single local-shard unit. With min_local_shape=1
        # (the default) this never happens; it only arises once a per-op floor is
        # imposed (e.g. NVFP4 nodes set min_local_shape=32 so each shard stays a
        # whole 16-element FP4 scale block). Raise an actionable error instead of
        # the bare ``ZeroDivisionError`` the ``world_size % max_split_size`` below
        # would otherwise throw.
        raise ValueError(
            f"Cannot tensor-parallel split dim {dim} of size {t.shape[dim]}: it is "
            f"smaller than min_local_shape ({min_local_shape}). For NVFP4 weights this "
            f"means the dimension is below the 16-element FP4 block / MIN_LOCAL_SHAPE "
            f"floor and cannot be sharded across {world_size} ranks; widen the dim or "
            f"exclude this node from sharding."
        )
    if world_size > max_split_size:
        assert world_size % max_split_size == 0, (
            f"world_size ({world_size}) must be divisible by max_split_size ({max_split_size}). "
            f"GQA with num_kv_heads not dividing world_size is not supported."
        )
        num_groups = world_size // max_split_size
        ad_logger.debug(
            f"World size {world_size} is greater than the max split size {max_split_size}. "
            f"Splitting tensor to {num_groups} chunks"
        )
        return torch.tensor_split(t, max_split_size, dim=dim)[rank // num_groups]

    assert max_split_size % world_size == 0, (
        f"Number of units ({max_split_size}, dim {dim} size {t.shape[dim]} / "
        f"min_local_shape {min_local_shape}) must be divisible by world_size "
        f"({world_size}). For attention heads, use a world_size that divides "
        f"num_heads evenly (e.g. for {max_split_size} heads, try world_size in "
        f"{[d for d in range(2, max_split_size + 1) if max_split_size % d == 0]})."
    )
    return torch.tensor_split(t, world_size, dim=dim)[rank]


def _shard_fp4_weight_scale(
    weight_scale,
    original_uint8_weight_shape,
    dim,
    rank,
    world_size,
    min_local_shape=1,
    fused_weight_dims=None,
):
    # Convert original uint8 shape to element shape (FP4 packs 2 elements per byte)
    weight_shape_elements = list(original_uint8_weight_shape)
    weight_shape_elements[-1] *= 2
    modelopt_weight_scale = cutlass_fp4_scale_to_modelopt_fp4_scale(
        weight_scale, tuple(weight_shape_elements)
    )
    if fused_weight_dims is not None:
        # Fused weights (e.g. Mamba in_proj) are split per-component then sharded.
        # The scale must follow the same per-component splitting to stay aligned.
        sharded_scale = torch.cat(
            [
                _split_tensor_for_tp(chunk, dim, rank, world_size, min_local_shape)
                for chunk in torch.split(modelopt_weight_scale, list(fused_weight_dims), dim=dim)
            ],
            dim=dim,
        )
    else:
        sharded_scale = _split_tensor_for_tp(
            modelopt_weight_scale, dim, rank, world_size, min_local_shape
        )
    return modelopt_fp4_scale_to_cutlass_fp4_scale(sharded_scale)


def shard_weight_tensor(
    gm: GraphModule,
    weight_tensor: torch.Tensor,
    param_key: str,
    dim: int,
    rank: int,
    world_size: int,
    min_local_shape: int = 1,
    fused_weight_dims: Optional[list] = None,
    requires_grad: bool = False,
    custom_shard_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Size]:
    """Shard a weight tensor across ranks and register load hook.

    Args:
        gm: GraphModule containing the weight
        weight_tensor: The weight tensor to shard
        param_key: Parameter key for registering load hook
        dim: Dimension to shard along
        rank: Current rank
        world_size: Total number of ranks
        min_local_shape: Minimum local shape constraint (for GQA)
        fused_weight_dims: List of dimensions for fused weights
        custom_shard_fn: Optional custom function to shard the tensor
        requires_grad: Whether the parameter should require gradients

    Returns:
        Tuple of (sharded_tensor, sharded_shape)
    """

    # Handle fused weights
    if fused_weight_dims is not None:

        def f_split(
            t: torch.Tensor,
            fused_dims: list = fused_weight_dims,
            d: int = dim,
        ) -> torch.Tensor:
            return torch.cat(
                [
                    _split_tensor_for_tp(w, dim, rank, world_size, min_local_shape)
                    for w in torch.split(t, fused_dims, dim=d)
                ],
                dim=d,
            )

    else:
        f_split = partial(
            _split_tensor_for_tp,
            dim=dim,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
        )

    sharded_weight = f_split(weight_tensor)
    sharded_shape = sharded_weight.shape

    # Update the parameter in the module
    modname, _, param_name = param_key.rpartition(".")
    submod = gm.get_submodule(modname)

    # Register load hook on the owning submodule (not the top-level gm).
    # This ensures the hook runs *after* any parent-level hooks that transform
    # the state_dict (e.g., unfusing fused MoE checkpoint weights into
    # individual expert keys). With the hook on gm, it would run before
    # unfusing and fail to find the individual expert keys.
    hook = partial(
        _load_hook,
        f_split=f_split,
        param_key=param_name,
        param_shape=sharded_shape,
    )
    submod._register_load_state_dict_pre_hook(
        mark_pipeline_cache_hook(
            hook,
            {
                "type": "shard_tp",
                "param_key": param_name,
                "param_shape": list(sharded_shape),
                "dim": dim,
                "rank": rank,
                "world_size": world_size,
                "min_local_shape": min_local_shape,
                "fused_weight_dims": list(fused_weight_dims) if fused_weight_dims else None,
            },
        )
    )
    param_new = nn.Parameter(sharded_weight.detach().clone(), requires_grad=requires_grad)
    setattr(submod, param_name, param_new)

    return sharded_weight, sharded_shape


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
    pipeline_cache_spec: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a sharded scale buffer and its corresponding load hook."""
    buf_name = sn.node_key.rsplit(".", 1)[-1]
    sn.submod.register_buffer(buf_name, sharded_scale)
    hook = partial(
        _load_hook,
        f_split=f_split,
        param_key=sn.node_key,
        param_shape=sharded_scale.shape,
    )
    if pipeline_cache_spec is not None:
        hook = mark_pipeline_cache_hook(hook, pipeline_cache_spec)
    gm._register_load_state_dict_pre_hook(hook)


def _fp8_block_scale_pipeline_cache_spec(
    sn: WeightNode,
    sharded_scale: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    return {
        "type": "shard_fp8_block_scale",
        "param_key": sn.node_key,
        "param_shape": list(sharded_scale.shape),
        "dim": int(dim),
        "rank": rank,
        "world_size": world_size,
    }


def _fp4_weight_scale_pipeline_cache_spec(
    sn: WeightNode,
    sharded_scale: torch.Tensor,
    weight_shape: torch.Size,
    dim: int,
    rank: int,
    world_size: int,
    min_local_shape: int,
    fused_weight_dims: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    return {
        "type": "shard_fp4_weight_scale",
        "param_key": sn.node_key,
        "param_shape": list(sharded_scale.shape),
        "original_uint8_weight_shape": list(weight_shape),
        "dim": int(dim),
        "rank": rank,
        "world_size": world_size,
        "min_local_shape": min_local_shape,
        "fused_weight_dims": list(fused_weight_dims) if fused_weight_dims else None,
    }


def _rowwise_bias_load_hook(state_dict, prefix, *args, rank, param_key):
    """Always-apply load hook for the rank0-only row-parallel bias: zero on rank != 0.

    Unlike the shape-gated ``_load_hook`` (which only transforms when the loaded
    shape differs from the sharded shape), the row-parallel bias keeps its full
    shape -- only its *value* changes. Idempotent (zeroing zeros / keeping rank 0).
    Takes plain ``rank``/``param_key`` (not a closure) so the pipeline cache can
    serialize and rebuild it via the importable-hook path.
    """
    key = prefix + param_key
    if key in state_dict and rank != 0:
        state_dict[key] = torch.zeros_like(state_dict[key])


def _replicate_rowwise_bias(bn: WeightNode, rank: int) -> None:
    """Keep a row-parallel linear's bias on rank 0 only; zero it on the others.

    The row-parallel output is summed across ranks by the trailing ``all_reduce``,
    so a full bias present on every rank would be added ``world_size`` times. Zeroing
    it on rank != 0 makes the all_reduce contribute the bias exactly once. The
    parameter shape is unchanged (unlike the column-parallel bias, which is split),
    so a dedicated always-apply load hook is used.
    """
    new_bias = bn.tensor if rank == 0 else torch.zeros_like(bn.tensor)
    pname = bn.node_key.rsplit(".", 1)[-1]
    bn.submod._register_load_state_dict_pre_hook(
        partial(_rowwise_bias_load_hook, rank=rank, param_key=pname)
    )
    setattr(bn.submod, pname, torch.nn.Parameter(new_bias.detach().clone(), requires_grad=False))


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


def _auto_deploy_ops(*names: str) -> Tuple[OpOverloadPacket, ...]:
    return tuple(
        op for name in names if (op := getattr(torch.ops.auto_deploy, name, None)) is not None
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

    # Minimum per-rank shard size (in elements along the split dim) this op
    # requires, independent of any model/layer/config hint. The TP split is
    # rounded down to a multiple of this value (see ``_split_tensor_for_tp``).
    # NVFP4 linears override this to 32 because the NVFP4 GEMM requires the
    # local ``n`` dimension to be divisible by 32 -- a hard dtype constraint,
    # not a tunable. The effective floor is ``max(MIN_LOCAL_SHAPE, hint)`` so
    # larger constraints (e.g. GQA head_dim) still win.
    MIN_LOCAL_SHAPE: int = 1

    def __init__(self, node: Node):
        self.node = node

    @classmethod
    def register(cls, *op_targets):
        """Class decorator that registers a ShardableNode subclass for the given op targets."""

        def decorator(subcls):
            for target in op_targets:
                if target is None:
                    continue
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
        # Honor the op's dtype floor (e.g. 32 for NVFP4) on top of any hint so the
        # per-rank shard stays GEMM-valid regardless of model/layer config. The
        # NVFP4 GEMM constrains the *output* (column) dim to a multiple of 32; the
        # input (row) dim is FP4-packed (2 values/byte) so a column floor must NOT
        # be applied there (it would over-constrain the packed dim and reject
        # otherwise-valid shards).
        col_floor = self.MIN_LOCAL_SHAPE if split_dim == SplitDimension.COLUMN else 1
        min_shape = max(tp_min_local_shape if tp_min_local_shape else 1, col_floor)

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
            else:
                # Row-parallel: the trailing all_reduce would sum a full bias
                # world_size times. Keep it on rank 0 only so it is added once.
                _replicate_rowwise_bias(bn, dc.tp_rank)

        ad_logger.debug(f"  sharded linear tp_mode={tp_mode}")
        return 1

    def _shard_scales(self, gm, dc, weight_nodes, dim, min_shape=1, fused=None):
        """Override in quantization subclasses to shard per-block scale buffers."""
        pass


@ShardableNode.register(
    torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear,
    torch.ops.auto_deploy.trtllm_finegrained_fp8_linear,
    torch.ops.auto_deploy.trtllm_fp8_deepgemm,
)
class FineGrainedFP8LinearShardableNode(LinearShardableNode):
    """FineGrained FP8 linear: shards per-block ``weight_scale_inv`` buffers."""

    def _shard_scales(self, gm, dc, weight_nodes, dim, min_shape=1, fused=None):
        for sn in weight_nodes.scales:
            f_split = partial(
                _split_fp8_block_scale, dim=dim, rank=dc.tp_rank, world_size=dc.tp_size
            )
            sharded = f_split(sn.tensor)
            _shard_scale_and_hook(
                gm,
                sn,
                sharded,
                f_split,
                _fp8_block_scale_pipeline_cache_spec(sn, sharded, dim, dc.tp_rank, dc.tp_size),
            )


@ShardableNode.register(
    torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear,
    torch.ops.auto_deploy.torch_quant_nvfp4_linear,
)
class FP4LinearShardableNode(LinearShardableNode):
    """NVFP4 linear: shards cutlass-format ``weight_scale`` buffers."""

    # NVFP4 GEMM requires the local ``n`` dimension divisible by 32; floor the
    # TP split granularity so every rank's shard stays kernel-valid.
    MIN_LOCAL_SHAPE: int = 32

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
            _shard_scale_and_hook(
                gm,
                sn,
                sharded,
                f_split,
                _fp4_weight_scale_pipeline_cache_spec(
                    sn, sharded, weight_shape, dim, dc.tp_rank, dc.tp_size, min_shape, fused
                ),
            )


@ShardableNode.register(torch.ops.auto_deploy.view)
class ViewShardableNode(ShardableNode):
    """View op: replace shape[tp_scaled_dim] with -1 so PyTorch infers the sharded size."""

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
            view_shape[tp_scaled_dim] = -1
            set_op_args(self.node, shape=view_shape)
            ad_logger.debug(f"  updated view shape at dim {tp_scaled_dim} to -1 (inferred)")
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
    torch.ops.auto_deploy.torch_attention,
)
class WeightedParamShardableNode(ShardableNode):
    """Ops whose weight parameters are sharded along dim 0 (head dimension).

    Covers SSM (A, D, dt_bias), GatedDeltaNet (A_log, dt_bias), MLA
    (kv_b_proj), and ``torch_attention`` (per-head free Parameters such as
    GPT-OSS's ``sinks``). All share identical sharding logic: when
    ``enable_sharding`` is ``True``, every discovered weight parameter is
    split along dim 0.

    For ``torch_attention``: q/k/v projection weights belong to the preceding
    ``torch_linear_simple`` nodes and are sharded by ``LinearShardableNode``;
    only direct ``get_attr`` args of the ``torch_attention`` node itself
    (e.g. ``sinks``) are sliced here. Models that pass no head-wise Parameter
    to ``torch_attention`` (qwen3, llama, smollm3, ...) leave
    ``enable_sharding`` at its default ``False`` and this handler no-ops for
    them.
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


@ShardableNode.register(torch.ops.auto_deploy.torch_attention)
class AttentionSinksShardableNode(ShardableNode):
    """``torch_attention`` with per-head ``sinks``: shard sinks along the head dim.

    Attention-sink models (e.g. GPT-OSS) add a learnable per-head sink scalar
    (shape ``[num_heads]``). When the heads are TP-sharded (q/k/v colwise), the
    ``sinks`` arg must follow the same head split, else the op sees full sinks
    against the sharded head count. Standard attention (no ``sinks``) is a no-op.

    Gating is handled by the apply loop: attention-DP skips all non-MoE nodes
    (so attention stays replicated and sinks stays full), and ``shard_layers``
    gates via the node's ``layer_type`` hint (default ``"mha"``).
    """

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        if dc.tp_size <= 1:
            return 0
        count = 0
        for wn in extract_weight_nodes(self.node).weights:
            # Only the per-head ``sinks`` (1-D) follows the head split; never a 2-D weight.
            if wn.tensor.dim() != 1:
                continue
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=0,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
            )
            count += 1
        if count:
            ad_logger.debug("  sharded attention sinks along head dim")
        return 1 if count > 0 else 0

    @classmethod
    def _strip_node_hints(cls, node: Node) -> bool:
        # Leave ``torch_attention`` untouched at strip time: its ``layer_type`` is a
        # benign op default that downstream backend selection reads as-is.
        return False


@ShardableNode.register(*_auto_deploy_ops("torch_rmsnorm_gated", "triton_rmsnorm_gated"))
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
            wdim = self._dim_for_key(wn.node_key)
            # Column floor (e.g. 32 for NVFP4) applies only to the output dim;
            # the row dim is FP4-packed and must not inherit it (see
            # LinearShardableNode.apply).
            min_ls = self.MIN_LOCAL_SHAPE if wdim == SplitDimension.COLUMN else 1
            shard_weight_tensor(
                gm=gm,
                weight_tensor=wn.tensor,
                param_key=wn.node_key,
                dim=wdim,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                min_local_shape=min_ls,
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
            sharded = f_split(sn.tensor)
            _shard_scale_and_hook(
                gm,
                sn,
                sharded,
                f_split,
                _fp8_block_scale_pipeline_cache_spec(sn, sharded, dim, dc.tp_rank, dc.tp_size),
            )


@ShardableNode.register(torch.ops.auto_deploy.torch_nvfp4_swiglu_mlp)
class FP4SwiGLUShardableNode(SwiGLUShardableNode):
    """NVFP4 SwiGLU: shards cutlass-format ``weight_scale`` buffers."""

    # NVFP4 GEMM requires the local ``n`` dimension divisible by 32 (see
    # FP4LinearShardableNode); apply the same floor to the fused SwiGLU split.
    MIN_LOCAL_SHAPE: int = 32

    def _shard_scales(self, gm, dc, weight_nodes):
        if not weight_nodes.weights:
            return

        # Each NVFP4 weight_scale must be de-swizzled with ITS OWN weight's uint8
        # shape. The fused SwiGLU carries gate/up/down weights whose shapes differ
        # (down_proj's in/out are transposed vs gate/up), so using weights[0] for
        # every scale corrupts the down_proj scale -> garbage MLP output. Pair each
        # scale with its weight by module prefix (strip the trailing ``.weight`` /
        # ``.weight_scale`` leaf).
        def _module_prefix(node_key: str) -> str:
            return node_key.rsplit(".", 1)[0]

        shape_by_prefix = {
            _module_prefix(wn.node_key): wn.tensor.shape for wn in weight_nodes.weights
        }
        fallback_shape = weight_nodes.weights[0].tensor.shape
        for sn in weight_nodes.scales:
            weight_shape = shape_by_prefix.get(_module_prefix(sn.node_key), fallback_shape)
            dim = self._dim_for_key(sn.node_key)
            min_ls = self.MIN_LOCAL_SHAPE if dim == SplitDimension.COLUMN else 1
            f_split = partial(
                _shard_fp4_weight_scale,
                original_uint8_weight_shape=weight_shape,
                dim=dim,
                rank=dc.tp_rank,
                world_size=dc.tp_size,
                min_local_shape=min_ls,
                fused_weight_dims=None,
            )
            sharded = f_split(sn.tensor)
            _shard_scale_and_hook(
                gm,
                sn,
                sharded,
                f_split,
                _fp4_weight_scale_pipeline_cache_spec(
                    sn, sharded, weight_shape, dim, dc.tp_rank, dc.tp_size, min_ls
                ),
            )


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

        # Localize global expert indices to per-rank-local indices on the all-reduce
        # path (attention-DP off): each rank computes only its expert slice and the
        # partials are summed by all_reduce. The all-to-all path (attention-DP on)
        # keeps GLOBAL expert IDs -- dispatch/combine handles routing.
        if not enable_alltoall:
            self._localize_expert_indices(
                gm, selected_experts, routing_weights, experts_per_rank, ep_rank, ep_size
            )

        # Always record the MoE grid + workspace inputs on the op, mirroring legacy
        # ``_insert_sharded_moe`` (which sets these unconditionally). The gate on the
        # parallelism mode is *localization* (above), not the mapping metadata -- the
        # latter is consumed by multiple backends/paths regardless of all-to-all:
        #   * all-to-all: ``mapping_config`` + ``max_num_tokens`` + ``batch_info_host``
        #     drive dispatch/combine and workspace sizing.
        #   * all-reduce + TRTLLM-Gen internal routing: ``mapping_config`` lets the
        #     fused kernel recover the GLOBAL expert count and per-rank expert offset;
        #     without it the op falls back to ``num_experts = local_num_experts`` while
        #     ``router_logits`` keeps the global width -> "routing_logits has incorrect
        #     shape" (or silent misrouting).
        # Args a given path does not consume (e.g. ``max_num_tokens`` on all-reduce)
        # are harmless.
        batch_info_host_nodes = gm.graph.find_nodes(op="placeholder", target="batch_info_host")
        batch_info_host_node = batch_info_host_nodes[0] if batch_info_host_nodes else None
        extra_kwargs = (
            {"batch_info_host": batch_info_host_node} if batch_info_host_node is not None else {}
        )
        set_op_args(
            self.node,
            mapping_config=dc.serialize(),
            max_num_tokens=max_num_tokens,
            **extra_kwargs,
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


@ShardableNode.register(torch.ops.auto_deploy.torch_moe_dense_mlp)
class DenseMLPMoEShardableNode(ShardableNode):
    """EP sharding for ``torch_moe_dense_mlp`` (stacked-tensor dense MoE).

    The op signature (see ``custom_ops/fused_moe/torch_moe.py``) is::

        torch_moe_dense_mlp(
            hidden_states,    # [B, S, H]
            routing_weights,  # [B*S, E]
            gate_up_w,        # [E, H, 2I]
            gate_up_b,        # [E, 2I]
            down_w,           # [E, I, H]
            down_b,           # [E, H]
            alpha,            # float
            limit,            # float
        ) -> [B, S, H]

    The op is additive over experts (``next_states.sum(dim=0)`` at the end), so
    EP sharding is just "give each rank a slice of experts and ``all_reduce`` at
    the end". We slice the four stacked expert tensors at dim 0 plus the
    routing_weights at dim 1 (matching ``num_experts``, which the op reads from
    ``routing_weights.shape[1]``), then insert an ``all_reduce`` after the op
    so partial per-rank sums combine into the full expert sum.

    Slicing is graph-level (``aten.slice``), mirroring
    :class:`StackedMoEShardableNode` for ``triton_mxfp4_moe``. Parameters stay
    full-size on every rank; only the runtime forward uses the rank's slice.
    Memory-wise not optimal, but the math is bit-correct for the equivalence
    test and matches the existing stacked-MoE precedent. A follow-up could
    replace this with true per-rank param slicing via load hooks.
    """

    _IDX_ROUTING_WEIGHTS = 1
    _IDX_GATE_UP_W = 2
    _IDX_GATE_UP_B = 3
    _IDX_DOWN_W = 4
    _IDX_DOWN_B = 5

    def apply(self, gm: GraphModule, dc: DistConfig, max_num_tokens: int = 0) -> int:
        ep_size = dc.moe_ep_size
        ep_rank = dc.moe_ep_rank

        if ep_size <= 1:
            return 0

        expert_shape = shape(self.node.args[self._IDX_GATE_UP_W])
        assert expert_shape is not None, (
            f"Cannot determine num_experts: gate_up_w arg has no shape metadata "
            f"(node: {self.node.name})"
        )
        num_experts = expert_shape[0]
        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )
        per = num_experts // ep_size
        lo = per * ep_rank
        hi = num_experts if ep_rank == ep_size - 1 else lo + per

        # Graph-level slice of every expert-stacked arg along the expert dim.
        args = list(self.node.args)
        for idx in (
            self._IDX_GATE_UP_W,
            self._IDX_GATE_UP_B,
            self._IDX_DOWN_W,
            self._IDX_DOWN_B,
        ):
            with gm.graph.inserting_after(args[idx]):
                args[idx] = gm.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    args=(args[idx], 0, lo, hi, 1),
                )
        # routing_weights is sliced along the expert column (dim 1) so the op's
        # internal ``num_experts = routing_weights.shape[1]`` matches the sliced
        # weight stacks.
        with gm.graph.inserting_before(self.node):
            args[self._IDX_ROUTING_WEIGHTS] = gm.graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(args[self._IDX_ROUTING_WEIGHTS], 1, lo, hi, 1),
            )
        self.node.args = tuple(args)

        # Sum the per-rank partial (over local experts) into the full expert sum.
        _, all_reduce_op = _get_dist_ops("auto")
        with gm.graph.inserting_after(self.node):
            red = gm.graph.call_function(
                all_reduce_op,
                args=(self.node, dc.allreduce_strategy),
            )
            self.node.replace_all_uses_with(red)
            red.replace_input_with(red, self.node)

        ad_logger.debug(
            f"  sharded torch_moe_dense_mlp: {num_experts} experts, "
            f"ep={ep_size}, rank slice [{lo}:{hi}]"
        )
        return 1


# =============================================================================
# IR sharding config
# =============================================================================


class IRShardingConfig(TransformConfig):
    """Minimal configuration for the hint-driven IR sharding transform.

    Carries only the fields the IR pipeline reads. ``ShardingTransformConfig``
    in ``sharding.py`` is the parallel config used by the heuristic-detection
    fallback for modeling files not yet ported to IR.
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
    simple_shard_filter: Optional[str] = Field(
        default=None,
        description="Comma-separated weight-name keywords (e.g. 'lm_head'). Matching linears are "
        "gather-sharded (column split + all_gather) regardless of shard_layers -- used for the "
        "lm_head vocab projection, which the hint-driven sharder would otherwise replicate.",
    )
    enable_attention_dp: bool = Field(default=False)
    dist_mapping: dict[str, int] = Field(default_factory=dict)
    dist_config: DistConfig = Field(default_factory=DistConfig)

    @field_validator("allreduce_strategy", mode="before")
    @classmethod
    def _validate_allreduce_strategy(cls, v):
        return validate_allreduce_strategy(v)

    def _init_dist_config(self, rank: int, world_size: int):
        """Initialize ``self.dist_config`` from ``dist_mapping`` (test-only fallback).

        Production path builds ``DistConfig`` in ``LlmArgs.init_dist_config``
        and passes it through ``SharedConfig.dist_config``.  This fallback is
        only entered when ``shared_config.dist_config is None`` (tests that
        construct ``InferenceOptimizer`` without a ``dist_config`` kwarg).
        ``rank`` and ``world_size`` come from ``shared_config``.
        """
        self.dist_config = DistConfig.from_sharding_params(
            rank=rank,
            world_size=world_size,
            dist_mapping=self.dist_mapping,
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
        num_updates += _simple_shard_node(gm, node, dc)
    return num_updates


def _simple_shard_node(gm: GraphModule, node: Node, dc: DistConfig) -> int:
    """Column-split one linear's weight/bias/scale, then all_gather (the "gather" mode).

    Used as the simple-shard fallback for every linear (``simple_shard_only``) and,
    via ``simple_shard_filter``, to gather-shard specific linears like ``lm_head``
    (huge vocab projection) that the hint-driven sharder would otherwise replicate.
    """
    weight_nodes = extract_weight_nodes(node)
    if not weight_nodes.weights:
        return 0
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
    sn = ShardableNode.from_node(node)
    if isinstance(sn, LinearShardableNode):
        sn._shard_scales(gm, dc, weight_nodes, dim=SplitDimension.COLUMN, min_shape=1, fused=None)
    # torch_dist_all_gather is the demollm backend op; signature is
    # (tensor, dim=0, sizes=None) — plain torch.distributed all_gather,
    # no strategy or symm_mem support (use the trtllm backend for those).
    with gm.graph.inserting_after(node):
        gather_node = gm.graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_gather.default,
            args=(node, -1),
        )
        node.replace_all_uses_with(gather_node)
        gather_node.replace_input_with(gather_node, node)
    return 1


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


def is_shardingIR_enabled(gm: GraphModule) -> bool:
    """Whether the FX graph contains sharding-IR marker nodes.

    The sharding-IR pipeline inserts ``torch.ops.auto_deploy.all_reduce`` nodes
    in the modeling source (per ad-sharding-ir-port skill rule A3) to mark
    rowwise-projection / MoE-merge points. Legacy ``detect_sharding`` never
    emits this op (it inserts ``dist.all_reduce`` later), so the presence of
    any such node is a sufficient signal that the modeling file was authored
    against the sharding IR.

    This is the marker the default-sharding dispatcher uses to decide between
    the IR pipeline (``apply_sharding_hints``) and the legacy pipeline
    (``detect_sharding`` + ``sharding_transform_executor``).
    """
    target = torch.ops.auto_deploy.all_reduce
    for node in gm.graph.nodes:
        if is_op(node, target):
            return True
    return False


@TransformRegistry.register("apply_sharding_hints")
class ApplyShardingHints(BaseTransform):
    """Deterministic, node-local sharding transform driven by hint kwargs.

    Iterates graph nodes and applies sharding based on explicit hint arguments
    (tp_mode, tp_scaled_dim, tp_scale_sizes, etc.) together with the runtime
    DistConfig.  No cross-node propagation, no topology inference.

    When the FX graph contains no sharding-IR markers (no
    ``torch.ops.auto_deploy.all_reduce`` node), this transform is a no-op and
    leaves the graph for the legacy ``detect_sharding`` pipeline. Otherwise it
    sets ``gm.meta["sharding_ir_applied"] = True`` after applying, which
    ``Sharding`` / ``ShardingTransformExecutor`` read to skip themselves.
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
            # Alias the shared DistConfig (already populated with allreduce_strategy
            # from YAML by LlmArgs.init_dist_config) so any later mutations stay
            # visible to downstream fusions.
            self.config.dist_config = shared_config.dist_config
        else:
            self.config._init_dist_config(shared_config.local_rank, shared_config.world_size)

        dc = self.config.dist_config
        _log_sharding_prelude(dc)

        if shared_config.world_size < 2:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Auto-detect dispatcher: if the FX graph contains no sharding-IR markers,
        # this modeling file was not authored against the sharding IR. Skip and
        # let the legacy detect_sharding pipeline handle it.
        if not is_shardingIR_enabled(gm):
            ad_logger.info(
                "apply_sharding_hints: no sharding-IR markers in graph "
                "(no torch.ops.auto_deploy.all_reduce node); deferring to legacy "
                "detect_sharding pipeline."
            )
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
            ssf = self.config.simple_shard_filter
            simple_shard_filter = [k.strip() for k in ssf.split(",") if k.strip()] if ssf else None
            num_skipped = 0
            all_dead_nodes = []

            with WeightBiasInfoCache():
                for node in list(gm.graph.nodes):
                    shardable_node = ShardableNode.from_node(node)
                    if shardable_node is None:
                        continue
                    if dc.enable_attention_dp and not isinstance(
                        shardable_node, (MoEShardableNode, StackedMoEShardableNode)
                    ):
                        continue
                    # Gather-shard simple_shard_filter matches (e.g. lm_head) regardless of
                    # shard_layers: column split + all_gather of the huge vocab projection.
                    if simple_shard_filter and is_any_lin_op(node):
                        wnodes = extract_weight_nodes(node)
                        key = wnodes.weights[0].node_key if wnodes.weights else ""
                        if any(kw in key for kw in simple_shard_filter):
                            num_updates += _simple_shard_node(gm, node, dc)
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

        # Signal to the legacy detect_sharding pipeline that IR has handled this
        # graph. Sharding (detect_sharding) and ShardingTransformExecutor read
        # this flag and short-circuit themselves; without it they would re-shard
        # the same nodes via the heuristic path.
        gm.meta["sharding_ir_applied"] = True

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_updates,
            is_clean=(num_updates == 0),
            has_valid_shapes=True,
        )
