"""Sharding config definitions for the inference optimizer."""

import math
import operator
import re
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.fx import GraphModule, Node

from ..models.factory import ShardingConfigSource
from ..utils.logger import ad_logger
from .node_utils import (
    bfs,
    extract_param_names_from_lin_node,
    is_linear_op,
    is_op,
    num_users_of_weight_node,
    subgraph,
)
from .quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    modelopt_fp4_scale_to_cutlass_fp4_scale,
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
    ad_logger.debug(f"Sharder LOAD hook is called for '{key}'")
    if key not in state_dict:
        return
    p_to_load = state_dict[key]

    p_to_load = p_to_load if param_shape == p_to_load.shape else f_split(p_to_load)

    state_dict[key] = p_to_load


def _load_hook_remove(
    state_dict: Dict,
    prefix: str,
    *args,
    param_key: str,
):
    key = prefix + param_key
    ad_logger.debug(f"Sharder LOAD hook is called for '{key}'")
    state_dict.pop(key, None)


def _validate_sharded_shapes(
    node: Node, fused_weight_dims: Optional[list] = None, world_size: int = None
) -> None:
    """
    Update the shapes of the view nodes and the split node parameters to account for the TP sharding.
    1. After sharding weights of the linear node using column split
    in attention module (Q, K, V),
    the output Y = X @ W^T shape is [batch, seq, num_heads // TP_size, head_dim].
    Some models hardcode the shape of the output to [batch, seq, num_heads, head_dim]
    instead of implicit [batch, seq, -1, head_dim].
    Detect such cases and update the shape of the view node accordingly.
    2. If the weights are fused (e.g,. QKV, gate_up, SSM, etc.), the follow-up split node parameters
    need to be updated to account for the TP sharding.
    """

    # get the subgraph of this module. Subgraph boundary is the next linear node.
    next_lin_node, depth = bfs(node, is_linear_op, include_root=False)
    nodes_to_validate = subgraph(
        [node],
        [next_lin_node],
        include=lambda n: is_op(n, [torch.ops.aten.view, torch.ops.aten.reshape]),
    )
    for view_node in nodes_to_validate:
        if len(view_node.args) < 2:
            continue
        view_shape = list(view_node.args[1])
        if not isinstance(view_shape, list):
            continue
        if len(view_shape) >= 3 and isinstance(view_shape[2], int) and view_shape[2] != -1:
            args = list(view_node.args)
            view_shape[2] = view_shape[2] // world_size
            args[1] = tuple(view_shape)
            view_node.args = tuple(args)
            ad_logger.debug(f"\nUpdated view node {view_node} arguments to {view_node.args}")

    # if fused_weight_dims is provided, we need to update all split sizes
    if fused_weight_dims is not None:
        assert world_size is not None, "World size is required to update the split node params"
        assert len(node.users) == 1, "Fused linear node should have only one user: a split node"
        # find all split nodes in the region between this linear node and the next
        split_nodes = subgraph(
            [node],
            [next_lin_node],
            include=lambda n: is_op(n, [torch.ops.aten.split, torch.ops.aten.split_with_sizes]),
        )
        for split_node in split_nodes:
            orig_sizes = split_node.args[1]
            new_sizes = [orig_sizes[i] // world_size for i in range(len(orig_sizes))]
            args = list(split_node.args)
            args[1] = new_sizes
            split_node.args = tuple(args)
            ad_logger.debug(f"\nUpdated split node {split_node} arguments to {split_node.args}")


def shard_weight_tensor(
    gm: GraphModule,
    weight_tensor: torch.Tensor,
    param_key: str,
    dim: int,
    rank: int,
    world_size: int,
    min_local_shape: int = 1,
    fused_weight_dims: Optional[list] = None,
    custom_shard_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    requires_grad: bool = False,
    update_param: bool = True,
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
        update_param: Whether to update the parameter in the module

    Returns:
        Tuple of (sharded_tensor, sharded_shape)
    """

    # Use custom shard function if provided
    if custom_shard_fn is not None:
        sharded_weight = custom_shard_fn(weight_tensor)
        sharded_shape = sharded_weight.shape
        # Register load hook with custom function
        gm._register_load_state_dict_pre_hook(
            partial(
                _load_hook,
                f_split=custom_shard_fn,
                param_key=param_key,
                param_shape=sharded_shape,
            )
        )

    else:

        def split_tensor(
            t: torch.Tensor,
            d: int = dim,
            r: int = rank,
            ws: int = world_size,
            min_d_shape: int = min_local_shape,
        ) -> torch.Tensor:
            # The local tensor shape has to be divisible by min_d_shape
            max_split_size = t.shape[d] // min_d_shape
            if ws > max_split_size:
                num_groups = math.ceil(ws / max_split_size)
                ad_logger.debug(
                    f"World size {ws} is greater than the max split size {max_split_size}. "
                    + f"Splitting tensor to {num_groups} chunks"
                )
                return torch.tensor_split(t, max_split_size, dim=d)[r // num_groups]
            return torch.tensor_split(t, ws, dim=d)[r]

        # Handle fused weights
        if fused_weight_dims is not None:
            # Split fused weights, apply TP sharding to each, then concatenate back
            sharded_weight = torch.cat(
                [split_tensor(w) for w in torch.split(weight_tensor, fused_weight_dims, dim=dim)],
                dim=dim,
            )

            # Create a function that applies the same logic for loading
            def split_fused_tensor(
                t: torch.Tensor,
                fused_dims: list = fused_weight_dims,
                d: int = dim,
            ) -> torch.Tensor:
                return torch.cat(
                    [split_tensor(w) for w in torch.split(t, fused_dims, dim=d)],
                    dim=d,
                )

            f_split = split_fused_tensor
        else:
            sharded_weight = split_tensor(weight_tensor)
            f_split = split_tensor

        sharded_shape = sharded_weight.shape

        # Register load hook
        gm._register_load_state_dict_pre_hook(
            partial(
                _load_hook,
                f_split=f_split,
                param_key=param_key,
                param_shape=sharded_shape,
            )
        )

    # Update the parameter in the module
    if update_param:
        modname, _, param_name = param_key.rpartition(".")
        submod = gm.get_submodule(modname)
        param_new = nn.Parameter(sharded_weight.detach().clone(), requires_grad=requires_grad)
        setattr(submod, param_name, param_new)

    return sharded_weight, sharded_shape


def get_all_weights_in_subgraph(
    sources: list[Node],
    sinks: list[Node],
):
    """Get all weight nodes (get_attr nodes) in the subgraph between sources and sinks."""
    weight_nodes = subgraph(
        sources, sinks, include_boundary_nodes=False, include=lambda n: n.op == "get_attr"
    )
    return weight_nodes


def _insert_sharded_mamba(
    gm: GraphModule,
    entry_node: Node,
    dim: int,
    rank: int,
    world_size: int,
    add_dist: bool = False,
    min_local_shape: int = 1,
    weights_to_shard: Optional[list[str]] = None,
    weight_shard_dims: Optional[Dict[str, int]] = None,
    fused_weight_dims: Optional[Dict[str, list]] = None,
    quantization_cb: Optional[
        Callable[[GraphModule, nn.Module, Node, str, torch.Size, int, int, int], None]
    ] = None,
) -> bool:
    """
    To shard Mamba layer, first column-shard the first linear layer: entry_node,
    then shard all remaining weight tensors found in the subgraph defined between
    entry_node and the next successor linear node.
    First, validate if this is indeed a mamba module: within the subgraph,
    there should be an torch_ssm node and conv1d node.

    Args:
        gm: GraphModule
        entry_node: The first linear node of the Mamba layer
        dim: Default shard dimension
        rank: Current rank
        world_size: Total number of ranks
        add_dist: Whether to add distribution op after entry_node
        min_local_shape: Minimum local shape constraint
        weights_to_shard: Optional list of regex patterns to match weight names
        weight_shard_dims: Optional dict mapping weight keys to their shard dimensions
        fused_weight_dims: Optional dict mapping weight keys to their fused dimension lists
        quantization_cb: Optional quantization callback
    """
    # Find next linear node to define subgraph boundary
    try:
        next_lin_node, depth = bfs(entry_node, is_linear_op, include_root=False)
    except RuntimeError:
        ad_logger.warning("Could not find next linear node after entry_node for Mamba sharding")
        return False

    # Get subgraph between entry_node and next linear node
    subgraph_nodes = subgraph([entry_node], [next_lin_node])

    ##############################################################
    ########## validate if this is a valid Mamba module ##########
    ##############################################################
    # has_ssm = any(is_op(n, torch.ops.auto_deploy.mamba.torch_ssm_transform) for n in subgraph_nodes)
    has_ssm = True
    conv1d_nodes = [
        n
        for n in subgraph_nodes
        if is_op(n, [torch.ops.aten.conv1d, torch.ops.auto_deploy.torch_causal_conv1d])
    ]
    if len(conv1d_nodes) != 1 or not has_ssm:
        ad_logger.warning(
            f"Subgraph does not contain exactly one conv1d node and torch_ssm_transform. "
            f"Skipping Mamba sharding. conv1d_nodes={conv1d_nodes}, has_ssm={has_ssm}"
        )
        return False

    ##############################################################
    ########## infer split sizes for in_proj and conv1d ##########
    ##############################################################
    # in_proj and conv1d are most likely fused, followed up by split nodes. Infer split sizes:
    if fused_weight_dims is None:
        split_nodes = [
            n
            for n in subgraph_nodes
            if is_op(n, [torch.ops.aten.split, torch.ops.aten.split_with_sizes])
        ]
        if len(split_nodes) != 2:
            ad_logger.warning(
                f"Subgraph does not contain exactly two split nodes. "
                f"Skipping Mamba sharding. split_nodes={split_nodes}"
            )
            return False
        split_sizes_1 = split_nodes[0].args[1]
        split_sizes_2 = split_nodes[1].args[1]
        if split_sizes_1[1] != sum(split_sizes_2):
            ad_logger.warning(
                f"Split nodes have different sizes. "
                f"Skipping Mamba sharding. split_sizes_1={split_sizes_1}, split_sizes_2={split_sizes_2}"
            )
            return False
        fused_weight_dims = {
            "in_proj": split_sizes_1[0:1] + split_sizes_2 + split_sizes_1[2:],
            "conv1d": split_sizes_2,
        }

    conv1d_node = conv1d_nodes[0]
    # conv1d_node last argument is the number of output channels.
    # This one is also sharded, so we need to update this parameter
    conv_args = list(conv1d_node.args)
    conv_args[-1] = conv1d_node.args[-1] // world_size
    conv1d_node.args = tuple(conv_args)

    # First, shard the entry_node (the first linear layer)
    # Extract entry node's fused_weight_dims by matching weight name against patterns
    entry_fused_dims = None
    if fused_weight_dims:
        entry_weight_key, _ = extract_param_names_from_lin_node(entry_node)
        for pattern, dims in fused_weight_dims.items():
            if re.search(pattern, entry_weight_key):
                entry_fused_dims = dims
                break

    _insert_sharded_matmul(
        gm=gm,
        node=entry_node,
        dim=dim,
        rank=rank,
        world_size=world_size,
        add_dist=add_dist,
        min_local_shape=min_local_shape,
        fused_weight_dims=entry_fused_dims,
        quantization_cb=quantization_cb,
    )

    # Get all weight nodes in the subgraph except for out_proj
    weight_nodes = [
        n
        for n in get_all_weights_in_subgraph([entry_node], [next_lin_node])
        if "out_proj" not in str(n)
    ]

    # Shard remaining weights, such as conv1d or RMSNorm
    for weight_node in weight_nodes:
        weight_key = weight_node.target

        # Filter by regex patterns if provided
        if weights_to_shard is not None:
            if not any(pattern in weight_key for pattern in weights_to_shard):
                continue

        # Determine shard dimension for this weight
        shard_dim = weight_shard_dims.get(weight_key, dim) if weight_shard_dims else dim

        # Get the weight parameter
        try:
            weight_param = gm.get_parameter(weight_key)
        except AttributeError:
            ad_logger.debug(f"Could not get parameter for {weight_key}, skipping")
            continue

        # Get fused dims for this weight if specified
        fused_dims = None
        for k, v in fused_weight_dims.items():
            if k in weight_key:
                fused_dims = v
                break

        # Shard the weight tensor (also updates the parameter in the module)
        _, sharded_shape = shard_weight_tensor(
            gm=gm,
            weight_tensor=weight_param,
            param_key=weight_key,
            dim=shard_dim,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
            fused_weight_dims=fused_dims,
        )

        ad_logger.debug(
            f"Sharded weight {weight_key} on dim {shard_dim}: "
            f"{weight_param.shape} -> {sharded_shape}"
        )


def _insert_sharded_matmul(
    gm: GraphModule,
    node: Node,
    dim: int,
    rank: int,
    world_size: int,
    add_dist: bool = False,
    min_local_shape: int = 1,
    fused_weight_dims: Optional[list] = None,
    quantization_cb: Optional[
        Callable[[GraphModule, nn.Module, Node, str, torch.Size, int, int, int], None]
    ] = None,
) -> None:
    """Replace the matmul node with a new matmul node that accepts sharded weights.

    The state_dict is also updated to contain the sharded weights.
    """
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    num_users = num_users_of_weight_node(node)
    if num_users > 1 or num_users == 0:
        ad_logger.warning(
            f"Weight node {node} has {num_users} users. This is not supported for sharding. Skipping."
        )
        return
    # get weight and bias key
    weight_key, bias_key = extract_param_names_from_lin_node(node)

    modname = weight_key.rpartition(".")[0]
    submod = gm.get_submodule(modname)

    # Shard weight using the unified function (also updates the parameter)
    original_weight = gm.get_parameter(weight_key)
    _, weight_new_shape = shard_weight_tensor(
        gm=gm,
        weight_tensor=original_weight,
        param_key=weight_key,
        dim=dim,
        rank=rank,
        world_size=world_size,
        min_local_shape=min_local_shape,
        fused_weight_dims=fused_weight_dims,
    )

    if bias_key is not None and dim == 0:
        # update bias for dim 0 --> we can handle it like the weight
        original_bias = gm.get_parameter(bias_key)
        shard_weight_tensor(
            gm=gm,
            weight_tensor=original_bias,
            param_key=bias_key,
            dim=dim,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
            fused_weight_dims=None,
        )
    elif bias_key is not None and rank != world_size - 1:
        # update the bias for dim 1 --> in this case only the last rank gets the bias to avoid
        # double counting it. For all other we will delete the bias.
        args = list(node.args)
        node_bias = args[2]
        args[2] = None
        node.args = tuple(args)
        gm.graph.erase_node(node_bias)
        bias_param_name = bias_key.rpartition(".")[-1]
        setattr(submod, bias_param_name, None)
        gm._register_load_state_dict_pre_hook(partial(_load_hook_remove, param_key=bias_key))

    if quantization_cb is not None:
        quantization_cb(
            gm=gm,
            submod=submod,
            node=node,
            weight_key=weight_key,
            weight_new_shape=weight_new_shape,
            dim=dim,
            rank=rank,
            world_size=world_size,
        )

    # column shard with no gather: the output is sharded
    if not add_dist:
        _validate_sharded_shapes(node, fused_weight_dims=fused_weight_dims, world_size=world_size)
        return

    # figure out the right dist op
    dist_lookup = {
        0: (torch.ops.auto_deploy.torch_dist_all_gather.default, -1),
        1: (torch.ops.auto_deploy.torch_dist_all_reduce.default,),
    }
    fn_dist, *dist_args = dist_lookup[dim]

    # add reduction node
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(fn_dist, args=(node, *dist_args))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


class SplitDimension(IntEnum):
    """Enum for tensor split dimensions in sharding."""

    # NOTE: The names COLUMN/ROW reflect the hugging face
    # base_tp_plan sharding notation, but since we assume Y = W @ X^T,
    # when splitting weight matrix W^T across columns, the actual split
    # is over dimension 0
    COLUMN = 0
    ROW = 1


class ShardingTransformInfo(BaseModel, ABC):
    """Abstract base class for transformation configurations."""

    model_config = ConfigDict(frozen=True)  # Makes the model immutable and hashable

    target_node: str
    rank: int
    world_size: int

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """
        Validate whether the transformation is valid.
        Execute right before applying the transformation.
        """
        return True

    @abstractmethod
    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply the transformation to the graph module.

        This method must be implemented by each transformation class.
        """
        pass

    def check_and_apply(self, gm: GraphModule, node: Node) -> bool:
        """
        Check if the transformation is valid and apply it if it is.
        Return True if the transformation is applied, False otherwise.
        """
        if not self.validate(gm, node):
            ad_logger.warning(f"Skipping invalid transformation {self}.")
            return False
        self.apply(gm, node)
        return True


class LayerType(Enum):
    ATTENTION = "attention"
    MAMBA = "mamba"
    MLP = "mlp"
    MOE = "moe"


class TPShardingInfo(ShardingTransformInfo):
    """Configuration for TP sharding transformations."""

    split_dim: SplitDimension
    dist_op: Optional[Literal["all_reduce", "all_gather"]] = None
    min_local_shape: int = 1
    layer_type: LayerType = LayerType.MLP
    # used for TP sharding of fused weights
    # For MLP/Attention: list of dimensions for fused weights (e.g., [dim1, dim2] for QKV)
    # For Mamba: dict mapping weight keys to their fused dimensions
    fused_weight_dims: Optional[Union[list, Dict[str, list]]] = None

    @classmethod
    def from_node(cls, node: Node, **kwargs) -> "TPShardingInfo":
        """
        Create the correct TPShardingInfo subclass (FP8/FP4/base) based on `node`.
        """
        subcls = _resolve_tp_cls_from_node(node)
        return subcls(target_node=node.name, **kwargs)

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if self.dist_op is not None:
            if self.split_dim == SplitDimension.COLUMN:
                if self.dist_op == "all_reduce":
                    ad_logger.warning(
                        f"Column split is only supported for all_gather. Skipping {self}."
                    )
                    return False
            if self.split_dim == SplitDimension.ROW:
                if self.dist_op == "all_gather":
                    ad_logger.warning(
                        f"Row split is only supported for all_reduce. Skipping {self}."
                    )
                    return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply TP sharding transformation to the graph module."""

        if self.layer_type == LayerType.MAMBA:
            _insert_sharded_mamba(
                gm=gm,
                entry_node=node,
                dim=self.split_dim.value,
                rank=self.rank,
                world_size=self.world_size,
                add_dist=self.dist_op is not None,
                min_local_shape=self.min_local_shape,
                fused_weight_dims=self.fused_weight_dims
                if isinstance(self.fused_weight_dims, dict)
                else None,
            )
        else:
            _insert_sharded_matmul(
                gm=gm,
                node=node,
                dim=self.split_dim.value,
                rank=self.rank,
                world_size=self.world_size,
                add_dist=self.dist_op is not None,
                min_local_shape=self.min_local_shape,
                fused_weight_dims=self.fused_weight_dims,
            )


class QuantizationShardingMixin(ABC):
    """
    Mixin that provides a callback to handle quantization-aware sharding:
      - shards/rewrites scale buffers
      - registers the quantized shard load hook
    """

    @abstractmethod
    def scale_names(self) -> List[str]: ...

    def shard_scales(
        self,
        dim: int,
        rank: int,
        world_size: int,
        weight_shape: torch.Size,
        **scales: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in scales.items() if isinstance(v, torch.Tensor)}

    def shard_load_hook(
        self,
        state_dict,
        prefix,
        *args,
        weight_name: str,
        weight_shape: torch.Size,
        dim: int,
        rank: int,
        world_size: int,
    ) -> None:
        return

    def quantization_cb(
        self,
        gm: GraphModule,
        submod: nn.Module,
        node: Node,
        weight_key: str,
        weight_new_shape: torch.Size,
        dim: int,
        rank: int,
        world_size: int,
    ) -> None:
        scales = {}
        for scale_name in self.scale_names():
            scales[scale_name] = submod.get_buffer(scale_name)
        scales["weight_shape"] = weight_new_shape
        sharded_scales = self.shard_scales(dim, rank, world_size, **scales)
        for k, v in sharded_scales.items():
            submod.register_buffer(k, v)

        gm._register_load_state_dict_pre_hook(
            partial(
                self.shard_load_hook,
                weight_name=weight_key,
                weight_shape=weight_new_shape,
            )
        )


class FP8TPShardingInfo(QuantizationShardingMixin, TPShardingInfo):
    """Tensor-parallel sharding for FP8-quantized linears."""

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale"]

    def shard_scales(
        self,
        dim: int,
        rank: int,
        world_size: int,
        weight_shape: torch.Size,
        *,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "input_scale": input_scale,
            "weight_scale": weight_scale,
        }

    def shard_load_hook(
        self,
        state_dict,
        prefix,
        *args,
        weight_name: str,
        weight_shape: torch.Size,
        dim: int,
        rank: int,
        world_size: int,
    ) -> None:
        return

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_matmul(
            gm=gm,
            node=node,
            dim=self.split_dim.value,
            rank=self.rank,
            world_size=self.world_size,
            add_dist=self.dist_op is not None,
            min_local_shape=self.min_local_shape,
            quantization_cb=self.quantization_cb,  # quant callback
        )


def _shard_fp4_weight_scale(weight_scale, sharded_uint8_weight_shape, dim, rank, world_size):
    assert weight_scale.dim() == 1
    weight_shape_original = list(sharded_uint8_weight_shape)
    weight_shape_original[dim] = weight_shape_original[dim] * world_size
    weight_shape_original[-1] *= 2
    modelopt_weight_scale = cutlass_fp4_scale_to_modelopt_fp4_scale(
        weight_scale, tuple(weight_shape_original)
    )
    return modelopt_fp4_scale_to_cutlass_fp4_scale(
        modelopt_weight_scale.tensor_split(world_size, dim=dim)[rank]
    )


class FP4TPShardingInfo(QuantizationShardingMixin, TPShardingInfo):
    """Tensor-parallel sharding for FP4-quantized linears."""

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale", "alpha"]

    def shard_scales(
        self,
        dim: int,
        rank: int,
        world_size: int,
        weight_shape: torch.Size,
        *,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "alpha": alpha,
            "input_scale": input_scale,
            "weight_scale": _shard_fp4_weight_scale(
                weight_scale, weight_shape, dim, rank, world_size
            ),
        }

    def shard_load_hook(
        self,
        state_dict,
        prefix,
        *args,
        weight_name: str,
        weight_shape: torch.Size,
        dim: int,
        rank: int,
        world_size: int,
    ) -> None:
        key = weight_name + "_scale"
        if key in state_dict:
            state_dict[key] = _shard_fp4_weight_scale(
                state_dict[key], weight_shape, dim, rank, world_size
            )

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_matmul(
            gm=gm,
            node=node,
            dim=self.split_dim.value,
            rank=self.rank,
            world_size=self.world_size,
            add_dist=self.dist_op is not None,
            min_local_shape=self.min_local_shape,
            quantization_cb=self.quantization_cb,  # quant callback
        )


TP_SHARDING_RULES = [
    (lambda n: is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear), FP8TPShardingInfo),
    (lambda n: is_op(n, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear), FP4TPShardingInfo),
]


def _resolve_tp_cls_from_node(node: Node):
    for pred, cls in TP_SHARDING_RULES:
        try:
            if pred(node):
                return cls
        except Exception:
            pass
    return TPShardingInfo


class BMMShardingInfo(ShardingTransformInfo):
    """Configuration for BMM sharding transformations."""

    rank: int
    world_size: int
    start_idx: int
    end_idx: int

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if not is_op(node, torch.ops.aten.bmm):
            ad_logger.warning(f"BMM sharding is only supported for BMM nodes. Skipping {self}.")
            return False

        # Get the input tensors
        lhs_tensor = node.args[0]
        rhs_tensor = node.args[1]

        # Check batch sizes from meta information
        lhs_batch_size = lhs_tensor.meta["val"].shape[0]
        rhs_batch_size = rhs_tensor.meta["val"].shape[0]

        assert lhs_batch_size == rhs_batch_size, "Batch sizes of both tensors must match"
        bmm_batch_size = lhs_batch_size

        # Check if the distribution is balanced
        remainder = bmm_batch_size % self.world_size

        # NOTE: our torch.ops.auto_deploy.torch_dist_all_gather doesn't support uneven splits at the moment.
        if remainder:
            ad_logger.warning(
                f"BMM batch size {bmm_batch_size} is not divisible by world size {self.world_size}. "
                f"This will result in uneven distribution of work across devices. Skipping."
            )
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply BMM sharding transformation to the graph module."""

        def handle_tensor(
            bmm_node: Node, tensor_node: Node, arg_idx: int, start_idx: int, end_idx: int
        ):
            """Unified helper function to shard either a parameter tensor or a dynamic tensor.

            Args:
                bmm_node: The BMM node that is being processed
                tensor_node: The input tensor node to shard
                arg_idx: The argument index of the tensor in the BMM node
                start_idx: Start index for sharding
                end_idx: End index for sharding
            """

            if tensor_node.op == "get_attr":
                # Handle parameter tensor using unified shard_weight_tensor
                weight_key = tensor_node.target
                param = gm.get_parameter(weight_key)

                # Define slice function for the sharding
                def slice_tensor(t: torch.Tensor) -> torch.Tensor:
                    return t[start_idx:end_idx]

                # Use shard_weight_tensor with custom shard function (also updates the parameter)
                shard_weight_tensor(
                    gm=gm,
                    weight_tensor=param,
                    param_key=weight_key,
                    dim=0,  # BMM slices along batch dimension
                    rank=self.rank,
                    world_size=self.world_size,
                    custom_shard_fn=slice_tensor,
                    requires_grad=True,  # BMM parameters require gradients
                )
            else:
                # Handle dynamic tensor
                with gm.graph.inserting_before(bmm_node):
                    tensor_slice = gm.graph.call_function(
                        torch.ops.aten.slice.Tensor, args=(tensor_node, 0, start_idx, end_idx, 1)
                    )
                # Update BMM node to use the sliced tensor
                bmm_node.update_arg(arg_idx, tensor_slice)

        # Get the input tensors
        lhs_tensor = node.args[0]
        rhs_tensor = node.args[1]
        # Handle both tensors
        handle_tensor(node, lhs_tensor, 0, self.start_idx, self.end_idx)
        handle_tensor(node, rhs_tensor, 1, self.start_idx, self.end_idx)

        # Add all_gather node after BMM to collect results
        with gm.graph.inserting_after(node):
            gather_node = gm.graph.call_function(
                torch.ops.auto_deploy.torch_dist_all_gather.default,
                args=(node, 0),  # Gather along batch dimension (0)
            )
            node.replace_all_uses_with(gather_node)
            gather_node.replace_input_with(gather_node, node)


def _insert_sharded_moe(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
    scale_names: Sequence[str] = (),
):
    """Update the torch_moe node with sharded weight lists,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.
    """
    scale_names = list(scale_names)

    num_experts = len(node.args[3])
    args = list(node.args)

    # -- Handle selected_experts and final_scales sharding --
    selected_experts = args[1]
    final_scales = args[2]

    experts_per_rank = num_experts // world_size

    with gm.graph.inserting_before(node):
        lower = experts_per_rank * rank
        # selected_experts_local = selected_experts - low
        selected_experts_local = gm.graph.create_node(
            "call_function", operator.sub, args=(selected_experts, lower), kwargs={}
        )

        # For num_experts % world_size != 0 case,
        # assign the last (num_experts % world_size) experts to the last rank
        # if rank == world_size -1:
        #     rank_mask = (selected_experts // experts_per_rank) >= rank
        # else:
        #     rank_mask = (selected_experts // experts_per_rank) == rank
        div_node = gm.graph.create_node(
            "call_function", operator.floordiv, args=(selected_experts, experts_per_rank), kwargs={}
        )
        comp_op = torch.ge if rank == world_size - 1 else torch.eq
        rank_mask = gm.graph.create_node("call_function", comp_op, args=(div_node, rank), kwargs={})

        # final_scales_local = final_scales * rank_mask
        final_scales_local = gm.graph.create_node(
            "call_function", operator.mul, args=(final_scales, rank_mask), kwargs={}
        )

    # -- Shard expert weights --
    def get_partition(lst, world_size, rank):
        num_experts = len(lst)
        expert_size_per_partition = num_experts // world_size
        expert_start = rank * expert_size_per_partition
        # For num_experts % world_size != 0 case,
        # assign the last (num_experts % world_size) experts to the last rank
        expert_end = (
            num_experts if (rank == world_size - 1) else expert_start + expert_size_per_partition
        )
        return lst[expert_start:expert_end]

    w1_list_sharded = get_partition(args[3], world_size, rank)
    w2_list_sharded = get_partition(args[4], world_size, rank)
    w3_list_sharded = get_partition(args[5], world_size, rank)

    # -- Update args --
    args[1] = selected_experts_local
    args[2] = final_scales_local
    args[3] = w1_list_sharded
    args[4] = w2_list_sharded
    args[5] = w3_list_sharded

    # Shard scales for quantized ops
    for i in range(len(scale_names) * 3):  # 3 layers (w1, w2, w3) × #scale_names per layer
        args[6 + i] = get_partition(args[6 + i], world_size, rank)

    ad_logger.debug(
        f"Updated node {node}: replaced original arguments {node.args} with sharded arguments {args}."
    )
    node.args = tuple(args)

    # -- add an all_reduce node --
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_reduce.default, args=(node,)
        )
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _slice_expert_dim(gm: GraphModule, tensor_node: Node, lo: int, hi: int) -> Node:
    """Return tensor_node[lo:hi, ...] via aten.slice along dim 0."""
    with gm.graph.inserting_after(tensor_node):
        # aten.slice.Tensor(self, dim, start, end, step)
        return gm.graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(tensor_node, 0, lo, hi, 1),
        )


def _split_range_last_remainder(n: int, world_size: int, rank: int):
    """[lo, hi) split along dim0; last rank gets remainder."""
    base = n // world_size
    lo = base * rank
    hi = n if rank == world_size - 1 else base * (rank + 1)
    return lo, hi


def _insert_sharded_mxfp4_mlp_ep(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
):
    """
    Transform a call to auto_deploy::triton_mxfp4_moe into:
      - sharded expert parameters along dim 0 (this rank's slice),
      - call to auto_deploy::triton_mxfp4_moe_ep(..., local_lo, local_hi),
      - followed by torch_dist_all_reduce.

    Expects the original op signature:
      (hidden_states,
       router_weight, router_bias, top_k,
       gate_up_blocks, gate_up_bias, gate_up_scales,
       alpha, limit,
       down_blocks, down_bias, down_scales)
    """

    IDX_GATE_UP_BLOCKS = 4
    IDX_GATE_UP_BIAS = 5
    IDX_GATE_UP_SCALES = 6
    IDX_DOWN_BLOCKS = 9
    IDX_DOWN_BIAS = 10
    IDX_DOWN_SCALES = 11

    gate_up_blocks_node = node.args[IDX_GATE_UP_BLOCKS]
    num_experts = int(gate_up_blocks_node.meta["val"].shape[0])

    local_lo, local_hi = _split_range_last_remainder(num_experts, world_size, rank)

    # Prepare new args with slices for this rank
    args = list(node.args)
    args[IDX_GATE_UP_BLOCKS] = _slice_expert_dim(gm, args[IDX_GATE_UP_BLOCKS], local_lo, local_hi)
    args[IDX_GATE_UP_BIAS] = _slice_expert_dim(gm, args[IDX_GATE_UP_BIAS], local_lo, local_hi)
    args[IDX_GATE_UP_SCALES] = _slice_expert_dim(gm, args[IDX_GATE_UP_SCALES], local_lo, local_hi)
    args[IDX_DOWN_BLOCKS] = _slice_expert_dim(gm, args[IDX_DOWN_BLOCKS], local_lo, local_hi)
    args[IDX_DOWN_BIAS] = _slice_expert_dim(gm, args[IDX_DOWN_BIAS], local_lo, local_hi)
    args[IDX_DOWN_SCALES] = _slice_expert_dim(gm, args[IDX_DOWN_SCALES], local_lo, local_hi)

    args_ep = tuple(args) + (int(world_size), int(rank))
    node.target = torch.ops.auto_deploy.triton_mxfp4_moe_ep.default
    node.args = args_ep

    # Add a dist all-reduce after the op (sum partial results across EP ranks)
    with gm.graph.inserting_after(node):
        red = gm.graph.call_function(torch.ops.auto_deploy.torch_dist_all_reduce, args=(node,))
        node.replace_all_uses_with(red)
        # keep dataflow: red(input=node)
        red.replace_input_with(red, node)


class EPShardingInfo(ShardingTransformInfo):
    """Configuration for EP sharding transformations."""

    rank: int
    world_size: int

    @classmethod
    def from_node(cls, node: Node, **kwargs) -> "EPShardingInfo":
        """
        Create the correct EPShardingInfo subclass (FP8/NVFP4/base) based on `node`.
        """
        subcls = _resolve_ep_cls_from_node(node)
        return subcls(target_node=node.name, **kwargs)

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if not is_op(node, torch.ops.auto_deploy.torch_moe):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply EP sharding transformation to the graph module."""
        _insert_sharded_moe(gm, node, self.rank, self.world_size, [])


class MXFP4EPShardingInfo(EPShardingInfo):
    """GPT-OSS style MXFP4-specific EP sharding behavior."""

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if not is_op(node, torch.ops.auto_deploy.triton_mxfp4_moe):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_mxfp4_mlp_ep(gm, node, self.rank, self.world_size)


class FP8EPShardingInfo(EPShardingInfo, QuantizationShardingMixin):
    """FP8-specific EP sharding behavior."""

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        if not is_op(node, torch.ops.auto_deploy.torch_quant_fp8_moe):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale"]

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_moe(gm, node, self.rank, self.world_size, self.scale_names())


class NVFP4EPShardingInfo(EPShardingInfo, QuantizationShardingMixin):
    """NVFP4-specific EP sharding behavior."""

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        if not is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_moe):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale", "alpha"]

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_moe(gm, node, self.rank, self.world_size, self.scale_names())


EP_SHARDING_RULES = [
    (lambda n: is_op(n, torch.ops.auto_deploy.torch_quant_fp8_moe), FP8EPShardingInfo),
    (lambda n: is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_moe), NVFP4EPShardingInfo),
    (lambda n: is_op(n, torch.ops.auto_deploy.torch_moe), EPShardingInfo),
    (lambda n: is_op(n, torch.ops.auto_deploy.triton_mxfp4_moe), MXFP4EPShardingInfo),
]


def _resolve_ep_cls_from_node(node: Node) -> type[EPShardingInfo]:
    for pred, cls in EP_SHARDING_RULES:
        try:
            if pred(node):
                return cls
        except Exception:
            # Missing op variant in this build or other harmless issues — keep trying.
            pass
    return EPShardingInfo


class ShardingConfig(BaseModel):
    """Configuration for sharding the model."""

    factory_source: ShardingConfigSource = Field(default=ShardingConfigSource.UNKNOWN)
    rank: int = Field(default=0)
    world_size: int = Field(default=1)
    predefined_config: Optional[Dict[str, Any]] = None
    simple_shard_only: bool = Field(default=False)
    use_sharding_from_factory: bool = False
    support_partial_config: bool = False
    sharding_dims: List[str] = Field(default_factory=list)
    tp_transforms: List[TPShardingInfo] = Field(default_factory=list)
    bmm_transforms: List[BMMShardingInfo] = Field(default_factory=list)
    ep_transforms: List[EPShardingInfo] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_and_normalize(self):
        # Normalize empty dict to None for "no config"
        if isinstance(self.predefined_config, dict) and not self.predefined_config:
            self.predefined_config = None
        # Validate only if provided
        if self.predefined_config is not None:
            self.validate_config()
        return self

    def validate_config(self) -> bool:
        if self.factory_source != ShardingConfigSource.HUGGINGFACE:
            ad_logger.warning(
                "Sharding config is currently only supported for HuggingFace. Skipping."
            )
            # invalidate the config
            self.predefined_config = {}
            return False

        if not isinstance(self.predefined_config, dict):
            ad_logger.warning("Sharding config is not a dictionary. Skipping.")
            # invalidate the config
            self.predefined_config = {}
            return False

        if "head_dim" not in self.predefined_config:
            ad_logger.warning("Sharding config does not contain head_dim. Skipping.")
            # invalidate the config
            self.predefined_config = {}
            return False

        if "tp_plan" not in self.predefined_config or self.predefined_config["tp_plan"] is None:
            ad_logger.warning("Sharding config does not contain tp_plan. Skipping.")
            # invalidate the config
            self.predefined_config = {}
            return False
        tp_plan = self.predefined_config["tp_plan"]

        values = set(tp_plan.values())
        supported_modes = {
            "colwise",  # row split and no collective
            "rowwise",  # column split and all-reduce
            "gather",  # simple shard (row + all_gather)
            # TODO: remaining values are not supported yet.
            # They require hybrid EP+TP and/or SP support.
            # "sequence_parallel", # sequence parallelism
            # "local_colwise",
            # "local_rowwise",
            # "local_packed_rowwise",
            # "local",
        }
        if not self.support_partial_config and not values.issubset(supported_modes):
            ad_logger.warning("Sharding config contains invalid values. Skipping.")
            # invalidate the config
            self.predefined_config = {}
            return False
        return True

    def get_predefined_config(self) -> Dict[str, Any]:
        return self.predefined_config
