"""Sharding config definitions for the inference optimizer."""

import math
import operator
import re
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.fx import GraphModule, Node

from ....functional import AllReduceStrategy
from ..models.factory import ShardingConfigSource
from ..utils.logger import ad_logger
from .node_utils import (
    bfs,
    extract_param_names_from_node,
    is_any_lin_op,
    is_op,
    num_users_of_weight_node,
    subgraph,
)
from .quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    modelopt_fp4_scale_to_cutlass_fp4_scale,
)

if TYPE_CHECKING:
    from ..transform.library.sharding import ShardingTransformConfig


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


def _get_dist_ops(backend: str):
    """Get the appropriate distributed ops based on backend availability.

    Args:
        backend: The distributed backend to use. Can be 'auto', 'trtllm', or 'torch'.
                 'auto' will automatically select based on availability.

    Returns tuple of (all_gather_op, all_reduce_op) for the current backend.
    """
    from ..custom_ops.trtllm_dist import is_trtllm_op_available

    # Handle DistBackend enum or string
    if hasattr(backend, "value"):
        backend = backend.value

    if backend == "trtllm":
        # Force TRT-LLM ops
        return (
            torch.ops.auto_deploy.trtllm_dist_all_gather.default,
            torch.ops.auto_deploy.trtllm_dist_all_reduce.default,
        )
    elif backend == "torch":
        # Force PyTorch distributed ops
        return (
            torch.ops.auto_deploy.torch_dist_all_gather.default,
            torch.ops.auto_deploy.torch_dist_all_reduce.default,
        )
    else:  # auto
        # Automatically select based on availability
        if is_trtllm_op_available():
            # Use TRT-LLM optimized ops in MPI mode
            return (
                torch.ops.auto_deploy.trtllm_dist_all_gather.default,
                torch.ops.auto_deploy.trtllm_dist_all_reduce.default,
            )
        else:
            # Use PyTorch distributed ops in demollm mode
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
    node: Node, fused_weight_dims: Optional[list] = None, world_size: Optional[int] = None
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
    next_lin_node, _ = bfs(node, is_any_lin_op, include_root=False)
    nodes_to_validate = subgraph(
        [node],
        include=lambda n: is_op(n, [torch.ops.aten.view, torch.ops.aten.reshape]),
        boundary_condition=is_any_lin_op,
    )
    for shape_node in nodes_to_validate:
        # Parameter update must be idempotent
        if "sharded" in shape_node.meta and shape_node.meta["sharded"]:
            continue
        if len(shape_node.args) < 2:
            continue
        view_shape = list(shape_node.args[1])
        if not isinstance(view_shape, list):
            continue
        if len(view_shape) >= 3 and isinstance(view_shape[2], int) and view_shape[2] != -1:
            args = list(shape_node.args)
            view_shape[2] = -1  # view_shape[2] // world_size
            args[1] = tuple(view_shape)
            shape_node.args = tuple(args)
            shape_node.meta["sharded"] = True
            ad_logger.debug(f"\nUpdated view node {shape_node} arguments to {shape_node.args}")

    # if fused_weight_dims is provided, we need to update all split sizes
    if fused_weight_dims is not None:
        assert world_size is not None, "World size is required to update the split node params"
        assert len(node.users) == 1, "Fused linear node should have only one user: a split node"
        # find all split nodes in the region between this linear node and the next
        split_nodes = subgraph(
            [node],
            [next_lin_node],
            include=lambda n: is_op(n, [torch.ops.aten.split_with_sizes]),
        )
        for split_node in split_nodes:
            # Parameter update must be idempotent
            if "sharded" in split_node.meta and split_node.meta["sharded"]:
                continue
            orig_sizes = split_node.args[1]
            new_sizes = [orig_sizes[i] // world_size for i in range(len(orig_sizes))]
            args = list(split_node.args)
            args[1] = new_sizes
            split_node.args = tuple(args)
            split_node.meta["sharded"] = True
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
    requires_grad: bool = False,
    update_param: bool = True,
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
        update_param: Whether to update the parameter in the module

    Returns:
        Tuple of (sharded_tensor, sharded_shape)
    """

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
        f_split = split_tensor

    sharded_weight = f_split(weight_tensor)
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
    weight_nodes = subgraph(sources, sinks, include=lambda n: n.op == "get_attr")
    return weight_nodes


def _insert_sharded_mamba(
    gm: GraphModule,
    entry_node: Node,
    dim: int,
    rank: int,
    world_size: int,
    allreduce_strategy: AllReduceStrategy,
    dist_backend: str,
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

    NOTE: allreduce_strategy is MANDATORY and must be explicitly provided.
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
    if allreduce_strategy is None:
        raise ValueError(
            f"allreduce_strategy must be set for Mamba sharding on node {entry_node.name}"
        )
    # Find next linear node to define subgraph boundary
    try:
        next_lin_node, depth = bfs(entry_node, is_any_lin_op, include_root=False)
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

    ##############################################################
    ####### shard the entry_node (the first linear layer) ########
    ##############################################################
    # Extract entry node's fused_weight_dims by matching weight name against patterns
    entry_fused_dims = None
    if fused_weight_dims:
        entry_weight_key, _ = extract_param_names_from_node(entry_node)
        for pattern, dims in fused_weight_dims.items():
            if re.search(pattern, entry_weight_key):
                entry_fused_dims = dims
                break

    _shard_parameter_node(
        gm=gm,
        node=entry_node,
        dim=SplitDimension.COLUMN,
        rank=rank,
        world_size=world_size,
        dist_backend=dist_backend,
        add_dist=False,
        min_local_shape=min_local_shape,
        fused_weight_dims=entry_fused_dims,
        quantization_cb=quantization_cb,
        allreduce_strategy=allreduce_strategy,
    )

    ##############################################################
    ######## Shard remaining weights: conv1d and RMSNorm #########
    ##############################################################
    # Get all weight nodes in the subgraph except for out_proj
    weight_nodes = [
        n
        for n in get_all_weights_in_subgraph([entry_node], [next_lin_node])
        if "out_proj" not in str(n)
    ]

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

    ##############################################################
    ############## update split node parameters ##################
    ##############################################################
    next_lin_node, _ = bfs(entry_node, is_any_lin_op, include_root=False)

    split_nodes = subgraph(
        [entry_node],
        [next_lin_node],
        include=lambda n: is_op(n, [torch.ops.aten.split_with_sizes]),
    )
    for split_node in split_nodes:
        orig_sizes = split_node.args[1]
        new_sizes = [orig_sizes[i] // world_size for i in range(len(orig_sizes))]
        args = list(split_node.args)
        args[1] = new_sizes
        split_node.args = tuple(args)
        ad_logger.debug(f"\nUpdated split node {split_node} arguments to {split_node.args}")

    nodes_to_validate = subgraph(
        [entry_node],
        include=lambda n: is_op(n, [torch.ops.aten.view, torch.ops.aten.reshape]),
        boundary_condition=is_any_lin_op,
    )
    for reshape_node in nodes_to_validate:
        if len(reshape_node.args) < 2:
            continue
        if "sharded" in reshape_node.meta and reshape_node.meta["sharded"]:
            continue
        view_shape = list(reshape_node.args[1])
        if not isinstance(view_shape, list):
            continue
        if len(view_shape) >= 3 and isinstance(view_shape[2], int) and view_shape[2] != -1:
            args = list(reshape_node.args)
            view_shape[2] = -1  # view_shape[2] // world_size
            args[1] = tuple(view_shape)
            reshape_node.args = tuple(args)
            reshape_node.meta["sharded"] = True
            ad_logger.debug(f"\nUpdated view node {reshape_node} arguments to {reshape_node.args}")


def _shard_parameter_node(
    gm: GraphModule,
    node: Node,
    dim: int,
    rank: int,
    world_size: int,
    allreduce_strategy: AllReduceStrategy,
    dist_backend: str,
    add_dist: bool = False,
    min_local_shape: int = 1,
    fused_weight_dims: Optional[list] = None,
    quantization_cb: Optional[
        Callable[[GraphModule, nn.Module, Node, str, torch.Size, int, int, int], None]
    ] = None,
) -> None:
    """Replace the node with parametrized weight tensor with a new node that accepts sharded weights.

    NOTE: allreduce_strategy is MANDATORY and must be explicitly provided.

    The state_dict is also updated to contain the sharded weights.
    """
    if allreduce_strategy is None:
        raise ValueError(
            f"allreduce_strategy must be set for parameter sharding on node {node.name}"
        )
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    num_users = num_users_of_weight_node(node)
    if num_users > 1 or num_users == 0:
        ad_logger.warning(
            f"Weight node {node} has {num_users} users. This is not supported for sharding. Skipping."
        )
        return
    # get weight and bias key
    weight_key, bias_key = extract_param_names_from_node(node)

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
            fused_weight_dims=fused_weight_dims,
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

    # # # column shard with no gather: the output is sharded
    if not add_dist:
        return

    # figure out the right dist op (backend-aware)
    all_gather_op, all_reduce_op = _get_dist_ops(dist_backend)
    dist_lookup = {
        0: (all_gather_op, -1),
        1: (all_reduce_op, allreduce_strategy.name),
    }
    fn_dist, *dist_args = dist_lookup[dim]

    # add reduction node
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(fn_dist, args=(node,) + tuple(dist_args))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _update_node_args(node: Node, args: tuple) -> None:
    """Update the node's arguments with the new sharded arguments."""
    if "sharded" in node.meta and node.meta["sharded"]:
        return
    node.args = args
    node.meta["sharded"] = True
    ad_logger.debug(
        f"Updated node {node}: replaced original arguments {node.args} with sharded arguments {args}."
    )


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


class WeightShardingInfo(ShardingTransformInfo):
    """Configuration for TP sharding transformations.

    NOTE: allreduce_strategy will be automatically injected by ShardingConfig.add()
    if not provided at creation time. The strategy comes from the parent ShardingConfig.
    """

    split_dim: SplitDimension
    dist_op: Optional[Literal["all_reduce", "all_gather"]] = None
    min_local_shape: int = 1
    layer_type: LayerType = LayerType.MLP
    # used for TP sharding of fused weights
    fused_weight_dims: Optional[list] = None
    allreduce_strategy: Optional[AllReduceStrategy] = None  # Set by ShardingConfig.add() if None
    dist_backend: Optional[str] = None  # Set by ShardingConfig.add() if None

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
        """Quantization callback. Default does nothing for non-quantized models."""
        return None

    @classmethod
    def from_node(cls, node: Node, **kwargs) -> "WeightShardingInfo":
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
        _shard_parameter_node(
            gm=gm,
            node=node,
            dim=self.split_dim.value,
            rank=self.rank,
            world_size=self.world_size,
            add_dist=self.dist_op is not None,
            dist_backend=self.dist_backend,
            min_local_shape=self.min_local_shape,
            fused_weight_dims=self.fused_weight_dims,
            quantization_cb=self.quantization_cb,
            allreduce_strategy=self.allreduce_strategy,
        )


class ParameterUpdateInfo(ShardingTransformInfo):
    """Configuration for node args sharding transformations."""

    args: tuple

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        return len(node.args) == len(self.args)

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply the transformation to the graph module."""
        _update_node_args(node, self.args)


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
                dim=dim,
                rank=rank,
                world_size=world_size,
            )
        )


class FP8TPShardingInfo(QuantizationShardingMixin, WeightShardingInfo):
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


class FP4TPShardingInfo(QuantizationShardingMixin, WeightShardingInfo):
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
    return WeightShardingInfo


class BMMShardingInfo(ShardingTransformInfo):
    """Configuration for BMM sharding transformations."""

    rank: int
    world_size: int
    start_idx: int
    end_idx: int
    dist_backend: Optional[str] = None  # Set by ShardingConfig.add() if None

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

        # NOTE: our torch.ops.auto_deploy.torch_dist_all_gather/trtllm_dist_all_gather
        #  doesn't support uneven splits at the moment.
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

            # Define slice function for the sharding
            def slice_tensor(t: torch.Tensor) -> torch.Tensor:
                return t[start_idx:end_idx]

            if tensor_node.op == "get_attr":
                # Handle parameter tensor
                weight_key = tensor_node.target
                modname, _, param_name = weight_key.rpartition(".")
                param = gm.get_parameter(weight_key)

                # Update the parameter with its shard
                param_new = nn.Parameter(slice_tensor(param).detach().clone(), requires_grad=True)
                gm.get_submodule(modname).register_parameter(param_name, param_new)

                # Register load state dict hook
                gm._register_load_state_dict_pre_hook(
                    partial(
                        _load_hook,
                        f_split=slice_tensor,
                        param_key=weight_key,
                        param_shape=param_new.shape,
                    )
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
        all_gather_op, _ = _get_dist_ops(self.dist_backend)
        with gm.graph.inserting_after(node):
            gather_node = gm.graph.call_function(
                all_gather_op,
                args=(node, 0),  # Gather along batch dimension (0)
            )
            node.replace_all_uses_with(gather_node)
            gather_node.replace_input_with(gather_node, node)


def _insert_sharded_moe(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
    allreduce_strategy: AllReduceStrategy,
    dist_backend: str,
    scale_names: Sequence[str] = (),
):
    """Update the torch_moe node with sharded weight lists or stacked tensors,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.

    Handles both:
    - Standard format: per-expert weight lists
    - Stacked format: single-element lists containing stacked 3D tensors (Llama4 pattern)

    NOTE: allreduce_strategy is MANDATORY.
    """
    if allreduce_strategy is None:
        raise ValueError(f"allreduce_strategy must be set for MoE sharding on node {node.name}")
    scale_names = list(scale_names)

    # Detect format: check if w1_weight is a single-element list with a 3D tensor (stacked format)
    w1_weight_arg = node.args[3]
    is_stacked = False

    # In FX graphs, the list might be a Node representing a list() call
    if isinstance(w1_weight_arg, Node):
        # Check if this is a list() call node
        if w1_weight_arg.target is list and len(w1_weight_arg.args) > 0:
            # Get the actual list content from the args
            list_content = w1_weight_arg.args[0]
            if isinstance(list_content, (list, tuple)) and len(list_content) == 1:
                first_elem = list_content[0]
                if isinstance(first_elem, Node) and first_elem.op == "get_attr":
                    try:
                        tensor = gm.get_parameter(first_elem.target)
                        is_stacked = tensor.ndim == 3
                    except (AttributeError, KeyError):
                        pass
                elif isinstance(first_elem, torch.Tensor):
                    is_stacked = first_elem.ndim == 3
    # Handle case where it's a direct Python list (not in FX graph context)
    elif isinstance(w1_weight_arg, (list, tuple)) and len(w1_weight_arg) == 1:
        first_elem = w1_weight_arg[0]
        if isinstance(first_elem, Node) and first_elem.op == "get_attr":
            try:
                tensor = gm.get_parameter(first_elem.target)
                is_stacked = tensor.ndim == 3
            except (AttributeError, KeyError):
                pass
        elif isinstance(first_elem, torch.Tensor):
            is_stacked = first_elem.ndim == 3

    if is_stacked:
        # Use stacked tensor sharding logic (similar to _insert_sharded_moe_bmm)
        _insert_sharded_moe_stacked(gm, node, rank, world_size, allreduce_strategy, scale_names)
        return

    # Standard per-expert list sharding
    # For FX graphs, get the list from the Node; for direct calls, use the list directly
    if isinstance(w1_weight_arg, Node) and w1_weight_arg.target is list:
        # Extract the list content from the list() call node
        num_experts = len(w1_weight_arg.args[0]) if w1_weight_arg.args else 0
    elif isinstance(w1_weight_arg, (list, tuple)):
        num_experts = len(w1_weight_arg)
    else:
        raise ValueError(f"Unexpected w1_weight format in node {node.name}: {type(w1_weight_arg)}")
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
    _, all_reduce_op = _get_dist_ops(dist_backend)
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(all_reduce_op, args=(node, allreduce_strategy.name))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _slice_expert_dim(
    gm: GraphModule,
    tensor_node_or_tensor: Union[Node, torch.Tensor],
    lo: int,
    hi: int,
) -> Union[Node, torch.Tensor]:
    """Slice expert weights along dim 0 and register load hook (simple version).

    This is the original simple slicing function used by MXFP4 EP sharding.
    For parameters, it modifies them in-place and returns the same node.

    Args:
        gm: The graph module
        tensor_node_or_tensor: Either a Node (from FX graph) or a Tensor
        lo: Start index for slicing
        hi: End index for slicing

    Returns:
        Node or Tensor depending on input type
    """
    # Handle raw tensor case
    if isinstance(tensor_node_or_tensor, torch.Tensor):
        return tensor_node_or_tensor[lo:hi]

    # Handle Node case
    tensor_node = tensor_node_or_tensor

    if tensor_node.op != "get_attr":
        # If not a parameter node, just add a runtime slice node after it
        with gm.graph.inserting_after(tensor_node):
            return gm.graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(tensor_node, 0, lo, hi, 1),
            )

    # Get the parameter
    param_key = str(tensor_node.target)
    modname, _, param_name = param_key.rpartition(".")
    submod = gm.get_submodule(modname) if modname else gm
    full_param = getattr(submod, param_name)

    # Slice the parameter
    sliced_param = full_param[lo:hi].detach().clone()
    sliced_shape = sliced_param.shape

    # Define slice function for load hook
    def slice_expert_tensor(t: torch.Tensor) -> torch.Tensor:
        return t[lo:hi]

    # Register load hook to slice during checkpoint loading
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook,
            f_split=slice_expert_tensor,
            param_key=param_key,
            param_shape=sliced_shape,
        )
    )

    # Replace the parameter with the sliced version
    new_param = nn.Parameter(sliced_param, requires_grad=False)
    setattr(submod, param_name, new_param)

    # Return the same node (it now points to the sliced parameter)
    return tensor_node


def _transform_bmm_moe_weight_param(
    gm: GraphModule,
    param_node: Node,
    lo: int,
    hi: int,
    swap_gate_up: bool = False,
) -> None:
    """Transform a parameter for BMM MoE: slice experts, optionally swap gate/up, transpose.

    This modifies the parameter in-place and registers a load hook.
    Does NOT create graph nodes - those should be created separately by the caller.

    Args:
        gm: Graph module
        param_node: The get_attr node for the parameter
        lo: Start index for expert slicing
        hi: End index for expert slicing
        swap_gate_up: If True, swap W1 and W3 (Llama4 -> TRT-LLM format)
    """
    if param_node.op != "get_attr":
        return  # Only works on parameters

    param_key = str(param_node.target)
    modname, _, param_name = param_key.rpartition(".")
    submod = gm.get_submodule(modname) if modname else gm
    full_param = getattr(submod, param_name)

    # Slice the parameter along expert dimension (dim 0)
    sliced_param = full_param[lo:hi].detach().clone()

    # Swap W1 and W3 if needed (for gate_up weights)
    # Llama4: (E, H, 2*I) with [W1, W3], TRT-LLM wants [W3, W1]
    if swap_gate_up and sliced_param.ndim == 3:
        intermediate_size = sliced_param.shape[2] // 2
        w1 = sliced_param[:, :, :intermediate_size]
        w3 = sliced_param[:, :, intermediate_size:]
        sliced_param = torch.cat([w3, w1], dim=2)

    # Transpose: Llama4 (E, H, X) -> TRT-LLM (E, X, H)
    transposed_param = sliced_param.transpose(1, 2)
    transposed_shape = transposed_param.shape

    # Define transformation function for load hook
    def transform_tensor(t: torch.Tensor) -> torch.Tensor:
        t_sliced = t[lo:hi]
        if swap_gate_up and t_sliced.ndim == 3:
            intermediate_size = t_sliced.shape[2] // 2
            w1 = t_sliced[:, :, :intermediate_size]
            w3 = t_sliced[:, :, intermediate_size:]
            t_sliced = torch.cat([w3, w1], dim=2)
        return t_sliced.transpose(1, 2).contiguous()

    # Register load hook
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook,
            f_split=transform_tensor,
            param_key=param_key,
            param_shape=transposed_shape,
        )
    )

    # Replace the parameter with the transformed version
    new_param = nn.Parameter(transposed_param, requires_grad=False)
    setattr(submod, param_name, new_param)


def _get_dim0_from_arg(gm: GraphModule, arg: Union[Node, torch.Tensor]) -> int:
    """Helper to get the first dimension size of an argument (Node or Tensor)."""
    if isinstance(arg, torch.Tensor):
        return arg.shape[0]
    if isinstance(arg, Node):
        if arg.op == "get_attr":
            # Traverse attributes to find the tensor
            obj = gm
            for atom in arg.target.split("."):
                obj = getattr(obj, atom)
            return obj.shape[0]
        if "val" in arg.meta:
            return arg.meta["val"].shape[0]
    raise ValueError(f"Cannot determine shape[0] for {arg}")


def _insert_sharded_moe_stacked(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
    allreduce_strategy: AllReduceStrategy,
    scale_names: Sequence[str] = (),
):
    """Update the torch_moe node with sliced stacked weight tensors,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.

    For torch_moe with stacked tensor format (single-element lists containing 3D tensors).

    NOTE: allreduce_strategy is MANDATORY and must be explicitly provided.
    """
    if allreduce_strategy is None:
        raise ValueError(f"allreduce_strategy must be set for MoE sharding on node {node.name}")

    # Extract the stacked tensors from single-element lists
    # args[3] = w1_weight (Node representing list with one 3D tensor, or direct list)
    # args[4] = w2_weight (Node representing list with one 3D tensor, or direct list)

    # Helper to extract tensor node from list (handles both Node and direct list)
    def extract_tensor_from_list_arg(list_arg):
        if isinstance(list_arg, Node) and list_arg.target is list:
            # It's a list() call node - extract from its args
            return list_arg.args[0][0]  # args[0] is the list content, [0] is first element
        elif isinstance(list_arg, (list, tuple)):
            # Direct list
            return list_arg[0]
        else:
            raise ValueError(f"Unexpected list format: {type(list_arg)}")

    w3_w1_tensor_node = extract_tensor_from_list_arg(node.args[3])
    w2_tensor_node = extract_tensor_from_list_arg(node.args[4])
    num_experts = _get_dim0_from_arg(gm, w3_w1_tensor_node)

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
        div_node = gm.graph.create_node(
            "call_function", operator.floordiv, args=(selected_experts, experts_per_rank), kwargs={}
        )

        comp_op = torch.ge if rank == world_size - 1 else torch.eq
        rank_mask = gm.graph.create_node("call_function", comp_op, args=(div_node, rank), kwargs={})

        # final_scales_local = final_scales * rank_mask
        final_scales_local = gm.graph.create_node(
            "call_function", operator.mul, args=(final_scales, rank_mask), kwargs={}
        )

    # -- Transform expert weight parameters --
    local_lo, local_hi = _split_range_last_remainder(num_experts, world_size, rank)

    # Transform w3_w1_stacked: slice experts, swap [W1,W3]->[W3,W1], transpose (E,H,2I)->(E,2I,H)
    if isinstance(w3_w1_tensor_node, Node):
        _transform_bmm_moe_weight_param(
            gm, w3_w1_tensor_node, local_lo, local_hi, swap_gate_up=True
        )

    # Transform w2_stacked: slice experts, transpose (E,I,H)->(E,H,I)
    if isinstance(w2_tensor_node, Node):
        _transform_bmm_moe_weight_param(gm, w2_tensor_node, local_lo, local_hi, swap_gate_up=False)

    # -- Update args (keep same lists/nodes, just with transformed parameters) --
    args[1] = selected_experts_local
    args[2] = final_scales_local
    # args[3] and args[4] stay the same - we modified the parameters in-place

    ad_logger.debug(
        f"Updated node {node}: replaced original arguments {node.args} with sharded arguments {args}."
    )

    node.args = tuple(args)

    # -- add an all_reduce node --
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_reduce.default,
            args=(node, allreduce_strategy.name),
        )
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


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
    allreduce_strategy: AllReduceStrategy,
    dist_backend: str,
):
    """Transform a call to auto_deploy::triton_mxfp4_moe into:
      - sharded expert parameters along dim 0 (this rank slice),
      - call to auto_deploy::triton_mxfp4_moe_ep(..., local_lo, local_hi),
      - followed by torch_dist_all_reduce/trtllm_dist_all_reduce.

    NOTE: allreduce_strategy is MANDATORY and must be explicitly provided.

    Expects the original op signature:
      (hidden_states,
       router_weight, router_bias, top_k,
       gate_up_blocks, gate_up_bias, gate_up_scales,
       alpha, limit,
       down_blocks, down_bias, down_scales)
    """
    if allreduce_strategy is None:
        raise ValueError(
            f"allreduce_strategy must be set for MXFP4 MLP EP sharding on node {node.name}"
        )

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
    _, all_reduce_op = _get_dist_ops(dist_backend)
    with gm.graph.inserting_after(node):
        red = gm.graph.call_function(all_reduce_op, args=(node, allreduce_strategy.name))
        node.replace_all_uses_with(red)
        # keep dataflow: red(input=node)
        red.replace_input_with(red, node)


class EPShardingInfo(ShardingTransformInfo):
    """Configuration for EP sharding transformations.

    NOTE: allreduce_strategy and dist_backend will be automatically injected by
    ShardingConfig.add() if not provided at creation time. The values come from
    the parent ShardingConfig.
    """

    allreduce_strategy: Optional[AllReduceStrategy] = None  # Set by ShardingConfig.add() if None
    dist_backend: Optional[str] = None  # Set by ShardingConfig.add() if None

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
        _insert_sharded_moe(
            gm, node, self.rank, self.world_size, self.allreduce_strategy, self.dist_backend, []
        )


class MXFP4EPShardingInfo(EPShardingInfo):
    """GPT-OSS style MXFP4-specific EP sharding behavior."""

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if not is_op(node, torch.ops.auto_deploy.triton_mxfp4_moe):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_mxfp4_mlp_ep(
            gm, node, self.rank, self.world_size, self.allreduce_strategy, self.dist_backend
        )


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
        _insert_sharded_moe(
            gm,
            node,
            self.rank,
            self.world_size,
            self.allreduce_strategy,
            self.dist_backend,
            self.scale_names(),
        )


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
        _insert_sharded_moe(
            gm,
            node,
            self.rank,
            self.world_size,
            self.allreduce_strategy,
            self.dist_backend,
            self.scale_names(),
        )


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


class ShardingSource(Enum):
    """Enum for sharding source."""

    HEURISTIC = "heuristic"
    FACTORY = "factory"
    MANUAL = "manual"


class ShardingDim(Enum):
    """Enum for sharding dimension."""

    TP = "tp"
    EP = "ep"
    BMM = "bmm"


class DistBackend(Enum):
    """Enum for distributed backend."""

    AUTO = "auto"
    TRTLLM = "trtllm"
    TORCH = "torch"


class ShardingTransformContainer(BaseModel):
    """Configuration for sharding the model."""

    factory_source: ShardingConfigSource = Field(default=ShardingConfigSource.UNKNOWN)
    rank: int = Field(default=0)
    world_size: int = Field(default=1)
    factory_config: Dict[str, Any] = Field(default_factory=dict)
    manual_config: Dict[str, Any] = Field(default_factory=dict)
    simple_shard_only: bool = Field(default=False)
    support_partial_config: bool = Field(default=True)
    sharding_source: List[ShardingSource] = Field(
        default_factory=lambda: [ShardingSource.HEURISTIC]
    )
    sharding_dims: List[ShardingDim] = Field(
        default_factory=lambda: [ShardingDim.TP, ShardingDim.EP, ShardingDim.BMM]
    )
    allreduce_strategy: AllReduceStrategy = Field(
        default=AllReduceStrategy.AUTO,
        description="AllReduce strategy for distributed operations. "
        "Options: AUTO, NCCL, ONESHOT, TWOSHOT, MIN_LATENCY, LOWPRECISION, UB, MNNVL, NCCL_SYMMETRIC, SYMM_MEM",
    )
    dist_backend: DistBackend = Field(default=DistBackend.AUTO)
    weight_sharding_transforms: List[WeightShardingInfo] = Field(default_factory=list)
    parameter_update_transforms: List[ParameterUpdateInfo] = Field(default_factory=list)
    bmm_transforms: List[BMMShardingInfo] = Field(default_factory=list)
    ep_transforms: List[EPShardingInfo] = Field(default_factory=list)

    @field_validator("allreduce_strategy", mode="before")
    @classmethod
    def _validate_allreduce_strategy(cls, v):
        """Convert string names like 'AUTO' to AllReduceStrategy enum."""
        return validate_allreduce_strategy(v)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transform_list_dict = {
            WeightShardingInfo: self.weight_sharding_transforms,
            BMMShardingInfo: self.bmm_transforms,
            EPShardingInfo: self.ep_transforms,
            ParameterUpdateInfo: self.parameter_update_transforms,
        }

    def init_params(
        self, other: "ShardingTransformConfig", rank: int = None, world_size: int = None
    ) -> None:
        """
        Copy parameters from ShardingTransformConfig. The class is not
        imported here to avoid circular imports.
        """
        if rank is not None:
            self.rank = rank
        if world_size is not None:
            self.world_size = world_size
        self.factory_config = other.factory_config
        self.manual_config = other.manual_config
        self.simple_shard_only = other.simple_shard_only
        self.support_partial_config = other.support_partial_config
        self.sharding_dims = other.sharding_dims
        self.sharding_source = other.sharding_source
        # Extract factory_source from factory_config if present
        self.factory_source = self.factory_config.get("source", ShardingConfigSource.UNKNOWN)
        self.allreduce_strategy = other.allreduce_strategy
        self.dist_backend = other.dist_backend
        self.validate_config(ShardingSource.MANUAL)
        self.validate_config(ShardingSource.FACTORY)

    def add(self, transform: ShardingTransformInfo) -> bool:
        """Append a transform only if that node was
        not sharded before. Do not overwrite existing transforms.

        Automatically propagates allreduce_strategy and dist_backend from this config
        to the transform if the transform doesn't already have them set.
        """
        # Inject allreduce_strategy and dist_backend from config into transform
        # if they have the attributes and they're None
        # This creates a new transform instance with the values set
        needs_injection = False
        transform_dict = None

        if hasattr(transform, "allreduce_strategy") and transform.allreduce_strategy is None:
            if transform_dict is None:
                transform_dict = transform.model_dump()
            transform_dict["allreduce_strategy"] = self.allreduce_strategy
            needs_injection = True

        if hasattr(transform, "dist_backend") and transform.dist_backend is None:
            if transform_dict is None:
                transform_dict = transform.model_dump()
            transform_dict["dist_backend"] = self.dist_backend
            needs_injection = True

        if needs_injection:
            transform = type(transform)(**transform_dict)

        # Find the appropriate list by checking inheritance
        transform_list = None
        for base_class, transform_list_candidate in self._transform_list_dict.items():
            if isinstance(transform, base_class):
                transform_list = transform_list_candidate
                break

        if transform_list is None:
            raise ValueError(f"Unknown transform type: {type(transform)}")

        # Check if node already has a transform
        for existing_transform in transform_list:
            if existing_transform.target_node == transform.target_node:
                return False
        transform_list.append(transform)
        return True

    def validate_config(self, source: ShardingSource) -> bool:
        if (
            source == ShardingSource.FACTORY
            and self.factory_source != ShardingConfigSource.HUGGINGFACE
        ):
            ad_logger.debug(
                "Sharding config is currently only supported for HuggingFace. Skipping."
            )
            # invalidate the config
            self.factory_config.clear()
            return False

        config = self.manual_config if source == ShardingSource.MANUAL else self.factory_config

        if "head_dim" not in config:
            ad_logger.debug("Sharding config does not contain head_dim. Skipping.")
            # invalidate the config
            config.clear()
            return False

        if "tp_plan" not in config or config["tp_plan"] is None:
            ad_logger.debug("Sharding config does not contain tp_plan. Skipping.")
            # invalidate the config
            config.clear()
            return False
        tp_plan = config["tp_plan"]

        values = set(tp_plan.values())
        supported_modes = {
            "colwise",  # row split and no collective
            "rowwise",  # column split and all-reduce
            "mamba",  # mamba SSM layer
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
            ad_logger.debug("Sharding config contains invalid values. Skipping.")
            # invalidate the config
            config.clear()
            return False
        return True

    def get_factory_config(self) -> Dict[str, Any]:
        return self.factory_config

    def get_manual_config(self) -> Dict[str, Any]:
        return self.manual_config
