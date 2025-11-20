"""Transformations to support graph sharding.

Our sharding algorithm for tensor parallelism (TP) is based on the following steps:

    1. Initialize/construct unsharded model. Ideally, this should be done on device="meta" to avoid
       unnecessary memory allocation. In some cases, this is necessary if the model is too large to
       fit on a single device.
    2. Shard the graph IR of the model:
        a. Identify linear nodes that correspond to TP tuples.
        b. Reduce/Shard shape of weights in the corresponding linear nodes accordingly (either in
           row or column dimension). Add all_reduce nodes where necessary (--> only needed for
           fusing results in final linear node of the TP tuple).
        c. Add a checkpoint loading hook to the sharded linear nodes so that only the correct shard
           of the weight from the checkpoint gets loaded.
    3. Load the checkpoint and allocate the tensor. Loading the correct shard from the checkpoint
       happens automatically via the checkpoint loading hook added in step 2c.
"""

import math
import operator
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, IntEnum
from functools import partial
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

import torch
from pydantic import Field, field_validator
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch.fx import GraphModule, Node

from .....functional import AllReduceStrategy
from ...models.factory import ModelFactory, ShardingConfigSource
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    bfs,
    extract_param_names_from_node,
    extract_weight_node,
    filtered_nodes,
    get_all_layer_subgraphs,
    is_any_attention_op,
    is_any_lin_op,
    is_any_moe_op,
    is_any_ssm_op,
    is_op,
    num_users_of_weight_node,
    subgraph,
)
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


class ShardingSource(Enum):
    """Enum for sharding source."""

    HEURISTIC = "heuristic"
    FACTORY = "factory"
    MANUAL = "manual"


class ShardingDim(Enum):
    """Enum for sharding dimension."""

    SSM = "ssm"
    TP = "tp"
    EP = "ep"
    BMM = "bmm"


class SplitDimension(IntEnum):
    """Enum for tensor split dimensions in sharding."""

    # NOTE: The names COLUMN/ROW reflect the hugging face
    # base_tp_plan sharding notation, but since we assume Y = W @ X^T,
    # when splitting weight matrix W^T across columns, the actual split
    # is over dimension 0
    COLUMN = 0
    ROW = 1


class ShardingTransformConfig(TransformConfig):
    """Configuration for sharding the model."""

    factory_source: ShardingConfigSource = Field(default=ShardingConfigSource.UNKNOWN)
    factory_config: Dict[str, Any] = Field(default_factory=dict)
    manual_config: Dict[str, Any] = Field(default_factory=dict)
    simple_shard_only: bool = Field(default=False)
    support_partial_config: bool = Field(default=True)
    sharding_source: List[ShardingSource] = Field(
        default_factory=lambda: [
            ShardingSource.MANUAL,
            ShardingSource.FACTORY,
            ShardingSource.HEURISTIC,
        ]
    )
    sharding_dims: List[ShardingDim] = Field(
        default_factory=lambda: [ShardingDim.TP, ShardingDim.EP, ShardingDim.BMM]
    )
    allreduce_strategy: AllReduceStrategy = Field(
        default=AllReduceStrategy.AUTO,
        description="AllReduce strategy for distributed operations. "
        "Options: AUTO (automatic selection), NCCL, ONESHOT, TWOSHOT, MIN_LATENCY, "
        "LOWPRECISION, UB, MNNVL, NCCL_SYMMETRIC",
    )

    @field_validator("allreduce_strategy", mode="before")
    @classmethod
    def _validate_allreduce_strategy(cls, v):
        """Convert string names like 'AUTO' to AllReduceStrategy enum."""
        return validate_allreduce_strategy(v)
    process_grid: Dict[ShardingDim, int] = Field(default_factory=dict)

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
    for view_node in nodes_to_validate:
        if len(view_node.args) < 2:
            continue
        if "sharded" in view_node.meta and view_node.meta["sharded"]:
            continue
        view_shape = list(view_node.args[1])
        if not isinstance(view_shape, list):
            continue
        if len(view_shape) >= 3 and isinstance(view_shape[2], int) and view_shape[2] != -1:
            args = list(view_node.args)
            view_shape[2] = -1  # view_shape[2] // world_size
            args[1] = tuple(view_shape)
            view_node.args = tuple(args)
            view_node.meta["sharded"] = True
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
            # dim_d = t.shape[d]
            # num_parts = 1
            # part_size = dim_d // num_parts
            # fused_dims = [part_size] * num_parts
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

    # First, shard the entry_node (the first linear layer)
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
        add_dist=False,
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


def _shard_parameter_node(
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
    """Replace the node with parametrized weight tensor with a new node that accepts sharded weights.

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
        if is_any_lin_op(node):
            _validate_sharded_shapes(
                node, fused_weight_dims=fused_weight_dims, world_size=world_size
            )
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


def _update_node_args(node: Node, args: tuple) -> None:
    """Update the node's arguments with the new sharded arguments."""
    if "sharded" in node.meta and node.meta["sharded"]:
        return
    node.args = args
    node.meta["sharded"] = True
    ad_logger.debug(
        f"Updated node {node}: replaced original arguments {node.args} with sharded arguments {args}."
    )


class ShardingTransformInfo(BaseModel, ABC):
    """Abstract base class for transformation configurations."""

    model_config = ConfigDict(frozen=True)  # Makes the model immutable and hashable

    target_node: str
    config: ShardingTransformConfig

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
    """Configuration for TP sharding transformations."""

    split_dim: SplitDimension
    dist_op: Optional[Literal["all_reduce", "all_gather"]] = None
    min_local_shape: int = 1
    layer_type: LayerType = LayerType.MLP
    # used for TP sharding of fused weights
    fused_weight_dims: Optional[list] = None

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
        if self.layer_type == LayerType.MAMBA:
            _insert_sharded_mamba(
                gm=gm,
                entry_node=node,
                dim=self.split_dim.value,
                rank=self.config.rank,
                world_size=self.config.world_size,
                add_dist=self.dist_op is not None,
                min_local_shape=self.min_local_shape,
                fused_weight_dims=self.fused_weight_dims
                if isinstance(self.fused_weight_dims, dict)
                else None,
                quantization_cb=self.quantization_cb,
            )
        else:
            _shard_parameter_node(
                gm=gm,
                node=node,
                dim=self.split_dim.value,
                rank=self.config.rank,
                world_size=self.config.world_size,
                add_dist=self.dist_op is not None,
                min_local_shape=self.min_local_shape,
                fused_weight_dims=self.fused_weight_dims,
                quantization_cb=self.quantization_cb,
            )


class ParameterUpdateInfo(ShardingTransformInfo):
    """Configuration for node args sharding transformations."""

    target_node: str
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
        remainder = bmm_batch_size % self.config.world_size

        # NOTE: our torch.ops.auto_deploy.torch_dist_all_gather doesn't support uneven splits at the moment.
        if remainder:
            ad_logger.warning(
                f"BMM batch size {bmm_batch_size} is not divisible by world size {self.config.world_size}. "
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
    config: ShardingTransformConfig,
    scale_names: Sequence[str] = (),
):
    """Update the torch_moe node with sharded weight lists,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.
    """
    scale_names = list(scale_names)
    # get 2D EP+TP process grid and corresponding ranks
    ep_rank = config.process_grid[ShardingDim.EP]["p"]
    ep_size = config.process_grid[ShardingDim.EP]["w"]
    tp_rank = config.process_grid[ShardingDim.TP]["p"]
    tp_size = config.process_grid[ShardingDim.TP]["w"]
    print(f"ep_rank: {ep_rank}, ep_size: {ep_size}, tp_rank: {tp_rank}, tp_size: {tp_size}")
    num_experts = len(node.args[3])
    args = list(node.args)

    # -- Handle selected_experts and final_scales sharding --
    selected_experts = args[1]
    final_scales = args[2]

    experts_per_rank = num_experts // ep_size

    with gm.graph.inserting_before(node):
        lower = experts_per_rank * ep_rank
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
        comp_op = torch.ge if ep_rank == ep_size - 1 else torch.eq
        rank_mask = gm.graph.create_node(
            "call_function", comp_op, args=(div_node, ep_rank), kwargs={}
        )

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

    w1_list_sharded = get_partition(args[3], ep_size, ep_rank)
    w2_list_sharded = get_partition(args[4], ep_size, ep_rank)
    w3_list_sharded = get_partition(args[5], ep_size, ep_rank)

    # if tp_size > 1, we do 2D EP+TP sharding.
    # we add TP sharding of all expert weights.
    # for w1 in w1_list_sharded:
    #     shard_weight_tensor(
    #         gm=gm,
    #         weight_tensor=w1,
    #         param_key=weight_key,
    #         dim=ShardingDimension.COLUMN,
    #         rank=tp_rank,
    #         world_size=tp_size,
    #         min_local_shape=1,
    #     )

    # -- Update args --
    args[1] = selected_experts_local
    args[2] = final_scales_local
    args[3] = w1_list_sharded
    args[4] = w2_list_sharded
    args[5] = w3_list_sharded

    # Shard scales for quantized ops
    for i in range(len(scale_names) * 3):  # 3 layers (w1, w2, w3) × #scale_names per layer
        args[6 + i] = get_partition(args[6 + i], ep_size, ep_rank)

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
        _insert_sharded_moe(gm, node, self.config)


class MXFP4EPShardingInfo(EPShardingInfo):
    """GPT-OSS style MXFP4-specific EP sharding behavior."""

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if not is_op(node, torch.ops.auto_deploy.triton_mxfp4_moe):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        _insert_sharded_mxfp4_mlp_ep(gm, node, self.config)


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
            self.config,
            scale_names=self.scale_names(),
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
        _insert_sharded_moe(gm, node, self.config, self.scale_names())


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


class ShardingTransformContainer(BaseModel):
    """Configuration for sharding the model."""

    config: ShardingTransformConfig = Field(default_factory=ShardingTransformConfig)
    weight_sharding_transforms: List[WeightShardingInfo] = Field(default_factory=list)
    parameter_update_transforms: List[ParameterUpdateInfo] = Field(default_factory=list)
    bmm_transforms: List[BMMShardingInfo] = Field(default_factory=list)
    ep_transforms: List[EPShardingInfo] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transform_list_dict = {
            WeightShardingInfo: self.weight_sharding_transforms,
            BMMShardingInfo: self.bmm_transforms,
            EPShardingInfo: self.ep_transforms,
            ParameterUpdateInfo: self.parameter_update_transforms,
        }

    def add(self, transform: ShardingTransformInfo) -> bool:
        """Append a transform only if that node was
        not sharded before. Do not overwrite existing transforms.
        """
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

    dist_backend: DistBackend = Field(default=DistBackend.AUTO)


@TransformRegistry.register("sharding_transform_executor")
class ShardingTransformExecutor(BaseTransform):
    """Apply transformations to the graph module.

    Args:
        gm: Graph module to apply transformations to
        sharding_config: Transformation configuration containing list of transformations to apply
    """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ShardingTransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # create a node dict for faster lookup
        node_dict = {n.name: n for n in gm.graph.nodes}

        def check_and_apply(transform: ShardingTransformInfo) -> bool:
            """Return True if the transformation is applied, False otherwise."""
            if transform.target_node is None or transform.target_node not in node_dict:
                ad_logger.warning(
                    f"Skipping transformation {transform} because target node "
                    + f"{transform.target_node} not found in graph"
                )
                return False
            return transform.check_and_apply(gm, node_dict[transform.target_node])

        num_matches = 0
        transforms = shared_config.sharding_transform_container
        for tp_transform in transforms.weight_sharding_transforms:
            if check_and_apply(tp_transform):
                num_matches += 1
        for bmm_transform in transforms.bmm_transforms:
            if check_and_apply(bmm_transform):
                num_matches += 1
        for ep_transform in transforms.ep_transforms:
            if check_and_apply(ep_transform):
                num_matches += 1

        # post-sharding cleanup transformations
        for update_transform in transforms.parameter_update_transforms:
            if not check_and_apply(update_transform):
                ad_logger.warning(f"Invalid parameter update transformation {update_transform}.")

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


def _process_simple_shard(
    nodes_linear: Union[Dict[Node, List[Node]], List[Node]],
    rank: int,
    world_size: int,
    transform_container: ShardingTransformContainer,
    layer_type: LayerType = LayerType.MLP,
) -> int:
    # for every linear node:
    # --> row_split (dim 0 of weight) + all_gather (dim -1 of output)
    # if nodes_linear is a dict, flatten it to a 1D list of nodes

    if isinstance(nodes_linear, dict):
        nodes_linear = [n for group in nodes_linear.values() for n in group]

    num_simple_shards = 0
    for n in nodes_linear:
        num_simple_shards += int(
            transform_container.add(
                WeightShardingInfo.from_node(
                    n,
                    split_dim=SplitDimension.COLUMN,
                    rank=rank,
                    world_size=world_size,
                    dist_op="all_gather",
                    min_local_shape=1,
                    layer_type=layer_type,
                )
            )
        )
    return num_simple_shards


@TransformRegistry.register("detect_sharding")
class Sharding(BaseTransform):
    """A transformation to apply sharding to the model following tensor parallelism.

    The transformation is based on the following steps:

    1. Identify boundary nodes between residual nodes to identify shardable regions.
    2. Identify the GEMM nodes that can be sharded
    3. Trace through the subgraph using DFS/BFS between each pair of boundary nodes
    4. Account for each node in the trace to ensure the op is correct even after sharding. This is
       necessary to ensure that the sharding is correct and we need to be able to account for
       **all** nodes in the subgraph. The subgraph here is defined as the region between the first
       linear node to the last linear node of an identified sharding region.
    # 5. Shard the GEMM nodes or skip accordingly.

    min_local_shape is the minimum size of the local tensor shard, to prevent TP parallelism
    splitting, e.g., the individual heads into smaller shards.
    """

    config: ShardingTransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ShardingTransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        local_rank, world_size = shared_config.local_rank, shared_config.world_size
        if world_size < 2:
            ad_logger.info("Skipping sharding for single device")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        assert isinstance(gm, GraphModule), "Expecting GraphModule"
        self.config.factory_config = factory.get_sharding_config() if factory else {}
        transform_container = shared_config.sharding_transform_container
        transform_container.init_params(self.config, local_rank, world_size)

        info = TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
        for source in config.sharding_source:
            if source == ShardingSource.FACTORY:
                if len(config.factory_config) == 0:
                    ad_logger.debug(
                        "No factory config found. Skipping sharding from factory config"
                    )
                    continue
                ad_logger.info("Applying sharding from factory config")
                info += detect_sharding_from_config(gm, transform_container, ShardingSource.FACTORY)
            elif source == ShardingSource.MANUAL:
                if len(config.manual_config) == 0:
                    ad_logger.debug("No manual config found. Skipping sharding from manual config")
                    continue
                ad_logger.info("Applying sharding from manual config")
                info += detect_sharding_from_config(gm, transform_container, ShardingSource.MANUAL)

            elif source == ShardingSource.HEURISTIC:
                ad_logger.info(
                    f"Running autodeploy sharding heuristics: {transform_container.sharding_dims}"
                )
                # run TP sharding across ranks
                if ShardingDim.TP in config.sharding_dims:
                    info += detect_column_row_shard(gm, transform_container)

                # run EP sharding across ranks
                if ShardingDim.EP in config.sharding_dims:
                    info += detect_ep_shard(gm, transform_container)

                # run BMM sharding across ranks
                if ShardingDim.BMM in config.sharding_dims:
                    info += detect_dp_bmm_shard(gm, transform_container)

        return gm, info


def _process_ssm_sharding(
    gm: GraphModule,
    entry_node: Node,
    transform_container: ShardingTransformContainer,
    rank: int,
    world_size: int,
    min_local_shape: int = 1,
) -> int:
    """
    Process the SSM sharding from the candidate nodes and update the view and split nodes accordingly.
    """
    # Find next linear node to define subgraph boundary
    try:
        out_proj_node, _ = bfs(entry_node, is_any_lin_op, include_root=False)
    except RuntimeError:
        ad_logger.warning("Could not find next linear node after entry_node for Mamba sharding")
        return 0

    # Get subgraph between entry_node and next linear node
    subgraph_nodes = subgraph([entry_node], [out_proj_node])

    ##############################################################
    ########## infer split sizes for in_proj and conv1d ##########
    ##############################################################
    # in_proj and conv1d are fused, followed up by split nodes. Infer split sizes:
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
        return 0
    split_sizes_0 = split_nodes[0].args[1]
    split_sizes_1 = split_nodes[1].args[1]
    if split_sizes_0[1] != sum(split_sizes_1):
        ad_logger.warning(
            f"Split nodes have different sizes. "
            f"Skipping Mamba sharding. split_sizes_1={split_sizes_0}, split_sizes_2={split_sizes_1}"
        )
        return 0
    fused_weight_dims = {
        "in_proj": split_sizes_0[0:1] + split_sizes_1 + split_sizes_0[2:],
        "conv1d": split_sizes_1,
    }

    # ##############################################################
    # ####### shard the entry_node (the first linear layer) ########
    # ##############################################################
    if not transform_container.add(
        WeightShardingInfo.from_node(
            entry_node,
            split_dim=SplitDimension.COLUMN,
            rank=rank,
            world_size=world_size,
            dist_op=None,
            min_local_shape=min_local_shape,
            fused_weight_dims=fused_weight_dims["in_proj"],
            layer_type=LayerType.MAMBA,
        )
    ):
        # the layer was already sharded. Skipping.
        return 0

    # # ##############################################################
    # # ############## update split nodes ############################
    # # ##############################################################
    split_args_0 = list(split_nodes[0].args)
    split_args_0[1] = [s // world_size for s in split_args_0[1]]
    split_args_1 = list(split_nodes[1].args)
    split_args_1[1] = [s // world_size for s in split_args_1[1]]
    transform_container.add(
        ParameterUpdateInfo(
            config=transform_container.config,
            target_node=split_nodes[0].name,
            args=tuple(split_args_0),
        )
    )
    transform_container.add(
        ParameterUpdateInfo(
            config=transform_container.config,
            target_node=split_nodes[1].name,
            args=tuple(split_args_1),
        )
    )

    # ##############################################################
    # ############# update conv1d num output channels ##############
    # ##############################################################
    conv1d_nodes = [
        n for n in subgraph_nodes if is_op(n, [torch.ops.auto_deploy.torch_causal_conv1d])
    ]
    assert len(conv1d_nodes) == 1, "Expecting exactly one conv1d node"
    conv1d_node = conv1d_nodes[0]
    # conv1d_node last argument is the number of output channels.
    # This one is also sharded, so we need to update this parameter
    conv_args = list(conv1d_node.args)
    conv_args[-1] = conv1d_node.args[-1] // world_size
    transform_container.add(
        ParameterUpdateInfo(
            config=transform_container.config, target_node=conv1d_node.name, args=tuple(conv_args)
        )
    )

    # ##############################################################
    # ############## shard the remaining weights ###################
    # ##############################################################
    # # Get all weight nodes in the subgraph except for out_proj (it has to be row-sharded)
    weight_nodes = [
        n
        for n in get_all_weights_in_subgraph([entry_node], [out_proj_node])
        if "out_proj" not in str(n)
    ]
    for weight_node in weight_nodes:
        weight_key = weight_node.target
        # Get the weight parameter
        try:
            gm.get_parameter(weight_key)
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
        transform_container.add(
            WeightShardingInfo.from_node(
                list(weight_node.users)[0],
                split_dim=SplitDimension.COLUMN,
                config=transform_container.config,
                dist_op=None,
                min_local_shape=min_local_shape,
                fused_weight_dims=fused_dims,
                layer_type=LayerType.MAMBA,
            )
        )

    # ##############################################################
    # ############## update the view and reshape nodes #############
    # ##############################################################
    nodes_to_validate = [
        n for n in subgraph_nodes if is_op(n, [torch.ops.aten.view, torch.ops.aten.reshape])
    ]
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
            transform_container.add(
                ParameterUpdateInfo(
                    rank=rank, world_size=world_size, target_node=view_node.name, args=tuple(args)
                )
            )
            ad_logger.debug(f"\nUpdated view node {view_node} arguments to {view_node.args}")

    ##############################################################
    ############## shard the out_proj node #######################
    ##############################################################
    transform_container.add(
        WeightShardingInfo.from_node(
            out_proj_node,
            split_dim=SplitDimension.ROW,
            config=transform_container.config,
            dist_op="all_reduce",
            layer_type=LayerType.MAMBA,
        )
    )
    return 1


def _process_column_sharding(
    linear_nodes: List[Node],
    subgraph_nodes: Union[List[Node], None],
    transform_container: ShardingTransformContainer,
    rank: int,
    world_size: int,
    min_local_shape: int = 1,
) -> None:
    """
    Parse the column sharding from the candidate nodes and update the view and split nodes accordingly.
    """
    if subgraph_nodes is None:
        subgraph_nodes = subgraph(linear_nodes, boundary_condition=is_any_lin_op)
    fused_weight_dims = None
    # check if there are split nodes in the subgraph. They may indicate fused weights (e.g., QKV)
    split_nodes = list(filtered_nodes(subgraph_nodes, ops=[torch.ops.aten.split_with_sizes]))
    if len(split_nodes) > 0:
        assert len(linear_nodes) == 1
        linear_node = linear_nodes[0]
        assert len(split_nodes) == 1, "Expecting exactly one split node for fused weights"
        fused_weight_dims = split_nodes[0].args[1]
    slice_nodes = list(filtered_nodes(subgraph_nodes, ops=[torch.ops.aten.slice]))
    if len(slice_nodes) > 0:
        # we are probably in fused QKV case with single linear node and 3 slice nodes
        assert len(linear_nodes) == 1
        linear_node = linear_nodes[0]
        assert all(
            s.args[1] == 2 for s in filtered_nodes(linear_node.users, ops=torch.ops.aten.slice)
        ), "Expecting slice nodes to slice tensor over dim=2"
        fused_weight_dims = [s.args[3] - s.args[2] for s in linear_node.users]
        weight_dim = linear_node.meta["val"].shape[2]
        if sum(fused_weight_dims) != weight_dim:
            if fused_weight_dims[-1] > weight_dim:
                fused_weight_dims[-1] = weight_dim - sum(fused_weight_dims[:-1])
            else:
                ad_logger.warning(
                    f"Fused weight dims {fused_weight_dims} do not sum to weight dim {weight_dim}. Skipping."
                )
                return
    chunk_nodes = list(filtered_nodes(subgraph_nodes, ops=torch.ops.aten.chunk))
    if len(chunk_nodes) > 0:
        assert len(linear_nodes) == 1
        linear_node = linear_nodes[0]
        assert len(chunk_nodes) == 1, "Expecting exactly one chunk node for fused weights"
        num_chunks = chunk_nodes[0].args[1]
        weight_dim = linear_node.meta["val"].shape[2]
        fused_weight_dims = [weight_dim // num_chunks] * num_chunks

    # check if there are any attention nodes in the subgraph
    attention_nodes = list(filtered_nodes(subgraph_nodes, is_any_attention_op))

    added_nodes: int = 0
    for linear_node in linear_nodes:
        added_nodes += transform_container.add(
            WeightShardingInfo.from_node(
                linear_node,
                split_dim=SplitDimension.COLUMN,
                config=transform_container.config,
                dist_op=None,  # for column sharding, no dist op is performed
                min_local_shape=min_local_shape,
                fused_weight_dims=fused_weight_dims,
                layer_type=LayerType.ATTENTION if len(attention_nodes) > 0 else LayerType.MLP,
            )
        )
    if added_nodes == 0:
        ad_logger.debug("No nodes were added for column sharding. Skipping.")
        return

    nodes_to_validate = [
        n for n in subgraph_nodes if is_op(n, [torch.ops.aten.view, torch.ops.aten.reshape])
    ]
    for view_node in nodes_to_validate:
        if len(view_node.args) < 2:
            continue
        view_shape = list(view_node.args[1])
        if not isinstance(view_shape, list):
            continue
        if len(view_shape) >= 3 and isinstance(view_shape[2], int) and view_shape[2] != -1:
            args = list(view_node.args)
            view_shape[2] = -1
            args[1] = tuple(view_shape)
            transform_container.add(
                ParameterUpdateInfo(
                    config=transform_container.config, target_node=view_node.name, args=tuple(args)
                )
            )
            ad_logger.debug(f"\nUpdated view node {view_node} arguments to {view_node.args}")

    # if fused_weight_dims is provided, we need to update all split sizes
    if fused_weight_dims is not None:
        assert world_size is not None, "World size is required to update the split node params"

        # fused weight may either be processed by several slice nodes or a single split node
        assert len(split_nodes) > 0 or len(slice_nodes) > 0 or len(chunk_nodes) > 0, (
            "Expecting at least one split or slice or chunk node for fused weights"
        )

        assert len(linear_nodes) == 1, "Expecting exactly one linear node for fused weights"
        linear_node = linear_nodes[0]
        if len(split_nodes) > 0:
            user = split_nodes[0]
            orig_sizes = user.args[1]
            new_sizes = [orig_sizes[i] // world_size for i in range(len(orig_sizes))]
            args = list(user.args)
            args[1] = new_sizes
            transform_container.add(
                ParameterUpdateInfo(
                    config=transform_container.config, target_node=user.name, args=tuple(args)
                )
            )
        elif len(slice_nodes) > 0:
            for slice_node in filtered_nodes(linear_node.users, ops=torch.ops.aten.slice):
                args = list(slice_node.args)
                args[2] = args[2] // world_size
                args[3] = args[3] // world_size
                transform_container.add(
                    ParameterUpdateInfo(
                        rank=rank,
                        world_size=world_size,
                        target_node=slice_node.name,
                        args=tuple(args),
                    )
                )
        # chunk nodes do not need to be updated


def detect_sharding_from_config(
    gm: GraphModule,
    transform_container: ShardingTransformContainer,
    source: ShardingSource,
) -> TransformInfo:
    """
    Create sharding transformations from the predefined config.
    TODO: currently, it applies only to TP sharding.
    Args:
        gm: Graph module to apply transformations to
        transform_container: Container for sharding transformations
        source: Sharding source
    """
    config = transform_container.config
    # check if config is valid.
    # 1. it is a Dict[str, str]
    # 2. the keys are of format "module.submodule.subsubmodule..."
    # 3. the wildcard "*" is allowed in the keys
    # 4. the allowed values are:
    #   - "colwise"
    #   - "rowwise"
    #   - "mamba"
    #   - "sequence_parallel"
    #   - "local_colwise"
    #   - "local_rowwise"
    #   - "local"
    #   - "gather"
    # The following constraints are based on
    # https://github.com/huggingface/transformers/blob/d8e05951b8efd4880acca9a3f291e8b65841a86d/src/transformers/models/llama4/configuration_llama4.py#L249
    if source == ShardingSource.FACTORY:
        config = transform_container.get_factory_config()
    elif source == ShardingSource.MANUAL:
        config = transform_container.get_manual_config()
    else:
        raise ValueError(f"Unsupported sharding source: {source}")

    head_dim = config["head_dim"]
    tp_plan = config["tp_plan"]

    rank, world_size = transform_container.rank, transform_container.world_size

    # If the node is inside the attention module, we need to set min_local_shape to the
    # head_dim - otherwise, we would risk splitting the heads into smaller shards.
    # TODO: is there a better way to check if we are in attention module?
    attn_names = [
        "attention",
        "Attention",
        "attn",
        "Attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    num_shards = 0
    num_simple_shards = 0
    num_row_col_shards = 0
    num_attention_shards = 0
    num_ssm_shards = 0

    for lin_node in filtered_nodes(gm.graph.nodes, is_any_lin_op):
        # use node's weight name to get the module name
        module_name = extract_weight_node(lin_node).target

        if any(attn_name in module_name for attn_name in attn_names):
            min_local_shape = head_dim
            layer_type = LayerType.ATTENTION
        else:
            min_local_shape = 1
            layer_type = LayerType.MLP

        # use regex to find if module_name matches any of the keys in sharding_config
        for key in tp_plan.keys():
            pattern_string = "*" + key + "*"
            # convert it to regex. Escape dots, replace * with .*
            # First, we substitute * with an unlikely character, e.g. @
            # Then we escape dots, and finally we replace @ with .*
            pattern_string = pattern_string.replace("*", "@")
            pattern_regex = re.escape(pattern_string).replace("@", ".*")
            if re.match(pattern_regex, module_name):
                # we have a match. Get the config for this layer

                config = tp_plan[key]
                if config == "colwise":
                    _process_column_sharding(
                        linear_nodes=[lin_node],
                        subgraph_nodes=None,
                        transform_container=transform_container,
                        rank=rank,
                        world_size=world_size,
                        min_local_shape=min_local_shape,
                    )
                elif config == "rowwise":
                    if transform_container.add(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.ROW,
                            config=transform_container.config,
                            dist_op="all_reduce",
                            min_local_shape=min_local_shape,
                            layer_type=layer_type,
                        )
                    ):
                        if layer_type == LayerType.ATTENTION:
                            num_attention_shards += 1
                        num_row_col_shards += 1
                elif config == "mamba":
                    if (
                        _process_ssm_sharding(gm, lin_node, transform_container, rank, world_size)
                        > 0
                    ):
                        num_ssm_shards += 1
                        num_row_col_shards += 1

                elif "sequence" in config:
                    transform_container.add(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            config=transform_container.config,
                            dist_op=None,
                            min_local_shape=min_local_shape,
                            layer_type=LayerType.MAMBA,
                        )
                    )
                    num_row_col_shards += 1
                elif "sequence" in tp_config:
                    # TODO: Sequence parallelism is not supported yet.
                    ad_logger.warning("Sequence parallelism is not supported yet. Skipping.")
                elif "local" in tp_config:
                    # Check if this applies to shared experts in EP parallelism.
                    # If yes, apply the TP col-row shard.
                    if "shared" in module_name:
                        col_row_action = tp_config.replace("local_", "")
                        if col_row_action == "colwise":
                            transform_container.add(
                                WeightShardingInfo.from_node(
                                    lin_node,
                                    split_dim=SplitDimension.COLUMN,
                                    config=transform_container.config,
                                    dist_op=None,
                                    min_local_shape=min_local_shape,
                                    layer_type=layer_type,
                                )
                            )
                        elif col_row_action == "rowwise":
                            if transform_container.add(
                                WeightShardingInfo.from_node(
                                    lin_node,
                                    split_dim=SplitDimension.ROW,
                                    config=transform_container.config,
                                    dist_op="all_reduce",
                                    min_local_shape=min_local_shape,
                                    layer_type=layer_type,
                                )
                            ):
                                num_row_col_shards += 1
                        else:
                            ad_logger.warning(f"Unsupported sharding action {tp_config}. Skipping.")
                    else:
                        # TODO: local refers to hybrid EP+TP parallelism. Not supported yet.
                        ad_logger.warning("Local EP+TP sharding is not supported yet. Skipping.")

                elif "gather" in tp_config:
                    # Simple shard (row + all_gather)
                    if transform_container.add(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            config=transform_container.config,
                            dist_op="all_gather",
                            min_local_shape=1,
                            layer_type=layer_type,
                        )
                    ):
                        num_simple_shards += 1
                else:
                    ad_logger.debug(
                        f"Unsupported sharding action {config}. "
                        f"Linear node {lin_node} will not be sharded."
                    )
                # after successful match, break the loop
                break

    num_shards = num_simple_shards + num_row_col_shards
    ad_logger.info(
        f"Applied {num_shards} TP shards from config. Simple: {num_simple_shards}, "
        f"row-col: {num_row_col_shards} (including: ssm: {num_ssm_shards}, attention: {num_attention_shards})"
    )

    num_matches = len(transform_container.weight_sharding_transforms)

    return TransformInfo(
        skipped=False,
        num_matches=num_matches,
        is_clean=False,
        has_valid_shapes=False,
    )


def detect_ssm_shard(
    gm: GraphModule,
    transform_container: ShardingTransformContainer,
) -> TransformInfo:
    """A transformation to apply sharding to the model following SSM parallelism.
    TODO: This is a TEMPORARY place for this logic due to the incompatibility between the
    identify_regions_between_residuals() and subgraph() methods to detect layers.
    The goal is to have a unified single pass over the graph to detect layers and apply
    appropriate sharding transformations.
    """
    config = transform_container.config
    rank, world_size = config.rank, config.world_size
    if world_size < 2:
        ad_logger.info("Skipping TP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
    ad_logger.info("Running SSM sharding detection")

    # find all ssm nodes in the graph
    ssm_nodes = filtered_nodes(gm.graph.nodes, is_any_ssm_op)
    num_ssm_shards = 0
    for ssm_node in ssm_nodes:
        # We assume that one ssm node defines a subgraph corresponding
        # to a single Mamba layer.
        # Find defining previous (in_proj) and next (out_proj) linear nodes.
        in_proj_node, _ = bfs(ssm_node, is_any_lin_op, attr_next="args", include_root=False)

        num_ssm_shards += int(
            _process_ssm_sharding(gm, in_proj_node, transform_container, rank, world_size)
        )

    ad_logger.info(f"Found {num_ssm_shards} SSM shards")
    return TransformInfo(
        skipped=False, num_matches=num_ssm_shards, is_clean=False, has_valid_shapes=False
    )


def detect_column_row_shard(
    gm: GraphModule,
    transfrom_container: ShardingTransformContainer,
) -> TransformInfo:
    """A transformation to apply sharding to the model following tensor parallelism.

    The transformation is based on the following steps:

    1. Identify boundary nodes between residual nodes to identify shardable regions.
    2. Identify the GEMM nodes that can be sharded
    3. Trace through the subgraph using DFS/BFS between each pair of boundary nodes
    4. Account for each node in the trace to ensure the op is correct even after sharding. This is
       necessary to ensure that the sharding is correct and we need to be able to account for
       **all** nodes in the subgraph. The subgraph here is defined as the region between the first
       linear node to the last linear node of an identified sharding region.
    # 5. Shard the GEMM nodes or skip accordingly.

    min_local_shape is the minimum size of the local tensor shard, to prevent TP parallelism
    splitting, e.g., the individual heads into smaller shards.
    """
    ad_logger.debug("Before sharding graph: " + str(gm))
    config = transform_container.config
    rank, world_size = config.rank, config.world_size
    if world_size < 2:
        ad_logger.info("Skipping TP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    assert isinstance(gm, GraphModule), "Expecting GraphModule"
    ad_logger.info("Running TP sharding detection")
    linear_nodes = list(filtered_nodes(gm.graph.nodes, is_any_lin_op))
    if len(linear_nodes) == 0:
        ad_logger.warning("Could not find any linear nodes in the graph. Skipping TP sharding.")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)

    num_shards = 0
    num_simple_shards = 0
    num_ssm_shards = 0
    num_attention_shards = 0
    num_column_row_shards = 0
    for opening, layer_subgraph, closing in layer_subgraphs:
        nodes_linear = opening + [closing]
        num_shards += 1

        ssm_nodes = list(filtered_nodes(layer_subgraph, is_any_ssm_op))
        attention_nodes = list(filtered_nodes(layer_subgraph, is_any_attention_op))
        min_local_shape = 1
        layer_type = (
            LayerType.MAMBA
            if len(ssm_nodes) > 0
            else LayerType.ATTENTION
            if len(attention_nodes) > 0
            else LayerType.MLP
        )

        if transfrom_container.simple_shard_only:
            ad_logger.debug(
                f"Forcing Simple Shard on nodes: {nodes_linear} with layer type: {layer_type}"
            )
            num_simple_shards += _process_simple_shard(
                nodes_linear, rank, world_size, transfrom_container, layer_type=layer_type
            )
            continue

        if len(ssm_nodes) > 0:
            # Mamba layers need special handling due to the fused weights for in_proj and conv1d
            assert len(ssm_nodes) == 1, "Expected exactly one SSM node in layer subgraph"
            assert len(opening) == 1, "Expected exactly one opening node in Mamba layer"
            ad_logger.debug(f"Found SSM nodes in layer subgraph: {ssm_nodes}")
            num_ssm_shards += _process_ssm_sharding(
                gm, opening[0], transfrom_container, rank, world_size
            )
            continue

        if len(attention_nodes) > 0:
            ad_logger.debug(f"Found attention nodes in layer subgraph: {attention_nodes}")
            if len(attention_nodes) > 1:
                # Column-row shard boundary region detection is probably wrong - there should be
                # only one attention operation. Fall back to simple shard.
                ad_logger.debug(f"More than one attention node: {attention_nodes}")
                num_simple_shards += _process_simple_shard(
                    nodes_linear, rank, world_size, transfrom_container, layer_type=layer_type
                )
                continue
            # Extract head dimension. We cannot shard below the head_dim size.
            # Assume that head_dim is the last (innermost) dimension of the tensor
            min_local_shape = attention_nodes.pop().meta["val"].shape[-1]
            # if the QKV projection is fused, check if num_kv_heads is divisible by world_size
            if len(opening) == 1:
                qkv_proj_node = opening[0]
                slice_nodes = list(filtered_nodes(qkv_proj_node.users, ops=torch.ops.aten.slice))
                if len(slice_nodes) > 0:
                    # extract num_kv_heads * head_dim from the second slice node
                    assert len(slice_nodes) == 3, "Expecting exactly 3 slice nodes for fused QKV"
                    num_kv_heads = (
                        slice_nodes[1].args[3] - slice_nodes[1].args[2]
                    ) // min_local_shape
                    if num_kv_heads % world_size != 0:
                        ad_logger.debug(
                            f"num_kv_heads {num_kv_heads} is not divisible by world_size {world_size}. "
                            f"Falling back to simple shard."
                        )
                        num_simple_shards += _process_simple_shard(
                            nodes_linear,
                            rank,
                            world_size,
                            transfrom_container,
                            layer_type=layer_type,
                        )
                        # TODO: handle the case where num_kv_heads is not divisible by world_size
                        continue

        # column-row sharding
        _process_column_sharding(
            linear_nodes=opening,
            subgraph_nodes=layer_subgraph,
            transform_container=transfrom_container,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
        )

        # shard single row node
        if transfrom_container.add(
            WeightShardingInfo.from_node(
                closing,
                split_dim=SplitDimension.ROW,
                config=transform_container.config,
                dist_op="all_reduce",
                min_local_shape=min_local_shape,
                layer_type=layer_type,
            )
        ):
            num_column_row_shards += 1
            if layer_type == LayerType.ATTENTION:
                num_attention_shards += 1

    # simple shard remaining linear nodes
    num_simple_shards += _process_simple_shard(
        unprocessed_linear_nodes, rank, world_size, transfrom_container
    )
    num_column_row_shards += num_ssm_shards
    ad_logger.info(
        f"Heuristics found {num_shards} TP shards. Simple: {num_simple_shards}, "
        f"row-col: {num_column_row_shards} (including: ssm: {num_ssm_shards}, attention: {num_attention_shards})"
    )
    return TransformInfo(
        skipped=False, num_matches=num_shards, is_clean=False, has_valid_shapes=False
    )


def detect_dp_bmm_shard(
    gm: GraphModule, transform_container: ShardingTransformContainer
) -> TransformInfo:
    """A transformation to apply sharding to batched matrix multiplications in the graph.

    We'll shard the BMM nodes by slicing the batch dimension of input tensors into world_size number of slices.
    After sharding each BMM node, we'll insert an all_gather node to gather the results across the different devices.
    This transformation handles any combination of tensor types for both inputs to the BMM operation.

    We'll also assume that the inputs to BMM are broadcasted across the devices already.
    """
    ad_logger.debug("Before sharding graph: " + str(gm))
    rank, world_size = transform_container.rank, transform_container.world_size
    if world_size < 2:
        ad_logger.info("Skipping DP BMM sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    num_bmm_shards = 0

    for node in gm.graph.nodes:
        if not is_op(node, {torch.ops.aten.bmm}):
            continue

        ad_logger.debug(f"Found BMM node: {node}")

        # Get the input tensors
        lhs_tensor = node.args[0]
        rhs_tensor = node.args[1]

        # Check batch sizes from meta information
        lhs_batch_size = lhs_tensor.meta["val"].shape[0]
        rhs_batch_size = rhs_tensor.meta["val"].shape[0]

        assert lhs_batch_size == rhs_batch_size, "Batch sizes of both tensors must match"
        bmm_batch_size = lhs_batch_size

        # Calculate balanced distribution
        base_size = bmm_batch_size // world_size
        remainder = bmm_batch_size % world_size

        # NOTE: our torch.ops.auto_deploy.torch_dist_all_gather/trtllm_dist_all_gather
        #  doesn't support uneven splits at the moment.
        if remainder:
            ad_logger.warning(
                f"BMM batch size {bmm_batch_size} is not divisible by world size {world_size}. "
                f"This will result in uneven distribution of work across devices. Skipping."
            )
            continue

        # Calculate start and end indices for this rank
        if rank < remainder:
            start_idx = rank * (base_size + 1)
            end_idx = start_idx + base_size + 1
        else:
            start_idx = remainder + rank * base_size
            end_idx = start_idx + base_size

        transform_container.add(
            BMMShardingInfo(
                start_idx=start_idx,
                end_idx=end_idx,
                target_node=node.name,
                config=transform_container.config,
            )
        )
        ad_logger.debug(
            f"Sharding BMM for rank {rank}: batch_size={bmm_batch_size}, start_idx={start_idx}, end_idx={end_idx}"
        )

        num_bmm_shards += 1

    ad_logger.debug("After sharding BMM: " + str(gm))
    ad_logger.info(f"Found {num_bmm_shards} BMM shards")

    return TransformInfo(
        skipped=False, num_matches=num_bmm_shards, is_clean=False, has_valid_shapes=False
    )


def get_process_grid_from_config(config: ShardingTransformConfig) -> Tuple[int, int]:
    rank, world_size = config.rank, config.world_size
    if len(config.process_grid) > 0:
        ad_logger.info(f"EP + TP sharding process grid: {config.process_grid}")
        ep_size = config.process_grid[ShardingDim.EP]
        tp_size = config.process_grid[ShardingDim.TP]
        # the order of the keys (ep,tp) vs (tp,ep) determines how ranks
        # are mapped to the 2D process grid
        if list(config.process_grid.keys())[-1] == ShardingDim.TP:
            tp_rank = rank % tp_size
            ep_rank = rank // tp_size
        else:
            tp_rank = rank // ep_size
            ep_rank = rank % ep_size

        if ep_size * tp_size != world_size:
            ad_logger.warning(
                f"EP + TP sharding process grid {config.process_grid} "
                f"does not match world size {world_size}. "
                f"Skipping 2D sharding, applying only 1D EP sharding."
            )
            return None
    else:
        ep_size = world_size
        tp_size = 1
        ep_rank = rank
        tp_rank = 0
    process_grid = {
        ShardingDim.EP: {"p": ep_rank, "w": ep_size},
        ShardingDim.TP: {"p": tp_rank, "w": tp_size},
    }
    return process_grid


def detect_ep_shard(
    gm: GraphModule, transform_container: ShardingTransformContainer
) -> TransformInfo:
    ad_logger.debug("Before sharding graph: " + str(gm))

    rank, world_size = transform_container.rank, transform_container.world_size
    if world_size < 2:
        ad_logger.info("Skipping EP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    assert isinstance(gm, GraphModule), "Expecting GraphModule"
    num_moe_patterns = 0
    for node in list(gm.graph.nodes):
        if not is_any_moe_op(node):
            continue
        if transform_container.add(
            EPShardingInfo.from_node(
                node,
                config=transform_container.config,
            )
        ):
            num_moe_patterns += 1

    ad_logger.info(f"Found {num_moe_patterns} MoE patterns")

    return TransformInfo(
        skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
    )
