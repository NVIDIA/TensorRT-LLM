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
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum
from functools import partial
from typing import Callable, DefaultDict, Dict, List, Literal, Optional, Set

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import (
    extract_param_names_from_lin_node,
    identify_regions_between_residuals,
    is_linear_op,
    is_op,
    num_users_of_weight_node,
)
from ...utils.quantization_utils import QuantizationImpl
from .._graph import canonicalize_graph


class SplitDimension(IntEnum):
    """Enum for tensor split dimensions in sharding."""

    ROW = 0  # Split along rows (first dimension)
    COLUMN = 1  # Split along columns (second dimension)


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

    def check_and_apply(self, gm: GraphModule, node: Node) -> None:
        """Check if the transformation is valid and apply it if it is."""
        if not self.validate(gm, node):
            ad_logger.warning(f"Skipping invalid transformation {self}.")
            return
        self.apply(gm, node)


class TPShardingInfo(ShardingTransformInfo):
    """Configuration for TP sharding transformations."""

    split_dim: SplitDimension
    dist_op: Optional[Literal["all_reduce", "all_gather"]] = None
    min_local_shape: int = 1

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if self.dist_op is not None:
            if self.split_dim == SplitDimension.ROW:
                if self.dist_op == "all_gather":
                    ad_logger.warning(
                        f"Row split is only supported for all_gather. Skipping {self}."
                    )
                    return False
            if self.split_dim == SplitDimension.COLUMN:
                if self.dist_op == "all_reduce":
                    ad_logger.warning(
                        f"Column split is only supported for all_reduce. Skipping {self}."
                    )
                    return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply TP sharding transformation to the graph module."""

        _insert_sharded_matmul(
            gm=gm,
            node=node,
            dim=self.split_dim.value,
            rank=self.rank,
            world_size=self.world_size,
            add_dist=self.dist_op is not None,
            min_local_shape=self.min_local_shape,
        )


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
                torch.ops.auto_deploy.torch_dist_all_gather,
                args=(node, 0),  # Gather along batch dimension (0)
            )
            node.replace_all_uses_with(gather_node)
            gather_node.replace_input_with(gather_node, node)


class EPShardingInfo(ShardingTransformInfo):
    """Configuration for EP sharding transformations."""

    rank: int
    world_size: int

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate the transformation configuration."""
        if not is_op(
            node,
            (
                torch.ops.auto_deploy.torch_moe,
                torch.ops.auto_deploy.torch_quant_fp8_moe,
                torch.ops.auto_deploy.torch_quant_fp4_moe,
            ),
        ):
            ad_logger.warning(f"EP sharding is only supported for MOE nodes. Skipping {self}.")
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Apply EP sharding transformation to the graph module."""
        _insert_sharded_moe(gm, node, self.rank, self.world_size)


class ShardingConfig(BaseModel):
    """Configuration for sharding the model."""

    tp_transforms: List[TPShardingInfo] = Field(default_factory=list)
    bmm_transforms: List[BMMShardingInfo] = Field(default_factory=list)
    ep_transforms: List[EPShardingInfo] = Field(default_factory=list)


def sharding_transform_executor(gm: GraphModule, sharding_config: ShardingConfig) -> None:
    """Apply transformations to the graph module.

    Args:
        gm: Graph module to apply transformations to
        sharding_config: Transformation configuration containing list of transformations to apply
    """
    # create a node dict for faster lookup
    node_dict = {n.name: n for n in gm.graph.nodes}

    def check_and_apply(transform: ShardingTransformInfo) -> None:
        if transform.target_node is None or transform.target_node not in node_dict:
            ad_logger.warning(
                f"Skipping transformation {transform} because target node "
                + f"{transform.target_node} not found in graph"
            )
            return
        transform.check_and_apply(gm, node_dict[transform.target_node])

    for tp_transform in sharding_config.tp_transforms:
        check_and_apply(tp_transform)
    for bmm_transform in sharding_config.bmm_transforms:
        check_and_apply(bmm_transform)
    for ep_transform in sharding_config.ep_transforms:
        check_and_apply(ep_transform)

    # canonicalize and return
    gm = canonicalize_graph(gm)
    ad_logger.debug("After applying sharding transformations: " + str(gm))


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


def _insert_sharded_matmul(
    gm: GraphModule,
    node: Node,
    dim: int,
    rank: int,
    world_size: int,
    add_dist: bool = False,
    min_local_shape: int = 1,
) -> None:
    """Replace the matmul node with a new matmul node that accepts sharded weights.

    The state_dict is also updated to contain the sharded weights.
    """
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    quantization_impl = QuantizationImpl.create(node)

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

    def set_new_param(submod: nn.Module, param_key: str, remove: bool = False) -> torch.Size:
        # split or remove it
        param_new = (
            None
            if remove
            else nn.Parameter(
                split_tensor(gm.get_parameter(param_key)).detach().clone(),
                requires_grad=quantization_impl is None,
            )
        )

        # update the parameter
        param_name = param_key.rpartition(".")[-1]
        setattr(submod, param_name, param_new)
        return torch.Size() if param_new is None else param_new.shape

    # update weight
    weight_new_shape = set_new_param(submod, weight_key)
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook, f_split=split_tensor, param_key=weight_key, param_shape=weight_new_shape
        )
    )

    if bias_key is not None and dim == 0:
        # update bias for dim 0 --> we can handle it like the weight
        bias_new_shape = set_new_param(submod, bias_key)
        gm._register_load_state_dict_pre_hook(
            partial(
                _load_hook, f_split=split_tensor, param_key=bias_key, param_shape=bias_new_shape
            )
        )
    elif bias_key is not None and rank != world_size - 1:
        # update the bias for dim 1 --> in this case only the last rank gets the bias to avoid
        # double counting it. For all other we will delete the bias.
        args = list(node.args)
        node_bias = args[2]
        args[2] = None
        node.args = tuple(args)
        gm.graph.erase_node(node_bias)
        set_new_param(submod, bias_key, remove=True)
        gm._register_load_state_dict_pre_hook(partial(_load_hook_remove, param_key=bias_key))

    if quantization_impl:
        scales = {}
        for scale_name in quantization_impl.scale_names():
            scales[scale_name] = submod.get_buffer(scale_name)
        scales["weight_shape"] = weight_new_shape
        sharded_scales = quantization_impl.shard_scales(dim, rank, world_size, **scales)
        for k, v in sharded_scales.items():
            submod.register_buffer(k, v)

        gm._register_load_state_dict_pre_hook(
            partial(
                quantization_impl.shard_load_hook,
                weight_name=weight_key,
                weight_shape=weight_new_shape,
                dim=dim,
                rank=rank,
                world_size=world_size,
            )
        )

    # no comm node needed for single device
    if not add_dist:
        return

    # figure out the right dist op
    dist_lookup = {
        0: (torch.ops.auto_deploy.torch_dist_all_gather, -1),
        1: (torch.ops.auto_deploy.torch_dist_all_reduce,),
    }
    fn_dist, *dist_args = dist_lookup[dim]

    # add reduction node
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(fn_dist, args=(node, *dist_args))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _append_simple_shard(
    nodes_linear: Dict[Node, List[Node]],
    rank: int,
    world_size: int,
    sharding_config: ShardingConfig,
) -> None:
    # for every linear node:
    # --> row_split (dim 0 of weight) + all_gather (dim -1 of output)
    tp_shards: List[TPShardingInfo] = []
    for node_group in nodes_linear.values():
        for n in node_group:
            tp_shards.append(
                TPShardingInfo(
                    target_node=n.name,
                    split_dim=SplitDimension.ROW,
                    rank=rank,
                    world_size=world_size,
                    dist_op="all_gather",
                    min_local_shape=1,
                )
            )
    sharding_config.tp_transforms.extend(tp_shards)


def detect_column_row_shard(
    gm: GraphModule,
    rank: int,
    world_size: int,
    sharding_config: ShardingConfig,
    simple_shard_only: bool = False,
) -> None:
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

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    # find boundary nodes of regions we want to shard
    boundary_nodes = identify_regions_between_residuals(gm)

    # TODO: continue updating these lists
    # pointwise ops that don't affect the sharder
    pointwise_ops = {
        torch.ops.aten.gelu,
        torch.ops.aten.leaky_relu,
        torch.ops.aten.mul,
        torch.ops.aten.relu,
        torch.ops.aten.sigmoid,
        torch.ops.aten.silu,
        torch.ops.aten.tanh,
        torch.ops.aten.contiguous,
    }

    # acceptable attention nodes between sharded GEMMs
    shardable_attention_nodes = {
        torch.ops.auto_deploy.torch_attention_sdpa,
        torch.ops.auto_deploy.torch_attention_grouped_sdpa,
        torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa,
    }

    # This is a heuristic. Basically, we assume those are okay to shard if we also encounter an
    # attention node because we know that those ops must be compatible with the attention op. Now
    # since the attention op is shardable, we will assume those are as well if used in conjunction
    # with the attention op.
    shardable_nodes_with_attention = {
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.auto_deploy.flashinfer_rope,
        operator.getitem,
    }

    # let's look at linear nodes we can identify between pairs of boundary nodes
    # There is three potential cases we can handle:
    # 1. No linear nodes:
    #       --> just continue
    # 2. Two groups of linear nodes and we can account for all to the view nodes:
    #       --> row_split (dim 0) 1st group + check for supported nodes +
    #           col_split (dim 1) 2nd group + all_reduce output of 2nd group
    # 3. Linear nodes that are not in two groups or we cannot account for all nodes:
    #       --> row_split (dim 0 of weight) + all_gather (dim -1 of output) output
    num_shards = 0
    for n_start, n_end in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        # we iterate through all nodes between the two boundary nodes and store linear nodes
        # sorted by their input activation node. We also store remaining nodes.
        nodes_linear: DefaultDict[Node, List[Node]] = defaultdict(list)
        attention_nodes: Set[Node] = set()
        attention_related_nodes: Set[Node] = set()
        unaccounted_nodes: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            if is_linear_op(current_node, include_quantization=True):
                nodes_linear[current_node.args[0]].append(current_node)
            elif is_op(current_node, shardable_attention_nodes):
                attention_nodes.add(current_node)
            elif is_op(current_node, shardable_nodes_with_attention):
                attention_related_nodes.add(current_node)
            elif not is_op(current_node, pointwise_ops):
                unaccounted_nodes.add(current_node)
            current_node = current_node.next
            assert current_node, "Could not identify next node"

        # nothing to shard
        if len(nodes_linear) == 0:
            continue

        num_shards += 1

        if simple_shard_only:
            ad_logger.debug(f"Forcing Simple Shard: Linear groups: {nodes_linear}")
            _append_simple_shard(nodes_linear, rank, world_size, sharding_config)
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:
            ad_logger.debug(f"Linear groups: {nodes_linear}")
            _append_simple_shard(nodes_linear, rank, world_size, sharding_config)
            continue

        # let's look at the unnacounted nodes. They are okay as long as they fall before the
        # first linear node or after the last linear node, i.e., outside the sharded region
        lin_nodes_flat: Set[Node] = {n for group in nodes_linear.values() for n in group}
        lin_nodes_passed: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            # check if this is another linear node
            if current_node in lin_nodes_flat:
                lin_nodes_passed.add(current_node)

            # check if we are OUTSIDE sharded region
            if len(lin_nodes_passed) == 0 or lin_nodes_passed == lin_nodes_flat:
                # remove node from unaccounted nodes since we are outside and it doesn't matter
                unaccounted_nodes.discard(current_node)
                attention_related_nodes.discard(current_node)
                attention_nodes.discard(current_node)

            current_node = current_node.next

        # let's post-process the attention-related nodes
        # we can disregard them if we also see attention nodes and we assume they are compatible
        if len(attention_nodes) > 0:
            attention_related_nodes.clear()

        # check if any unaccounted nodes are left. If so, do a simply shard
        if unaccounted_nodes or attention_related_nodes:
            ad_logger.debug(f"Unaccounted nodes: {unaccounted_nodes}")
            _append_simple_shard(nodes_linear, rank, world_size, sharding_config)
            continue

        # If we can account for all sharded nodes, we can do a two-way shard
        # --> row_split (dim 0) + col_split (dim 1) + all_reduce

        # check if we are sharding the attention block
        if attention_nodes:
            if len(attention_nodes) > 1:
                # Column-row shard boundary region detection is probably wrong - there should be
                # only one attention operation. Fall back to simple shard.
                ad_logger.debug(f"More than one attention node: {unaccounted_nodes}")
                _append_simple_shard(nodes_linear, rank, world_size, sharding_config)
                continue
            # Extract head dimension. We cannot shard below the head_dim size.
            # Assume that head_dim is the last (innermost) dimension of the tensor
            min_local_shape = attention_nodes.pop().meta["val"].shape[-1]
        else:
            min_local_shape = 1
        for i, group in enumerate(nodes_linear.values()):
            for n in group:
                if i > 0:
                    dist_op = "all_reduce"
                else:
                    dist_op = None
                sharding_config.tp_transforms.append(
                    TPShardingInfo(
                        target_node=n.name,
                        split_dim=i,
                        rank=rank,
                        world_size=world_size,
                        dist_op=dist_op,
                        min_local_shape=min_local_shape,
                    )
                )

    ad_logger.info(f"Found {num_shards} TP shards")


def detect_dp_bmm_shard(
    gm: GraphModule, rank: int, world_size: int, sharding_config: ShardingConfig
) -> None:
    """A transformation to apply sharding to batched matrix multiplications in the graph.

    We'll shard the BMM nodes by slicing the batch dimension of input tensors into world_size number of slices.
    After sharding each BMM node, we'll insert an all_gather node to gather the results across the different devices.
    This transformation handles any combination of tensor types for both inputs to the BMM operation.

    We'll also assume that the inputs to BMM are broadcasted across the devices already.
    """
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return

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

        # NOTE: our torch.ops.auto_deploy.torch_dist_all_gather doesn't support uneven splits at the moment.
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

        sharding_config.bmm_transforms.append(
            BMMShardingInfo(
                target_node=node.name,
                rank=rank,
                world_size=world_size,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        )
        ad_logger.debug(
            f"Sharding BMM for rank {rank}: batch_size={bmm_batch_size}, start_idx={start_idx}, end_idx={end_idx}"
        )

        num_bmm_shards += 1

    # Canonicalize and return
    if num_bmm_shards:
        gm = canonicalize_graph(gm)
    ad_logger.debug("After sharding BMM: " + str(gm))
    ad_logger.info(f"Found {num_bmm_shards} BMM shards")


def detect_ep_shard(
    gm: GraphModule, rank: int, world_size: int, sharding_config: ShardingConfig
) -> None:
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return

    assert isinstance(gm, GraphModule), "Expecting GraphModule"
    num_moe_patterns = 0
    for node in list(gm.graph.nodes):
        if not is_op(
            node,
            (
                torch.ops.auto_deploy.torch_moe,
                torch.ops.auto_deploy.torch_quant_fp8_moe,
                torch.ops.auto_deploy.torch_quant_fp4_moe,
            ),
        ):
            continue
        sharding_config.ep_transforms.append(
            EPShardingInfo(
                target_node=node.name,
                rank=rank,
                world_size=world_size,
            )
        )
        num_moe_patterns += 1

    ad_logger.info(f"Found {num_moe_patterns} MoE patterns")


def _insert_sharded_moe(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
):
    """Update the torch_moe node with sharded weight lists,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.
    """
    quant_impl = QuantizationImpl.create(node)
    scale_names = quant_impl.scale_names() if quant_impl else []

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
            torch.ops.auto_deploy.torch_dist_all_reduce, args=(node,)
        )
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)
