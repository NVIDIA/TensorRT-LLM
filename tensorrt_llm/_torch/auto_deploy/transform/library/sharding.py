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

import operator
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory, ShardingConfigSource
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    filtered_nodes,
    identify_regions_between_residuals,
    is_fake_quantized_linear_op,
    is_linear_op,
    is_op,
)
from ...utils.sharding_utils import (
    BMMShardingInfo,
    EPShardingInfo,
    ShardingConfig,
    ShardingTransformInfo,
    SplitDimension,
    TPShardingInfo,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


@TransformRegistry.register("sharding_transform_executor")
class ShardingTransformExecutor(BaseTransform):
    """Apply transformations to the graph module.

    Args:
        gm: Graph module to apply transformations to
        sharding_config: Transformation configuration containing list of transformations to apply
    """

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
        for tp_transform in shared_config.sharding_config.tp_transforms:
            if check_and_apply(tp_transform):
                num_matches += 1
        for bmm_transform in shared_config.sharding_config.bmm_transforms:
            if check_and_apply(bmm_transform):
                num_matches += 1
        for ep_transform in shared_config.sharding_config.ep_transforms:
            if check_and_apply(ep_transform):
                num_matches += 1

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
        return gm, info


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
                TPShardingInfo.from_node(
                    n,
                    split_dim=SplitDimension.COLUMN,
                    rank=rank,
                    world_size=world_size,
                    dist_op="all_gather",
                    min_local_shape=1,
                )
            )
    sharding_config.tp_transforms.extend(tp_shards)


class ShardingTransformConfig(TransformConfig):
    """Configuration for sharding transformations."""

    simple_shard_only: bool = Field(default=False)
    use_sharding_from_factory: bool = Field(default=False)
    support_partial_config: bool = Field(default=False)
    # Which sharding families to run: any subset of {"tp", "ep", "bmm"}
    sharding_dims: List[str] = Field(default_factory=lambda: ["tp", "ep", "bmm"])


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
        shared_config.sharding_config.rank = local_rank
        shared_config.sharding_config.world_size = world_size
        shared_config.sharding_config.predefined_config = (
            factory.get_sharding_config() if factory else {}
        )
        shared_config.sharding_config.factory_source = (
            shared_config.sharding_config.predefined_config.get(
                "source", ShardingConfigSource.UNKNOWN
            )
            if factory
            else ShardingConfigSource.UNKNOWN
        )
        shared_config.sharding_config.simple_shard_only = self.config.simple_shard_only
        shared_config.sharding_config.support_partial_config = self.config.support_partial_config
        shared_config.sharding_config.sharding_dims = self.config.sharding_dims

        shared_config.sharding_config.use_sharding_from_factory = (
            self.config.use_sharding_from_factory
        )

        sharding_config = shared_config.sharding_config
        sharding_config.validate_config()

        if (
            shared_config.sharding_config.use_sharding_from_factory
            and len(shared_config.sharding_config.get_predefined_config()) > 0
        ):
            ad_logger.info("Applying sharding from config")
            factory_info = detect_sharding_from_factory_config(gm, sharding_config)
            return gm, factory_info

        ad_logger.info(
            f"Running autodeploy sharding heuristics: {shared_config.sharding_config.sharding_dims}"
        )
        # run TP sharding across ranks
        if "tp" in shared_config.sharding_config.sharding_dims:
            tp_info = detect_column_row_shard(gm, sharding_config)
        else:
            tp_info = TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # run EP sharding across ranks
        if "ep" in shared_config.sharding_config.sharding_dims:
            ep_info = detect_ep_shard(gm, sharding_config)
        else:
            ep_info = TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # run BMM sharding across ranks
        if "bmm" in shared_config.sharding_config.sharding_dims:
            dp_bmm_info = detect_dp_bmm_shard(gm, sharding_config)
        else:
            dp_bmm_info = TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        info = TransformInfo(
            skipped=tp_info.skipped and ep_info.skipped and dp_bmm_info.skipped,
            num_matches=tp_info.num_matches + ep_info.num_matches + dp_bmm_info.num_matches,
            is_clean=tp_info.is_clean and ep_info.is_clean and dp_bmm_info.is_clean,
            has_valid_shapes=tp_info.has_valid_shapes
            and ep_info.has_valid_shapes
            and dp_bmm_info.has_valid_shapes,
        )
        return gm, info


def detect_sharding_from_factory_config(
    gm: GraphModule,
    sharding_config: ShardingConfig,
) -> TransformInfo:
    """
    Create sharding transformations from the predefined config.
    TODO: currently, it applies only to TP sharding.
    Args:
        gm: Graph module to apply transformations to
        sharding_config: Predefined sharding configuration
    """
    # check if config is valid.
    # 1. it is a Dict[str, str]
    # 2. the keys are of format "module.submodule.subsubmodule..."
    # 3. the wildcard "*" is allowed in the keys
    # 4. the allowed values are:
    #   - "colwise"
    #   - "rowwise"
    #   - "sequence_parallel"
    #   - "local_colwise"
    #   - "local_rowwise"
    #   - "local"
    #   - "gather"
    # The following constraints are based on
    # https://github.com/huggingface/transformers/blob/d8e05951b8efd4880acca9a3f291e8b65841a86d/src/transformers/models/llama4/configuration_llama4.py#L249

    factory_config = sharding_config.get_predefined_config()
    head_dim = factory_config["head_dim"]
    tp_plan = factory_config["tp_plan"]

    rank, world_size = sharding_config.rank, sharding_config.world_size

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

    for lin_node in filtered_nodes(gm.graph.nodes, [is_linear_op, is_fake_quantized_linear_op]):
        # use node's weight name to get the module name
        module_name = lin_node.args[1].target

        if any(attn_name in module_name for attn_name in attn_names):
            min_local_shape = head_dim
        else:
            min_local_shape = 1

        # use regex to find if module_name matches any of the keys in sharding_config
        for key in tp_plan.keys():
            pattern_string = "*" + key + "*"
            # convert it to regex. Escape dots, replace * with .*
            # First, we substitute * with an unlikely character, e.g. @
            # Then we escape dots, and finally we replace @ with .*
            pattern_string = pattern_string.replace("*", "@")
            pattern_regex = re.escape(pattern_string).replace("@", ".*")
            if re.match(pattern_regex, module_name):
                num_shards += 1
                # we have a match. Get the config for this layer
                config = tp_plan[key]
                if config == "colwise":
                    sharding_config.tp_transforms.append(
                        TPShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            rank=rank,
                            world_size=world_size,
                            dist_op=None,
                            min_local_shape=min_local_shape,
                        )
                    )
                elif config == "rowwise":
                    sharding_config.tp_transforms.append(
                        TPShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.ROW,
                            rank=rank,
                            world_size=world_size,
                            dist_op="all_reduce",
                            min_local_shape=min_local_shape,
                        )
                    )
                    num_row_col_shards += 1
                elif "sequence" in config:
                    # TODO: Sequence parallelism is not supported yet.
                    ad_logger.warning("Sequence parallelism is not supported yet. Skipping.")
                elif "local" in config:
                    # Check if this applies to shared experts in EP parallelism.
                    # If yes, apply the TP col-row shard.
                    if "shared" in module_name:
                        col_row_action = config.replace("local_", "")
                        if col_row_action == "colwise":
                            sharding_config.tp_transforms.append(
                                TPShardingInfo(
                                    target_node=lin_node.name,
                                    split_dim=SplitDimension.COLUMN,
                                    rank=rank,
                                    world_size=world_size,
                                    dist_op=None,
                                    min_local_shape=min_local_shape,
                                )
                            )
                        elif col_row_action == "rowwise":
                            sharding_config.tp_transforms.append(
                                TPShardingInfo(
                                    target_node=lin_node.name,
                                    split_dim=SplitDimension.ROW,
                                    rank=rank,
                                    world_size=world_size,
                                    dist_op="all_reduce",
                                    min_local_shape=min_local_shape,
                                )
                            )
                            num_row_col_shards += 1
                        else:
                            ad_logger.warning(f"Unsupported sharding action {config}. Skipping.")
                    else:
                        # TODO: local refers to hybrid EP+TP parallelism. Not supported yet.
                        ad_logger.warning("Local EP+TP sharding is not supported yet. Skipping.")

                elif "gather" in config:
                    # Simple shard (row + all_gather)
                    sharding_config.tp_transforms.append(
                        TPShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            rank=rank,
                            world_size=world_size,
                            dist_op="all_gather",
                            min_local_shape=1,
                        )
                    )
                    num_simple_shards += 1
                else:
                    ad_logger.warning(
                        f"Unsupported sharding action {config}. Fallback to simple shard"
                    )
                    sharding_config.tp_transforms.append(
                        TPShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            rank=rank,
                            world_size=world_size,
                            dist_op="all_gather",
                            min_local_shape=1,
                        )
                    )
                # after successful match, break the loop
                break

    ad_logger.info(
        f"Applied {num_shards} TP shards (simple: {num_simple_shards}, "
        f"row-col pattern: {num_row_col_shards})"
    )

    num_matches = len(sharding_config.tp_transforms)

    if sharding_config.support_partial_config:
        ad_logger.info(
            f"Partial factory config applied only for TP. "
            f"Applying heuristics for {sharding_config.sharding_dims}."
        )

        # run EP sharding across ranks
        if "ep" in sharding_config.sharding_dims:
            ep_info = detect_ep_shard(gm, sharding_config)
        else:
            ep_info = TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # run BMM sharding across ranks
        if "bmm" in sharding_config.sharding_dims:
            dp_bmm_info = detect_dp_bmm_shard(gm, sharding_config)
        else:
            dp_bmm_info = TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        num_matches += ep_info.num_matches + dp_bmm_info.num_matches

    return TransformInfo(
        skipped=False,
        num_matches=num_matches,
        is_clean=False,
        has_valid_shapes=False,
    )


def detect_column_row_shard(
    gm: GraphModule,
    sharding_config: ShardingConfig,
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

    rank, world_size = sharding_config.rank, sharding_config.world_size
    if world_size < 2:
        ad_logger.info("Skipping TP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    ad_logger.info("Running TP sharding detection")

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
        torch.ops.auto_deploy.torch_attention,
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
    num_simple_shards = 0
    num_row_col_shards = 0
    for n_start, n_end in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        # we iterate through all nodes between the two boundary nodes and store linear nodes
        # sorted by their input activation node. We also store remaining nodes.
        nodes_linear: DefaultDict[Node, List[Node]] = defaultdict(list)
        attention_nodes: Set[Node] = set()
        attention_related_nodes: Set[Node] = set()
        unaccounted_nodes: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            if is_linear_op(current_node) or is_fake_quantized_linear_op(current_node):
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

        if sharding_config.simple_shard_only:
            ad_logger.debug(f"Forcing Simple Shard: Linear groups: {nodes_linear}")
            _append_simple_shard(nodes_linear, rank, world_size, sharding_config)
            num_simple_shards += 1
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:
            ad_logger.debug(f"Linear groups: {nodes_linear}")
            _append_simple_shard(nodes_linear, rank, world_size, sharding_config)
            num_simple_shards += 1
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
            num_simple_shards += 1
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
                num_simple_shards += 1
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
                    TPShardingInfo.from_node(
                        n,
                        split_dim=i,
                        rank=rank,
                        world_size=world_size,
                        dist_op=dist_op,
                        min_local_shape=min_local_shape,
                    )
                )
        num_row_col_shards += 1

    ad_logger.info(
        f"Found {num_shards} TP shards (simple: {num_simple_shards}, row-col: {num_row_col_shards})"
    )
    return TransformInfo(
        skipped=False, num_matches=num_shards, is_clean=False, has_valid_shapes=False
    )


def detect_dp_bmm_shard(gm: GraphModule, sharding_config: ShardingConfig) -> TransformInfo:
    """A transformation to apply sharding to batched matrix multiplications in the graph.

    We'll shard the BMM nodes by slicing the batch dimension of input tensors into world_size number of slices.
    After sharding each BMM node, we'll insert an all_gather node to gather the results across the different devices.
    This transformation handles any combination of tensor types for both inputs to the BMM operation.

    We'll also assume that the inputs to BMM are broadcasted across the devices already.
    """
    ad_logger.debug("Before sharding graph: " + str(gm))
    rank, world_size = sharding_config.rank, sharding_config.world_size
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

    ad_logger.debug("After sharding BMM: " + str(gm))
    ad_logger.info(f"Found {num_bmm_shards} BMM shards")

    return TransformInfo(
        skipped=False, num_matches=num_bmm_shards, is_clean=False, has_valid_shapes=False
    )


def detect_ep_shard(gm: GraphModule, sharding_config: ShardingConfig) -> TransformInfo:
    ad_logger.debug("Before sharding graph: " + str(gm))

    rank, world_size = sharding_config.rank, sharding_config.world_size
    if world_size < 2:
        ad_logger.info("Skipping EP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    assert isinstance(gm, GraphModule), "Expecting GraphModule"
    num_moe_patterns = 0
    for node in list(gm.graph.nodes):
        if not is_op(
            node,
            (
                torch.ops.auto_deploy.torch_moe,
                torch.ops.auto_deploy.torch_quant_fp8_moe,
                torch.ops.auto_deploy.torch_quant_nvfp4_moe,
                torch.ops.auto_deploy.triton_mxfp4_moe,
            ),
        ):
            continue
        sharding_config.ep_transforms.append(
            EPShardingInfo.from_node(
                node,
                rank=rank,
                world_size=world_size,
            )
        )
        num_moe_patterns += 1

    ad_logger.info(f"Found {num_moe_patterns} MoE patterns")

    return TransformInfo(
        skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
    )
