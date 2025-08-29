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
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import identify_regions_between_residuals, is_linear_op, is_op
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


class ColumnRowShardConfig(TransformConfig):
    """Configuration for column-row sharding."""

    simple_shard_only: bool = Field(default=False)


@TransformRegistry.register("detect_column_row_shard")
class ColumnRowShard(BaseTransform):
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

    config: ColumnRowShardConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ColumnRowShardConfig

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

            if self.config.simple_shard_only:
                ad_logger.debug(f"Forcing Simple Shard: Linear groups: {nodes_linear}")
                _append_simple_shard(
                    nodes_linear, local_rank, world_size, shared_config.sharding_config
                )
                continue

            # simple shard when we have != 2 groups of linear nodes
            if len(nodes_linear) != 2:
                ad_logger.debug(f"Linear groups: {nodes_linear}")
                _append_simple_shard(
                    nodes_linear, local_rank, world_size, shared_config.sharding_config
                )
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
                _append_simple_shard(
                    nodes_linear, local_rank, world_size, shared_config.sharding_config
                )
                continue

            # If we can account for all sharded nodes, we can do a two-way shard
            # --> row_split (dim 0) + col_split (dim 1) + all_reduce

            # check if we are sharding the attention block
            if attention_nodes:
                if len(attention_nodes) > 1:
                    # Column-row shard boundary region detection is probably wrong - there should be
                    # only one attention operation. Fall back to simple shard.
                    ad_logger.debug(f"More than one attention node: {unaccounted_nodes}")
                    _append_simple_shard(
                        nodes_linear, local_rank, world_size, shared_config.sharding_config
                    )
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
                    shared_config.sharding_config.tp_transforms.append(
                        TPShardingInfo(
                            target_node=n.name,
                            split_dim=i,
                            rank=local_rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=min_local_shape,
                        )
                    )

        info = TransformInfo(
            skipped=False, num_matches=num_shards, is_clean=False, has_valid_shapes=False
        )
        return gm, info


@TransformRegistry.register("detect_dp_bmm_shard")
class DpBmmShard(BaseTransform):
    """A transformation to apply sharding to batched matrix multiplications in the graph.

    We'll shard the BMM nodes by slicing the batch dimension of input tensors into world_size number of slices.
    After sharding each BMM node, we'll insert an all_gather node to gather the results across the different devices.
    This transformation handles any combination of tensor types for both inputs to the BMM operation.

    We'll also assume that the inputs to BMM are broadcasted across the devices already.
    """

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

        num_bmm_shards = 0

        for node in gm.graph.nodes:
            if not is_op(node, {torch.ops.aten.bmm}):
                continue

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
            if local_rank < remainder:
                start_idx = local_rank * (base_size + 1)
                end_idx = start_idx + base_size + 1
            else:
                start_idx = remainder + local_rank * base_size
                end_idx = start_idx + base_size

            shared_config.sharding_config.bmm_transforms.append(
                BMMShardingInfo(
                    target_node=node.name,
                    rank=local_rank,
                    world_size=world_size,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )
            ad_logger.debug(
                f"Sharding BMM for rank {local_rank}: "
                f"batch_size={bmm_batch_size}, "
                f"start_idx={start_idx}, end_idx={end_idx}"
            )

            num_bmm_shards += 1

        info = TransformInfo(
            skipped=False, num_matches=num_bmm_shards, is_clean=False, has_valid_shapes=False
        )
        return gm, info


@TransformRegistry.register("detect_ep_shard")
class DetectEpShard(BaseTransform):
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
            shared_config.sharding_config.ep_transforms.append(
                EPShardingInfo(
                    target_node=node.name,
                    rank=local_rank,
                    world_size=world_size,
                )
            )
            num_moe_patterns += 1

        info = TransformInfo(
            skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
        )
        return gm, info
