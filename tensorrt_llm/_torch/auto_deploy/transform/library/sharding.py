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
    bfs,
    extract_weight_node,
    filtered_nodes,
    identify_regions_between_residuals,
    is_any_lin_op,
    is_op,
    subgraph,
)
from ...utils.sharding_utils import (
    BMMShardingInfo,
    EPShardingInfo,
    LayerType,
    ParameterUpdateInfo,
    ShardingConfig,
    ShardingDim,
    ShardingSource,
    ShardingTransformInfo,
    SplitDimension,
    WeightShardingInfo,
    get_all_weights_in_subgraph,
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
        for tp_transform in shared_config.sharding_config.weight_sharding_transforms:
            if check_and_apply(tp_transform):
                num_matches += 1
        for bmm_transform in shared_config.sharding_config.bmm_transforms:
            if check_and_apply(bmm_transform):
                num_matches += 1
        for ep_transform in shared_config.sharding_config.ep_transforms:
            if check_and_apply(ep_transform):
                num_matches += 1

        # post-sharding cleanup transformations
        for update_transform in shared_config.sharding_config.parameter_update_transforms:
            if not check_and_apply(update_transform):
                ad_logger.warning(f"Invalid parameter update transformation {update_transform}.")

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        # exit()
        return gm, info


def _process_simple_shard(
    nodes_linear: Dict[Node, List[Node]],
    rank: int,
    world_size: int,
    sharding_config: ShardingConfig,
) -> int:
    # for every linear node:
    # --> row_split (dim 0 of weight) + all_gather (dim -1 of output)
    num_simple_shards = 0
    for node_group in nodes_linear.values():
        for n in node_group:
            num_simple_shards += int(
                sharding_config.add(
                    WeightShardingInfo.from_node(
                        n,
                        split_dim=SplitDimension.COLUMN,
                        rank=rank,
                        world_size=world_size,
                        dist_op="all_gather",
                        min_local_shape=1,
                    )
                )
            )
    return num_simple_shards


class ShardingTransformConfig(TransformConfig):
    """Configuration for sharding transformations."""

    simple_shard_only: bool = Field(default=False)
    sharding_source: List[ShardingSource] = Field(
        default_factory=lambda: [ShardingSource.HEURISTIC]
    )
    support_partial_config: bool = Field(default=False)
    # Which sharding dimensions to run: any subset of {"tp", "ep", "bmm"}
    sharding_dims: List[ShardingDim] = Field(
        default_factory=lambda: [ShardingDim.SSM, ShardingDim.TP, ShardingDim.EP, ShardingDim.BMM]
    )


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
        # world_size = 2

        if world_size < 2:
            ad_logger.info("Skipping sharding for single device")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        assert isinstance(gm, GraphModule), "Expecting GraphModule"
        sharding_config = shared_config.sharding_config
        sharding_config.rank = local_rank
        sharding_config.world_size = world_size
        sharding_config.predefined_config = factory.get_sharding_config() if factory else {}
        sharding_config.factory_source = (
            sharding_config.predefined_config.get("source", ShardingConfigSource.UNKNOWN)
            if factory
            else ShardingConfigSource.UNKNOWN
        )
        sharding_config.simple_shard_only = self.config.simple_shard_only
        sharding_config.support_partial_config = self.config.support_partial_config
        sharding_config.sharding_dims = self.config.sharding_dims
        sharding_config.sharding_source = self.config.sharding_source

        sharding_config.validate_config()

        info = TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
        for source in sharding_config.sharding_source:
            if source == ShardingSource.FACTORY:
                if len(sharding_config.get_predefined_config()) == 0:
                    ad_logger.warning(
                        "No factory config found. Skipping sharding from factory config"
                    )
                    continue
                ad_logger.info("Applying sharding from factory config")
                info += detect_sharding_from_factory_config(gm, sharding_config)

            elif source == ShardingSource.HEURISTIC:
                ad_logger.info(
                    f"Running autodeploy sharding heuristics: {sharding_config.sharding_dims}"
                )
                if ShardingDim.SSM in sharding_config.sharding_dims:
                    info += detect_ssm_shard(gm, sharding_config)

                # run TP sharding across ranks
                if ShardingDim.TP in sharding_config.sharding_dims:
                    info += detect_column_row_shard(gm, sharding_config)

                # run EP sharding across ranks
                if ShardingDim.EP in sharding_config.sharding_dims:
                    info += detect_ep_shard(gm, sharding_config)

                # run BMM sharding across ranks
                if ShardingDim.BMM in sharding_config.sharding_dims:
                    info += detect_dp_bmm_shard(gm, sharding_config)

        return gm, info


def _process_ssm_sharding(
    gm: GraphModule,
    entry_node: Node,
    sharding_config: ShardingConfig,
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

    # # ##############################################################
    # # ############## update split nodes ############################
    # # ##############################################################
    split_args_0 = list(split_nodes[0].args)
    split_args_0[1] = [s // world_size for s in split_args_0[1]]
    split_args_1 = list(split_nodes[1].args)
    split_args_1[1] = [s // world_size for s in split_args_1[1]]
    sharding_config.add(
        ParameterUpdateInfo(
            rank=rank,
            world_size=world_size,
            target_node=split_nodes[0].name,
            args=tuple(split_args_0),
        )
    )
    sharding_config.add(
        ParameterUpdateInfo(
            rank=rank,
            world_size=world_size,
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
    sharding_config.add(
        ParameterUpdateInfo(
            rank=rank, world_size=world_size, target_node=conv1d_node.name, args=tuple(conv_args)
        )
    )

    # ##############################################################
    # ####### shard the entry_node (the first linear layer) ########
    # ##############################################################
    sharding_config.add(
        WeightShardingInfo.from_node(
            entry_node,
            split_dim=SplitDimension.COLUMN,
            rank=rank,
            world_size=world_size,
            dist_op=None,
            min_local_shape=min_local_shape,
            fused_weight_dims=fused_weight_dims["in_proj"],
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
        sharding_config.add(
            WeightShardingInfo.from_node(
                list(weight_node.users)[0],
                split_dim=SplitDimension.COLUMN,
                rank=rank,
                world_size=world_size,
                dist_op=None,
                min_local_shape=min_local_shape,
                fused_weight_dims=fused_dims,
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
            sharding_config.add(
                ParameterUpdateInfo(
                    rank=rank, world_size=world_size, target_node=view_node.name, args=tuple(args)
                )
            )
            ad_logger.debug(f"\nUpdated view node {view_node} arguments to {view_node.args}")

    ##############################################################
    ############## shard the out_proj node #######################
    ##############################################################
    sharding_config.add(
        WeightShardingInfo.from_node(
            out_proj_node,
            split_dim=SplitDimension.ROW,
            rank=rank,
            world_size=world_size,
            dist_op="all_reduce",
        )
    )
    return 1


def _process_column_sharding(
    gm: GraphModule,
    linear_nodes: List[Node],
    sharding_config: ShardingConfig,
    rank: int,
    world_size: int,
    min_local_shape: int = 1,
    fused_weight: bool = False,
) -> None:
    """
    Parse the column sharding from the candidate nodes and update the view and split nodes accordingly.
    """
    for linear_node in linear_nodes:
        sharding_config.add(
            WeightShardingInfo.from_node(
                linear_node,
                split_dim=SplitDimension.COLUMN,
                rank=rank,
                world_size=world_size,
                dist_op=None,  # for column sharding, no dist op is performed
                min_local_shape=min_local_shape,
            )
        )

    # get the subgraph of this module. Subgraph boundary is the next linear node.
    next_lin_node, _ = bfs(linear_nodes[0], is_any_lin_op, include_root=False)
    subgraph_nodes = subgraph(
        linear_nodes,
        [next_lin_node],
    )

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
            sharding_config.add(
                ParameterUpdateInfo(
                    rank=rank, world_size=world_size, target_node=view_node.name, args=tuple(args)
                )
            )
            ad_logger.debug(f"\nUpdated view node {view_node} arguments to {view_node.args}")

    # if fused_weight_dims is provided, we need to update all split sizes
    if fused_weight:
        assert len(linear_nodes) == 1, "Fused weight should be only one linear node"
        node = linear_nodes[0]
        assert world_size is not None, "World size is required to update the split node params"
        assert len(node.users) == 1, "Fused linear node should have only one user: a split node"
        user = list(node.users)[0]
        if is_op(user, [torch.ops.aten.split_with_sizes]):
            orig_sizes = user.args[1]
            new_sizes = [orig_sizes[i] // world_size for i in range(len(orig_sizes))]
            args = list(user.args)
            args[1] = new_sizes
            sharding_config.add(
                ParameterUpdateInfo(
                    rank=rank, world_size=world_size, target_node=user.name, args=tuple(args)
                )
            )
            ad_logger.debug(
                f"\nInserted parameter update transformation for split node {user} arguments to {user.args}"
            )


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
    #   - "mamba"
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

    for lin_node in filtered_nodes(gm.graph.nodes, is_any_lin_op):
        # use node's weight name to get the module name
        module_name = extract_weight_node(lin_node).target

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
                    sharding_config.weight_sharding_transforms.append(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            rank=rank,
                            world_size=world_size,
                            dist_op=None,
                            min_local_shape=min_local_shape,
                        )
                    )
                    num_row_col_shards += 1
                elif config == "rowwise":
                    sharding_config.weight_sharding_transforms.append(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.ROW,
                            rank=rank,
                            world_size=world_size,
                            dist_op="all_reduce",
                            min_local_shape=min_local_shape,
                        )
                    )
                    num_row_col_shards += 1
                elif config == "mamba":
                    sharding_config.weight_sharding_transforms.append(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            rank=rank,
                            world_size=world_size,
                            dist_op=None,
                            min_local_shape=min_local_shape,
                            layer_type=LayerType.MAMBA,
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
                            sharding_config.weight_sharding_transforms.append(
                                WeightShardingInfo(
                                    target_node=lin_node.name,
                                    split_dim=SplitDimension.COLUMN,
                                    rank=rank,
                                    world_size=world_size,
                                    dist_op=None,
                                    min_local_shape=min_local_shape,
                                )
                            )
                        elif col_row_action == "rowwise":
                            sharding_config.weight_sharding_transforms.append(
                                WeightShardingInfo(
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
                    sharding_config.weight_sharding_transforms.append(
                        WeightShardingInfo.from_node(
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
                    sharding_config.weight_sharding_transforms.append(
                        WeightShardingInfo.from_node(
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

    num_matches = len(sharding_config.weight_sharding_transforms)

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


def detect_ssm_shard(
    gm: GraphModule,
    sharding_config: ShardingConfig,
) -> TransformInfo:
    """A transformation to apply sharding to the model following SSM parallelism.
    TODO: This is a TEMPORARY place for this logic due to the incompatibility between the
    identify_regions_between_residuals() and subgraph() methods to detect layers.
    The goal is to have a unified single pass over the graph to detect layers and apply
    appropriate sharding transformations.
    """
    rank, world_size = sharding_config.rank, sharding_config.world_size
    if world_size < 2:
        ad_logger.info("Skipping TP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
    ad_logger.info("Running SSM sharding detection")

    # find all ssm nodes in the graph
    ssm_nodes = filtered_nodes(gm.graph.nodes, ops=torch.ops.auto_deploy.torch_ssm)
    num_ssm_shards = 0
    for ssm_node in ssm_nodes:
        # We assume that one ssm node defines a subgraph corresponding
        # to a single Mamba layer.
        # Find defining previous (in_proj) and next (out_proj) linear nodes.
        in_proj_node, _ = bfs(ssm_node, is_any_lin_op, attr_next="args", include_root=False)

        num_ssm_shards += int(
            _process_ssm_sharding(gm, in_proj_node, sharding_config, rank, world_size)
        )

    ad_logger.info(f"Found {num_ssm_shards} SSM shards")
    return TransformInfo(
        skipped=False, num_matches=num_ssm_shards, is_clean=False, has_valid_shapes=False
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
            if is_any_lin_op(current_node):
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
            num_simple_shards += _process_simple_shard(
                nodes_linear, rank, world_size, sharding_config
            )
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:
            ad_logger.debug(f"Linear groups: {nodes_linear}")
            num_simple_shards += _process_simple_shard(
                nodes_linear, rank, world_size, sharding_config
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
            num_simple_shards += _process_simple_shard(
                nodes_linear, rank, world_size, sharding_config
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
                num_simple_shards += _process_simple_shard(
                    nodes_linear, rank, world_size, sharding_config
                )
                continue
            # Extract head dimension. We cannot shard below the head_dim size.
            # Assume that head_dim is the last (innermost) dimension of the tensor
            min_local_shape = attention_nodes.pop().meta["val"].shape[-1]
        else:
            min_local_shape = 1

        # We are inserting column-row shard for each group of linear enodes
        # This may require parameter update of nodes whose args depend on (sharded) dimensions,
        # such as view or split nodes.
        nodes_to_column_shard = list(nodes_linear.values())[0]
        nodes_to_row_shard = list(nodes_linear.values())[1]
        if len(nodes_to_row_shard) != 1:
            ad_logger.warning(
                "Expecting only one linear node for row sharding, but got %s",
                len(nodes_to_row_shard),
            )
            num_simple_shards += _process_simple_shard(
                nodes_linear, rank, world_size, sharding_config
            )
            continue

        # column-row sharding
        _process_column_sharding(
            gm,
            linear_nodes=nodes_to_column_shard,
            sharding_config=sharding_config,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
        )

        # shard single row node
        sharding_config.weight_sharding_transforms.append(
            WeightShardingInfo.from_node(
                nodes_to_row_shard[0],
                split_dim=SplitDimension.ROW,
                rank=rank,
                world_size=world_size,
                dist_op="all_reduce",
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
