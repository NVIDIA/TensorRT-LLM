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

import re
from typing import Any, Dict, List, Tuple, Type, Union

import torch
from pydantic import Field, field_validator
from torch.fx import GraphModule, Node

from .....functional import AllReduceStrategy
from ...models.factory import ModelFactory, ShardingConfigSource
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    bfs,
    extract_weight_node,
    filtered_nodes,
    get_all_layer_subgraphs,
    is_any_attention_op,
    is_any_lin_op,
    is_any_moe_op,
    is_any_ssm_op,
    is_op,
    subgraph,
)
from ...utils.sharding_utils import (
    BMMShardingInfo,
    DistBackend,
    EPShardingInfo,
    LayerType,
    ParameterUpdateInfo,
    ShardingDim,
    ShardingSource,
    ShardingTransformContainer,
    ShardingTransformInfo,
    SplitDimension,
    WeightShardingInfo,
    get_all_weights_in_subgraph,
    validate_allreduce_strategy,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


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
        for source in transform_container.sharding_source:
            if source == ShardingSource.FACTORY:
                if len(transform_container.get_factory_config()) == 0:
                    ad_logger.debug(
                        "No factory config found. Skipping sharding from factory config"
                    )
                    continue
                ad_logger.info("Applying sharding from factory config")
                info += detect_sharding_from_config(gm, transform_container, ShardingSource.FACTORY)
            elif source == ShardingSource.MANUAL:
                if len(transform_container.get_manual_config()) == 0:
                    ad_logger.debug("No manual config found. Skipping sharding from manual config")
                    continue
                ad_logger.info("Applying sharding from manual config")
                info += detect_sharding_from_config(gm, transform_container, ShardingSource.MANUAL)

            elif source == ShardingSource.HEURISTIC:
                ad_logger.info(
                    f"Running autodeploy sharding heuristics: {transform_container.sharding_dims}"
                )
                # run TP sharding across ranks
                if ShardingDim.TP in transform_container.sharding_dims:
                    info += detect_column_row_shard(gm, transform_container)

                # run EP sharding across ranks
                if ShardingDim.EP in transform_container.sharding_dims:
                    info += detect_ep_shard(gm, transform_container)

                # run BMM sharding across ranks
                if ShardingDim.BMM in transform_container.sharding_dims:
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
            rank=rank,
            world_size=world_size,
            target_node=split_nodes[0].name,
            args=tuple(split_args_0),
        )
    )
    transform_container.add(
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
    transform_container.add(
        ParameterUpdateInfo(
            rank=rank, world_size=world_size, target_node=conv1d_node.name, args=tuple(conv_args)
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
                rank=rank,
                world_size=world_size,
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
            rank=rank,
            world_size=world_size,
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
                rank=rank,
                world_size=world_size,
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
                    rank=rank, world_size=world_size, target_node=view_node.name, args=tuple(args)
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
                    rank=rank, world_size=world_size, target_node=user.name, args=tuple(args)
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
        transform_container: containing predefined sharding configuration
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
                            rank=rank,
                            world_size=world_size,
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
                    # TODO: Sequence parallelism is not supported yet.
                    ad_logger.warning("Sequence parallelism is not supported yet. Skipping.")
                elif "local" in config:
                    # Check if this applies to shared experts in EP parallelism.
                    # If yes, apply the TP col-row shard.
                    if "shared" in module_name:
                        col_row_action = config.replace("local_", "")
                        if col_row_action == "colwise":
                            transform_container.add(
                                WeightShardingInfo.from_node(
                                    lin_node,
                                    split_dim=SplitDimension.COLUMN,
                                    rank=rank,
                                    world_size=world_size,
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
                                    rank=rank,
                                    world_size=world_size,
                                    dist_op="all_reduce",
                                    min_local_shape=min_local_shape,
                                    layer_type=layer_type,
                                )
                            ):
                                num_row_col_shards += 1
                        else:
                            ad_logger.warning(f"Unsupported sharding action {config}. Skipping.")
                    else:
                        # TODO: local refers to hybrid EP+TP parallelism. Not supported yet.
                        ad_logger.warning("Local EP+TP sharding is not supported yet. Skipping.")

                elif "gather" in config:
                    # Simple shard (row + all_gather)
                    if transform_container.add(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.COLUMN,
                            rank=rank,
                            world_size=world_size,
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
    rank, world_size = transform_container.rank, transform_container.world_size
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
    rank, world_size = transfrom_container.rank, transfrom_container.world_size

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
                rank=rank,
                world_size=world_size,
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
                rank=rank,
                world_size=world_size,
            )
        ):
            num_moe_patterns += 1

    ad_logger.info(f"Found {num_moe_patterns} MoE patterns")

    return TransformInfo(
        skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
    )
