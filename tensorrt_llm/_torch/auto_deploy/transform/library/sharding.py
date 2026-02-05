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
from enum import Enum, IntEnum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator
from torch.fx import GraphModule, Node

from .....functional import AllReduceStrategy
from ...custom_ops.trtllm_dist import is_trtllm_op_available
from ...models.factory import ModelFactory, ShardingConfigSource
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import del_attr_by_name, eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    LayerSubgraph,
    LayerType,
    bfs,
    extract_weight_name,
    filtered_nodes,
    get_all_layer_subgraphs,
    get_all_weight_infos,
    get_all_weights_in_subgraph,
    is_any_attention_op,
    is_any_lin_op,
    is_any_moe_op,
    is_any_ssm_op,
    is_op,
    num_users_of_weight_node,
    shape,
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


########################################################
#  Helper enums
########################################################
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


class DistBackend(Enum):
    """Enum for distributed backend."""

    AUTO = "auto"
    TRTLLM = "trtllm"
    TORCH = "torch"


class MLPType(Enum):
    """Enum for MLP type."""

    GATED_MLP = "gated_mlp"  # explicit three weights: up, down, gate (in this order)
    MLP = "mlp"  # two weights: up, down
    FUSED_GATED_MLP = (
        "fused_gated_mlp"  # fused three weights (two inputs) up_gate, down (in this order)
    )


########################################################
#  Sharding classes
########################################################
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
    shard_all_unprocessed: bool = Field(
        default=True,
        description="When True, apply simple shard (column split + all_gather) to "
        "'leftover' linear nodes that are not part of any layer subgraph.",
    )
    allreduce_strategy: AllReduceStrategy = Field(
        default=AllReduceStrategy.AUTO,
        description="AllReduce strategy for distributed operations. "
        "Options: AUTO (automatic selection), NCCL, ONESHOT, TWOSHOT, MIN_LATENCY, "
        "LOWPRECISION, UB, MNNVL, NCCL_SYMMETRIC",
    )

    process_grid: Dict[ShardingDim, int] = Field(default_factory=dict)

    enable_attention_dp: bool = Field(
        default=False,
        description="When True, skip TP sharding as attention data parallelism is enabled.",
    )

    def validate_config(self, sources: Union[ShardingSource, List[ShardingSource]] = None) -> bool:
        init_process_grid_from_config(self)
        if sources is None:
            sources = [ShardingSource.FACTORY, ShardingSource.MANUAL]
        if not isinstance(sources, list):
            sources = [sources]
        for source in sources:
            config = self.manual_config if source == ShardingSource.MANUAL else self.factory_config
            if (
                source == ShardingSource.FACTORY
                and self.factory_source != ShardingConfigSource.HUGGINGFACE
            ):
                if "source" in config:
                    self.factory_source = config["source"]
                if self.factory_source != ShardingConfigSource.HUGGINGFACE:
                    ad_logger.debug(
                        "Sharding config is currently only supported for HuggingFace. Skipping."
                    )
                    config.clear()
                    continue

            if "tp_plan" not in config or config["tp_plan"] is None or len(config["tp_plan"]) == 0:
                ad_logger.debug("Sharding config does not contain tp_plan. Skipping.")
                # invalidate the config
                config.clear()
                continue

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
                continue

    @field_validator("allreduce_strategy", mode="before")
    @classmethod
    def _validate_allreduce_strategy(cls, v):
        """Convert string names like 'AUTO' to AllReduceStrategy enum."""
        return validate_allreduce_strategy(v)

    dist_backend: DistBackend = Field(default=DistBackend.AUTO)


class ShardingTransformInfo(BaseModel, ABC):
    """Abstract base class for transformation configurations."""

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

    def __hash__(self) -> int:
        """Make the transform info hashable by excluding the config field.

        The config field is excluded because:
        1. It may not be hashable (ShardingTransformConfig is mutable)
        2. Tests set config=None before comparison anyway
        """
        # Get all fields except 'config' for hashing
        field_values = []
        for field_name, field_info in self.model_fields.items():
            if field_name != "config":
                value = getattr(self, field_name)
                # Handle enums
                if isinstance(value, (Enum, IntEnum)):
                    field_values.append(value.value)
                else:
                    field_values.append(value)
        return hash(tuple(field_values))


class WeightShardingInfo(ShardingTransformInfo):
    """Configuration for TP sharding transformations."""

    split_dim: SplitDimension
    dist_op: Optional[Literal["all_reduce", "all_gather"]] = None
    min_local_shape: int = 1
    layer_type: LayerType = LayerType.MLP
    # used for TP sharding of fused weights
    fused_weight_dims: Optional[tuple] = None

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
            config=self.config,
            add_dist=self.dist_op is not None,
            min_local_shape=self.min_local_shape,
            fused_weight_dims=self.fused_weight_dims,
            quantization_cb=self.quantization_cb,
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


class FP8WeightShardingInfo(QuantizationShardingMixin, WeightShardingInfo):
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
    # assert weight_scale.dim() == 1
    weight_shape_original = list(sharded_uint8_weight_shape)
    weight_shape_original[dim] = weight_shape_original[dim] * world_size
    weight_shape_original[-1] *= 2
    modelopt_weight_scale = cutlass_fp4_scale_to_modelopt_fp4_scale(
        weight_scale, tuple(weight_shape_original)
    )
    return modelopt_fp4_scale_to_cutlass_fp4_scale(
        modelopt_weight_scale.tensor_split(world_size, dim=dim)[rank]
    )


class FP4WeightShardingInfo(QuantizationShardingMixin, WeightShardingInfo):
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
        lhs_batch_size = shape(lhs_tensor)[0]
        rhs_batch_size = shape(rhs_tensor)[0]

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
        _insert_sharded_moe(gm, node, self.config, scale_names=self.scale_names())


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


class RMSNormShardingInfo(ShardingTransformInfo):
    """Configuration for replacing RMSNorm with sharded version.

    When RMSNorm (torch_rmsnorm) operates on sharded activations with
    weight shape [num_heads * head_dim] (full hidden size), it needs to be
    replaced with sharded_rmsnorm which uses all_reduce to compute
    the correct global mean across shards.

    The detection of whether an RMSNorm needs this treatment is done in
    _shard_qk_norm based on weight shape matching q/k projection
    output dimensions.
    """

    world_size: int

    @classmethod
    def from_node(cls, node: Node, **kwargs) -> "RMSNormShardingInfo":
        """Create a RMSNormShardingInfo from a node."""
        return cls(target_node=node.name, **kwargs)

    def validate(self, gm: GraphModule = None, node: Node = None) -> bool:
        """Validate that the node is a torch_rmsnorm op."""
        if not is_op(node, torch.ops.auto_deploy.torch_rmsnorm):
            ad_logger.debug(
                f"RMSNormShardingInfo only applies to torch_rmsnorm ops. "
                f"Got {node.target}. Skipping."
            )
            return False
        return True

    def apply(self, gm: GraphModule, node: Node) -> None:
        """Replace torch_rmsnorm with sharded_rmsnorm and shard the weight.

        This handles both:
        1. Weight sharding: Shard the RMSNorm weight across ranks
        2. Op replacement: Replace with sharded_rmsnorm which uses all_reduce
           for computing the correct global mean across shards.
        """
        # Get original arguments: (input, weight, eps)
        input_node = node.args[0]
        weight_node = node.args[1]
        eps = node.args[2]

        # Shard the weight parameter (column-wise for 1D RMSNorm weight)
        weight_key = weight_node.target
        weight_tensor = gm.get_parameter(weight_key)
        shard_weight_tensor(
            gm=gm,
            weight_tensor=weight_tensor,
            param_key=weight_key,
            dim=0,  # Column shard for 1D weight
            rank=self.config.rank,
            world_size=self.world_size,
        )
        ad_logger.debug(f"Sharded RMSNorm weight: {weight_key}")

        # Insert the new node with world_size parameter
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                torch.ops.auto_deploy.sharded_rmsnorm.default,
                args=(input_node, weight_node, eps, self.world_size),
            )

        # Replace all uses with the new node and remove the old node
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        ad_logger.debug(
            f"Replaced torch_rmsnorm with sharded_rmsnorm (world_size={self.world_size})"
        )


########################################################
#  Transform API classes
########################################################


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
        assert isinstance(gm, GraphModule), "Expecting GraphModule"
        config = self.config
        config.factory_config = factory.get_sharding_config() if factory else {}
        config.rank = local_rank
        config.world_size = world_size
        # validate the config
        config.validate_config()
        # initialize the transform container
        transform_container = ShardingTransformContainer(config=config)
        shared_config.sharding_transform_container = transform_container
        ad_logger.info(
            f"Using allreduce strategy: {config.allreduce_strategy.name}, dist backend: {config.dist_backend}"
        )

        if world_size < 2 or config.enable_attention_dp:
            reason = "single device" if world_size < 2 else "attention DP enabled"
            ad_logger.info(f"Skipping sharding: {reason}")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

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
                ad_logger.info(f"Running autodeploy sharding heuristics: {config.sharding_dims}")
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
        for rmsnorm_transform in transforms.rmsnorm_transforms:
            if check_and_apply(rmsnorm_transform):
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


class ShardingTransformContainer(BaseModel):
    """Configuration for sharding the model."""

    config: ShardingTransformConfig = Field(default_factory=ShardingTransformConfig)
    weight_sharding_transforms: List[WeightShardingInfo] = Field(default_factory=list)
    parameter_update_transforms: List[ParameterUpdateInfo] = Field(default_factory=list)
    bmm_transforms: List[BMMShardingInfo] = Field(default_factory=list)
    ep_transforms: List[EPShardingInfo] = Field(default_factory=list)
    rmsnorm_transforms: List[RMSNormShardingInfo] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transform_list_dict = {
            WeightShardingInfo: self.weight_sharding_transforms,
            BMMShardingInfo: self.bmm_transforms,
            EPShardingInfo: self.ep_transforms,
            ParameterUpdateInfo: self.parameter_update_transforms,
            RMSNormShardingInfo: self.rmsnorm_transforms,
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


########################################################
#  Helper functions
########################################################


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


TP_SHARDING_RULES = [
    (lambda n: is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear), FP8WeightShardingInfo),
    (
        lambda n: is_op(n, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear),
        FP4WeightShardingInfo,
    ),
]


def _resolve_tp_cls_from_node(node: Node):
    for pred, cls in TP_SHARDING_RULES:
        try:
            if pred(node):
                return cls
        except Exception:
            pass
    return WeightShardingInfo


def init_process_grid_from_config(
    config: ShardingTransformConfig,
) -> Dict[ShardingDim, Dict[str, int]]:
    rank, world_size = config.rank, config.world_size
    if len(config.process_grid) > 0:
        ad_logger.debug(f"EP + TP sharding process grid: {config.process_grid}")
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
            ep_size = world_size
            tp_size = 1
            ep_rank = rank
            tp_rank = 0
    else:
        ep_size = world_size
        tp_size = 1
        ep_rank = rank
        tp_rank = 0
    process_grid = {
        ShardingDim.EP: {"p": ep_rank, "w": ep_size},
        ShardingDim.TP: {"p": tp_rank, "w": tp_size},
    }
    ad_logger.info(f"EP + TP sharding process grid: {process_grid}")
    config.process_grid = process_grid
    return process_grid


########################################################
#  Sharding transform functions
########################################################
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
    modname, _, param_name = param_key.rpartition(".")
    submod = gm.get_submodule(modname)
    param_new = nn.Parameter(sharded_weight.detach().clone(), requires_grad=requires_grad)
    setattr(submod, param_name, param_new)

    return sharded_weight, sharded_shape


def _shard_parameter_node(
    gm: GraphModule,
    node: Node,
    dim: int,
    config: ShardingTransformConfig,
    add_dist: bool = False,
    min_local_shape: int = 1,
    fused_weight_dims: Optional[tuple] = None,
    quantization_cb: Optional[
        Callable[[GraphModule, nn.Module, Node, str, torch.Size, int, int, int], None]
    ] = None,
) -> None:
    """Replace the node with parametrized weight tensor with a new node that accepts sharded weights.

    The state_dict is also updated to contain the sharded weights.
    """
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    rank, world_size = config.rank, config.world_size
    allreduce_strategy = config.allreduce_strategy.name

    if "sharded" in node.meta and node.meta["sharded"]:
        # Node was already sharded, skip
        return
    node.meta["sharded"] = True

    num_users = num_users_of_weight_node(node)
    if num_users > 1 or num_users == 0:
        ad_logger.warning(
            f"Expected exactly one user for the weight node {node.name}, but found {num_users}"
        )
        return

    # Shard weight using the unified function (also updates the parameter)
    all_weight_infos = get_all_weight_infos(node)
    # Parametrized nodes must have at least one weight (for debugging)
    assert len(all_weight_infos.weights) > 0, (
        f"Node {node.name} has no weights - weight mapping may be incorrect"
    )

    for weight_info in all_weight_infos.weights:
        _, weight_new_shape = shard_weight_tensor(
            gm=gm,
            weight_tensor=weight_info.tensor,
            param_key=weight_info.node_key,
            dim=dim,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
            fused_weight_dims=fused_weight_dims,
        )
        if quantization_cb is not None:
            quantization_cb(
                gm=gm,
                submod=weight_info.submod,
                node=node,
                weight_key=weight_info.node_key,
                weight_new_shape=weight_new_shape,
                dim=dim,
                rank=rank,
                world_size=world_size,
            )

    for bias_info in all_weight_infos.biases:
        if dim == 0:
            # update bias for dim 0 --> we can handle it like the weight
            shard_weight_tensor(
                gm=gm,
                weight_tensor=bias_info.tensor,
                param_key=bias_info.node_key,
                dim=dim,
                rank=rank,
                world_size=world_size,
                min_local_shape=min_local_shape,
                fused_weight_dims=fused_weight_dims,
            )
        elif rank != world_size - 1:
            # update the bias for dim 1 --> in this case only the last rank gets the bias to avoid
            # double counting it. For all other we will delete the bias.
            args = list(node.args)
            node_bias = args[2]
            args[2] = None
            node.args = tuple(args)
            gm.graph.erase_node(node_bias)
            bias_param_name = bias_info.node_key.rpartition(".")[-1]
            setattr(bias_info.submod, bias_param_name, None)
            gm._register_load_state_dict_pre_hook(
                partial(_load_hook_remove, param_key=bias_info.node_key)
            )

    # # # column shard with no gather: the output is sharded
    if not add_dist:
        return

    # figure out the right dist op (backend-aware)
    all_gather_op, all_reduce_op = _get_dist_ops(config.dist_backend)
    dist_lookup = {
        0: (all_gather_op, -1),
        1: (all_reduce_op, allreduce_strategy),
    }
    fn_dist, *dist_args = dist_lookup[dim]

    # add reduction node
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(fn_dist, args=(node,) + tuple(dist_args))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _update_node_args(node: Node, args: tuple) -> None:
    """Update the node's arguments with the new sharded arguments.

    For Node args: preserve the current value (may have been updated by other transforms).
    For non-Node args (shapes, sizes, indices): use the stored sharded value.

    This prevents ParameterUpdateInfo from reverting Node references that were
    intentionally updated by other transforms.
    """
    if "sharded" in node.meta and node.meta["sharded"]:
        return

    # Build new args: preserve current Node refs, apply stored non-Node values
    new_args = []
    for i, stored_arg in enumerate(args):
        if isinstance(stored_arg, Node):
            # Node args: preserve current value (may have been updated by other transforms)
            new_args.append(node.args[i])
        else:
            # Non-Node args (shapes, sizes, indices): use stored sharded value
            new_args.append(stored_arg)

    node.args = tuple(new_args)
    node.meta["sharded"] = True
    ad_logger.debug(f"Updated node {node}: sharded arguments are now {node.args}.")


def _insert_sharded_moe(
    gm: GraphModule,
    node: Node,
    config: ShardingTransformConfig,
    scale_names: Sequence[str] = (),
):
    """Update the torch_moe node with sharded weight lists,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.


    NOTE: allreduce_strategy is MANDATORY.
    """
    # get 2D EP+TP process grid and corresponding ranks
    ep_rank = config.process_grid[ShardingDim.EP]["p"]
    ep_size = config.process_grid[ShardingDim.EP]["w"]
    tp_rank = config.process_grid[ShardingDim.TP]["p"]
    tp_size = config.process_grid[ShardingDim.TP]["w"]
    allreduce_strategy = config.allreduce_strategy.name
    args = list(node.args)
    if allreduce_strategy is None:
        raise ValueError(f"allreduce_strategy must be set for MoE sharding on node {node.name}")
    scale_names = list(scale_names)

    # -- Handle selected_experts and final_scales sharding --
    selected_experts = args[1]
    final_scales = args[2]
    num_experts = len(args[3])

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

    args[1] = selected_experts_local
    args[2] = final_scales_local

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

        return lst[expert_start:expert_end], lst[:expert_start] + lst[expert_end:]

    w_up_list_sharded, w_up_list_to_remove = get_partition(args[3], ep_size, ep_rank)
    w_down_list_sharded, w_down_list_to_remove = get_partition(args[4], ep_size, ep_rank)
    w_gate_list_sharded, w_gate_list_to_remove = get_partition(args[5], ep_size, ep_rank)

    # if tp_size > 1, we do 2D EP+TP sharding.
    # we add TP sharding of all expert weights.
    for w_up in w_up_list_sharded + w_gate_list_sharded:
        shard_weight_tensor(
            gm=gm,
            weight_tensor=gm.get_parameter(w_up.target),
            param_key=w_up.target,
            dim=SplitDimension.COLUMN,
            rank=tp_rank,
            world_size=tp_size,
        )
    # here we don't need to add all-reduce: it's enough to have
    # just one all-reduce after the whole EP+TP sharded MoE node.
    for w_down in w_down_list_sharded:
        shard_weight_tensor(
            gm=gm,
            weight_tensor=gm.get_parameter(w_down.target),
            param_key=w_down.target,
            dim=SplitDimension.ROW,
            rank=tp_rank,
            world_size=tp_size,
        )

    # -- Update args --
    args[3] = w_up_list_sharded
    args[4] = w_down_list_sharded
    args[5] = w_gate_list_sharded

    # Shard scales for quantized ops
    scales_to_remove = []
    for i in range(len(scale_names) * 3):  # 3 layers (w1, w2, w3) × #scale_names per layer
        sharded, to_remove = get_partition(args[6 + i], ep_size, ep_rank)
        args[6 + i] = sharded
        scales_to_remove.extend(to_remove)

    ad_logger.debug(
        f"Updated node {node}: replaced original arguments {node.args} with sharded arguments {args}."
    )
    node.args = tuple(args)

    # -- add an all_reduce node --
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_reduce.default, args=(node, allreduce_strategy)
        )
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)

    eliminate_dead_code(gm)
    # Expert weights registered via gm.register_parameter() are top-level attributes.
    # Unlike submodules, these aren't cleaned up by eliminate_dead_code() or
    # delete_all_unused_submodules() - must delete manually after removing their get_attr nodes.
    for expert in (
        w_up_list_to_remove + w_down_list_to_remove + w_gate_list_to_remove + scales_to_remove
    ):
        try:
            del_attr_by_name(gm, expert.target)
        except AttributeError:
            ad_logger.warning(
                f"Failed to delete unused parameter {expert.target} from GraphModule."
            )


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
    config: ShardingTransformConfig,
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
    num_experts = shape(gate_up_blocks_node)[0]

    rank, world_size = config.rank, config.world_size
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
        red = gm.graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_reduce, args=(node, config.allreduce_strategy.name)
        )
        node.replace_all_uses_with(red)
        # keep dataflow: red(input=node)
        red.replace_input_with(red, node)


def _process_simple_shard(
    nodes_linear: Dict[Node, List[Node]],
    transform_container: ShardingTransformContainer,
    layer_type: LayerType = LayerType.MLP,
) -> int:
    # for every linear node:
    # --> row_split (dim 0 of weight) + all_gather (dim -1 of output)
    # if nodes_linear is a dict, flatten it to a 1D list of nodes
    config = transform_container.config
    if isinstance(nodes_linear, dict):
        nodes_linear = [n for group in nodes_linear.values() for n in group]

    num_simple_shards = 0
    for n in nodes_linear:
        num_simple_shards += int(
            transform_container.add(
                WeightShardingInfo.from_node(
                    n,
                    split_dim=SplitDimension.COLUMN,
                    config=config,
                    dist_op="all_gather",
                    min_local_shape=1,
                    layer_type=layer_type,
                )
            )
        )
    return num_simple_shards


def _process_ssm_sharding(
    layer_subgraph: LayerSubgraph,
    transform_container: ShardingTransformContainer,
) -> int:
    """
    Process the SSM sharding from the Mamba layer subgraph and update the view and split nodes accordingly.
    """
    config = transform_container.config
    world_size = config.world_size
    assert len(layer_subgraph.opening_nodes) == 1, (
        "Expecting exactly one opening node for SSM layer"
    )
    # Get subgraph between entry_node and next linear node
    subgraph_nodes = layer_subgraph.subgraph_nodes
    entry_node = layer_subgraph.opening_nodes[0]
    out_proj_node = layer_subgraph.terminating_node
    gm = entry_node.graph.owning_module

    ##############################################################
    ########## infer split sizes for in_proj and conv1d ##########
    ##############################################################
    # in_proj and conv1d are fused, followed up by split nodes. Infer split sizes:
    assert len(entry_node.users) == 1, "Expecting exactly one user for the entry node"
    split_node_0 = list(entry_node.users)[0]
    assert is_op(split_node_0, [torch.ops.aten.split_with_sizes]), (
        "Expecting split_with_sizes node for the entry node"
    )
    split_sizes_0 = split_node_0.args[1]
    # extract the single conv1d node
    conv1d_nodes = [
        n for n in subgraph_nodes if is_op(n, [torch.ops.auto_deploy.torch_causal_conv1d])
    ]
    assert len(conv1d_nodes) == 1, "Expecting exactly one conv1d node"
    conv1d_node = conv1d_nodes[0]
    assert len(conv1d_node.users) == 1, "Expecting exactly one user for the conv1d node"
    silu_node_1 = list(conv1d_node.users)[0]
    assert len(silu_node_1.users) == 1, "Expecting exactly one user for the silu node"
    split_node_1 = list(silu_node_1.users)[0]
    assert is_op(split_node_1, [torch.ops.aten.split_with_sizes]), (
        "Expecting split_with_sizes node for the split node"
    )
    split_sizes_1 = split_node_1.args[1]
    assert split_sizes_0[1] == sum(split_sizes_1)
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
            config=config,
            dist_op=None,
            min_local_shape=1,
            fused_weight_dims=tuple(fused_weight_dims["in_proj"]),
            layer_type=LayerType.SSM,
        )
    ):
        # the layer was already sharded. Skipping.
        return 0

    # # ##############################################################
    # # ############## update split nodes ############################
    # # ##############################################################
    split_args_0 = list(split_node_0.args)
    split_args_0[1] = [s // world_size for s in split_args_0[1]]
    split_args_1 = list(split_node_1.args)
    split_args_1[1] = [s // world_size for s in split_args_1[1]]
    transform_container.add(
        ParameterUpdateInfo(
            config=config,
            target_node=split_node_0.name,
            args=tuple(split_args_0),
        )
    )
    transform_container.add(
        ParameterUpdateInfo(
            config=config,
            target_node=split_node_1.name,
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
                fused_dims = tuple(v)
                break

        # Shard the weight tensor (also updates the parameter in the module)
        transform_container.add(
            WeightShardingInfo.from_node(
                list(weight_node.users)[0],
                split_dim=SplitDimension.COLUMN,
                config=config,
                dist_op=None,
                min_local_shape=1,
                fused_weight_dims=fused_dims,
                layer_type=LayerType.SSM,
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
                    config=transform_container.config, target_node=view_node.name, args=tuple(args)
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
            layer_type=LayerType.SSM,
        )
    )
    return 1


def _process_mla_sharding(
    layer_subgraph: LayerSubgraph,
    transform_container: ShardingTransformContainer,
) -> int:
    """
    Process the MLA sharding from the MLA layer subgraph and update the view and split nodes accordingly.

    We expect 5 linear nodes in the subgraph with the following sharding strategies:
    - q_a_proj: # gather (simple shard, output is replicated)
    - q_b_proj: # column-sharding  (output is head-distributed)
    - kv_a_proj # gather (simple shard, output is replicated)
    - kv_b_proj # column-sharding (output is head-distributed)
    - o_proj # row-sharding + all-reduce

    # Furthermore, we need to update the split nodes for the q_a_proj and kv_a_proj nodes.
    """
    config = transform_container.config
    world_size = config.world_size
    # check if we have exactly 2 openinng linear nodes (q_a_proj and kv_a_proj)
    assert len(layer_subgraph.opening_nodes) == 2, (
        "Expecting exactly two opening nodes for MLA layer"
    )
    q_a_proj, kv_a_proj = layer_subgraph.opening_nodes
    # extract q_b_proj and kv_b_proj nodes
    lin_nodes = list(filtered_nodes(layer_subgraph.subgraph_nodes, is_any_lin_op))
    assert len(lin_nodes) == 2, "Expecting exactly two linear nodes in the interior of the subgraph"
    q_b_proj, kv_b_proj = lin_nodes

    # extract o_proj node
    o_proj = layer_subgraph.terminating_node

    # add the sharding strategies for the q_a_proj and kv_a_proj nodes
    num_simple_shards = _process_simple_shard(
        [q_a_proj, kv_a_proj], transform_container, layer_type=LayerType.MLA
    )
    if num_simple_shards < 2:
        # it means that "someone else" already sharded these nodes. Skipping.
        return 0

    # extract the sub-subgraph from q_b_proj and kv_b_proj to o_proj
    sub_subgraph = subgraph(
        sources=[q_b_proj, kv_b_proj],
        boundary_condition=is_any_lin_op,
    )
    attention_subgraph = LayerSubgraph(
        opening_nodes=[q_b_proj, kv_b_proj],
        subgraph_nodes=sub_subgraph,
        terminating_node=o_proj,
        layer_type=LayerType.MLA,
        min_local_shape=layer_subgraph.min_local_shape,
    )
    # shard q_b_proj and kv_b_proj nodes
    num_column_row_shards = _process_column_sharding(attention_subgraph, transform_container)
    if num_column_row_shards < 2:
        # it means that "someone else" already sharded these nodes. Skipping.
        return 0

    # update "empty" and "expand" nodes' args. Reference in modeling_deepseek.py:
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    # We need to change the second argument num_heads to num_heads // world_size
    nodes_to_update = [
        n
        for n in layer_subgraph.subgraph_nodes
        if is_op(n, [torch.ops.aten.new_empty, torch.ops.aten.expand])
    ]
    for node_to_update in nodes_to_update:
        args = list(node_to_update.args)
        node_args = list(args[1])  # make the immutable list mutable
        node_args[1] = node_args[1] // world_size
        args[1] = node_args
        transform_container.add(
            ParameterUpdateInfo(
                target_node=node_to_update.name, config=transform_container.config, args=tuple(args)
            )
        )

    # shard o_proj node
    transform_container.add(
        WeightShardingInfo.from_node(
            o_proj,
            split_dim=SplitDimension.ROW,
            config=transform_container.config,
            dist_op="all_reduce",
            min_local_shape=layer_subgraph.min_local_shape,
            layer_type=layer_subgraph.layer_type,
        )
    )
    return 1


def _determine_fused_weight_dims(
    linear_nodes: List[Node],
) -> None:
    """
    Determine the fused weight dims for the given linear nodes and subgraph nodes.
    """
    if len(linear_nodes) != 1:
        return None
    linear_node = linear_nodes[0]
    fused_weight_dims = None
    if len(linear_nodes) == 1:
        linear_node = linear_nodes[0]
        # check if there are split nodes in the subgraph. They may indicate fused weights (e.g., QKV)
        linear_split_users = list(
            filtered_nodes(linear_node.users, ops=torch.ops.aten.split_with_sizes)
        )
        linear_slice_users = list(filtered_nodes(linear_node.users, ops=torch.ops.aten.slice))
        linear_chunk_users = list(filtered_nodes(linear_node.users, ops=torch.ops.aten.chunk))
        if len(linear_split_users) > 0:
            assert len(linear_split_users) == 1, (
                "Expecting exactly one split node for fused weights"
            )
            fused_weight_dims = linear_split_users[0].args[1]

        elif len(linear_slice_users) > 0:
            # we are probably in fused QKV case with single linear node and 3 slice nodes
            assert all(s.args[1] == 2 for s in linear_slice_users), (
                "Expecting slice nodes to slice tensor over dim=2"
            )
            fused_weight_dims = [s.args[3] - s.args[2] for s in linear_slice_users]
            assert fused_weight_dims, "fused weight dims cannot be empty"
            weight_dim = linear_node.meta["val"].shape[2]
            if sum(fused_weight_dims) != weight_dim:
                if fused_weight_dims[-1] > weight_dim:
                    fused_weight_dims[-1] = weight_dim - sum(fused_weight_dims[:-1])
                else:
                    ad_logger.warning(
                        f"Fused weight dims {fused_weight_dims} do not sum to weight dim {weight_dim}. Skipping."
                    )
                    return

        elif len(linear_chunk_users) > 0:
            assert len(linear_chunk_users) == 1, (
                "Expecting exactly one chunk node for fused weights"
            )
            num_chunks = linear_chunk_users[0].args[1]
            weight_dim = linear_node.meta["val"].shape[2]
            fused_weight_dims = [weight_dim // num_chunks] * num_chunks


def _find_upstream_qk_proj(node: Node, gm: GraphModule) -> Optional[str]:
    """
    Find the upstream q/k projection linear node from an RMSNorm input.

    Traverses backwards through pass-through tensor operations (view, reshape,
    type conversions, quantize/dequantize, etc.) to find the immediate producer.
    If that producer is a q_proj or k_proj linear, returns the weight name.

    Args:
        node: The input node to start traversing from (typically RMSNorm's activation input)
        gm: The graph module containing the nodes

    Returns:
        The weight name if a q_proj or k_proj is found as immediate upstream, None otherwise.

    TODO: Is there a more efficient way to do this?
    """
    # Pass-through ops that we traverse through (these don't change the semantic meaning)
    passthrough_ops = [
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.aten.contiguous,
        torch.ops.aten.clone,
        torch.ops.aten.to,
        torch.ops.aten._to_copy,
        torch.ops.aten.slice,
        torch.ops.aten.transpose,
        torch.ops.aten.permute,
    ]

    visited = set()
    current = node

    # Traverse through pass-through ops to find the actual producer
    while current is not None and current not in visited:
        visited.add(current)

        # Check if this is a linear operation (the producer we're looking for)
        if is_any_lin_op(current):
            try:
                weight_name = extract_weight_name(current)
                if weight_name and ("q_proj" in weight_name or "k_proj" in weight_name):
                    return weight_name
            except (AttributeError, AssertionError):
                pass
            # Found a linear but not q/k proj - this is not a QK norm
            return None

        # If this is a pass-through op, continue to its first input
        if is_op(current, passthrough_ops):
            # Get the first tensor input (skip non-tensor args like dims)
            tensor_inputs = [arg for arg in current.all_input_nodes if isinstance(arg, Node)]
            if tensor_inputs:
                current = tensor_inputs[0]
                continue

        # Hit a non-passthrough, non-linear op - stop searching
        break

    return None


def _shard_qk_norm(
    layer_subgraph: LayerSubgraph,
    linear_nodes: List[Node],
    transform_container: ShardingTransformContainer,
) -> int:
    """
    Shard RMSNorm ops in attention layers that operate on full hidden size.

    This function detects torch_rmsnorm ops that are true QK norms - i.e., norms that
    operate on the output of q_proj or k_proj. These global QK norms operate on
    flattened Q/K output [batch, seq, hidden_size] before reshape and need special
    handling for tensor parallelism:
    1. Weight sharding: The norm weight is column-sharded across ranks
    2. Global mean: Replaced with sharded_rmsnorm which uses all_reduce

    Detection criteria:
    1. The RMSNorm's input must trace back to a q_proj or k_proj linear operation
    2. The weight shape must match q/k projection output dimensions

    Example1: - Global Norm on all heads directly on flattened Q/K output (e.g. MiniMax):
        self.q_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_attention_heads, eps=config.rms_norm_eps)
        weight shape: [num_heads * head_dim] -> matches valid_sizes -> will be sharded
        Status: Needs sharding (weight matches q projection output dim)

    Example2: - Norm per head after reshaping to [batch, seq, num_heads, head_dim] (e.g. GLM 4.7):
        self.q_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        weight shape: [head_dim] -> does NOT match valid_sizes -> skipped
        Status: No sharding needed (weight doesn't match q/k output dims)

    Args:
        layer_subgraph: The attention layer subgraph
        linear_nodes: The linear nodes (q/k/v projections)
        transform_container: Container for sharding transformations

    Returns:
        Number of nodes added for sharding
    """
    if layer_subgraph.layer_type != LayerType.ATTENTION or layer_subgraph.terminating_node is None:
        return 0

    config = transform_container.config
    world_size = config.world_size
    gm = linear_nodes[0].graph.owning_module
    added_nodes = 0

    # Collect valid output dimensions from q/k/v projections
    # These are the only valid sizes for intermediate weights (e.g., q_norm, k_norm)
    valid_sizes = set()
    for lin_node in linear_nodes:
        try:
            wkey = extract_weight_name(lin_node)
            w = gm.get_parameter(wkey)
            valid_sizes.add(w.shape[0])  # q: num_heads*head_dim, k/v: num_kv_heads*head_dim
        except (AttributeError, AssertionError):
            pass

    # Find all intermediate weight nodes between q/k/v projections and o_proj.
    intermediate_weight_nodes = subgraph(
        sources=linear_nodes,
        sinks=[layer_subgraph.terminating_node],
        include=lambda n: n.op == "get_attr",
    )

    for weight_node in intermediate_weight_nodes:
        weight_key = weight_node.target

        # First check: is this weight consumed by a torch_rmsnorm op
        if len(list(weight_node.users)) == 0:
            continue

        user_node = list(weight_node.users)[0]
        if not is_op(user_node, torch.ops.auto_deploy.torch_rmsnorm):
            continue

        # Verify this is a true QK norm by checking its input traces back to q_proj or k_proj
        # This filters out input_layernorm which feeds INTO q/k/v projections (not after them)
        rmsnorm_input = user_node.args[0]  # activation input (not the weight)
        upstream_proj = _find_upstream_qk_proj(rmsnorm_input, gm)
        if upstream_proj is None:
            ad_logger.debug(
                f"Skipping {user_node.name} - input does not trace back to q_proj or k_proj "
                f"(likely input_layernorm or other non-QK norm)"
            )
            continue

        ad_logger.debug(f"Found QK norm {user_node.name} with upstream projection: {upstream_proj}")

        # Try to get the parameter
        try:
            param = gm.get_parameter(weight_key)
        except AttributeError:
            ad_logger.debug(f"Could not get parameter for {weight_key}, skipping")
            continue

        # Only shard 1D weights (RMSNorm weights are always 1D)
        if param.dim() != 1:
            ad_logger.debug(
                f"Skipping intermediate weight {weight_key} with dim={param.dim()} (not 1D)"
            )
            continue

        # Check if the weight size is divisible by world_size
        if param.shape[0] % world_size != 0:
            ad_logger.debug(
                f"Skipping intermediate weight {weight_key} with shape {param.shape} "
                f"(not divisible by world_size={world_size})"
            )
            continue

        # Check if weight size matches one of the q/k/v projection output dimensions
        # This filters out per-head norms and unrelated weights like inv_freq
        if valid_sizes and param.shape[0] not in valid_sizes:
            ad_logger.debug(
                f"Skipping intermediate weight {weight_key} with shape {param.shape} "
                f"(not in valid_sizes={valid_sizes}, likely per-head norm)"
            )
            continue

        # Add RMSNormShardingInfo to replace with sharded_rmsnorm
        # This handles both weight sharding and op replacement in one transform
        if transform_container.add(
            RMSNormShardingInfo.from_node(
                user_node,
                config=config,
                world_size=world_size,
            )
        ):
            ad_logger.debug(
                f"Added RMSNormShardingInfo for {user_node.name} "
                f"(will replace with sharded_rmsnorm for global mean)"
            )
            added_nodes += 1

    return added_nodes


def _process_column_sharding(
    layer_subgraph: LayerSubgraph,
    transform_container: ShardingTransformContainer,
) -> int:
    """
    Parse the column sharding from the candidate nodes and update the view and split nodes accordingly.
    """
    config = transform_container.config
    world_size = config.world_size
    linear_nodes = layer_subgraph.opening_nodes
    subgraph_nodes = layer_subgraph.subgraph_nodes
    fused_weight_dims = _determine_fused_weight_dims(linear_nodes)

    added_nodes: int = 0
    for linear_node in linear_nodes:
        added_nodes += transform_container.add(
            WeightShardingInfo.from_node(
                linear_node,
                split_dim=SplitDimension.COLUMN,
                config=config,
                dist_op=None,  # for column sharding, no dist op is performed
                min_local_shape=layer_subgraph.min_local_shape,
                fused_weight_dims=fused_weight_dims,
                layer_type=layer_subgraph.layer_type,
            )
        )
    if added_nodes == 0:
        ad_logger.debug("No nodes were added for column sharding. Skipping.")
        return 0

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
                ParameterUpdateInfo(target_node=view_node.name, config=config, args=tuple(args))
            )
            ad_logger.debug(f"\nUpdated view node {view_node} arguments to {view_node.args}")

    # if fused_weight_dims is provided, we need to update all split sizes
    if fused_weight_dims is not None:
        assert world_size is not None, "World size is required to update the split node params"

        # fused weight may either be processed by several slice nodes or a single split node
        linear_node = linear_nodes[0]
        split_nodes = list(filtered_nodes(linear_node.users, ops=[torch.ops.aten.split_with_sizes]))
        slice_nodes = list(filtered_nodes(linear_node.users, ops=[torch.ops.aten.slice]))
        if len(split_nodes) > 0:
            user = split_nodes[0]
            orig_sizes = user.args[1]
            new_sizes = [orig_sizes[i] // world_size for i in range(len(orig_sizes))]
            args = list(user.args)
            args[1] = new_sizes
            transform_container.add(
                ParameterUpdateInfo(config=config, target_node=user.name, args=tuple(args))
            )
        elif len(slice_nodes) > 0:
            for slice_node in slice_nodes:
                args = list(slice_node.args)
                args[2] = args[2] // world_size
                args[3] = args[3] // world_size
                transform_container.add(
                    ParameterUpdateInfo(
                        config=config,
                        target_node=slice_node.name,
                        args=tuple(args),
                    )
                )
        # chunk nodes do not need to be updated

    # Shard intermediate weights (e.g. q/k/v -> q_norm, k_norm ... -> o_proj) for attention layers
    added_nodes += _shard_qk_norm(layer_subgraph, linear_nodes, transform_container)

    return added_nodes


########################################################
#  Topological pattern matching functions
########################################################


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
        config = transform_container.config.factory_config
    elif source == ShardingSource.MANUAL:
        config = transform_container.config.manual_config
    else:
        raise ValueError(f"Unsupported sharding source: {source}")
    tp_plan = config["tp_plan"]

    num_shards = 0
    num_simple_shards = 0
    num_row_col_shards = 0
    num_attention_shards = 0
    num_ssm_shards = 0
    linear_nodes = list(filtered_nodes(gm.graph.nodes, is_any_lin_op))

    # use layer_subgraphs to determine the layer_type
    # and check the validity of the sharding transform
    layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)

    for lin_node in linear_nodes:
        # use node's weight name to get the module name
        weight_name = extract_weight_name(lin_node)
        # get the parent layer_subgraph
        layer_subgraph = [
            layer
            for layer in layer_subgraphs
            if lin_node in layer.opening_nodes or lin_node == layer.terminating_node
        ]
        if len(layer_subgraph) == 1:
            layer_subgraph = layer_subgraph[0]
            layer_type = layer_subgraph.layer_type
        else:
            if lin_node in unprocessed_linear_nodes:
                layer_type = LayerType.UNKNOWN
            else:
                ad_logger.warning(
                    f"Failed to find the parent layer_subgraph for linear node {lin_node}. "
                    f"May result in incorrect sharding."
                )

        # use regex to find if module_name matches any of the keys in sharding_config
        for key in tp_plan.keys():
            pattern_string = "*" + key + "*"
            # convert it to regex. Escape dots, replace * with .*
            # First, we substitute * with an unlikely character, e.g. @
            # Then we escape dots, and finally we replace @ with .*
            pattern_string = pattern_string.replace("*", "@")
            pattern_regex = re.escape(pattern_string).replace("@", ".*")
            if re.match(pattern_regex, weight_name):
                # we have a match. Get the config for this layer
                config = tp_plan[key]

                if config == "colwise":
                    _process_column_sharding(
                        layer_subgraph=layer_subgraph,
                        transform_container=transform_container,
                    )
                elif config == "rowwise":
                    if transform_container.add(
                        WeightShardingInfo.from_node(
                            lin_node,
                            split_dim=SplitDimension.ROW,
                            config=transform_container.config,
                            dist_op="all_reduce",
                            layer_type=layer_type,
                        )
                    ):
                        if layer_type == LayerType.ATTENTION:
                            num_attention_shards += 1
                        num_row_col_shards += 1
                elif config == "mamba":
                    if _process_ssm_sharding(layer_subgraph, transform_container) > 0:
                        num_ssm_shards += 1
                        num_row_col_shards += 1

                elif "sequence" in config:
                    # TODO: Sequence parallelism is not supported yet.
                    ad_logger.warning("Sequence parallelism is not supported yet. Skipping.")
                elif "local" in config:
                    # Check if this applies to shared experts in EP parallelism.
                    # If yes, apply the TP col-row shard.
                    if "shared" in weight_name:
                        col_row_action = config.replace("local_", "")
                        if col_row_action == "colwise":
                            transform_container.add(
                                WeightShardingInfo.from_node(
                                    lin_node,
                                    split_dim=SplitDimension.COLUMN,
                                    config=transform_container.config,
                                    dist_op=None,
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
                            config=transform_container.config,
                            dist_op="all_gather",
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
    world_size = config.world_size
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
            _process_ssm_sharding(gm, in_proj_node, transform_container, config=config)
        )

    ad_logger.info(f"Found {num_ssm_shards} SSM shards")
    return TransformInfo(
        skipped=False, num_matches=num_ssm_shards, is_clean=False, has_valid_shapes=False
    )


def detect_column_row_shard(
    gm: GraphModule,
    transform_container: ShardingTransformContainer,
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
    world_size = config.world_size

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
    num_mha_shards = 0
    num_mla_shards = 0
    num_column_row_shards = 0
    for layer in layer_subgraphs:
        opening = layer.opening_nodes
        closing = layer.terminating_node
        layer_subgraph = layer.subgraph_nodes
        nodes_linear = opening + [closing]

        attention_nodes = list(filtered_nodes(layer_subgraph, is_any_attention_op))
        min_local_shape = 1

        if config.simple_shard_only or layer.layer_type == LayerType.UNKNOWN:
            ad_logger.debug(
                f"Forcing Simple Shard on nodes: {nodes_linear} with layer type: {layer.layer_type}"
            )
            num_simple_shards += _process_simple_shard(
                nodes_linear, transform_container, layer_type=layer.layer_type
            )
            continue

        if layer.layer_type == LayerType.SSM:
            # Mamba layers need special handling due to the fused weights for in_proj and conv1d
            num_ssm_shards += _process_ssm_sharding(
                layer,
                transform_container,
            )
            continue

        if layer.layer_type == LayerType.MLA:
            num_mla_shards += _process_mla_sharding(
                layer,
                transform_container,
            )
            continue

        if layer.layer_type == LayerType.ATTENTION:
            ad_logger.debug(f"Found attention nodes in layer subgraph: {attention_nodes}")
            # Extract head dimension. We cannot shard below the head_dim size.
            # Assume that head_dim is the last (innermost) dimension of the tensor
            min_local_shape = shape(attention_nodes[0])[-1]
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
                            transform_container,
                            layer_type=layer.layer_type,
                        )
                        # TODO: handle the case where num_kv_heads is not divisible by world_size
                        continue

        # column-row sharding
        _process_column_sharding(
            layer_subgraph=layer,
            transform_container=transform_container,
        )

        # shard single row node
        if transform_container.add(
            WeightShardingInfo.from_node(
                closing,
                split_dim=SplitDimension.ROW,
                config=config,
                dist_op="all_reduce",
                min_local_shape=min_local_shape,
                layer_type=layer.layer_type,
            )
        ):
            num_column_row_shards += 1
            if layer.layer_type == LayerType.ATTENTION:
                num_mha_shards += 1

    # simple shard remaining linear nodes
    if config.shard_all_unprocessed:
        num_simple_shards += _process_simple_shard(unprocessed_linear_nodes, transform_container)
    num_column_row_shards += num_ssm_shards + num_mla_shards
    num_shards = num_simple_shards + num_column_row_shards
    ad_logger.info(
        f"Heuristics found {num_shards} TP shards. Simple: {num_simple_shards}, "
        f"row-col: {num_column_row_shards} (including: ssm: {num_ssm_shards}, "
        f"mha: {num_mha_shards}, mla: {num_mla_shards})"
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
    config = transform_container.config
    rank, world_size = config.rank, config.world_size
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
        lhs_batch_size = shape(lhs_tensor)[0]
        rhs_batch_size = shape(rhs_tensor)[0]

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
                config=config,
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

    config = transform_container.config
    world_size = config.world_size
    if world_size < 2:
        ad_logger.info("Skipping EP sharding for single device")
        return TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)

    assert isinstance(gm, GraphModule), "Expecting GraphModule"
    num_moe_patterns = 0
    for node in list(gm.graph.nodes):
        if not is_any_moe_op(node):
            continue
        if transform_container.add(EPShardingInfo.from_node(node, config=config)):
            num_moe_patterns += 1

    ad_logger.info(f"Found {num_moe_patterns} MoE patterns")

    return TransformInfo(
        skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
    )
