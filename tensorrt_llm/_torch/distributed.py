import enum
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.functional import (AllReduceConfig, AllReduceFusionOp,
                                     AllReduceParams, AllReduceStrategy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

_thread_local = threading.local()


class TensorParallelMode(str, enum.Enum):
    COLUMN = 'column'
    ROW = 'row'

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0


@dataclass(kw_only=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    gpus_per_node: int = 8
    tensor_parallel_mode: Optional[TensorParallelMode] = None
    gather_output: bool = False


def get_allreduce_workspace(mapping: Mapping) -> torch.LongTensor:
    if not hasattr(_thread_local, 'allreduce_workspaces'):
        _thread_local.allreduce_workspaces = {}
    allreduce_workspaces = _thread_local.allreduce_workspaces
    if mapping not in allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_auto(mapping.tp_size),
        )
        allreduce_workspaces[mapping] = (ipc_buffers, workspace)
    return allreduce_workspaces[mapping][1]


def allreduce(
    input: torch.Tensor,
    workspace: Optional[torch.LongTensor],
    parallel_config: ParallelConfig,
    strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    config: AllReduceConfig = AllReduceConfig(0),
    all_reduce_params: Optional[AllReduceParams] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    Add an operation that performs a collective all-reduce.

    The input tensors in the different ranks must have the same shape.
    The output tensor will have that same shape with the input tensor.
    The output tensor will be replicated among the TP group.
    Noting that it is not an in-place operation like torch.distributed.all_reduce.

    That operation is implemented using a torch op that wraps the NCCL all-reduce
    collective operation and custom one-shot/two-shot allreduce kernels. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
    for details.

    Args:
        input (Tensor): The input tensor.
        parallel_config (ParallelConfig):  The parallel config.
        strategy (AllReduceStrategy): NCCL delegates all-reduce to NCCL while ONESHOT and TWOSHOT are custom latency-optimal algorithms.
            AUTO chooses amongst the three based on a message-size heuristic.
        config (AllReduceConfig): The config for custom allreduce kernels.
        all_reduce_params (AllReduceParams): The parameters for the fused ops into the allreduce op.
    Returns:
        The reduced tensor and an optional intermediate tensor if fused.
    '''
    if parallel_config.tensor_parallel_size == 1 or (
            all_reduce_params is not None
            and all_reduce_params.enable_allreduce == False):
        return input

    mapping = Mapping(
        world_size=parallel_config.tensor_parallel_size,
        tp_size=parallel_config.tensor_parallel_size,
        rank=parallel_config.tensor_parallel_rank,
        gpus_per_node=parallel_config.gpus_per_node,
    )

    if all_reduce_params is None:
        all_reduce_params = AllReduceParams()
    is_fused = all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    reduce_fusion_inputs = []
    if is_fused:
        if all_reduce_params.has_bias() == 1:
            reduce_fusion_inputs.append(all_reduce_params.bias)
        reduce_fusion_inputs.append(all_reduce_params.residual)
        if all_reduce_params.has_affine() == 1:
            reduce_fusion_inputs.append(all_reduce_params.norm_weight)
        if all_reduce_params.has_scale() == 1:
            reduce_fusion_inputs.append(all_reduce_params.scale)

    final_output, inter_output = torch.ops.trtllm.allreduce(
        input,
        workspace,
        reduce_fusion_inputs,
        mapping.tp_group,
        int(strategy),
        int(config),
        int(all_reduce_params.fusion_op),
        float(all_reduce_params.eps),
        all_reduce_params.has_affine(),
        all_reduce_params.has_bias(),
        all_reduce_params.has_scale(),
    )

    if is_fused:
        return final_output, inter_output
    else:
        return final_output


def userbuffers_allreduce_finalize(input: torch.Tensor) -> torch.Tensor:
    output = torch.ops.trtllm.userbuffers_allreduce_finalize(input)
    return output


def allgather(input: torch.Tensor,
              parallel_config: ParallelConfig,
              gather_dim: int = -1) -> torch.Tensor:
    '''
    Add an operation that performs a collective all-gather.

    The input tensors in the different ranks must have the same shape.
    The output tensor will be replicated among the TP group.

    Given the 'section_size = input.shape[gather_dim]', each rank
    contributes a section of its input tensor that correspond to
    'rank*section_size:(rank+1)*section_size',
    and 'output.shape[gather_dim] = input.shape[gather_dim] * tp_group_size'.

    That operation is implemented using a torch op that wraps the NCCL all-gather
    collective operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather
    for details.

    Args:
        input (Tensor): The input tensor.
        parallel_config (ParallelConfig):  The parallel config.
        gather_dim (int): Gather along given dimension. By default -1.
    Returns:
        The gathered tensor.
    '''
    if parallel_config.tensor_parallel_size == 1:
        return input

    mapping = Mapping(
        world_size=parallel_config.tensor_parallel_size,
        tp_size=parallel_config.tensor_parallel_size,
        rank=parallel_config.tensor_parallel_rank,
        gpus_per_node=parallel_config.gpus_per_node,
    )

    output = torch.ops.trtllm.allgather(
        input,
        mapping.tp_group,
    )

    if gather_dim < 0:
        gather_dim += input.ndim

    output = torch.movedim(output, 0, gather_dim)
    input_shape = input.size()
    output = output.reshape(input_shape[:gather_dim] +
                            (parallel_config.tensor_parallel_size *
                             input_shape[gather_dim], ) +
                            input_shape[gather_dim + 1:])
    return output


def reducescatter(input: torch.Tensor,
                  parallel_config: ParallelConfig,
                  scatter_dim: int = -1) -> torch.Tensor:
    if parallel_config.tensor_parallel_size == 1:
        return input

    mapping = Mapping(
        world_size=parallel_config.tensor_parallel_size,
        tp_size=parallel_config.tensor_parallel_size,
        rank=parallel_config.tensor_parallel_rank,
    )

    output = torch.ops.trtllm.reducescatter(
        input,
        mapping.tp_group,
    )

    if scatter_dim < 0:
        scatter_dim += input.ndim

    output = torch.movedim(output, 0, scatter_dim)
    input_shape = input.size()
    output = output.reshape(input_shape[:scatter_dim] +
                            (input_shape[scatter_dim] //
                             parallel_config.tensor_parallel_size, ) +
                            input_shape[scatter_dim + 1:])
    return output


class AllReduce(nn.Module):

    def __init__(self,
                 parallel_config: ParallelConfig,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO):
        super().__init__()

        self.parallel_config = parallel_config
        self.tp_size = self.parallel_config.tensor_parallel_size
        self.tp_rank = self.parallel_config.tensor_parallel_rank
        self.gpus_per_node = self.parallel_config.gpus_per_node

        self.workspace = None
        self.strategy = strategy
        if self.tp_size > 1:
            mapping = Mapping(
                world_size=self.tp_size,
                tp_size=self.tp_size,
                rank=self.tp_rank,
                gpus_per_node=self.gpus_per_node,
            )
            if self.strategy != AllReduceStrategy.UB:
                self.workspace = get_allreduce_workspace(mapping)

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        output = allreduce(input,
                           self.workspace,
                           self.parallel_config,
                           all_reduce_params=all_reduce_params,
                           strategy=self.strategy)
        return output
