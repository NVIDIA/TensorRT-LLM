import atexit
import os
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from tensorrt_llm.functional import (AllReduceConfig, AllReduceFusionOp,
                                     AllReduceParams, AllReduceStrategy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

_thread_local = threading.local()


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


def get_deepseek_allreduce_workspace(mapping: Mapping) -> torch.LongTensor:
    if not hasattr(_thread_local, 'deepseek_allreduce_workspaces'):
        _thread_local.deepseek_allreduce_workspaces = {}
    deepseek_allreduce_workspaces = _thread_local.deepseek_allreduce_workspaces
    if mapping not in deepseek_allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_allreduce_fusion_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_auto(mapping.tp_size),
        )
        deepseek_allreduce_workspaces[mapping] = (ipc_buffers, workspace)
    return deepseek_allreduce_workspaces[mapping][1]


def allreduce(
    input: torch.Tensor,
    workspace: Optional[torch.LongTensor],
    mapping: Mapping,
    strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    config: AllReduceConfig = AllReduceConfig(0),
    all_reduce_params: Optional[AllReduceParams] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        mapping (Mapping):  The parallel mapping.
        strategy (AllReduceStrategy): NCCL delegates all-reduce to NCCL while ONESHOT and TWOSHOT are custom latency-optimal algorithms.
            AUTO chooses amongst the three based on a message-size heuristic.
        config (AllReduceConfig): The config for custom allreduce kernels.
        all_reduce_params (AllReduceParams): The parameters for the fused ops into the allreduce op.
    Returns:
        The reduced tensor and an optional intermediate tensor if fused.
    '''
    if mapping.tp_size == 1 or (all_reduce_params is not None and
                                all_reduce_params.enable_allreduce == False):
        return input

    if all_reduce_params is None:
        all_reduce_params = AllReduceParams()
    is_fused = all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM or \
        all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8 or \
        all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
    reduce_fusion_inputs = []
    if is_fused:
        if all_reduce_params.has_bias() == 1:
            reduce_fusion_inputs.append(all_reduce_params.bias)
        reduce_fusion_inputs.append(all_reduce_params.residual)
        if all_reduce_params.has_affine() == 1:
            reduce_fusion_inputs.append(all_reduce_params.norm_weight)
        if all_reduce_params.has_scale() == 1:
            reduce_fusion_inputs.append(all_reduce_params.scale)

    out = torch.ops.trtllm.allreduce(
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
    if all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        return out[0], out[1], out[2]
    elif is_fused:
        return out[0], out[1]
    else:
        return out[0]


def userbuffers_allreduce_finalize(
        input: torch.Tensor,
        force_applying_finalize: bool = False) -> torch.Tensor:
    output = torch.ops.trtllm.userbuffers_allreduce_finalize(
        input, force_applying_finalize)
    return output


def allgather(input: torch.Tensor,
              mapping: Mapping,
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
        mapping (Mapping):  The parallel mapping.
        gather_dim (int): Gather along given dimension. By default -1.
    Returns:
        The gathered tensor.
    '''
    if mapping.tp_size == 1:
        return input

    output = torch.ops.trtllm.allgather(
        input,
        mapping.tp_group,
    )

    if gather_dim < 0:
        gather_dim += input.ndim

    output = torch.movedim(output, 0, gather_dim)
    input_shape = input.size()
    output = output.reshape(input_shape[:gather_dim] +
                            (mapping.tp_size * input_shape[gather_dim], ) +
                            input_shape[gather_dim + 1:])
    return output


def reducescatter(input: torch.Tensor,
                  mapping: Mapping,
                  scatter_dim: int = -1) -> torch.Tensor:
    if mapping.tp_size == 1:
        return input

    output = torch.ops.trtllm.reducescatter(
        input,
        mapping.tp_group,
    )

    if scatter_dim < 0:
        scatter_dim += input.ndim

    output = torch.movedim(output, 0, scatter_dim)
    input_shape = input.size()
    output = output.reshape(input_shape[:scatter_dim] +
                            (input_shape[scatter_dim] // mapping.tp_size, ) +
                            input_shape[scatter_dim + 1:])
    return output


class AllReduce(nn.Module):

    def __init__(self,
                 mapping: Mapping,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO):
        super().__init__()

        self.mapping = mapping
        self.workspace = None
        self.strategy = strategy
        if self.mapping.tp_size > 1:
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
                           self.mapping,
                           all_reduce_params=all_reduce_params,
                           strategy=self.strategy)
        return output


class DeepseekAllReduce(nn.Module):

    def __init__(self, mapping: Mapping):
        super().__init__()
        self.mapping = mapping
        self.workspace = None
        if self.mapping.tp_size > 1:
            self.workspace = get_deepseek_allreduce_workspace(mapping)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reduce_fusion_inputs: List[torch.Tensor],
        eps: float,
        fusion_op: AllReduceFusionOp,
    ) -> List[torch.Tensor]:
        """
        hidden_states: hidden_states of the model
        reduce_fusion_inputs: [residual, norm_weight, scale (if using FP4 quantization)]
        eps: epsilon for RMSNorm
        fusion_op: AllReduceFusionOp Type, currently supports RMSNorm:
          * RESIDUAL_RMS_NORM: allreduce + residual + Norm
          * RESIDUAL_RMS_NORM_QUANT_NVFP4: allreduce + residual + Norm + fp4 quantization

        output:
          * [hidden_states, residual] if using RESIDUAL_RMS_NORM fusion_op
          * [act_fp4, act_sf, residual] if using RESIDUAL_RMS_NORM_QUANT_NVFP4 fusion_op
        """

        output = torch.ops.trtllm.deepseek_allreduce_fusion(
            input=hidden_states,
            workspace=self.workspace,
            reduce_fusion_inputs=reduce_fusion_inputs,
            rank=self.mapping.tp_rank,
            nranks=self.mapping.tp_size,
            eps=eps,
            fusion_op=fusion_op,
        )

        if len(output) == 0:
            raise ValueError(f"Unsupported fusion op: {fusion_op}")

        return output


class PPComm:
    # PP communication using torch.distributed with nccl backend
    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        if not dist.is_initialized():
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="nccl",
                                    init_method=init_method,
                                    world_size=global_mapping.world_size,
                                    rank=global_mapping.rank)
            atexit.register(self._cleanup)

        # Force NCCL initialization and rank population via PyTorch distributed barrier.
        # This is necessary for NOW if using pp + tp because our custom nccl allreduce
        # op for tp groups can interfere with PyTorch's NCCL initialization when PyTorch
        # distributed performs the first comm. op and kick off nccl init. The barrier here
        # ensures proper NCCL setup and GPU-procs binding at beginning.
        dist.barrier(device_ids=[torch.cuda.current_device()])

    def _cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def send(self,
             tensor: torch.Tensor,
             dest: Optional[int] = None,
             tag: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()
        dist.send(tensor, dest, tag=tag)

    def recv(self,
             tensor: torch.Tensor,
             src: Optional[int] = None,
             tag: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()
        dist.recv(tensor, src, tag=tag)
