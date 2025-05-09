import threading
from typing import Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

_thread_local = threading.local()


def get_allreduce_workspace(mapping: Mapping) -> torch.LongTensor:
    if not hasattr(_thread_local, 'allreduce_workspaces'):
        _thread_local.allreduce_workspaces = [{}
                                              for _ in range(mapping.pp_size)]
    allreduce_workspaces = _thread_local.allreduce_workspaces[mapping.pp_rank]
    if mapping not in allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_allreduce_fusion_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_auto(
                mapping.tp_size, support_deterministic=False),
        )
        allreduce_workspaces[mapping] = (ipc_buffers, workspace)
    return allreduce_workspaces[mapping][1]


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
        """
        AllReduce is a module that performs an all-reduce operation on a tensor.

        Args:
            mapping (Mapping):  The parallel mapping config.
            strategy (AllReduceStrategy):
                Three types of all-reduce strategies are supported:
                - UB: AllReduce uses user-buffer based all-reduce kernel. Supported ops:
                    - RESIDUAL_RMS_NORM
                    - RESIDUAL_RMS_NORM_QUANT_FP8
                    - RESIDUAL_RMS_NORM_QUANT_NVFP4

                - NCCL: AllReduce delegates all-reduce to NCCL MIN_LATENCY mode kernel. Supported ops:
                    - NONE (AllReduce only)
                    - RESIDUAL_RMS_NORM

                - MIN_LATENCY: AllReduce uses MIN_LATENCY mode kernel. Supported ops:
                    - NONE (AllReduce only)
                    - RESIDUAL_RMS_NORM
                    - RESIDUAL_RMS_NORM_QUANT_FP8
                    - RESIDUAL_RMS_NORM_QUANT_NVFP4
                    - RESIDUAL_RMS_NORM_OUT_QUANT_FP8
                    - RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4

                - AUTO: AUTO chooses between NCCL and MIN_LATENCY mode based on a heuristic policy.

        Note:
            For the reference implementation for each pattern, please refer to the following unit test:
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/multi_gpu/test_allreduce.py
        """

        self.mapping = mapping
        self.workspace = None
        self.strategy = strategy

        if self.mapping.tp_size > 1:
            # When Strategy is UB, it is guaranteed that the workspace is not used.
            if self.strategy != AllReduceStrategy.UB:
                self.workspace = get_allreduce_workspace(self.mapping)

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        '''
        The input tensors in the different ranks must have the same shape.
        The output tensor will have that same shape with the input tensor.
        The output tensor will be replicated among the TP group.
        Note that it is not an in-place operation like torch.distributed.all_reduce.

        That operation is implemented using a torch op that wraps the NCCL all-reduce
        collective operation and custom one-shot/two-shot allreduce kernels. See
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
        for details.

        Args:
            input (Tensor): The input tensor.
            all_reduce_params (AllReduceParams): The parameters for the fused ops into the allreduce op.
        Returns:
            A tensor lists with different tensor outptus according to the fusion_op.
            NONE: [hidden_states]
            RESIDUAL_RMS_NORM: [hidden_states, residual]
            RESIDUAL_RMS_NORM_QUANT_FP8: [norm_quant, residual]
            RESIDUAL_RMS_NORM_OUT_QUANT_FP8: [norm, norm_quant, residual]
            RESIDUAL_RMS_NORM_QUANT_NVFP4: [norm_quant_fp4, scale_factor, residual]
            RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: [norm, norm_quant_fp4, scale_factor, residual]
        '''
        if self.mapping.tp_size == 1 or (all_reduce_params is not None
                                         and all_reduce_params.enable_allreduce
                                         == False):
            return input

        # Assume using no fusion allreduce here
        if all_reduce_params is None:
            all_reduce_params = AllReduceParams()

        output = torch.ops.trtllm.allreduce(
            input=input,
            residual=all_reduce_params.residual,
            norm_weight=all_reduce_params.norm_weight,
            scale=all_reduce_params.scale,
            bias=all_reduce_params.bias,
            workspace=self.workspace,
            group=self.mapping.tp_group,
            strategy=self.strategy,
            op=all_reduce_params.fusion_op,
            eps=all_reduce_params.eps,
        )

        return output if len(output) > 1 else output[0]


def allreduce_argmax(
    input: torch.Tensor,
    all_reduce: AllReduce,
    padding: int = 0,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:

    assert dim == -1 or dim == input.dim(
    ) - 1, "Only supports allreduce_argmax along last dimension."

    tp_size = all_reduce.mapping.tp_size
    tp_rank = all_reduce.mapping.tp_rank

    # If TP1, call argmax directly.
    if tp_size == 1:
        return torch.argmax(input, dim, keepdim=keepdim)

    # Get per-rank number of elements.
    per_rank_elements = input.shape[dim]

    # For last rank, we need to remove the padding.
    if tp_rank == tp_size - 1 and padding > 0:
        input = input[..., :-padding]

    # First, take argmax locally.
    local_argmax = torch.argmax(input, dim, keepdim=True)
    local_max = input.gather(dim=dim, index=local_argmax).float()
    local_argmax = local_argmax.squeeze(dim)
    local_max = local_max.squeeze(dim)

    # Create a tensor to store both local argmax and local max, and then use allreduce to communicate across ranks.
    # To reduce the number of transfers, we pack int32 as float32, hoping that adding 0.0f results in identical result.
    # Since the vocab size is almost always <2^30, there should be no NaNs or infs that will mess up the argmax.
    expanded_max = torch.zeros(2,
                               *local_argmax.shape,
                               tp_size,
                               device=input.device,
                               dtype=torch.float)
    expanded_max[0, ..., tp_rank] = local_argmax.int().view(dtype=torch.float)
    expanded_max[1, ..., tp_rank] = local_max
    expanded_max = all_reduce(expanded_max)

    # Finally, take argmax again to get the global argmax.
    argmax_rank = torch.argmax(expanded_max[1], dim=-1, keepdim=True)
    output = expanded_max[0].gather(dim=-1, index=argmax_rank).view(
        dtype=torch.int32).long() + argmax_rank * per_rank_elements

    if not keepdim:
        output = output.squeeze(dim)

    return output


class MoEAllReduce(nn.Module):

    def __init__(self, mapping: Mapping):
        """
        MoEAllReduce is a module that performs a specific fused MoE reduction
        followed by a regular AR + RMS norm.

        Args:
            mapping (Mapping):  The parallel mapping config.

        Notes:
            Support pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
            expert_reduction = torch.sum(active_experts_token_input *
                                        scale.unsqueeze(-1),
                                        dim=0)
            output_add = expert_reduction + shared_expert_output
            output_residual = output_add + residual
            output_hidden_states = rms_norm(output_residual, norm_weight, eps)
        """
        super().__init__()
        self.mapping = mapping
        self.workspace = get_allreduce_workspace(self.mapping)

    def forward(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        device_num_experts: torch.Tensor,
        scale_input: torch.Tensor,
        active_experts_token_input: torch.Tensor,
        token_input: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Args:
            residual: residual tensor
            norm_weight: RMS norm weight
            device_num_experts: number of experts per device
            scale_input: experts to token score
            active_experts_token_input: per token per expert input
            token_input: per token input, shared expert output
            eps: epsilon for RMSNorm

        Output:
            hidden_states: hidden_states of the model
            residual: residual tensor
        """
        return torch.ops.trtllm.moe_allreduce(
            residual=residual,
            norm_weight=norm_weight,
            device_num_experts=device_num_experts,
            scale_input=scale_input,
            active_experts_token_input=active_experts_token_input,
            token_input=token_input,
            workspace=self.workspace,
            rank=self.mapping.tp_rank,
            nranks=self.mapping.tp_size,
            eps=eps,
        )
