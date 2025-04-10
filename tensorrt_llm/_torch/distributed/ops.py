import threading
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

_thread_local = threading.local()


def get_allreduce_workspace(mapping: Mapping) -> torch.LongTensor:
    if not hasattr(_thread_local, f'allreduce_workspaces_{mapping.pp_rank}'):
        setattr(_thread_local, f'allreduce_workspaces_{mapping.pp_rank}', {})

    allreduce_workspaces = getattr(_thread_local,
                                   f'allreduce_workspaces_{mapping.pp_rank}')
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


def allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    gather_dim: int = -1,
    all_rank_split_size: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Add an operation that performs a collective all-gather.

    If 'all_rank_split_size' is 'None', the input tensors in the different ranks must have the same shape.
    Otherwise, 'all_rank_split_size[i]' must be 'input.shape[gather_dim]' at rank i, and the input tensors in
    the different ranks can only differ in shape at dimension `gather_dim`.

    The input tensors in the same TP group are concatenated at dimension 'gather_dim' to produce the output tensor.
    If 'all_rank_split_size' is 'None', 'output.shape[gather_dim] = input.shape[gather_dim] * tp_group_size'.
    Otherwise, 'output.shape[gather_dim] = sum(all_rank_split_size)'.

    That operation is implemented using a torch op that wraps the NCCL all-gather collective operation or
    the NCCL group call of a series of NCCL broadcast collective operations. See the following materials for details.
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        mapping (Mapping):  The parallel mapping.
        gather_dim (int): Gather along given dimension. By default -1.
        all_rank_split_size(Optional[List[int]]): An optional list indicating 'input.shape[gather_dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    if mapping.tp_size == 1:
        return input

    if all_rank_split_size is not None:
        assert len(all_rank_split_size) == len(mapping.tp_group)
        if isinstance(input, torch.Tensor):
            assert input.shape[gather_dim] == all_rank_split_size[
                mapping.tp_rank]
        else:
            assert all([
                val.shape[gather_dim] == all_rank_split_size[mapping.tp_rank]
                for val in input
            ])
        # 'all_rank_split_size' is not needed if all inputs in the same TP group have the same shape
        for split_size in all_rank_split_size[1:]:
            if split_size != all_rank_split_size[0]:
                break
        else:
            all_rank_split_size = None

    if isinstance(input, torch.Tensor):
        torch_op = torch.ops.trtllm.allgather
        input = input.movedim(gather_dim, 0).contiguous()
    else:
        torch_op = torch.ops.trtllm.allgather_list
        input = [val.movedim(gather_dim, 0).contiguous() for val in input]

    output = torch_op(
        input,
        all_rank_split_size,
        mapping.tp_group,
    )

    if isinstance(input, torch.Tensor):
        output = output.movedim(0, gather_dim).contiguous()
    else:
        output = [val.movedim(0, gather_dim).contiguous() for val in output]
    return output


def reducescatter(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    scatter_dim: int = -1,
    all_rank_split_size: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if mapping.tp_size == 1:
        return input

    if all_rank_split_size is not None:
        assert len(all_rank_split_size) == len(mapping.tp_group)
        sum_split_size = sum(all_rank_split_size)
        if isinstance(input, torch.Tensor):
            assert input.shape[scatter_dim] == sum_split_size
        else:
            assert all(
                [val.shape[scatter_dim] == sum_split_size for val in input])
        # 'all_rank_split_size' is not needed if all outputs in the same TP group have the same shape
        for split_size in all_rank_split_size[1:]:
            if split_size != all_rank_split_size[0]:
                break
        else:
            all_rank_split_size = None

    if isinstance(input, torch.Tensor):
        torch_op = torch.ops.trtllm.reducescatter
        input = input.movedim(scatter_dim, 0).contiguous()
    else:
        torch_op = torch.ops.trtllm.reducescatter_list
        input = [val.movedim(scatter_dim, 0).contiguous() for val in input]

    output = torch_op(
        input,
        all_rank_split_size,
        mapping.tp_group,
    )

    if isinstance(input, torch.Tensor):
        output = output.movedim(0, scatter_dim).contiguous()
    else:
        output = [val.movedim(0, scatter_dim).contiguous() for val in output]
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
