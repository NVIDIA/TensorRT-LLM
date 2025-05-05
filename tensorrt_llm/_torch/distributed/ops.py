import threading
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceParams,
                                     AllReduceStrategy)
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
        self.max_workspace_size = CustomAllReduceHelper.max_workspace_size_auto(
            self.mapping.tp_size, support_deterministic=False)

        self.fallback_func_mapping = {
            AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
            self.fallback_residual_rms_norm_quant_fp8,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
            self.fallback_residual_rms_norm_out_quant_fp8,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
            self.fallback_residual_rms_norm_quant_nvfp4,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
            self.fallback_residual_rms_norm_out_quant_nvfp4,
        }

        if self.mapping.tp_size > 1:
            # When Strategy is UB, it is guaranteed that the workspace is not used.
            if self.strategy != AllReduceStrategy.UB:
                self.workspace = get_allreduce_workspace(self.mapping)

    @staticmethod
    def fallback_residual_rms_norm_quant_fp8(
        output: Tuple[torch.Tensor, ...],
        all_reduce_params: AllReduceParams,
    ):
        norm_out, residual_out = output
        quant_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            norm_out, all_reduce_params.scale)
        return quant_fp8, residual_out

    @staticmethod
    def fallback_residual_rms_norm_out_quant_fp8(
        output: Tuple[torch.Tensor, ...],
        all_reduce_params: AllReduceParams,
    ):
        norm_out, residual_out = output
        quant_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            norm_out, all_reduce_params.scale)
        return norm_out, quant_fp8, residual_out

    @staticmethod
    def fallback_residual_rms_norm_quant_nvfp4(
        output: Tuple[torch.Tensor, ...],
        all_reduce_params: AllReduceParams,
    ):
        norm_out, residual_out = output
        quant_fp4, scale_factor = torch.ops.trtllm.fp4_quantize(
            norm_out, all_reduce_params.scale, 16, False)

        return quant_fp4, scale_factor, residual_out

    @staticmethod
    def fallback_residual_rms_norm_out_quant_nvfp4(
        output: Tuple[torch.Tensor, ...],
        all_reduce_params: AllReduceParams,
    ):
        norm_out, residual_out = output
        quant_fp4, scale_factor = torch.ops.trtllm.fp4_quantize(
            norm_out, all_reduce_params.scale, 16, False)

        return norm_out, quant_fp4, scale_factor, residual_out

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

        strategy = self.strategy
        fusion_op = all_reduce_params.fusion_op

        # If the input size is larger than the max workspace size, fallback to NCCL strategy
        if input.numel() > self.max_workspace_size \
            and all_reduce_params.fusion_op != AllReduceFusionOp.NONE \
            and all_reduce_params.fusion_op != AllReduceFusionOp.RESIDUAL_RMS_NORM:
            strategy = AllReduceStrategy.NCCL
            fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

        output = torch.ops.trtllm.allreduce(
            input=input,
            residual=all_reduce_params.residual,
            norm_weight=all_reduce_params.norm_weight,
            scale=all_reduce_params.scale,
            bias=all_reduce_params.bias,
            workspace=self.workspace,
            group=self.mapping.tp_group,
            strategy=strategy,
            op=fusion_op,
            eps=all_reduce_params.eps,
        )

        if input.numel() > self.max_workspace_size \
            and all_reduce_params.fusion_op != AllReduceFusionOp.NONE \
            and all_reduce_params.fusion_op != AllReduceFusionOp.RESIDUAL_RMS_NORM:
            output = self.fallback_func_mapping[all_reduce_params.fusion_op](
                output, all_reduce_params)

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


class DeepseekAllReduce(nn.Module):

    def __init__(self, mapping: Mapping):
        super().__init__()
        self.mapping = mapping
        self.workspace = None
        if self.mapping.tp_size > 1:
            self.workspace = get_allreduce_workspace(mapping)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reduce_fusion_inputs: List[torch.Tensor],
        eps: float,
        fusion_op: AllReduceFusionOp,
    ) -> Tuple[torch.Tensor, ...]:
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
