import math
import os
import threading
from typing import Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._utils import mpi_barrier
from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer
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


class LowLatencyTwoShotAllReduce(nn.Module):
    # Use singleton pattern since memory allocation is required and expensive
    # TODO: Do we need to limit the number of instances?
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls,
                mapping: Mapping,
                init_dim: int = 7168,
                dtype: torch.dtype = torch.bfloat16):
        with cls._lock:
            if init_dim not in cls._instances:
                cls._instances[init_dim] = super().__new__(cls)
                cls._instances[init_dim]._initialized = False
        # __init__ is called after returning this instance
        return cls._instances[init_dim]

    def __init__(self,
                 mapping: Mapping,
                 init_dim: int = 7168,
                 dtype: torch.dtype = torch.bfloat16):
        # Prevent re-initialization if __init__ is called again on the same instance
        if getattr(self, "_initialized", False):
            return
        super().__init__()

        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.gpus_per_node = mapping.gpus_per_node
        self.dtype = dtype

        # Use intra-node to avoid confusing and avoid using 'Local Rank'
        self.force_mn = os.environ.get("TRTLLM_LLAR_FORCE_MN", "0") == "1"
        self.intra_node_rank = self.tp_rank % self.gpus_per_node
        self.local_device = torch.device("cuda", self.intra_node_rank)
        self.is_multi_node = mapping.is_multi_node() or self.force_mn
        self.max_num_tokens = int(os.environ.get("TRTLLM_LLAR_MAX_M", "128"))

        # Predefined N used to align the allocation
        self.hidden_dim = init_dim
        self.buf_size = self._alloc_buf()

        self._initialized = True

    def _alloc_buf(self):
        # Triple-buffer, one buffer for the reduce-scatter and one for the allgather, M*N
        buffer_tokens = math.ceil(
            self.max_num_tokens / self.tp_size) * self.tp_size
        self._mcast_buffer = McastGPUBuffer(
            buffer_tokens * self.hidden_dim * 3 * 2 * self.dtype.itemsize,
            self.tp_size,
            self.tp_rank,
            self.local_device,
            self.is_multi_node,
        )

        self._buffer = self._mcast_buffer.get_uc_buffer(
            self.tp_rank, (3, 2, buffer_tokens, self.hidden_dim), self.dtype, 0)
        # Only initialize the buffer when we need to resize it
        self._buffer.fill_(-0.0)
        # CPU barrier since we assume this should not be called in cuda graph
        torch.cuda.synchronize()
        mpi_barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # [Buffer_ptr, Clear_ptr, Buffer_size, atomic access counter]
        self._buffer_flags = torch.tensor(
            [0, 2, self._buffer.size()[3] * self._buffer.size()[2], 0],
            dtype=torch.uint32,
            device=self.local_device)

        return buffer_tokens * self.hidden_dim

    def __call__(self, shard_in: torch.Tensor) -> torch.Tensor:
        buffer_stride = self._buffer.size()[3] * self._buffer.size(
        )[2] * self._buffer.size()[1]
        shape = shard_in.shape
        shard_in = shard_in.view(-1, shard_in.shape[-1])

        shard_out = torch.empty_like(shard_in)

        torch.ops.trtllm.lowlat_twoshot_allreduce(
            shard_out,
            shard_in,
            self._buffer,
            self._buffer_flags,
            True,
        )
        return shard_out.view(shape)

    def all_reduce_res_norm(
        self,
        gamma: torch.Tensor,
        x: torch.Tensor,
        residual_in: torch.Tensor,
        residual_out: Optional[torch.Tensor] = None,
        eps: float = torch.finfo(torch.bfloat16).eps,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buffer_stride = self._buffer.size()[3] * self._buffer.size(
        )[2] * self._buffer.size()[1]
        shape = x.shape
        x_flattened = x.view(-1, x.shape[-1])
        shard_out = torch.empty_like(x_flattened)
        residual_in = residual_in.view(-1, residual_in.shape[-1])
        assert x_flattened.shape[-1] == self.hidden_dim

        torch.ops.trtllm.lowlat_twoshot_allreduce(
            shard_out,
            x_flattened,
            self._buffer,
            self._buffer_flags,
            False,
        )
        if residual_out is None:
            residual_out = torch.empty_like(residual_in)

        torch.ops.trtllm.lowlat_twoshot_rmsnorm(residual_out, shard_out,
                                                self._buffer, gamma, eps,
                                                residual_in, self._buffer_flags)

        return shard_out.view(shape), residual_out.view(shape)
