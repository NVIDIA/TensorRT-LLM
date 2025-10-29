import math
import os
import platform
import threading
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._utils import mpi_comm, mpi_disabled
from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer
from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceParams,
                                     AllReduceStrategy, MoEAllReduceParams)
from tensorrt_llm.logger import logger
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


def allocate_low_presicion_allreduce_workspace(mapping: Mapping) -> None:
    if not hasattr(_thread_local, 'lowprecision_allreduce_workspaces'):
        _thread_local.lowprecision_allreduce_workspaces = {}
    lowprecision_allreduce_workspaces = _thread_local.lowprecision_allreduce_workspaces
    if mapping not in lowprecision_allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_lowprecision_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_lowprecision(
                mapping.tp_size),
        )
        lowprecision_allreduce_workspaces[mapping] = (ipc_buffers, workspace)
        CustomAllReduceHelper.initialize_lowprecision_buffers(
            workspace, mapping.tp_size)
    return


def get_allreduce_mnnvl_workspace(
    mapping: Mapping, dtype: torch.dtype
) -> Tuple[McastGPUBuffer, torch.Tensor, torch.Tensor, int]:

    if not hasattr(_thread_local,
                   f'allreduce_mnnvl_workspaces_{mapping.pp_rank}'):
        setattr(_thread_local, f'allreduce_mnnvl_workspaces_{mapping.pp_rank}',
                {})
    # Support topology split
    comm = mpi_comm().Split(
        int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank),
        mapping.tp_rank)
    force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"

    allreduce_mnnvl_workspaces = getattr(
        _thread_local, f'allreduce_mnnvl_workspaces_{mapping.pp_rank}')
    if mapping not in allreduce_mnnvl_workspaces:
        # buffer shape: [3, 2, buffer_tokens, hidden_dim]
        stride = 3 * 2 * dtype.itemsize
        # Max hidden_size_to_support
        max_hidden_dim = 16384
        buffer_size_in_bytes = math.ceil(
            12_000_000 / (max_hidden_dim * stride)) * (max_hidden_dim * stride)
        max_num_elements = buffer_size_in_bytes // stride

        mcast_buffer = McastGPUBuffer(
            buffer_size_in_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            # Split the communicator according to the topology
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank,
            mapping.local_rank,
            True,  # mnNvlink
        )

        buffer = mcast_buffer.get_uc_buffer(mapping.tp_rank,
                                            (3, 2, max_num_elements), dtype, 0)
        # Only initialize the buffer when we need to resize it
        buffer.fill_(-0.0)
        # CPU barrier since we assume this should not be called in cuda graph
        torch.cuda.synchronize()
        comm.Barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # [Buffer_ptr, Clear_ptr, num_tokens_to_clear,atomic access counter]
        buffer_flags = torch.tensor([0, 2, 0, 0],
                                    dtype=torch.uint32,
                                    device=torch.device("cuda",
                                                        mapping.local_rank))

        allreduce_mnnvl_workspaces[mapping] = (mcast_buffer, buffer,
                                               buffer_flags, max_num_elements)
    return allreduce_mnnvl_workspaces[mapping]


def userbuffers_allreduce_finalize(
        input: torch.Tensor,
        force_applying_finalize: bool = False) -> torch.Tensor:
    output = torch.ops.trtllm.userbuffers_allreduce_finalize(
        input, force_applying_finalize)
    return output


def get_output_info(input: torch.Tensor, dim: int) -> List[int]:
    dim = dim % input.ndim
    output_shape = [
        val if idx != dim else -1 for idx, val in enumerate(input.shape)
    ]
    numel_base = -math.prod(output_shape)
    return {'output_shape': output_shape, 'numel_base': numel_base}


def filter_valid_input(
        input_list: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], List[bool]]:
    func_valid = lambda x: x is not None
    valid_list = list(map(func_valid, input_list))
    input_list = list(filter(func_valid, input_list))
    return input_list, valid_list


def restore_full_output(valid_outputs: List[torch.Tensor],
                        valid_list: List[bool]) -> List[torch.Tensor]:
    idx = 0
    full_outputs = []
    for v in valid_list:
        full_outputs.append(valid_outputs[idx] if v else None)
        idx += int(v)
    return full_outputs


def allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Add an operation that performs a collective all-gather.

    If 'sizes' is 'None', the input tensors in the different ranks must have the same shape.
    Otherwise, 'sizes[i]' must be 'input.shape[dim]' at rank i, and the input tensors in
    the different ranks can only differ in shape at dimension `dim`.

    The input tensors in the same TP group are concatenated at dimension 'dim' to produce the output tensor.
    If 'sizes' is 'None', 'output.shape[dim] = input.shape[dim] * tp_group_size'.
    Otherwise, 'output.shape[dim] = sum(sizes)'.

    That operation is implemented using a torch op that wraps the NCCL all-gather collective operation or
    the NCCL group call of a series of NCCL broadcast collective operations. See the following materials for details.
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        mapping (Mapping):  The parallel mapping.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    if mapping.tp_size == 1:
        return input

    if sizes is not None:
        assert len(sizes) == len(mapping.tp_group)
        if isinstance(input, torch.Tensor):
            assert input.shape[dim] == sizes[mapping.tp_rank]
        else:
            assert all([
                val.shape[dim] == sizes[mapping.tp_rank] for val in input
                if val is not None
            ])
    # Inputs are reshaped in this way to pass necessary shape information to the allgather op
    if isinstance(input, torch.Tensor):
        if mpi_disabled():
            torch_op = torch.ops.trtllm.allgather_pg
        else:
            torch_op = torch.ops.trtllm.allgather

        output_info = get_output_info(input, dim)
        input = input.contiguous().view(-1, output_info['numel_base'])
    else:
        input, valid = filter_valid_input(input)
        if mpi_disabled():
            torch_op = torch.ops.trtllm.allgather_list_pg
        else:
            torch_op = torch.ops.trtllm.allgather_list

        output_info = [get_output_info(val, dim) for val in input]
        input = [
            val.contiguous().view(-1, val_info['numel_base'])
            for val, val_info in zip(input, output_info)
        ]

    if mpi_disabled():
        output = torch_op(input, sizes, mapping.tp_group,
                          mapping.tp_group_pg.boxed())
    else:
        output = torch_op(input, sizes, mapping.tp_group)

    def convert_output(x, x_info):
        if dim == 0:
            x = x.view(x_info['output_shape'])
        else:
            if sizes is None:
                x_list = x.chunk(mapping.tp_size)
            else:
                x_list = x.split(sizes)
            x = torch.cat([x.reshape(x_info['output_shape']) for x in x_list],
                          dim=dim)
        return x

    if isinstance(input, torch.Tensor):
        output = convert_output(output, output_info)
    else:
        output = [
            convert_output(val, val_info)
            for val, val_info in zip(output, output_info)
        ]
        output = restore_full_output(output, valid)
    return output


def alltoall_helix(
    inputs: List[torch.Tensor],
    group: List[int],
) -> List[torch.Tensor]:
    '''
    Add an operation that performs a collective all-to-all across a given group.
    The operation is implemented using a torch op that wraps a NCCL group call of a series of
    NCCL send/recv operations to implement the all-to-all. See the following materials for details.
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-all,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html.
    Args:
        inputs (List[Tensor]): The input tensors.
            Its length must be a multiple of the group size,
            and all tensors in a group must have the same shape.
        group (List[int]): The group of ranks to participate in the all-to-all.
    Returns:
        The output tensors.
        For each group of input tensors (of size group size),
        there is one output tensor with shape (group size, *input shape).
    '''
    n_ranks = len(group)
    if n_ranks == 1:
        return inputs

    assert n_ranks > 0, "group must be non-empty"
    assert n_ranks == len(set(group)), "group must be unique"

    assert len(inputs) % n_ranks == 0,\
        "inputs length must be a multiple of the group size"
    num_lists = len(inputs) // n_ranks
    for il in range(num_lists):
        ref_input = inputs[il * n_ranks]
        assert all([inputs[i].shape == ref_input.shape for i in range(il * n_ranks + 1, (il + 1) * n_ranks)]),\
            "all input tensors in a group must have the same shape"

    return torch.ops.trtllm.alltoall_helix(inputs, group, num_lists)


def reducescatter(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if mapping.tp_size == 1:
        return input

    if sizes is not None:
        assert len(sizes) == len(mapping.tp_group)
        sum_split_size = sum(sizes)
        if isinstance(input, torch.Tensor):
            assert input.shape[dim] == sum_split_size
        else:
            assert all([
                val.shape[dim] == sum_split_size for val in input
                if val is not None
            ])

    def convert_input(x, x_info):
        # Inputs are reshaped in this way to pass necessary shape information to the reducescatter op
        if dim == 0:
            x = x.contiguous().view(-1, x_info['numel_base'])
        else:
            if sizes is None:
                x_list = x.chunk(mapping.tp_size, dim=dim)
            else:
                x_list = x.split(sizes, dim=dim)
            x = torch.cat([x.reshape(-1, x_info['numel_base']) for x in x_list])
        return x

    if isinstance(input, torch.Tensor):
        if mpi_disabled():
            torch_op = torch.ops.trtllm.reducescatter_pg
        else:
            torch_op = torch.ops.trtllm.reducescatter
        output_info = get_output_info(input, dim)
        input = convert_input(input, output_info)
    else:
        input, valid = filter_valid_input(input)
        if mpi_disabled():
            torch_op = torch.ops.trtllm.reducescatter_list_pg
        else:
            torch_op = torch.ops.trtllm.reducescatter_list
        output_info = [get_output_info(val, dim) for val in input]
        input = [
            convert_input(val, val_info)
            for val, val_info in zip(input, output_info)
        ]

    if mpi_disabled():
        output = torch_op(input, sizes, mapping.tp_group,
                          mapping.tp_group_pg.boxed())
    else:
        output = torch_op(input, sizes, mapping.tp_group)

    if isinstance(input, torch.Tensor):
        output = output.view(output_info['output_shape'])
    else:
        output = [
            val.view(val_info['output_shape'])
            for val, val_info in zip(output, output_info)
        ]
        output = restore_full_output(output, valid)
    return output


class MNNVLAllReduce(nn.Module):
    """A specialized AllReduce implementation for Multi-Node NVLink communication.

    This class handles the MNNVL-specific allreduce operations, which can be more efficient
    for certain operations when using NVLink for multi-node communication.
    """

    SUPPORTED_FUSION_HIDDEN_DIMS = [2048, 2880, 4096, 5120, 7168, 8192]

    def __init__(self, mapping: Mapping, dtype: torch.dtype):
        super().__init__()
        self.mapping = mapping
        self.dtype = dtype
        assert (
            dtype in MNNVLAllReduce.get_supported_dtypes()
            and (not mapping.has_cp())
        ), "MNNVL all reduce only supports dtype {MNNVLAllReduce.get_supported_dtypes()} and without cp."

        self.mcast_buffer_mnnvl, self.buffer_mnnvl, self.buffer_flags_mnnvl, self.max_num_elements_mnnvl = get_allreduce_mnnvl_workspace(
            self.mapping, dtype)

    @staticmethod
    def get_supported_dtypes():
        return (torch.float16, torch.bfloat16, torch.float32)

    # Check if MNNVL is supported
    @staticmethod
    def is_mnnvl(mapping: Mapping, dtype: torch.dtype) -> bool:
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        return (dtype in MNNVLAllReduce.get_supported_dtypes()
                and not mapping.has_cp() and mapping.is_multi_node()
                and MnnvlMemory.supports_mnnvl() and is_on_aarch64)

    def forward(
        self,
        input: torch.Tensor,
        all_reduce_params: AllReduceParams,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass for MNNVL AllReduce.

        Args:
            input (torch.Tensor): Input tensor to be reduced
            all_reduce_params (Optional[AllReduceParams]): Parameters for fused operations

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: Reduced tensor(s)
        """

        fusion_op = all_reduce_params.fusion_op
        shape = input.shape
        input = input.view(-1, shape[-1])
        (num_tokens, hidden_dim) = input.shape

        # Slice the buffer according to the hidden size, need to pass this numel as the new buffer size
        max_num_tokens = self.max_num_elements_mnnvl // hidden_dim
        num_elements_in_use = max_num_tokens * hidden_dim
        if num_tokens > max_num_tokens:
            logger.debug(
                f"MNNVL AllReduce can't be enabled due to {num_tokens=} larger than {max_num_tokens=}."
            )
            return None

        # This should not happen but leave this check for future code changes
        if num_elements_in_use > self.max_num_elements_mnnvl:
            logger.debug(
                f"MNNVL AllReduce can't be enabled due to {num_elements_in_use=} larger than {self.max_num_elements_mnnvl=}."
            )
            return None

        output = torch.empty_like(input)
        buffer_mnnvl = self.buffer_mnnvl.view(-1)[:(3 * 2 *
                                                    num_elements_in_use)].view(
                                                        3, 2, -1, hidden_dim)

        if fusion_op == AllReduceFusionOp.NONE:
            output = torch.ops.trtllm.mnnvl_twoshot_allreduce(
                input,
                buffer_mnnvl,
                self.buffer_flags_mnnvl,
                num_elements_in_use,
                True,
            )
            return output.view(shape)
        # Fallback to use other allreduce if hidden_size is not supported
        elif (fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
              and hidden_dim in MNNVLAllReduce.SUPPORTED_FUSION_HIDDEN_DIMS):
            torch.ops.trtllm.mnnvl_twoshot_allreduce(
                input,
                buffer_mnnvl,
                self.buffer_flags_mnnvl,
                num_elements_in_use,
                False,
            )
            residual_in = all_reduce_params.residual

            output, residual_out = torch.ops.trtllm.mnnvl_twoshot_rmsnorm(
                buffer_mnnvl,
                all_reduce_params.norm_weight,
                all_reduce_params.eps,
                residual_in,
                self.buffer_flags_mnnvl,
                num_elements_in_use,
            )
            return output.view(shape), residual_out.view(shape)
        return None


class AllReduce(nn.Module):

    def __init__(self,
                 mapping: Mapping,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        """
        AllReduce is a module that performs an all-reduce operation on a tensor.

        Args:
            mapping (Mapping):  The parallel mapping config.
            strategy (AllReduceStrategy):
                The following all-reduce strategies are supported:

                - UB: AllReduce uses user-buffer based all-reduce kernel.

                - NCCL: Use NCCL allreduce.

                - MIN_LATENCY: AllReduce uses MIN_LATENCY mode kernel.

                - AUTO: AUTO chooses between NCCL and MIN_LATENCY mode based on a heuristic policy.

                - LOWPRECISION: AllReduce quantizes data to lower precision for transmission.
                  Should only be used on topologies with PCIe switches and without NVLink.
                  This strategy may result in some precision loss but can improve performance
                  on specific hardware configurations.

            All strategies support the following operations:
                - NONE (AllReduce only)
                - RESIDUAL_RMS_NORM
                - RESIDUAL_RMS_NORM_QUANT_FP8
                - RESIDUAL_RMS_NORM_QUANT_NVFP4
                - RESIDUAL_RMS_NORM_OUT_QUANT_FP8
                - RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4

            Note: NCCL, UB, and LOWPRECISION strategies only support consequent kernel calls
        instead of fused operations.

        Note:
            For the reference implementation for each pattern, please refer to the following unit test:
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/multi_gpu/test_allreduce.py

            The LOWPRECISION strategy can be selected either by directly specifying it in the constructor.
        """

        self.mapping = mapping
        self.workspace = None
        self.strategy = strategy
        self.mnnvl_allreduce = None
        self._disable_mpi = mpi_disabled()

        self.all_reduce_op = torch.ops.trtllm.allreduce_pg if self._disable_mpi else torch.ops.trtllm.allreduce

        if self.mapping.tp_size > 1:
            # When Strategy is UB, it is guaranteed that the workspace is not used.
            if self.strategy != AllReduceStrategy.UB:
                if self.strategy == AllReduceStrategy.LOWPRECISION:
                    allocate_low_presicion_allreduce_workspace(self.mapping)
                self.workspace = get_allreduce_workspace(self.mapping)

            # Initialize MNNVL AllReduce if needed
            if self.strategy in (AllReduceStrategy.AUTO,
                                 AllReduceStrategy.MNNVL):
                if MNNVLAllReduce.is_mnnvl(self.mapping, dtype):
                    try:
                        self.mnnvl_allreduce = MNNVLAllReduce(
                            self.mapping, dtype) if dtype else None
                        if self.mnnvl_allreduce:
                            logger.debug(f"MNNVLAllReduce is enabled")
                        else:
                            logger.debug(f"MNNVLAllReduce is disabled")
                    except Exception as e:
                        logger.debug(
                            f"MNNVL AllReduce can't be enabled due to {e}.")
                        self.mnnvl_allreduce = None
                else:
                    logger.debug(
                        f"MNNVLAllReduce can't be enabled due to failing the is_mnnvl check."
                    )
                    self.mnnvl_allreduce = None

    def is_mnnvl(self) -> bool:
        return self.mnnvl_allreduce is not None

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

        input = input.contiguous()  # Underlying op requires contiguous input

        allreduce_strategy = self.strategy
        if all_reduce_params is None:
            all_reduce_params = AllReduceParams()

        # Try MNNVL AllReduce first if available
        if self.mnnvl_allreduce:
            mnnvl_output = self.mnnvl_allreduce(
                input, all_reduce_params=all_reduce_params)
            if mnnvl_output is not None:
                return mnnvl_output

        # Fall back to regular AllReduce if MNNVL is not available or not applicable
        # Make sure the strategy is AUTO since allreduceOp does not have the branch for MNNVL
        if allreduce_strategy == AllReduceStrategy.MNNVL:
            allreduce_strategy = AllReduceStrategy.AUTO

        additional_args = {}
        if self._disable_mpi:
            pg = self.mapping.tp_group_pg
            assert pg is not None, "TP ProcessGroup not initialised"
            additional_args = {
                "rank": torch.distributed.get_rank(),
                "pg": pg.boxed(),
            }

        output = self.all_reduce_op(
            input=input,
            residual=all_reduce_params.residual,
            norm_weight=all_reduce_params.norm_weight,
            scale=all_reduce_params.scale,
            bias=all_reduce_params.bias,
            workspace=self.workspace,
            group=self.mapping.tp_group,
            strategy=allreduce_strategy,
            op=all_reduce_params.fusion_op,
            eps=all_reduce_params.eps,
            trigger_completion_at_end=all_reduce_params.
            trigger_completion_at_end,
            **additional_args,
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
            * min latency mode:

            Support pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
            expert_reduction = torch.sum(active_experts_token_input *
                                        scale.unsqueeze(-1),
                                        dim=0)
            output_add = expert_reduction + shared_expert_output
            output_residual = output_add + residual
            output_hidden_states = rms_norm(output_residual, norm_weight, eps)

            * regular mode:

            Support pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
            expert_reduction = local_reduction(input, expanded_idx_to_permuted_idx, expert_scale_factor)
            output_add = expert_reduction + shared_expert_output
            output_residual = output_add + residual
            output_hidden_states = rms_norm(output_residual, norm_weight, eps)
        """
        super().__init__()
        self.mapping = mapping
        self.workspace = get_allreduce_workspace(self.mapping)
        # Pls keep this value in sync with the kOneShotMaxToken in moeAllReduceFusionKernels.h
        self.max_token = 128

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: MoEAllReduceParams,
    ) -> torch.Tensor:

        assert all_reduce_params.is_valid(), "MoEAllReduceParams is not valid"

        if all_reduce_params.is_cutlass_min_latency:
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
                active_experts_token_input=input,
                residual=all_reduce_params.residual,
                norm_weight=all_reduce_params.norm_weight,
                device_num_experts=all_reduce_params.device_num_experts,
                scale_input=all_reduce_params.expert_scale_factor,
                token_input=all_reduce_params.shared_expert_output,
                workspace=self.workspace,
                rank=self.mapping.tp_rank,
                nranks=self.mapping.tp_size,
                eps=all_reduce_params.eps,
            )
        else:
            assert all_reduce_params.residual.shape[
                0] <= self.max_token, "Num tokens must be less than or equal to max_token"

            return torch.ops.trtllm.moe_finalize_allreduce(
                input=input,
                residual=all_reduce_params.residual,
                norm_weight=all_reduce_params.norm_weight,
                expanded_idx_to_permuted_idx=all_reduce_params.
                expanded_idx_to_permuted_idx,
                shared_expert_output=all_reduce_params.shared_expert_output,
                expert_scale_factor=all_reduce_params.expert_scale_factor,
                workspace=self.workspace,
                rank=self.mapping.tp_rank,
                nranks=self.mapping.tp_size,
                eps=all_reduce_params.eps,
            )
