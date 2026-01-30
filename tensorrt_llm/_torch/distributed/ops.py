import math
import os
import platform
import threading
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._mnnvl_utils import HelixCpMnnvlMemory, MnnvlMemory
from tensorrt_llm._torch.distributed.symm_mem_allreduce import \
    SymmetricMemoryAllReduce
from tensorrt_llm._utils import mpi_comm, mpi_disabled
from tensorrt_llm.bindings import internal as _tllm_internal
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


def get_or_scale_allreduce_mnnvl_workspace(
    mapping: Mapping,
    dtype: torch.dtype,
    buffer_size_bytes: Optional[int] = None
) -> Tuple[McastGPUBuffer, torch.Tensor, torch.Tensor, int]:
    """
    WORKSPACE is a entire memory allocation used for allreduce, while BUFFER refers to single lamport buffer.
    Each WORKSPACE contains NUM_LAMPORT_BUFFERS buffers.
    """

    NUM_LAMPORT_BUFFERS = 3

    # Use MNNVLAllReduce class to share across threads
    allreduce_mnnvl_workspaces = MNNVLAllReduce.allreduce_mnnvl_workspaces

    # A safe method to get the element size of the dtype
    elem_size = torch.tensor([], dtype=dtype).element_size()
    force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
    use_fabric_handle = force_mn or mapping.is_multi_node()

    if mapping not in allreduce_mnnvl_workspaces or allreduce_mnnvl_workspaces[
            mapping]["buffer_size_bytes"] < (buffer_size_bytes or 0):
        # Initial buffer to be large enough to support 1024 tokens * 8192 hidden_dim
        init_buffer_size_bytes = max(1024 * 8192 * elem_size, buffer_size_bytes
                                     or 0)
        # Creating the workspace if it doesn't exist
        if mapping not in allreduce_mnnvl_workspaces:
            # Do the communicator split if there is no communicator in the workspace
            comm = mpi_comm().Split(
                int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank),
                mapping.tp_rank)
            # Use the predefined buffer size if no buffer size is provided
            buffer_size_bytes = buffer_size_bytes or init_buffer_size_bytes
            if mapping.tp_rank == 0:
                logger.debug(
                    f"[MNNVL] Creating workspace for pp_rank {mapping.pp_rank}, tp_size {mapping.tp_size} with {buffer_size_bytes} bytes"
                )

        else:
            comm = allreduce_mnnvl_workspaces[mapping]["mpi_comm"]
            # Safeguard against when buffer_size_bytes is None
            req_buffer_size_bytes = buffer_size_bytes or init_buffer_size_bytes
            # Increase the buffer size in 8 MiB granularity to avoid frequently scaling the buffer
            buffer_size_bytes = math.ceil(req_buffer_size_bytes /
                                          (8 * 1024 * 1024)) * (8 * 1024 * 1024)
            logger.debug(
                f"[MNNVL] Requested {req_buffer_size_bytes} bytes, is larger than the current workspace size. Scaling workspace for pp_rank {mapping.pp_rank}, tp_size {mapping.tp_size} from {allreduce_mnnvl_workspaces[mapping]['buffer_size_bytes']} to {buffer_size_bytes} bytes"
            )
        # Each workspace contains NUM_LAMPORT_BUFFERS buffers.
        workspace_size_bytes = NUM_LAMPORT_BUFFERS * buffer_size_bytes
        # Pass the pre-split MPI communicator's Fortran handle to avoid redundant splitting in C++
        mcast_buf_handle = McastGPUBuffer(
            workspace_size_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            mapping.local_rank,
            use_fabric_handle,  # whether to use fabric handle or POSIX FD ipc
            comm.py2f(),  # Fortran handle for the MPI communicator
        )

        # We use per FP32 element in the buffer for lamport sync
        buffer = mcast_buf_handle.get_uc_buffer(mapping.tp_rank,
                                                (workspace_size_bytes //
                                                 (torch.float32.itemsize), ),
                                                torch.float32, 0)
        buffer.fill_(-0.0)
        # Wait until the initialization is done
        torch.cuda.synchronize()
        comm.Barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # The flag should be binded to each buffer allocation
        # Layout: [cur idx, dirty idx, bytes per buffer, dirty num stages, numBytesToClear[4], access count ptr]
        num_bytes_to_clear = [0] * 4
        buffer_flags = torch.tensor(
            [0, 2, buffer_size_bytes, 0, *num_bytes_to_clear, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

        allreduce_mnnvl_workspaces[mapping] = {
            "handle": mcast_buf_handle,
            "uc_buffer": buffer,
            "buffer_flags": buffer_flags,
            "buffer_size_bytes": buffer_size_bytes,
            "mpi_comm": comm,
        }
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


def _allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    group: List[int],
    rank: int,
    group_boxed: Optional[object] = None,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Performs a collective all-gather across the given parallel group, for the given rank.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        group (List[int]): The list of ranks to participate in the all-gather.
        rank (int): The rank of the current process.
        group_boxed (object): The boxed ProcessGroup object for the list of ranks, if available.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    if len(group) == 1:
        return input

    if sizes is not None:
        assert len(sizes) == len(group)
        if isinstance(input, torch.Tensor):
            assert input.shape[dim] == sizes[rank]
        else:
            assert all([
                val.shape[dim] == sizes[rank] for val in input
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
        output = torch_op(input, sizes, group, group_boxed)
    else:
        output = torch_op(input, sizes, group)

    def convert_output(x, x_info):
        if dim == 0:
            x = x.view(x_info['output_shape'])
        else:
            if sizes is None:
                x_list = x.chunk(len(group))
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


def allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Add an operation that performs a collective all-gather across the TP group.

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
        mapping (Mapping): The parallel mapping.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    group_boxed = mapping.tp_group_pg.boxed() if mpi_disabled() else None
    return _allgather(input, mapping.tp_group, mapping.tp_rank, group_boxed,
                      dim, sizes)


def cp_allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Add an operation that performs a collective all-gather across the CP group.

    See `allgather` for more details on the inputs and implementation constraints.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        mapping (Mapping): The parallel mapping.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    group_boxed = mapping.cp_group_pg.boxed() if mpi_disabled() else None
    return _allgather(input, mapping.cp_group, mapping.cp_rank, group_boxed,
                      dim, sizes)


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
            All input tensors must be contiguous.
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


class HelixAllToAllNative:
    """
    Manager for Helix All-to-All operations with MNNVL workspace management.

    Exchanges data along the cp_size dimension:
    - partial_o: [..., cp_size, kv_lora_rank] half-precision
    - softmax_stats: [..., cp_size, 2] float32
    """

    # Global cache: mapping -> instance
    _cache: Dict[Mapping, "HelixAllToAllNative"] = {}

    def __init__(self, mapping: Mapping, workspace: HelixCpMnnvlMemory,
                 workspace_tensor: torch.Tensor):
        """Private constructor - use get() instead."""
        self.mapping = mapping
        self.workspace = workspace
        self.workspace_tensor = workspace_tensor

    @staticmethod
    def get(mapping: Mapping) -> "HelixAllToAllNative":
        """
        Get or create a HelixAllToAllNative instance for the given configuration.

        Args:
            mapping: TensorRT-LLM mapping object containing cp_size and cp_rank

        Returns:
            Cached or newly-created HelixAllToAllNative instance
        """
        if mapping not in HelixAllToAllNative._cache:
            logger.info(
                f"Rank {mapping.cp_rank} initializing HelixCpMnnvlMemory for Helix"
            )
            MnnvlMemory.initialize()

            # Get workspace size (in bytes)
            workspace_size_per_rank = _tllm_internal.thop.get_helix_workspace_size_per_rank(
                mapping.cp_size)

            # Allocate MNNVL memory using CP communicator for Helix
            workspace = HelixCpMnnvlMemory(mapping, workspace_size_per_rank)
            workspace_tensor = workspace.as_torch_strided_tensor(torch.uint64)

            torch.ops.trtllm.initialize_helix_workspace(workspace_tensor,
                                                        mapping.cp_rank,
                                                        mapping.cp_size)
            torch.cuda.synchronize()
            HelixCpMnnvlMemory.get_comm(mapping).barrier()

            HelixAllToAllNative._cache[mapping] = HelixAllToAllNative(
                mapping, workspace, workspace_tensor)

        return HelixAllToAllNative._cache[mapping]

    def alltoall_native(self, partial_o: torch.Tensor,
                        softmax_stats: torch.Tensor):
        """
        Perform all-to-all data exchange.

        Args:
            partial_o: Tensor with shape [..., cp_size, kv_lora_rank], dtype half.
            softmax_stats: Tensor with shape [..., cp_size, 2], dtype float32.

        Returns:
            Tuple of (partial_o_out, softmax_stats_out) with same shapes as inputs.
        """
        partial_o_out, softmax_stats_out = torch.ops.trtllm.alltoall_helix_native(
            partial_o,
            softmax_stats,
            self.workspace_tensor,
            self.mapping.cp_rank,
            self.mapping.cp_size,
        )

        return partial_o_out, softmax_stats_out


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
    allreduce_mnnvl_workspaces: Dict[Mapping, Dict] = {}

    def __init__(self, mapping: Mapping, dtype: torch.dtype):
        super().__init__()
        self.mapping = mapping
        self.dtype = dtype
        if dtype not in MNNVLAllReduce.get_supported_dtypes() or (
                mapping.has_cp()):
            # This is safe as we always capture the exception when create this object
            raise ValueError(
                f"MNNVL all reduce only supports dtype {MNNVLAllReduce.get_supported_dtypes()} and without cp."
            )

        # Initialize the workspace
        get_or_scale_allreduce_mnnvl_workspace(self.mapping, self.dtype)

    @staticmethod
    def get_supported_dtypes():
        return (torch.float16, torch.bfloat16, torch.float32)

    # Check if MNNVL is supported
    @staticmethod
    def is_mnnvl(mapping: Mapping, dtype: torch.dtype) -> bool:
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        # Add a bypass so that we can run the unittest on single-node
        is_testing = os.environ.get("TLLM_TEST_MNNVL", "0") == "1"
        return is_testing or (dtype in MNNVLAllReduce.get_supported_dtypes() and
                              not mapping.has_cp() and mapping.is_multi_node()
                              and MnnvlMemory.supports_mnnvl()
                              and is_on_aarch64)

    @staticmethod
    def get_required_workspace_size(num_tokens: int, hidden_dim: int,
                                    group_size: int, dtype: torch.dtype) -> int:
        elem_size = torch.tensor([], dtype=dtype).element_size()
        # This should match the heuristic in allreduceOp.cpp
        is_one_shot = num_tokens * hidden_dim * group_size * elem_size <= 64 * 1024 * 8
        if is_one_shot:
            # For one-shot, each rank needs to store num_tokens * group_size tokens
            workspace_size = num_tokens * hidden_dim * group_size * elem_size
        else:
            # For two-shot, each rank stores a slices of tokens. We need to round up to the nearest group_size.
            # 2 Stage is required for the two-shot allreduce.
            workspace_size = 2 * math.ceil(
                num_tokens / group_size) * group_size * hidden_dim * elem_size
        return workspace_size

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

        workspace_size_bytes = self.get_required_workspace_size(
            num_tokens, hidden_dim, self.mapping.tp_size, self.dtype)

        # We use uint32_t to store workspace size related info. Safeguard against overflow.
        if workspace_size_bytes >= 2**32 - 1:
            # Raise an error so we can fallback to other allreduce strategies
            raise ValueError(
                f"[MNNVL AllReduce] Required workspace {workspace_size_bytes} bytes exceeds uint32 limits "
                f"for shard ({num_tokens}, {hidden_dim}), TP {self.mapping.tp_size}."
            )

        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping,
            self.dtype,
            buffer_size_bytes=workspace_size_bytes,
        )

        # We don't expect the buffer to be directly used in this level. The tensor is only used for passing the pointer to the kernel
        buffer_base = workspace["uc_buffer"].view(self.dtype).view(3, -1)
        # The buffer flags is tied to the buffer and used to save the state of the buffer
        buffer_flags = workspace["buffer_flags"]

        if fusion_op == AllReduceFusionOp.NONE:
            output, _ = torch.ops.trtllm.mnnvl_fusion_allreduce(
                input,
                None,  # gamma
                None,  # residual
                1e-6,  # epsilon
                buffer_base,  # comm_buffer
                buffer_flags,  # buffer_flags
                False,  # rmsnorm_fusion
            )
            return output.view(shape)
        # Fallback to use other allreduce if hidden_size is not supported
        elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
            output, residual_out = torch.ops.trtllm.mnnvl_fusion_allreduce(
                input,
                all_reduce_params.norm_weight,  # gamma
                all_reduce_params.residual,  # residual
                all_reduce_params.eps,  # epsilon
                buffer_base,  # comm_buffer
                buffer_flags,  # buffer_flags
                True,  # rmsnorm_fusion
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

                - SYMM_MEM: Uses PyTorch's symmetric memory with MULTIMEM hardware instructions.
                  Falls back automatically if not supported.

                - UB: AllReduce uses user-buffer based all-reduce kernel.

                - NCCL: Use NCCL allreduce.

                - MIN_LATENCY: AllReduce uses MIN_LATENCY mode kernel.

                - AUTO: AUTO chooses the best available strategy. Will try MNNVL,
                  then choose between NCCL and MIN_LATENCY based on a heuristic policy.

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
        self.symm_mem_allreduce = None
        self._disable_mpi = mpi_disabled()

        self.all_reduce_op = torch.ops.trtllm.allreduce_pg if self._disable_mpi else torch.ops.trtllm.allreduce
        if self.mapping.tp_size > 1:
            # Initialize Symmetric Memory AllReduce if needed (before workspace allocation)
            if self.strategy == AllReduceStrategy.SYMM_MEM:
                try:
                    symm_mem = SymmetricMemoryAllReduce(
                        self.mapping,
                        dtype=dtype if dtype else torch.bfloat16,
                    )
                    if not symm_mem.disabled:
                        self.symm_mem_allreduce = symm_mem
                        logger.info(
                            f"SymmetricMemoryAllReduce (MULTIMEM) is enabled with fallback support for world_size={self.mapping.tp_size}"
                        )
                        # Keep SYMM_MEM strategy but allocate workspace for fallback to regular allreduce
                    else:
                        logger.info(
                            f"SymmetricMemoryAllReduce is disabled (not supported or unavailable), falling back to AUTO strategy"
                        )
                        # Fall back to AUTO if SYMM_MEM can't be enabled
                        self.strategy = AllReduceStrategy.AUTO
                except Exception as e:
                    logger.info(
                        f"Symmetric Memory AllReduce can't be enabled due to {e}, falling back to AUTO strategy"
                    )
                    self.symm_mem_allreduce = None
                    # Fall back to AUTO if SYMM_MEM initialization fails
                    self.strategy = AllReduceStrategy.AUTO

            # Allocate workspace for strategies that need it
            # Note: SYMM_MEM now also needs workspace for fallback scenarios (fused ops, etc.)
            # Only UB doesn't need workspace
            if self.strategy != AllReduceStrategy.UB:
                if self.strategy == AllReduceStrategy.LOWPRECISION:
                    allocate_low_presicion_allreduce_workspace(self.mapping)
                if self.strategy not in (AllReduceStrategy.UB,
                                         AllReduceStrategy.NCCL,
                                         AllReduceStrategy.NCCL_SYMMETRIC):
                    self.workspace = get_allreduce_workspace(self.mapping)

            # Initialize MNNVL if using AUTO or MNNVL strategy
            if self.strategy in (AllReduceStrategy.AUTO,
                                 AllReduceStrategy.MNNVL):
                # Try to initialize MNNVL
                if MNNVLAllReduce.is_mnnvl(self.mapping, dtype):
                    # ALWAYS capture the exception when creating this instance
                    try:
                        self.mnnvl_allreduce = MNNVLAllReduce(
                            self.mapping, dtype) if dtype else None
                    except Exception as e:
                        logger.debug(
                            f"MNNVL AllReduce can't be enabled due to {e}.")
                        self.mnnvl_allreduce = None
                else:
                    logger.debug(
                        f"MNNVLAllReduce can't be enabled due to failing the is_mnnvl check."
                    )
                    self.mnnvl_allreduce = None

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

        # Try Symmetric Memory AllReduce first if available
        # Note: Currently only supports NONE fusion op (plain allreduce)
        if self.symm_mem_allreduce and all_reduce_params.fusion_op == AllReduceFusionOp.NONE:
            symm_mem_output = self.symm_mem_allreduce(input)
            if symm_mem_output is not None:
                logger.debug(
                    f"Using SymmetricMemoryAllReduce (MULTIMEM) for input shape {input.shape}"
                )
                return symm_mem_output
        elif self.symm_mem_allreduce and all_reduce_params.fusion_op != AllReduceFusionOp.NONE:
            # Log once per rank that we're skipping symm_mem due to fusion
            logger.debug_once(
                f"Skipping SymmetricMemoryAllReduce for fused operation (fusion_op={all_reduce_params.fusion_op}), using regular allreduce",
                key=(self.mapping.tp_rank, all_reduce_params.fusion_op,
                     "debug_fusion_skip"),
            )

        # Try MNNVL AllReduce if symm_mem didn't handle it
        if self.mnnvl_allreduce:
            mnnvl_output = self.mnnvl_allreduce(
                input, all_reduce_params=all_reduce_params)
            if mnnvl_output is not None:
                return mnnvl_output

        # Fall back to regular AllReduce if specialized methods are not available or not applicable
        # Make sure the strategy is AUTO since allreduceOp does not have the branch for MNNVL/SYMM_MEM
        if allreduce_strategy in (AllReduceStrategy.MNNVL,
                                  AllReduceStrategy.SYMM_MEM):
            allreduce_strategy = AllReduceStrategy.AUTO

        additional_args = {}
        if self._disable_mpi:
            # Get ProcessGroup from mapping
            pg = self.mapping.tp_group_pg
            assert pg is not None, "TP ProcessGroup not initialised"
            additional_args = {
                "rank": torch.distributed.get_rank(),
                "pg": pg.boxed(),
            }

        # In case that AutoTuner brings potential perf regression
        # TODO: Remove this if no perf regression is observed.
        disable_allreduce_autotune = os.environ.get(
            "TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0") == "1"

        if allreduce_strategy == AllReduceStrategy.AUTO and not disable_allreduce_autotune and not self._disable_mpi:
            output = torch.ops.trtllm.tunable_allreduce(
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
            )
        else:
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


def all_to_all_4d(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    All-to-all for 4D tensors (batch, seq, heads, head_dim).

    Redistributes a 4D tensor along two dimensions using all-to-all communication.
    This is used for Ulysses-style sequence parallelism to transform between:
    - Sequence sharding [B, S/P, H, D] → Head sharding [B, S, H/P, D]
    - Head sharding [B, S, H/P, D] → Sequence sharding [B, S/P, H, D]

    Args:
        input: Input tensor with shape [batch, seq, heads, head_dim]
        scatter_dim: Dimension to split and scatter (1 for seq, 2 for heads)
        gather_dim: Dimension to gather (1 for seq, 2 for heads)
        process_group: PyTorch distributed process group. If None, uses default process group.

    Returns:
        Redistributed tensor with same shape as input

    Example:
        # Transform from sequence sharding to head sharding
        # Input: [B, S/P, H, D] (each rank has S/P sequence)
        output = all_to_all_4d(input, scatter_dim=2, gather_dim=1, process_group=pg)
        # Output: [B, S, H/P, D] (each rank has H/P heads)

        # Transform back from head sharding to sequence sharding
        output = all_to_all_4d(input, scatter_dim=1, gather_dim=2, process_group=pg)
    """
    # Only support PyTorch distributed mode (not MPI mode)
    if not mpi_disabled():
        raise NotImplementedError(
            "all_to_all_4d currently only supports PyTorch distributed mode. "
            "MPI mode is not supported.")

    # Get world size from process group
    world_size = torch.distributed.get_world_size(group=process_group)

    # If world_size is 1, no communication needed
    if world_size == 1:
        return input

    # Validate dimensions
    assert scatter_dim in [1, 2], "scatter_dim must be 1 (seq) or 2 (heads)"
    assert gather_dim in [1, 2], "gather_dim must be 1 (seq) or 2 (heads)"
    assert scatter_dim != gather_dim, "scatter_dim and gather_dim must be different"

    batch, seq, heads, head_dim = input.shape

    # Validate that the scatter dimension is divisible by world_size
    scatter_size = input.shape[scatter_dim]
    assert scatter_size % world_size == 0, \
        f"Dimension {scatter_dim} size {scatter_size} must be divisible by world_size {world_size}"

    # For all-to-all, we need to:
    # 1. Split input along scatter_dim into world_size chunks
    # 2. Send chunk i to rank i
    # 3. Receive chunk from each rank and concatenate along gather_dim

    # Reshape for all-to-all: move scatter_dim chunks to a new dimension
    if scatter_dim == 1:  # Scatter along seq dimension
        # [B, S, H, D] -> [B, P, S/P, H, D] where P = world_size
        input_reshaped = input.view(batch, world_size, seq // world_size, heads,
                                    head_dim)
        # Transpose to group by destination rank: [B, P, S/P, H, D] -> [P, B, S/P, H, D]
        input_transposed = input_reshaped.permute(1, 0, 2, 3, 4).contiguous()
    else:  # scatter_dim == 2, scatter along heads dimension
        # [B, S, H, D] -> [B, S, P, H/P, D] where P = world_size
        input_reshaped = input.view(batch, seq, world_size, heads // world_size,
                                    head_dim)
        # Transpose to group by destination rank: [B, S, P, H/P, D] -> [P, B, S, H/P, D]
        input_transposed = input_reshaped.permute(2, 0, 1, 3, 4).contiguous()

    # Flatten to [P * ...] for all-to-all communication
    # Shape: [P, B, ...] -> [P * B * ...]
    input_flat = input_transposed.flatten()
    output_flat = torch.empty_like(input_flat)

    # Perform all-to-all communication using PyTorch distributed
    # all_to_all_single splits input into world_size chunks and exchanges them
    torch.distributed.all_to_all_single(output_flat,
                                        input_flat,
                                        group=process_group)

    # Reshape output back to [P, B, ...] form
    output_transposed = output_flat.view_as(input_transposed)

    # Transpose back and reshape to final form
    if gather_dim == 1:  # Gather along seq dimension
        # [P, B, S/P, H, D] -> [B, P, S/P, H, D]
        output_reshaped = output_transposed.permute(1, 0, 2, 3, 4).contiguous()
        # [B, P, S/P, H, D] -> [B, S, H, D] where S = P * (S/P)
        # When scattering heads and gathering seq: seq needs to be multiplied, heads needs to be divided
        if scatter_dim == 2:
            # Scattered heads, so we have H/P heads and need to gather S/P -> S sequence
            gathered_seq = seq * world_size
            sharded_heads = heads // world_size
            output = output_reshaped.view(batch, gathered_seq, sharded_heads,
                                          head_dim)
        else:
            # Scattered seq (should be impossible if gather_dim == 1), keep as is
            output = output_reshaped.view(batch, seq, heads, head_dim)
    else:  # gather_dim == 2, gather along heads dimension
        # [P, B, S, H/P, D] -> [B, S, P, H/P, D]
        output_reshaped = output_transposed.permute(1, 2, 0, 3, 4).contiguous()
        # [B, S, P, H/P, D] -> [B, S, H, D] where H = P * (H/P)
        # When scattering seq and gathering heads: heads needs to be multiplied, seq needs to be divided
        if scatter_dim == 1:
            # Scattered seq, so we have S/P seq and need to gather H/P -> H heads
            gathered_heads = heads * world_size
            sharded_seq = seq // world_size
            output = output_reshaped.view(batch, sharded_seq, gathered_heads,
                                          head_dim)
        else:
            # Scattered heads (should be impossible if gather_dim == 2), keep as is
            output = output_reshaped.view(batch, seq, heads, head_dim)

    return output
