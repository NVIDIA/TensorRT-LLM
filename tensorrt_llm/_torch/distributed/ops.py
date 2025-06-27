import math
import os
import threading
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._utils import getSMVersion
from tensorrt_llm._utils import mpi_barrier
from tensorrt_llm.bindings.allreduceUtil import check_nvlink_p2p_supported
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


def get_all_reduce_mnnvl_max_workspace_elements(dtype):
    stride = 3 * 2 * dtype.itemsize
    # LCM for hidden_dim: 2048, 4096, 5120, 7168, 8192 = 286720
    # max_num_elements must be a multiple of 286720
    lcm_hidden_dim = 286720
    buffer_size_in_bytes = math.ceil(
        12_000_000 / (lcm_hidden_dim * stride)) * (lcm_hidden_dim * stride)
    max_num_elements = buffer_size_in_bytes // stride
    return max_num_elements, buffer_size_in_bytes, (3, 2, max_num_elements)


def get_allreduce_mnnvl_workspace(
    mapping: Mapping, dtype: torch.dtype
) -> Tuple[McastGPUBuffer, torch.Tensor, torch.Tensor, int]:
    if not hasattr(_thread_local,
                   f'allreduce_mnnvl_workspaces_{mapping.pp_rank}'):
        setattr(_thread_local, f'allreduce_mnnvl_workspaces_{mapping.pp_rank}',
                {})

    force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"

    allreduce_mnnvl_workspaces = getattr(
        _thread_local, f'allreduce_mnnvl_workspaces_{mapping.pp_rank}')
    if mapping not in allreduce_mnnvl_workspaces:
        max_num_elements, buffer_size_in_bytes, buffer_shape = get_all_reduce_mnnvl_max_workspace_elements(
            dtype)

        mcast_buffer = McastGPUBuffer(
            buffer_size_in_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            torch.device("cuda", mapping.local_rank),
            mapping.is_multi_node() or force_mn,
        )

        buffer = mcast_buffer.get_uc_buffer(mapping.tp_rank, buffer_shape,
                                            dtype, 0)
        # Only initialize the buffer when we need to resize it
        buffer.fill_(-0.0)
        # CPU barrier since we assume this should not be called in cuda graph
        torch.cuda.synchronize()
        mpi_barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # [Buffer_ptr, Clear_ptr, Buffer_size, atomic access counter]
        buffer_flags = torch.tensor([0, 2, max_num_elements, 0],
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
        torch_op = torch.ops.trtllm.allgather
        output_info = get_output_info(input, dim)
        input = input.contiguous().view(-1, output_info['numel_base'])
    else:
        input, valid = filter_valid_input(input)
        torch_op = torch.ops.trtllm.allgather_list
        output_info = [get_output_info(val, dim) for val in input]
        input = [
            val.contiguous().view(-1, val_info['numel_base'])
            for val, val_info in zip(input, output_info)
        ]

    output = torch_op(
        input,
        sizes,
        mapping.tp_group,
    )

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
        torch_op = torch.ops.trtllm.reducescatter
        output_info = get_output_info(input, dim)
        input = convert_input(input, output_info)
    else:
        input, valid = filter_valid_input(input)
        torch_op = torch.ops.trtllm.reducescatter_list
        output_info = [get_output_info(val, dim) for val in input]
        input = [
            convert_input(val, val_info)
            for val, val_info in zip(input, output_info)
        ]

    output = torch_op(
        input,
        sizes,
        mapping.tp_group,
    )

    if isinstance(input, torch.Tensor):
        output = output.view(output_info['output_shape'])
    else:
        output = [
            val.view(val_info['output_shape'])
            for val, val_info in zip(output, output_info)
        ]
        output = restore_full_output(output, valid)
    return output


NVLINK_P2P_SUPPORTED = {}
MNNVL_BUFFER_SHAPE = {}
SM_VERSION = getSMVersion()

def is_FP8_enabled():
    global SM_VERSION
    return (SM_VERSION >= 90) and (SM_VERSION < 100)

def get_mnnvl_supported_dtypes():
    return (torch.bfloat16, torch.float32)


def get_UB_supported_dtype():
    return (torch.half, torch.bfloat16)


def _strategy_supports(strategy, input, fusion_op, nvlink_supported, p2p_supported, buffer_mnnvl_shape):
    # If UB is 
    if strategy == AllReduceStrategy.NCCL:
        return True

    if strategy == AllReduceStrategy.UB and input.dtype(
    ) in get_UB_supported_dtype():
        return True

    override_strategy = os.getenv("OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY",
                                    False)
    if override_strategy and strategy != AllReduceStrategy.AUTO:
        return True

    if strategy == AllReduceStrategy.LOWPRECISION:
        low_precision_min_message_size = 2 * 1024 * 1024
        input_size = input.numel()
        if is_FP8_enabled(
        ) and input_size >= low_precision_min_message_size and not nvlink_supported and p2p_supported:
            return True

    if strategy == AllReduceStrategy.MNNVL:
        assert buffer_mnnvl_shape is not None, "buffer_mnnvl_shape shall not be None."
        shape = input.shape
        if input.numel() <= get_all_reduce_mnnvl_max_workspace_elements(
                input.dtype())[0] and (buffer_mnnvl_shape[-1] % shape[-1] == 0):
            return True

    return False


def _fallback_nccl(message_size_bytes, max_workspace_size, is_auto, nvlink_supported, p2p_supported):
    # If messageSize is less than maxWorkspaceSize, use NCCL, regardless of the fusion type.
    if message_size_bytes > max_workspace_size:
        if not is_auto:
            logger.warn(
                "Since messageSize is greater than maxWorkspaceSize, fallback to AllReduceStrategy: NCCL"
            )
        return True

    # If Peer to Peer is not supported, fallback to NCCL.
    if not p2p_supported:
        if not is_auto:
            logger.warn(
                "Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL"
            )
        return True

    # If NVLINK is not supported, fallback to NCCL.
    if not nvlink_supported:
        if not is_auto:
            logger.warn(
                "Since NVLINK not supported, fallback to AllReduceStrategy: NCCL"
            )
        return True

    return False


def _get_max_required_workspace_size(world_size):
    forceDeterministic = bool(
        os.environ.get("FORCE_ALL_REDUCE_DETERMINISTIC", False)) or bool(
            os.environ.get("FORCE_DETERMINISTIC", False))
    if forceDeterministic:
        workspaceSize = int(
            os.environ.get("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE",
                            1000 * 1000 * 1000))
        return workspaceSize

    if world_size <= 2:
        return 16 * 1000 * 1000
    return 8 * 1000 * 1000


def _match_min_latency_msg_size(world_size, message_size_bytes):
    if world_size <= 2 or (message_size_bytes < 500 * 1000) or (
            world_size <= 4 & message_size_bytes < 1 * 1000 * 1000):
        return True


def _infer_strategy(input, fusion_op: AllReduceFusionOp,
                    stratey: AllReduceStrategy, tp_group, nvlink_supported, p2p_supported) -> AllReduceStrategy:
    is_auto = stratey == AllReduceStrategy.AUTO
    message_size_bytes = input.numel() * input.element_size()
    max_workspace_size = _get_max_required_workspace_size(
        len(tp_group))

    if _fallback_nccl(message_size_bytes, max_workspace_size, is_auto, nvlink_supported, p2p_supported):
        return AllReduceStrategy.NCCL

    if fusion_op in [
            AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4
    ]:
        return AllReduceStrategy.MIN_LATENCY
    # Suppose NCCL has fallback implementations for all fusion types.
    elif fusion_op not in [
            AllReduceFusionOp.NONE, AllReduceFusionOp.RESIDUAL_RMS_NORM
    ]:
        return AllReduceStrategy.NCCL

    if not is_auto:
        if stratey in [
                AllReduceStrategy.ONESHOT, AllReduceStrategy.TWOSHOT
        ]:
            return AllReduceStrategy.MIN_LATENCY
        return stratey
    else:
        if _match_min_latency_msg_size(len(tp_group),
                                    message_size_bytes):
            stratey = AllReduceStrategy.MIN_LATENCY
        else:
            stratey = AllReduceStrategy.NCCL
    return stratey


def _prepare_for_strategy(strategy, input, workspace, mnnvl_buffer, mnnvl_buff_flags):
    if strategy == AllReduceStrategy.MNNVL:
        workspace = mnnvl_buffer.view(3, 2, -1, input.shape[-1])
        mnnvl_buff_flags = mnnvl_buff_flags
    else:
        workspace = workspace
        mnnvl_buff_flags = None
    return workspace, mnnvl_buff_flags


def get_runtime_strategy(input, fusion_op, origin_strategy, tp_group, buffer_mnnvl_shape):
    strategy = origin_strategy

    tp_group_set = tuple(tp_group)
    global NVLINK_P2P_SUPPORTED
    nvlink_supported, p2p_supported = NVLINK_P2P_SUPPORTED[
        tp_group_set]

    if strategy != AllReduceStrategy.AUTO:
        if _strategy_supports(strategy, input,
                                    fusion_op, nvlink_supported, p2p_supported, buffer_mnnvl_shape):
            return strategy
        elif strategy == AllReduceStrategy.LOWPRECISION or strategy == AllReduceStrategy.MNNVL or strategy == AllReduceStrategy.UB:
            strategy = AllReduceStrategy.AUTO
    return _infer_strategy(input, fusion_op, strategy, tp_group, nvlink_supported, p2p_supported)


@torch.library.custom_op("trtllm::all_reduce", mutates_args=())
def all_reduce_forward(input: torch.Tensor, residual: Optional[torch.Tensor], norm_weight: Optional[torch.Tensor],
                        scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], tp_group: list[int],
                        strategy: int, fusion_op: int, eps: float, trigger_completion_at_end: bool,
                        buffer_mnnvl_shape: Optional[list[int]], workspace: Optional[torch.Tensor],
                        mnnvl_buffer: Optional[torch.Tensor], mnnvl_buff_flags: Optional[torch.Tensor]) -> List[torch.Tensor]:
    strategy = get_runtime_strategy(input, fusion_op, strategy, tp_group, buffer_mnnvl_shape)
    workspace, buffer_flags_mnnvl = _prepare_for_strategy(strategy, input, workspace, mnnvl_buffer, mnnvl_buff_flags)

    output = torch.ops.trtllm.allreduce(
        input=input,
        residual=residual,
        norm_weight=norm_weight,
        scale=scale,
        bias=bias,
        workspace=workspace,
        mnnvl_buffer_flags=buffer_flags_mnnvl,
        group=tp_group,
        strategy=strategy,
        op=fusion_op,
        eps=eps,
        trigger_completion_at_end=
        trigger_completion_at_end,
        )

    return output

@all_reduce_forward.register_fake
def _(
    input: torch.Tensor, residual: Optional[torch.Tensor], norm_weight: Optional[torch.Tensor],
    scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], tp_group: list[int],
    strategy: int, fusion_op: int, eps: float, trigger_completion_at_end: bool,
    buffer_mnnvl_shape: Optional[list[int]], workspace: Optional[torch.Tensor],
    mnnvl_buffer: Optional[torch.Tensor], mnnvl_buff_flags: Optional[torch.Tensor]
) -> List[torch.Tensor]:
    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
        return [torch.empty_like(input), torch.empty_like(residual)]
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
        return [norm_quant, torch.empty_like(residual)]
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        return [norm, norm_quant, torch.empty_like(residual)]
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        return [norm_quant_fp4, scale_factor, torch.empty_like(residual)]
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        return [norm, norm_quant_fp4, scale_factor, torch.empty_like(residual)]
    return torch.empty_like(input)

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
        self.mnnvl_buffer = None
        self.mnnvl_buff_flags = None
        self.strategy = strategy
        self.dtype = dtype

        if self.strategy == AllReduceStrategy.MNNVL:
            if (not is_FP8_enabled()) or (dtype is None or dtype not in get_mnnvl_supported_dtypes()) or (mapping.has_cp()):
                logger.warn(
                    f"MNNVL doesn't support cp and only support datatype {get_mnnvl_supported_dtypes()}: required cp {mapping.has_cp()} dtype {dtype}. Failback to auto."
                )
                self.strategy = AllReduceStrategy.AUTO

        self.buffer_mnnvl_shape = None

        tp_group = self.mapping.tp_group
        tp_group_set = tuple(tp_group)
        if tp_group_set in NVLINK_P2P_SUPPORTED:
            self.nvlink_supported, self.p2p_supported = NVLINK_P2P_SUPPORTED[
                tp_group_set]
        else:
            self.nvlink_supported, self.p2p_supported = check_nvlink_p2p_supported(
                set(tp_group))
            NVLINK_P2P_SUPPORTED[tp_group_set] = [
                self.nvlink_supported, self.p2p_supported
            ]

        if self.strategy == AllReduceStrategy.MNNVL:
            assert dtype is not None, "dtype shall be set with MNNVL."
            if dtype in MNNVL_BUFFER_SHAPE:
                self.buffer_mnnvl_shape = MNNVL_BUFFER_SHAPE(dtype)
            else:
                self.buffer_mnnvl_shape = get_all_reduce_mnnvl_max_workspace_elements(
                    dtype)[2]
                MNNVL_BUFFER_SHAPE[dtype] = self.buffer_mnnvl_shape
            mcast_buffer_mnnvl, self.mnnvl_buffer, self.mnnvl_buff_flags, max_num_elements_mnnvl = get_allreduce_mnnvl_workspace(
                self.mapping, dtype)

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

        if all_reduce_params is None:
            all_reduce_params = AllReduceParams()

        output = torch.ops.trtllm.all_reduce(input, all_reduce_params.residual, all_reduce_params.norm_weight, all_reduce_params.scale, all_reduce_params.bias, self.mapping.tp_group, self.strategy,
                                    all_reduce_params.fusion_op, all_reduce_params.eps, all_reduce_params.trigger_completion_at_end, self.buffer_mnnvl_shape, self.workspace,
                        self.mnnvl_buffer, self.mnnvl_buff_flags)

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
