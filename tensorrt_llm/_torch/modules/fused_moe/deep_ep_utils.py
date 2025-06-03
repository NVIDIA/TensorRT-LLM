from typing import List, Optional, Tuple, Union

import torch
from deep_ep import Buffer, EventOverlap

from tensorrt_llm._utils import local_mpi_size

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(comm, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    world_size = comm.Get_size()
    for config in (Buffer.get_dispatch_config(world_size),
                   Buffer.get_combine_config(world_size)):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
            num_nvl_bytes)
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, world_size),
            num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        if _buffer is not None:
            raise NotImplementedError("not implemented buffer change")
        _buffer = Buffer(None,
                         num_nvl_bytes,
                         num_rdma_bytes,
                         num_nvl_peers=local_mpi_size(),
                         comm=comm)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(
        x,
        handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


# You may call this function at the framework initialization
def low_latency_get_buffer(comm, num_max_dispatch_tokens_per_rank: int,
                           hidden: int, num_experts: int) -> Buffer:
    # NOTES: the low-latency mode will consume much more space than the normal mode
    # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256
    global _buffer
    world_size = comm.Get_size()
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, world_size, num_experts)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or not _buffer.low_latency_mode or _buffer.num_rdma_bytes < num_rdma_bytes:
        # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
        assert num_experts % world_size == 0
        _buffer = Buffer(None,
                         0,
                         num_rdma_bytes,
                         low_latency_mode=True,
                         num_qps_per_rank=num_experts // world_size,
                         comm=comm)
    return _buffer


def low_latency_dispatch(hidden_states: torch.Tensor, topk_idx: torch.Tensor,
                         num_max_dispatch_tokens_per_rank: int,
                         num_experts: int):
    global _buffer

    # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
    recv_hidden_states, recv_expert_count, handle, event, hook = \
        _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                                     use_fp8=False, async_finish=False, return_recv_hook=False)

    # NOTES: the actual tensor will not be received only if you call `hook()`,
    # it is useful for double-batch overlapping, but **without any SM occupation**
    # If you don't want to overlap, please set `return_recv_hook=False`
    # Later, you can use our GEMM library to do the computation with this specific format
    return recv_hidden_states, recv_expert_count, handle


def low_latency_combine(hidden_states: torch.Tensor, topk_idx: torch.Tensor,
                        topk_weights: torch.Tensor, handle: Tuple):
    global _buffer

    # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
    combined_hidden_states, event_overlap, hook = \
        _buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle,
                                    async_finish=False, return_recv_hook=False)

    # NOTES: the same behavior as described in the dispatch kernel
    return combined_hidden_states
