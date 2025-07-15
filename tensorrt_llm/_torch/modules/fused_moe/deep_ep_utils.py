# Adapted from
# https://github.com/deepseek-ai/DeepEP/blob/aae9fa9a6dd0fec2a723fbb85ec4b22460fab670/README.md
import os
import weakref
from typing import List, Tuple, Union

import torch

from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.mapping import Mapping

try:
    from tensorrt_llm.deep_ep import Buffer
    deep_ep_installed = True
except ImportError:
    deep_ep_installed = False


class VariableLengthBuffer:
    """ A wrapper of deep_ep.Buffer that accepts future size change
    """

    def __init__(self, mapping: Mapping):
        self.comm = mpi_comm().Split(mapping.pp_rank, mapping.moe_ep_rank)
        self.buffer = None

    def __del__(self):
        self.comm.Free()

    def reserve(self, hidden_size: int, hidden_dtype: torch.dtype):
        """ Ensure the buffer capacity is large enough.

            Reserve is a collective operation that requires all EP ranks to be sync
        """
        # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
        num_nvl_bytes, num_rdma_bytes = 0, 0
        hidden_bytes = hidden_size * max(hidden_dtype.itemsize,
                                         torch.bfloat16.itemsize)
        world_size = self.comm.Get_size()
        for config in (Buffer.get_dispatch_config(world_size),
                       Buffer.get_combine_config(world_size)):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
                num_nvl_bytes)
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, world_size),
                num_rdma_bytes)

        # Allocate a buffer if not existed or not enough buffer size
        if self.buffer is None or self.buffer.num_nvl_bytes < num_nvl_bytes or self.buffer.num_rdma_bytes < num_rdma_bytes:
            if self.buffer is not None:
                num_nvl_bytes = max(num_nvl_bytes, self.buffer.num_nvl_bytes)
                num_rdma_bytes = max(num_rdma_bytes, self.buffer.num_rdma_bytes)
            del self.buffer  # Destruct before Construct
            self.buffer = Buffer(None,
                                 num_nvl_bytes,
                                 num_rdma_bytes,
                                 comm=self.comm)

    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                 num_experts: int) -> \
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple]:
        # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
        # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
        # refer to the docs of `Buffer.dispatch`

        # Calculate layout before actual dispatch
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
            self.buffer.get_dispatch_layout(topk_idx, num_experts)
        assert event.event is None

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
        # For more advanced usages, please refer to the docs of the `dispatch` function
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
            self.buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                                 num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                                 is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert)
        assert event.event is None

        # For event management, please refer to the docs of the `EventOverlap` class
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle

    def combine(self, x: torch.Tensor, handle: Tuple) -> torch.Tensor:
        # Do MoE combine
        # For more advanced usages, please refer to the docs of the `combine` function
        combined_x, _, event = self.buffer.combine(x, handle)
        assert event.event is None

        # For event management, please refer to the docs of the `EventOverlap` class
        return combined_x


class VariableLengthLowLatencyBuffer:
    """ A wrapper of deep_ep.Buffer that accepts future size change
    """

    def __init__(self, mapping: Mapping):
        self.comm = mpi_comm().Split(mapping.pp_rank, mapping.moe_ep_rank)
        self.buffer = None
        self.num_max_dispatch_tokens_per_rank = None

    def __del__(self):
        self.comm.Free()

    def reserve(self, num_max_dispatch_tokens_per_rank: int, hidden_size: int,
                num_experts: int):
        """ Ensure the buffer capacity is large enough.

            Reserve is a collective operation that requires all EP ranks to be sync
        """
        # NOTES: the low-latency mode will consume much more space than the normal mode
        # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256
        world_size = self.comm.Get_size()
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden_size, world_size,
            num_experts)
        allow_nvlink_for_low_latency_mode = (os.environ.get(
            "TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE", "0") == "0")

        # Allocate a buffer if not existed or not enough buffer size
        if self.buffer is None or self.buffer.num_rdma_bytes < num_rdma_bytes:
            # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
            assert num_experts % world_size == 0
            del self.buffer  # Destruct before Construct
            self.buffer = Buffer(None,
                                 0,
                                 num_rdma_bytes,
                                 low_latency_mode=True,
                                 num_qps_per_rank=num_experts // world_size,
                                 allow_nvlink_for_low_latency_mode=
                                 allow_nvlink_for_low_latency_mode,
                                 comm=self.comm)

    def low_latency_dispatch(self, hidden_states: torch.Tensor,
                             topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int,
                             num_experts: int):
        if self.num_max_dispatch_tokens_per_rank is None:
            self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        if num_max_dispatch_tokens_per_rank != self.num_max_dispatch_tokens_per_rank:
            raise NotImplementedError(
                "There are issues if `low_latency_dispatch` calls use different `num_max_dispatch_tokens_per_rank` values"
            )

        # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
        recv_hidden_states, recv_expert_count, handle, event, hook = \
            self.buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, use_fp8=False)
        assert event.event is None
        assert hook is None

        # NOTES: the actual tensor will not be received only if you call `hook()`,
        # it is useful for double-batch overlapping, but **without any SM occupation**
        # If you don't want to overlap, please set `return_recv_hook=False`
        # Later, you can use our GEMM library to do the computation with this specific format
        return recv_hidden_states, recv_expert_count, handle

    def low_latency_combine(self, hidden_states: torch.Tensor,
                            topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: Tuple):
        # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
        combined_hidden_states, event, hook = \
            self.buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle)
        assert event.event is None
        assert hook is None

        # NOTES: the same behavior as described in the dispatch kernel
        return combined_hidden_states

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int,
                                 hidden: int, num_experts: int) -> None:
        self.buffer.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank,
                                             hidden, num_experts)


class BufferPool:
    """ A pool that allocates buffers on demand.

        Although the pool interface allows creating multiple buffers, the
        current version of DeepEP supports at most one `deep_ep.Buffer` at a
        time. Please ensure that all references to `VariableLengthBuffer` are
        released before getting another buffer.
    """

    def __init__(self):
        self.buffers: Map[Mapping,
                          weakref.ReferenceType[VariableLengthBuffer]] = {}
        self.low_latency_buffers: Map[
            Mapping,
            weakref.ReferenceType[VariableLengthLowLatencyBuffer]] = {}

    def get_buffer(self, mapping: Mapping) -> VariableLengthBuffer:
        """ Get_buffer is a collective operation that requires all ranks to be sync
        """
        if mapping in self.buffers and self.buffers[mapping]() is not None:
            buffer = self.buffers[mapping]()
        else:
            buffer = VariableLengthBuffer(mapping)
            self.buffers[mapping] = weakref.ref(buffer)
        return buffer

    def get_low_latency_buffer(
            self, mapping: Mapping) -> VariableLengthLowLatencyBuffer:
        """ Get_low_latency_buffer is a collective operation that requires all ranks to be sync
        """
        if mapping in self.low_latency_buffers and self.low_latency_buffers[
                mapping]() is not None:
            buffer = self.low_latency_buffers[mapping]()
        else:
            buffer = VariableLengthLowLatencyBuffer(mapping)
            self.low_latency_buffers[mapping] = weakref.ref(buffer)
        return buffer


# The default pool
# You may create own pools for better resource management.
buffer_pool = BufferPool()
