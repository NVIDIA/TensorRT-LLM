import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BufferBlock:
    buffer: torch.Tensor = None
    pin_memory: bool = False


# Intention to have this buffer is to reuse buffer tensors across graph and non-graph
# situation (across layer/round).
# When forward is under graph capturing, one stream is created and all tensors' memory
# is associated with this stream and be kept in a graph pool. Then, all buffer memory
# allocated during graph capture won't be released back to allocator/system.
# Then, in non-graph mode, additional buffers are allocated which give bigger pressure
# on memory consumption at runtime.
# Timeline example:
#   [t0] start cudagraph capture
#   [t1] A = torch.zeros(....) -> allocate buffer A and put into graph pool
#   [t2] end cudagraph capture
#   [t3] in non-graph forward
#   [t4] A = torch.zeros(....) -> allocate buffer A in allocator but not use memory in cudagraph pool
#        OOM may happen
# TODO:
# The final resolution to this problem shall be supported in pytorch that to allocate memory
#    from a give pool, it's the graph pool here.
# It will be like
#    try:
#        with torch.cuda.use_mem_pool(graphpool):
#            allocate_memory_here
#    except exception as ex:
#        allocate_memory_outside of graphpool
# Need some archeteture change:
#    1. a. set a thread local graphpool context object when cudagraphRunner start a fn
#       b. check and get the thread local graphpool
#       b. allocate memory
#    2. aggregate workspaces in the same OP to be a big one in graph pool
#       allocate memory for the big workspace and slice them into small ones.
#       However, in non-graph mode, allocate workspace one by one
class Buffers:

    def __init__(self):
        self.buffers: dict[str, list[BufferBlock]] = {}

    def get_buffer(self, tensor_shape: list[int], dtype: torch.dtype,
                   buffer_name: str, pin_memory: bool):

        def select_buffer_with_more_elements(
            pinned_buffer: Optional[torch.Tensor],
            runtime_buffer: Optional[torch.Tensor]
        ) -> tuple[Optional[torch.Tensor]]:
            if pinned_buffer is None:
                return runtime_buffer
            if runtime_buffer is None:
                return pinned_buffer

            return runtime_buffer if runtime_buffer.buffer.numel(
            ) > pinned_buffer.buffer.numel() else pinned_buffer

        def view_to(buffer: torch.Tensor, dtype: torch.dtype,
                    tensor_shape: list[int]) -> torch.Tensor:
            return buffer[0:math.prod(tensor_shape) *
                          dtype.itemsize].view(dtype).view(tensor_shape)

        # all buffers are allocated with 1 byte element size
        element_size = dtype.itemsize
        required_memory_size = math.prod(tensor_shape) * element_size
        candidate_buffers = self.buffers.get(buffer_name, [])
        pinned_buffer = None
        free_buffer = None
        for buffer in candidate_buffers:
            buffer_size = buffer.buffer.numel()
            if buffer_size >= required_memory_size:
                if buffer.pin_memory:
                    pinned_buffer = buffer
                else:
                    free_buffer = buffer

            if free_buffer is not None and pinned_buffer is not None:
                break

        if pin_memory:
            if pinned_buffer is not None:
                return view_to(pinned_buffer.buffer, dtype, tensor_shape)
            elif free_buffer is not None:
                free_buffer.pin_memory = True
                return view_to(free_buffer.buffer, dtype, tensor_shape)

        if buffer_name in self.buffers:
            candidate_buffers = self.buffers.get(buffer_name, [])
            for buffer in list(candidate_buffers):
                if not buffer.pin_memory:
                    # Need to call del BufferBlock.buffer, otherwise memory isn't
                    # released and OOM may happen.
                    del buffer.buffer
                    candidate_buffers.remove(buffer)

        new_buffer = torch.zeros((required_memory_size, ),
                                 device='cuda',
                                 dtype=torch.uint8)
        self.buffers.setdefault(buffer_name, []).append(
            BufferBlock(buffer=new_buffer, pin_memory=pin_memory))
        return view_to(new_buffer, dtype, tensor_shape)


_buffer = Buffers()


def get_memory_buffer():
    global _buffer
    return _buffer
