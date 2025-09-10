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

        if pin_memory and pinned_buffer is not None:
            return view_to(pinned_buffer.buffer, dtype, tensor_shape)
        else:
            buffer = select_buffer_with_more_elements(pinned_buffer,
                                                      free_buffer)
            if buffer is not None:
                buffer.pin_memory = True if pin_memory else buffer.pin_memory
                return view_to(buffer.buffer, dtype, tensor_shape)

        if buffer_name in self.buffers:
            candidate_buffers = self.buffers.get(buffer_name, [])
            remove_idx = []
            for idx, buffer in enumerate(candidate_buffers):
                if buffer.pin_memory == False:
                    remove_idx.append(idx)

            for idx in remove_idx[::-1]:
                removed = candidate_buffers.pop(idx)
                del removed.buffer

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
