import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BufferBlock:
    """A container for a buffer tensor and its state."""
    buffer: torch.Tensor = None
    is_reserved: bool = False


class Buffers:
    """
    Manages and reuses CUDA memory buffers to reduce allocation overhead,
    especially when interacting with CUDA graphs.

    This class maintains a pool of named buffers. When a buffer is requested,
    it tries to find an existing, available buffer that is large enough.
    If none is found, a new one is allocated and added to the pool. This helps
    avoid repeated allocations, which can be slow and cause memory fragmentation,
    particularly when the same operations are run inside and outside of a
    CUDA graph context.
    """

    def __init__(self):
        self.buffers: dict[str, list[BufferBlock]] = {}

    @staticmethod
    def _view_as(buffer: torch.Tensor, target_shape: list[int],
                 target_dtype: torch.dtype) -> torch.Tensor:
        """Safely creates a view of a raw byte buffer with the desired shape and dtype."""
        # The buffer is stored as uint8, so its numel is its size in bytes.
        required_size_in_bytes = math.prod(target_shape) * target_dtype.itemsize
        if buffer.numel() < required_size_in_bytes:
            raise ValueError(
                "Buffer is too small for the requested shape and dtype.")

        # Slice the buffer to the exact required size, then view it with the correct type and shape.
        return buffer[:required_size_in_bytes].view(target_dtype).view(
            target_shape)

    def get_buffer(self, tensor_shape: list[int], dtype: torch.dtype,
                   buffer_name: str, reserve_buffer: bool):

        # all buffers are allocated with 1 byte element size
        required_memory_size = math.prod(tensor_shape) * dtype.itemsize
        candidate_blocks = self.buffers.get(buffer_name, [])

        # Find the best-fit available buffer.
        best_fit_block: Optional[BufferBlock] = None
        smallest_sufficient_size = float('inf')
        for block in candidate_blocks:
            # Skip buffers that are too small.
            if block.buffer.numel() < required_memory_size:
                continue

            # Find the smallest buffer that is still large enough (best-fit).
            if block.buffer.numel() < smallest_sufficient_size:
                # Use reserved block if find one.
                if best_fit_block is not None and best_fit_block.is_reserved and not block.is_reserved:
                    continue

                best_fit_block = block
                smallest_sufficient_size = block.buffer.numel()

        if reserve_buffer and best_fit_block is not None:
            # A suitable buffer was found, so reuse it.
            best_fit_block.is_reserved = True
            return self._view_as(best_fit_block.buffer, tensor_shape, dtype)

        for block in list(candidate_blocks):
            if not block.is_reserved:
                # Need to call del BufferBlock.buffer, otherwise memory isn't
                # released and OOM may happen.
                del block.buffer
                candidate_blocks.remove(block)

        # No suitable buffer was found, so allocate a new one.
        # The new buffer is created with uint8 to represent raw bytes.
        new_buffer_tensor = torch.zeros((required_memory_size, ),
                                        device='cuda',
                                        dtype=torch.uint8)
        new_block = BufferBlock(buffer=new_buffer_tensor,
                                is_reserved=reserve_buffer)

        # Add the new buffer to the pool for this name.
        self.buffers.setdefault(buffer_name, []).append(new_block)
        return self._view_as(new_block.buffer, tensor_shape, dtype)


_buffer = Buffers()


def get_memory_buffers():
    global _buffer
    return _buffer
