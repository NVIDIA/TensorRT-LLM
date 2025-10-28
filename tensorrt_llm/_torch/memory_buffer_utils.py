import contextlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger


def get_smallest_key_greater_than(ordered_dict, target_value):
    """
    Return (k, ordered_dict[k]) where k is the smallest key with k >= target_value,
    or (None, None) if not found.
    """
    min_key = min((k for k in ordered_dict.keys() if k >= target_value),
                  default=None)
    return (min_key, ordered_dict[min_key]) if min_key is not None else (None,
                                                                         None)


def get_biggest_key_smaller_than(ordered_dict, target_value):
    """
    Return (k, ordered_dict[k]) where k is the largest key with k < target_value,
    or (None, None) if not found.
    """
    max_key = max((k for k in ordered_dict.keys() if k < target_value),
                  default=None)
    return (max_key, ordered_dict[max_key]) if max_key is not None else (None,
                                                                         None)


def get_size_in_byte(target_shape: list[int], target_dtype: torch.dtype):
    return math.prod(target_shape) * target_dtype.itemsize


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
        self.managed_buffers = OrderedDict()
        self.max_buffer_concurrency = 0

    @staticmethod
    def _view_as(buffer: torch.Tensor, target_shape: list[int],
                 target_dtype: torch.dtype) -> torch.Tensor:
        """Safely creates a view of a raw byte buffer with the desired shape and dtype."""
        # The buffer is stored as uint8, so its numel is its size in bytes.
        required_memory_size = get_size_in_byte(target_shape, target_dtype)
        if buffer.numel() < required_memory_size:
            raise ValueError(
                "Buffer is too small for the requested shape and dtype.")

        # Slice the buffer to the exact required size, then view it with the correct type and shape.
        return buffer[:required_memory_size].view(target_dtype).view(
            target_shape)

    def _get_managed_buffer(self, required_memory_size: int):
        size, buffer = get_smallest_key_greater_than(self.managed_buffers,
                                                     required_memory_size)

        if size is not None and buffer is not None:
            return buffer

        size_1, buffer_1 = get_biggest_key_smaller_than(self.managed_buffers,
                                                        required_memory_size)
        if size_1 is not None and buffer is not None:
            del self.managed_buffers[size_1]

        new_buffer_tensor = None
        try:
            with torch.cuda.memory.use_mem_pool(get_shared_pool()):
                new_buffer_tensor = torch.zeros((required_memory_size, ),
                                                device='cuda',
                                                dtype=torch.uint8)
        except Exception as ex:
            # Need to check if this is an OOM exception
            logger.debug(
                f"Exception happened to create tensor from given memory pool: {str(ex)}"
            )
            # if exception happens during allocating memory from shared pool, retry
            # to allocate from default pool
            new_buffer_tensor = torch.zeros((required_memory_size, ),
                                            device='cuda',
                                            dtype=torch.uint8)

        self.managed_buffers[required_memory_size] = new_buffer_tensor

        return new_buffer_tensor

    def get_buffer(self, tensor_shape: list[int], dtype: torch.dtype,
                   buffer_name: str, reserve_buffer: bool):

        # all buffers are allocated with 1 byte element size
        required_memory_size = math.prod(tensor_shape) * dtype.itemsize

        if buffer_name is None or len(buffer_name) == 0:
            return _get_managed_buffer(required_memory_size)

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
        new_buffer_tensor = None
        try:
            with torch.cuda.memory.use_mem_pool(get_shared_pool()):
                new_buffer_tensor = torch.zeros((required_memory_size, ),
                                                device='cuda',
                                                dtype=torch.uint8)
        except Exception as ex:
            # Need to check if this is an OOM exception
            logger.debug(
                f"Exception happened to create tensor from given memory pool: {str(ex)}"
            )
            # if exception happens during allocating memory from shared pool, retry
            # to allocate from default pool
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


_shared_pool = None


def set_shared_pool(shared_pool):
    """Sets the global memory pool for buffer allocation.

    Args:
        shared_pool: A CUDA memory pool object to use for allocations.
    """
    global _shared_pool
    _shared_pool = shared_pool


def get_shared_pool():
    """Retrieves the current global memory pool.

    Returns:
        The current memory pool, or None if not set.
    """
    return _shared_pool


@contextlib.contextmanager
def with_shared_pool(shared_pool) -> contextlib.AbstractContextManager:
    """Temporarily sets a preferred memory pool and restores the previous one on exit.

    This context manager allows temporarily switching to a different memory pool
    for CUDA graph operations, ensuring the original pool is restored even if
    an exception occurs.

    Args:
        shared_pool: The memory pool to use within the context.

    Yields:
        None

    Example:
        >>> with with_shared_pool(shared_pool):
        ...     # Allocations within this block use shared_pool
        ...     tensor = allocate_buffer(...)
    """
    old_shared_pool = get_shared_pool()
    set_shared_pool(shared_pool)
    try:
        yield
    finally:
        set_shared_pool(old_shared_pool)
