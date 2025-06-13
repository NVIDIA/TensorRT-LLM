from contextlib import contextmanager

import torch

from tensorrt_llm.bindings.internal.runtime import \
    CudaVirtualAddressAllocatorBackedMode as BackedMode
from tensorrt_llm.bindings.internal.runtime import (
    get_virtual_memory_manager, pop_virtual_address_allocator,
    push_virtual_address_allocator)

__all__ = ["BackedMode", "scope", "release_with_mark", "materialize_with_mark"]

_virtual_memory_allocator = None


def _get_torch_pluggable_virtual_memory_allocator():
    global _virtual_memory_allocator
    if _virtual_memory_allocator is not None:
        return _virtual_memory_allocator.allocator()

    th_common = next(path for path in torch.classes.loaded_libraries
                     if 'th_common' in path)
    _virtual_memory_allocator = torch.cuda.CUDAPluggableAllocator(
        th_common, 'tensorrt_llm_virtual_memory_alloc',
        'tensorrt_llm_virtual_memory_free')
    return _virtual_memory_allocator.allocator()


@contextmanager
def _virtual_address_helper(mark: str, mode: BackedMode):
    stream = torch.cuda.current_stream()
    push_virtual_address_allocator(mark, mode, stream.cuda_stream)
    try:
        yield
    finally:
        pop_virtual_address_allocator()


@contextmanager
def scope(mark: str, mode: BackedMode = BackedMode.NONE):
    """A context manager that routes allocations to virtual memory allocator
    using given mark and backed mode.

    :param mark: The mark to reference the memory for release and materialize
    :param mode: The backed mode to choose how the memory content is backed up
    """
    pool = torch.cuda.MemPool(_get_torch_pluggable_virtual_memory_allocator())
    with _virtual_address_helper(mark, mode), torch.cuda.use_mem_pool(pool):
        yield pool


def release_with_mark(*marks: str):
    """Release virtual memory allocated with given marks

    :param marks: The mark of the scope when the virtual memory is allocated
    :return: Number of memory blobs released
    """
    manager = get_virtual_memory_manager()
    released_blobs = sum(manager.release_with_mark(mark) for mark in marks)
    return released_blobs


def materialize_with_mark(*marks: str):
    """Materialize virtual memory allocated with given marks

    :param marks: The mark of the scope when the virtual memory is allocated
    :return: Number of memory blobs materialized
    """
    manager = get_virtual_memory_manager()
    materialized_blobs = sum(
        manager.materialize_with_mark(mark) for mark in marks)
    return materialized_blobs
