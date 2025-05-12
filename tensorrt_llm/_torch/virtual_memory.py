import functools
from contextlib import contextmanager
from typing import Generator

import torch

from tensorrt_llm.bindings.internal.runtime import \
    CudaVirtualMemoryAllocatorRestoreMode as RestoreMode
from tensorrt_llm.bindings.internal.runtime import (
    get_virtual_memory_manager, pop_virtual_memory_allocator,
    push_virtual_memory_allocator)

__all__ = [
    "RestoreMode", "maybe_scope", "scope", "release_with_mark",
    "materialize_with_mark"
]


@functools.cache
def _get_torch_pluggable_virtual_memory_allocator():
    th_common = next(path for path in torch.classes.loaded_libraries
                     if 'th_common' in path)
    virtual_memory_allocator = torch.cuda.CUDAPluggableAllocator(
        th_common, 'tensorrt_llm_virtual_memory_alloc',
        'tensorrt_llm_virtual_memory_free')
    return virtual_memory_allocator.allocator()


@contextmanager
def _virtual_address_helper(mark: str, mode: RestoreMode):
    stream = torch.cuda.current_stream()
    push_virtual_memory_allocator(mark, mode, stream.cuda_stream)
    try:
        yield
    finally:
        pop_virtual_memory_allocator()


def _scope(
    mark: str,
    mode: RestoreMode = RestoreMode.NONE
) -> Generator[torch.cuda.MemPool, None, None]:
    """A context manager that routes allocations to virtual memory allocator
    using given mark and backed mode.

    :param mark: The mark to reference the memory for release and materialize
    :param mode: The backed mode to choose how the memory content is backed up
    """
    pool = torch.cuda.MemPool(_get_torch_pluggable_virtual_memory_allocator())
    with _virtual_address_helper(mark, mode), torch.cuda.use_mem_pool(pool):
        yield pool


scope = contextmanager(_scope)


@contextmanager
def maybe_scope(
    enable: bool,
    mark: str,
    mode: RestoreMode = RestoreMode.NONE
) -> Generator[torch.cuda.MemPool | None, None, None]:
    if enable:
        yield from _scope(mark, mode)
    else:
        yield


def release_with_mark(*marks: str) -> int:
    """Release virtual memory allocated with given marks

    :param marks: The mark of the scope when the virtual memory is allocated
    :return: Number of memory blobs released
    """
    manager = get_virtual_memory_manager()
    released_blobs = sum(manager.release_with_mark(mark) for mark in marks)
    return released_blobs


def materialize_with_mark(*marks: str) -> int:
    """Materialize virtual memory allocated with given marks

    :param marks: The mark of the scope when the virtual memory is allocated
    :return: Number of memory blobs materialized
    """
    manager = get_virtual_memory_manager()
    materialized_blobs = sum(
        manager.materialize_with_mark(mark) for mark in marks)
    return materialized_blobs
