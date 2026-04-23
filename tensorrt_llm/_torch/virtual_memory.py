import contextlib
import functools
import gc
import time
from contextlib import contextmanager
from typing import Callable, Generator, TypeVar

import torch

from tensorrt_llm.bindings.internal.runtime import \
    CudaVirtualMemoryAllocatorRestoreMode as RestoreMode
from tensorrt_llm.bindings.internal.runtime import (
    get_virtual_memory_manager, pop_virtual_memory_allocator,
    push_virtual_memory_allocator)
from tensorrt_llm.logger import logger

__all__ = [
    "RestoreMode", "maybe_scope", "scope", "release_with_tag",
    "materialize_with_tag", "run_with_oom_retry"
]

T = TypeVar("T")
_OOM_RETRY_INTERVAL_SECONDS = 1.0


@functools.cache
def _get_torch_pluggable_virtual_memory_allocator():
    th_common = next(path for path in torch.classes.loaded_libraries
                     if 'th_common' in path)
    virtual_memory_allocator = torch.cuda.CUDAPluggableAllocator(
        th_common, 'tensorrt_llm_virtual_memory_alloc',
        'tensorrt_llm_virtual_memory_free')
    return virtual_memory_allocator.allocator()


class _MultiPoolProxy:
    """Reference holder for MemPool objects allocated during a scope.

    Each MemPool can only be used in ``torch.cuda.use_mem_pool`` once and
    nesting is not supported.  When a nested scope exits we must resume
    the parent with a *fresh* MemPool.  This proxy accumulates every pool
    created for its scope so they all stay alive until the caller deletes
    the proxy.
    """

    def __init__(self):
        self._pools: list[torch.cuda.MemPool] = []

    def _add(self, pool: torch.cuda.MemPool):
        self._pools.append(pool)

    def __del__(self):
        self._pools.clear()


_pool_stack: list[tuple[contextlib.AbstractContextManager,
                        _MultiPoolProxy]] = []


def _scope(
    tag: str,
    mode: RestoreMode = RestoreMode.NONE
) -> Generator[_MultiPoolProxy, None, None]:
    """A context manager that routes allocations to virtual memory allocator
    using given tag and backed mode.  Supports nesting.

    :param tag: The tag to reference the memory for release and materialize
    :param mode: The backed mode to choose how the memory content is backed up
    """

    # TODO(ytong): Remove these ugly code after we upgrade to PyTorch 2.10,
    #  which natively supports MemPool nesting:
    #  https://github.com/pytorch/pytorch/commit/d038b0130ec7c20ebcac219301292fd8e98a1ace
    if _pool_stack:
        parent_ctx, _ = _pool_stack[-1]
        parent_ctx.__exit__(None, None, None)

    pushed_allocator = False
    child_pushed = False
    try:
        stream = torch.cuda.current_stream()
        push_virtual_memory_allocator(tag, mode, stream.cuda_stream)
        pushed_allocator = True

        proxy = _MultiPoolProxy()
        pool = torch.cuda.MemPool(
            _get_torch_pluggable_virtual_memory_allocator())
        pool_ctx = torch.cuda.use_mem_pool(pool)
        pool_ctx.__enter__()
        proxy._add(pool)
        _pool_stack.append((pool_ctx, proxy))
        child_pushed = True
        yield proxy
    finally:
        if child_pushed:
            current_ctx, _ = _pool_stack.pop()
            current_ctx.__exit__(None, None, None)
        if pushed_allocator:
            pop_virtual_memory_allocator()

        if _pool_stack:
            new_pool = torch.cuda.MemPool(
                _get_torch_pluggable_virtual_memory_allocator())
            new_ctx = torch.cuda.use_mem_pool(new_pool)
            new_ctx.__enter__()
            _, parent_proxy = _pool_stack[-1]
            parent_proxy._add(new_pool)
            _pool_stack[-1] = (new_ctx, parent_proxy)


scope = contextmanager(_scope)


@contextmanager
def maybe_scope(
    enable: bool,
    tag: str,
    mode: RestoreMode = RestoreMode.NONE
) -> Generator[_MultiPoolProxy | None, None, None]:
    if enable:
        yield from _scope(tag, mode)
    else:
        yield


def release_with_tag(*tags: str) -> int:
    """Release virtual memory allocated with given tags

    :param tags: The tag of the scope when the virtual memory is allocated
    :return: Number of memory blobs released
    """
    manager = get_virtual_memory_manager()
    released_blobs = sum(manager.release_with_tag(tag) for tag in tags)
    return released_blobs


def _is_oom_error(error: Exception) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(
        error).lower()


def _cleanup_after_oom() -> None:
    for cleanup in (torch.cuda.synchronize, gc.collect, torch.cuda.empty_cache):
        try:
            cleanup()
        except Exception as cleanup_error:
            logger.debug("OOM cleanup step failed: %s", cleanup_error)


def run_with_oom_retry(action: Callable[[], T], *, description: str) -> T:
    retries = 0
    while True:
        try:
            result = action()
            if retries:
                logger.info("%s succeeded after %d retries", description,
                            retries)
            return result
        except Exception as error:
            if not _is_oom_error(error):
                raise

            retries += 1
            logger.warning(
                "%s hit OOM, waiting for capacity before retry %d: %s",
                description, retries, error)
            _cleanup_after_oom()
            time.sleep(_OOM_RETRY_INTERVAL_SECONDS)


def materialize_with_tag(*tags: str) -> int:
    """Materialize virtual memory allocated with given tags

    :param tags: The tag of the scope when the virtual memory is allocated
    :return: Number of memory blobs materialized
    """
    manager = get_virtual_memory_manager()
    materialized_blobs = 0
    for tag in tags:
        materialized_blobs += run_with_oom_retry(
            lambda tag=tag: manager.materialize_with_tag(tag),
            description=f"Materializing virtual memory for tag {tag}")
    return materialized_blobs
