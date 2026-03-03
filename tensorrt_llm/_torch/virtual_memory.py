import contextlib
import functools
from contextlib import contextmanager
from typing import Generator, List

import torch
from strenum import StrEnum

from tensorrt_llm.bindings.internal.runtime import \
    CudaVirtualMemoryAllocatorRestoreMode as RestoreMode
from tensorrt_llm.bindings.internal.runtime import (
    get_virtual_memory_manager, pop_virtual_memory_allocator,
    push_virtual_memory_allocator)

__all__ = [
    "RestoreMode", "maybe_scope", "scope", "release_with_tag",
    "materialize_with_tag"
]


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

    stream = torch.cuda.current_stream()
    push_virtual_memory_allocator(tag, mode, stream.cuda_stream)

    proxy = _MultiPoolProxy()
    pool = torch.cuda.MemPool(_get_torch_pluggable_virtual_memory_allocator())
    pool_ctx = torch.cuda.use_mem_pool(pool)
    pool_ctx.__enter__()
    proxy._add(pool)
    _pool_stack.append((pool_ctx, proxy))

    try:
        yield proxy
    finally:
        current_ctx, _ = _pool_stack[-1]
        current_ctx.__exit__(None, None, None)
        _pool_stack.pop()
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


class ExecutorMemoryType(StrEnum):
    SAMPLER = "sampler"
    DRAFTER = "drafter"
    GUIDED_DECODER = "guided_decoder"
    SPEC_RESOURCES = "spec_resource_manager"
    INIT_KV_CACHE = "_no_capture_init_kv_cache"
    INIT_EXTRA_RESOURCES = "_no_capture_init_extra_resources"
    MODEL_EXTRA = "model_extra"
    EXTRA_RESOURCES = "executor_extra"
    KV_CACHE = "kv_cache"
    MODEL_ENGINE_MAIN = "model"
    MODEL_ENGINE_DRAFT = "draft_model"
    MODEL_WEIGHTS_MAIN = "model_weights"
    MODEL_WEIGHTS_DRAFT = "draft_model_weights"


def verify_sleep_wakeup_tags(tags_strs: List[str]) -> List[ExecutorMemoryType]:
    tags = []
    for tag_str in tags_strs:
        try:
            tags.append(ExecutorMemoryType(tag_str))
        except ValueError:
            raise ValueError(
                f"Unknown memory tag '{tag_str}'."
                f"Valid tags are: {[t.value for t in ExecutorMemoryType]}")
    return tags


def release_with_tag(*tags: str) -> int:
    """Release virtual memory allocated with given tags

    :param tags: The tag of the scope when the virtual memory is allocated
    :return: Number of memory blobs released
    """
    manager = get_virtual_memory_manager()
    released_blobs = sum(manager.release_with_tag(tag) for tag in tags)
    return released_blobs


def materialize_with_tag(*tags: str) -> int:
    """Materialize virtual memory allocated with given tags

    :param tags: The tag of the scope when the virtual memory is allocated
    :return: Number of memory blobs materialized
    """
    manager = get_virtual_memory_manager()
    materialized_blobs = sum(manager.materialize_with_tag(tag) for tag in tags)
    return materialized_blobs
