import functools
from contextlib import contextmanager
from typing import Generator, List

import torch
from strenum import StrEnum

from tensorrt_llm.bindings.internal.runtime import \
    CudaVirtualMemoryAllocatorRestoreMode as RestoreMode
from tensorrt_llm.bindings.internal.runtime import (
    clear_virtual_memory_allocator, get_virtual_memory_manager,
    set_virtual_memory_allocator)

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


@contextmanager
def _virtual_memory_helper(tag: str, mode: RestoreMode):
    stream = torch.cuda.current_stream()
    set_virtual_memory_allocator(tag, mode, stream.cuda_stream)
    try:
        yield
    finally:
        clear_virtual_memory_allocator()


def _scope(
    tag: str,
    mode: RestoreMode = RestoreMode.NONE
) -> Generator[torch.cuda.MemPool, None, None]:
    """A context manager that routes allocations to virtual memory allocator
    using given tag and backed mode.

    :param tag: The tag to reference the memory for release and materialize
    :param mode: The backed mode to choose how the memory content is backed up
    """
    pool = torch.cuda.MemPool(_get_torch_pluggable_virtual_memory_allocator())
    with _virtual_memory_helper(tag, mode), torch.cuda.use_mem_pool(pool):
        yield pool


scope = contextmanager(_scope)


@contextmanager
def maybe_scope(
    enable: bool,
    tag: str,
    mode: RestoreMode = RestoreMode.NONE
) -> Generator[torch.cuda.MemPool | None, None, None]:
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
    # MODEL_EXTRA = "_no_capture_model_extra"  # TODO: remove _no_capture after torch fix crash on torch.cuda.empty_cache()
    MODEL_EXTRA = "model_extra"
    EXTRA_RESOURCES = "executor_extra"
    KV_CACHE = "kv_cache"
    MODEL_ENGINE_MAIN = "model"
    MODEL_ENGINE_DRAFT = "draft_model"


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
