# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind NCCL symmetric-window buffers to PyTorch CUDA graph memory pools.

Every TRT-LLM CUDA graph capture that may use NCCL symmetric windows must bind
its shared graph-pool handle with ``nccl_window_graph_owner``. Most captures
should use ``nccl_window_graph_capture``, which also starts the PyTorch graph.
Bypassing ownership emits a warning and uses unregistered buffers to prevent unsafe reuse,
which preserves correctness but loses symmetric-window performance.
"""

import contextlib
import operator
import threading
from typing import Any, Iterator

import torch

_EAGER_OWNER = -1
_OWNER_BASE = 1 << 62
_next_pool_owner = _OWNER_BASE
_pool_owners: dict[tuple[int, int], int] = {}
_pool_owners_lock = threading.Lock()
_capture_owner = threading.local()


def _pool_key(pool: Any) -> tuple[int, int]:
    if not isinstance(pool, tuple) or len(pool) != 2:
        raise TypeError(f"Expected a CUDA graph pool handle, got {pool!r}")
    try:
        return operator.index(pool[0]), operator.index(pool[1])
    except TypeError as error:
        raise TypeError(f"Invalid CUDA graph pool handle: {pool!r}") from error


def _get_or_create_pool_owner(pool: Any) -> tuple[int, bool]:
    global _next_pool_owner

    key = _pool_key(pool)
    with _pool_owners_lock:
        owner = _pool_owners.get(key)
        created = owner is None
        if owner is None:
            _next_pool_owner += 1
            owner = _next_pool_owner
            _pool_owners[key] = owner
        return owner, created


def _shared_pool_owner(pool: Any) -> int:
    owner, _ = _get_or_create_pool_owner(pool)
    return owner


@contextlib.contextmanager
def nccl_window_graph_owner(pool: Any) -> Iterator[None]:
    """Bind NCCL-window requests in this scope to a CUDA graph memory pool."""
    owner, created = _get_or_create_pool_owner(pool)
    previous_owner = getattr(_capture_owner, "value", _EAGER_OWNER)
    torch.ops.trtllm.set_nccl_window_graph_owner(owner)
    _capture_owner.value = owner
    completed = False
    try:
        yield
        completed = True
    finally:
        torch.ops.trtllm.set_nccl_window_graph_owner(previous_owner)
        _capture_owner.value = previous_owner
        if created and not completed:
            abandon_nccl_window_graph_owner(pool)


@contextlib.contextmanager
def nccl_window_graph_capture(
    graph: torch.cuda.CUDAGraph,
    pool: Any,
    **capture_kwargs: Any,
) -> Iterator[None]:
    with nccl_window_graph_owner(pool):
        device = torch.cuda.current_device()
        release_epoch = torch.ops.trtllm.get_nccl_window_buffer_release_epoch()
        with torch.cuda.graph(graph, pool=pool, **capture_kwargs):
            # capture_begin has synchronized PyTorch warmup work. Promote explicitly
            # released windows before the captured body can request them.
            torch.ops.trtllm.promote_nccl_window_buffer_releases(device, release_epoch)
            yield


def abandon_nccl_window_graph_owner(pool: Any) -> None:
    """Forget a locally failed pool without making its C++ buffers reusable.

    Failure and destructor timing need not match across ranks. Keeping any C++
    buffers bound to their now-unreachable owner prevents that timing from
    changing later best-fit selection; communicator teardown reclaims them.
    """
    key = _pool_key(pool)
    with _pool_owners_lock:
        _pool_owners.pop(key, None)


def release_nccl_window_graph_owner(pool: Any) -> None:
    key = _pool_key(pool)
    with _pool_owners_lock:
        owner = _pool_owners.get(key)
        if owner is not None:
            torch.ops.trtllm.release_nccl_window_graph_owner(owner)
            del _pool_owners[key]
