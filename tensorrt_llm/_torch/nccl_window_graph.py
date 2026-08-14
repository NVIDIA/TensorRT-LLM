# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind NCCL symmetric-window buffers to PyTorch CUDA graph memory pools."""

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


def _pool_key(pool: Any) -> tuple[int, int]:
    if not isinstance(pool, tuple) or len(pool) != 2:
        raise TypeError(f"Expected a CUDA graph pool handle, got {pool!r}")
    try:
        return operator.index(pool[0]), operator.index(pool[1])
    except TypeError as error:
        raise TypeError(f"Invalid CUDA graph pool handle: {pool!r}") from error


def _shared_pool_owner(pool: Any) -> int:
    global _next_pool_owner

    key = _pool_key(pool)
    with _pool_owners_lock:
        owner = _pool_owners.get(key)
        if owner is None:
            _next_pool_owner += 1
            owner = _next_pool_owner
            _pool_owners[key] = owner
        return owner


@contextlib.contextmanager
def nccl_window_graph_capture(
    graph: torch.cuda.CUDAGraph,
    pool: Any,
    **capture_kwargs: Any,
) -> Iterator[None]:
    owner = _shared_pool_owner(pool)
    torch.ops.trtllm.set_nccl_window_graph_owner(owner)
    try:
        with torch.cuda.graph(graph, pool=pool, **capture_kwargs):
            yield
    finally:
        torch.ops.trtllm.set_nccl_window_graph_owner(_EAGER_OWNER)


def release_nccl_window_graph_owner(pool: Any) -> None:
    key = _pool_key(pool)
    with _pool_owners_lock:
        owner = _pool_owners.get(key)
        if owner is not None:
            torch.ops.trtllm.release_nccl_window_graph_owner(owner)
            del _pool_owners[key]
