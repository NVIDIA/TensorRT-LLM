# Adapted from SGLang's breakable CUDA graph implementation.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_breakable_cuda_graph_active: ContextVar[bool] = ContextVar(
    "breakable_cuda_graph_active", default=False
)
_parent_moe_graph_enabled: ContextVar[bool] = ContextVar("parent_moe_graph_enabled", default=False)


def is_in_breakable_cuda_graph() -> bool:
    """Return whether the current context is executing a BCG region."""
    return _breakable_cuda_graph_active.get()


@contextmanager
def enable_breakable_cuda_graph() -> Iterator[None]:
    """Mark capture or replay work as breakable CUDA graph execution."""
    token = _breakable_cuda_graph_active.set(True)
    try:
        yield
    finally:
        _breakable_cuda_graph_active.reset(token)


def is_parent_moe_graph_enabled() -> bool:
    """Return whether a dynamic MoE parent graph is warming up or capturing."""
    return _parent_moe_graph_enabled.get()


@contextmanager
def enable_parent_moe_graph() -> Iterator[None]:
    """Opt into device-backed READY for the runner's warmup and capture."""
    token = _parent_moe_graph_enabled.set(True)
    try:
        yield
    finally:
        _parent_moe_graph_enabled.reset(token)
