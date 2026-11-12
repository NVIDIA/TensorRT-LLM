# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two launches on two CUDA streams, joined on the calling stream.

``fork_join(fn0, fn1, fork, join, aux)`` runs ``fn0`` on the calling stream and ``fn1`` on ``aux``:

    fork.record()                      # calling stream
    r0 = fn0()                         # calling stream
    with torch.cuda.stream(aux):
        fork.wait(); r1 = fn1(); join.record()
    join.wait()                        # calling stream, before anything reads r1

Both run on the calling stream, in that order, unless ``do_multi_stream()`` (the engine's
per-thread flag, set while CUDA graphs are captured) holds and ``aux`` is given. Outside graph
capture the CUDA tensors ``fn1`` returns (directly or in nested tuples / lists / dict values) are
recorded on the calling stream, so the caching allocator does not hand their memory to later
``aux`` work while the calling stream still reads them.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from ...common.multi_stream import do_multi_stream


def _record_on(obj: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            try:
                obj.record_stream(stream)
            except RuntimeError as error:
                # cudaMallocAsync asserts on memory it did not allocate (an external buffer wrapped
                # as a tensor), which it never frees: nothing to record.
                if (
                    torch.cuda.memory.get_allocator_backend() != "cudaMallocAsync"
                    or "ptr not found in ptr_info" not in str(error)
                ):
                    raise
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _record_on(item, stream)
    elif isinstance(obj, dict):
        for item in obj.values():
            _record_on(item, stream)


def fork(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    join_event: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any, Optional[torch.cuda.Event]]:
    """``fn0()`` and ``fn1()`` and the event the calling stream must wait on before it reads
    ``fn1()``'s outputs (None when both ran on the calling stream)."""
    if not (do_multi_stream() and aux_stream is not None):
        return fn0(), fn1(), None
    fork_event.record()
    result0 = fn0()
    with torch.cuda.stream(aux_stream):
        fork_event.wait()
        result1 = fn1()
        join_event.record()
    if not torch.cuda.is_current_stream_capturing():
        _record_on(result1, torch.cuda.current_stream())
    return result0, result1, join_event


def fork_join(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    join_event: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any]:
    """``(fn0(), fn1())``, ``fn1`` on ``aux_stream`` where :func:`fork` forks; joined."""
    result0, result1, join = fork(fn0, fn1, fork_event, join_event, aux_stream)
    if join is not None:
        join.wait()
    return result0, result1
