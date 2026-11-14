# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two callables on two CUDA streams, when the engine runs multi-stream.

:func:`do_multi_stream` is the engine's flag: the CUDA graph runner sets it for its warmup and
capture forwards (``with_multi_stream(True)``) and it is off everywhere else, so eager steps (the
context servers, which run without CUDA graphs) keep every side-stream branch on the calling
stream. This module reads that one flag; it does not keep a second one.

:func:`maybe_execute_in_parallel` runs ``fn0`` on the calling stream and ``fn1`` on ``aux_stream``
when the flag is on and an aux stream is given, else both in order on the calling stream:

* calling stream: ``event0.record()`` -> ``fn0()``;
* ``aux_stream``: ``event0.wait()`` -> ``fn1()`` -> ``event1.record()``;
* calling stream: ``event1.wait()``.

Outside graph capture and ``torch.compile`` tracing, the CUDA tensors ``fn1`` returns (directly
or in nested tuples, lists and dict values) are recorded on the calling stream, so their memory
outlives the caller's use. During capture nothing is recorded: a captured graph relies on every
later use of ``aux_stream`` first waiting for the calling stream, as the next call's
``event0.wait()`` does.
"""

from typing import Any, Callable, Optional

import torch

from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream

__all__ = [
    "do_multi_stream",
    "maybe_execute_in_parallel",
    "maybe_execute_in_parallel_deferred_join",
]


def _record_stream(obj: Any, stream: torch.cuda.Stream) -> None:
    """Record ``stream`` on every CUDA tensor in ``obj``: a tensor, or tuples, lists and dict
    values of them, nested. Other objects are not walked."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            try:
                obj.record_stream(stream)
            except RuntimeError as e:
                # The cudaMallocAsync backend asserts on memory it did not allocate (a buffer
                # from outside PyTorch, wrapped as a tensor). It never frees that memory: nothing
                # to record. The text is the TORCH_INTERNAL_ASSERT in
                # CudaMallocAsyncAllocator::recordStream.
                if (
                    torch.cuda.memory.get_allocator_backend() != "cudaMallocAsync"
                    or "ptr not found in ptr_info" not in str(e)
                ):
                    raise
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _record_stream(item, stream)
    elif isinstance(obj, dict):
        for item in obj.values():
            _record_stream(item, stream)


def maybe_execute_in_parallel_deferred_join(
    fn0: Callable,
    fn1: Callable,
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream] = None,
    disable_on_compile: bool = False,
) -> tuple[Any, Any, Optional[torch.cuda.Event]]:
    """:func:`maybe_execute_in_parallel` without the final join.

    Returns ``fn0()``, ``fn1()`` and the event the calling stream must wait on before it reads
    ``fn1()``'s outputs (``None`` when both ran on the calling stream).
    """
    multi_stream = (
        do_multi_stream()
        and aux_stream is not None
        and not (disable_on_compile and torch.compiler.is_compiling())
    )
    if not multi_stream:
        return fn0(), fn1(), None

    event0.record()
    result0 = fn0()

    with torch.cuda.stream(aux_stream):
        event0.wait()
        result1 = fn1()
        event1.record()
    if not (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()):
        _record_stream(result1, torch.cuda.current_stream())
    return result0, result1, event1


def maybe_execute_in_parallel(
    fn0: Callable,
    fn1: Callable,
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream] = None,
    disable_on_compile: bool = False,
) -> tuple[Any, Any]:
    """``(fn0(), fn1())``, ``fn1`` on ``aux_stream`` when multi-stream is on (see the module).

    ``disable_on_compile``: keep both on the calling stream while ``torch.compile`` traces.
    """
    result0, result1, event1 = maybe_execute_in_parallel_deferred_join(
        fn0, fn1, event0, event1, aux_stream, disable_on_compile
    )
    if event1 is not None:
        event1.wait()
    return result0, result1
