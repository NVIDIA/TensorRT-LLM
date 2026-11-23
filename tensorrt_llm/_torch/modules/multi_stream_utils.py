# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm._utils import CUASSERT

_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")
_R = TypeVar("_R")


class _MultiStreamLocal(threading.local):

    def __init__(self) -> None:
        self.do_multi_stream = False
        self.eager_forks = True
        self.forks_in_capture = False


_local = _MultiStreamLocal()


def set_do_multi_stream(enable: bool):
    _local.do_multi_stream = enable


def do_multi_stream() -> bool:
    return _local.do_multi_stream


@contextmanager
def with_multi_stream(enable: bool):
    prev_do_multi_stream = _local.do_multi_stream
    set_do_multi_stream(enable)
    try:
        yield
    finally:
        set_do_multi_stream(prev_do_multi_stream)


def _record_stream(obj: Any, stream: torch.cuda.Stream) -> None:
    """Record ``stream`` on every CUDA tensor in ``obj``: a tensor, or tuples,
    lists and dict values of them, nested. Other objects are not walked."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            try:
                obj.record_stream(stream)
            except RuntimeError as e:
                # The cudaMallocAsync backend asserts on memory it did not
                # allocate (a buffer from outside PyTorch, wrapped as a
                # tensor). It never frees that memory: nothing to record.
                # The text is the TORCH_INTERNAL_ASSERT in
                # CudaMallocAsyncAllocator::recordStream
                # (c10/cuda/CUDAMallocAsyncAllocator.cpp, torch
                # 2.14.0a0+4fdf77b940.nv26.08); re-check it on torch upgrades.
                if (torch.cuda.memory.get_allocator_backend()
                        != "cudaMallocAsync"
                        or "ptr not found in ptr_info" not in str(e)):
                    raise
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _record_stream(item, stream)
    elif isinstance(obj, dict):
        for item in obj.values():
            _record_stream(item, stream)


@functools.cache
def private_stream(device: int, purpose: str) -> torch.cuda.ExternalStream:
    """A non-blocking stream per ``purpose``: a ``torch.cuda.Stream`` comes from a
    pool and can alias the execution stream."""
    with torch.cuda.device(device):
        (stream, ) = CUASSERT(
            cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    return torch.cuda.ExternalStream(int(stream), device=device)


@contextmanager
def no_eager_forks() -> Iterator[None]:
    """Disable the eager forks of maybe_execute_in_parallel_and_combine().

    Tuning contexts enter it, so that no op is profiled on the aux stream next
    to the other branch, with its buffers left in that stream's pool.
    """
    prev_eager_forks = _local.eager_forks
    _local.eager_forks = False
    try:
        yield
    finally:
        _local.eager_forks = prev_eager_forks


def forks_now(fork_in_eager: bool) -> bool:
    """Whether maybe_execute_in_parallel_and_combine() forks onto its aux stream
    here (not while tracing)."""
    multi_stream = do_multi_stream()
    eager = fork_in_eager and _local.eager_forks
    capturing = ((multi_stream or eager)
                 and torch.cuda.is_current_stream_capturing())
    return multi_stream or (eager and
                            (not capturing or _local.forks_in_capture))


@contextmanager
def eager_forks_in_capture() -> Iterator[None]:
    """Keep the eager forks of maybe_execute_in_parallel_and_combine() in CUDA graph captures,
    as a fork / join inside the graph, for captures that replace eager forwards."""
    prev_forks_in_capture = _local.forks_in_capture
    _local.forks_in_capture = True
    try:
        yield
    finally:
        _local.forks_in_capture = prev_forks_in_capture


def _run_in_parallel(fn0: Callable[[], _T0], fn1: Callable[[], _T1],
                     event0: torch.cuda.Event, event1: torch.cuda.Event,
                     aux_stream: torch.cuda.Stream | None,
                     fork: bool) -> tuple[_T0, _T1]:
    if fork:
        event0.record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)


def maybe_execute_in_parallel(
        fn0: Callable,
        fn1: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream] = None,
        disable_on_compile: bool = False) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.
    aux_stream does not wait for the later use of fn1's results; use
    maybe_execute_in_parallel_and_combine() when it must.

    fn1()'s outputs are allocated in aux_stream order and used on the calling
    stream. Outside CUDA graph capture and torch.compile tracing, the CUDA
    tensors fn1() returns (directly, or in nested tuples, lists and dict
    values; not tensors held by other objects) are recorded on the calling
    stream, so their memory outlives the caller's use. During capture nothing
    is recorded: a captured graph relies on every later use of aux_stream
    first waiting for the calling stream, as the next call's event0.wait()
    does. Other work captured on aux_stream must do the same. Callers should
    not record fn1()'s outputs themselves.

    A caller that torch.compile traces with disable_on_compile=False runs
    fn1() on aux_stream without this record, outside CUDA graphs too, so it
    must order the reuse of fn1()'s outputs itself.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.
        disable_on_compile (bool): if True, disable multi-stream when
            torch.compile is tracing. Callers that are not inside a custom op
            should set this to True so that stream/event ops are not captured
            by dynamo. Callers inside custom ops (e.g. attention, MoE) should
            leave this as False since custom ops are opaque to the compiler.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1().
    """

    result0, result1, event1 = maybe_execute_in_parallel_deferred_join(
        fn0, fn1, event0, event1, aux_stream, disable_on_compile)
    if event1 is not None:
        event1.wait()
    return (result0, result1)


def maybe_execute_in_parallel_deferred_join(
    fn0: Callable,
    fn1: Callable,
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream] = None,
    disable_on_compile: bool = False
) -> tuple[Any, Any, Optional[torch.cuda.Event]]:
    """maybe_execute_in_parallel without the final join.

    Returns:
        fn0() and fn1(), and the event the calling stream must wait on before
        it reads fn1()'s outputs (None when both ran on the calling stream).
    """
    multi_stream = (do_multi_stream() and aux_stream is not None and
                    not (disable_on_compile and torch.compiler.is_compiling()))
    if not multi_stream:
        return fn0(), fn1(), None

    event0.record()
    result0 = fn0()

    with torch.cuda.stream(aux_stream):
        event0.wait()
        result1 = fn1()
        event1.record()
    # Without the record, freeing fn1's outputs hands their memory to later
    # aux_stream work (caching allocator) or frees it in aux_stream order
    # (cudaMallocAsync) while the calling stream's reads may still be
    # queued. Under graph capture the caching allocator would hold every
    # recorded block freed in the capture until it ends, and under
    # cudaMallocAsync freeing a recorded output inside the capture fails it
    # ("capturing stream has unjoined work"). Dynamo traces neither the
    # capture query nor record_stream, so compiled callers skip both.
    if not (torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()):
        _record_stream(result1, torch.cuda.current_stream())
    return result0, result1, event1


def maybe_execute_in_parallel_and_combine(
        fn0: Callable[[], _T0],
        fn1: Callable[[], _T1],
        combine: Callable[[_T0, _T1], _R],
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: torch.cuda.Stream | None = None,
        *,
        fork_in_eager: bool | torch.SymBool = False) -> _R:
    """Run fn0 and fn1 like maybe_execute_in_parallel(), then return
    combine(result0, result1).

    Also forks when fork_in_eager is set, unless tracing, capturing (outside
    eager_forks_in_capture()) or within no_eager_forks(). combine is the last
    use of fn1's results and must not return or alias them: after it,
    aux_stream waits for the calling stream, so their memory is not reused or
    released early.

    Args:
        fn0, fn1, event0, event1, aux_stream: as in maybe_execute_in_parallel().
        combine (Callable): called on the calling stream with both results.
        fork_in_eager (bool | torch.SymBool): also fork in eager mode.
    """

    if aux_stream is None:
        fork = order = False
    elif torch.compiler.is_compiling():
        # fork_in_eager may be symbolic while tracing; don't evaluate it.
        fork, order = do_multi_stream(), False
    else:
        fork = forks_now(fork_in_eager)
        order = fork and not torch.cuda.is_current_stream_capturing()

    result0, result1 = _run_in_parallel(fn0, fn1, event0, event1, aux_stream,
                                        fork)
    result = combine(result0, result1)
    if order:
        event0.record()
        event0.wait(aux_stream)
    return result
