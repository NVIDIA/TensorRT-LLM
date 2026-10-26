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
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    multi_stream = (do_multi_stream() and aux_stream is not None and
                    not (disable_on_compile and torch.compiler.is_compiling()))
    return _run_in_parallel(fn0, fn1, event0, event1, aux_stream, multi_stream)


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

    Also forks when fork_in_eager is set, unless tracing, capturing or within
    no_eager_forks(). combine is the last use of fn1's results and must not
    return or alias them: after it, aux_stream waits for the calling stream,
    so their memory is not reused or released early.

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
        multi_stream = do_multi_stream()
        eager = fork_in_eager and _local.eager_forks
        capturing = ((multi_stream or eager)
                     and torch.cuda.is_current_stream_capturing())
        fork = multi_stream or (eager and not capturing)
        order = fork and not capturing

    result0, result1 = _run_in_parallel(fn0, fn1, event0, event1, aux_stream,
                                        fork)
    result = combine(result0, result1)
    if order:
        event0.record()
        event0.wait(aux_stream)
    return result
