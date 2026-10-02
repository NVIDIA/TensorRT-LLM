import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch


class do_multi_stream_local(threading.local):

    def __init__(self):
        self.do_multi_stream = False


_local = do_multi_stream_local()


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

    fn1()'s outputs are allocated in aux_stream order and used on the calling
    stream. Outside CUDA graph capture and torch.compile tracing, the CUDA
    tensors fn1() returns (directly, or in nested tuples, lists and dict
    values; not tensors held by other objects) are recorded on the calling
    stream, so their memory outlives the caller's use. During capture nothing
    is recorded: a captured graph relies on every later use of aux_stream
    first waiting for the calling stream, as the next call's event0.wait()
    does. Other work captured on aux_stream must do the same.

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

    multi_stream = (do_multi_stream() and aux_stream is not None and
                    not (disable_on_compile and torch.compiler.is_compiling()))

    if multi_stream:
        event0.record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
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
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)
