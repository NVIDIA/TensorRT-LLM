from typing import Any, Callable, Optional

import torch

from ..pyexecutor.cuda_graph_runner import is_graph_capturing


def maybe_execute_in_parallel(
        fn0: Callable,
        fn1: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream] = None) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    do_multi_stream = is_graph_capturing() and aux_stream is not None
    print(f"[DEBUG] maybe_execute_in_parallel - do_multi_stream: {do_multi_stream}")

    if do_multi_stream:
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
