# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two callables on two CUDA streams, forked and joined with events, where the engine runs multi-stream.

``do_multi_stream`` is the engine's thread-local switch: the CUDA-graph runner turns it on around the captures it
records with an auxiliary stream (eager steps and piecewise captures run with it off). The targets read it here.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream

__all__ = ["do_multi_stream", "maybe_execute_in_parallel"]


def _record_stream(obj: Any, stream: torch.cuda.Stream) -> None:
    """Record ``stream`` on every CUDA tensor in ``obj``: a tensor, or tuples, lists and dict values of them,
    nested. Other objects are not walked."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            try:
                obj.record_stream(stream)
            except RuntimeError as e:
                # The cudaMallocAsync backend asserts on memory it did not allocate (a buffer from outside PyTorch,
                # wrapped as a tensor). It never frees that memory: nothing to record.
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


def maybe_execute_in_parallel(
    fn0: Callable,
    fn1: Callable,
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream] = None,
    disable_on_compile: bool = False,
) -> tuple[Any, Any]:
    """``(fn0(), fn1())``: ``fn0`` on the current stream and ``fn1`` on ``aux_stream``, forked at ``event0`` and joined
    at ``event1``, when ``do_multi_stream()`` is on and ``aux_stream`` is given (and, with ``disable_on_compile``,
    torch.compile is not tracing); otherwise one after the other on the current stream.

    ``fn1``'s outputs are allocated in ``aux_stream`` order and read on the calling stream. Outside CUDA-graph capture
    and torch.compile tracing, the CUDA tensors ``fn1`` returns (directly or nested in tuples, lists and dict values)
    are recorded on the calling stream so their memory outlives the caller's use. A capture records nothing: a captured
    graph relies on every later use of ``aux_stream`` first waiting for the calling stream, as the next call's
    ``event0.wait()`` does.
    """
    multi_stream = (
        do_multi_stream()
        and aux_stream is not None
        and not (disable_on_compile and torch.compiler.is_compiling())
    )
    if multi_stream:
        event0.record()
        result0 = fn0()
        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
        # Without the record, freeing fn1's outputs hands their memory to later aux_stream work while the calling
        # stream's reads may still be queued. Dynamo traces neither the capture query nor record_stream.
        if not (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()):
            _record_stream(result1, torch.cuda.current_stream())
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)
