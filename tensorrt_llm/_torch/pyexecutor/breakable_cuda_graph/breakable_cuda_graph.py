# Adapted from SGLang's breakable CUDA graph implementation.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import logging
import sys
import threading
from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any, Callable, Hashable, Optional

import torch
from cuda.bindings import runtime as rt

from tensorrt_llm._utils import CUASSERT

from ...nccl_window_graph import nccl_window_graph_owner
from ...utils import make_weak_ref

logger = logging.getLogger(__name__)

__all__ = [
    "BreakableCUDAGraph",
    "BreakableCUDAGraphCapture",
    "break_graph",
    "eager_on_graph",
    "get_current_replay_token",
]

_current_capture: ContextVar[Optional["BreakableCUDAGraphCapture"]] = ContextVar(
    "breakable_cuda_graph_capture", default=None
)
_current_stream: ContextVar[Optional[torch.cuda.Stream]] = ContextVar(
    "breakable_cuda_graph_stream", default=None
)
_current_replay_token: ContextVar[Optional[int]] = ContextVar(
    "breakable_cuda_graph_replay_token", default=None
)
_forked_streams: ContextVar[Optional[set[torch.cuda.Stream]]] = ContextVar(
    "breakable_cuda_graph_forked_streams", default=None
)
_replay_token_counter = itertools.count(1)

_original_wait_stream: Optional[Callable] = None
_wait_stream_hook_lock = threading.Lock()
_wait_stream_hook_refcount = 0


def get_current_stream(device: Optional[torch.device] = None) -> torch.cuda.Stream:
    """Return the active BCG stream or PyTorch's current stream."""
    stream = _current_stream.get()
    return torch.cuda.current_stream(device) if stream is None else stream


def get_current_replay_token() -> Optional[int]:
    """Return a unique token for the active BCG replay."""
    return _current_replay_token.get()


def _capture_status(stream_ptr: int) -> rt.cudaStreamCaptureStatus:
    status, *_ = CUASSERT(rt.cudaStreamGetCaptureInfo(stream_ptr))
    return status


def _is_stream_capturing(stream: torch.cuda.Stream) -> bool:
    return (
        _capture_status(stream.cuda_stream)
        == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    )


def _hooked_wait_stream(self: torch.cuda.Stream, other: torch.cuda.Stream) -> None:
    assert _original_wait_stream is not None
    forked = _forked_streams.get()
    capturing = _current_stream.get()
    if forked is None or capturing is None:
        _original_wait_stream(self, other)
        return

    capture_ptr = capturing.cuda_stream
    self_is_capture = self is capturing or self.cuda_stream == capture_ptr
    other_is_capture = other is capturing or other.cuda_stream == capture_ptr
    if self_is_capture and not other_is_capture:
        if not _is_stream_capturing(other):
            logger.warning(
                "Dropping a wait from the breakable CUDA graph capture stream "
                "to a non-capturing stream; this dependency will not be "
                "preserved during replay.",
                stacklevel=2,
            )
            return
        _original_wait_stream(self, other)
        forked.discard(other)
    elif other_is_capture and not self_is_capture:
        _original_wait_stream(self, other)
        forked.add(self)
    else:
        _original_wait_stream(self, other)


def _install_wait_stream_hook() -> None:
    global _original_wait_stream, _wait_stream_hook_refcount
    with _wait_stream_hook_lock:
        if _wait_stream_hook_refcount == 0:
            _original_wait_stream = torch.cuda.Stream.wait_stream
            torch.cuda.Stream.wait_stream = _hooked_wait_stream
        _wait_stream_hook_refcount += 1


def _uninstall_wait_stream_hook() -> None:
    global _original_wait_stream, _wait_stream_hook_refcount
    with _wait_stream_hook_lock:
        _wait_stream_hook_refcount -= 1
        if _wait_stream_hook_refcount == 0:
            assert _original_wait_stream is not None
            torch.cuda.Stream.wait_stream = _original_wait_stream
            _original_wait_stream = None


def _copy_output(destination: Any, source: Any) -> Any:
    if torch.is_tensor(destination) and torch.is_tensor(source):
        destination.copy_(source)
        return destination

    if (
        isinstance(destination, (tuple, list))
        and isinstance(source, (tuple, list))
        and len(destination) == len(source)
    ):
        copied = [_copy_output(dst, src) for dst, src in zip(destination, source)]
        return tuple(copied) if isinstance(destination, tuple) else copied

    if hasattr(destination, "__dict__") and hasattr(source, "__dict__"):
        for key, source_value in source.__dict__.items():
            destination_value = getattr(destination, key, None)
            if torch.is_tensor(destination_value) and torch.is_tensor(source_value):
                destination_value.copy_(source_value)
            else:
                setattr(destination, key, source_value)
        return destination

    if isinstance(destination, dict) and isinstance(source, dict):
        for key, source_value in source.items():
            destination_value = destination.get(key)
            if torch.is_tensor(destination_value) and torch.is_tensor(source_value):
                destination_value.copy_(source_value)
            else:
                destination[key] = source_value
        return destination

    return source


class _GraphedBreak:
    """An eager break that replays as one CUDA graph per key of its launch sequence."""

    def __init__(
        self,
        eager_fn: Callable[[], Any],
        graph_fn: Callable[[], Any],
        graph_key: Callable[[], Optional[Hashable]],
    ) -> None:
        self._eager_fn = eager_fn
        self._graph_fn = graph_fn
        self._graph_key = graph_key
        self.graphs: dict[Hashable, torch.cuda.CUDAGraph] = {}

    def __call__(self, capture_pool=None) -> None:
        key = self._graph_key()
        graph = None if key is None else self.graphs.get(key)
        if graph is not None:
            graph.replay()
        elif key is not None and capture_pool is not None:
            # Run the key's launches once eagerly (JIT compilation, autotuning),
            # then capture them; the eager run is this replay's execution.
            self._graph_fn()
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(pool=capture_pool)
            try:
                self._graph_fn()
            finally:
                graph.capture_end()
            self.graphs[key] = graph
        else:
            self._eager_fn()

    def reset(self) -> None:
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()


def eager_on_graph(
    inner: Optional[Callable] = None,
    *,
    graph_key: Optional[Callable[..., Optional[Hashable]]] = None,
    graph_body: Optional[Callable] = None,
) -> Callable:
    """Run a callable eagerly between captured CUDA graph segments.

    With ``graph_key``, the break can replay as CUDA graphs instead. Called with
    the break's arguments at each replay, ``graph_key`` returns a key naming
    everything besides those arguments that fixes the launches of ``graph_body``
    (default: ``inner``), or None to run ``inner``. A key's graph is captured
    only by a replay given a capture pool (``BreakableCUDAGraph.replay``); other
    replays run ``inner`` for a key without one.
    """
    if inner is None:
        return functools.partial(eager_on_graph, graph_key=graph_key, graph_body=graph_body)

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):
        if torch.compiler.is_compiling():
            return inner(*args, **kwargs)

        capture = _current_capture.get()
        if capture is None:
            return inner(*args, **kwargs)

        logger.debug(
            "Break CUDA graph for function %s", getattr(inner, "__name__", type(inner).__name__)
        )
        capture._end_current_segment()
        output = inner(*args, **kwargs)

        captured_args = tuple(make_weak_ref(arg) for arg in args)
        captured_kwargs = {key: make_weak_ref(value) for key, value in kwargs.items()}
        captured_output = output

        def replay_fn(fn: Callable = inner) -> Any:
            new_output = fn(*captured_args, **captured_kwargs)
            return _copy_output(captured_output, new_output)

        if graph_key is None:
            capture.cuda_graph._break_functions.append(replay_fn)
        else:
            capture.cuda_graph._break_functions.append(
                _GraphedBreak(
                    replay_fn,
                    functools.partial(replay_fn, graph_body or inner),
                    functools.partial(graph_key, *captured_args, **captured_kwargs),
                )
            )
        capture._begin_new_segment()
        return output

    return wrapper


class BreakableCUDAGraph:
    """A sequence of CUDA graph segments separated by eager functions."""

    def __init__(self) -> None:
        self._segments: list[torch.cuda.CUDAGraph] = []
        self._break_functions: list[Callable[[], Any]] = []

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    @property
    def num_breaks(self) -> int:
        return len(self._break_functions)

    @property
    def has_graphable_breaks(self) -> bool:
        """Whether a break can replay as CUDA graphs (see ``eager_on_graph``)."""
        return any(isinstance(fn, _GraphedBreak) for fn in self._break_functions)

    @property
    def num_break_graphs(self) -> int:
        """CUDA graphs captured for the breaks (see ``eager_on_graph``)."""
        return sum(len(fn.graphs) for fn in self._break_functions if isinstance(fn, _GraphedBreak))

    def pool(self):
        if not self._segments:
            raise RuntimeError("Cannot get the pool of an empty BCG")
        return self._segments[0].pool()

    def replay(self, break_graph_pool=None) -> None:
        """Replay the segments and breaks.

        Args:
            break_graph_pool: Memory pool to capture the CUDA graph of a graphable
                break's key in when it has none; None runs such breaks eagerly.
                Capturing needs a non-default current stream.
        """
        stream_token = _current_stream.set(torch.cuda.current_stream())
        replay_token = _current_replay_token.set(next(_replay_token_counter))
        try:
            for index, segment in enumerate(self._segments):
                segment.replay()
                if index < len(self._break_functions):
                    fn = self._break_functions[index]
                    if isinstance(fn, _GraphedBreak):
                        fn(break_graph_pool)
                    else:
                        fn()
        finally:
            _current_replay_token.reset(replay_token)
            _current_stream.reset(stream_token)

    def reset(self) -> None:
        for segment in self._segments:
            segment.reset()
        for fn in self._break_functions:
            if isinstance(fn, _GraphedBreak):
                fn.reset()
        self._segments.clear()
        self._break_functions.clear()


class BreakableCUDAGraphCapture:
    """Capture a region as CUDA graph segments separated by eager work."""

    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool=None,
        stream: Optional[torch.cuda.Stream] = None,
        capture_error_mode: str = "global",
    ) -> None:
        if not isinstance(cuda_graph, BreakableCUDAGraph):
            raise TypeError("cuda_graph must be a BreakableCUDAGraph")
        self.cuda_graph = cuda_graph
        self._pool = (0, 0) if pool is None else pool
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._exit_stack: Optional[ExitStack] = None

    def __enter__(self) -> "BreakableCUDAGraphCapture":
        stack = ExitStack()
        self._exit_stack = stack
        try:
            _install_wait_stream_hook()
            stack.callback(_uninstall_wait_stream_hook)
            stack.enter_context(nccl_window_graph_owner(self._pool))
            if self._stream is not None:
                stack.enter_context(torch.cuda.stream(self._stream))
            capture_token = _current_capture.set(self)
            stack.callback(_current_capture.reset, capture_token)
            stream_token = _current_stream.set(self._stream or torch.cuda.current_stream())
            stack.callback(_current_stream.reset, stream_token)
            forked_token = _forked_streams.set(set())
            stack.callback(_forked_streams.reset, forked_token)
            self._begin_new_segment()
        except BaseException:
            self._exit_stack = None
            stack.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *args: object) -> bool:
        try:
            self._end_current_segment()
        except BaseException:
            args = sys.exc_info()
            raise
        finally:
            stack = self._exit_stack
            self._exit_stack = None
            if stack is not None:
                stack.__exit__(*args)
        return False

    def _begin_new_segment(self) -> None:
        segment = torch.cuda.CUDAGraph()
        segment.capture_begin(pool=self._pool, capture_error_mode=self._capture_error_mode)
        self.cuda_graph._segments.append(segment)

    def _end_current_segment(self) -> None:
        main_stream = get_current_stream()
        forked = _forked_streams.get()
        if forked:
            assert _original_wait_stream is not None
            for side_stream in list(forked):
                if _is_stream_capturing(side_stream):
                    _original_wait_stream(main_stream, side_stream)
            forked.clear()
        self.cuda_graph._segments[-1].capture_end()


@eager_on_graph
def break_graph() -> None:
    """Insert an empty eager break between CUDA graph segments."""
    return None
