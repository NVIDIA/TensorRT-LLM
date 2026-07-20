# Adapted from SGLang's breakable CUDA graph implementation.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import logging
import threading
from contextvars import ContextVar
from typing import Any, Callable, Optional

import torch
from cuda.bindings import runtime as rt

from ...utils import make_weak_ref
from .cuda_utils import check_cuda_errors

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
    status, *_ = check_cuda_errors(rt.cudaStreamGetCaptureInfo(stream_ptr))
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


def _weak_ref_if_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        return make_weak_ref(value)
    if isinstance(value, tuple):
        return tuple(_weak_ref_if_tensor(item) for item in value)
    if isinstance(value, list):
        return [_weak_ref_if_tensor(item) for item in value]
    if isinstance(value, dict):
        return {key: _weak_ref_if_tensor(item) for key, item in value.items()}
    return value


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


def eager_on_graph(enable: bool) -> Callable[[Callable], Callable]:
    """Run a callable eagerly between captured CUDA graph segments."""

    def decorator(inner: Callable) -> Callable:
        if not enable:
            return inner

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            capture = _current_capture.get()
            if capture is None:
                return inner(*args, **kwargs)

            logger.debug(
                "Break CUDA graph for function %s", getattr(inner, "__name__", type(inner).__name__)
            )
            capture._end_current_segment()
            output = inner(*args, **kwargs)

            captured_args = tuple(_weak_ref_if_tensor(arg) for arg in args)
            captured_kwargs = {key: _weak_ref_if_tensor(value) for key, value in kwargs.items()}
            captured_output = _weak_ref_if_tensor(output)

            def replay_fn() -> Any:
                new_output = inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_output)

            capture.cuda_graph._break_functions.append(replay_fn)
            capture._begin_new_segment()
            return output

        return wrapper

    return decorator


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

    def pool(self):
        if not self._segments:
            raise RuntimeError("Cannot get the pool of an empty BCG")
        return self._segments[0].pool()

    def replay(self) -> None:
        stream_token = _current_stream.set(torch.cuda.current_stream())
        replay_token = _current_replay_token.set(next(_replay_token_counter))
        try:
            for index, segment in enumerate(self._segments):
                segment.replay()
                if index < len(self._break_functions):
                    self._break_functions[index]()
        finally:
            _current_replay_token.reset(replay_token)
            _current_stream.reset(stream_token)

    def reset(self) -> None:
        for segment in self._segments:
            segment.reset()
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
        self._stream_context = None
        self._capture_token = None
        self._stream_token = None
        self._forked_token = None

    def __enter__(self) -> "BreakableCUDAGraphCapture":
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_context = torch.cuda.stream(self._stream)
            self._stream_context.__enter__()
        self._capture_token = _current_capture.set(self)
        self._stream_token = _current_stream.set(self._stream or torch.cuda.current_stream())
        self._forked_token = _forked_streams.set(set())
        self._begin_new_segment()
        return self

    def __exit__(self, *args: object) -> bool:
        try:
            self._end_current_segment()
        finally:
            _forked_streams.reset(self._forked_token)
            _current_stream.reset(self._stream_token)
            _current_capture.reset(self._capture_token)
            if self._stream_context is not None:
                self._stream_context.__exit__(*args)
                self._stream_context = None
            _uninstall_wait_stream_hook()
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


@eager_on_graph(True)
def break_graph() -> None:
    """Insert an empty eager break between CUDA graph segments."""
    return None
