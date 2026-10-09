# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-forward scopes for native CUTLASS scratch reclamation."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from itertools import count
from typing import Iterator, Protocol

import torch

from ..modules.multi_stream_utils import do_multi_stream


class _WorkspaceRunner(Protocol):
    def begin_workspace_forward(self, owner: int, warmup: bool, num_tokens: int) -> bool: ...

    def finish_workspace_forward(self, completed: bool) -> None: ...


class _WorkspaceMethods:
    """Keep the native owner alive and resolve its TorchBind methods once."""

    def __init__(self, runner: _WorkspaceRunner) -> None:
        self._runner = runner
        self.begin_workspace_forward = runner.begin_workspace_forward
        self.finish_workspace_forward = runner.finish_workspace_forward


_owner_ids = count(1)
_current_scope: ContextVar["CutlassWorkspaceReclaimer | None"] = ContextVar(
    "cutlass_workspace_scope", default=None
)


class CutlassWorkspaceReclaimer:
    """Group shared runners' layer/chunk calls into one model forward.

    Native owners accumulate exact workspace sizing; only runners observed
    during warmup become eligible. Captured storage and shared-engine owners
    opt out in native code, where their allocation lifetimes are known.
    """

    def __init__(self) -> None:
        self._owner = next(_owner_ids)
        self._runners: dict[
            tuple[int, torch.device, int], tuple[_WorkspaceRunner | None, torch.cuda.Stream]
        ] = {}
        self._cached_runners: dict[
            tuple[int, torch.device, int], tuple[_WorkspaceMethods, torch.cuda.Stream]
        ] = {}
        self._last_runner: _WorkspaceRunner | None = None
        self._last_device_index: int | None = None
        self._last_stream_id = 0
        self._num_tokens = 0
        self._warmup = False
        self._active = False

    def register(self, runner: _WorkspaceRunner, device: torch.device) -> None:
        # Query the raw stream on the input's explicit device. Constructing a
        # Stream and resolving the current device on every layer is expensive.
        device_index = device.index
        stream_id = torch._C._cuda_getCurrentRawStream(device_index)
        # Consecutive layers usually share a runner. Still check the stream,
        # but avoid constructing and hashing the same key for every layer.
        if (
            runner is self._last_runner
            and device_index == self._last_device_index
            and stream_id == self._last_stream_id
        ):
            return
        key = (id(runner), device, stream_id)
        if key not in self._runners:
            cached = self._cached_runners.get(key)
            if cached is None:
                cached = (_WorkspaceMethods(runner), torch.cuda.current_stream(device))
                self._cached_runners[key] = cached
            methods, stream = cached
            accepted = methods.begin_workspace_forward(self._owner, self._warmup, self._num_tokens)
            self._runners[key] = (methods if accepted else None, stream)
        self._last_runner = runner
        self._last_device_index = device_index
        self._last_stream_id = stream_id

    @staticmethod
    def _finish(runner: _WorkspaceRunner, stream: torch.cuda.Stream, completed: bool) -> None:
        device = torch.cuda.current_device()
        if (
            device == stream.device_index
            and torch._C._cuda_getCurrentRawStream(device) == stream.cuda_stream
        ):
            runner.finish_workspace_forward(completed)
        else:
            # A forward may leave another device/stream current. Cleanup must
            # still release and replace scratch on its allocation stream.
            with torch.cuda.stream(stream):
                runner.finish_workspace_forward(completed)

    @contextmanager
    def forward(self, *, warmup: bool, num_tokens: int = 0) -> Iterator[None]:
        if self._active or _current_scope.get() is not None:
            raise RuntimeError("Overlapping CUTLASS workspace scopes are unsupported")
        if do_multi_stream() or torch.cuda.is_current_stream_capturing():
            yield
            return
        self._num_tokens = num_tokens
        self._warmup = warmup
        self._runners = {}
        self._last_runner = None
        self._active = True
        token = _current_scope.set(self)
        completed = False
        try:
            if num_tokens > 0 and self._cached_runners:
                device = torch.cuda.current_device()
                stream_id = torch._C._cuda_getCurrentRawStream(device)
                for key, (methods, stream) in self._cached_runners.items():
                    if key[1].index == device and key[2] == stream_id:
                        accepted = methods.begin_workspace_forward(self._owner, warmup, num_tokens)
                        self._runners[key] = (methods if accepted else None, stream)
            yield
            completed = True
        finally:
            _current_scope.reset(token)
            self._active = False
            self._last_runner = None
            runners, self._runners = self._runners, {}
            if len(runners) == 1:
                runner, stream = next(iter(runners.values()))
                if runner is not None:
                    self._finish(runner, stream, completed)
            else:
                # Finish every owner even when another owner's cleanup fails.
                with ExitStack() as stack:
                    for runner, stream in runners.values():
                        if runner is not None:
                            stack.callback(self._finish, runner, stream, completed)


def register_cutlass_workspace(runner: _WorkspaceRunner, device: torch.device) -> None:
    """Register the native owner immediately before a CUTLASS kernel call."""
    scope = _current_scope.get()
    if scope is not None:
        scope.register(runner, device)
