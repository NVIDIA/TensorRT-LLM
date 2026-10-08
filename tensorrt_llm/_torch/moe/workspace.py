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
    def begin_workspace_forward(self, owner: int, warmup: bool) -> bool: ...

    def finish_workspace_forward(self, completed: bool) -> None: ...


_owner_ids = count(1)
_current_scope: ContextVar["CutlassWorkspaceReclaimer | None"] = ContextVar(
    "cutlass_workspace_scope", default=None
)


class CutlassWorkspaceReclaimer:
    """Group shared runners' layer/chunk calls into one model forward.

    Native owners accumulate exact workspace sizing; only runners observed
    during warmup acquire a floor. Captured storage and shared-engine owners
    opt out in native code, where their allocation lifetimes are known.
    """

    def __init__(self) -> None:
        self._owner = next(_owner_ids)
        self._runners: dict[
            tuple[int, torch.device, int], tuple[_WorkspaceRunner | None, torch.cuda.Stream]
        ] = {}
        self._warmup = False
        self._active = False

    def register(self, runner: _WorkspaceRunner) -> None:
        stream = torch.cuda.current_stream()
        key = (id(runner), stream.device, stream.cuda_stream)
        if key not in self._runners:
            accepted = runner.begin_workspace_forward(self._owner, self._warmup)
            self._runners[key] = (runner if accepted else None, stream)

    @staticmethod
    def _finish(runner: _WorkspaceRunner, stream: torch.cuda.Stream, completed: bool) -> None:
        with torch.cuda.stream(stream):
            runner.finish_workspace_forward(completed)

    @contextmanager
    def forward(self, *, warmup: bool) -> Iterator[None]:
        if self._active or _current_scope.get() is not None:
            raise RuntimeError("Overlapping CUTLASS workspace scopes are unsupported")
        if do_multi_stream() or torch.cuda.is_current_stream_capturing():
            yield
            return
        self._warmup = warmup
        self._runners = {}
        self._active = True
        token = _current_scope.set(self)
        completed = False
        try:
            yield
            completed = True
        finally:
            _current_scope.reset(token)
            self._active = False
            runners, self._runners = self._runners, {}
            with ExitStack() as stack:
                for runner, stream in runners.values():
                    if runner is not None:
                        stack.callback(self._finish, runner, stream, completed)


def register_cutlass_workspace(runner: _WorkspaceRunner) -> None:
    """Register the native owner immediately before a CUTLASS kernel call."""
    scope = _current_scope.get()
    if scope is not None:
        scope.register(runner)
