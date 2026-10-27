# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-loop behaviour a model may opt into, read through ``runtime_hooks_of``.

A model declares its hooks as ``runtime_hooks`` (an instance attribute, a class
attribute or a property). Every field defaults to off, and a model that declares
nothing gets ``RuntimeHooks()``: the step loop then runs exactly as it does
without this module. Every engine call site
of a hook runs its original statements when the hook is off.

The hooks change *where* and *how* the step loop moves data, never what a
forward computes. The protocols below state only the methods an engine call
site needs; the objects that implement them are owned by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


class DecodePrologue(Protocol):
    """Replaces the decode step's input prologue with one launch.

    Called first in the step's input preprocessing, inside CUDA graph capture
    as well. It decides on the host before any device work; returning False
    keeps the native prologue, and returning True means every engine-visible
    tensor the native prologue writes holds the value it would have written.
    """

    def __call__(
        self, preparer: Any, inputs: Any, *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> bool:
        """``preparer`` is the ``DecoderInputPreparer`` running the step's preprocessing."""


class ExecutorEvents(Protocol):
    """Points in the executor's life a model may act at."""

    def after_warmup(self) -> None:
        """After warmup and CUDA graph capture, before the first request is served."""

    def before_overlap_loop(self) -> None:
        """Right before the overlap executor loop starts, on every rank; must not raise."""

    def at_shutdown(self) -> None:
        """When the executor loop has completed, during cleanup; must not raise."""


@dataclass(frozen=True)
class RuntimeHooks:
    """The step-loop hooks a model opts into. All off by default."""

    # Run the overlap executor loop with the execution stream current, so the
    # loop's own stream work and the forward are ordered on one stream.
    loop_on_execution_stream: bool = False
    # The decode step's input prologue in one launch (see DecodePrologue).
    decode_prologue: Optional[DecodePrologue] = None
    # Actions after warmup, before the overlap loop and at shutdown (see ExecutorEvents).
    executor_events: Optional[ExecutorEvents] = None


DEFAULT_RUNTIME_HOOKS = RuntimeHooks()


def runtime_hooks_of(model: object) -> RuntimeHooks:
    """The hooks ``model`` declares, or the all-off default when it declares none.

    Declaring hooks is optional for every model class, so absence is the common
    case and means the default.
    """
    hooks = getattr(model, "runtime_hooks", None)
    if hooks is None:
        return DEFAULT_RUNTIME_HOOKS
    if not isinstance(hooks, RuntimeHooks):
        raise TypeError(
            f"{type(model).__name__}.runtime_hooks must be a RuntimeHooks, got {type(hooks).__name__}"
        )
    return hooks
