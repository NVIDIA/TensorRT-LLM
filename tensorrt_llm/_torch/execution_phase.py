# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process-wide execution phase of the PyTorch backend.

The executor publishes whether the process is in its startup warmup or serving. Components whose
behavior depends on the phase subscribe to transitions, so the model engine does not need to know
about them.
"""

import contextlib
import enum
import threading
from collections.abc import Callable, Iterator

import torch

__all__ = [
    "ExecutionPhase",
    "ExecutionPhaseListener",
    "add_execution_phase_listener",
    "execution_phase",
    "get_execution_phase",
    "set_execution_phase",
]


class ExecutionPhase(enum.Enum):
    """Lifecycle phase shared by every model engine in the process."""

    WARMUP = "warmup"
    """One-time per-rank work (JIT compilation, autotuning, module loading) can skew ranks."""

    SERVING = "serving"
    """Serving, including the capture of the CUDA graphs that serving replays."""


ExecutionPhaseListener = Callable[[ExecutionPhase], None]

# Held across a transition and its notifications, so a listener added concurrently either sees
# the transition or is added after it; it never misses one.
_lock = threading.RLock()
_current_phase = ExecutionPhase.SERVING
_listeners: list[ExecutionPhaseListener] = []


def get_execution_phase() -> ExecutionPhase:
    """Return the phase most recently published by the executor."""
    return _current_phase


def set_execution_phase(phase: ExecutionPhase) -> None:
    """Publish ``phase`` and notify listeners if it differs from the current phase.

    Args:
        phase: The phase the process enters.

    Raises:
        TypeError: If ``phase`` is not an ``ExecutionPhase``.
        RuntimeError: If the current CUDA stream is capturing a graph. Listeners may issue
            host-synchronizing CUDA calls, which are illegal during capture.
    """
    global _current_phase
    if not isinstance(phase, ExecutionPhase):
        raise TypeError(f"phase must be an ExecutionPhase, got {type(phase).__name__}")
    if _is_capturing_cuda_graph():
        raise RuntimeError(
            f"Cannot enter execution phase {phase.value!r} while a CUDA graph is being captured"
        )
    with _lock:
        if phase is _current_phase:
            return
        _current_phase = phase
        for listener in list(_listeners):
            listener(phase)


@contextlib.contextmanager
def execution_phase(phase: ExecutionPhase) -> Iterator[None]:
    """Run the enclosed block in ``phase``, then restore the phase that was active on entry.

    Args:
        phase: The phase for the enclosed block.
    """
    previous = get_execution_phase()
    set_execution_phase(phase)
    try:
        yield
    finally:
        set_execution_phase(previous)


def add_execution_phase_listener(listener: ExecutionPhaseListener) -> None:
    """Subscribe ``listener`` to subsequent phase transitions; adding it again has no effect.

    Args:
        listener: Called with the new phase after each transition.
    """
    with _lock:
        if listener not in _listeners:
            _listeners.append(listener)


def _is_capturing_cuda_graph() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
