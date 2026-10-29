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
"""Naming PyTorch memory by the code that allocated it.

Tags and claims cover what exists when startup ends. What an op allocates
while serving -- activations, scratch buffers, a library's lazily created
workspace -- carries no tag and is reported as "PyTorch, unattributed".
With ``TLLM_MEM_MONITOR_STACKS=1`` the allocator keeps, for every live
block, the Python stack that allocated it, and that part of the report is
broken down by allocating code. Memory a C++ op allocates through
``torch::empty`` is named by the Python call site of the op.

Off by default because it is not free. It records allocator *state*, not an
event history: each live block keeps its stack and nothing else is kept, so
there is no ring buffer to wrap and a snapshot stays cheap. Measured on
torch 2.14 with 20000 small allocations: 41.1 ms against 32.5 ms
unrecorded (+26%), and a snapshot in 6 ms -- against 47.7 ms and 2279 ms
when the full event history is kept as well.

Only allocations made after recording starts carry a stack, so turn it on
before reproducing the problem; it cannot be applied to an OOM after the
fact. It covers the PyTorch allocator only; memory a library obtains from
CUDA directly is outside its reach.
"""

from __future__ import annotations

import os

import torch

from tensorrt_llm.logger import logger

__all__ = ["ENV_VAR", "site_of", "start_stack_recording", "stacks_enabled"]

ENV_VAR = "TLLM_MEM_MONITOR_STACKS"

#: Frames from these paths say how memory was allocated, not who wanted it.
_SKIP = ("/torch/", "/memory/monitor/", "<frozen", "/contextlib.py", "/functools.py")
_PACKAGE = "/tensorrt_llm/"


def stacks_enabled() -> bool:
    """Whether stack recording was asked for and can work here."""
    from . import enabled
    from .tagging import tagging_supported

    return os.environ.get(ENV_VAR, "") == "1" and enabled() and tagging_supported()


def start_stack_recording() -> bool:
    """Keep each live block's Python allocation stack from now on.

    Replaces any history recording in progress -- the startup tagging window
    hands over to this once its map is frozen.
    """
    try:
        torch.cuda.memory._record_memory_history(enabled="state", context="alloc", stacks="python")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not break serving
        logger.warning(
            f"memory monitor: could not start stack recording ({type(exc).__name__}: {exc})"
        )
        return False
    logger.info(
        "memory monitor: recording allocation stacks (TLLM_MEM_MONITOR_STACKS=1); "
        "PyTorch allocations are slower while this is on"
    )
    return True


def site_of(frames) -> str | None:
    """A short name for the code that made an allocation.

    The innermost TensorRT-LLM frame when there is one, otherwise the
    innermost frame outside PyTorch itself (a third-party library), as
    ``path:function`` with the path trimmed to something readable.
    """
    if not frames:
        return None
    outside = [f for f in frames if not any(s in f.get("filename", "") for s in _SKIP)]
    ours = [f for f in outside if _PACKAGE in f.get("filename", "")]
    frame = (ours or outside or [None])[0]
    if frame is None:
        return None
    path = frame.get("filename", "?")
    if _PACKAGE in path:
        path = path.rsplit(_PACKAGE, 1)[1]
        path = path[len("_torch/") :] if path.startswith("_torch/") else path
    elif "/site-packages/" in path:
        path = path.rsplit("/site-packages/", 1)[1]
    return f"{path}:{frame.get('name', '?')}"
