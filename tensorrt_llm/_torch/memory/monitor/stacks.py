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
With ``memory_monitor_config.record_stacks`` (or
``TLLM_MEM_MONITOR_STACKS=1``) in full mode, the allocator keeps, for every live
block, the Python stack that allocated it, and that part of the report is
broken down by allocating code. Memory a C++ op allocates through
``torch::empty`` is named by the Python call site of the op.

Off by default because it slows every PyTorch allocation. It records
allocator *state*, not an event history: each live block keeps its stack
and nothing else is kept, so there is no event ring to grow and a snapshot
stays cheap. Switching to it first stops the startup recording, which
discards that recording's event ring: PyTorch keeps a ring it is no longer
sized for otherwise, and writes past its end once it has wrapped.

Recording runs until :func:`stop_stack_recording`, which executor shutdown
calls, so the slowdown does not outlive the executor that asked for it.

Only allocations made after recording starts carry a stack, so turn it on
before reproducing the problem; it cannot be applied to an OOM after the
fact. It covers the PyTorch allocator only; memory a library obtains from
CUDA directly is outside its reach.
"""

from __future__ import annotations

import re

import torch

from tensorrt_llm.logger import logger

from .capability import tagging_supported
from .settings import STACKS_ENV_VAR, full_mode, stacks_requested

__all__ = [
    "STACKS_ENV_VAR",
    "site_of",
    "start_stack_recording",
    "stacks_enabled",
    "stop_stack_recording",
]

#: Frames from these paths say how memory was allocated, not who wanted it.
_SKIP = ("/torch/", "/memory/monitor/", "<frozen", "/contextlib.py", "/functools.py")
_PACKAGE = "/tensorrt_llm/"
#: Paths that hold installed libraries and the standard library.
_LIBRARY_PATHS = ("/site-packages/", "/dist-packages/", "/lib/python3")


def stacks_enabled() -> bool:
    """Whether stack recording was asked for and can work here."""
    return stacks_requested() and full_mode() and tagging_supported()


#: Whether this module started the recording in progress.
_recording = False


def start_stack_recording() -> bool:
    """Keep each live block's Python allocation stack from now on.

    Takes over from the startup tagging window once its map is frozen. The
    window's recording is stopped first, which discards its event ring.
    """
    global _recording
    try:
        torch.cuda.memory._record_memory_history(None)
        torch.cuda.memory._record_memory_history(enabled="state", context="alloc", stacks="python")
    except RuntimeError as exc:
        logger.warning(f"memory monitor: could not start stack recording ({exc})")
        return False
    _recording = True
    logger.info(
        "memory monitor: recording allocation stacks (record_stacks); "
        "PyTorch allocations are slower while this is on"
    )
    return True


def stop_stack_recording() -> None:
    """Stop stack recording if it was started here; otherwise do nothing."""
    global _recording
    if not _recording:
        return
    try:
        torch.cuda.memory._record_memory_history(None)
    except RuntimeError as exc:
        logger.warning(f"memory monitor: could not stop stack recording ({exc})")
        return
    _recording = False


def _is_library(path: str) -> bool:
    """Installed third-party code or the standard library, not ours or the user's."""
    if _PACKAGE in path:
        return False
    return any(part in path for part in _LIBRARY_PATHS)


def site_of(frames: list[dict]) -> str | None:
    """A short name for the code that made an allocation.

    The innermost frame that is either TensorRT-LLM or the user's own code
    -- a logits processor, say -- wins: that is the code a reader can change.
    Only when there is neither does a third-party library frame name the
    allocation. Given as ``path:function``, the path trimmed to something
    readable.
    """
    if not frames:
        return None
    outside = [f for f in frames if not any(s in f.get("filename", "") for s in _SKIP)]
    own = [f for f in outside if not _is_library(f.get("filename", ""))]
    frame = (own or outside or [None])[0]
    if frame is None:
        return None
    path = frame.get("filename", "?")
    if _PACKAGE in path:
        path = path.rsplit(_PACKAGE, 1)[1]
        path = path[len("_torch/") :] if path.startswith("_torch/") else path
    elif any(part in path for part in ("/site-packages/", "/dist-packages/")):
        path = re.split(r"/(?:site|dist)-packages/", path)[-1]
    else:
        path = "/".join(path.split("/")[-2:])
    return f"{path}:{frame.get('name', '?')}"
