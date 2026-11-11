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
"""Opt-in reset of the driver's graph-launch scheduling state after warmup.

A process can end up in a launch-scheduling mode in which every dependent kernel node of a
CUDA-graph replay costs more than in a fresh process. Creating and destroying one transient
driver context in the process ends that mode. With ``TRTLLM_GRAPH_LAUNCH_RESET=1`` each
worker does that once, after warmup and CUDA-graph capture and before the executor loop
starts.

The primary context, its graphs, streams, allocations and IPC/multicast buffers are
untouched; no kernel, graph or value changes, so outputs are bitwise equal. Any failed
driver call or check raises (no silent fallback). A second reset in one process is skipped:
context churn next to graph replays is a known driver hazard.
"""

import os
import time

import torch

RESET_ENV_VAR_NAME = "TRTLLM_GRAPH_LAUNCH_RESET"

_done_pid = None


def _ok(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"graph launch reset: {what} failed: {err}")
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


def _create_context(drv, cudev):
    # cuda-python 13: cuCtxCreate(params, flags, dev); 12.x: cuCtxCreate_v4 or cuCtxCreate(flags, dev).
    if hasattr(drv, "cuCtxCreate_v4"):
        return _ok(drv.cuCtxCreate_v4(None, 0, cudev), "cuCtxCreate_v4")
    try:
        return _ok(drv.cuCtxCreate(None, 0, cudev), "cuCtxCreate")
    except TypeError:
        return _ok(drv.cuCtxCreate(0, cudev), "cuCtxCreate")


def reset_once() -> dict:
    """Drain the device, create and destroy one context on it, restore the primary context,
    drain, check.

    Raises:
        RuntimeError: on a failed driver call, a current context that is not the device's
            primary one, or a context, device or stream that differs after the reset.
    """
    global _done_pid
    from cuda.bindings import driver as drv

    t0 = time.perf_counter()
    torch.cuda.synchronize()
    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    saved = _ok(drv.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if saved is None or not int(saved):
        raise RuntimeError("graph launch reset: no current context on this thread")
    cudev = _ok(drv.cuDeviceGet(device), "cuDeviceGet")
    primary = _ok(drv.cuDevicePrimaryCtxRetain(cudev), "cuDevicePrimaryCtxRetain")
    _ok(drv.cuDevicePrimaryCtxRelease(cudev), "cuDevicePrimaryCtxRelease")
    if int(primary) != int(saved):
        raise RuntimeError(
            "graph launch reset: the current context is not the device's primary context"
        )
    t1 = time.perf_counter()
    tmp = _create_context(drv, cudev)
    t2 = time.perf_counter()
    try:
        _ok(drv.cuCtxDestroy(tmp), "cuCtxDestroy")
    finally:
        _ok(drv.cuCtxSetCurrent(saved), "cuCtxSetCurrent")
    t3 = time.perf_counter()
    torch.cuda.synchronize()
    now = _ok(drv.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if (
        int(now) != int(saved)
        or torch.cuda.current_device() != device
        or torch.cuda.current_stream() != stream
    ):
        raise RuntimeError("graph launch reset: context, device or stream changed across the reset")
    _done_pid = os.getpid()
    return {
        "device": device,
        "create_ms": round((t2 - t1) * 1e3, 1),
        "destroy_ms": round((t3 - t2) * 1e3, 1),
        "total_ms": round((time.perf_counter() - t0) * 1e3, 1),
    }


def maybe_reset_graph_launch_mode(logger) -> None:
    """Resets the graph-launch scheduling state when ``TRTLLM_GRAPH_LAUNCH_RESET=1``.

    Called once per worker after the final executor's warmup, before ``start_worker()``;
    a no-op for any other value, and for a second call in the same process.
    """
    if os.environ.get(RESET_ENV_VAR_NAME) != "1":
        return
    if _done_pid == os.getpid():
        logger.warning("Graph launch reset already done in this process; skipped.")
        return
    info = reset_once()
    logger.info(
        f"Graph launch reset: transient CUDA context created and destroyed after warmup ({info})."
    )
