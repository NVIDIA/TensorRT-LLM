# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-armed Nsight capture window (diagnostic only, off by default).

``TRTLLM_M31_PHASE_NSYS_ARM=<marker path>`` arms a capture of a chosen window of a running server, e.g. one phase
of an E2E benchmark, for nsys run with ``--capture-range=cudaProfilerApi``. Rank 0 reads the marker (JSON,
``{"schema_version": 1, "capture_at_unix_s": <float>, "num_steps": <1..10000, default 200>,
"max_seconds": <0..120, default 10>}``) at most once per second. At ``capture_at_unix_s`` it starts a
``CUDA_PROFILER`` window through ``PyExecutorProfileManager.start_profile``, whose broadcast starts every rank on
the same iteration. The window stops after ``num_steps`` iterations or ``max_seconds``, whichever comes first. One
window per process; never during warmup; refused with the iteration-range or torch-trace profiling env set.

Adapted from dongfengy/tekit 8b5caa029 (GLM-5: a per-rank marker poll calling cudaProfilerStart directly).
"""

import json
import os
import time
from typing import Optional

from tensorrt_llm.logger import logger

PHASE_NSYS_ARM_ENV_VAR_NAME = "TRTLLM_M31_PHASE_NSYS_ARM"
_MAX_MARKER_BYTES = 4096


class PhaseCapture:
    """Rank 0's per-iteration driver of one phase-armed profile window."""

    def __init__(self, manager, executor, marker_path: str):
        self._manager = manager
        self._executor = executor
        self._path = marker_path
        self._next_poll = 0.0
        self._capture_at: Optional[float] = None
        self._num_steps = 200
        self._max_seconds = 10.0
        self._started_at: Optional[float] = None
        self._seen_enabled = False
        self.done = False

    def _read_marker(self) -> bool:
        """Parse the marker once it exists; returns True once armed. Any malformed marker disarms for good."""
        try:
            with open(self._path, encoding="utf-8") as f:
                text = f.read(_MAX_MARKER_BYTES + 1)
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.error(f"[M31 phase capture] marker read failed ({e}); disarmed")
            self.done = True
            return False
        try:
            if len(text) > _MAX_MARKER_BYTES:
                raise ValueError("marker exceeds 4096 bytes")
            control = json.loads(text)
            if not isinstance(control, dict) or control.get("schema_version") != 1:
                raise ValueError("unsupported marker schema")
            capture_at = float(control["capture_at_unix_s"])
            num_steps = int(control.get("num_steps", 200))
            max_seconds = float(control.get("max_seconds", 10.0))
            if not (
                0 < capture_at < float("inf") and 1 <= num_steps <= 10000 and 0 < max_seconds <= 120
            ):
                raise ValueError("marker values out of range")
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"[M31 phase capture] bad marker {self._path} ({e}); disarmed")
            self.done = True
            return False
        self._capture_at, self._num_steps, self._max_seconds = capture_at, num_steps, max_seconds
        logger.info(
            f"[M31 phase capture] armed at iteration {self._executor.iter_counter}: capture_at_unix_s="
            f"{capture_at:.3f} num_steps={num_steps} max_seconds={max_seconds}"
        )
        return True

    def _finish(self, reason: str) -> None:
        self.done = True
        logger.info(
            f"[M31 phase capture] window ended at iteration {self._executor.iter_counter}: reason={reason} "
            f"seconds={time.monotonic() - self._started_at:.3f}"
        )

    def tick(self) -> None:
        """Called once per executor iteration on rank 0."""
        if self.done or self._executor.is_warmup:
            return
        if self._capture_at is None:
            now = time.monotonic()
            if now < self._next_poll:
                return
            self._next_poll = now + 1.0
            if not self._read_marker():
                return
        if self._started_at is None:
            if time.time() < self._capture_at:
                return
            try:
                self._manager.start_profile(num_steps=self._num_steps, activities=["CUDA_PROFILER"])
            except Exception as e:  # noqa: BLE001 - a diagnostic must never take the server down
                logger.error(f"[M31 phase capture] start_profile refused ({e}); disarmed")
                self.done = True
                return
            self._started_at = time.monotonic()
            logger.info(
                f"[M31 phase capture] window scheduled at iteration {self._executor.iter_counter}"
            )
            return
        enabled = self._executor._profile_enabled
        if enabled:
            self._seen_enabled = True
            if time.monotonic() - self._started_at >= self._max_seconds:
                # The non-blocking half of stop_profile(): it would wait for this very thread to run the stop.
                self._manager.apply_stop_config()
                queue = getattr(self._executor, "executor_request_queue", None)
                if queue is not None:
                    queue.enqueue_profile_stop_request()
                self._finish("max_seconds")
        elif self._seen_enabled:
            self._finish("num_steps")


def make_phase_capture(manager, executor, conflicting_env: bool) -> Optional[PhaseCapture]:
    """The rank-0 phase capture when TRTLLM_M31_PHASE_NSYS_ARM is set, else None."""
    path = os.environ.get(PHASE_NSYS_ARM_ENV_VAR_NAME)
    if not path:
        return None
    if conflicting_env:
        logger.error(
            f"[M31 phase capture] {PHASE_NSYS_ARM_ENV_VAR_NAME} is exclusive with TLLM_PROFILE_START_STOP / "
            "TLLM_TORCH_PROFILE_TRACE; disarmed"
        )
        return None
    if getattr(getattr(executor, "dist", None), "rank", 0) != 0:
        return None
    logger.info(f"[M31 phase capture] waiting for marker {path}")
    return PhaseCapture(manager, executor, path)
