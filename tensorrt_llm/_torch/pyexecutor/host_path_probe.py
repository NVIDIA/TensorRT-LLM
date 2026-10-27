# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rank host-path probe (diagnostic only, off by default).

``TRTLLM_M31_HOST_PATH_PROBE=<dir>`` records, per executor iteration and rank, host-only wall-clock stamps (no
CUDA events, no extra synchronization point) into ``<dir>/hostpath-rank<r>-pid<p>.bin`` as int64 rows:

    t_top      loop top (profile_step)
    t_launch   _forward_step entry
    t_fwd_ret  _forward_step return (the forward is issued)
    t_wait0    before waiting on the previous step's sampler event (in _update_requests)
    t_wait1    the previous step's sampler event has completed
    n_ctx      context requests in the forward
    n_gen      generation requests in the forward

``t_wait1 - t_wait0`` is how long the host waited for the GPU: the host's headroom. A step with no wait was
host-bound. The probe waits on the same event the sampler then waits on (``SamplerEvent.synchronize`` is
idempotent), so it moves no work and adds no synchronization. Rows are flushed every 4096 steps and at exit.
Reader: the Host fixture's host_path_read_v1.py. Without the variable the executor installs nothing.
"""

import array
import atexit
import os
import time
from typing import Optional

from tensorrt_llm.logger import logger

HOST_PATH_PROBE_ENV_VAR_NAME = "TRTLLM_M31_HOST_PATH_PROBE"
_FIELDS = 7
_FLUSH_ROWS = 4096


class HostPathProbe:
    """One rank's per-iteration host stamps; installed on an executor by ``make_host_path_probe``."""

    def __init__(self, out_dir: str, rank: int):
        os.makedirs(out_dir, exist_ok=True)
        self._path = os.path.join(out_dir, f"hostpath-rank{rank}-pid{os.getpid()}.bin")
        self._rows = array.array("q")
        self._row = [0] * _FIELDS
        self._ns = time.time_ns
        atexit.register(self.flush)

    def top(self) -> None:
        row = self._row
        if row[0]:
            self._rows.extend(row)
            if len(self._rows) >= _FLUSH_ROWS * _FIELDS:
                self.flush()
        self._row = [self._ns(), 0, 0, 0, 0, 0, 0]

    def flush(self) -> None:
        if self._rows:
            with open(self._path, "ab") as f:
                self._rows.tofile(f)
            del self._rows[:]

    def install(self, executor) -> None:
        """Wrap this executor's _forward_step and _update_requests (instance attributes only)."""
        forward_step = executor._forward_step
        update_requests = executor._update_requests
        ns = self._ns

        def _forward_step(scheduled_requests, *args, **kwargs):
            row = self._row
            row[1] = ns()
            row[5] = scheduled_requests.num_context_requests
            row[6] = len(scheduled_requests.generation_requests)
            try:
                return forward_step(scheduled_requests, *args, **kwargs)
            finally:
                row[2] = ns()

        def _update_requests(sample_state, *args, **kwargs):
            row = self._row
            event = getattr(sample_state, "sampler_event", None)
            row[3] = ns()
            if event is not None:
                event.synchronize()
            row[4] = ns()
            return update_requests(sample_state, *args, **kwargs)

        executor._forward_step = _forward_step
        executor._update_requests = _update_requests


def make_host_path_probe(executor) -> Optional[HostPathProbe]:
    """The installed probe when TRTLLM_M31_HOST_PATH_PROBE is set, else None."""
    out_dir = os.environ.get(HOST_PATH_PROBE_ENV_VAR_NAME)
    if not out_dir:
        return None
    rank = getattr(executor, "global_rank", getattr(getattr(executor, "dist", None), "rank", 0))
    probe = HostPathProbe(out_dir, rank)
    probe.install(executor)
    logger.info(f"[M31 host path probe] rank {rank} writing {probe._path}")
    return probe
