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
"""A VisualGen worker's part of the GPU memory monitor.

The worker is one monitored executor, like an LLM's ``PyExecutor``: it
registers while it loads, arms out-of-memory reporting once loaded, passes
the errors it catches while serving to the monitor, and unregisters when it
stops. Settings come from the ``TLLM_MEM_MONITOR*`` environment variables;
``docs/source/features/memory-monitor.md`` describes the modes and reports.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch

from ..memory import monitor
from ..memory.monitor import pools, providers, tagging, triggers
from ..memory.monitor.claims import claim_module_memory
from ..memory.monitor.names import WEIGHTS

__all__ = ["WorkerMemoryMonitor"]

#: Startup step the pipeline load and warmup are recorded under.
_LOAD_STEP = "model"


class WorkerMemoryMonitor:
    """Monitor registration of one VisualGen worker on ``device_id``.

    The monitor follows one executor per process: a worker started while
    another executor holds the monitor runs without it.
    """

    def __init__(self, device_id: int, rank: int, world_size: int) -> None:
        self._device_id = device_id
        self._rank = rank
        self._world_size = world_size
        self._token: providers.ExecutorToken | None = None

    @property
    def monitored(self) -> bool:
        """Whether the monitor follows this worker."""
        return self._token is not None and self._token.monitored

    @contextlib.contextmanager
    def startup(self) -> Iterator[None]:
        """Register the worker for the duration of the pipeline load.

        In full mode the load, warmup included, is recorded as one startup
        step and a startup report follows. A load that raises unregisters
        the worker again.
        """
        monitor.configure(None, world_size=self._world_size)
        self._token = providers.begin_executor()
        try:
            # Loading and warmup may run allocations out of memory and catch
            # the error, as LLM startup does, so nothing is reported meanwhile.
            with (
                triggers.suspend_reporting(),
                tagging.startup_window(self._rank),
                tagging.tag(_LOAD_STEP),
            ):
                yield
        except BaseException:
            self.release()
            raise

    def claim_weights(self, pipeline: torch.nn.Module) -> None:
        """Name the pipeline's GPU parameters and buffers as model weights."""
        if self.monitored:
            claim_module_memory(pipeline, WEIGHTS)

    def arm(self) -> None:
        """Arm out-of-memory reporting for the loaded worker."""
        if not self.monitored:
            return
        if triggers.install_oom_reporter(
            rank=self._rank,
            device=self._device_id,
            owner=self._token,
            world_size=self._world_size,
        ):
            # Classify the pools the load created now rather than inside the
            # first report, which may run while a request is failing.
            pools.prime()

    def report_error(self, exc: BaseException) -> None:
        """Report ``exc`` if it is a device out-of-memory error. Never raises."""
        triggers.notify_error(error=exc, owner=self._token)

    def release(self) -> None:
        """Unregister the worker; later calls do nothing."""
        token, self._token = self._token, None
        if token is not None:
            providers.end_executor(token)
