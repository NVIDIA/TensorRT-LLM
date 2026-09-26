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

"""Process-local backing for an idle diffusion pipeline's GPU allocations."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Condition, Lock, get_ident
from typing import Literal
from uuid import uuid4

import torch

from tensorrt_llm._torch import virtual_memory


class PipelineSleepManager:
    """Keep one pipeline's allocation pools alive across sleep/wake transitions.

    Generation must enter generation() for its entire GPU workload. Sleep closes
    admission and waits for that call before releasing memory. This manager does
    not drain an external scheduler or preserve a terminated process.
    """

    def __init__(
        self,
        restore_mode: Literal["CPU", "PINNED"],
        device: torch.device,
        *,
        release_cpu_backup: bool = False,
    ) -> None:
        if restore_mode not in ("CPU", "PINNED"):
            raise ValueError("Pipeline sleep requires CPU or PINNED memory backing")
        self._mode = virtual_memory.RestoreMode[restore_mode]
        self._device = device
        self._tag = f"visual_gen_{uuid4().hex}"
        self._pool: object | None = None
        self._released_blobs = 0
        self._state: Literal["uninitialized", "awake", "sleeping", "asleep", "waking", "failed"] = (
            "uninitialized"
        )
        self._condition = Condition()
        self._transition_lock = Lock()
        self._generation_thread: int | None = None
        self._release_backup_on_wake = release_cpu_backup

    @contextmanager
    def loading(self) -> Iterator[None]:
        """Capture persistent allocations, excluding inference warmup buffers."""
        if self._state != "uninitialized":
            raise RuntimeError("A pipeline sleep manager can load only once")
        self._state = "failed"
        with torch.cuda.device(self._device):
            with virtual_memory.scope(self._tag, self._mode) as pool:
                self._pool = pool
                yield
        self._state = "awake"

    @property
    def is_sleeping(self) -> bool:
        with self._condition:
            return self._state == "asleep"

    def ensure_awake(self) -> None:
        """Reject GPU work if memory is absent or a transition was interrupted."""
        with self._condition:
            if self._state != "awake":
                raise RuntimeError(
                    f"Pipeline is {self._state}; generation requires an awake pipeline"
                )

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Admit one forward call; overlapping forwards are not supported."""
        with self._condition:
            self.ensure_awake()
            if self._generation_thread is not None:
                raise RuntimeError(
                    "Concurrent generation on a sleep-enabled pipeline is unsupported"
                )
            self._generation_thread = get_ident()
        try:
            yield
        finally:
            with self._condition:
                self._generation_thread = None
                self._condition.notify_all()

    def _reject_generation_callback(self) -> None:
        # A callback cannot wait for its own forward call to finish.
        with self._condition:
            if self._generation_thread == get_ident():
                raise RuntimeError("Cannot sleep or wake from inside an active generation call")

    def sleep(self) -> None:
        """Stop admission, wait for generation, then back up and release GPU memory."""
        self._reject_generation_callback()
        with self._transition_lock:
            with self._condition:
                if self._state == "asleep":
                    return
                self.ensure_awake()
                self._state = "sleeping"
            try:
                with self._condition:
                    self._condition.wait_for(lambda: self._generation_thread is None)
                with torch.cuda.device(self._device):
                    torch.cuda.synchronize(self._device)
                    self._released_blobs = virtual_memory.release_with_tag(self._tag)
                    if self._released_blobs == 0:
                        raise RuntimeError("No sleep-managed GPU allocations were captured")
                    torch.cuda.synchronize(self._device)
                    torch.cuda.empty_cache()
                with self._condition:
                    self._state = "asleep"
            finally:
                with self._condition:
                    if self._state == "sleeping":
                        self._state = "failed"

    def wake_up(self) -> None:
        """Restore backed allocations at their original GPU virtual addresses."""
        self._reject_generation_callback()
        with self._transition_lock:
            with self._condition:
                if self._state == "awake":
                    return
                if self._state != "asleep":
                    raise RuntimeError(f"Cannot wake a pipeline in state {self._state}")
                self._state = "waking"
            try:
                with torch.cuda.device(self._device):
                    restored_blobs = virtual_memory.materialize_with_tag(self._tag)
                    if restored_blobs != self._released_blobs:
                        raise RuntimeError(
                            f"Restored {restored_blobs} sleep-managed allocations; "
                            f"expected {self._released_blobs}"
                        )
                    torch.cuda.synchronize(self._device)
                    if self._release_backup_on_wake:
                        virtual_memory.release_host_backups_with_tag(self._tag)
                with self._condition:
                    self._state = "awake"
            finally:
                with self._condition:
                    if self._state == "waking":
                        self._state = "failed"
