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

import atexit
import sys
import threading
from _thread import LockType
from collections.abc import Callable, Iterator
from dataclasses import dataclass

# avoid importing the whole tensorrt_llm module, which takes time during debugging.
from importlib.util import find_spec
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Sequence, cast

import cuda.bindings.driver as drv

from ._common import Address, CacheTier, CudaStream, MemAddress
from ._utils import (
    CachedCudaEvent,
    HomoTuple,
    HostMem,
    _unwrap,
    div_up,
    stream_wait_events,
    temporary_sys_path,
)

_NixlGdsCopyEngine: Any

if "tensorrt_llm" in sys.modules:
    from tensorrt_llm.bindings.internal.batch_manager import (
        kv_cache_manager_v2_utils as _native_utils,
    )
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (  # noqa # type: ignore
        DiskAddress,
        DiskToDiskTask,
        DiskToHostTask,
        HostToDiskTask,
        MemToMemTask,
        copy_device_to_device,
        copy_device_to_host,
        copy_disk_to_disk,
        copy_disk_to_host,
        copy_host_to_device,
        copy_host_to_disk,
        copy_host_to_host,
    )

    _NixlGdsCopyEngine = getattr(_native_utils, "NixlGdsCopyEngine", None)
else:
    # fast path for dev, avoids importing the whole tensorrt_llm module
    spec = find_spec("kv_cache_manager_v2")
    assert spec is not None and spec.origin is not None
    with temporary_sys_path(str(Path(spec.origin).parent.parent.parent)):
        from bindings.internal.batch_manager import kv_cache_manager_v2_utils as _native_utils  # noqa
        from bindings.internal.batch_manager.kv_cache_manager_v2_utils import (  # noqa
            DiskAddress,
            DiskToDiskTask,
            DiskToHostTask,
            HostToDiskTask,
            MemToMemTask,
            copy_device_to_device,
            copy_device_to_host,
            copy_disk_to_disk,
            copy_disk_to_host,
            copy_host_to_device,
            copy_host_to_disk,
            copy_host_to_host,
        )

        _NixlGdsCopyEngine = getattr(_native_utils, "NixlGdsCopyEngine", None)


class CopyTask(NamedTuple):
    dst: Address
    src: Address


def _copy_gpu_to_gpu(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_device_to_device([MemToMemTask(dst, src) for dst, src in tasks], num_bytes, stream)
        )
    )


def _copy_host_to_host(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_host_to_host([MemToMemTask(dst, src) for dst, src in tasks], num_bytes, stream)
        )
    )


def _copy_disk_to_disk(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_disk_to_disk(
                [
                    DiskToDiskTask(
                        DiskAddress(
                            cast(DiskAddress, dst).fd,
                            cast(DiskAddress, dst).pos,
                        ),
                        DiskAddress(
                            cast(DiskAddress, src).fd,
                            cast(DiskAddress, src).pos,
                        ),
                    )
                    for dst, src in tasks
                ],
                num_bytes,
                stream,
            )
        )
    )


def _copy_gpu_to_host(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_device_to_host([MemToMemTask(dst, src) for dst, src in tasks], num_bytes, stream)
        )
    )


def _copy_host_to_gpu(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_host_to_device([MemToMemTask(dst, src) for dst, src in tasks], num_bytes, stream)
        )
    )


def _copy_disk_to_host(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_disk_to_host(
                [
                    DiskToHostTask(
                        cast(MemAddress, dst),
                        DiskAddress(cast(DiskAddress, src).fd, cast(DiskAddress, src).pos),
                    )
                    for dst, src in tasks
                ],
                num_bytes,
                stream,
            )
        )
    )


def _copy_host_to_disk(tasks: Sequence[CopyTask], num_bytes: int, stream: CudaStream):
    _unwrap(
        drv.CUresult(
            copy_host_to_disk(
                [
                    HostToDiskTask(
                        DiskAddress(
                            cast(DiskAddress, dst).fd,
                            cast(DiskAddress, dst).pos,
                        ),
                        cast(MemAddress, src),
                    )
                    for dst, src in tasks
                ],
                num_bytes,
                stream,
            )
        )
    )


Copier = Callable[[Sequence[CopyTask], int, CudaStream], None]


def get_copier(dst: CacheTier, src: CacheTier) -> Copier | HomoTuple[Copier]:
    copiers: HomoTuple[HomoTuple[Copier | HomoTuple[Copier]]] = (
        # dst = GPU_MEM
        (
            _copy_gpu_to_gpu,  # src = GPU_MEM
            _copy_host_to_gpu,  # src = HOST_MEM
            (_copy_disk_to_host, _copy_host_to_gpu),  # src = DISK
        ),
        # dst = HOST_MEM
        (
            _copy_gpu_to_host,  # src = GPU_MEM
            _copy_host_to_host,  # src = HOST_MEM
            _copy_disk_to_host,  # src = DISK
        ),
        # dst = DISK
        (
            (_copy_gpu_to_host, _copy_host_to_disk),  # src = GPU_MEM
            _copy_host_to_disk,  # src = HOST_MEM
            _copy_disk_to_disk,  # src = DISK
        ),
    )
    return copiers[dst][src]


@dataclass(slots=True)
class GrainMetadata:
    mutex: LockType
    ready_event: CachedCudaEvent  # protects the buffer grain.


class StagingBuffer:
    __slots__ = ("manager", "min_size", "max_size", "_size", "start_grain", "stream")
    manager: "StagingBufferManager"
    min_size: int
    max_size: int
    _size: int
    start_grain: int
    stream: CudaStream

    def __init__(
        self, manager: "StagingBufferManager", min_size: int, max_size: int, stream: CudaStream
    ):
        self.manager = manager
        self.min_size = min_size
        self.max_size = max_size
        self.stream = stream

    @property
    def address(self) -> MemAddress:
        return MemAddress(self.manager.buffer.address + self.start_grain * self.manager.GRANULARITY)

    @property
    def size(self) -> int:
        return self._size

    @property
    def num_grains(self) -> int:
        return div_up(self._size, self.manager.GRANULARITY)

    @property
    def grains(self) -> list[GrainMetadata]:
        return self.manager.grains[self.start_grain : self.start_grain + self.num_grains]

    def __enter__(self) -> "StagingBuffer":
        manager = self.manager
        if self.min_size > manager.size:
            raise ValueError(f"Requested min_size {self.min_size} is too large for the manager")
        with manager.mutex:
            self._size = min(self.max_size, manager._suggest_next_max_size_unsafe())
            self.start_grain = manager.next
            manager.next += self.num_grains
            assert manager.next <= manager.num_grains
            if manager.next == manager.num_grains:
                manager.next = 0

            def lock_and_consume_events() -> Iterator[CachedCudaEvent]:
                for grain in self.grains:
                    grain.mutex.acquire()
                    yield grain.ready_event
                    grain.ready_event = CachedCudaEvent.NULL

            stream_wait_events(self.stream, lock_and_consume_events())
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        event = CachedCudaEvent(self.stream)
        for grain in reversed(self.grains):
            grain.ready_event = event
            grain.mutex.release()


class StagingBufferManager:
    __slots__ = ("mutex", "buffer", "grains", "next")
    GRANULARITY: ClassVar[int] = 1 << 20

    mutex: LockType
    buffer: HostMem
    grains: list[GrainMetadata]
    next: int

    def __init__(self, size: int) -> None:
        assert size % self.GRANULARITY == 0
        self.mutex = threading.Lock()
        num_grains = size // self.GRANULARITY
        self.buffer = HostMem(size)
        self.grains = [
            GrainMetadata(threading.Lock(), CachedCudaEvent.NULL) for _ in range(num_grains)
        ]
        self.next = 0

    @property
    def size(self) -> int:
        "Requesting more than this will fail."
        assert len(self.grains) * self.GRANULARITY == self.buffer.size
        return self.buffer.size

    @property
    def num_grains(self) -> int:
        return len(self.grains)

    def _suggest_next_max_size_unsafe(self) -> int:
        "Requesting more than this may degrade performance. Must be called with self.mutex held."
        return self.GRANULARITY * (self.num_grains - self.next)

    # max_size is just a hint, the actual size may be smaller.
    def new(self, min_size: int, max_size: int, stream: CudaStream) -> StagingBuffer:
        """
        min_size is the min required size. max_size is for best efforts. Your should query the actual
        size after entering the context.
        """
        return StagingBuffer(self, min_size, max_size, stream)


class CopyEngine:
    __slots__ = ("_staging_buffer_manager", "_nixl_gds_engine", "_nixl_gds_thread_count")
    _staging_buffer_manager: StagingBufferManager | None
    _nixl_gds_engine: Any | None
    _nixl_gds_thread_count: int | None

    def __init__(self) -> None:
        self._staging_buffer_manager = None
        self._nixl_gds_engine = None
        self._nixl_gds_thread_count = None

    def close(self) -> None:
        self._staging_buffer_manager = None
        self._nixl_gds_engine = None
        self._nixl_gds_thread_count = None

    def initialize_nixl_gds(self, thread_count: int) -> Any:
        """Create the one GDS_MT engine owned by this worker process."""
        if thread_count <= 0:
            raise ValueError("NIXL GDS_MT thread count must be positive")
        if self._nixl_gds_engine is None:
            if _NixlGdsCopyEngine is None:
                raise RuntimeError(
                    "NIXL GDS support is unavailable in the loaded TensorRT-LLM bindings"
                )
            self._nixl_gds_engine = _NixlGdsCopyEngine(thread_count)
            self._nixl_gds_thread_count = thread_count
        elif thread_count != self._nixl_gds_thread_count:
            raise RuntimeError(
                "NIXL GDS_MT is already initialized with thread_count="
                f"{self._nixl_gds_thread_count}; cannot reconfigure it to {thread_count}"
            )
        return self._nixl_gds_engine

    @property
    def staging_buffer_manager(self) -> StagingBufferManager:
        if self._staging_buffer_manager is None:
            self._staging_buffer_manager = StagingBufferManager(64 << 20)
        return self._staging_buffer_manager

    # @TODO: Use a dedicated stream for each different Copier, take set[CachedCudaEvent] instead of
    # stream, and return a new CachedCudaEvent.
    def transfer(
        self,
        dst_cache_tier: CacheTier,
        src_cache_tier: CacheTier,
        num_bytes: int,
        tasks: Sequence[CopyTask],
        stream: CudaStream,
        disk_backend: str = "posix",
        gds_thread_count: int = 8,
    ) -> None:
        if disk_backend not in ("posix", "nixl_gds"):
            raise ValueError(f"Unsupported disk cache backend: {disk_backend}")
        if (
            disk_backend == "nixl_gds"
            and CacheTier.DISK in (dst_cache_tier, src_cache_tier)
            and dst_cache_tier != src_cache_tier
        ):
            return self._transfer_nixl_gds(
                dst_cache_tier, src_cache_tier, num_bytes, tasks, stream, gds_thread_count
            )

        copier = get_copier(dst_cache_tier, src_cache_tier)
        if not isinstance(copier, tuple):
            return copier(tasks, num_bytes, stream)
        assert len(copier) == 2, "for now, we only support 2 copiers via host memory"
        manager = self.staging_buffer_manager
        remaining = tasks
        while remaining:
            with manager.new(num_bytes, num_bytes * len(remaining), stream) as buf:
                addr = buf.address
                n = buf.size // num_bytes
                assert n <= len(remaining)
                batch = remaining[:n]
                copier[0](
                    [
                        CopyTask(MemAddress(addr + num_bytes * i), t.src)
                        for i, t in enumerate(batch)
                    ],
                    num_bytes,
                    buf.stream,
                )
                copier[1](
                    [
                        CopyTask(t.dst, MemAddress(addr + num_bytes * i))
                        for i, t in enumerate(batch)
                    ],
                    num_bytes,
                    buf.stream,
                )
                remaining = remaining[n:]

    def _transfer_nixl_gds(
        self,
        dst_cache_tier: CacheTier,
        src_cache_tier: CacheTier,
        num_bytes: int,
        tasks: Sequence[CopyTask],
        stream: CudaStream,
        thread_count: int,
    ) -> None:
        engine = self.initialize_nixl_gds(thread_count)

        if dst_cache_tier == CacheTier.DISK:
            disk_tasks = [
                HostToDiskTask(
                    DiskAddress(cast(DiskAddress, task.dst).fd, cast(DiskAddress, task.dst).pos),
                    cast(MemAddress, task.src),
                )
                for task in tasks
            ]
            if src_cache_tier == CacheTier.GPU_MEM:
                result = engine.copy_device_to_disk(disk_tasks, num_bytes, stream)
            elif src_cache_tier == CacheTier.HOST_MEM:
                result = engine.copy_host_to_disk(disk_tasks, num_bytes, stream)
            else:
                raise ValueError(f"Unsupported NIXL GDS source tier: {src_cache_tier}")
        elif src_cache_tier == CacheTier.DISK:
            memory_tasks = [
                DiskToHostTask(
                    cast(MemAddress, task.dst),
                    DiskAddress(cast(DiskAddress, task.src).fd, cast(DiskAddress, task.src).pos),
                )
                for task in tasks
            ]
            if dst_cache_tier == CacheTier.GPU_MEM:
                result = engine.copy_disk_to_device(memory_tasks, num_bytes, stream)
            elif dst_cache_tier == CacheTier.HOST_MEM:
                result = engine.copy_disk_to_host(memory_tasks, num_bytes, stream)
            else:
                raise ValueError(f"Unsupported NIXL GDS destination tier: {dst_cache_tier}")
        else:
            raise ValueError(
                f"NIXL GDS requires one disk tier, got {src_cache_tier} -> {dst_cache_tier}"
            )
        _unwrap(drv.CUresult(result))


_copy_engine = CopyEngine()
atexit.register(_copy_engine.close)


def initialize_nixl_gds(thread_count: int) -> None:
    _copy_engine.initialize_nixl_gds(thread_count)


def batched_copy(
    dst_cache_tier: CacheTier,
    src_cache_tier: CacheTier,
    num_bytes: int,
    tasks: Sequence[CopyTask],
    stream: CudaStream,
    disk_backend: str = "posix",
    gds_thread_count: int = 8,
) -> None:
    _copy_engine.transfer(
        dst_cache_tier,
        src_cache_tier,
        num_bytes,
        tasks,
        stream,
        disk_backend,
        gds_thread_count,
    )
