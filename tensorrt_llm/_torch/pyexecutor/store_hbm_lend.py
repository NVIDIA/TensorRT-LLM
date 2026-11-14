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
"""Lend GPU memory (HBM) to a Mooncake store pool.

``store_host_donor.hbm_lend_size_per_gpu`` makes every rank lend that many
bytes of its GPU's memory to the pool, next to the host memory the donor
lends. The rank allocates the buffer itself after the final executor sized
its KV cache and captured its CUDA graphs, opens its own Mooncake client and
mounts the buffer as a store segment with
``MooncakeDistributedStore.mount_buffer``. The buffer stays allocated until
the segment is unmounted at shutdown; if unmounting fails, the buffer is kept
until the process exits so the store never points at freed memory.

The segment uses the transport of the store it joins: RDMA, where a CUDA
buffer is registered for GPUDirect RDMA, or NVLink under
``TLLM_MOONCAKE_STORE_EGM=1``, where the buffer is a FABRIC allocation from
Mooncake's NVLink allocator so that peers can map it.

Mooncake releases without ``mount_buffer`` cannot mount memory the caller
owns. The lend then logs one ``STORE_HBM_LEND_UNSUPPORTED`` line and lends
nothing; the executor runs as without the setting.
"""

import ctypes
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger

LOG_TAG = "STORE_HBM_LEND"
#: GPU memory that must stay free after the lend.
RESERVE_BYTES = 8 << 30
#: Mooncake's FABRIC allocator, shipped in the ``mooncake`` package.
NVLINK_ALLOCATOR = "nvlink_allocator.so"
SUPPORTED_PROTOCOLS = ("rdma", "nvlink")


class CudaBuffer:
    """A CUDA buffer from the PyTorch allocator, for RDMA."""

    def __init__(self, size: int, device: int) -> None:
        import torch

        self._tensor = torch.empty(size, dtype=torch.uint8, device=device)
        self.pointer = self._tensor.data_ptr()

    def free(self) -> None:
        self._tensor = None
        self.pointer = 0


class FabricBuffer:
    """A FABRIC allocation from Mooncake's NVLink allocator, for NVLink."""

    def __init__(self, size: int, device: int, path: Optional[str] = None) -> None:
        if path is None:
            import mooncake

            path = os.path.join(
                os.path.dirname(os.path.abspath(mooncake.__file__)), NVLINK_ALLOCATOR
            )
        lib = ctypes.CDLL(path)
        lib.mc_nvlink_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
        lib.mc_nvlink_malloc.restype = ctypes.c_void_p
        lib.mc_nvlink_free.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ssize_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.mc_nvlink_free.restype = None
        self._lib, self._size, self._device = lib, size, device
        self.pointer = lib.mc_nvlink_malloc(size, device, None) or 0
        if not self.pointer:
            raise RuntimeError(
                f"{LOG_TAG}: FABRIC allocation of {size} bytes failed on device {device}"
            )

    def free(self) -> None:
        if self.pointer:
            self._lib.mc_nvlink_free(self.pointer, self._size, self._device, None)
            self.pointer = 0


def _default_allocate(size: int, device: int, protocol: str) -> Any:
    if protocol == "nvlink":
        return FabricBuffer(size, device)
    return CudaBuffer(size, device)


def _default_free_memory(device: int) -> int:
    import torch

    torch.cuda.synchronize(device)
    return torch.cuda.mem_get_info(device)[0]


def _default_device() -> int:
    import torch

    return torch.cuda.current_device()


def store_supports_mount_buffer(store: Any) -> bool:
    """Whether a Mooncake store can mount memory its caller owns."""
    return callable(getattr(store, "mount_buffer", None))


class StoreHbmLend:
    """One rank's GPU memory lent to the store: a client and a mounted buffer."""

    def __init__(
        self,
        *,
        size: int,
        master: str,
        protocol: str,
        device_name: str,
        metadata_server: str,
        rank: int,
        store_factory: Callable[[], Any],
        allocate: Callable[[int, int, str], Any] = _default_allocate,
        free_memory: Callable[[int], int] = _default_free_memory,
        device: Callable[[], int] = _default_device,
        reserve: int = RESERVE_BYTES,
    ) -> None:
        if size <= 0:
            raise ValueError(f"{LOG_TAG}: size must be positive, got {size}")
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"{LOG_TAG}: the store's protocol {protocol!r} cannot reach GPU "
                f"memory; lending HBM needs one of {SUPPORTED_PROTOCOLS}"
            )
        self.size = size
        self.master = master
        self.protocol = protocol
        self.device_name = device_name
        self.metadata_server = metadata_server
        self.rank = rank
        self._store_factory = store_factory
        self._allocate = allocate
        self._free_memory = free_memory
        self._device = device
        self._reserve = reserve
        self.store: Any = None
        self.buffer: Any = None
        self.segment_ids: List[str] = []
        self.record: Optional[Dict[str, Any]] = None

    @property
    def mounted(self) -> bool:
        return bool(self.segment_ids)

    def mount(self, hostname: str) -> bool:
        """Lend the buffer under ``hostname``.

        Returns False when the Mooncake release cannot mount caller memory.
        """
        if self.store is not None:
            raise RuntimeError(f"{LOG_TAG}: already lent")
        store = self._store_factory()
        if not store_supports_mount_buffer(store):
            logger.warning(
                f"{LOG_TAG}_UNSUPPORTED "
                + json.dumps(
                    {
                        "rank": self.rank,
                        "bytes": self.size,
                        "reason": "the installed Mooncake has no "
                        "MooncakeDistributedStore.mount_buffer; nothing lent",
                    },
                    sort_keys=True,
                )
            )
            return False
        device = self._device()
        free_before = self._free_memory(device)
        if free_before < self.size + self._reserve:
            raise RuntimeError(
                f"{LOG_TAG}: device {device} has {free_before} bytes free; lending "
                f"{self.size} bytes needs {self.size + self._reserve} including a "
                f"{self._reserve}-byte reserve. Lower kv_cache_config."
                "free_gpu_memory_fraction or hbm_lend_size_per_gpu."
            )
        status = store.setup(
            hostname, self.metadata_server, 0, 0, self.protocol, self.device_name, self.master
        )
        if status != 0:
            raise RuntimeError(
                f"{LOG_TAG}: MooncakeDistributedStore.setup failed with "
                f"status {status} (master={self.master})"
            )
        self.store = store
        try:
            self.buffer = self._allocate(self.size, device, self.protocol)
            result = store.mount_buffer(
                self.buffer.pointer, self.size, protocol=self.protocol, location="*"
            )
            ret, segment_ids = result["ret"], list(result["segment_ids"])
            if ret != 0 or not segment_ids:
                raise RuntimeError(f"{LOG_TAG}: mount_buffer failed with status {ret}")
        except BaseException:
            # A failed mount leaves nothing mounted: release what was taken.
            self._release()
            raise
        self.segment_ids = segment_ids
        free_after = self._free_memory(device)
        get_hostname = getattr(store, "get_hostname", None)
        self.record = {
            "rank": self.rank,
            "device": device,
            "bytes": self.size,
            "protocol": self.protocol,
            "segment": str(get_hostname()) if callable(get_hostname) else hostname,
            "segment_ids": self.segment_ids,
            "free_before": free_before,
            "free_after": free_after,
        }
        logger.info(f"{LOG_TAG}_MOUNTED {json.dumps(self.record, sort_keys=True)}")
        if free_after < self._reserve:
            logger.warning(
                f"{LOG_TAG}: device {device} has {free_after} bytes free after "
                f"the lend, below the {self._reserve}-byte reserve"
            )
        return True

    def _release(self) -> Tuple[bool, str]:
        """Free the buffer, then close the client; only call with nothing mounted."""
        if self.buffer is not None:
            self.buffer.free()
            self.buffer = None
        status: Any = 0
        if self.store is not None:
            try:
                status = self.store.close()
            except Exception as error:  # noqa: BLE001 - reported by the caller
                status = f"{type(error).__name__}: {error}"
            self.store = None
        return status in (0, None), str(status)

    def close(self) -> bool:
        """Unmount the segment, then free the buffer. Idempotent.

        Returns False when the segment could not be unmounted; the buffer and
        the client are then kept until the process exits.
        """
        if self.store is None:
            return True
        status: Any = 0
        if self.segment_ids:
            try:
                status = self.store.unmount_segment(self.segment_ids, 0)
            except Exception as error:  # noqa: BLE001 - reported, memory retained
                status = f"{type(error).__name__}: {error}"
        record = {"rank": self.rank, "segment_ids": self.segment_ids, "status": str(status)}
        if status not in (0, None):
            logger.error(
                f"{LOG_TAG}_RETIRE_FAILED {json.dumps(record, sort_keys=True)}; "
                "GPU memory retained until process exit"
            )
            return False
        self.segment_ids = []
        closed, close_status = self._release()
        if not closed:
            # The segment is gone and the buffer freed; only the client is left.
            logger.error(f"{LOG_TAG}: closing the lend client failed: {close_status}")
        logger.info(f"{LOG_TAG}_RETIRED {json.dumps(record, sort_keys=True)}")
        return True
