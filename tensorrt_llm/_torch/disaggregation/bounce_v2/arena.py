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
"""NIXL-registered bounce arena: FabricArena + agent region registration.

Integration-layer module (may import the compiled binding) — NOT part of the
pure-logic core.

ORDERING INVARIANT (design doc Section 5.4): the arena must be registered
with the transfer agent BEFORE ``get_local_agent_desc()`` is called, so every
peer's loaded metadata includes the arena and per-chunk 1:1 writes into it
resolve to a single NIXL descriptor. The caller (``BounceEngine`` inside
``TransferWorker.__init__``) constructs this object before the rank-info /
AgentDesc exchange.

FABRIC POLICY: fabric memory (MNNVL / GPUDirect-RDMA capable) is REQUIRED
where the platform supports it — a silent degradation to plain device memory
on GB200 would quietly lose the NVLink-fabric data path. Support is probed
with a tiny throwaway allocation; when unsupported (x86 CI boxes) or
explicitly disabled via ``BounceV2Config.disable_fabric_memory`` the arena
falls back to ``cudaMalloc`` with a warning (mirroring the C++ CI escape
hatch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from .config import BounceV2Config

__all__ = ["BounceArena"]

_PROBE_BYTES = 1 << 10


class BounceArena:
    """One contiguous device staging buffer, registered with the NIXL agent.

    ``raw_agent`` must be the C++ binding agent (exposing ``register_region``
    / ``deregister_region``), not the Python wrapper. ``close()`` deregisters;
    the underlying device memory is freed when the last reference to this
    object drops.
    """

    def __init__(self, raw_agent, config: "BounceV2Config", device_id: int) -> None:
        # Deferred import: keeps this module importable (e.g. by engine.py's
        # NoBounceEngine path) on hosts without the compiled binding.
        from tensorrt_llm.tensorrt_llm_transfer_agent_binding import FabricArena

        self._agent = raw_agent
        self._device_id = device_id
        self._registered = False

        if config.disable_fabric_memory:
            require_fabric = False
        else:
            # Probe fabric support with a tiny allocation, then REQUIRE it
            # when available: never silently degrade on a fabric-capable box.
            probe = FabricArena(_PROBE_BYTES, device_id, require_fabric=False)
            require_fabric = bool(probe.is_fabric)
            del probe

        self._arena = FabricArena(config.arena_size_bytes, device_id, require_fabric=require_fabric)
        if not self._arena.is_fabric:
            logger.warning(
                f"bounce_v2: arena on device {device_id} is NOT fabric memory "
                f"(unsupported platform or disable_fabric_memory=True); "
                f"NVLink-fabric / GPUDirect-RDMA data paths are unavailable"
            )

        # Register BEFORE any peer can fetch our agent metadata (see module
        # docstring). register_region sits below the VMM splitter: one
        # registration, single-descriptor per-chunk writes.
        if not raw_agent.register_region(self._arena.base_ptr, self._arena.size, device_id):
            raise RuntimeError(
                f"bounce_v2: NIXL registration of the bounce arena failed "
                f"(base=0x{self._arena.base_ptr:x} bytes={self._arena.size} dev={device_id})"
            )
        self._registered = True
        logger.info(
            f"bounce_v2: arena ready base=0x{self._arena.base_ptr:x} "
            f"bytes={self._arena.size} fabric={self._arena.is_fabric} dev={device_id}"
        )

    @property
    def base_ptr(self) -> int:
        return int(self._arena.base_ptr)

    @property
    def size(self) -> int:
        return int(self._arena.size)

    @property
    def is_fabric(self) -> bool:
        return bool(self._arena.is_fabric)

    def close(self) -> None:
        """Deregister from NIXL (idempotent). Call BEFORE the agent shuts
        down and AFTER no transfer can still target the arena."""
        if not self._registered:
            return
        self._registered = False
        try:
            self._agent.deregister_region(self._arena.base_ptr, self._arena.size, self._device_id)
        except RuntimeError as e:
            logger.warning(f"bounce_v2: arena deregistration failed: {e}")
