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

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Protocol, cast
from weakref import WeakSet

import torch

from tensorrt_llm._mnnvl_utils import (
    MnnvlCheckpointCommunicator,
    MnnvlMemory,
    _checkpoint_allgather,
)
from tensorrt_llm._torch.alltoall_watchdog import (
    AlltoAllWatchdog,
    AlltoAllWatchdogCoordinator,
    AlltoAllWatchdogTimeout,
    EPGroupHealthLike,
)

_WORKSPACE_LIFECYCLE_KEY = "mnnvl_alltoall_workspace_lifecycle"


def _collect_active_ranks(
    comm: MnnvlCheckpointCommunicator,
    *,
    local_clients_idle: bool,
    expected_size: int,
) -> list[int]:
    """Collectively return ranks whose local workspace clients are active."""
    clients_idle_by_rank = _checkpoint_allgather(
        comm,
        local_clients_idle,
        operation="workspace idle readiness",
    )
    if len(clients_idle_by_rank) != expected_size:
        raise RuntimeError(
            "MNNVL workspace communicator size does not match the MoE EP group: "
            f"{len(clients_idle_by_rank)} != {expected_size}"
        )
    return [rank for rank, clients_idle in enumerate(clients_idle_by_rank) if not clients_idle]


def _collect_unready_ranks(
    comm: MnnvlCheckpointCommunicator,
    *,
    local_ready: bool,
    expected_size: int,
) -> list[int]:
    """Collectively return ranks that could not finish local restore work."""
    ready_by_rank = _checkpoint_allgather(
        comm,
        local_ready,
        operation="frontend restore readiness",
    )
    if len(ready_by_rank) != expected_size:
        raise RuntimeError(
            "MNNVL workspace communicator size does not match the MoE EP group: "
            f"{len(ready_by_rank)} != {expected_size}"
        )
    return [rank for rank, ready in enumerate(ready_by_rank) if not ready]


class _WorkspaceClient(Protocol):
    def _mnnvl_checkpoint_is_idle(self) -> bool:
        """Return whether this client can safely enter a checkpoint."""
        ...

    def _mnnvl_checkpoint_reset(self) -> None:
        """Reset frontend protocol state after a successful restore."""


@dataclass(frozen=True)
class _WatchdogConfig:
    ep_size: int
    timeout_s: float
    poll_interval_s: float
    on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None


class _MnnvlAlltoAllWorkspaceLifecycle:
    """Own checkpoint and watchdog transitions for one shared MoE workspace.

    ``PyExecutor`` invokes this resource hook from the engine sleep/wakeup
    PREPARE/COMMIT/ABORT control path. That coordinator stops admission, drains
    in-flight work, aggregates bounded per-rank results, and reopens admission
    only after every rank has completed COMMIT. The local client and subgroup
    idle checks here remain defense-in-depth validation while the executor is
    parked.
    """

    def __init__(
        self,
        *,
        workspace_state: MutableMapping[str, object],
        memory: MnnvlMemory,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        metainfo_index: Mapping[str, int],
        ep_rank: int,
        ep_size: int,
        health: EPGroupHealthLike | None,
    ) -> None:
        self._workspace_state = workspace_state
        self._memory = memory
        self._workspace = workspace
        self._metainfo = metainfo
        self._metainfo_index = dict(metainfo_index)
        self._ep_rank = int(ep_rank)
        self._ep_size = int(ep_size)
        self._health = health
        self._clients: WeakSet[_WorkspaceClient] = WeakSet()
        self._watchdog_clients: WeakSet[_WorkspaceClient] = WeakSet()
        self._watchdog_config: _WatchdogConfig | None = None
        self._watchdog: AlltoAllWatchdog | None = None
        self._coordinator = self._create_coordinator()

    @classmethod
    def get_or_create(
        cls,
        *,
        workspace_state: MutableMapping[str, object],
        memory: MnnvlMemory,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        metainfo_index: Mapping[str, int],
        ep_rank: int,
        ep_size: int,
        health: EPGroupHealthLike | None,
    ) -> "_MnnvlAlltoAllWorkspaceLifecycle":
        lifecycle = workspace_state.get(_WORKSPACE_LIFECYCLE_KEY)
        if lifecycle is None:
            lifecycle = cls(
                workspace_state=workspace_state,
                memory=memory,
                workspace=workspace,
                metainfo=metainfo,
                metainfo_index=metainfo_index,
                ep_rank=ep_rank,
                ep_size=ep_size,
                health=health,
            )
            workspace_state[_WORKSPACE_LIFECYCLE_KEY] = lifecycle
            return lifecycle
        if not isinstance(lifecycle, cls):
            raise TypeError("invalid MNNVL All-to-All workspace lifecycle state")
        lifecycle._validate_shared_context(
            memory=memory,
            workspace=workspace,
            metainfo=metainfo,
            metainfo_index=metainfo_index,
            ep_rank=ep_rank,
            ep_size=ep_size,
            health=health,
        )
        return lifecycle

    @property
    def metainfo(self) -> torch.Tensor:
        return self._metainfo

    @property
    def coordinator(self) -> AlltoAllWatchdogCoordinator:
        return self._coordinator

    def watchdog_for(self, client: _WorkspaceClient) -> AlltoAllWatchdog | None:
        if client not in self._watchdog_clients:
            return None
        return self._watchdog

    def register(
        self,
        client: _WorkspaceClient,
        *,
        watchdog_timeout_s: float | None,
        watchdog_poll_interval_s: float,
        watchdog_on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None,
    ) -> None:
        if client in self._clients:
            return
        if watchdog_timeout_s is None:
            self._clients.add(client)
            return

        config = _WatchdogConfig(
            ep_size=self._ep_size,
            timeout_s=float(watchdog_timeout_s),
            poll_interval_s=float(watchdog_poll_interval_s),
            on_timeout=watchdog_on_timeout,
        )
        self._validate_watchdog_config(config)
        self._clients.add(client)
        self._watchdog_clients.add(client)
        try:
            if self._memory.mapped:
                self._start_watchdog()
        except Exception:
            self._watchdog_clients.discard(client)
            self._clients.discard(client)
            if not self._watchdog_clients:
                self._watchdog_config = None
            raise

    def unregister(self, client: _WorkspaceClient) -> None:
        self._clients.discard(client)
        self._watchdog_clients.discard(client)
        if not self._watchdog_clients:
            self._stop_watchdog()
            self._watchdog_config = None

    def checkpoint_prepare(self) -> None:
        """Preflight shared readers, then collectively detach backing handles."""
        if not self._memory.mapped:
            self._stop_watchdog()
            self._memory.checkpoint_prepare()
            return
        local_clients_idle = all(
            client._mnnvl_checkpoint_is_idle() for client in list(self._clients)
        )
        comm = cast(MnnvlCheckpointCommunicator | None, self._memory.comm)
        if comm is None:
            raise RuntimeError("MNNVL workspace communicator is not initialized")
        try:
            active_ranks = _collect_active_ranks(
                comm,
                local_clients_idle=local_clients_idle,
                expected_size=self._ep_size,
            )
        except TimeoutError:
            self._memory.checkpoint_fail_closed()
            self._stop_watchdog()
            raise
        if active_ranks:
            raise RuntimeError(
                f"Cannot checkpoint during an active MoE All-to-All phase on ranks {active_ranks}"
            )
        self._stop_watchdog()
        self._memory.checkpoint_prepare()

    def checkpoint_restore(
        self,
        comm: MnnvlCheckpointCommunicator,
        initialize_frontend: Callable[[], torch.Tensor],
    ) -> None:
        """Restore backing handles and publish the workspace after frontend readiness."""
        if self._memory.mapped:
            return
        restored = self._memory.checkpoint_restore(comm)
        if restored is False:
            return
        local_error: Exception | None = None
        try:
            try:
                refreshed_metainfo = initialize_frontend()
                if not torch.equal(refreshed_metainfo, self._metainfo):
                    raise RuntimeError(
                        "MoE All-to-All metainfo changed during MNNVL restore; "
                        "captured CUDA graphs cannot be replayed safely"
                    )
                self._metainfo = refreshed_metainfo
                self._workspace_state["metainfo"] = refreshed_metainfo
                self._coordinator = self._create_coordinator()
                torch.cuda.synchronize()
                self._start_watchdog()
                for client in list(self._clients):
                    client._mnnvl_checkpoint_reset()
            except Exception as error:
                local_error = error

            unready_ranks = _collect_unready_ranks(
                comm,
                local_ready=local_error is None,
                expected_size=self._ep_size,
            )
            if unready_ranks:
                self._stop_watchdog()
                if local_error is not None:
                    raise local_error
                raise RuntimeError(
                    "MNNVL workspace restore failed on ranks "
                    f"{unready_ranks}; refusing to publish the restored workspace"
                )
            self._memory._checkpoint_restore_complete()
        except Exception:
            self._memory._checkpoint_restore_failed()
            self._stop_watchdog()
            raise

    def _create_coordinator(self) -> AlltoAllWatchdogCoordinator:
        return AlltoAllWatchdogCoordinator(
            workspace_state=self._workspace_state,
            workspace=self._workspace,
            metainfo=self._metainfo,
            metainfo_index=self._metainfo_index,
            ep_rank=self._ep_rank,
            health=self._health,
        )

    def _start_watchdog(self) -> None:
        config = self._watchdog_config
        if config is None or not self._watchdog_clients or self._watchdog is not None:
            return
        self._watchdog = self._coordinator.acquire_watchdog(
            ep_size=config.ep_size,
            timeout_s=config.timeout_s,
            poll_interval_s=config.poll_interval_s,
            on_timeout=config.on_timeout,
        )

    def _stop_watchdog(self) -> None:
        if self._watchdog is None:
            return
        self._coordinator.release_watchdog(self._watchdog)
        self._watchdog = None

    def _validate_shared_context(
        self,
        *,
        memory: MnnvlMemory,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        metainfo_index: Mapping[str, int],
        ep_rank: int,
        ep_size: int,
        health: EPGroupHealthLike | None,
    ) -> None:
        if (
            self._memory is not memory
            or self._workspace is not workspace
            or self._metainfo is not metainfo
            or self._metainfo_index != dict(metainfo_index)
            or self._ep_rank != ep_rank
            or self._ep_size != ep_size
        ):
            raise ValueError(
                "MNNVL All-to-All wrappers sharing a workspace must use the "
                "same allocation, metadata layout, and rank layout"
            )
        if self._health is health:
            return
        if self._clients or self._watchdog is not None or self._watchdog_config is not None:
            raise ValueError(
                "MNNVL All-to-All wrappers sharing a workspace must use the same EP health object"
            )
        self._health = health
        self._coordinator = self._create_coordinator()

    def _validate_watchdog_config(self, requested: _WatchdogConfig) -> None:
        existing = self._watchdog_config
        if existing is None:
            self._watchdog_config = requested
            return
        if (
            existing.ep_size != requested.ep_size
            or existing.timeout_s != requested.timeout_s
            or existing.poll_interval_s != requested.poll_interval_s
            or existing.on_timeout is not requested.on_timeout
        ):
            raise ValueError(
                "MNNVL All-to-All wrappers sharing a workspace must use the "
                "same watchdog configuration"
            )
