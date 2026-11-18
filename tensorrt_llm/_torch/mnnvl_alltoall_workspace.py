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

from typing import Callable, MutableMapping, Protocol, cast
from weakref import WeakSet

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import (
    MnnvlCheckpointCommunicator,
    MnnvlMemory,
    _checkpoint_allgather,
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


class _MnnvlAlltoAllWorkspaceLifecycle:
    """Own checkpoint transitions for one shared MoE workspace.

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
        ep_rank: int,
        ep_size: int,
    ) -> None:
        self._workspace_state = workspace_state
        self._memory = memory
        self._workspace = workspace
        self._metainfo = metainfo
        self._ep_rank = int(ep_rank)
        self._ep_size = int(ep_size)
        self._clients: WeakSet[_WorkspaceClient] = WeakSet()

    @classmethod
    def get_or_create(
        cls,
        *,
        workspace_state: MutableMapping[str, object],
        memory: MnnvlMemory,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> "_MnnvlAlltoAllWorkspaceLifecycle":
        lifecycle = workspace_state.get(_WORKSPACE_LIFECYCLE_KEY)
        if lifecycle is None:
            lifecycle = cls(
                workspace_state=workspace_state,
                memory=memory,
                workspace=workspace,
                metainfo=metainfo,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )
            workspace_state[_WORKSPACE_LIFECYCLE_KEY] = lifecycle
            return lifecycle
        if not isinstance(lifecycle, cls):
            raise TypeError("invalid MNNVL All-to-All workspace lifecycle state")
        lifecycle._validate_shared_context(
            memory=memory,
            workspace=workspace,
            metainfo=metainfo,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        return lifecycle

    @property
    def metainfo(self) -> torch.Tensor:
        return self._metainfo

    def register(self, client: _WorkspaceClient) -> None:
        self._clients.add(client)

    def unregister(self, client: _WorkspaceClient) -> None:
        self._clients.discard(client)

    def checkpoint_prepare(self) -> None:
        """Preflight shared readers, then collectively detach backing handles."""
        if not self._memory.mapped:
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
            raise
        if active_ranks:
            raise RuntimeError(
                f"Cannot checkpoint during an active MoE All-to-All phase on ranks {active_ranks}"
            )
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
                torch.cuda.synchronize()
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
                if local_error is not None:
                    raise local_error
                raise RuntimeError(
                    "MNNVL workspace restore failed on ranks "
                    f"{unready_ranks}; refusing to publish the restored workspace"
                )
            self._memory._checkpoint_restore_complete()
        except Exception:
            self._memory._checkpoint_restore_failed()
            raise

    def _validate_shared_context(
        self,
        *,
        memory: MnnvlMemory,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        if (
            self._memory is not memory
            or self._workspace is not workspace
            or self._metainfo is not metainfo
            or self._ep_rank != ep_rank
            or self._ep_size != ep_size
        ):
            raise ValueError(
                "MNNVL All-to-All wrappers sharing a workspace must use the "
                "same allocation, metadata layout, and rank layout"
            )
