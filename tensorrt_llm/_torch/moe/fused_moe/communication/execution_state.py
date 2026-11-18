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
"""The active-rank mask and abort state that the MoE all-to-all kernels read under Wide-EP fault tolerance.

When a peer stops, ``request_abort`` releases the collectives that wait for it. Then, with the model
idle, ``rearm`` returns the bound workspaces and layers to their first round, and ``install_mask``
writes the new rank mask. Long waits on peers publish the round flag they wait for in the abort
state (``waited_flags``). After each forward, ``check_incomplete_rounds`` marks the abort state
cancelled if a peer may have overwritten rows that the forward's combines read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import internal as _tllm_internal

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.membership import StoreGroup

_MASK_WORD_BITS = 64
_ABORT_REQUESTED = 0
# The words ``requested`` and ``status``, then the round flag awaited from each peer.
_ABORT_WORDS = slice(0, int(_tllm_internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX))
_WAITED_FLAGS = slice(int(_tllm_internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX), None)


def _mask_words(mask: int, ep_rank: int) -> tuple[int, ...]:
    """Split ``mask`` into the uint64 words of the kernels' mask, which must mark ``ep_rank``."""
    word_mask = (1 << _MASK_WORD_BITS) - 1
    words = tuple((mask >> (i * _MASK_WORD_BITS)) & word_mask for i in range(EP_MASK_NUM_WORDS))
    if not (words[ep_rank // _MASK_WORD_BITS] >> (ep_rank % _MASK_WORD_BITS)) & 1:
        raise RuntimeError(f"active_rank_mask must mark the local ep_rank ({ep_rank}) as active")
    return words


class MembershipWorkspace(Protocol):
    """Peer-writable memory that a membership change remaps among the active ranks in place."""

    @property
    def local_device_bytes(self) -> int:
        """Return the device bytes that :meth:`remap` allocates for this rank's row."""

    def rows_needing_fresh_pages(self, comm: StoreGroup) -> frozenset[int]:
        """Return every row if a former member may still write into the workspace, else no row."""

    def remap(self, comm: StoreGroup, fresh_rows: frozenset[int]) -> None:
        """Map the workspace among the active ranks with ``fresh_rows`` on fresh pages."""


class BindableCommunication(Protocol):
    """MoE communication strategy whose collectives can run under a :class:`MoeExecutionState`."""

    def bind_execution_state(self, state: MoeExecutionState) -> None:
        """Run every later collective with ``state``'s rank mask and abort state."""


class MoeExecutionState:
    """Own the active-rank mask and abort state that the kernels of a model's NVLinkOneSided layers read.

    Both stay at one address and change only in place, so CUDA graphs that captured the all-to-all
    kernels follow every mask that :meth:`install_mask` writes and every abort request.

    Args:
        ep_group_health: The EP membership whose active ranks the mask holds.
        ep_rank: This rank's EP rank, which every mask must mark active.
        device: The device of the all-to-all workspace.
    """

    def __init__(self, ep_group_health: EPGroupHealth, ep_rank: int, device: torch.device) -> None:
        self.ep_group_health = ep_group_health
        self.ep_rank = ep_rank
        self._device = device
        # The kernels read the abort state at its host address, which must be pinned memory.
        self.abort = torch.zeros(
            int(_tllm_internal.thop.MOE_A2A_ABORT_STATE_WORDS),
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )
        if not self.abort.is_pinned():
            raise RuntimeError("The MoE All-to-All abort state requires pinned host memory")
        self._abort_words = self.abort.numpy()[_ABORT_WORDS]
        self._workspaces: dict[
            MembershipWorkspace, tuple[Callable[[], None], Callable[[], None], torch.Tensor]
        ] = {}
        self._layer_resets: list[Callable[[], None]] = []
        snapshot = ep_group_health.snapshot()
        self._generation = snapshot.generation
        # A layer bound after a membership change has a workspace that the change's remaps left out.
        self._bound_generation = snapshot.generation
        self.active_rank_mask = torch.tensor(
            _mask_words(snapshot.mask, ep_rank), dtype=torch.uint64, device=device
        )

    def register_layer(self, reset_state: Callable[[], None]) -> None:
        """Register the reset of a bound layer's dispatch/combine state machine.

        Raises:
            RuntimeError: The EP membership changed since this state was created.
        """
        if self.ep_group_health.generation != self._bound_generation:
            raise RuntimeError("Bind MoE layers only before the EP membership changes")
        self._layer_resets.append(reset_state)

    def register_workspace(
        self,
        workspace: MembershipWorkspace,
        reset_workspace: Callable[[], None],
        check_incomplete_round: Callable[[], None],
        rows: torch.Tensor,
    ) -> None:
        """Register a bound layer's workspace, once per workspace that layers share.

        ``reset_workspace`` resets this rank's row of it, ``check_incomplete_round`` enqueues the
        check of this rank's incomplete-round word in it (:meth:`check_incomplete_rounds`), and
        ``rows`` is the workspace's ``[ep_size, bytes]`` uint8 view.
        """
        self._workspaces.setdefault(workspace, (reset_workspace, check_incomplete_round, rows))

    def staging_workspace(self) -> torch.Tensor:
        """Return the rows of the largest bound workspace.

        The workspace is idle from :meth:`remap_workspaces` until :meth:`rearm`, so restores may
        stage expert weights through it.
        """
        return max(
            (rows for _, _, rows in self._workspaces.values()), key=lambda rows: rows.shape[1]
        )

    def workspace_bytes(self) -> int:
        """Return the device memory :meth:`remap_workspaces` allocates on this rank."""
        return sum(workspace.local_device_bytes for workspace in self._workspaces)

    def workspace_rows_needing_fresh_pages(self, comm: StoreGroup) -> frozenset[int]:
        """Return every row if a former member may still write into any bound workspace."""
        return frozenset().union(
            *(workspace.rows_needing_fresh_pages(comm) for workspace in self._workspaces)
        )

    def remap_workspaces(self, comm: StoreGroup, fresh_rows: frozenset[int]) -> None:
        """Map every bound workspace at the same addresses among the active members of ``comm``.

        Call only while no rank is in a collective. Every rank must :meth:`rearm` before the next
        collective.
        """
        for workspace in self._workspaces:
            workspace.remap(comm, fresh_rows)

    def request_abort(self) -> None:
        """Make running and future collectives return without their peers. Safe from any thread."""
        self._abort_words[_ABORT_REQUESTED] = 1

    def aborted(self) -> bool:
        """Return whether an abort was requested or a collective aborted since the last :meth:`rearm`."""
        return bool(self._abort_words.any())

    def check_incomplete_rounds(self) -> None:
        """Enqueue the check of this rank's incomplete-round word in every bound workspace.

        A peer sets the word before it overwrites rows that this rank's combines may have read, and
        the check then marks the abort state cancelled, so :meth:`aborted` holds. Call once per
        forward, on the stream of its last combine, before the event that the host waits on to
        accept the forward.
        """
        for _, check_incomplete_round, _ in self._workspaces.values():
            check_incomplete_round()

    def waited_flags(self) -> dict[int, int]:
        """Return each EP rank that a kernel has long waited on, with the round flag it waits for."""
        # The abort state stores the uint32 flags in int32 words.
        return {
            peer: round_flag & 0xFFFFFFFF
            for peer, round_flag in enumerate(self.abort[_WAITED_FLAGS].tolist())
            if round_flag
        }

    def rearm(self) -> None:
        """Clear the abort state and return every bound workspace and layer to its first round.

        Call only while the model is idle and before :meth:`install_mask`. The device is
        synchronized before and after, so no queued collective sees a partial reset.
        """
        torch.cuda.synchronize(self._device)
        self.abort.zero_()
        for reset_workspace, _, _ in self._workspaces.values():
            reset_workspace()
        for reset_state in self._layer_resets:
            reset_state()
        torch.cuda.synchronize(self._device)

    def install_mask(self) -> None:
        """Write the active ranks of the EP group health into the device mask.

        Call only while no all-to-all collective is queued or running, and after :meth:`rearm` if a
        collective aborted. The device is synchronized first, so no collective reads a partly
        written mask.
        """
        snapshot = self.ep_group_health.snapshot()
        if snapshot.generation == self._generation:
            return
        words = _mask_words(snapshot.mask, self.ep_rank)
        torch.cuda.synchronize(self._device)
        if self.aborted():
            raise RuntimeError(f"MoE All-to-All abort state is {self.abort[_ABORT_WORDS].tolist()}")
        self.active_rank_mask.copy_(torch.tensor(words, dtype=torch.uint64))
        self._generation = snapshot.generation
