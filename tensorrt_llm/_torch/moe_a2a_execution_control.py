# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch


def validate_execution_mode(rank_mask_enabled: bool, can_use_cft: bool) -> None:
    """Reject a transport mode without cooperative execution-abort support.

    Args:
        rank_mask_enabled: Whether this communicator uses WideEP FT kernels.
        can_use_cft: Whether dispatch or combine may select counted writes.

    Raises:
        RuntimeError: If the FT communicator can select the unqualified CFT path.
    """
    if rank_mask_enabled and can_use_cft:
        raise RuntimeError(
            "WideEP FT execution abort currently supports only non-CFT transport; "
            "disable CFT explicitly before constructing a rank-mask communicator"
        )


@dataclass(frozen=True)
class MoeA2AExecutionAbortStatus:
    """First recoverable failure observed by an NVLinkOneSided kernel."""

    execution_epoch: int
    phase: str
    reason: str
    waiting_peer: int | None
    raw_status: int

    @classmethod
    def from_raw(cls, raw_status: int) -> MoeA2AExecutionAbortStatus | None:
        """Decode the first failure without inferring completion or quiescence.

        Args:
            raw_status: Packed nonnegative status word returned by the native op.

        Returns:
            Decoded diagnostic evidence, or None when no failure was observed.
        """
        if raw_status == 0:
            return None

        reason_code = raw_status & 0xFF
        phase_code = (raw_status >> 8) & 0xFF
        peer_code = (raw_status >> 16) & 0x1FF
        execution_epoch = (raw_status >> 25) & ((1 << 38) - 1)
        phase = {1: "dispatch", 2: "combine"}.get(phase_code, f"unknown({phase_code})")
        reason = {1: "host_requested", 2: "timeout"}.get(reason_code, f"unknown({reason_code})")
        return cls(
            execution_epoch=execution_epoch,
            phase=phase,
            reason=reason,
            waiting_peer=peer_code - 1 if peer_code else None,
            raw_status=raw_status,
        )


class MoeA2AExecutionControl:
    """Stream-independent abort token shared by dispatch and combine.

    ``request_abort`` only invalidates the current execution epoch. It does not
    change EP membership, the active-rank mask, or the committed membership
    generation. ``begin_epoch`` must be called by the recovery coordinator only
    after it has stopped new launch admission and all work tagged with the old
    epoch has quiesced. This primitive does not provide the admission gate.
    Communication wrappers expose recoverable abort only in the opt-in
    rank-mask/FT specialization; ordinary MoE retains its fail-stop timeout.
    Completion flags are progress only. A local CUDA stream drain is not proof
    that a failed peer can no longer write to reusable storage.
    """

    def __init__(self, workspace: torch.Tensor, ep_rank: int) -> None:
        """Register a mapped-host token tied to one rank's CUDA workspace.

        Args:
            workspace: Two-dimensional uint8 CUDA symmetric workspace tensor.
            ep_rank: Local logical rank indexing the workspace's first dimension.
        """
        self._lock = Lock()
        self._closed = False
        self._workspace = workspace
        self._ep_rank = ep_rank
        tensor = torch.ops.trtllm.moe_a2a_create_execution_control(workspace, ep_rank)
        try:
            live_epoch, _ = torch.ops.trtllm.moe_a2a_get_execution_abort_state(tensor)
        except Exception:
            torch.ops.trtllm.moe_a2a_release_execution_control(tensor)
            raise
        self._tensor: torch.Tensor | None = tensor
        self._expected_epoch = int(live_epoch)
        self._has_acknowledged_abort = False

    def _check_open(self) -> None:
        """Reject use after the mapped control registration has been released."""
        if self._closed:
            raise RuntimeError("MoE A2A execution control has been released")

    @property
    def tensor(self) -> torch.Tensor:
        """Mapped host allocation passed to the CUDA kernels."""
        self._check_open()
        assert self._tensor is not None
        return self._tensor

    def capture_epoch(self) -> int:
        """Capture the epoch that both halves of one dispatch/combine pair use."""
        with self._lock:
            self._check_open()
            return self._expected_epoch

    def request_abort(self) -> int:
        """Invalidate in-flight work without enqueueing anything on a CUDA stream."""
        with self._lock:
            self._check_open()
            return int(torch.ops.trtllm.moe_a2a_request_execution_abort(self.tensor))

    def requested_epoch(self) -> int:
        """Read the latest host-requested epoch without admitting new execution."""
        with self._lock:
            self._check_open()
            live_epoch, _ = torch.ops.trtllm.moe_a2a_get_execution_abort_state(self.tensor)
            return int(live_epoch)

    def status(self) -> MoeA2AExecutionAbortStatus | None:
        """Read the first failure; a nonzero status does not imply grid quiescence."""
        with self._lock:
            self._check_open()
            _, raw_status = torch.ops.trtllm.moe_a2a_get_execution_abort_state(self.tensor)
            return MoeA2AExecutionAbortStatus.from_raw(int(raw_status))

    def has_pending_abort(self) -> bool:
        """Report invalidated or failed work, without asserting quiescence.

        Checkpoint eligibility must not let workspace reinitialization clear
        device status while its host-side abort remains unacknowledged.
        """
        with self._lock:
            self._check_open()
            live_epoch, raw_status = torch.ops.trtllm.moe_a2a_get_execution_abort_state(self.tensor)
            return int(live_epoch) != self._expected_epoch or int(raw_status) != 0

    def begin_epoch(self, execution_epoch: int | None = None) -> int:
        """Reset local status after the caller establishes transport quiescence.

        The caller must stop launch admission and prove that old local and peer
        writes cannot reach reused storage. This method supplies no such proof.

        Args:
            execution_epoch: Latest requested epoch, or None to read it here.

        Returns:
            Acknowledged execution epoch shared by dispatch and combine.

        Raises:
            ValueError: If the epoch is stale or no new abort was requested.
            RuntimeError: If this control has already been released.
        """
        with self._lock:
            self._check_open()
            live_epoch, raw_status = torch.ops.trtllm.moe_a2a_get_execution_abort_state(self.tensor)
            live_epoch = int(live_epoch)
            if execution_epoch is None:
                execution_epoch = live_epoch
            if execution_epoch != live_epoch:
                raise ValueError(
                    "execution_epoch must match the latest requested epoch "
                    f"({live_epoch}), got {execution_epoch}"
                )
            if execution_epoch == self._expected_epoch and self._has_acknowledged_abort:
                if int(raw_status) != 0:
                    raise ValueError(
                        "begin_epoch cannot reset a kernel abort without a newly requested "
                        "execution epoch; call request_abort() first"
                    )
                # Multiple communication wrappers can share one workspace-owned
                # control. Once one wrapper acknowledges the epoch, subsequent
                # wrapper resets for that epoch are intentionally idempotent.
                return execution_epoch
            if execution_epoch <= self._expected_epoch:
                raise ValueError(
                    "begin_epoch requires a newly requested execution epoch; "
                    f"current={self._expected_epoch}, requested={execution_epoch}"
                )
            torch.ops.trtllm.moe_a2a_begin_execution_epoch(
                self._workspace,
                self._ep_rank,
                self.tensor,
                execution_epoch,
            )
            self._expected_epoch = execution_epoch
            self._has_acknowledged_abort = True
            return execution_epoch

    def close(self) -> None:
        """Release the async-lifetime hold after all kernels have quiesced."""
        with self._lock:
            if self._closed:
                return
            tensor = self.tensor
            torch.ops.trtllm.moe_a2a_release_execution_control(tensor)
            self._tensor = None
            self._closed = True
