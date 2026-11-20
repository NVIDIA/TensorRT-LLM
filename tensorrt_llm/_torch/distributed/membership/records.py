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
"""Membership records, rank reports and rank masks for Wide-EP fault tolerance.

Every rank sends a :class:`RankReport` to the membership authority every ``REPORT_INTERVAL_S``. The
authority publishes each decision as a :class:`MembershipRecord` to every rank. A rank mask is an int
whose bit ``r`` is set when it holds rank ``r``.
"""

import enum
import os
from collections.abc import Iterable, Mapping
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from tensorrt_llm._wide_ep_fault_tolerance import RENDEZVOUS_ENV, SHARED_SECRET_ENV
from tensorrt_llm.llmapi.utils import _set_affinity_all_threads

# The active mask has room for as many ranks as the all-to-all kernels support.
MAX_RANKS = 256

# The membership's timing. A report interval is shorter than a heartbeat interval, which is shorter
# than the silence timeout, which is shorter than the hang timeout. The admission timeout is shorter
# than the hang timeout too. The all-to-all kernels must wait at least twice the hang timeout.

# Long enough that a slow but healthy host step is not taken as a hang. A hang found during a
# membership change can take a second timeout to remove, and twice this stays within
# the default 300 s executor hang detector and all-to-all timeouts.
HANG_TIMEOUT_S = 60.0
REPORT_INTERVAL_S = 0.5
# The authority sends an empty line to a rank that it has sent nothing for this long. A rank then
# hears from it at least every heartbeat plus report interval, so it outlasts an authority stall of
# the silence timeout less both (28.5 s).
HEARTBEAT_INTERVAL_S = 1.0
# The age at which a rank takes the authority as silent. It is half the hang timeout, as the
# authority takes a rank's reports as stale after half of its own hang timeout.
AUTHORITY_SILENCE_TIMEOUT_S = HANG_TIMEOUT_S / 2
# A joining rank that makes no progress in its admission for this long is left out, so a stalled
# joiner pauses the survivors far less than a hang timeout.
ADMISSION_TIMEOUT_S = 20.0
# Longer than a network stall that TCP recovers from. A peer that stays connected but stops making
# progress is bounded by the hang timeout instead.
DEAD_TIMEOUT_S = 60.0
HELLO_TIMEOUT_S = 10.0
RENDEZVOUS_TIMEOUT_S = 600.0
# Node supervisors start with the job and connect within seconds of the authority, so one still
# missing after this is a launch mistake.
NODE_SUPERVISOR_TIMEOUT_S = 60.0

NODE_SUPERVISOR_LAUNCH = (
    "mpirun --enable-recovery -np <nodes> --map-by ppr:1:node --bind-to none "
    f"-x {SHARED_SECRET_ENV} -x {RENDEZVOUS_ENV} trtllm-llmapi-launch <command>"
)
# A process started outside the launcher's job must not inherit these.
_LAUNCHER_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_")
# The deployment's opt-in to running Open MPI as root, which a rank keeps.
_RUN_AS_ROOT_VARS = (
    "OMPI_ALLOW_RUN_AS_ROOT",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
    "PRTE_ALLOW_RUN_AS_ROOT",
    "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM",
)
# Open MPI's shared-memory transport, named vader before Open MPI 5, which still accepts that name.
_SHARED_MEMORY_BTLS = ("sm", "vader")


def launcher_free_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return ``environ`` without the launcher's variables, for a process that runs no MPI."""
    return {
        name: value
        for name, value in environ.items()
        if not name.startswith(_LAUNCHER_ENV_PREFIXES)
    }


def remove_cpu_affinity_constraints() -> None:
    """Let this process run on every CPU unless ``TLLM_NUMA_AWARE_WORKER_AFFINITY`` opts out.

    As ``configure_cpu_affinity`` does for a worker, override the inherited binding when the variable
    is unset or ``1``, and keep it for any other value. A process that uses no GPU has no NUMA node to
    pick, so it runs on every CPU. Without this, a process inherits its parent's binding, which a
    launcher can set to the one CPU that rank 0 starts on.
    """
    if os.environ.get("TLLM_NUMA_AWARE_WORKER_AFFINITY") in (None, "1"):
        _set_affinity_all_threads(list(range(os.cpu_count())))


def rank_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return ``environ`` for a rank, a one-process MPI world started outside the launcher's job.

    Keep the opt-in to running as root, and drop the shared-memory ``btl``, which has no peer in a
    one-process world and leaves a /dev/shm segment behind after a kill.
    """
    env = launcher_free_env(environ)
    env.update((name, environ[name]) for name in _RUN_AS_ROOT_VARS if name in environ)
    btl = environ.get("OMPI_MCA_btl", "")
    if btl.startswith("^"):
        excluded = btl[1:].split(",")
        env["OMPI_MCA_btl"] = btl if "sm" in excluded else f"{btl},sm"
    else:
        included = [name for name in btl.split(",") if name and name not in _SHARED_MEMORY_BTLS]
        env["OMPI_MCA_btl"] = ",".join(included) or "^sm"
    return env


def all_ranks(nranks: int) -> int:
    """Return the rank mask of ranks ``0`` to ``nranks - 1``."""
    return (1 << nranks) - 1


def has_rank(mask: int, rank: int) -> bool:
    """Return whether rank mask ``mask`` holds ``rank``."""
    return bool(mask >> rank & 1)


def rank_mask(ranks: Iterable[int], nranks: int) -> int:
    """Return the rank mask of ``ranks``, which must be ranks ``0`` to ``nranks - 1``.

    Raises:
        ValueError: A rank is outside ``[0, nranks)``.
    """
    mask = 0
    for rank in ranks:
        if not 0 <= rank < nranks:
            raise ValueError(f"rank must be in [0, {nranks}), got {rank}")
        mask |= 1 << rank
    return mask


# The host collectives over MNNVL memory number their rounds with an epoch in [1, _EPOCHS]. Epoch 0
# is never sent, because the buffers start zeroed. A round uses the buffer half that its epoch's
# parity selects, so the epoch wraps from the even _EPOCHS to 1 to keep the parity alternating.
_EPOCHS = 0xFFFFFFFE


def next_epoch(epoch: int) -> int:
    """Return the host-exchange epoch after ``epoch``."""
    return epoch % _EPOCHS + 1


def latest_epoch(epochs: list[int]) -> int:
    """Return the latest of ``epochs``. They must lie within half the epoch range of each other.

    The epochs compare on the circle relative to ``epochs[0]``, so 1 counts as later than
    ``_EPOCHS``.
    """
    return max(epochs, key=lambda epoch: (epoch - epochs[0] + _EPOCHS // 2) % _EPOCHS)


class MembershipState(enum.IntEnum):
    """The state of a membership record.

    STARTING until every rank started, then COMMITTED while the ranks serve. A change goes through
    PROPOSE (every active rank pauses and replies READY) and INSTALL (every active rank applies the
    change and replies INSTALLED) back to COMMITTED. ABORTED ends the deployment with an error.
    SHUTDOWN is a normal exit, in which a closed connection is not a failure.
    """

    COMMITTED = 0
    PROPOSE = 1
    INSTALL = 2
    ABORTED = 3
    SHUTDOWN = 4
    STARTING = 5


class Reply(enum.IntEnum):
    """A rank's reply to the current PROPOSE (READY) or INSTALL (INSTALLED), or FAILED."""

    NONE = 0
    READY = 1
    INSTALLED = 2
    FAILED = 3


class _Message(BaseModel):
    """A membership protocol message, sent as JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ResumeState(_Message):
    """Where every rank continues after a membership change."""

    iteration: int = 0
    # The host-collective epoch that every rank resumes from. Communication backends derive their
    # per-round counters from it, so all ranks agree after a change.
    epoch: int = 0


class MembershipRecord(_Message):
    """A membership decision by the authority.

    Attributes:
        generation: Membership change counter, incremented by each PROPOSE.
        state: The change's state.
        active_mask: The ranks in the membership.
        resume: Where the ranks resume, set by the INSTALL of a change.
        incarnations: Current process per rank. 0 is the original, n the n-th replacement.
        joining_mask: The active ranks that are replacements being added by this change.
        given_up_mask: The ranks that no replacement is started for, as their node supervisor gave
            them up or left.
        reason: Why a change left a rank out, or why the deployment aborted.
    """

    generation: int
    state: MembershipState
    active_mask: int
    resume: ResumeState = ResumeState()
    incarnations: tuple[int, ...] = ()
    joining_mask: int = 0
    given_up_mask: int = 0
    reason: str = ""

    def is_active(self, rank: int) -> bool:
        """Return whether ``rank`` is in the membership, whichever process serves it."""
        return has_rank(self.active_mask, rank)

    def is_joining(self, rank: int) -> bool:
        """Return whether ``rank`` is a replacement that this change adds."""
        return has_rank(self.joining_mask, rank)

    def is_given_up(self, rank: int) -> bool:
        """Return whether no replacement of ``rank`` is started."""
        return has_rank(self.given_up_mask, rank)

    def is_member(self, rank: int, incarnation: int) -> bool:
        """Return whether ``rank`` is active with ``incarnation`` as its current process."""
        current = self.incarnations[rank] if rank < len(self.incarnations) else 0
        return self.is_active(rank) and current == incarnation

    def admits(self, rank: int, incarnation: int) -> bool:
        """Return whether the process ``incarnation`` of ``rank`` is in a committed membership."""
        return self.state == MembershipState.COMMITTED and self.is_member(rank, incarnation)


class NodeView(_Message):
    """What the authority sends a node supervisor."""

    record: MembershipRecord
    # The ranks whose replacement made no startup progress for the rendezvous timeout.
    stalled_replacements: tuple[int, ...] = ()
    # Rank mask of the ranks whose current process has a liveness connection to the authority.
    connected_mask: int = 0


class NodeReport(_Message):
    """What a node supervisor last reported to the authority.

    Attributes:
        given_up: The node's ranks that the supervisor gave up.
        paused_mask: Rank mask of the node's ranks whose hang detector is paused and whose process
            is not stopped.
        received_s: When the authority received the report, on its monotonic clock.
    """

    given_up: tuple[int, ...] = ()
    paused_mask: int = 0
    received_s: float = 0.0


class RankReport(_Message):
    """What one rank last reported to the authority.

    Attributes:
        progress: The executor's iteration, or during the startup the number of startup
            collectives the rank completed. A replacement also counts the steps of its autotuner
            warmup.
        waiting_for: Rank mask of peers this rank currently waits on in a collective.
        reply_generation: The generation that ``reply`` answers.
        reply: The rank's reply to that record.
        paused_at: The iteration and epoch at which the rank paused for the current change.
        incarnation: The rank's process, set by the authority from the connection's hello.
        received_s: When the authority received the report, on its monotonic clock. Reports
            that stop arriving tell the authority that the process stopped.
        waited_flags: (peer, round flag) for each all-to-all round that the rank waits on, so a
            peer that still delivers new rounds is not hung.
        waited_epochs: (peer, epoch) for each host-collective round that the rank waits on.
        install_steps: Completed steps in the current INSTALL. Each one counts as progress.
        joining_ready: A replacement is ready to join. It counts as its READY to the change that
            adds it.
        started: The rank finished its startup.
        shutdown: Rank 0 shuts the deployment down.
    """

    progress: int = 0
    waiting_for: int = 0
    reply_generation: int = 0
    reply: Reply = Reply.NONE
    paused_at: ResumeState = ResumeState()
    incarnation: int = 0
    received_s: float = 0.0
    waited_flags: tuple[tuple[int, int], ...] = ()
    waited_epochs: tuple[tuple[int, int], ...] = ()
    install_steps: int = 0
    joining_ready: bool = False
    started: bool = False
    shutdown: bool = False


class TransactionSummary(_Message):
    """An ended membership transaction, with its milestones in Unix milliseconds."""

    generation: int
    outcome: Literal["COMMITTED", "ABORTED"]
    removed_ranks: tuple[int, ...]
    # Why each removed rank was removed: "disconnected" (its liveness connection closed) or "hung".
    causes: dict[int, str] = {}
    added_ranks: tuple[int, ...]
    # When the first removed rank stalled or disconnected, or when the first replacement was ready.
    detected_unix_ms: float
    began_unix_ms: float
    ready_unix_ms: Optional[float] = None
    ended_unix_ms: float
    # Ended transactions per kind (failover, admission or abort), this one included.
    changes: dict[str, int] = {}
    # Why the deployment aborted, for an aborted transaction.
    reason: str = ""


def message_bytes(message: _Message) -> bytes:
    """Return ``message`` as JSON bytes."""
    return message.model_dump_json().encode()
