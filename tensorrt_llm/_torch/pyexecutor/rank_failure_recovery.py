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
"""Keep a Wide-EP deployment serving when a rank fails, and let a replacement rank rejoin.

Every rank runs a RankFailureRecovery, which takes part in each membership change that the
authority process drives with PROPOSE, INSTALL and COMMITTED records. The ranks pause the forward,
install the new membership in lockstep, restore the lost experts and resume serving.
"""

from __future__ import annotations

import atexit
import contextlib
import enum
import os
import pickle  # nosec B403
import subprocess
import threading
import time
import traceback
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch

from tensorrt_llm._wide_ep_fault_tolerance import check_fault_tolerance_requirements
from tensorrt_llm.executor.rank_failure_requests import RankFailureError
from tensorrt_llm.logger import logger

from ..distributed import Distributed
from ..distributed.membership import (
    AUTHORITY_SILENCE_TIMEOUT_S,
    RENDEZVOUS_ENV,
    REPORT_INTERVAL_S,
    MembershipClient,
    MembershipRecord,
    MembershipState,
    Reply,
    ResumeState,
    SignedStore,
    StoreGroup,
    next_epoch,
    open_pause_word,
    pause_word_path,
    rank_mask,
    rendezvous_store,
    wait_for_keys,
)
from ..distributed.membership.startup import (
    authority_process,
    end_process,
    membership_client,
    publish_startup_record,
    publish_tuned_tactics,
    wait_for_every_rank,
)
from ..distributed.mnnvl_memory import MnnvlMappingError, MnnvlPeerHandleError
from ..moe.fused_moe.eplb_expert_restore import ExpertRestorePlan, ExpertRestorePlanMismatchError
from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from .kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .rank_failure_memory import (
    KvCacheShrinkError,
    RecoveryMemory,
    StartupMemory,
    check_largest_failure_fits,
)
from .rank_failure_model_load import moe_comms
from .resource_manager import KVCacheManager

if TYPE_CHECKING:
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from .llm_request import LlmRequest, PyResult
    from .py_executor import BatchState, PyExecutor
    from .scheduler import ScheduledRequests
    from .scheduler.adp_router import RankState
    from .scheduler.waiting_queue import WaitingQueue

_AUTHORITY_TIMEOUT_S = 10.0

# The context reuse commits of a step, held until the step is accepted.
ContextCommits = List[Tuple["KVCacheManagerV2", Dict[int, int]]]
# Where the stored outputs of a step's requests end, rewound if the step is discarded.
OutputsMarks = List[Tuple["LlmRequest", "PyResult.OutputsMark"]]


class _MembershipRestarted(Exception):
    """A newer membership change replaced the one being installed."""


class _ExpertsNotRestorable(Exception):
    """The active ranks cannot hold, load or agree on every expert after a membership change."""


class _Phase(enum.Enum):
    """Where a rank is in the membership changes.

    A record for a change that the rank has not installed can arrive in any phase. It gets no
    phase, because the install in progress still runs (see ``_change_not_installed``).
    """

    # No change is installed. The forward runs unless the rank has a change it has not installed.
    SERVING = enum.auto()
    # A forward vote paused every rank's forward for a change. The rank stops in its next host
    # exchange round once it has the change's record.
    STOPPED = enum.auto()
    # The host exchange runs over the installed record's members. The forward stays paused.
    INSTALLING = enum.auto()
    # Past the resume iteration. The rank finishes the install and waits for COMMITTED.
    RESUMING = enum.auto()
    # A replacement that no COMMITTED record has admitted yet.
    JOINING = enum.auto()


# The phases each phase may move to. Only the client thread moves JOINING to RESUMING.
_PHASE_TRANSITIONS = {
    _Phase.SERVING: {_Phase.STOPPED, _Phase.INSTALLING},  # forward vote, host exchange install
    _Phase.STOPPED: {_Phase.SERVING, _Phase.INSTALLING},  # forward vote, host exchange install
    _Phase.INSTALLING: {_Phase.INSTALLING, _Phase.RESUMING},  # newer change, resume iteration
    _Phase.RESUMING: {_Phase.INSTALLING, _Phase.SERVING},  # newer change, COMMITTED
    _Phase.JOINING: {_Phase.RESUMING},  # COMMITTED admits this rank
}
# The phases from an install until its commit, in which no forward runs.
_INSTALL_IN_PROGRESS = frozenset({_Phase.INSTALLING, _Phase.RESUMING, _Phase.JOINING})


def _moe_load_balancer_of(executor: PyExecutor):
    balancer = executor.model_engine.moe_load_balancer
    return balancer if isinstance(balancer, MoeLoadBalancer) else None


def _admission_epoch_key(generation: int) -> str:
    return f"host_exchange_epoch/{generation}"


class _PhaseTimer:
    """Record how long each membership change phase takes, in milliseconds."""

    def __init__(self):
        self.times_ms: Dict[str, float] = {}
        self._began_s = time.perf_counter()

    def begin(self) -> None:
        self._began_s = time.perf_counter()

    def end(self, phase: str) -> None:
        """Record the time since the last ``begin`` or ``end`` as ``phase``."""
        now_s = time.perf_counter()
        self.times_ms[phase] = (now_s - self._began_s) * 1e3
        self._began_s = now_s

    def summary(self) -> str:
        return ", ".join(f"{phase} {ms:.0f} ms" for phase, ms in self.times_ms.items())


# Per MoE layer, the expert in each of one rank's slots (-1 if empty).
OwnSlots = List[List[int]]


def agreed_placement(
    own_slots: List[Optional[OwnSlots]], record: MembershipRecord
) -> List[np.ndarray]:
    """Return the per-layer placement that every active rank plans the restore from.

    Row r holds the slots that rank r gathered once it abandoned its EPLB iterations, because the
    ranks' views can differ after an interrupted EPLB update or restore. An inactive rank's slots
    are empty (-1).

    Raises:
        _MembershipRestarted: An active rank's slots are missing, which only a newer change causes.
    """
    active = [r for r in range(len(own_slots)) if record.is_active(r)]
    if any(own_slots[r] is None for r in active):
        raise _MembershipRestarted
    return [
        np.array(
            [
                own_slots[r][layer] if record.is_active(r) else [-1] * len(slots)
                for r in range(len(own_slots))
            ],
            dtype=np.int32,
        )
        for layer, slots in enumerate(own_slots[active[0]])
    ]


class RankFailureRecovery:
    """Take part in every membership change on one rank, so that serving survives a rank failure.

    The membership client thread calls ``_on_record``, ``_on_tick`` and ``_on_client_error``.
    Every other method runs on the executor thread. Both threads read and write ``_record``,
    ``_installed`` and ``_phase`` under ``_record_changed``. The client thread makes one phase
    transition, JOINING to RESUMING when a COMMITTED record admits this replacement. The
    executor thread makes every other one. Only the client thread writes ``_record`` and leaves
    JOINING, so it reads both without the lock.
    """

    def __init__(
        self,
        executor: PyExecutor,
        exchange,
        moe_state: MoeExecutionState,
        store: SignedStore,
        startup_memory: StartupMemory,
        authority: Optional[subprocess.Popen] = None,
        incarnation: int = 0,
    ):
        self._executor = executor
        self._exchange = exchange
        self._moe = moe_state
        self._store = store
        self._memory = RecoveryMemory(executor, moe_state, startup_memory)
        self._authority = authority
        self.rank = executor.dist.rank
        self.world_size = executor.dist.world_size
        self.incarnation = incarnation
        # On rank 0, maps request id to client id for the requests without a final response.
        # The executor worker attaches it before it reports READY, so before any request arrives.
        self._outstanding_requests: Optional[Callable[[], Dict[int, int]]] = None
        self._client: Optional[MembershipClient] = None
        self._record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, active_mask=(1 << self.world_size) - 1
        )
        self._record_changed = threading.Condition()
        self._phase = _Phase.SERVING if incarnation == 0 else _Phase.JOINING
        self._committed_mask = self._record.active_mask
        self._committed_generation = self._record.generation
        # The last record whose host exchange rows this rank mapped.
        self._installed: Optional[MembershipRecord] = None
        self._waiting_in_store = 0
        # Mask of the peers whose MNNVL handles the active ranks could not import.
        self._unimportable_peers = 0
        self._install_steps = 0
        self._published_admissions: List[int] = []
        # Store groups of the uncommitted changes, closed at the next commit.
        self._change_groups: List[StoreGroup] = []
        self._interrupted: List[LlmRequest] = []
        # Per request id, the outputs mark at the first step discarded since the last recompute.
        self._discarded_marks: Dict[int, PyResult.OutputsMark] = {}
        exchange.set_interrupt_handler(self._membership_changing, self._on_exchange_interrupt)

    @classmethod
    def create(cls, executor: PyExecutor) -> RankFailureRecovery:
        """Take over the membership this process joined at startup.

        Called once, on the thread that starts the executor, before its worker thread starts. It:

        - closes the startup record, and checks that no torch.distributed group exists;
        - checks that every KV cache manager can defer its reuse commits;
        - on an original rank with an MoE load balancer, checks that the largest failure fits;
        - publishes this rank's tuned tactics and, on rank 0, the startup record, for replacements;
        - on an original rank, reports it started and waits until every rank has;
        - binds the membership client's callbacks to the returned object (:meth:`start`), after
          which the client thread calls ``_on_record``, ``_on_tick`` and ``_on_client_error``;
        - on a replacement, runs the change that admits it, and resumes the executor at the
          iteration where the active ranks resume.

        An original rank returns once every rank has started, a replacement once the change that
        adds it is installed.
        """
        dist = executor.dist
        replacement = Distributed.startup_record.replacement
        Distributed.startup_record.close()
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "With Wide-EP fault tolerance no rank may create a torch.distributed process group "
                "(NCCL or Gloo) during its startup. Every rank is its own process, and a rank's "
                "replacement could never join the group."
            )
        check_fault_tolerance_requirements(
            [
                (
                    type(manager).try_commit_blocks is KVCacheManagerV2.try_commit_blocks,
                    f"a KV cache manager with the default reuse commits (not "
                    f"{type(manager).__name__}, which overrides try_commit_blocks), because a "
                    "step's reuse commits wait until the step is accepted and then bypass the "
                    "override",
                )
                for manager in executor.resource_manager.resource_managers.values()
                if isinstance(manager, KVCacheManagerV2)
            ]
            + [
                (
                    False,
                    f"the KV cache manager V2 (kv_cache_config.use_kv_cache_manager_v2=True, not "
                    f"{type(manager).__name__}), because a step's reuse commits wait until the "
                    "step is accepted, which only the V2 manager supports",
                )
                for manager in executor.resource_manager.resource_managers.values()
                if isinstance(manager, KVCacheManager)
            ]
        )
        balancer = _moe_load_balancer_of(executor)
        startup_memory = StartupMemory(executor.kv_cache_manager.gpu_quota)
        moe_execution_state = moe_comms(executor.model_engine.model)[0].execution_state
        if not replacement and balancer is not None:
            check_largest_failure_fits(executor, balancer, startup_memory, moe_execution_state)
        store = rendezvous_store()
        publish_tuned_tactics(store, dist.rank)
        if dist.rank == 0:
            publish_startup_record(store)
        recovery = cls(
            executor,
            dist.host_exchange,
            moe_execution_state,
            store,
            startup_memory,
            authority_process(),
            incarnation=membership_client().incarnation,
        )
        if not replacement:
            wait_for_every_rank()
        recovery.start()
        if replacement:
            recovery._join()
        logger.info(
            f"Rank {dist.rank} joined the fault-tolerance membership at {os.environ[RENDEZVOUS_ENV]}"
        )
        return recovery

    def set_outstanding_requests(self, outstanding_requests: Callable[[], Dict[int, int]]) -> None:
        """Attach the executor worker's map from request id to client id of unfinished requests.

        After a membership change, rank 0 fails the ones that no active rank holds.
        """
        self._outstanding_requests = outstanding_requests

    def _join(self) -> None:
        """Admit this replacement and resume the executor loop where the active ranks resume."""
        # A replacement may miss the PROPOSE that adds it, so it reports its readiness up front.
        self._end_eplb_iterations()
        self._client.report(joining_ready=True)
        self._change_membership()
        self._executor.resume_at_iteration(self._installed.resume.iteration + 1)

    def start(self) -> None:
        """Take over the membership client that this process started at worker entry.

        Also share the hang detector's pause through the rank's pause word, which the node
        supervisor reports to the authority. A paused call may hold the GIL for longer than the
        hang timeout, and the client thread cannot report while it does.
        """
        self._client = membership_client()
        self._client.bind(self._on_record, on_tick=self._on_tick, on_error=self._on_client_error)
        self._executor.hang_detector.share_pause_word(open_pause_word(pause_word_path(self.rank)))

    def stop(self, completed: bool) -> None:
        """Leave the membership after a completed run, or end the process after a failed one.

        After a completed run, rank 0 records SHUTDOWN, so that the node supervisors do not take
        the ranks' exits for failures. If the deployment aborts instead, as when rank 0 is lost
        first, the process ends with status 1. MPI_Finalize waits for every rank, so after a rank
        failure an atexit handler ends the process before mpi4py finalizes MPI. Handlers
        registered later still run before it.
        """
        if completed:
            if self.rank == 0:
                self._client.report(shutdown=True)
            with self._record_changed:
                self._record_changed.wait_for(
                    lambda: self._record.state
                    in (MembershipState.SHUTDOWN, MembershipState.ABORTED),
                    _AUTHORITY_TIMEOUT_S,
                )
                record = self._record
            if record.state == MembershipState.ABORTED:
                end_process(self.rank, 1, f"the membership record is {record!r}")
            if record.state != MembershipState.SHUTDOWN:
                logger.error(
                    f"Rank {self.rank} received no SHUTDOWN record in {_AUTHORITY_TIMEOUT_S:.0f} "
                    "s, so its node supervisor may take its exit for a failure. Membership "
                    f"record {record!r}, phase {self._phase.name}"
                )
            self._client.stop()
            if self._authority is not None:
                try:
                    self._authority.wait(_AUTHORITY_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self._authority.kill()
            if self._record.generation > 0:
                atexit.register(end_process, self.rank, 0, "the executor completed")
            return
        end_process(self.rank, 1, "the executor stopped")

    def _on_record(self, record: MembershipRecord) -> None:
        with self._record_changed:
            self._record = record
            if self._phase is _Phase.JOINING and record.admits(self.rank, self.incarnation):
                self._set_phase(_Phase.RESUMING, f"generation {record.generation} admits the rank")
            if record.state == MembershipState.COMMITTED:
                self._committed_mask = record.active_mask
                self._committed_generation = record.generation
            change_not_installed = self._change_not_installed()
            joining = self._phase is _Phase.JOINING
            self._record_changed.notify_all()
        if change_not_installed:
            self._exchange.wake()
        if joining:
            return
        if record.state in (MembershipState.PROPOSE, MembershipState.ABORTED) or self._excluded(
            record
        ):
            if self._interrupts_steps(record):
                self._moe.request_abort()
            if self.rank == 0:  # rank 0 may be waiting for requests rather than in a collective
                self._executor.executor_request_queue.wake()
            loop = self._executor.worker_thread
            if (record.state == MembershipState.ABORTED or self._excluded(record)) and (
                loop is not None and not loop.is_alive()
            ):
                end_process(self.rank, 1, f"the membership record is {record!r}")

    def _on_client_error(self, error: Exception) -> None:
        """End the process once its membership client failed, unless the deployment shuts down.

        Without the authority no later failure can be recovered, as when rank 0 is lost.
        """
        if self._record.state != MembershipState.SHUTDOWN:
            end_process(self.rank, 1, f"the membership client failed ({error!r})")

    def _on_tick(self) -> None:
        waited_flags = self._moe.waited_flags()
        self._client.report(
            progress=self._executor.iter_counter,
            waiting_for=self._exchange.waiting_for
            | self._waiting_in_store
            | self._unimportable_peers
            | rank_mask(waited_flags, self.world_size),
            waited_flags=tuple(sorted(waited_flags.items())),
            waited_epochs=self._exchange.waited_epochs,
        )

    def _set_phase(self, phase: _Phase, reason: str) -> None:
        with self._record_changed:
            if phase not in _PHASE_TRANSITIONS[self._phase]:
                raise RuntimeError(
                    f"Rank {self.rank} cannot move from membership phase {self._phase.name} to "
                    f"{phase.name} ({reason})"
                )
            logger.debug(f"Rank {self.rank}: {self._phase.name} -> {phase.name}, {reason}")
            self._phase = phase

    def _excluded(self, record: MembershipRecord) -> bool:
        return self._phase is not _Phase.JOINING and not record.is_member(
            self.rank, self.incarnation
        )

    def _interrupts_steps(self, record: MembershipRecord) -> bool:
        """Check if ``record`` aborts the steps in flight rather than waiting for them to finish."""
        survivors = record.active_mask & ~record.joining_mask
        return (
            record.state == MembershipState.ABORTED
            or self._excluded(record)
            or bool(self._committed_mask & ~survivors)
        )

    def _change_not_installed(self) -> bool:
        """Return whether the latest record is a change that this rank has not installed.

        Call it under ``_record_changed``.
        """
        record = self._record
        return (
            record.state in (MembershipState.PROPOSE, MembershipState.ABORTED)
            or (record.state == MembershipState.INSTALL and record != self._installed)
            or self._excluded(record)
        )

    def _membership_changing(self) -> bool:
        """Decide if the host exchange stops for a change that this rank has not installed yet.

        A change that does not interrupt the steps waits for a forward vote that launched no
        forward. This way every active rank stops in the same round and after the same forwards.
        """
        with self._record_changed:
            return self._change_not_installed() and (
                self._phase is not _Phase.SERVING or self._interrupts_steps(self._record)
            )

    @contextlib.contextmanager
    def _hang_detection_paused(self):
        """Pause the hang detector during a membership change. The authority bounds its duration."""
        detector = self._executor.hang_detector
        armed = not detector.paused()
        detector.disarm()
        try:
            yield
        finally:
            if armed:
                detector.checkpoint()

    def _on_exchange_interrupt(self) -> None:
        """Run the membership change that stopped a host exchange round.

        The change communicates over the rendezvous store and calls ``HostExchange.install`` but
        starts no host exchange round, so it never re-enters the round it stopped. A joining rank
        whose install is in progress then abandons the install it was finishing, and
        ``before_iteration`` finishes the newer change.
        """
        self._change_membership()
        if self._phase in _INSTALL_IN_PROGRESS and self._installed.is_joining(self.rank):
            raise _MembershipRestarted

    def _change_membership(self) -> None:
        """Take part in the membership change until the active ranks' host exchange is installed."""
        with self._hang_detection_paused():
            record = self._installed
            listed = record is not None
            installed = False
            while not installed:
                record = self._next_record(record)
                installed = self._take_part_in(record, listed)
                listed = listed or record.is_member(self.rank, self.incarnation)

    def _take_part_in(self, record: MembershipRecord, listed: bool) -> bool:
        """Handle one record of a membership change and return whether the change is installed.

        ``listed`` tells if an earlier record of this change listed this rank. A record that ends
        the deployment or no longer includes this rank ends the process.
        """
        if record.state == MembershipState.ABORTED:
            end_process(self.rank, 1, "the deployment cannot recover from the rank failure")
        if record.state == MembershipState.SHUTDOWN:
            end_process(self.rank, 0, "the deployment is shutting down")
        if not record.is_member(self.rank, self.incarnation):
            if self._phase is not _Phase.JOINING or listed:
                end_process(self.rank, 1, "the membership no longer includes this rank")
            return False
        if record.state == MembershipState.PROPOSE:
            self._on_propose(record)
            return False
        if record.state == MembershipState.INSTALL:
            return self._on_install(record)
        return True

    def _on_propose(self, record: MembershipRecord) -> None:
        self._end_eplb_iterations()
        if self._record == record:
            paused_at = ResumeState(
                iteration=self._executor.iter_counter, epoch=self._exchange.stopped_epoch
            )
            self._client.report(
                reply=Reply.READY, reply_generation=record.generation, paused_at=paused_at
            )

    def _on_install(self, record: MembershipRecord) -> bool:
        self._client.report(joining_ready=False)
        installed = False
        with self._install_attempt(record):
            self._install(record)
            installed = True
        return installed

    @contextlib.contextmanager
    def _install_attempt(self, record: MembershipRecord) -> Iterator[None]:
        """Stop installing ``record`` at an expected error.

        A newer change only stops the attempt. After the other expected errors, the rank tells the
        authority which ranks failed and waits for its next record. Any other error ends this rank's
        process, which the authority then removes like any failed rank.
        """
        try:
            yield
        except _MembershipRestarted as error:
            logger.warning(
                f"Rank {self.rank} stopped installing membership generation "
                f"{record.generation}: {error!r}"
            )
        except MnnvlPeerHandleError as error:
            self._wait_on_unimportable_peers(record, error)
        except (MnnvlMappingError, _ExpertsNotRestorable) as error:
            self._fail_install(record, error)

    def _next_record(self, seen: Optional[MembershipRecord]) -> MembershipRecord:
        with self._record_changed:
            self._record_changed.wait_for(lambda: self._record != seen)
            return self._record

    def _end_eplb_iterations(self) -> None:
        """Abandon the online EPLB iterations in progress, fixing the placement until the resume."""
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is not None:
            balancer.abandon_iterations()

    def _install(self, record: MembershipRecord) -> None:
        """Map the host exchange rows among the active ranks.

        The rows move to fresh pages if a process that maps them may be gone.
        """
        group = self._active_group(record, "host_exchange_rows")
        self._exchange.install(
            group, record.active_mask & ~record.joining_mask, record.resume.epoch
        )
        with self._record_changed:
            self._installed = record
            if self._phase is not _Phase.JOINING:
                self._set_phase(_Phase.INSTALLING, f"installed generation {record.generation}")
        self._install_step_done()

    def _install_step_done(self) -> None:
        self._install_steps += 1
        self._client.report(install_steps=self._install_steps)

    def _active_group(self, record: MembershipRecord, name: str) -> StoreGroup:
        """Return an active-rank store group that stops waiting once ``record`` changes."""

        def on_wait(waiting: int) -> None:
            self._waiting_in_store = waiting
            if waiting and self._record != record:
                raise _MembershipRestarted

        group = StoreGroup(
            self._store,
            f"{name}/{record.generation}",
            self.rank,
            range(self.world_size),
            on_wait,
            active=[r for r in range(self.world_size) if record.is_active(r)],
        )
        self._change_groups.append(group)
        return group

    def _fail_install(self, record: MembershipRecord, error: Exception) -> None:
        """Reply FAILED and wait for the next membership record.

        A rank that only saw other ranks fail to map MNNVL memory does not reply. Those ranks do.
        """
        if isinstance(error, MnnvlMappingError) and self.rank not in error.ranks:
            logger.warning(f"Rank {self.rank} waits for the next membership record: {error!r}")
        else:
            logger.error(
                f"Rank {self.rank} could not install membership generation {record.generation}: "
                f"{error!r}\n{traceback.format_exc()}"
            )
            self._client.report(reply=Reply.FAILED, reply_generation=record.generation)
        self._next_record(record)

    def _wait_on_unimportable_peers(
        self, record: MembershipRecord, error: MnnvlPeerHandleError
    ) -> None:
        """Wait for the next membership record, reporting a wait on the peers that ``error`` names.

        Every active rank raises the same error, so those peers end every wait-for chain, and the
        authority's hang rule removes them even if their processes still run.
        """
        logger.error(
            f"Rank {self.rank} stopped installing membership generation {record.generation} and "
            f"waits for the authority to remove ranks {list(error.ranks)}: {error!r}"
        )
        self._unimportable_peers = rank_mask(error.ranks, self.world_size)
        try:
            self._next_record(record)
        finally:
            self._unimportable_peers = 0

    def before_iteration(self) -> None:
        """Finish installing the membership change at the first iteration past the resume point.

        An original rank tries once. A joining rank whose admission a newer change replaced installs
        that change and tries again.
        """
        if self._memory.release_agreed:
            self._release_at_step_boundary()
        record = self._installed
        if (
            self._phase not in (_Phase.INSTALLING, _Phase.JOINING)
            or self._executor.iter_counter <= record.resume.iteration
        ):
            return
        with self._hang_detection_paused():
            while True:
                if self._phase is _Phase.INSTALLING:
                    self._set_phase(_Phase.RESUMING, "past the resume iteration")
                self._executor.resume_at_iteration(record.resume.iteration + 1)
                # A joining rank whose admission was committed stops for a newer change in the host
                # exchange, as the other active ranks do.
                if self._finish_install(record) or self._phase is not _Phase.JOINING:
                    return
                if self._installed is record:
                    self._change_membership()
                record = self._installed

    def _finish_install(self, record: MembershipRecord) -> bool:
        """Restore the active ranks, reply INSTALLED and return whether the change was committed.

        Every active rank runs the same host collectives in the same order: the survivors retire
        their in-flight batch, the joining ranks enter the host exchange at their admission, then
        every active rank gathers the known requests and its slots, and restores the all-to-all
        workspaces and the experts. A rank that installed a newer change meanwhile skips the
        admission but still joins the gather. Past the gather no active rank has an all-to-all
        kernel in flight.
        """
        timer = _PhaseTimer()
        torch.cuda.synchronize()
        restored = False
        with self._install_attempt(record):
            recomputed, outstanding = self._prepare_install(record, timer)
            known, own_slots = self._gather_rank_state(record, timer)
            self._restore(record, own_slots, timer)
            restored = True
        if not restored:
            return False
        self._client.report(reply=Reply.INSTALLED, reply_generation=record.generation)
        # The next change may replace COMMITTED before this rank reads it.
        self._next_record(record)
        if self._committed_generation < record.generation:
            return False
        self._resume_serving(record, recomputed, outstanding, known)
        return True

    def _prepare_install(
        self, record: MembershipRecord, timer: _PhaseTimer
    ) -> Tuple[int, Dict[int, int]]:
        """Settle the interrupted steps and admit the joining ranks.

        Returns:
            The number of requests paused for recompute, and on rank 0 a map from request id to
            client id for the requests without a final response. The map is read before the known
            requests are gathered, so that a request enqueued in between does not count as lost.
        """
        # The step a rank stopped in may have started an EPLB iteration after READY. Its updates
        # complete here, before the rank gathers its slots.
        self._end_eplb_iterations()
        if not record.is_joining(self.rank):
            self._executor.retire_inflight_batch()
        recomputed = self._recompute_interrupted_requests()
        if record.joining_mask:
            if self._installed is record:
                self._admit_joining_ranks(record)
            self._install_step_done()
        outstanding = self._outstanding()
        timer.end("prepare")
        return recomputed, outstanding

    def _gather_rank_state(
        self, record: MembershipRecord, timer: _PhaseTimer
    ) -> Tuple[Set[int], List[Optional[OwnSlots]]]:
        """Return the request ids that the active ranks know and each rank's slots.

        A rank's slots are None without EPLB or if the rank is inactive. Rank 0 also hands its
        waiting queue to the joining ranks. A joining rank knows no request, because its queue is
        only a copy of rank 0's, possibly from an earlier admission attempt.
        """
        executor = self._executor
        balancer = _moe_load_balancer_of(executor)
        own_slots = None
        if balancer is not None:
            own_slots = [
                ranks[self.rank] for _, ranks in balancer.expert_restore.current_placement()
            ]
        own_requests = set() if record.is_joining(self.rank) else executor.known_request_ids()
        gathered = executor.dist.allgather((own_requests, own_slots))
        if record.joining_mask:
            self._hand_over_waiting_queue(record)
        timer.end("request gather")
        self._install_step_done()
        known = set().union(*(payload[0] for payload in gathered if payload is not None))
        return known, [None if payload is None else payload[1] for payload in gathered]

    def _hand_over_waiting_queue(self, record: MembershipRecord) -> None:
        """Hand rank 0's waiting queue, pending cancellations and shutdown flag to the joining ranks.

        Every active rank then drops the same canceled requests from the queue, and a joining rank
        admitted after the shutdown request drains and stops with the others. The queue goes
        through the host exchange, so user prompts stay out of the rendezvous store.
        """
        if self.rank == 0:
            waiting = self._executor.export_waiting_state()
            joining = [r for r in range(self.world_size) if record.is_joining(r)]
            self._exchange.exchange(pickle.dumps(waiting), readers=[], recipients=joining)
            return
        joining = record.is_joining(self.rank)
        payloads = self._exchange.exchange(
            pickle.dumps(None), readers=[0] if joining else [], recipients=[]
        )
        if joining:
            # Rank 0's queue replaces the one an earlier attempt of this admission took.
            self._executor.import_waiting_state(pickle.loads(payloads[0]))  # nosec B301

    def _resume_serving(
        self,
        record: MembershipRecord,
        recomputed: int,
        outstanding: Dict[int, int],
        known: Set[int],
    ) -> None:
        """Resume serving after the commit and fail the requests that no active rank holds.

        A request counts as lost only while it is still outstanding at the commit, so a final
        response that the worker handled during the install does not make it count.
        """
        self._set_phase(_Phase.SERVING, f"generation {record.generation} is committed")
        # The joining ranks read their epoch before they replied INSTALLED.
        for generation in self._published_admissions:
            self._store.delete_key(_admission_epoch_key(generation))
        self._published_admissions.clear()
        # Every active rank replied INSTALLED, so it made its last call in these groups.
        active = [r for r in range(self.world_size) if record.is_active(r)]
        for group in self._change_groups:
            group.close(active)
        self._change_groups.clear()
        # Serving resumes first. Every rank returns the empty spare slots' memory to the KV cache
        # at the step boundary that the next forward vote picks.
        self._memory.schedule_release()
        still_outstanding = self._outstanding()
        lost = {r: c for r, c in outstanding.items() if r not in known and r in still_outstanding}
        self._executor.fail_requests_by_id(lost, RankFailureError())
        logger.info(
            f"Rank {self.rank} resumed serving with membership generation {record.generation}, "
            f"ranks {active}. {recomputed} interrupted requests continue by recompute"
        )
        if self.rank == 0:
            logger.info(
                f"Membership generation {record.generation} ended {len(lost)} requests that no "
                "active rank holds"
            )

    def _outstanding(self) -> Dict[int, int]:
        """Return, on rank 0, the client ids of the requests without a final response."""
        return {} if self._outstanding_requests is None else self._outstanding_requests()

    def _release_at_step_boundary(self) -> None:
        """Release the last committed change's memory, or retry a release that fell short.

        Every rank retires its step in flight, which runs collectives, even if only another rank
        releases memory.
        """
        release = self._memory.end_release_vote()
        torch.cuda.synchronize()
        self._executor.retire_inflight_batch()
        # A change that arrived after the vote keeps the memory until it is committed.
        if release and not self.pauses_forward():
            self._memory.release(_moe_load_balancer_of(self._executor), self._installed.generation)

    def _restore(
        self, record: MembershipRecord, own_slots: List[Optional[OwnSlots]], timer: _PhaseTimer
    ) -> None:
        """Restore the all-to-all workspaces and the lost experts on every active rank.

        ``own_slots`` holds each rank's slots, as gathered after the ranks abandoned their EPLB
        iterations.
        """
        if self._record != record:
            raise _MembershipRestarted
        started_s = time.perf_counter()
        balancer = _moe_load_balancer_of(self._executor)
        plan, loaded_experts = None, 0
        if balancer is not None:
            plan, loaded_experts = self._plan_expert_restore(record, own_slots, balancer)
        self._mark_rank_health(record)
        group = self._active_group(record, "moe_a2a_workspaces")
        fresh_rows = self._moe.workspace_rows_needing_fresh_pages(group)
        # The remap maps fresh pages for this rank's rows before it frees the pages they replace.
        workspace_bytes = self._moe.workspace_bytes() if self.rank in fresh_rows else 0
        restore_bytes = workspace_bytes + (0 if plan is None else plan.device_bytes)
        try:
            self._memory.check_restore_fits(restore_bytes, balancer)
        except ValueError as error:
            raise _ExpertsNotRestorable(str(error)) from error
        timer.begin()
        with contextlib.ExitStack() as kv_cache_shrunk:
            try:
                kv_cache_quota = kv_cache_shrunk.enter_context(
                    self._memory.kv_cache_shrunk_by(restore_bytes, balancer, self._recomputable)
                )
            except KvCacheShrinkError as error:
                raise _ExpertsNotRestorable(str(error)) from error
            if restore_bytes:
                timer.end("KV cache shrink")
                self._install_step_done()
            timer.begin()
            self._moe.remap_workspaces(group, fresh_rows)
            if workspace_bytes:
                self._memory.owe_workspace_rows()
            timer.end("remap")
            self._install_step_done()
            if plan is not None:

                def restore_progress() -> None:
                    # A newer change stops the loads that this one planned, as the store groups do.
                    if self._record != record:
                        raise _MembershipRestarted
                    self._install_step_done()

                # Experts are staged through the remapped workspace, which no former member writes.
                try:
                    balancer.expert_restore.restore_lost_experts(
                        plan,
                        self._moe.staging_workspace(),
                        self._active_group(record, "expert_staging").allgather,
                        timer.times_ms,
                        progress=restore_progress,
                    )
                except ExpertRestorePlanMismatchError as error:
                    raise _ExpertsNotRestorable(str(error)) from error
                self._install_step_done()
        timer.begin()
        self._rearm(record)
        timer.end("rearm")
        logger.info(
            f"Rank {self.rank} restored membership generation {record.generation} in "
            f"{(time.perf_counter() - started_s) * 1e3:.0f} ms. Loaded {loaded_experts} experts "
            f"({0 if plan is None else plan.device_bytes} bytes), KV cache quota {kv_cache_quota}. "
            f"Phases: {timer.summary()}"
        )

    def _plan_expert_restore(
        self,
        record: MembershipRecord,
        own_slots: List[Optional[OwnSlots]],
        balancer: MoeLoadBalancer,
    ) -> Tuple[ExpertRestorePlan, int]:
        """Plan the expert placement of ``record`` and count the experts this rank loads."""
        agreed = agreed_placement(own_slots, record)
        try:
            plan = balancer.expert_restore.plan_expert_restore(
                [record.is_active(r) for r in range(self.world_size)],
                agreed,
                rejoining=[record.is_joining(r) for r in range(self.world_size)],
            )
        except ValueError as error:
            raise _ExpertsNotRestorable(str(error)) from error
        loaded_experts = 0
        for planned, agreed_layer in zip(plan.placements, agreed):
            new_slots, old_slots = planned[balancer.ep_rank], agreed_layer[balancer.ep_rank]
            loaded_experts += int(np.count_nonzero((new_slots != old_slots) & (new_slots >= 0)))
        self._install_step_done()
        return plan, loaded_experts

    def _mark_rank_health(self, record: MembershipRecord) -> None:
        for r in range(self.world_size):
            if record.is_active(r):
                self._moe.ep_group_health.mark_active(r)
            else:
                self._moe.ep_group_health.mark_failed(r)

    def _rearm(self, record: MembershipRecord) -> None:
        """Rearm the all-to-all and install its active rank mask, unless ``record`` is outdated."""
        # The client thread takes this lock for every record, so wait for the device before it.
        torch.cuda.synchronize()
        # The client thread updates the record under this lock before it requests the abort of a
        # newer change. Once this check passes, that abort follows the rearm and stays in effect.
        with self._record_changed:
            if self._record != record:
                raise _MembershipRestarted
            self._moe.rearm()
            self._moe.install_mask()

    def _admit_joining_ranks(self, record: MembershipRecord) -> None:
        """Bring the joining ranks into the host exchange with rank 0's next epoch."""
        epoch_key = _admission_epoch_key(record.generation)
        if record.is_joining(self.rank):

            def check() -> None:
                self._waiting_in_store = 1
                if self._record != record:
                    raise _MembershipRestarted

            try:
                (epoch,) = wait_for_keys(self._store, [epoch_key], REPORT_INTERVAL_S, check)
            finally:
                self._waiting_in_store = 0
            self._exchange.admit(record.active_mask, int(epoch))
            return
        epoch = next_epoch(self._exchange.epoch)
        if self.rank == 0:
            self._store.set(epoch_key, str(epoch).encode())
            self._published_admissions.append(record.generation)
        self._exchange.admit(record.active_mask, epoch)

    def pauses_forward(self) -> bool:
        """Return True while a membership change that is not yet committed pauses the forward."""
        with self._record_changed:
            return self._phase in _INSTALL_IN_PROGRESS or self._change_not_installed()

    def forward_vote(self, batch_size: int) -> Tuple[bool, bool]:
        """Run the attention-DP forward vote with the change and memory release flags.

        A rank whose forward a membership change pauses reports batch size 0, so no rank launches
        a forward with it. A rank flags a change it has not installed, and a vote with a flag moves
        a serving rank to STOPPED.

        Returns:
            Whether every active rank has a batch to forward, and whether this rank has one.
        """
        with self._record_changed:
            change_not_installed = self._change_not_installed()
        forward_paused_here = change_not_installed or self._phase in _INSTALL_IN_PROGRESS
        release = self._memory.wants_release(forward_paused_here)
        own_batch_size = 0 if forward_paused_here else batch_size
        votes = self._executor.dist.tp_allgather_int64(
            [own_batch_size, int(change_not_installed), int(release)], active_only=True
        )
        change_voted = bool(votes[:, 1].any())
        if self._phase in (_Phase.SERVING, _Phase.STOPPED):
            phase = _Phase.STOPPED if change_voted else _Phase.SERVING
            if phase is not self._phase:
                self._set_phase(phase, "the forward vote")
        self._memory.record_release_vote(bool(votes[:, 2].any()) and not change_voted)
        if forward_paused_here and self.rank == 0:
            self._executor.executor_request_queue.wake()
        return bool(votes[:, 0].all()), own_batch_size > 0

    @staticmethod
    def mark_outputs(scheduled_batch: ScheduledRequests) -> OutputsMarks:
        """Return where the stored outputs end for each request that returns logits or outputs."""
        return [
            (request, request.py_result.mark_outputs())
            for request in scheduled_batch.all_requests()
            if request.py_return_context_logits
            or request.py_return_generation_logits
            or request.py_additional_outputs
        ]

    def check_step(self) -> None:
        """Enqueue, after a forward and ahead of its sampler event, the all-to-all check that
        :meth:`accept_step` reads: whether a peer may have overwritten rows that the forward read."""
        self._moe.check_incomplete_rounds()

    def accept_step(self, batch: BatchState) -> bool:
        """Wait until the step of ``batch`` is sampled and return whether it completed.

        An abort may have cut the step's all-to-all rounds short. Such a step's requests continue
        by recompute, and the outputs it stored are dropped.
        """
        event = batch.sample_state.sampler_event if batch.sample_state is not None else None
        if event is not None:
            event.synchronize()
        if not self._step_aborted():
            return True
        self._interrupted.extend(batch.scheduled_requests.all_requests())
        for request, mark in batch.outputs_marks:
            # The next step in flight was marked after this step's outputs, so it rewinds to the
            # first discarded step's mark too.
            first_mark = self._discarded_marks.setdefault(request.py_request_id, mark)
            request.py_result.rewind_outputs(first_mark)
        return False

    def discard_failed_forward(self, scheduled_requests: ScheduledRequests) -> bool:
        """Mark the requests of a failed forward for recompute if its all-to-all was aborted.

        Returns:
            False if the forward failed for another reason.
        """
        if not self._step_aborted():
            return False
        self._interrupted.extend(scheduled_requests.all_requests())
        return True

    def _step_aborted(self) -> bool:
        """Return whether a membership change aborted the all-to-all, which discards the step.

        Every abort comes from a change that pauses the forward. An abort that a peer requested
        also cancels this rank's collectives, possibly before this rank receives the change. So
        wait for a record that pauses the forward, or for the SHUTDOWN record that may follow it.

        Raises:
            RuntimeError: No such record arrived within ``AUTHORITY_SILENCE_TIMEOUT_S``.
        """
        if not self._moe.aborted():
            return False
        with self._record_changed:
            if self._record_changed.wait_for(
                lambda: self._record.state == MembershipState.SHUTDOWN or self.pauses_forward(),
                AUTHORITY_SILENCE_TIMEOUT_S,
            ):
                return True
            record = self._record
        raise RuntimeError(
            f"The MoE all-to-all of rank {self.rank} aborted, and no membership change arrived in "
            f"{AUTHORITY_SILENCE_TIMEOUT_S:g} s (membership record {record!r})"
        )

    @staticmethod
    def _recomputable(request: LlmRequest) -> bool:
        return not request.is_dummy and not request.is_finished and request.py_seq_slot is not None

    def _recompute_interrupted_requests(self) -> int:
        """Pause the discarded steps' requests for recompute like a preemption and count them."""
        interrupted = {request.py_request_id for request in self._interrupted}
        self._interrupted = []
        self._discarded_marks = {}
        requests = [
            request
            for request in self._executor.active_requests
            if request.py_request_id in interrupted and self._recomputable(request)
        ]
        self._executor.recompute_requests(requests)
        return len(requests)

    def fail_unservable_requests(
        self, waiting_queue: WaitingQueue, rank_states: List[RankState]
    ) -> None:
        """Fail the waiting requests pinned strictly to an attention-DP rank that does not serve.

        A request to a rank whose replacement may still join fails as one to send again.
        """
        if len(rank_states) == self._executor.dist.tp_size:
            return
        serving = {state.rank for state in rank_states}
        unservable = {}
        for item in waiting_queue:
            params = getattr(item.request, "py_scheduling_params", None)
            if (
                params is not None
                and params.attention_dp_rank is not None
                and not params.attention_dp_relax
                and params.attention_dp_rank not in serving
            ):
                unservable[item.id] = item
        if not unservable:
            return
        waiting_queue.remove_by_ids(set(unservable))
        if self.rank != 0:
            return
        record = self._record
        for item in unservable.values():
            attention_dp_rank = item.request.py_scheduling_params.attention_dp_rank
            if record.is_given_up(attention_dp_rank):
                error = (
                    f"Attention DP rank {attention_dp_rank} failed and no replacement serves it. "
                    "Send the request without attention_dp_rank or with attention_dp_relax."
                )
            else:
                error = RankFailureError(
                    f"Attention DP rank {attention_dp_rank} failed and serves again once its "
                    "replacement joins. Send the request again, or without attention_dp_rank, or "
                    "with attention_dp_relax.",
                    resubmit=False,
                )
            self._executor.fail_requests_by_id(
                {item.id: getattr(item.request, "client_id", None)}, error
            )
