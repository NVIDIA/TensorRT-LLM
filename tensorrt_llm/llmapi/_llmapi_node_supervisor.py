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
"""Start and replace the ranks of one node under Wide-EP fault tolerance.

``trtllm-llmapi-launch`` runs one node supervisor per node, as the task that
``mpirun --enable-recovery`` starts there. The supervisor starts every rank on its node as a
separate process, follows the record that the membership authority sends, and starts a
replacement on the same GPU for a rank that the membership drops.
"""

import argparse
import concurrent.futures
import contextlib
import glob
import os
import selectors
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import threading
import time
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Dict, List, NoReturn, Optional, Set

_SHM_DIR = "/dev/shm"
_CONNECT_TIMEOUT_S = 2.0
_CONNECT_WARNING_INTERVAL_S = 30.0
# How long the ranks may drain after a stop signal, and then end after SIGTERM. At a job end, mpirun
# kills the supervisor within a second of its signal, so the ranks get no such grace there.
_STOP_GRACE_S = 10.0
# Delay before the next replacement after a failed one. It doubles with each failure in a row.
_BACKOFF_S = 1.0
# Failed replacements in a row, for any reason, after which a rank is given up.
_MAX_FAILED_REPLACEMENTS = 3
# A replacement that served this long before it ended counts as healthy. Its end is a new failure,
# not a failed replacement, so its rank's failure count resets.
_HEALTHY_SERVING_S = 600.0
# Run as ``_SCRIPT --exec-rank <supervisor PID> <command>``, this script execs a rank's command.
_EXEC_RANK = "--exec-rank"
_SCRIPT = os.path.abspath(__file__)


def _exit_status(returncode: int) -> int:
    """Return the exit status as a shell reports it: 128 + N for a process killed by signal N."""
    return 128 - returncode if returncode < 0 else returncode


@dataclass
class _Process:
    """A rank process that this supervisor started."""

    child: subprocess.Popen
    replacement: bool
    pidfd: Optional[int]
    incarnation: Optional[int] = None
    listed: bool = False
    # When a COMMITTED record first listed it as an active member.
    admitted_s: Optional[float] = None
    # The reason the membership left it out while it joined.
    left_out_reason: str = ""
    killed_s: Optional[float] = None
    status: Optional[int] = None
    started_s: float = 0.0


# Selector data for descriptors other than rank pidfds, which carry the rank instead.
_WAKE = "wake"
_PARENT = "parent"
_AUTHORITY = "authority"


class NodeSupervisor:
    """Start, supervise and replace the ranks of one node.

    Every report interval, it reports to the authority which of its ranks are in a call that their
    hang detector's pause covers. It unlinks the node's host weights and its ranks' pause words once
    the deployment ends. On node 0 under ``mpirun``, it ends the job as failed.

    Args:
        command: The command that starts each rank process. With ``rank_0_in_session``, it
            starts every rank but 0.
        end_job: The function that ends the whole job as failed. Set on node 0 under ``mpirun``.
        rejoin: The node returns to a running deployment that lost it. The supervisor connects
            until the authority's first view, and starts only replacements.
        parent: The process whose end stops the node, by default this process's parent. 0 for
            none.
    """

    def __init__(
        self,
        node: int,
        nodes: int,
        gpus_per_node: int,
        command: List[str],
        environ: Dict[str, str],
        rank_0_in_session: bool = False,
        end_job: Optional[Callable[[], None]] = None,
        rejoin: bool = False,
        parent: Optional[int] = None,
    ):
        from tensorrt_llm._torch.distributed import membership
        from tensorrt_llm.llmapi._llmapi_process_guard import _signal_group
        from tensorrt_llm.logger import logger

        self._membership = membership
        self._logger = logger
        self._signal_group = _signal_group
        self._node = node
        self._gpus_per_node = gpus_per_node
        self._world_size = nodes * gpus_per_node
        self._ranks = list(range(node * gpus_per_node, (node + 1) * gpus_per_node))
        self._command = command
        self._rank_0_in_session = rank_0_in_session
        self._rejoin = rejoin
        self._environ = membership.rank_env(environ)
        self._parent = os.getppid() if parent is None else parent
        self._parent_ended = False
        self._selector = selectors.DefaultSelector()
        # Written by the signal handler and the connector, so that they end the selector's wait.
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_write.setblocking(False)
        self._connecting: Optional[concurrent.futures.Future] = None
        self._connection = None
        # A connection that waits for the authority's first view until _view_due_s.
        self._connection_awaiting_view = None
        self._view_due_s = 0.0
        self._view = None
        self._view_received_s = 0.0
        # Set once a record is past STARTING. Before that, the failure of any rank stops the node.
        self._deployment_started = False
        # The authority closed the connection before its first view, or its store holds values not
        # signed with this node's shared secret.
        self._refused_by_authority = False
        # The last connect reached the authority, which then sent a view or refused the supervisor.
        # While it is reachable, the node's ranks are replaced rather than stopped with the node.
        self._authority_reached = False
        self._connect_due_s = 0.0
        self._connect_warning_due_s = time.monotonic() + _CONNECT_WARNING_INTERVAL_S
        # Times at which the loop must step again. Each step rebuilds the list.
        self._deadlines_s: List[float] = []
        self._processes: Dict[int, _Process] = {}
        self._given_up: Set[int] = set()
        # Per rank, the failed replacements in a row and when the next one may start.
        self._failed_replacements: Dict[int, int] = {}
        self._replacement_due_s: Dict[int, float] = {}
        # Per rank this supervisor has not started, since when a process runs on its GPU.
        self._gpu_in_use_since_s: Dict[int, float] = {}
        self._first_failure: Optional[int] = None
        self._signum: Optional[int] = None
        self._signal_handled = False
        self._shutdown_handled = False
        # The shared-memory segments that the signal handler unlinked, for the loop to log.
        self._unlinked_on_signal = 0
        # When the supervisor stops the ranks still running after SHUTDOWN, once the authority left.
        self._shutdown_grace_due_s: Optional[float] = None
        self._signal_grace_due_s = 0.0
        # This process may have threads, so no Python code runs between a rank's fork and exec. A
        # new interpreter arms the rank's parent-death signal instead.
        self._exec_rank = [sys.executable, "-S", _SCRIPT, _EXEC_RANK, str(os.getpid())]
        self._end_job_fn = end_job
        self._host_weights_prefix = f"{membership.host_weights_name(node)}_"
        self._pause_word_paths = {rank: membership.pause_word_path(rank) for rank in self._ranks}
        # When the next report to the authority is due, and the part of the last one not sent yet.
        self._report_due_s = 0.0
        self._unsent = b""
        self._warned_unreadable_state = False

    def on_signal(self, signum: int, frame: Optional[FrameType] = None) -> None:
        """Unlink the node's shared memory, then stop the supervisor and signal rank 0 the same.

        The unlink runs here, before the loop wakes, since the launcher may kill this process soon
        after the signal (mpirun within a second). The loop logs it, since the handler may
        interrupt a log write, and a log write from the handler would then fail as a reentrant call.
        """
        if self._signum is None:
            self._unlinked_on_signal = self._unlink_shared_memory()
            self._signal_grace_due_s = time.monotonic() + _STOP_GRACE_S
            # Last, since the loop acts on the signal once it sees it.
            self._signum = signum
        self._wake()

    def _wake(self) -> None:
        with contextlib.suppress(OSError):
            self._wake_write.send(b"\0")

    def run(self) -> int:
        try:
            self._watch_parent()
            self._selector.register(self._wake_read, selectors.EVENT_READ, _WAKE)
            self._connect()
            # The first view tells whether the node starts its ranks. A rejoining node connects again
            # until it has one, for up to DEAD_TIMEOUT_S: within that time the authority drops the
            # connection of the node's previous supervisor once it ended (at once on a close, else
            # through the keepalive), and an authority silent for that long counts as gone, as any
            # silent peer does.
            rejoin_due_s = time.monotonic() + self._membership.DEAD_TIMEOUT_S
            while self._signum is None and (
                self._connecting is not None
                or self._connection_awaiting_view is not None
                or (
                    self._rejoin
                    and self._view is None
                    and not self._refused_by_authority
                    and not self._parent_ended
                    and time.monotonic() < rejoin_due_s
                )
            ):
                if self._connecting is None and self._connection_awaiting_view is None:
                    if time.monotonic() >= self._connect_due_s:
                        self._connect()
                    else:
                        self._deadlines_s = [self._connect_due_s]
                self._wait()
                self._deadlines_s = []
                self._take_connection()
                self._check_first_view(time.monotonic())
            if self._signum is not None or self._parent_ended:
                return self._step(time.monotonic())
            if self._refused_by_authority:
                return 1
            if self._rejoin and self._view is None:
                self._logger.error(
                    "No membership authority at "
                    f"{os.environ[self._membership.RENDEZVOUS_ENV]} admitted node {self._node} "
                    f"within {self._membership.DEAD_TIMEOUT_S:.0f} s: the deployment ended, the "
                    "node's previous supervisor still runs, or the node's ranks are not in this "
                    "deployment (check --rejoin-node, --nodes and the step's GPUs)"
                )
                return 1
            if self._view is not None and self._view.record.state in (
                self._membership.MembershipState.SHUTDOWN,
                self._membership.MembershipState.ABORTED,
            ):
                self._logger.error(
                    "The deployment at the rendezvous address ended. Use a new shared secret."
                )
                return 1
            if self._rejoin and not self._deployment_started:
                self._logger.error(
                    f"The deployment has not started, so start node {self._node} without --rejoin"
                )
                return 1
            if self._rejoin and len(self._view.record.incarnations) != self._world_size:
                self._logger.error(
                    f"The deployment has {len(self._view.record.incarnations)} ranks, not "
                    f"{self._world_size}: check --nodes and the step's GPUs"
                )
                return 1
            self._start_first_processes()
            while True:
                status = self._step(time.monotonic())
                if status is not None:
                    return status
                self._wait()
        except Exception as error:
            # No step stops the node any more, so stop its ranks and, on node 0, end the job.
            self._stop(1, end_job_reason=f"the node supervisor failed: {error!r}")
            raise
        finally:
            for key in list(self._selector.get_map().values()):
                if isinstance(key.fileobj, int):
                    os.close(key.fileobj)
                else:
                    key.fileobj.close()
            self._selector.close()
            self._wake_write.close()

    def _start_first_processes(self) -> None:
        """Start every rank's first process, unless the deployment has started."""
        if not self._deployment_started:
            for rank in self._ranks:
                if rank != 0 or not self._rank_0_in_session:
                    self._start(rank, replacement=False)

    def _watch_parent(self) -> None:
        # A process without a parent, such as a container's init process, has none to watch.
        if self._parent == 0:
            return
        try:
            pidfd = os.pidfd_open(self._parent)
        except ProcessLookupError:
            self._parent_ended = True
            return
        self._selector.register(pidfd, selectors.EVENT_READ, _PARENT)
        if os.getppid() != self._parent:
            self._parent_ended = True

    def _wait(self) -> None:
        """Wait for the next event or the next due step, and take the events."""
        timeout_s = None
        if self._deadlines_s:
            timeout_s = max(0.0, min(self._deadlines_s) - time.monotonic())
        for key, _ in self._selector.select(timeout_s):
            self._on_event(key)

    def _on_event(self, key: selectors.SelectorKey) -> None:
        if key.data == _WAKE:
            self._wake_read.recv(1 << 12)
        elif key.data == _PARENT:
            self._parent_ended = True
            self._close_pidfd(key)
        elif key.data == _AUTHORITY:
            self._receive_view()
        else:
            self._reap(key.data, time.monotonic())

    def _close_pidfd(self, key: selectors.SelectorKey) -> None:
        self._selector.unregister(key.fileobj)
        os.close(key.fileobj)

    def _step(self, now_s: float) -> Optional[int]:
        """Handle the latest events and timers, and return the exit status once the node stops."""
        self._deadlines_s = []
        if self._signum is not None:
            return self._stop_on_signal(now_s)
        if self._parent_ended:
            self._logger.warning("Stopping the node's ranks because the supervisor's parent ended")
            return self._stop(128 + signal.SIGTERM)
        self._take_connection()
        if self._refused_by_authority:
            return self._stop(1)
        record = None if self._view is None else self._view.record
        if record is not None and record.state == self._membership.MembershipState.SHUTDOWN:
            return self._stop_after_shutdown(record, now_s)
        if (
            self._connection is None
            and self._connection_awaiting_view is None
            and self._connecting is None
        ):
            if now_s >= self._connect_due_s:
                self._connect()
            self._deadlines_s.append(self._connect_due_s)
        self._check_first_view(now_s)
        if self._connection is not None:
            if now_s >= self._report_due_s:
                self._report_due_s = now_s + self._membership.REPORT_INTERVAL_S
                self._report()
            self._deadlines_s.append(self._report_due_s)
        if record is not None:
            status = self._check_abort(record)
            if status is not None:
                return status
        for rank in self._ranks:
            status = self._supervise(rank, record, now_s)
            if status is not None:
                return status
        if self._connection is None and not self._authority_reached and self._all_ranks_ended():
            if self._deployment_started:
                self._logger.error("Every rank of the node ended with no authority to follow")
            return self._stop(self._first_failure or 0)
        return None

    def _stop_on_signal(self, now_s: float) -> Optional[int]:
        """Stop the node on a signal, and return the exit status once it stopped.

        The launcher signals every node supervisor, and the signal makes rank 0 shut the deployment
        down. The ranks then drain their requests and end on their own. Once they ended, the
        SHUTDOWN record decides the exit status, so while the authority is connected the supervisor
        waits for the record up to the grace period. The ranks still running a grace period after
        the signal are stopped. A launcher that kills the supervisor sooner, as mpirun does at a job
        end, ends the ranks through their parent-death signal.
        """
        if not self._signal_handled:
            self._signal_handled = True
            self._log_unlinked(self._unlinked_on_signal)
            rank_0 = self._processes.get(0)
            # Through the pidfd, which unlike Popen.send_signal never reaps the process.
            if rank_0 is not None and rank_0.status is None:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(rank_0.pidfd, self._signum)
        record = None if self._view is None else self._view.record
        if record is not None:
            status = self._check_abort(record)
            if status is not None:
                return status
        if not self._active_ranks_running(record):
            if record is not None and record.state == self._membership.MembershipState.SHUTDOWN:
                return self._stop(self._shutdown_status(record))
            # The ranks end on the SHUTDOWN record, which may reach them before this supervisor.
            if self._connection is None:
                return self._stop(128 + self._signum)
        if now_s < self._signal_grace_due_s:
            self._deadlines_s.append(self._signal_grace_due_s)
            return None
        self._logger.warning(f"Stopping the node's ranks on signal {self._signum}")
        return self._stop(128 + self._signum)

    def _stop_after_shutdown(self, record, now_s: float) -> Optional[int]:
        """Stop the node once its active ranks ended after a shutdown, and return the exit status.

        The active ranks end after they leave the membership, and the authority ends once all of
        them left. Ranks still running a grace period after the authority's connection closed are
        stopped. Rank 0 is never stopped here, since its end is the deployment's end, which its
        launcher bounds. Each rank's end wakes the supervisor.

        The node's host weights and pause words are unlinked at once. The ranks still map them, so
        the unlink only removes the names, and the pages are freed once the last rank ends. After
        the ranks ended, the unlink would free the pages, which takes seconds and may not finish
        before the launcher kills this process at the job's end. Nothing opens them by name after
        SHUTDOWN.
        """
        if not self._shutdown_handled:
            self._shutdown_handled = True
            self._log_unlinked(self._unlink_shared_memory())
        running = self._active_ranks_running(record)
        if not running:
            self._logger.info("The deployment shut down")
            return self._stop(self._shutdown_status(record))
        if self._connection is not None or 0 in running:
            return None
        if self._shutdown_grace_due_s is None:
            self._shutdown_grace_due_s = now_s + _STOP_GRACE_S
        if now_s < self._shutdown_grace_due_s:
            self._deadlines_s.append(self._shutdown_grace_due_s)
            return None
        self._logger.error(
            f"Stopping ranks {running}, which did not end within {_STOP_GRACE_S:.0f} s of the "
            "authority's end after the shutdown"
        )
        return self._stop(self._shutdown_status(record) or 1)

    def _shutdown_status(self, record) -> int:
        """Return the first nonzero exit status of an active rank, in rank order, or 0."""
        return next(
            (
                _exit_status(process.status)
                for rank, process in sorted(self._processes.items())
                if process.status and record.is_active(rank)
            ),
            0,
        )

    def _check_abort(self, record) -> Optional[int]:
        """Stop the node if ``record`` aborts the deployment, and return the exit status."""
        state = self._membership.MembershipState
        if record.state == state.ABORTED:
            self._logger.error(f"Deployment aborted: {record.reason}")
            return self._stop(1, end_job_reason=f"the deployment aborted: {record.reason}")
        return None

    def _all_ranks_ended(self) -> bool:
        return bool(self._processes) and not self._ranks_running()

    def _ranks_running(self) -> bool:
        return any(process.status is None for process in self._processes.values())

    def _active_ranks_running(self, record) -> List[int]:
        """Return the node's running ranks that ``record`` lists as active, or all without one."""
        return [
            rank
            for rank, process in sorted(self._processes.items())
            if process.status is None and (record is None or record.is_active(rank))
        ]

    def _supervise(self, rank: int, record, now_s: float) -> Optional[int]:
        """Act on ``rank``, and return the exit status once the node stopped."""
        process = self._processes.get(rank)
        if not self._deployment_started:
            return self._check_before_serving(rank, process)
        if rank == 0:
            return self._check_rank_0(process)
        if self._connection is None:
            return None
        if process is not None and process.status is None:
            self._check_running(rank, process, record, now_s)
        else:
            self._replace_when_due(rank, record, now_s)
        return None

    def _check_before_serving(self, rank: int, process: Optional[_Process]) -> Optional[int]:
        """Stop the node if ``rank`` failed before the deployment serves, and return its status."""
        if process is None or process.status in (None, 0):
            return None
        self._logger.error(f"Rank {rank} failed before serving (status {process.status})")
        reason = f"rank 0 failed before serving (status {process.status})" if rank == 0 else None
        return self._stop(_exit_status(process.status), end_job_reason=reason)

    def _check_rank_0(self, process: Optional[_Process]) -> Optional[int]:
        """Stop the node once rank 0 ended, which ends the deployment, and return its status."""
        if process is None or process.status is None:
            return None
        self._logger.error("Rank 0 ended, so the deployment ends with it")
        return self._stop(
            _exit_status(process.status), end_job_reason=f"rank 0 ended (status {process.status})"
        )

    def _check_running(self, rank: int, process: _Process, record, now_s: float) -> None:
        """Track the membership of a running rank, and kill it once the membership drops it."""
        if record.is_active(rank):
            process.listed = True
            process.incarnation = record.incarnations[rank]
            if (
                process.admitted_s is None
                and record.state == self._membership.MembershipState.COMMITTED
            ):
                process.admitted_s = now_s
        elif process.killed_s is not None:
            self._check_killed(rank, process, now_s)
        elif process.listed:
            self._logger.warning(
                f"Killing rank {rank} (incarnation {process.incarnation}), "
                "which the membership no longer lists"
            )
            self._kill(process)
        # A view received before this replacement started reports its predecessor's stall.
        elif rank in self._view.stalled_replacements and self._view_received_s > process.started_s:
            self._logger.warning(
                f"Killing the replacement of rank {rank}, which made no progress for "
                f"{self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s"
            )
            self._kill(process)
        # The authority times a replacement's startup only once it connects, so time the part before
        # that here.
        elif not self._membership.has_rank(self._view.connected_mask, rank):
            due_s = process.started_s + self._membership.RENDEZVOUS_TIMEOUT_S
            if now_s < due_s:
                self._deadlines_s.append(due_s)
                return
            self._logger.warning(
                f"Killing rank {rank}'s process, which is not connected to the membership "
                f"authority {self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s after it started"
            )
            self._kill(process)

    def _check_killed(self, rank: int, process: _Process, now_s: float) -> None:
        """Give up ``rank`` if its killed process did not end within the rendezvous timeout."""
        if rank in self._given_up:
            return
        due_s = process.killed_s + self._membership.RENDEZVOUS_TIMEOUT_S
        if now_s < due_s:
            self._deadlines_s.append(due_s)
            return
        self._logger.error(
            f"Rank {rank}'s process (PID {process.child.pid}) did not end within "
            f"{self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s of SIGKILL, so its GPU memory may not "
            f"be free. The deployment serves without rank {rank}"
        )
        self._give_up(rank)

    def _replace_when_due(self, rank: int, record, now_s: float) -> None:
        """Replace ``rank`` if the membership left it out and its backoff has passed."""
        if record.is_active(rank) or rank in self._given_up:
            return
        due_s = self._replacement_due_s.get(rank, 0.0)
        if now_s < due_s:
            self._deadlines_s.append(due_s)
            return
        self._replace(rank, now_s)

    def _kill(self, process: _Process) -> None:
        """Kill a rank's process group, helper processes included.

        Step again at the rendezvous timeout, so that :meth:`_check_killed` gives the rank up if its
        process has not ended by then.
        """
        process.killed_s = time.monotonic()
        self._deadlines_s.append(process.killed_s + self._membership.RENDEZVOUS_TIMEOUT_S)
        self._signal_group(process.child.pid, signal.SIGKILL)

    def _start(self, rank: int, replacement: bool) -> None:
        from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV, RankIdentity

        identity = RankIdentity(rank, self._world_size, self._gpus_per_node, replacement)
        environ = {**self._environ, FAULT_TOLERANCE_RANK_ENV: identity.env_value()}
        child = subprocess.Popen(  # nosec B603
            [*self._exec_rank, *self._command],
            env=environ,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        pidfd = os.pidfd_open(child.pid)
        self._selector.register(pidfd, selectors.EVENT_READ, rank)
        self._processes[rank] = _Process(
            child, replacement, pidfd, None if replacement else 0, started_s=time.monotonic()
        )

    def _replace(self, rank: int, now_s: float) -> None:
        if self._waits_for_gpu(rank, now_s):
            return
        try:
            self._logger.warning(f"Starting a replacement of rank {rank}")
            self._start(rank, replacement=True)
        except (OSError, subprocess.SubprocessError) as error:
            self._logger.error(f"Cannot start a replacement of rank {rank}: {error!r}")
            self._count_failed_replacement(rank, f"it could not start: {error!r}", now_s)

    def _waits_for_gpu(self, rank: int, now_s: float) -> bool:
        """Return whether ``rank``'s replacement waits until no process runs on its GPU.

        A rank's process that this supervisor started has ended once it is reaped. One that a
        previous supervisor started ends through its parent-death signal, which proves nothing, so
        wait for the GPU, up to the rendezvous timeout. Then give the rank up.

        Any process on the GPU counts. Each rank holds a context only on its own GPU, and NVML may
        report the PIDs of another PID namespace. A GPU that NVML cannot check counts as a
        replacement that could not start.
        """
        if rank in self._processes:
            return False
        visible = self._environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            gpus = [str(index) for index in range(_local_gpu_count(self._environ))]
        else:
            gpus = [device for device in visible.split(",") if device]
        if not gpus:
            return False
        # The GPU that the rank's worker selects, as BaseWorker._configure_device does.
        gpu = gpus[rank % len(gpus)]
        try:
            pids = _gpu_compute_processes(gpu)
        except RuntimeError as error:
            self._logger.error(
                f"Cannot start a replacement of rank {rank}: cannot check that no process runs on "
                f"its GPU {gpu}: {error}"
            )
            self._count_failed_replacement(
                rank, f"its GPU {gpu} could not be checked: {error}", now_s
            )
            return True
        if not pids:
            return False
        timeout_s = self._membership.RENDEZVOUS_TIMEOUT_S
        since_s = self._gpu_in_use_since_s.setdefault(rank, now_s)
        if now_s >= since_s + timeout_s:
            self._logger.error(
                f"Processes {pids} still run on GPU {gpu} of rank {rank} after {timeout_s:.0f} s. "
                f"The deployment serves without rank {rank}"
            )
            self._give_up(rank)
            return True
        if since_s == now_s:
            self._logger.warning(
                f"The replacement of rank {rank} waits until processes {pids} end on its GPU {gpu}"
            )
        self._replacement_due_s[rank] = min(now_s + _BACKOFF_S, since_s + timeout_s)
        self._deadlines_s.append(self._replacement_due_s[rank])
        return True

    def _give_up(self, rank: int) -> None:
        self._given_up.add(rank)
        self._report()

    def _report(self) -> None:
        """Report the given-up ranks and the paused ranks to the authority.

        The authority publishes the given-up ranks in the record. The send never blocks, since a
        supervisor blocked on a stalled authority would stop supervising its ranks. While part of
        the last report is unsent, this sends only that part, and the next report interval sends a
        current one. Without a connection, the next connection reports once the authority admitted
        it.
        """
        connection = self._connection
        if connection is None:
            return
        if not self._unsent:
            report = self._membership.NodeReport(
                given_up=tuple(sorted(self._given_up)), paused_mask=self._paused_mask()
            )
            self._unsent = connection.lines.encode(report.model_dump_json().encode())
        try:
            sent = connection.sock.send(self._unsent)
        except BlockingIOError:
            return
        except OSError as error:
            self._logger.warning(f"Cannot report to the membership authority: {error!r}")
            self._close_connection(connection)
            return
        self._unsent = self._unsent[sent:]

    def _paused_mask(self) -> int:
        """Return the rank mask of the node's ranks whose hang detector is paused.

        A rank counts while its pause word holds the PID of a process that still exists and is not
        stopped, as by SIGSTOP or a debugger.
        """
        mask = 0
        for rank, path in self._pause_word_paths.items():
            pid = self._membership.read_pause_word(path)
            if pid and self._in_pause(rank, pid):
                mask |= 1 << rank
        return mask

    def _in_pause(self, rank: int, pid: int) -> bool:
        """Return whether process ``pid``, which ``rank``'s pause word names, can be in a pause.

        A process that no longer exists or is stopped is not. One whose state cannot be read
        otherwise, as under ``hidepid``, counts as not stopped.
        """
        try:
            with open(f"/proc/{pid}/stat", "rb") as stat:
                # The state follows the command name, which may hold spaces and parentheses.
                return stat.read().rsplit(b")", 1)[1].split()[0] not in (b"T", b"t")
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError as error:
            if not self._warned_unreadable_state:
                self._warned_unreadable_state = True
                self._logger.warning(
                    f"Cannot read the state of process {pid}, which rank {rank}'s pause word "
                    f"names, so it counts as not stopped: {error!r}"
                )
            return True

    def _reap(self, rank: int, now_s: float) -> None:
        process = self._processes[rank]
        self._selector.unregister(process.pidfd)
        os.close(process.pidfd)
        process.pidfd = None
        # Until the rank's process is reaped, its PID and so its process group's ID cannot be reused.
        # Kill the group now to end the helpers the rank left, then reap the rank.
        self._signal_group(process.child.pid, signal.SIGKILL)
        process.status = process.child.wait()
        self._logger.warning(
            f"Rank {rank} ({'replacement' if process.replacement else 'first process'}, "
            f"incarnation {process.incarnation}) ended with status {process.status}"
        )
        if process.status != 0 and self._first_failure is None:
            self._first_failure = _exit_status(process.status)
        healthy = (
            process.admitted_s is not None and now_s - process.admitted_s >= _HEALTHY_SERVING_S
        )
        if not process.replacement or healthy:
            self._failed_replacements.pop(rank, None)
            self._replacement_due_s.pop(rank, None)
            return
        if process.status == self._membership.REPLACEMENT_CANNOT_START:
            self._logger.error(
                f"The replacement of rank {rank} cannot start, nor could another one"
            )
            self._give_up(rank)
            return
        reason = (
            process.left_out_reason or f"it ended before serving for {_HEALTHY_SERVING_S:.0f} s"
        )
        self._count_failed_replacement(rank, reason, now_s)

    def _count_failed_replacement(self, rank: int, reason: str, now_s: float) -> None:
        """Count a replacement of ``rank`` that failed because of ``reason``.

        Give the rank up after too many in a row. Otherwise, start the next one after the backoff.
        """
        count = self._failed_replacements.get(rank, 0) + 1
        self._failed_replacements[rank] = count
        if count == _MAX_FAILED_REPLACEMENTS:
            self._logger.error(
                f"{count} replacements of rank {rank} failed in a row, the last because {reason}. "
                f"The deployment serves without rank {rank}"
            )
            self._give_up(rank)
            return
        self._replacement_due_s[rank] = now_s + _BACKOFF_S * 2 ** (count - 1)
        self._deadlines_s.append(self._replacement_due_s[rank])

    def _connect(self) -> None:
        """Start connecting to the authority, whose first view then arrives as an event.

        The connect wakes the loop when it ends, with a connection or an error. The first view may
        take up to the protocol's hello timeout. The supervisor keeps taking events meanwhile.
        """
        self._connect_due_s = time.monotonic() + _CONNECT_TIMEOUT_S
        self._connecting = concurrent.futures.Future()
        self._connecting.add_done_callback(lambda _: self._wake())
        # Off the main loop, so that a signal is acted on at once. The process exits without
        # waiting for a daemon thread.
        connector = threading.Thread(
            target=self._open_connection, args=(self._connecting,), daemon=True
        )
        connector.start()

    def _open_connection(self, connecting: concurrent.futures.Future) -> None:
        """Open the connection to the authority as the result of ``connecting``.

        Send no hello if the supervisor stopped while the store client connected, since the
        authority would count the node as connected. A supervisor that stops after this check may
        still send the hello. Its exit then closes the connection, which the authority treats as the
        end of the node's ranks, as the stop makes them end.
        """
        membership = self._membership
        try:
            store = membership.open_store(os.environ[membership.RENDEZVOUS_ENV], _CONNECT_TIMEOUT_S)
            if self._connecting is not connecting:
                raise ConnectionAbortedError("the node supervisor stopped")
            connection = membership.open_node_supervisor_connection(
                store, self._node, self._ranks, socket.gethostname(), _CONNECT_TIMEOUT_S
            )
        except Exception as error:
            # Any error, which the main loop raises again when it takes the connection.
            connecting.set_exception(error)
        else:
            connecting.set_result(connection)

    def _take_connection(self) -> None:
        """Take the connection to the authority once the connect ended."""
        if self._connecting is None or not self._connecting.done():
            return
        connecting, self._connecting = self._connecting, None
        membership = self._membership
        address = os.environ[membership.RENDEZVOUS_ENV]
        try:
            connection = connecting.result()
        except membership.UnsignedValueError as error:
            # This node's shared secret differs from the authority's, and no later connect can fix
            # that.
            self._logger.error(f"Cannot connect to the membership authority at {address}: {error}")
            self._refused_by_authority = True
            return
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            # Until the deployment serves, the store may not listen yet. Warn about those failures
            # once per interval.
            now_s = time.monotonic()
            quiet = not self._deployment_started and now_s < self._connect_warning_due_s
            if not quiet:
                self._connect_warning_due_s = now_s + _CONNECT_WARNING_INTERVAL_S
            (self._logger.debug if quiet else self._logger.warning)(
                f"Cannot connect to the membership authority at {address}: {error!r}"
            )
            self._authority_reached = False
            return
        self._authority_reached = True
        connection.sock.setblocking(False)
        self._connection_awaiting_view = connection
        self._view_due_s = time.monotonic() + membership.HELLO_TIMEOUT_S
        self._selector.register(connection.sock, selectors.EVENT_READ, _AUTHORITY)

    def _check_first_view(self, now_s: float) -> None:
        """Close the pending connection if the authority sent no view within the hello timeout.

        After the hello, the authority treats the closed connection as the node's ranks failing.
        """
        if self._connection_awaiting_view is None:
            return
        if now_s < self._view_due_s:
            self._deadlines_s.append(self._view_due_s)
            return
        self._logger.warning(
            "Cannot connect to the membership authority, which sent no view within "
            f"{self._membership.HELLO_TIMEOUT_S:.0f} s"
        )
        self._authority_reached = False
        self._close_connection(self._connection_awaiting_view)

    def _receive_view(self) -> None:
        connection = self._connection_awaiting_view or self._connection
        try:
            data = connection.sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = None
        if data:
            try:
                views = [
                    self._membership.NodeView.model_validate_json(payload)
                    for payload in connection.lines.decode(data)
                ]
            except ValueError as error:
                self._logger.error(
                    f"Received an invalid line from the membership authority: {error}"
                )
            else:
                if views and self._connection_awaiting_view is not None:
                    self._connection = self._connection_awaiting_view
                    self._connection_awaiting_view = None
                    # The authority gave up the node's ranks when the last connection closed.
                    self._report()
                for view in views:
                    self._on_view(view)
                return
        # Once the deployment has started, the authority has admitted this node's supervisor before.
        # It then refuses a reconnect, or a rejoin, only while it still holds the previous
        # connection, so connect again.
        if (
            data == b""
            and self._connection_awaiting_view is not None
            and not (self._deployment_started or self._rejoin)
        ):
            self._logger.error(
                f"The membership authority refused the supervisor of node {self._node}"
            )
            self._refused_by_authority = True
        else:
            self._logger.warning("The connection to the membership authority closed")
        self._close_connection(connection)

    def _close_connection(self, connection) -> None:
        self._selector.unregister(connection.sock)
        connection.sock.close()
        self._connection = self._connection_awaiting_view = None
        self._unsent = b""

    def _on_view(self, view) -> None:
        if self._view is not None:
            self._note_left_out(self._view.record, view.record)
        self._view = view
        self._view_received_s = time.monotonic()
        if view.record.state not in (
            self._membership.MembershipState.STARTING,
            self._membership.MembershipState.SHUTDOWN,
            self._membership.MembershipState.ABORTED,
        ):
            self._deployment_started = True

    def _note_left_out(self, previous, record) -> None:
        """Note why ``record`` leaves out a replacement that ``previous`` listed as joining."""
        for rank in self._ranks:
            process = self._processes.get(rank)
            if (
                process is not None
                and previous.is_joining(rank)
                and not record.is_active(rank)
                and record.reason
            ):
                process.left_out_reason = record.reason

    def _stop(
        self,
        status: int,
        end_job_reason: Optional[str] = None,
    ) -> int:
        """Stop the node's ranks, and return ``status``.

        With ``end_job_reason``, end the whole job as failed first. Unlink the node's shared memory.
        """
        # Before the job ends, since the launcher then kills this process.
        self._log_unlinked(self._unlink_shared_memory())
        # A connect whose store client is still connecting then sends no hello.
        self._connecting = None
        if end_job_reason is not None:
            self._end_job(end_job_reason)
        # Every rank not reaped yet, even one whose leader exited, may leave helpers in its group.
        ranks = [p for p in self._processes.values() if p.status is None]
        for process in ranks:
            self._signal_group(process.child.pid, signal.SIGTERM)
        self._wait_for(ranks)
        for process in ranks:
            self._signal_group(process.child.pid, signal.SIGKILL)
        # A rank stopped above may have created a segment after the first pass.
        self._log_unlinked(self._unlink_shared_memory())
        return status

    def _wait_for(self, processes: List[_Process]) -> None:
        """Wait up to the stop grace period for ``processes`` to end, without reaping them.

        Until a rank's process is reaped, its process group's ID cannot be reused.
        """
        deadline_s = time.monotonic() + _STOP_GRACE_S
        with selectors.DefaultSelector() as running:
            for process in processes:
                running.register(process.pidfd, selectors.EVENT_READ)
            while running.get_map() and time.monotonic() < deadline_s:
                for key, _ in running.select(max(0.0, deadline_s - time.monotonic())):
                    running.unregister(key.fileobj)

    def _end_job(self, reason: str) -> None:
        """End the whole job as failed on node 0 under ``mpirun``. Elsewhere, do nothing."""
        if self._end_job_fn is None:
            return
        self._logger.error(f"Ending the job: {reason}")
        try:
            self._end_job_fn()
        except OSError as error:
            self._logger.error(f"Cannot end the job: {error!r}")

    def _unlink_shared_memory(self) -> int:
        """Unlink the node's host weights and pause words, and return the number of segments.

        Only a supervisor that the authority admitted, or that started a rank, owns them. One that
        the authority refuses, such as a second supervisor started for a node whose supervisor
        still runs, leaves them to the running ranks.
        """
        if self._view is None and not self._processes:
            return 0
        host_weights = [
            os.path.join(_SHM_DIR, name)
            for name in os.listdir(_SHM_DIR)
            if name.startswith(self._host_weights_prefix)
        ]
        unlinked = 0
        for path in [*host_weights, *self._pause_word_paths.values()]:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
                unlinked += 1
        return unlinked

    def _log_unlinked(self, segments: int) -> None:
        if segments:
            self._logger.info(f"Unlinked the node's {segments} shared-memory segments")


def _local_gpu_count(environ: Dict[str, str]) -> int:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device])
    return len(glob.glob("/dev/nvidia[0-9]*"))


def _gpu_compute_processes(gpu: str) -> List[int]:
    """Return the PIDs of the compute processes on ``gpu``, an index or a UUID.

    Raises:
        RuntimeError: NVML cannot list them.
    """
    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError(f"cannot import pynvml: {error}") from error
    try:
        pynvml.nvmlInit()
        try:
            if gpu.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            else:
                handle = pynvml.nvmlDeviceGetHandleByUUID(gpu)
            return [process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError as error:
        raise RuntimeError(f"NVML failed: {error}") from error


def start_session_node_supervisor(n_workers: int, env: Dict[str, str]) -> subprocess.Popen:
    """Start the node supervisor of ranks 1 to ``n_workers - 1`` for a session that runs rank 0.

    Raises:
        RuntimeError: trtllm-llmapi-launch is not on PATH.
    """
    launcher = shutil.which("trtllm-llmapi-launch")
    if launcher is None:
        raise RuntimeError(
            "Wide-EP fault tolerance starts the ranks with trtllm-llmapi-launch, not on PATH"
        )
    return subprocess.Popen(  # nosec B603
        [
            sys.executable,
            _SCRIPT,
            "--node-rank",
            "0",
            "--nodes",
            "1",
            "--gpus-per-node",
            str(n_workers),
            "--rank-0-in-session",
            "--",
            launcher,
        ],
        env=env,
        # Out of the session's process group, so that a terminal's signal reaches only rank 0, which
        # then shuts the deployment down. The supervisor follows the record that ends it.
        start_new_session=True,
    )


def stop_session_node_supervisor(supervisor: subprocess.Popen, grace_s: Optional[float]) -> None:
    """Stop ``supervisor`` after waiting ``grace_s`` seconds for it to exit, unless that is None."""
    if grace_s is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            supervisor.wait(grace_s)
            return
    supervisor.terminate()
    # The supervisor waits one grace period for its ranks after the signal and one more after it sends
    # them SIGTERM. Allow a third for its own exit.
    try:
        supervisor.wait(3 * _STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        supervisor.kill()
        supervisor.wait()


def _exec_rank(supervisor: int, command: List[str]) -> NoReturn:
    """Run a rank's ``command`` in place of this process, to end once ``supervisor`` ends."""
    # The process guard sits next to this script, and this script runs without site packages.
    from _llmapi_process_guard import _arm_parent_death_signal

    # Only this signal ends the rank, and so frees its GPU, once its supervisor ends. A replacement
    # that the next supervisor starts would otherwise share the GPU with it.
    if not _arm_parent_death_signal():
        print("Cannot start the rank, which would outlive its node supervisor", file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    # The supervisor ended before the parent-death signal was armed.
    if os.getppid() != supervisor:
        os._exit(1)
    os.execvp(command[0], command)  # nosec B606


def main(argv: Optional[List[str]] = None) -> NoReturn:
    # Before the imports below, which take seconds, so that a parent ending meanwhile is noticed.
    parent = os.getppid()
    parser = argparse.ArgumentParser(
        prog="_llmapi_node_supervisor.py",
        description="Supervise the ranks of one node under Wide-EP fault tolerance",
    )
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, help="default: the GPUs this process sees")
    parser.add_argument(
        "--rank-0-in-session",
        action="store_true",
        help="start every rank but 0, which runs in the MpiPoolSession that starts this supervisor",
    )
    parser.add_argument(
        "--rejoin",
        action="store_true",
        help="return this node to a running deployment that lost it, with replacements of its ranks",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command of each rank>")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command of each rank after --")
    environ = dict(os.environ)
    # A Slurm step of one task, such as one that starts a lost node's supervisor again, ends with
    # that task alone.
    if (
        "SLURM_PROCID" in environ
        and environ.get("SLURM_STEP_NUM_TASKS") != "1"
        and "OMPI_COMM_WORLD_RANK" not in environ
        and not args.rank_0_in_session
    ):
        parser.error(
            "Wide-EP fault tolerance requires `mpirun --enable-recovery`, not srun, to start "
            "trtllm-llmapi-launch, because Slurm's PMIx plugin cancels the job step when one of its "
            "tasks exits"
        )
    visible_gpus = _local_gpu_count(environ)
    gpus_per_node = args.gpus_per_node or visible_gpus
    if gpus_per_node <= 0:
        parser.error("this process sees no GPU: give --gpus-per-node")
    # A replacement starts once no process runs on its rank's GPU, which a rank sharing it would hold.
    if gpus_per_node > visible_gpus > 0:
        parser.error(
            f"--gpus-per-node {gpus_per_node} is more than the {visible_gpus} GPUs this process "
            "sees: each rank needs a GPU of its own"
        )
    # This process uses neither MPI nor a GPU, so it skips the library initialization (MPI_Init).
    os.environ.update(
        TRT_LLM_NO_LIB_INIT="1", MPI4PY_RC_INITIALIZE="false", CUDA_VISIBLE_DEVICES=""
    )
    from tensorrt_llm._torch.distributed.membership import (
        NODE_SUPERVISOR_LAUNCH,
        RENDEZVOUS_ENV,
        SHARED_SECRET_ENV,
        parse_shared_secret,
        remove_cpu_affinity_constraints,
        rendezvous_host_port,
    )
    from tensorrt_llm.llmapi._mpirun_job import end_job_function

    local_size = environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")
    if local_size != "1":
        parser.error(
            f"mpirun started {local_size} node supervisors on this node, so start one task per "
            f"node: {NODE_SUPERVISOR_LAUNCH}"
        )
    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV):
        if not environ.get(name):
            parser.error(
                f"export the same {name} on every node and start one task per node: "
                f"{NODE_SUPERVISOR_LAUNCH}"
            )
    # Check both values before any rank starts. Every connect with a malformed value fails, and the
    # supervisor would keep retrying.
    try:
        parse_shared_secret(environ[SHARED_SECRET_ENV])
    except ValueError as error:
        parser.error(str(error))
    try:
        rendezvous_host_port(environ[RENDEZVOUS_ENV])
    except ValueError:
        parser.error(f"{RENDEZVOUS_ENV} must be host:port, not {environ[RENDEZVOUS_ENV]!r}")
    end_job = None
    if args.node_rank == 0:
        try:
            end_job = end_job_function(environ, args.nodes)
        except ValueError as error:
            parser.error(str(error))
    remove_cpu_affinity_constraints()
    supervisor = NodeSupervisor(
        args.node_rank,
        args.nodes,
        gpus_per_node,
        command,
        environ,
        rank_0_in_session=args.rank_0_in_session,
        end_job=end_job,
        rejoin=args.rejoin,
        parent=parent,
    )
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, supervisor.on_signal)
    status = supervisor.run()
    # Skip the interpreter shutdown. A connect thread that leaves the store client during the
    # shutdown aborts the process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    if sys.argv[1:2] == [_EXEC_RANK]:
        _exec_rank(int(sys.argv[2]), sys.argv[3:])
    # The script's directory must not shadow the modules TensorRT-LLM imports.
    del sys.path[0]
    main()
