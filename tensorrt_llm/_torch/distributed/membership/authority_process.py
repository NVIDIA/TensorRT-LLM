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
"""Membership authority process for Wide-EP fault tolerance.

:class:`AuthorityProcess` runs an :class:`~.authority.Authority`, serves the liveness connections
from ranks and node supervisors, and publishes each new record in the rendezvous store.
"""

import argparse
import hmac
import logging
import logging.handlers
import os
import queue
import resource
import selectors
import signal
import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._wide_ep_fault_tolerance import RENDEZVOUS_ENV, SHARED_SECRET_ENV
from tensorrt_llm.logger import logger

from .authority import Authority
from .liveness import Lines, configure_connection, hello_token
from .records import (
    HEARTBEAT_INTERVAL_S,
    HELLO_TIMEOUT_S,
    MAX_RANKS,
    NODE_SUPERVISOR_TIMEOUT_S,
    REPORT_INTERVAL_S,
    MembershipRecord,
    MembershipState,
    NodeReport,
    NodeView,
    RankReport,
    all_ranks,
    has_rank,
    message_bytes,
    rank_mask,
    remove_cpu_affinity_constraints,
)
from .rendezvous import (
    LIVENESS_KEY,
    RECORD_KEY,
    STALLED_SINCE_KEY,
    TRANSACTION_KEY,
    rendezvous_address,
    rendezvous_host_port,
)
from .signed_store import SignedStore

# A peer with more unsent bytes queued than this reads nothing, so its connection is closed.
_MAX_UNSENT_BYTES = 1 << 22
# Selector data for rank 0's pipe. End of file on it means rank 0's process ended; any bytes
# written to it are read and ignored.
_PARENT = object()


def _decided_at_once(report: RankReport) -> tuple:
    """Return the report fields that the authority acts on as soon as they change."""
    return (
        report.reply_generation,
        report.reply,
        report.joining_ready,
        report.started,
        report.shutdown,
    )


@dataclass
class _Connection:
    """A liveness connection the authority accepted."""

    accepted_s: float
    sent_s: float = 0.0
    received: bytes = b""
    member: Optional[tuple[int, int]] = None  # (rank, incarnation)
    node: Optional[int] = None
    node_ranks: tuple[int, ...] = ()
    lines: Optional[Lines] = None
    report: Optional[RankReport] = None
    node_report: Optional[NodeReport] = None
    # The node report is older than half the hang timeout, so the authority ignores it.
    node_report_stale: bool = False
    unsent: bytes = b""


class AuthorityProcess:
    """Run :class:`Authority` and publish every record in the store and on every connection.

    A replacement is admitted only for a rank that the membership no longer lists and that has no
    connected process, so only the authority's verdict removes a process, and each rank has at most
    one connection. The deployment aborts once ``parent``, held by rank 0's process, ends.
    """

    def __init__(
        self,
        world_size: int,
        gpus_per_node: int,
        store: SignedStore,
        authority: Authority,
        parent: Optional[int] = None,
    ) -> None:
        """Prepare to serve. :meth:`run` listens and publishes the first record.

        Args:
            world_size: The deployment's ranks.
            gpus_per_node: The ranks per node, by which a missing node supervisor is named.
            store: The rendezvous store, which this process hosts.
            authority: The authority that decides the membership.
            parent: The read end of a pipe whose write end rank 0's process holds, or None.
        """
        self.world_size = world_size
        self._gpus_per_node = gpus_per_node
        self._store = store
        self._authority = authority
        self._hello_token = hello_token().encode()
        self._parent = parent
        self._parent_ended_s: Optional[float] = None
        self._changed = False
        self._incarnation = [0] * world_size
        self._connections: dict[socket.socket, _Connection] = {}
        self._selector = selectors.DefaultSelector()
        self._record: Optional[MembershipRecord] = None
        # Ranks whose node supervisor connected, and when the authority started.
        self._supervised_mask = 0
        self._started_s = 0.0
        self._sent_stalled_replacements: tuple[int, ...] = ()
        self._sent_connected_mask = 0
        self._published_stalled_since_s: Optional[float] = None
        # The ranks that no replacement is started for, which every published record carries.
        self._given_up_mask = 0

    def run(self, host: str) -> MembershipRecord:
        """Serve on ``host`` until the record is ABORTED or SHUTDOWN, and return that record."""
        try:
            with socket.create_server((host, 0)) as listener:
                listener.setblocking(False)
                self._selector.register(listener, selectors.EVENT_READ)
                if self._parent is not None:
                    self._selector.register(self._parent, selectors.EVENT_READ, _PARENT)
                self._publish(self._authority.record)
                self._started_s = time.monotonic()
                self._store.set(
                    LIVENESS_KEY,
                    f"{host}:{listener.getsockname()[1]}".encode(),
                )
                try:
                    return self._serve()
                except Exception as error:
                    # The store ends with this process, so the ranks learn why from their
                    # connections.
                    record = self._authority.abort(f"the authority failed: {error!r}")
                    self._record = self._stamped(record)
                    self._send_record()
                    raise
        finally:
            for sock in list(self._connections):
                sock.close()
            self._selector.close()

    def _serve(self) -> MembershipRecord:
        decide_s = time.monotonic()
        while True:
            self._poll(max(0.0, decide_s - time.monotonic()))
            now_s = time.monotonic()
            if not self._changed and now_s < decide_s:
                continue
            if now_s - decide_s > HEARTBEAT_INTERVAL_S:
                logger.warning(
                    f"The membership authority decided {now_s - decide_s:.1f} s late, so its "
                    "heartbeats to the ranks were late too"
                )
            self._changed = False
            decide_s = now_s + REPORT_INTERVAL_S
            self._send_heartbeats(time.monotonic())
            summary = self._authority.last_transaction
            if self._parent_ended_s is not None:
                record = self._authority.abort("rank 0's process ended")
            else:
                record = self._check_supervisors(time.monotonic())
            connected = self._connected()
            if record is None:
                tick_s = time.monotonic()
                record = self._authority.tick(
                    tick_s, self._reports(), connected, self._node_reports(tick_s)
                )
            self._publish_stall()
            if record is None:
                if self._record.given_up_mask != self._given_up_mask:
                    # Resend the record to the ranks only once committed. During a change a rank
                    # would take a resent record as the change's next step.
                    self._publish(
                        self._record, ranks=self._record.state == MembershipState.COMMITTED
                    )
                elif (
                    self._authority.stalled_replacements != self._sent_stalled_replacements
                    or connected != self._sent_connected_mask
                ):
                    self._send_record(ranks=False)
                continue
            logger.info(f"Membership: {record!r}")
            transaction = self._authority.last_transaction
            if transaction is not summary:
                logger.info(f"Membership transaction: {transaction!r}")
                self._store.set(TRANSACTION_KEY, transaction.model_dump_json().encode())
            record = self._publish(record)
            if record.state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
                # The store lives in this process, so keep it until every active rank has read the
                # record and left, for at most a hang timeout.
                ended_s = time.monotonic()
                while self._connected() & record.active_mask:
                    if time.monotonic() - ended_s >= self._authority.hang_timeout_s:
                        break
                    # Ranks that have not left yet take a silent authority as hung.
                    self._send_heartbeats(time.monotonic())
                    self._poll(REPORT_INTERVAL_S)
                return record

    def _check_supervisors(self, now_s: float) -> Optional[MembershipRecord]:
        """Abort a startup in which some nodes' supervisors did not connect within the timeout.

        Return the ABORTED record, or None if the startup goes on.
        """
        unsupervised = all_ranks(self.world_size) & ~self._supervised_mask
        if (
            self._authority.record.state != MembershipState.STARTING
            or not unsupervised
            or now_s - self._started_s < NODE_SUPERVISOR_TIMEOUT_S
        ):
            return None
        nodes = sorted(
            {
                rank // self._gpus_per_node
                for rank in range(self.world_size)
                if has_rank(unsupervised, rank)
            }
        )
        return self._authority.abort(
            f"the supervisors of nodes {nodes} did not connect within "
            f"{NODE_SUPERVISOR_TIMEOUT_S:.0f} s (one per node, with {SHARED_SECRET_ENV} and "
            f"{RENDEZVOUS_ENV} set)"
        )

    def _stamped(self, record: MembershipRecord) -> MembershipRecord:
        """Return ``record`` with the current incarnations and given-up ranks."""
        return record.model_copy(
            update={"incarnations": tuple(self._incarnation), "given_up_mask": self._given_up_mask}
        )

    def _publish(self, record: MembershipRecord, ranks: bool = True) -> MembershipRecord:
        record = self._stamped(record)
        self._store.set(RECORD_KEY, record.model_dump_json().encode())
        self._record = record
        self._send_record(ranks)
        return record

    def _publish_stall(self) -> None:
        """Publish when the active ranks started to wait on a stalled rank, in Unix milliseconds."""
        stalled_since_s = self._authority.stalled_since_s
        if stalled_since_s == self._published_stalled_since_s:
            return
        self._published_stalled_since_s = stalled_since_s
        if stalled_since_s is None:
            value = b""
        else:
            stalled_for_s = time.monotonic() - stalled_since_s
            value = str(round((time.time() - stalled_for_s) * 1e3)).encode()
        self._store.set(STALLED_SINCE_KEY, value)

    def _send_record(self, ranks: bool = True) -> None:
        record = message_bytes(self._record)
        self._sent_stalled_replacements = self._authority.stalled_replacements
        self._sent_connected_mask = self._connected()
        view = message_bytes(
            NodeView(
                record=self._record,
                stalled_replacements=self._sent_stalled_replacements,
                connected_mask=self._sent_connected_mask,
            )
        )
        for sock, connection in list(self._connections.items()):
            # A connection an earlier send in this loop dropped is gone.
            if sock in self._connections and connection.lines is not None:
                if connection.node is not None or ranks:
                    self._send(sock, connection, record if connection.node is None else view)

    def _send_heartbeats(self, now_s: float) -> None:
        """Send an empty line to each rank that has received nothing for the heartbeat interval."""
        for sock, connection in list(self._connections.items()):
            if (
                sock in self._connections
                and connection.member is not None
                and now_s - connection.sent_s >= HEARTBEAT_INTERVAL_S
            ):
                self._send(sock, connection, b"")

    def _send(self, sock: socket.socket, connection: _Connection, payload: bytes) -> None:
        connection.unsent += connection.lines.encode(payload)
        connection.sent_s = time.monotonic()
        self._flush(sock, connection)

    def _flush(self, sock: socket.socket, connection: _Connection) -> None:
        try:
            sent = sock.send(connection.unsent)
        except BlockingIOError:
            sent = 0
        except OSError:
            self._drop(sock)
            return
        connection.unsent = connection.unsent[sent:]
        if len(connection.unsent) > _MAX_UNSENT_BYTES:
            if connection.node is None:
                logger.warning(
                    f"Closing the connection of {connection.member}, which reads nothing"
                )
            self._drop(sock, problem="it reads nothing")
            return
        write = selectors.EVENT_WRITE if connection.unsent else 0
        self._selector.modify(sock, selectors.EVENT_READ | write, connection)

    def _poll(self, timeout_s: float) -> None:
        for key, events in self._selector.select(timeout_s):
            sock, connection = key.fileobj, key.data
            if connection is None:
                self._accept(sock)
                continue
            if connection is _PARENT:
                if not os.read(sock, 1 << 12):
                    self._selector.unregister(sock)
                    self._parent_ended_s = time.monotonic()
                    self._changed = True
                continue
            if sock not in self._connections:
                continue
            if events & selectors.EVENT_WRITE:
                self._flush(sock, connection)
            if events & selectors.EVENT_READ and sock in self._connections:
                self._receive(sock, connection)
        now_s = time.monotonic()
        for sock, connection in list(self._connections.items()):
            if (
                sock in self._connections
                and connection.lines is None
                and now_s - connection.accepted_s > HELLO_TIMEOUT_S
            ):
                logger.warning("Closing a liveness connection without a valid hello")
                self._drop(sock)

    def _accept(self, listener: socket.socket) -> None:
        """Accept a connection, dropping the oldest of ``MAX_RANKS`` that wait for their hello."""
        waiting = [sock for sock, c in self._connections.items() if c.lines is None]
        if len(waiting) >= MAX_RANKS:
            logger.warning("Closing the oldest liveness connection waiting for its hello")
            self._drop(waiting.pop(0))
        try:
            sock, _ = listener.accept()
        except BlockingIOError:
            return
        except OSError as error:  # out of descriptors (EMFILE, ENFILE) or memory
            logger.warning(f"Cannot accept a liveness connection: {error!r}")
            if waiting:
                self._drop(waiting[0])
            return
        sock.setblocking(False)
        configure_connection(sock)
        self._connections[sock] = _Connection(time.monotonic())
        self._selector.register(sock, selectors.EVENT_READ, self._connections[sock])

    def _receive(self, sock: socket.socket, connection: _Connection) -> None:
        try:
            data = sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._drop(sock)
            return
        try:
            if connection.lines is None:
                data = self._read_hello(sock, connection, data)
                # The connection is gone if sending the first record failed.
                if sock not in self._connections:
                    return
            for line in connection.lines.decode(data) if data else ():
                if connection.node is not None:
                    self._on_node_report(connection, NodeReport.model_validate_json(line))
                    continue
                if connection.member is None:
                    raise ValueError("a report arrived on a liveness connection of no rank")
                report = RankReport.model_validate_json(line).model_copy(
                    update={"incarnation": connection.member[1], "received_s": time.monotonic()}
                )
                if connection.report is None or _decided_at_once(
                    connection.report
                ) != _decided_at_once(report):
                    self._changed = True
                connection.report = report
        except ValueError as error:
            if connection.node is None:
                logger.warning(f"Closing a liveness connection: {error}")
            self._drop(sock, problem=f"it sent an invalid line: {error}")

    def _read_hello(self, sock: socket.socket, connection: _Connection, data: bytes) -> bytes:
        """Admit the peer whose hello line ``data`` completes and return the bytes after it.

        A rank sends ``<token> <rank> <1 if a replacement, else 0>`` and a node supervisor sends
        ``<token> node <node> <rank,rank,...> <host>``.
        """
        connection.received += data
        if b"\n" not in connection.received:
            if len(connection.received) > 256:
                raise ValueError("no hello")
            return b""
        line, rest = connection.received.split(b"\n", 1)
        token, *announcement = line.split()
        if not hmac.compare_digest(token, self._hello_token):
            raise ValueError("the hello does not carry this deployment's token")
        announcement = [field.decode() for field in announcement]
        if announcement[:1] == ["node"]:
            _, node, ranks, host = announcement
            node, ranks = int(node), tuple(int(rank) for rank in ranks.split(","))
            refusal = self._node_refusal(node, ranks)
            if refusal is not None:
                raise ValueError(refusal)
            connection.node, connection.node_ranks, connection.lines = node, ranks, Lines()
            self._supervised_mask |= rank_mask(ranks, self.world_size)
            logger.info(f"The supervisor of node {node} ({host}, ranks {list(ranks)}) connected")
            self._send_record(ranks=False)
            return rest
        rank, replacement = (int(field) for field in announcement)
        refusal = self._refusal(rank, replacement)
        if refusal is not None:
            raise ValueError(refusal)
        if replacement:
            self._incarnation[rank] += 1
            self._authority.request_join(rank)
        connection.member = (rank, self._incarnation[rank])
        self._changed = True
        connection.lines = Lines()
        if replacement:
            # The stored record and the node views name the replacement's process from now on. The
            # ranks get no resent record, which they would take as the change's next step.
            self._publish(self._record, ranks=False)
        self._send(sock, connection, message_bytes(self._record))
        return rest

    def _on_node_report(self, connection: _Connection, report: NodeReport) -> None:
        """Keep ``report`` from the supervisor on ``connection`` and take the ranks it gave up."""
        if not set(report.given_up) <= set(connection.node_ranks):
            raise ValueError(
                f"node {connection.node} gave up ranks {list(report.given_up)}, which are not all "
                "its own"
            )
        node_mask = rank_mask(connection.node_ranks, self.world_size)
        if report.paused_mask & ~node_mask:
            raise ValueError(
                f"node {connection.node} reported the ranks of mask {report.paused_mask:#x} "
                "paused, which are not all its own"
            )
        connection.node_report = report.model_copy(update={"received_s": time.monotonic()})
        given_up_mask = (self._given_up_mask & ~node_mask) | rank_mask(
            report.given_up, self.world_size
        )
        # The supervisor reports every report interval, and a change in the given-up ranks is
        # decided at once.
        if given_up_mask != self._given_up_mask:
            self._given_up_mask = given_up_mask
            self._changed = True

    def _node_refusal(self, node: int, ranks: tuple[int, ...]) -> Optional[str]:
        if not ranks or not all(0 <= rank < self.world_size for rank in ranks):
            return f"node {node}'s ranks {list(ranks)} are not all in the deployment"
        for c in self._connections.values():
            if c.node == node or set(c.node_ranks) & set(ranks):
                return f"node {node} or one of its ranks {list(ranks)} has a supervisor already"
        return None

    def _refusal(self, rank: int, replacement: int) -> Optional[str]:
        """Return why a hello from ``rank`` is refused, or None to admit it.

        A replacement is admitted only once the membership no longer lists the rank, so its
        connection never hides that its predecessor ended. The startup's record lists every rank,
        so no replacement joins before the deployment has started.
        """
        if not 0 <= rank < self.world_size:
            return f"rank {rank} is not in the deployment"
        if replacement:
            if rank == 0:
                return "rank 0 is never replaced"
            if self._authority.record.is_active(rank):
                return f"rank {rank} is active: no replacement joins for it"
        elif self._authority.record.state != MembershipState.STARTING:
            return f"rank {rank} joins only as a replacement once the deployment has started"
        # A first process and a replacement alike join only while the rank has no connected process.
        if any(c.member is not None and c.member[0] == rank for c in self._connections.values()):
            return f"rank {rank} has a process connected already"
        return None

    def _drop(self, sock: socket.socket, problem: str = "") -> None:
        """Close ``sock``, and fail the node's ranks if it is a supervisor's connection.

        No replacement for those ranks starts until their node's supervisor connects again, so the
        authority publishes them as given up.

        Args:
            problem: Why the authority closes the connection, or empty if the peer closed it.
        """
        if sock not in self._connections:
            return
        self._selector.unregister(sock)
        connection = self._connections.pop(sock)
        sock.close()
        self._changed = True
        if connection.node is None:
            return
        if self._record.state not in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
            ranks = list(connection.node_ranks)
            if problem:
                logger.error(
                    f"Closing the connection of the supervisor of node {connection.node}, "
                    f"because {problem}. Its ranks {ranks} count as failed"
                )
            else:
                logger.warning(
                    f"The supervisor of node {connection.node} left, "
                    f"so its ranks {ranks} count as failed"
                )
            self._given_up_mask |= rank_mask(ranks, self.world_size)
            for other, peer in list(self._connections.items()):
                if peer.member is not None and peer.member[0] in connection.node_ranks:
                    self._drop(other)
        self._send_record(ranks=False)

    def _connected(self) -> int:
        return rank_mask(
            (c.member[0] for c in self._connections.values() if c.member is not None),
            self.world_size,
        )

    def _reports(self) -> dict[int, RankReport]:
        return {c.member[0]: c.report for c in self._connections.values() if c.report is not None}

    def _node_reports(self, now_s: float) -> list[NodeReport]:
        """Return each node supervisor's latest report, and log when one goes stale or resumes.

        The authority ignores a report older than half the hang timeout, so the node's paused ranks
        lose their exemption from the hang rule until the supervisor reports again.
        """
        stale_s = self._authority.hang_timeout_s / 2
        reports = []
        for c in self._connections.values():
            if c.node_report is None:
                continue
            stale = now_s - c.node_report.received_s > stale_s
            if stale != c.node_report_stale:
                c.node_report_stale = stale
                if stale:
                    logger.warning(
                        f"The supervisor of node {c.node} has not reported for {stale_s:g} s, so "
                        "its paused ranks are no longer exempt from the hang rule"
                    )
                else:
                    logger.info(
                        f"The supervisor of node {c.node} reports again, so its paused ranks are "
                        "exempt from the hang rule again"
                    )
            reports.append(c.node_report)
        return reports


def log_from_own_thread() -> logging.handlers.QueueListener:
    """Write this process's log records from their own thread, and return its listener.

    The authority shares rank 0's standard output, whose writes can block. A blocked write must not
    hold up the heartbeats.
    """
    python_logger = logging.getLogger(logger.PREFIX)
    records = queue.SimpleQueue()
    listener = logging.handlers.QueueListener(
        records, *python_logger.handlers, respect_handler_level=True
    )
    python_logger.handlers = [logging.handlers.QueueHandler(records)]
    listener.start()
    return listener


def main(argv: Optional[list[str]] = None) -> int:
    """Run the authority process.

    Start the rendezvous store at the address in ``TLLM_FAULT_TOLERANCE_RENDEZVOUS``, then write
    that address and a newline to ``--address-fd`` to tell the parent that the store is up. Abort
    the deployment once rank 0's process closes the standard input.

    Args:
        argv: The command-line arguments, by default ``sys.argv[1:]``.

    Returns:
        The exit status: 0 once the deployment shut down, 1 once it aborted.
    """
    parser = argparse.ArgumentParser(description="Wide-EP fault-tolerance membership authority")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--address-fd", type=int, required=True)
    args = parser.parse_args(argv)
    remove_cpu_affinity_constraints()
    # This process holds a liveness and a store connection per rank and per node supervisor, and
    # the connections that wait for their hello, which can exceed a soft limit of 1024 open files.
    _, max_open_files = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max_open_files, max_open_files))
    log_listener = log_from_own_thread()

    from torch.distributed import TCPStore

    address = rendezvous_address()
    host, port = rendezvous_host_port(address)
    # Like the liveness listener, the store listens only on the address the deployment uses.
    listener = socket.create_server((host, port))
    store = SignedStore(
        TCPStore(
            host,
            port,
            is_master=True,
            wait_for_workers=False,
            master_listen_fd=listener.detach(),
        )
    )
    os.write(args.address_fd, f"{address}\n".encode())
    os.close(args.address_fd)
    authority = Authority(args.world_size)
    # A signal for the deployment (Ctrl-C in its terminal) is its ranks' to act on.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    process = AuthorityProcess(
        args.world_size, args.gpus_per_node, store, authority, parent=sys.stdin.fileno()
    )
    try:
        record = process.run(host)
    finally:
        log_listener.stop()
    return 0 if record.state == MembershipState.SHUTDOWN else 1
