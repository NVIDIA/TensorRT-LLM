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
"""Wide-EP fault-tolerance status for the server's status endpoint and metrics."""

from __future__ import annotations

import socket
import threading
import time
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    get_args,
)

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tensorrt_llm._torch.distributed.membership import (
    AUTHORITY_SILENCE_TIMEOUT_S,
    HANG_TIMEOUT_S,
    RECORD_KEY,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    STALLED_SINCE_KEY,
    TRANSACTION_KEY,
    MembershipRecord,
    MembershipState,
    TransactionSummary,
    UnsignedValueError,
    open_store,
    parse_shared_secret,
    rendezvous_host_port,
)
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from tensorrt_llm._torch.distributed.membership import SignedStore

_STATUS_READ_TIMEOUT_S = 1.0
_STATUS_REFRESH_PERIOD_S = 1.0
# How long closing waits for a read in progress, whose few store requests each time out after
# _STATUS_READ_TIMEOUT_S.
_CLOSE_TIMEOUT_S = 10.0
# Report "stalled" once the active ranks have waited this long on a rank that makes no progress.
# Half the hang timeout keeps one long step, such as a long prefill, from reading as stalled.
_STALLED_STATE_AFTER_S = HANG_TIMEOUT_S / 2

# Routes answered from the status. With several serving frontends, only frontend 0 reads the store
# and the others forward these routes to it, so every scrape sees one series. With
# return_perf_metrics, its Prometheus endpoint also reports the metrics from all frontends.
FORWARDED_ROUTES = (
    ("GET", "/fault_tolerance_status"),
    ("GET", "/prometheus/metrics{rest:path}"),
)


FaultToleranceState = Literal[
    "healthy", "degraded", "stalled", "recovering", "unprotected", "failed", "stopped"
]
ReplacementState = Literal["pending", "given_up"]


class FaultToleranceStatus(BaseModel):
    """The membership status that the server's status endpoint answers."""

    state: FaultToleranceState
    authority_alive: bool
    # How long the active ranks have waited on a rank that makes no progress.
    stalled_seconds: float
    # The fields below are None until a membership record was read.
    generation: Optional[int] = None
    active_ranks: Optional[List[int]] = None
    failed_ranks: Optional[List[int]] = None
    replacements: Optional[Dict[int, ReplacementState]] = None
    incarnations: Optional[List[int]] = None
    last_transaction: Optional[TransactionSummary] = None


_Generational = TypeVar("_Generational", MembershipRecord, TransactionSummary)


def _latest(
    last: Optional[_Generational], read: Optional[_Generational]
) -> Optional[_Generational]:
    """Return ``read`` unless it is None or of an older generation than ``last``."""
    if read is None or (last is not None and read.generation < last.generation):
        return last
    return read


class FaultToleranceHealth:
    """Read the membership state for the server's status endpoint and metrics."""

    def __init__(
        self,
        address: str,
        secret: str,
        metric_labels: Optional[Mapping[str, str]] = None,
        on_failed: Optional[Callable[[], None]] = None,
    ):
        """Prepare to read the membership status from the rendezvous store at ``address``.

        The store is read only by :meth:`read_status` until :meth:`start` starts the refresh thread.

        Args:
            address: The rendezvous store's ``host:port``.
            secret: The deployment's shared secret in hex, which signs the store's values.
            metric_labels: Label values by label name, added to every metric.
            on_failed: Called once, from the reading thread, when the status first reads ``failed``.

        Raises:
            ValueError: ``secret`` is not a hex secret of the length the ranks require.
        """
        parse_shared_secret(secret)
        self._address = address
        self._secret = secret
        self._metric_labels = dict(metric_labels or {})
        self._on_failed = on_failed
        self._store: Optional[SignedStore] = None
        self._record: Optional[MembershipRecord] = None
        self._transaction: Optional[TransactionSummary] = None
        self._stalled_since_unix_ms: Optional[float] = None
        self._next_open = 0.0
        self._answered_s: Optional[float] = None
        self._authority_ended = False
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._last_status = self._status(authority_alive=False)
        self._refresh_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Read the status once, then start the thread that refreshes it."""
        # Read once, so that a server answers its first probe with the membership in the store.
        self.read_status()
        self._refresh_thread = threading.Thread(
            target=self._refresh, name="FaultToleranceHealth", daemon=True
        )
        self._refresh_thread.start()

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str],
        metric_labels: Mapping[str, str],
        on_failed: Callable[[], None],
    ) -> FaultToleranceHealth:
        """Read the rendezvous store named in the ranks' environment ``env``.

        Args:
            env: The environment variables the ranks start with.
            metric_labels: Label values by label name, added to every metric.
            on_failed: Called once when the deployment is found failed.
        """
        return cls(env[RENDEZVOUS_ENV], env[SHARED_SECRET_ENV], metric_labels, on_failed)

    def _read_store(self) -> bool:
        """Read the membership state from the store, and return whether the authority is alive.

        The authority counts as alive while the store answered within the timeout after which the
        ranks consider it hung. A failed read keeps the last state and reopens the store at most
        once per read timeout. Once a record was read, a refused connection means the authority
        ended, and every later read returns False at once. A record or transaction older than the
        last one read is ignored.
        """
        with self._lock:
            if self._authority_ended or self._closed.is_set():
                return False
            try:
                if self._store is None:
                    if time.monotonic() < self._next_open:
                        return self._answered_recently()
                    self._next_open = time.monotonic() + _STATUS_READ_TIMEOUT_S
                    self._store = open_store(self._address, _STATUS_READ_TIMEOUT_S, self._secret)
                record, transaction, stalled_since = (
                    self._get_if_set(key)
                    for key in (RECORD_KEY, TRANSACTION_KEY, STALLED_SINCE_KEY)
                )
                self._record = _latest(
                    self._record, self._parse(MembershipRecord.model_validate_json, record)
                )
            except (RuntimeError, OSError):
                self._store = None
                # Before a record was read, a refused port means the authority has not started.
                self._authority_ended = self._record is not None and self._store_refuses()
                return not self._authority_ended and self._answered_recently()
            except UnsignedValueError as error:
                # A value that the deployment did not sign keeps the last state read.
                logger.error(f"Ignored the rendezvous store's values: {error}")
                return self._answered_recently()
            self._transaction = _latest(
                self._transaction, self._parse(TransactionSummary.model_validate_json, transaction)
            )
            # The authority clears the stall with an empty value.
            self._stalled_since_unix_ms = self._parse(float, stalled_since or None)
            self._answered_s = time.monotonic()
            return True

    def _get_if_set(self, key: str) -> Optional[bytes]:
        return self._store.get(key) if self._store.check([key]) else None

    def _answered_recently(self) -> bool:
        return (
            self._answered_s is not None
            and time.monotonic() - self._answered_s <= AUTHORITY_SILENCE_TIMEOUT_S
        )

    def _store_refuses(self) -> bool:
        """Return whether the store's port refuses connections, so no process listens on it."""
        host, port = rendezvous_host_port(self._address)
        try:
            socket.create_connection((host, port), _STATUS_READ_TIMEOUT_S).close()
        except ConnectionRefusedError:
            return True
        except OSError:
            pass
        return False

    @staticmethod
    def _parse(parse: Callable[[bytes], object], value: Optional[bytes]):
        if value is None:
            return None
        try:
            return parse(value)
        except ValueError as error:
            logger.error(f"Cannot parse a value of the rendezvous store: {error!r}")
            return None

    def _state(self, authority_alive: bool, stalled_seconds: float) -> FaultToleranceState:
        record = self._record
        if record is None:
            return "unprotected"
        # After the deployment ends, the authority keeps its store up until every active rank left.
        if record.state == MembershipState.SHUTDOWN:
            return "stopped"
        if self._authority_ended or record.state == MembershipState.ABORTED:
            return "failed"
        if not authority_alive:
            return "unprotected"
        if record.state != MembershipState.COMMITTED:
            return "recovering"
        if stalled_seconds >= _STALLED_STATE_AFTER_S:
            return "stalled"
        world = (1 << len(record.incarnations)) - 1
        return "healthy" if record.active_mask == world else "degraded"

    def _status(self, authority_alive: bool) -> FaultToleranceStatus:
        """Return the status of the membership state read last."""
        record = self._record
        stalled_since = self._stalled_since_unix_ms
        stalled_seconds = (
            0.0 if stalled_since is None else max(0.0, time.time() - stalled_since / 1e3)
        )
        membership = {}
        if record is not None:
            world_size = len(record.incarnations)
            failed_ranks = [r for r in range(world_size) if not record.is_active(r)]
            membership = dict(
                generation=record.generation,
                active_ranks=[r for r in range(world_size) if record.is_active(r)],
                failed_ranks=failed_ranks,
                replacements={
                    r: "given_up" if record.is_given_up(r) else "pending" for r in failed_ranks
                },
                incarnations=list(record.incarnations),
            )
        return FaultToleranceStatus(
            state=self._state(authority_alive, stalled_seconds),
            authority_alive=authority_alive,
            stalled_seconds=stalled_seconds,
            last_transaction=self._transaction,
            **membership,
        )

    def read_status(self) -> FaultToleranceStatus:
        """Read the membership from the store, waiting for it, and return the status.

        Probes call :meth:`last_status`, which never waits for the store.
        """
        status = self._status(self._read_store())
        self._last_status = status
        return status

    def last_status(self) -> FaultToleranceStatus:
        """Return the last status read, without waiting for the store.

        Once started, a thread reads the status every :data:`_STATUS_REFRESH_PERIOD_S`, so the
        result is at most that period plus one read old. Each store call times out after
        :data:`_STATUS_READ_TIMEOUT_S`.
        """
        return self._last_status

    async def status_response(self) -> JSONResponse:
        """Answer the server's status endpoint with the last status read."""
        return JSONResponse(content=self.last_status().model_dump(mode="json"))

    def register_metrics(self, registry: CollectorRegistry) -> None:
        """Add the fault-tolerance metrics to ``registry``."""
        registry.register(FaultToleranceMetricsCollector(self, self._metric_labels))

    def close(self) -> None:
        """Stop reading the status, and wait for a read in progress to end."""
        self._closed.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(_CLOSE_TIMEOUT_S)

    def _refresh(self) -> None:
        try:
            while not self._closed.wait(_STATUS_REFRESH_PERIOD_S):
                if self.read_status().state == "failed" and self._on_failed is not None:
                    on_failed, self._on_failed = self._on_failed, None
                    on_failed()
        finally:
            if not self._closed.is_set():
                # A read raised, so report the store as unreadable rather than the last status.
                logger.error("Stopped reading the rendezvous store for the fault-tolerance status")
                self._closed.set()
                self.read_status()


class FaultToleranceMetricsCollector:
    """Expose the last fault-tolerance status as Prometheus metrics, without a store read."""

    STATES = get_args(FaultToleranceState)
    REPLACEMENT_STATES = get_args(ReplacementState)
    CHANGE_KINDS = ("failover", "admission", "abort")

    def __init__(self, health: FaultToleranceHealth, labels: Dict[str, str]):
        self._health = health
        self._labels = labels

    def collect(self):
        """Yield the membership state, the ranks, the replacements, the changes and the stalls."""
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

        status = self._health.last_status()
        values = list(self._labels.values())
        state = self._family(GaugeMetricFamily, "state", "Membership state", "state")
        for name in self.STATES:
            state.add_metric([*values, name], float(status.state == name))
        yield state
        stalled = self._family(
            GaugeMetricFamily,
            "stalled_seconds",
            "How long the active ranks have waited on a rank that makes no progress",
        )
        stalled.add_metric(values, status.stalled_seconds)
        yield stalled
        if status.active_ranks is not None:
            for key, ranks_of_kind in (
                ("active_ranks", status.active_ranks),
                ("failed_ranks", status.failed_ranks),
            ):
                ranks = self._family(GaugeMetricFamily, key, key.replace("_", " ").capitalize())
                ranks.add_metric(values, len(ranks_of_kind))
                yield ranks
            replacements = self._family(
                GaugeMetricFamily, "replacements", "Failed ranks by replacement state", "state"
            )
            for name in self.REPLACEMENT_STATES:
                replacements.add_metric(
                    [*values, name], list(status.replacements.values()).count(name)
                )
            yield replacements
            restarts = self._family(CounterMetricFamily, "rank_restarts", "Rank restarts")
            restarts.add_metric(values, sum(status.incarnations))
            yield restarts
        transaction = status.last_transaction
        changes = self._family(
            CounterMetricFamily, "membership_changes", "Membership changes", "kind"
        )
        for kind in self.CHANGE_KINDS:
            count = 0 if transaction is None else transaction.changes.get(kind, 0)
            changes.add_metric([*values, kind], count)
        yield changes
        if transaction is not None:
            pause = self._family(
                GaugeMetricFamily, "last_change_pause_seconds", "Last change pause"
            )
            pause.add_metric(
                values, (transaction.ended_unix_ms - transaction.detected_unix_ms) / 1e3
            )
            yield pause

    def _family(self, family: type, name: str, documentation: str, *labels: str):
        return family(
            "trtllm_fault_tolerance_" + name, documentation, labels=[*self._labels, *labels]
        )
