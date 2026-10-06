# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort, initialization-time observations of worker-owned settings.

Workers send one small authenticated datagram, never a usage event. Loss is
represented as incomplete coverage; no collective, acknowledgement or worker
wait is necessary. Only the explicitly supported scalar fields are observed.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import os
import secrets
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

PARTIAL_REUSE_FIELD = "kv_cache_config.enable_partial_reuse"
FIELDS = (
    "kv_cache_config.host_cache_size",
    "kv_cache_config.disk_cache_size",
    PARTIAL_REUSE_FIELD,
)
_MAX_WORKERS = 4096
_MAX_PACKET = 2048
_PROTOCOL = 1
_COLLECTION_SECONDS = 1.0
Endpoint = tuple[str, int, bytes]


def _enabled(args: Any) -> bool:
    from .usage_lib import is_usage_stats_enabled

    return is_usage_stats_enabled(args.telemetry_config.disabled)


def _valid_values(values: object) -> bool:
    if not isinstance(values, dict) or not values.keys() <= set(FIELDS):
        return False
    for path, value in values.items():
        if path == PARTIAL_REUSE_FIELD:
            if type(value) is not bool:
                return False
        elif value is not None and (type(value) is not int or not 0 <= value < 2**64):
            return False
    return True


def send_snapshot(endpoint: Endpoint, rank: int, snapshot: dict) -> None:
    """Send once, without waiting for a receiver or exposing raw config objects."""
    try:
        host, port, key = endpoint
        body = json.dumps(
            {"protocol": _PROTOCOL, "rank": rank, "snapshot": snapshot},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        if len(body) + 32 > _MAX_PACKET:
            return
        packet = hmac.digest(key, body, "sha256") + body
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.setblocking(False)
            sender.sendto(packet, (host, port))
    except Exception:
        # This boundary must never turn optional telemetry into a startup error.
        return


def publish_worker_config(args: Any, rank: int) -> None:
    """Observe the final worker arguments without starting a telemetry session."""
    try:
        endpoint = args._worker_config_endpoint
        if endpoint is None or not _enabled(args):
            return
        from .llmapi_config import collect_llm_api_config_payloads

        config, meta = map(json.loads, collect_llm_api_config_payloads(args))
        values = {path: config[path] for path in FIELDS if path in config}
        if not meta["capture_succeeded"] or not _valid_values(values):
            return
        # Collection can outlast an opt-out change.
        if not _enabled(args):
            return
        snapshot = {
            "manifest": meta["capture_manifest_digest"],
            "policy": meta["field_policy_version"],
            "values": values,
        }

        def send() -> None:
            # Hostname resolution and socket I/O must not delay engine startup.
            try:
                host, port, key = endpoint
                address = socket.gethostbyname(host)
                if _enabled(args):
                    send_snapshot((address, port, key), rank, snapshot)
            except Exception:
                pass  # Optional telemetry must not raise from a worker daemon.

        threading.Thread(target=send, daemon=True, name="trtllm-worker-snapshot").start()
    except Exception:
        return


class WorkerConfigCollector:
    """Bounded receiver owned by one LLM construction, not a process session."""

    def __init__(self, expected: int, host: str | None = None) -> None:
        if type(expected) is not int or not 1 <= expected <= _MAX_WORKERS:
            raise ValueError("Unsupported model-worker count")
        self.expected = expected
        self._key = secrets.token_bytes(32)
        self._snapshots: dict[int, dict] = {}
        self._invalid_ranks: set[int] = set()
        self._stop = threading.Event()
        self._complete = threading.Event()
        self._finished = threading.Event()
        self._deadline: float | None = None
        # Binding a local socket needs neither DNS nor a receiver-readiness wait.
        # Workers resolve the advertised hostname in their background senders.
        self._receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._receiver.bind((host or "0.0.0.0", 0))
            self._receiver.settimeout(0.02)
            self.endpoint = (
                host or socket.gethostname(),
                self._receiver.getsockname()[1],
                self._key,
            )
        except Exception:
            self._receiver.close()
            raise
        self._thread = threading.Thread(
            target=self._receive, daemon=True, name="trtllm-worker-config"
        )
        try:
            self._thread.start()
        except Exception:
            self._receiver.close()
            raise

    def _receive(self) -> None:
        try:
            with self._receiver as receiver:
                while not self._stop.is_set():
                    if self._deadline is not None and (
                        self._complete.is_set() or time.monotonic() >= self._deadline
                    ):
                        break
                    try:
                        packet = receiver.recv(_MAX_PACKET + 1)
                    except socket.timeout:
                        continue
                    self._accept(packet)
        except OSError:
            pass
        finally:
            self._finished.set()

    def _accept(self, packet: bytes) -> None:
        if not 32 < len(packet) <= _MAX_PACKET:
            return
        signature, body = packet[:32], packet[32:]
        if not hmac.compare_digest(signature, hmac.digest(self._key, body, "sha256")):
            return
        try:
            message = json.loads(body)
            if (
                not isinstance(message, dict)
                or type(message.get("protocol")) is not int
                or message["protocol"] != _PROTOCOL
            ):
                return
            rank = message.get("rank")
            snapshot = message.get("snapshot")
            if type(rank) is not int or not 0 <= rank < self.expected:
                return
            if not isinstance(snapshot, dict) or set(snapshot) != {"manifest", "policy", "values"}:
                return
            if (
                not isinstance(snapshot["manifest"], str)
                or len(snapshot["manifest"]) != 64
                or not isinstance(snapshot["policy"], str)
                or len(snapshot["policy"]) > 16
                or not _valid_values(snapshot["values"])
            ):
                return
            if rank in self._invalid_ranks:
                return
            previous = self._snapshots.get(rank)
            if previous is not None and previous != snapshot:
                self._invalid_ranks.add(rank)
                self._snapshots.pop(rank)
                self._complete.clear()
                return
            self._snapshots[rank] = snapshot
            if len(self._snapshots) == self.expected:
                self._complete.set()
        except (ValueError, TypeError, KeyError):
            return

    def start_deadline(self) -> None:
        """Arm background cleanup after startup, even if no reporter takes ownership."""
        self._deadline = time.monotonic() + _COLLECTION_SECONDS

    def cancel(self) -> None:
        """Cancel without waiting on the application's startup/shutdown thread."""
        self._stop.set()

    def finish(self, wait: bool = True) -> dict:
        """Read the final observations; only the telemetry thread may wait here."""
        if self._deadline is None:
            self.start_deadline()
        if wait:
            self._finished.wait(max(0, self._deadline - time.monotonic()) + 0.05)
        self.cancel()
        # Never read a dictionary still being modified by a delayed receiver.
        snapshots = list(self._snapshots.values()) if self._finished.is_set() else []
        return {"expected": self.expected, "snapshots": snapshots}


@contextlib.contextmanager
def observe_workers(args: Any) -> Iterator[WorkerConfigCollector | None]:
    """Add optional collection around engine creation; always clean up on failure."""
    collector = None
    expected = None
    try:
        args._worker_config_endpoint = None
        args._worker_config_observation = None
        if _enabled(args) and not args.encode_only:
            expected = args.parallel_config.world_size
            # Attached/deferred frontends do not initialize workers here.
            placement = getattr(args, "ray_placement_config", None)
            if not os.getenv("TLLM_EXECUTOR_ATTACH_INFO") and not getattr(
                placement, "defer_workers_init", False
            ):
                collector = WorkerConfigCollector(expected)
                args._worker_config_endpoint = collector.endpoint
    except Exception:
        pass
    try:
        yield collector
    except BaseException:
        if collector is not None:
            collector.cancel()
        raise
    finally:
        if expected is not None:
            if collector is not None:
                collector.start_deadline()
            args._worker_config_observation = {
                "expected": expected,
                "snapshots": [],
                "pending": collector is not None,
                "capture_id": uuid.uuid4().hex,
            }
            args._worker_config_endpoint = None


def merge_observation(config: dict, meta: dict, observation: dict) -> None:
    """Replace only supported paths, never use stale parent values as fallback."""
    expected = observation["expected"]
    compatible_snapshots = [
        snapshot
        for snapshot in observation["snapshots"]
        if snapshot["manifest"] == meta["capture_manifest_digest"]
        and snapshot["policy"] == meta["field_policy_version"]
    ]
    # Complete coverage does not imply agreement or availability for every field.
    all_workers_responded = len(compatible_snapshots) == expected
    conflicting, unavailable, verified = [], [], []
    for path in FIELDS:
        config.pop(path, None)
        if not all_workers_responded or any(
            path not in snapshot["values"] for snapshot in compatible_snapshots
        ):
            unavailable.append(path)
            continue
        values = [snapshot["values"][path] for snapshot in compatible_snapshots]
        if len({json.dumps(value, allow_nan=False) for value in values}) != 1:
            conflicting.append(path)
        else:
            config[path] = values[0]
            verified.append(path)
    if observation.get("pending"):
        status = "pending"
    elif all_workers_responded:
        status = "complete"
    elif compatible_snapshots:
        status = "partial"
    else:
        status = "unavailable"
    meta.update(
        {
            "capture_version": "3",
            "source": "parent_with_worker_observations",
            "worker_capture": {
                "status": status,
                "expected": expected,
                "received": len(compatible_snapshots),
                "verified_fields": verified,
                "conflicting_fields": conflicting,
                "unavailable_fields": unavailable,
            },
        }
    )
    if "capture_id" in observation:
        meta["capture_id"] = observation["capture_id"]
