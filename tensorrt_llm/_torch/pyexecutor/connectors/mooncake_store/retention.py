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
"""Opt-in cross-owner checkpoint retention on a job-scoped shared filesystem.

Requires coherent cross-host flock and atomic rename (validate before use).
Locks are never unlinked: replacing a lock inode would break mutual exclusion.
Pins survive process death because process death alone does not prove RDMA
retirement. A failed metadata operation may leak a reference; it must not cause
an unreferenced object to be reclaimed while another owner is using it.
"""

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol


class RetentionStore(Protocol):
    """Store deletion surface; results must prove non-forced removal completed."""

    def remove(self, key: str, force: bool) -> int: ...


Claim = tuple[BinaryIO, Path, dict[str, Any]]


class RetentionError(RuntimeError):
    """Unsafe/inconsistent retention metadata: stop rather than guess ownership."""


class EndpointUnavailable(RetentionError):
    """A busy or quarantined endpoint cannot admit new transfers."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _read(path: Path, default: Any) -> Any:
    try:
        with path.open() as source:
            return json.load(source)
    except FileNotFoundError:
        return default


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as output:
            json.dump(value, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        # Required for crash-safe insertion-before-removal ordering.
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path: Path, *, exclusive: bool = True) -> Iterator[None]:
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass
class TransferLease:
    """Worker-only lease: descriptors/paths, never CUDA/native cache ownership."""

    endpoint: Path
    pin: Path
    handle: BinaryIO

    def close(self, *, retired: bool) -> None:
        try:
            if retired:
                self.pin.unlink(missing_ok=True)
        finally:
            # If retirement is uncertain, keep the persistent pin after unlock.
            self.handle.close()


class CheckpointRetention:
    """One instance per transfer worker, shared catalog across every CTX owner."""

    def __init__(self, directory: str, namespace: str, max_turns: int = 5) -> None:
        if max_turns <= 0 or not namespace or not Path(directory).is_absolute():
            raise ValueError("retention requires an absolute shared path and positive turn limit")
        self.namespace = namespace
        self.max_turns = max_turns
        self.root = Path(directory) / _digest(namespace)
        for child in ("endpoints", "conversations", "pending"):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        self._pending_cursor = None
        self._uncertain_claims = []
        with _lock(self.root / "catalog.lock"):
            policy = {"schema": 1, "namespace": namespace, "max_turns": max_turns}
            path = self.root / "policy.json"
            previous = _read(path, None)
            if previous is not None and previous != policy:
                raise RetentionError("all CTX workers must use the same retention policy")
            if previous is None:
                _atomic(path, policy)

    def _endpoint(self, marker: str) -> Path:
        prefix = self.namespace + "/complete/"
        if not marker.startswith(prefix):
            raise RetentionError("marker does not belong to this representation namespace")
        suffix = marker.removeprefix(prefix)
        if len(suffix) != 64 or any(c not in "0123456789abcdef" for c in suffix):
            raise RetentionError("marker does not belong to this representation namespace")
        path = self.root / "endpoints" / _digest(marker)
        for child in ("refs", "pins"):
            (path / child).mkdir(parents=True, exist_ok=True)
        return path

    def acquire(self, marker: str) -> TransferLease:
        endpoint = self._endpoint(marker)
        handle = (endpoint / "transfer.lock").open("a+b")
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise EndpointUnavailable("endpoint has an exclusive deletion claim") from error
            if (endpoint / "uncertain-delete.json").exists():
                raise EndpointUnavailable("endpoint quarantined after uncertain Store deletion")
            pin = endpoint / "pins" / uuid.uuid4().hex
            # Persist before a single GET/PUT can begin.
            _atomic(pin, {"pid": os.getpid(), "created_ns": time.time_ns()})
            return TransferLease(endpoint, pin, handle)
        except BaseException:
            handle.close()
            raise

    def prepare_save(
        self, lease: TransferLease, marker: str, private_keys: tuple[str, ...]
    ) -> None:
        """Record an immutable reclaim manifest before publishing anything."""
        if self._endpoint(marker) != lease.endpoint:
            raise RetentionError("marker does not match the held transfer lease")
        if not private_keys or len(set(private_keys)) != len(private_keys):
            raise RetentionError("checkpoint requires unique endpoint-private payload keys")
        if any(not key.startswith(self.namespace + "/group/") for key in private_keys):
            raise RetentionError("private payload outside the representation namespace")
        endpoint_hash = marker.rsplit("/", 1)[1]
        if any(key.rsplit("/", 1)[-1] != endpoint_hash for key in private_keys):
            raise RetentionError("private payload is not keyed to this checkpoint endpoint")
        manifest = {"marker": marker, "private_keys": sorted(private_keys)}
        with _lock(self.root / "catalog.lock"):
            path = lease.endpoint / "manifest.json"
            existing = _read(path, None)
            if existing is not None and existing != manifest:
                raise RetentionError("same endpoint has inconsistent reclaim manifests")
            if existing is None:
                _atomic(path, manifest)
            # A cancelled/failed save without a published owner can be collected
            # once all transfers retire. Failed transfers keep their pins.
            _atomic(self.root / "pending" / lease.endpoint.name, {})

    def publish(self, lease: TransferLease, conversation: str, turn: str) -> int:
        """Called after successful marker PUT; return newly retired turn count."""
        if not turn:
            raise RetentionError("missing immutable request-turn identity")
        endpoint_id = lease.endpoint.name
        with _lock(self.root / "catalog.lock"):
            if not conversation:
                _atomic(lease.endpoint / "refs" / "unscoped", {})
                return 0
            conversation_id = _digest(conversation)
            path = self.root / "conversations" / (conversation_id + ".json")
            turns = _read(path, [])
            for row in turns:
                if row["turn"] == turn:
                    row["endpoints"] = sorted(set(row["endpoints"]) | {endpoint_id})
                    break
            else:
                turns.append({"turn": turn, "endpoints": [endpoint_id]})
            retained = turns[-self.max_turns :]
            expired = turns[: -self.max_turns]
            retained_ids = {key for row in retained for key in row["endpoints"]}
            dropped_ids = {key for row in expired for key in row["endpoints"]} - retained_ids
            # Insert new reference first, then commit ledger, then release old
            # references. A crash between steps can only retain extra objects.
            _atomic(lease.endpoint / "refs" / conversation_id, {})
            _atomic(path, retained)
            for old_id in dropped_ids:
                _atomic(self.root / "pending" / old_id, {})
                (self.root / "endpoints" / old_id / "refs" / conversation_id).unlink(
                    missing_ok=True
                )
            return len(expired)

    def claim(self, limit: int = 8) -> tuple[list[Claim], dict[str, int]]:
        """Reserve deletions; caller must hold returned handles through Store RPCs."""
        stats = {
            "gc_examined": 0,
            "gc_deleted_checkpoints": 0,
            "gc_deleted_objects": 0,
            "gc_deferred_active": 0,
            "gc_deferred_lease": 0,
            "gc_errors": 0,
        }
        claims = []
        if self._pending_cursor is None:
            self._pending_cursor = os.scandir(self.root / "pending")
        try:
            for _ in range(limit):
                try:
                    pending = next(self._pending_cursor)
                except StopIteration:
                    self._pending_cursor.close()
                    self._pending_cursor = None
                    break
                if len(pending.name) != 64 or not pending.is_file():
                    continue
                stats["gc_examined"] += 1
                endpoint = self.root / "endpoints" / pending.name
                handle = (endpoint / "transfer.lock").open("a+b")
                keep = False
                try:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        stats["gc_deferred_active"] += 1
                        continue
                    with _lock(self.root / "catalog.lock"):
                        if (endpoint / "uncertain-delete.json").exists():
                            stats["gc_errors"] += 1
                            continue
                        if any((endpoint / "refs").iterdir()):
                            Path(pending.path).unlink(missing_ok=True)
                            continue
                        if any((endpoint / "pins").iterdir()):
                            stats["gc_deferred_active"] += 1
                            continue
                        manifest = _read(endpoint / "manifest.json", None)
                        if manifest is None:
                            raise RetentionError("retirement candidate has no manifest")
                    claims.append((handle, Path(pending.path), manifest))
                    keep = True
                finally:
                    if not keep:
                        handle.close()
            return claims, stats
        except BaseException:
            for claim in claims:
                self.finish_claim(claim, success=False)
            raise

    def finish_claim(self, claim: Claim, *, success: bool, uncertain: bool = False) -> None:
        handle, pending, _ = claim
        if uncertain:
            # Persist a fence before releasing EX: an ambiguous deletion may
            # still complete after an RPC error. Never allow a new GET/PUT.
            self._uncertain_claims.append(claim)
            endpoint = self.root / "endpoints" / pending.name
            _atomic(endpoint / "uncertain-delete.json", {})
            self._uncertain_claims.remove(claim)
        try:
            if success:
                pending.unlink(missing_ok=True)
        finally:
            handle.close()

    def collect(self, store: RetentionStore, limit: int = 8) -> dict[str, int]:
        """Bounded worker-side collection; never waits for an active transfer."""
        claims, stats = self.claim(limit)
        for claim in claims:
            success = False
            deferred = False
            try:
                manifest = claim[2]
                for key in [manifest["marker"], *manifest["private_keys"]]:
                    result = store.remove(key, False)
                    if type(result) is int and result == -706:
                        # A live Store read lease explicitly rejects deletion.
                        # Nothing is in flight: retry later, without quarantine.
                        deferred = True
                        stats["gc_deferred_lease"] += 1
                        break
                    if type(result) is not int or result not in (0, -704):
                        raise RetentionError("non-forced Store deletion did not complete")
                    stats["gc_deleted_objects"] += int(result == 0)
                success = not deferred
                stats["gc_deleted_checkpoints"] += int(success)
            except (RuntimeError, OSError):
                stats["gc_errors"] += 1
            finally:
                self.finish_claim(claim, success=success, uncertain=not success and not deferred)
        return stats

    def close(self) -> None:
        """Close only the local directory iterator; never clear uncertain pins."""
        if self._pending_cursor is not None:
            self._pending_cursor.close()
            self._pending_cursor = None
