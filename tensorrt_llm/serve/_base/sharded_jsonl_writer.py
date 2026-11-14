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
"""A durable, sharded variant of the shared JSONL record writer.

:class:`ShardedJsonlRecordWriter` keeps the queue, the batching, the drain task and the
error containment of :class:`~tensorrt_llm.serve._base.jsonl_writer.JsonlRecordWriter`
and changes where and how a batch lands:

- records are grouped by the shard a subclass names for them (:meth:`_shard_key`, for
  example one directory per UTC hour and one file per record kind), and each group is
  appended to its own file ``<output_dir>/<shard key><suffix>.jsonl``;
- an append is flushed and ``fsync()``\\ ed before its records count as persisted;
- a file whose last append stopped mid-line gets a newline first, so the fragment is one
  bad line a JSONL reader skips and the next record is never welded onto it;
- an open that fails with ``EEXIST`` -- which some network filesystems report while many
  processes create files in a directory that was just created -- is retried;
- the writer publishes its accounting beside the shards (``<output_dir>/_writer/
  <generation>.json``): every submitted record classified as persisted, dropped by
  reason, unknown (its bytes may have landed, but the write or fsync holding them
  failed) or pending, and the byte range of every shard this writer appended to, which
  :func:`verify_shards` checks against the files;
- files are created with mode 0600 and directories with mode 0700, whatever the umask,
  and a shard directory that disappears under the writer is made again on the next
  append.

Records are serialized and encoded on the worker thread, once each, so a large record
costs the event loop nothing beyond the queue. Lines are UTF-8 with non-ASCII text kept
as is; a record whose text UTF-8 cannot encode (an unpaired surrogate) is written with
``\\u`` escapes instead, which JSON reads back as the same code points.
"""

import asyncio
import contextlib
import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.jsonl_writer import JsonlRecordWriter

__all__ = [
    "DROP_REASONS",
    "SIDECAR_DIR",
    "SIDECAR_SCHEMA",
    "ShardedJsonlRecordWriter",
    "verify_shards",
]

#: Why a record never reached its shard. Every submitted record ends in exactly one of:
#: persisted, one of these, unknown, or (until it does) pending.
DROP_REASONS = ("queue_full", "unserializable", "write_error", "shutdown")
#: Directory under the output directory that holds one accounting document per writer.
SIDECAR_DIR = "_writer"
SIDECAR_SCHEMA = "trtllm-sharded-jsonl-writer/1"
DURABILITY = (
    "a record is persisted once the append holding it, its terminating newline "
    "included, has been written, flushed and fsync()ed without an error"
)
# Seconds to wait before each retry of a shard open that fails with EEXIST.
_SHARD_OPEN_RETRY_DELAYS = (0.05, 0.25, 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ShardedJsonlRecordWriter(JsonlRecordWriter):
    """Append records to one durable file per shard and account for every record.

    Subclasses implement :meth:`_shard_key`, which names the shard of a record relative
    to the output directory (``"<directory>/<file stem>"``); the writer adds a suffix
    that tells writers apart (``-<hostname>-<pid>-<start time>``) and ``.jsonl``.

    The accounting is published at start, whenever a batch fails, every
    ``sidecar_interval`` seconds while the books have moved, and at close. See
    :meth:`accounting` for the document and :func:`verify_shards` for its check.
    """

    # A record without the key its shard is named from is malformed, like one JSON
    # cannot represent.
    malformed_record_errors = (KeyError, TypeError, ValueError)

    def __init__(
        self,
        output_dir: Optional[str],
        *,
        subject: str,
        queue_size: int,
        batch_size: int,
        shutdown_timeout: float,
        shard_suffix: Optional[str] = None,
        sidecar_interval: float = 10.0,
    ):
        """Configure the writer; nothing is created until ``start``.

        Args:
            output_dir: Directory the shards and the sidecar are created in. ``None``
                disables the writer.
            subject: What the records are, as the log messages name them.
            queue_size: Records held between the event loop and the drain task.
            batch_size: Most records serialized and appended in one write.
            shutdown_timeout: Seconds ``close`` waits for the queue to drain, and for a
                sidecar publication to land.
            shard_suffix: End of every shard file name before ``.jsonl``. ``None`` derives
                ``-<hostname>-<pid>-<YYYYmmddTHHMMSSZ>`` at ``start``.
            sidecar_interval: Seconds between publications of the accounting while the
                books move.
        """
        # The base names one file from ``file_stem``; this layout names its files from
        # ``_shard_key`` instead, so the stem is unused.
        super().__init__(
            output_dir,
            file_stem="",
            subject=subject,
            queue_size=queue_size,
            batch_size=batch_size,
            shutdown_timeout=shutdown_timeout,
        )
        self._shard_suffix = shard_suffix
        self._sidecar_interval = sidecar_interval
        #: One writer generation per start. Shard names carry the pid, which a restarted
        #: container hands out again; the generation tells two lifetimes apart.
        self.generation = ""
        self.started_at: Optional[str] = None
        # The disposition of every record handed to ``submit`` while enabled:
        #   persisted -- in its shard, newline-terminated, flushed and fsync()ed;
        #   dropped   -- by reason (DROP_REASONS), never written;
        #   unknown   -- some or all of its bytes may have landed, but the write or
        #                fsync holding it failed, so it is not claimed;
        #   pending   -- the rest: queued or being written right now.
        # ``dropped_records`` (the base class's total) always equals ``sum(dropped)``.
        self.submitted = 0
        self.persisted = 0
        self.unknown = 0
        self.dropped: dict[str, int] = dict.fromkeys(DROP_REASONS, 0)
        #: Records written with ``\u`` escapes because UTF-8 could not encode their text.
        #: An attribute of persisted (or other) records, not a disposition.
        self.sanitized_records = 0
        #: When the last batch persisted in full, in the ISO form the records carry, so a
        #: monitor can tell an idle writer from one whose writes stopped landing.
        self.last_write_at: Optional[str] = None
        self._last_error: Optional[str] = None
        # Records of the batch handed to the write thread and not yet classified.
        self._in_flight = 0
        self._batch_outcomes: list[dict[str, Any]] = []
        # Per shard this generation appended to: where its first append started, where
        # its last durable append ended, and how many records lie between. A shard whose
        # file was replaced or truncated under the writer keeps its old entry, marked
        # superseded, and gets a fresh one.
        self._shards: dict[str, dict[str, Any]] = {}
        self._superseded_shards: list[dict[str, Any]] = []
        self._sidecar_task: Optional[asyncio.Task[None]] = None
        self._publications = 0
        self._sidecar_errors = 0
        self._publishing: Optional[asyncio.Future[None]] = None
        # The books (see _counts_key) of the last sidecar that landed.
        self._published_counts: Optional[tuple[int, int, int, int, int]] = None

    # -- layout hook ----------------------------------------------------------------

    def _shard_key(self, record: Any) -> str:
        """Return the shard of ``record`` relative to the output directory, without suffix.

        For example ``"2026-11-10T12/requests"``. A record this raises for is dropped as
        malformed (``malformed_record_errors``) or, for any other exception, as an
        unexpected failure with its traceback logged.
        """
        raise NotImplementedError

    # -- lifecycle ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Records neither persisted, dropped nor unknown: still on their way."""
        return self.submitted - self.persisted - self.dropped_records - self.unknown

    async def start(self) -> None:
        """Create the output directory, the drain task and the first sidecar."""
        if self._output_dir is None or self._task is not None:
            return
        try:
            self._output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            logger.error("Disabling %s JSONL output: %s", self._subject, error)
            self._output_dir = None
            return
        started = datetime.now(timezone.utc)
        if self._shard_suffix is None:
            self._shard_suffix = (
                f"-{socket.gethostname()}-{os.getpid()}-{started.strftime('%Y%m%dT%H%M%SZ')}"
            )
        self.generation = f"{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        self.started_at = started.isoformat()
        self._task = asyncio.create_task(self._run())
        self._sidecar_task = asyncio.create_task(self._publish_periodically())
        # Published before the first record, so a writer that dies at once still leaves
        # a generation behind for a reconciliation to account for.
        await self._publish(closed=False)

    def submit(self, record: Any) -> None:
        """Queue ``record`` without blocking; a record the full queue refuses is counted."""
        if self._task is None:
            return
        self.submitted += 1
        dropped_before = self.dropped_records
        super().submit(record)
        if self.dropped_records != dropped_before:
            self.dropped["queue_full"] += 1

    async def close(self) -> None:
        """Write the queued records, stop the tasks and publish the final accounting.

        What is still queued after the shutdown timeout was never written and is dropped
        as ``shutdown``; a batch the write thread still holds may yet land, so its records
        are ``unknown`` rather than claimed either way.
        """
        if self._task is None:
            return
        if self._sidecar_task is not None:
            self._sidecar_task.cancel()
            await asyncio.gather(self._sidecar_task, return_exceptions=True)
            self._sidecar_task = None
        await super().close()
        leftover = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None:
                leftover += 1
        self.dropped["shutdown"] += leftover
        self.dropped_records += leftover
        self.unknown += self._in_flight
        self._in_flight = 0
        await self._publish(closed=True)

    # -- serialization --------------------------------------------------------------

    def _encode_record(self, record: Any) -> tuple[bytes, int]:
        """One UTF-8 JSON line as bytes, and 1 when it had to be ``\\u``-escaped.

        ``json.dumps`` checks JSON shape, not encodability: with ``ensure_ascii=False`` a
        lone surrogate in client text rides through it into a str no UTF-8 file can take,
        and the failure would surface in the file write and cost the whole group. Escaped,
        the record is pure ASCII and reads back as the code points the client sent.

        Runs on the worker thread, once per record. A subclass whose records are completed
        late -- from parts the event loop only queued -- overrides this and calls it.
        """
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        try:
            return (line + "\n").encode("utf-8"), 0
        except UnicodeEncodeError:
            line = json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            return (line + "\n").encode("ascii"), 1

    def _serialize(self, record: Any) -> str:
        """The base class's line contract; this layout encodes through ``_encode_record``."""
        return self._encode_record(record)[0].decode("utf-8")

    # -- writing --------------------------------------------------------------------

    async def _write_batch(self, records: list[Any]) -> None:
        """Group one batch by shard, hand it to the worker thread and book the outcomes.

        The event loop only names each record's shard; serializing, encoding and appending
        happen on the worker thread (:meth:`_append_groups`). Everything a batch can fail
        on is contained here, because the drain task has no other protection: a record
        whose shard cannot be named or whose line cannot be serialized is dropped alone,
        and a failed append is classified per group. ``asyncio.CancelledError`` still
        passes, so ``close`` can cancel a task stuck in a write; the records that batch
        held stay unknown.
        """
        groups: dict[str, list[Any]] = {}
        for record in records:
            try:
                key = self._shard_key(record)
            except self.malformed_record_errors as error:
                self._drop_malformed(error)
                continue
            except Exception as error:
                self.dropped["unserializable"] += 1
                self._drop_after_unexpected_error(1, "place", error)
                continue
            groups.setdefault(key, []).append(record)
        if not groups:
            return
        total = sum(len(group) for group in groups.values())
        self._in_flight = total
        # Filled in by _append_groups group by group, so a surprise exception out of it
        # still leaves what was already classified.
        self._batch_outcomes = []
        try:
            await asyncio.to_thread(self._append_groups, groups)
        except Exception as error:
            # _append_groups classifies its own failures, so only a surprise lands here.
            # A group whose bytes were attempted may have landed: unknown. Every other
            # unclassified record never reached a file: dropped.
            outcomes = self._batch_outcomes
            unclaimed = sum(
                o["records"] - o["persisted"] - o["dropped"] - o["unknown"] - o["unserializable"]
                for o in outcomes
                if o["attempted"]
            )
            classified = sum(
                o["persisted"] + o["dropped"] + o["unknown"] + o["unserializable"] for o in outcomes
            )
            self.unknown += unclaimed
            self._drop("write_error", total - classified - unclaimed)
            self._note_write_error(f"{type(error).__name__}: {error}")
        failed = False
        outcomes = self._batch_outcomes
        for outcome in outcomes:
            self._apply_outcome(outcome)
            if outcome["error"] is not None:
                failed = True
                self._note_write_error(outcome["error"])
        self._in_flight = 0
        if (
            not failed
            and len(outcomes) == len(groups)
            and any(outcome["persisted"] for outcome in outcomes)
        ):
            self.last_write_at = _utc_now()
        if failed:
            await self._publish(closed=False)

    def _drop(self, reason: str, count: int) -> None:
        self.dropped[reason] += count
        self.dropped_records += count

    def _drop_malformed(self, error: Exception) -> None:
        self._drop("unserializable", 1)
        if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
            logger.warning("Dropped malformed %s record: %s", self._subject, error)

    def _note_write_error(self, error: str) -> None:
        self._write_error_count += 1
        self._last_error = error
        if self._write_error_count == 1 or self._write_error_count % 1000 == 0:
            logger.warning("Failed to write %s JSONL: %s", self._subject, error)

    def _apply_outcome(self, outcome: dict[str, Any]) -> None:
        """Book one group's outcome: its counts, its escapes and its shard's watermarks."""
        if outcome["sanitized"]:
            before = self.sanitized_records
            self.sanitized_records += outcome["sanitized"]
            if before == 0 or before // 1000 != self.sanitized_records // 1000:
                logger.warning(
                    "Escaped %d %s record(s) whose text UTF-8 cannot encode",
                    self.sanitized_records,
                    self._subject,
                )
        for error in outcome["malformed_errors"]:
            self._drop_malformed(error)
        for error in outcome["unexpected_errors"]:
            self.dropped["unserializable"] += 1
            self._drop_after_unexpected_error(1, "serialize", error)
        self.persisted += outcome["persisted"]
        self._drop("write_error", outcome["dropped"])
        self.unknown += outcome["unknown"]
        if outcome["size_before"] is None:
            return  # nothing reached the file
        shard = self._shards.get(outcome["file"])
        if shard is not None and outcome["size_before"] < shard["persisted_end"]:
            # The file is shorter than the last durable append left it: it was replaced or
            # truncated under the writer. The old entry describes bytes that are gone; it
            # stays, marked, and the new file gets a fresh range.
            shard["superseded"] = True
            self._superseded_shards.append(shard)
            shard = None
        if shard is None:
            shard = self._shards[outcome["file"]] = {
                "file": outcome["file"],
                # Where this generation's first append to the shard began.
                "start_offset": outcome["size_before"],
                "persisted_end": outcome["size_before"],
                "persisted_records": 0,
                "fragments_isolated": 0,
                # Failed appends: where their bytes start, how many landed, and how
                # many records those bytes touch.
                "unknown_extents": [],
            }
        if outcome["isolated"]:
            shard["fragments_isolated"] += 1
        if outcome["end"] is not None:
            shard["persisted_end"] = outcome["end"]
            shard["persisted_records"] += outcome["persisted"]
        if outcome["extent"] is not None:
            shard["unknown_extents"].append(outcome["extent"])

    def _append_groups(self, groups: dict[str, list[Any]]) -> None:
        """Encode each group and append it to its shard, durably. Runs on a worker thread.

        One group is one file and one append: the records are encoded one by one (one
        that cannot be is set aside for the event loop to drop), their lines written in
        one call, flushed and ``fsync()``\\ ed, and only then counted as persisted, with
        the shard's watermark moved to the end of the append. The shard's directory is
        created on every append, so one an operator removed is made again. Files are
        created with mode 0600 and directories with mode 0700.

        A failure anywhere in the append is classified rather than raised: nothing
        attempted (the directory or file would not open) drops every line as
        ``write_error``; bytes that may have landed (the write, flush or fsync failed)
        leave every line whose bytes reached the file, wholly or partly, ``unknown`` and
        drop the rest.

        Each outcome is appended to ``_batch_outcomes`` as it completes, and nothing here
        touches a counter: the event loop books the outcome (:meth:`_apply_outcome`), so a
        batch abandoned by a cancelled drain never moves a watermark and the counts and
        the shards always agree.
        """
        assert self._output_dir is not None and self._shard_suffix is not None
        for key, group in groups.items():
            outcome: dict[str, Any] = {
                "file": f"{key}{self._shard_suffix}.jsonl",
                "records": len(group),
                "persisted": 0,
                "dropped": 0,
                "unknown": 0,
                "unserializable": 0,
                "sanitized": 0,
                "malformed_errors": [],
                "unexpected_errors": [],
                "attempted": False,
                "error": None,
                # Offsets in the shard: its size before this append, whether a fragment
                # was isolated first, where the durable append ended, and what a failed
                # one left behind.
                "size_before": None,
                "isolated": False,
                "end": None,
                "extent": None,
            }
            self._batch_outcomes.append(outcome)
            parts: list[bytes] = []
            for record in group:
                try:
                    data, sanitized = self._encode_record(record)
                except self.malformed_record_errors as error:
                    outcome["malformed_errors"].append(error)
                    outcome["unserializable"] += 1
                    continue
                except Exception as error:
                    outcome["unexpected_errors"].append(error)
                    outcome["unserializable"] += 1
                    continue
                parts.append(data)
                outcome["sanitized"] += sanitized
            if not parts:
                continue
            payload = b"".join(parts)
            lines = len(parts)
            path: Optional[Path] = None
            payload_start: Optional[int] = None
            try:
                path = self._output_dir / outcome["file"]
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                needs_newline = False
                try:
                    size = path.stat().st_size
                    if size > 0:
                        with path.open("rb") as tail:
                            tail.seek(-1, os.SEEK_END)
                            needs_newline = tail.read(1) != b"\n"
                except FileNotFoundError:
                    size = 0
                outcome["size_before"] = size
                with self._open_shard_for_append(path) as output:
                    outcome["attempted"] = True
                    if needs_newline:
                        output.write(b"\n")
                        outcome["isolated"] = True
                    output.flush()
                    payload_start = output.tell()
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                    outcome["end"] = output.tell()
                outcome["persisted"] = lines
            except Exception as error:
                outcome["error"] = f"{type(error).__name__}: {error}"
                landed = self._landed_bytes(path, payload_start, len(payload))
                if not outcome["attempted"] or payload_start is None:
                    outcome["dropped"] = lines
                elif landed is None:
                    outcome["unknown"] = lines
                else:
                    touched = payload[:landed].count(b"\n")
                    if landed and payload[landed - 1 : landed] != b"\n":
                        touched += 1  # the line the write stopped inside
                    outcome["unknown"] = touched
                    outcome["dropped"] = lines - touched
                    if landed:
                        outcome["extent"] = {
                            "start": payload_start,
                            "landed": landed,
                            "records": touched,
                        }

    @staticmethod
    def _open_shard_for_append(path: Path) -> IO[bytes]:
        """Open a shard for appending, created with mode 0600, retrying an ``EEXIST``.

        An open with ``O_CREAT`` and without ``O_EXCL`` cannot legitimately fail with
        ``EEXIST``, but a network filesystem client can still report it while several
        processes create files in a directory that was just created. Nothing has been
        written when the open fails, so it is simply tried again.
        """
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        for delay in _SHARD_OPEN_RETRY_DELAYS:
            try:
                return os.fdopen(os.open(path, flags, 0o600), "ab")
            except FileExistsError:
                time.sleep(delay)
        return os.fdopen(os.open(path, flags, 0o600), "ab")

    @staticmethod
    def _landed_bytes(path: Optional[Path], start: Optional[int], length: int) -> Optional[int]:
        """How much of a failed append reached the file, or None if unknowable."""
        if path is None or start is None:
            return 0
        try:
            return max(0, min(length, path.stat().st_size - start))
        except OSError:
            return None

    # -- accounting -----------------------------------------------------------------

    def accounting(self, closed: bool = False) -> dict[str, Any]:
        """This writer's sidecar document: what it was given and where it went.

        ``submitted == persisted + sum(dropped) + unknown + pending`` holds at every
        publication. ``shards`` gives, per file this generation appended to, the byte
        range its durable appends occupy -- from ``start_offset`` to ``persisted_end`` --
        and how many records that range holds; :func:`verify_shards` reads exactly that
        range. A file another lifetime of the same pid also appended to is told apart by
        these offsets.
        """
        return {
            "schema": SIDECAR_SCHEMA,
            "generation": self.generation,
            "pid": os.getpid(),
            "subject": self._subject,
            "shard_suffix": self._shard_suffix,
            "started_at": self.started_at,
            "published_at": _utc_now(),
            "sequence": self._publications,
            "closed": closed,
            "durability": DURABILITY,
            "counts": {
                "submitted": self.submitted,
                "persisted": self.persisted,
                "dropped": dict(self.dropped),
                "unknown": self.unknown,
                "pending": self.pending,
            },
            "attributes": {"sanitized": self.sanitized_records},
            "shards": [dict(shard) for shard in (*self._superseded_shards, *self._shards.values())],
            "errors": {
                "write_errors": self._write_error_count,
                "last_error": self._last_error,
                "sidecar_errors": self._sidecar_errors,
            },
        }

    def _counts_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.submitted,
            self.persisted,
            self.dropped_records,
            self.unknown,
            self._sidecar_errors,
        )

    def _unpublished(self) -> bool:
        """True if the books moved since a sidecar last landed."""
        return self._counts_key() != self._published_counts

    async def _publish_periodically(self) -> None:
        """Catch the sidecar up with the books while the writer is open.

        Covers the records of a final burst before a lull, which no later batch would
        publish, and retries a publication the filesystem refused.
        """
        while True:
            await asyncio.sleep(self._sidecar_interval)
            if self._unpublished():
                await self._publish(closed=False)

    async def _publish(self, *, closed: bool) -> None:
        """Atomically replace this generation's sidecar.

        Never raises: a sidecar that cannot be written costs the reconciliation its
        freshest numbers until a later one lands. Only a sidecar that landed marks its
        books published, so a failure is retried by the periodic task.
        """
        if self._output_dir is None:
            return
        if self._publishing is not None and not self._publishing.done():
            # The previous publication is still stuck in the filesystem; a thread queued
            # behind it would only pile up. Counted, and retried at the next opportunity.
            self._sidecar_errors += 1
            return
        document = self.accounting(closed=closed)
        books = self._counts_key()
        self._publications += 1
        self._publishing = asyncio.ensure_future(asyncio.to_thread(self._write_sidecar, document))
        # Retrieved here or, after a timeout, whenever the thread finishes.
        self._publishing.add_done_callback(lambda future: future.cancelled() or future.exception())
        try:
            await asyncio.wait_for(asyncio.shield(self._publishing), timeout=self._shutdown_timeout)
        except Exception as error:
            self._sidecar_errors += 1
            if self._sidecar_errors == 1 or self._sidecar_errors % 100 == 0:
                logger.warning("Failed to publish %s writer accounting: %s", self._subject, error)
        else:
            self._published_counts = books

    def _write_sidecar(self, document: dict[str, Any]) -> None:
        assert self._output_dir is not None
        directory = self._output_dir / SIDECAR_DIR
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        final = directory / f"{self.generation}.json"
        partial = directory / f".{self.generation}.json.partial"
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, final)
        with contextlib.suppress(OSError):
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def verify_shards(output_dir: str, sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    """Check a writer sidecar against the shards it describes.

    For each shard, reads the byte range the sidecar claims, from ``start_offset`` to
    ``persisted_end``, and counts the newline-terminated lines that parse as JSON --
    leaving out the bytes of failed appends (``unknown_extents``), which are counted on
    their own. A consistent writer has ``records == persisted_records`` and
    ``bad_lines == 0``; whatever its failed appends left behind shows up under
    ``unknown_lines``, never as persisted.
    """
    results = []
    for shard in sidecar.get("shards", []):
        path = Path(output_dir) / shard["file"]
        result = dict(shard, records=0, bad_lines=0, unknown_lines=0, size=None, short=False)
        begin, end = shard["start_offset"], shard["persisted_end"]
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                result["size"] = handle.tell()
                handle.seek(begin)
                data = bytearray(handle.read(end - begin))
        except OSError as error:
            result["error"] = f"{type(error).__name__}: {error}"
            results.append(result)
            continue
        result["short"] = len(data) < end - begin
        for extent in shard.get("unknown_extents", []):
            lo = extent["start"] - begin
            hi = min(len(data), lo + extent["landed"])
            if lo < 0 or lo >= len(data):
                continue
            result["unknown_lines"] += bytes(data[lo:hi]).count(b"\n") + (
                1 if hi > lo and data[hi - 1 : hi] != b"\n" else 0
            )
            data[lo:hi] = b"\n" * (hi - lo)  # neither persisted nor bad
        for line in bytes(data).split(b"\n"):
            if not line:
                continue
            try:
                json.loads(line)
                result["records"] += 1
            except ValueError:
                result["bad_lines"] += 1
        results.append(result)
    return results
