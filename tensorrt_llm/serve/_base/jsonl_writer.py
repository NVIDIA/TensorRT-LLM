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
"""Append JSON records to a JSONL file from the event loop without blocking it.

:class:`JsonlRecordWriter` is the mechanism behind the serving apps' record outputs
(today the per-request performance metrics): a bounded queue drained by one asyncio
task that batches records, serializes them and hands the blocking append to a worker
thread. A configuration subclass names the file and decides how a record becomes a
line; the queue, the batching, the drain task, the error containment and the drop
accounting are shared.
"""

import asyncio
import json
import os
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

from tensorrt_llm.logger import logger


class JsonlRecordWriter:
    """Best-effort bounded JSONL writer: one file, one drain task, appends on a worker thread.

    ``submit`` never blocks and never raises: a record that finds the queue full is dropped
    and counted. ``start`` creates the output directory and the drain task; ``close`` writes
    what is still queued and stops the task, cancelling it if that takes longer than
    ``shutdown_timeout``. A record that cannot be serialized or a batch that cannot be
    written is dropped and logged, and the task goes on, because it is the queue's only
    consumer (see ``_run``).

    Subclasses configure the writer through :meth:`_serialize` (how a submitted record
    becomes one newline-terminated line) and ``malformed_record_errors`` (which of its
    exceptions mean "this record cannot be represented"). ``_write`` appends one batch's
    joined lines to the file; a layout other than one append-only file (several files, a
    file per record key) overrides :meth:`_write_batch` instead and takes its lines from
    :meth:`_serialize_batch`, called once per group of records that share a file, so the
    malformed-record accounting stays shared.
    """

    #: Exceptions ``_serialize`` raises for a record that cannot be represented. Such a
    #: record is dropped as malformed; any other exception is unexpected and logged with
    #: its traceback.
    malformed_record_errors: Tuple[Type[Exception], ...] = (TypeError, ValueError)

    def __init__(
        self,
        output_dir: Optional[str],
        *,
        file_stem: str,
        subject: str,
        queue_size: int,
        batch_size: int,
        shutdown_timeout: float,
    ):
        """Configure the writer; nothing is created until ``start``.

        Args:
            output_dir: Directory the file is created in. ``None`` disables the writer:
                ``start``, ``submit`` and ``close`` do nothing.
            file_stem: Start of the file name. The host name, the pid and a UTC timestamp
                follow: ``<file_stem>-<hostname>-<pid>-<YYYYmmddTHHMMSSZ>.jsonl``.
            subject: What the records are, as the log messages name them
                (for example ``"performance metrics"``).
            queue_size: Records held between the event loop and the drain task; a record
                submitted while the queue is full is dropped.
            batch_size: Most records serialized and appended in one write.
            shutdown_timeout: Seconds ``close`` waits for the queue to drain before it
                cancels the drain task.
        """
        self._output_dir = Path(output_dir) if output_dir else None
        self._file_stem = file_stem
        self._subject = subject
        self._batch_size = batch_size
        self._shutdown_timeout = shutdown_timeout
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_size)
        self._task: Optional[asyncio.Task[None]] = None
        self._path: Optional[Path] = None
        self.dropped_records = 0
        self._write_error_count = 0
        self._unexpected_error_counts: dict[str, int] = {}

    async def start(self) -> None:
        """Create the output directory and the drain task.

        Does nothing when the writer is disabled or already started. When the directory
        cannot be created the writer disables itself and logs an error, so a bad output
        path costs the records, not the server.
        """
        if self._output_dir is None or self._task is not None:
            return
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._path = self._output_dir / (
                f"{self._file_stem}-{socket.gethostname()}-{os.getpid()}-{timestamp}.jsonl"
            )
            self._task = asyncio.create_task(self._run())
        except OSError as error:
            logger.error("Disabling %s JSONL output: %s", self._subject, error)
            self._output_dir = None

    def submit(self, record: Any) -> None:
        """Queue ``record`` for writing without blocking.

        Nothing happens before ``start`` or after ``close``. A record submitted while the
        queue is full is dropped and counted in ``dropped_records``.
        """
        if self._task is None:
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped_records += 1
            if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                logger.warning("Dropped %d %s records", self.dropped_records, self._subject)

    async def close(self) -> None:
        """Write the queued records and stop the drain task.

        Waits ``shutdown_timeout`` seconds for the queue to accept the stop marker and
        again for the task to finish; a task still running then is cancelled, and the
        records it had not written are lost.
        """
        if self._task is None:
            return
        task = self._task
        try:
            await asyncio.wait_for(self._queue.put(None), timeout=self._shutdown_timeout)
            await asyncio.wait_for(task, timeout=self._shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out flushing %s JSONL; dropping remaining records", self._subject)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._task = None

    async def _run(self) -> None:
        """Drain queued records into the file until ``close()`` stops the writer.

        This task is the queue's only consumer: if an exception ended it, ``submit()`` would
        keep accepting records that are never written. A record or batch that fails for an
        unexpected reason is therefore dropped and logged, and the loop goes on.
        ``asyncio.CancelledError`` is not an ``Exception``, so ``close()`` can still cancel it.
        """
        stop = False
        while not stop:
            item = await self._queue.get()
            if item is None:
                return
            records = [item]
            while len(records) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop = True
                    break
                records.append(item)
            await self._write_batch(records)

    async def _write_batch(self, records: List[Any]) -> None:
        """Serialize one batch and append it, charging every failure to ``dropped_records``.

        A batch the append fails on is dropped whole and logged, rate-limited; neither that
        nor a record ``_serialize_batch`` dropped ends the drain task.
        """
        lines = self._serialize_batch(records)
        if not lines:
            return
        try:
            await asyncio.to_thread(self._write, "".join(lines))
        except OSError as error:
            self.dropped_records += len(lines)
            self._write_error_count += 1
            if self._write_error_count == 1:
                logger.warning("Failed to write %s JSONL: %s", self._subject, error)
        except Exception as error:
            self._drop_after_unexpected_error(len(lines), "write", error)

    def _serialize_batch(self, records: List[Any]) -> List[str]:
        """Return the lines of ``records``, dropping and counting the records that have none.

        A record ``_serialize`` refuses (``malformed_record_errors``) is dropped alone with a
        rate-limited warning; one it fails on unexpectedly is dropped alone with its traceback.
        The result holds the lines only, so a dropped record cannot be told apart afterwards:
        a layout that writes several files calls this once per group of records that share one.
        """
        lines: List[str] = []
        for record in records:
            try:
                lines.append(self._serialize(record))
            except self.malformed_record_errors as error:
                self.dropped_records += 1
                if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                    logger.warning("Dropped malformed %s record: %s", self._subject, error)
            except Exception as error:
                self._drop_after_unexpected_error(1, "serialize", error)
        return lines

    def _drop_after_unexpected_error(self, num_records: int, stage: str, error: Exception) -> None:
        """Count records lost to an unexpected exception and log it, rate-limited.

        Each stage has its own rate limit so its first failure is always logged.
        The message carries the traceback, which shows where serializing or writing failed.
        """
        self.dropped_records += num_records
        count = self._unexpected_error_counts.get(stage, 0) + 1
        self._unexpected_error_counts[stage] = count
        if count == 1 or count % 1000 == 0:
            logger.error(
                "Dropped %d %s record(s) that failed to %s (%s: %s)\n%s",
                num_records,
                self._subject,
                stage,
                type(error).__name__,
                error,
                "".join(traceback.format_exception(error)).rstrip(),
            )

    def _serialize(self, record: Any) -> str:
        """Return ``record`` as one JSON line, newline included.

        Raises ``TypeError`` or ``ValueError`` -- the default ``malformed_record_errors`` --
        for a record JSON cannot represent; NaN and infinity are refused rather than written
        as tokens no JSON reader accepts.
        """
        return json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"

    def _write(self, data: str) -> None:
        """Append ``data`` to the file. Runs on a worker thread, never on the event loop."""
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as output:
                output.write(data)
