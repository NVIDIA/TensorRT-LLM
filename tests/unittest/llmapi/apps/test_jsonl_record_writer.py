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
"""The shared JSONL record writer: queue, batching, drain task, error containment, drops."""

import asyncio
import json
import logging
import os
import re
import socket
import threading

import pytest

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.jsonl_writer import JsonlRecordWriter

pytestmark = pytest.mark.cpu_only


def make_writer(output_dir, *, queue_size=16, batch_size=4, shutdown_timeout=5):
    return JsonlRecordWriter(
        None if output_dir is None else str(output_dir),
        file_stem="records-test",
        subject="test",
        queue_size=queue_size,
        batch_size=batch_size,
        shutdown_timeout=shutdown_timeout,
    )


def written_lines(output_dir):
    (output_file,) = output_dir.glob("records-test-*.jsonl")
    return output_file.read_text(encoding="utf-8").splitlines(keepends=True)


def written_values(output_dir):
    return [json.loads(line)["n"] for line in written_lines(output_dir)]


async def wait_until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


@pytest.fixture()
def logged():
    """Capture the warnings and errors the TRT-LLM logger emits, as (level, message) pairs."""
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append((record.levelname, record.getMessage()))

    handler = Capture(level=logging.WARNING)
    stdlib_logger = logger._logger
    previous_level = stdlib_logger.level
    previous_min_severity = logger._min_severity
    stdlib_logger.addHandler(handler)
    stdlib_logger.setLevel(logging.WARNING)
    logger._min_severity = "warning"
    try:
        yield messages
    finally:
        logger._min_severity = previous_min_severity
        stdlib_logger.setLevel(previous_level)
        stdlib_logger.removeHandler(handler)


def warnings_in(logged):
    return [message for level, message in logged if level == "WARNING"]


def errors_in(logged):
    return [message for level, message in logged if level == "ERROR"]


@pytest.mark.asyncio
async def test_records_become_one_compact_json_line_each_in_submission_order(tmp_path):
    writer = make_writer(tmp_path)
    await writer.start()
    records = [{"n": 1, "text": "café"}, {"n": 2, "nested": {"a": [1, 2.5, None]}}, {"n": 3}]
    for record in records:
        writer.submit(record)
    await writer.close()

    assert written_lines(tmp_path) == [
        json.dumps(record, separators=(",", ":")) + "\n" for record in records
    ]
    assert writer.dropped_records == 0


@pytest.mark.asyncio
async def test_the_file_is_named_after_the_stem_host_pid_and_start_time(tmp_path):
    writer = make_writer(tmp_path)
    await writer.start()
    writer.submit({"n": 1})
    await writer.close()

    (output_file,) = tmp_path.glob("*.jsonl")
    host = re.escape(socket.gethostname())
    assert re.fullmatch(
        rf"records-test-{host}-{os.getpid()}-\d{{8}}T\d{{6}}Z\.jsonl", output_file.name
    )


@pytest.mark.asyncio
async def test_a_second_start_changes_nothing(tmp_path):
    writer = make_writer(tmp_path)
    await writer.start()
    path, task = writer._path, writer._task
    await writer.start()

    assert writer._path == path and writer._task is task, "start() is idempotent"
    writer.submit({"n": 1})
    await writer.close()
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]
    assert written_values(tmp_path) == [1]


@pytest.mark.asyncio
async def test_queued_records_are_appended_in_batches_of_at_most_batch_size(tmp_path, monkeypatch):
    writer = make_writer(tmp_path, batch_size=4)
    await writer.start()
    appends = []
    real_write = writer._write

    def counting_write(data):
        appends.append(data.count("\n"))
        real_write(data)

    monkeypatch.setattr(writer, "_write", counting_write)
    for index in range(10):  # all queued before the drain task first runs
        writer.submit({"n": index})
    await writer.close()

    assert appends == [4, 4, 2]
    assert written_values(tmp_path) == list(range(10))


@pytest.mark.asyncio
async def test_a_full_queue_drops_the_newest_records_and_counts_them(tmp_path, logged):
    writer = make_writer(tmp_path, queue_size=2)
    await writer.start()
    for index in range(5):  # no await in between: the drain task cannot keep up
        writer.submit({"n": index})
    await writer.close()

    assert written_values(tmp_path) == [0, 1]
    assert writer.dropped_records == 3
    # Rate-limited: the first drop is logged, the following ones are only counted.
    assert len(warnings_in(logged)) == 1
    assert "Dropped 1 test records" in warnings_in(logged)[0]


@pytest.mark.asyncio
async def test_a_record_json_cannot_represent_is_dropped_alone(tmp_path, logged):
    writer = make_writer(tmp_path)
    await writer.start()
    writer.submit({"n": 1})
    writer.submit({"n": float("nan")})  # refused: allow_nan=False
    writer.submit({"n": {1, 2}})  # a set has no JSON form
    writer.submit({"n": 4})
    await writer.close()

    assert written_values(tmp_path) == [1, 4]
    assert writer.dropped_records == 2
    assert len(warnings_in(logged)) == 1, (
        "the first malformed record is logged, the next only counted"
    )
    assert "Dropped malformed test record: " in warnings_in(logged)[0]


class _IdOnlyWriter(JsonlRecordWriter):
    """A configuration: writes ``{"n": ...}`` only, refuses records without ``n`` (KeyError)."""

    malformed_record_errors = (KeyError, TypeError, ValueError)

    def _serialize(self, record):
        if record.get("break"):
            raise RuntimeError("serializer bug")
        return super()._serialize({"n": record["n"]})


@pytest.mark.asyncio
async def test_a_configuration_decides_the_line_and_which_errors_are_malformed(tmp_path, logged):
    writer = _IdOnlyWriter(
        str(tmp_path),
        file_stem="records-test",
        subject="test",
        queue_size=16,
        batch_size=4,
        shutdown_timeout=5,
    )
    await writer.start()
    writer.submit({"n": 1, "ignored": True})
    writer.submit({"no_id": True})  # KeyError: malformed for this configuration
    writer.submit({"n": 3, "break": True})  # RuntimeError: unexpected, dropped with a traceback
    writer.submit({"n": 4})
    await writer.close()

    assert written_lines(tmp_path) == ['{"n":1}\n', '{"n":4}\n']
    assert writer.dropped_records == 2
    assert ["Dropped malformed test record: 'n'" in message for message in warnings_in(logged)] == [
        True
    ]
    (error,) = errors_in(logged)
    assert (
        "Dropped 1 test record(s) that failed to serialize (RuntimeError: serializer bug)" in error
    )
    assert "Traceback (most recent call last)" in error and "_serialize" in error


@pytest.mark.asyncio
async def test_an_unexpected_write_failure_costs_one_batch_not_the_writer(
    tmp_path, monkeypatch, logged
):
    writer = make_writer(tmp_path)
    await writer.start()
    real_write = writer._write
    failures = [RuntimeError("cannot schedule new futures after shutdown")]

    def write_failing_once(data):
        if failures:
            raise failures.pop()
        real_write(data)

    monkeypatch.setattr(writer, "_write", write_failing_once)
    writer.submit({"n": 1})
    writer.submit({"n": 2})  # queued with the first: one batch, one failed append
    await wait_until(lambda: writer.dropped_records > 0 or writer._task.done())
    assert not writer._task.done(), "a failed append must not end the drain task"
    writer.submit({"n": 3})
    await writer.close()

    assert written_values(tmp_path) == [3]
    assert writer.dropped_records == 2, "every record of the failed batch counts as dropped"
    (error,) = errors_in(logged)
    assert "Dropped 2 test record(s) that failed to write (RuntimeError: cannot schedule" in error
    assert "Traceback (most recent call last)" in error and "write_failing_once" in error


@pytest.mark.asyncio
async def test_an_append_the_filesystem_refuses_drops_the_batch_and_warns_once(tmp_path, logged):
    writer = make_writer(tmp_path)
    await writer.start()
    writer._path.mkdir()  # appending to a directory fails with an OSError
    writer.submit({"n": 1})
    await wait_until(lambda: writer.dropped_records == 1)
    writer.submit({"n": 2})
    await wait_until(lambda: writer.dropped_records == 2)
    assert not writer._task.done()
    await writer.close()

    assert writer.dropped_records == 2
    assert len(warnings_in(logged)) == 1, "only the first write error is logged"
    assert "Failed to write test JSONL: " in warnings_in(logged)[0]


def test_unexpected_errors_are_logged_first_then_every_thousandth_per_stage(logged):
    writer = make_writer(None)
    logged_at = []
    for count in range(1, 2001):
        for stage in ("serialize", "write"):
            num_logged = len(logged)
            writer._drop_after_unexpected_error(1, stage, RuntimeError("failed"))
            if len(logged) > num_logged:
                logged_at.append((stage, count))

    assert logged_at == [
        (stage, count) for count in (1, 1000, 2000) for stage in ("serialize", "write")
    ]
    assert writer.dropped_records == 4000


@pytest.mark.asyncio
async def test_close_writes_every_queued_record_before_returning(tmp_path, logged):
    writer = make_writer(tmp_path, queue_size=8, batch_size=2)
    await writer.start()
    for index in range(7):
        writer.submit({"n": index})
    await writer.close()

    assert written_values(tmp_path) == list(range(7))
    assert writer._task is None
    # A closed writer ignores records: they are neither queued (more than the queue holds
    # would be dropped) nor counted as dropped.
    for index in range(9):
        writer.submit({"n": f"after close {index}"})
    assert written_values(tmp_path) == list(range(7))
    assert writer.dropped_records == 0
    assert logged == []


@pytest.mark.asyncio
async def test_close_cancels_a_write_that_outlives_the_shutdown_timeout(
    tmp_path, monkeypatch, logged
):
    """CancelledError is not an Exception: the error containment lets it end the task."""
    writer = make_writer(tmp_path, shutdown_timeout=0.1)
    await writer.start()
    entered = threading.Event()
    release = threading.Event()

    def stuck_write(data):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(writer, "_write", stuck_write)
    writer.submit({"n": 1})
    await wait_until(entered.is_set)
    task = writer._task
    try:
        await writer.close()
    finally:
        release.set()

    assert task.cancelled(), "close() must be able to cancel a drain task stuck in a write"
    assert writer._task is None
    assert any("Timed out flushing test JSONL" in message for message in warnings_in(logged))


@pytest.mark.asyncio
async def test_a_writer_without_an_output_directory_does_nothing(logged):
    writer = make_writer(None, queue_size=1)
    await writer.start()
    writer.submit({"n": 1})
    writer.submit({"n": 2})  # more than the queue holds: ignored, not dropped
    await writer.close()

    assert writer._task is None
    assert writer.dropped_records == 0
    assert logged == []


@pytest.mark.asyncio
async def test_an_output_directory_that_cannot_be_created_disables_the_writer(tmp_path, logged):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    writer = make_writer(blocker / "records")  # its parent is a file: mkdir fails
    await writer.start()
    writer.submit({"n": 1})
    await writer.close()

    assert writer._task is None
    assert writer._output_dir is None
    (error,) = errors_in(logged)
    assert "Disabling test JSONL output: " in error
