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
"""The sharded JSONL writer lands every record durably and accounts for every record.

Each submitted record is classified exactly once -- persisted (in its shard,
newline-terminated, flushed and fsync()ed), dropped by reason, unknown (bytes may
have landed but the write or fsync holding them failed) or pending -- and the
accounting is published beside the shards with each shard's durable byte range.
These drive the writer through the failures that break that claim: a write cut
mid-record inside a UTF-8 character, a full disk, a failed fsync, an open that
fails with EEXIST, a stalled filesystem with a saturated queue, a writer that dies
and a successor appending to the same file, and a sidecar write that fails.
"""

import asyncio
import errno
import json
import logging
import os
import threading
from pathlib import Path

import pytest

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base import sharded_jsonl_writer
from tensorrt_llm.serve._base.sharded_jsonl_writer import (
    DROP_REASONS,
    SIDECAR_DIR,
    SIDECAR_SCHEMA,
    ShardedJsonlRecordWriter,
    verify_shards,
)

pytestmark = pytest.mark.cpu_only

BUCKET = "2026-01-01T00"
KEY = f"{BUCKET}/requests"
SHARD = f"{KEY}-w.jsonl"
QUEUE_SIZE = 16


class _ShardFieldWriter(ShardedJsonlRecordWriter):
    """A layout: the record names its own shard in ``shard``."""

    def _shard_key(self, record):
        return record["shard"]


def make_writer(directory, *, shard_suffix="-w", sidecar_interval=10.0, shutdown_timeout=5):
    return _ShardFieldWriter(
        None if directory is None else str(directory),
        subject="test",
        queue_size=QUEUE_SIZE,
        batch_size=4,
        shutdown_timeout=shutdown_timeout,
        shard_suffix=shard_suffix,
        sidecar_interval=sidecar_interval,
    )


async def started(directory, **kwargs):
    writer = make_writer(directory, **kwargs)
    await writer.start()
    return writer


def record(index, text="ok", key=KEY):
    return {"shard": key, "index": index, "text": text}


def submit(writer, count, start=0, text="ok", key=KEY):
    for index in range(start, start + count):
        writer.submit(record(index, text, key))


def lines_of(directory, file=SHARD):
    return [
        json.loads(line)
        for line in (Path(directory) / file).read_text(encoding="utf-8").splitlines()
        if line
    ]


def sidecar(directory, writer):
    return json.loads((Path(directory) / SIDECAR_DIR / f"{writer.generation}.json").read_text())


def balanced(document):
    """The books balance, and the shards' watermarks agree with the counts."""
    counts = document["counts"]
    accounted = (
        counts["persisted"]
        + sum(counts["dropped"].values())
        + counts["unknown"]
        + counts["pending"]
    )
    in_shards = sum(shard["persisted_records"] for shard in document["shards"])
    return (
        counts["submitted"] == accounted
        and counts["pending"] >= 0
        and in_shards == counts["persisted"]
    )


def totals_agree(writer):
    """The base class's drop total and the per-reason books never diverge."""
    return writer.dropped_records == sum(writer.dropped.values())


async def until(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never became true"
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


class _CutWrite:
    """A shard file whose first payload write lands only a prefix, then fails."""

    def __init__(self, handle, cut, error):
        self._handle = handle
        self._cut = cut
        self._error = error

    def write(self, data):
        if self._cut is not None and len(data) > 1:
            cut, self._cut = self._cut, None
            self._handle.write(data[:cut])
            self._handle.flush()
            raise self._error
        return self._handle.write(data)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


def cut_next_shard_write(monkeypatch, cut, error):
    """Make the next payload append to a shard land ``cut`` bytes, then raise.

    Shards are opened with ``os.open`` and wrapped by ``os.fdopen(fd, "ab")``; the
    wrapper is where the write can be cut.
    """
    real_fdopen = os.fdopen
    armed = {"on": True}

    def fdopen(fd, mode="r", *args, **kwargs):
        handle = real_fdopen(fd, mode, *args, **kwargs)
        if armed["on"] and mode == "ab":
            armed["on"] = False
            return _CutWrite(handle, cut, error)
        return handle

    monkeypatch.setattr(sharded_jsonl_writer.os, "fdopen", fdopen)


def fail_next_shard_opens(monkeypatch, count):
    """Make the next ``count`` opens of a shard for appending fail with EEXIST."""
    real_open = os.open
    left = {"opens": count}

    def opener(path, flags, *args, **kwargs):
        if left["opens"] and flags & os.O_APPEND and str(path).endswith(".jsonl"):
            left["opens"] -= 1
            raise FileExistsError(errno.EEXIST, "File exists", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sharded_jsonl_writer.os, "open", opener)
    return left


# -- layout -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_land_in_their_shard_as_one_json_line_each(tmp_path):
    writer = await started(tmp_path)
    submit(writer, 2)
    submit(writer, 1, start=10, key=f"{BUCKET}/responses")
    submit(writer, 1, start=20, key="2026-01-01T01/requests")
    await writer.close()

    assert [line["index"] for line in lines_of(tmp_path)] == [0, 1]
    assert [line["index"] for line in lines_of(tmp_path, f"{BUCKET}/responses-w.jsonl")] == [10]
    assert [line["index"] for line in lines_of(tmp_path, "2026-01-01T01/requests-w.jsonl")] == [20]
    assert (tmp_path / SHARD).read_text(encoding="utf-8") == "".join(
        json.dumps(record(index), separators=(",", ":")) + "\n" for index in (0, 1)
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "2026-01-01T00",
        "2026-01-01T01",
        SIDECAR_DIR,
    ]


@pytest.mark.asyncio
async def test_the_default_suffix_names_the_host_pid_and_start_time(tmp_path):
    writer = await started(tmp_path, shard_suffix=None)
    submit(writer, 1)
    await writer.close()

    (path,) = (tmp_path / BUCKET).glob("*.jsonl")
    assert path.name == f"requests{writer._shard_suffix}.jsonl"
    import re
    import socket

    host = re.escape(socket.gethostname())
    assert re.fullmatch(rf"requests-{host}-{os.getpid()}-\d{{8}}T\d{{6}}Z\.jsonl", path.name)


@pytest.mark.asyncio
async def test_non_ascii_text_is_kept_and_a_lone_surrogate_is_escaped_losslessly(tmp_path, logged):
    writer = await started(tmp_path)
    clean = record(0, "你好")
    poisoned = record(1, "你 \ud83d")  # half a surrogate pair: UTF-8 cannot encode it
    writer.submit(clean)
    writer.submit(poisoned)  # same batch as its clean batchmate
    writer.submit(record(2, "cut \udfff here"))
    await writer.close()

    clean_line, poisoned_line, second_line = (
        (tmp_path / SHARD).read_text(encoding="utf-8").splitlines()
    )
    assert "你好" in clean_line, "a clean record keeps raw UTF-8"
    assert poisoned_line.isascii() and second_line.isascii()
    assert json.loads(clean_line) == clean
    assert json.loads(poisoned_line) == poisoned, "escaping is lossless"
    assert writer.sanitized_records == 2 and writer.dropped_records == 0
    assert sidecar(tmp_path, writer)["attributes"] == {"sanitized": 2}
    assert [m for m in logged if "Escaped 2 test record(s)" in m[1]], (
        "the first batch with escapes is logged, with the running total"
    )


@pytest.mark.asyncio
async def test_a_record_without_a_shard_is_dropped_alone(tmp_path, logged):
    writer = await started(tmp_path)
    submit(writer, 1)
    writer.submit({"index": 1})  # KeyError in _shard_key: malformed for this layout
    writer.submit({"shard": KEY, "index": 2, "bad": float("nan")})  # refused by allow_nan=False
    submit(writer, 1, start=3)
    await writer.close()

    assert [line["index"] for line in lines_of(tmp_path)] == [0, 3]
    assert writer.dropped == {
        "queue_full": 0,
        "unserializable": 2,
        "write_error": 0,
        "shutdown": 0,
    }
    assert totals_agree(writer)
    assert balanced(sidecar(tmp_path, writer))
    assert len([m for m in logged if "Dropped malformed test record" in m[1]]) == 1


@pytest.mark.asyncio
async def test_an_unexpected_shard_key_failure_is_dropped_with_its_traceback(tmp_path, logged):
    class _BrokenLayout(_ShardFieldWriter):
        def _shard_key(self, record):
            if record.get("break"):
                raise RuntimeError("layout bug")
            return super()._shard_key(record)

    writer = _BrokenLayout(
        str(tmp_path),
        subject="test",
        queue_size=QUEUE_SIZE,
        batch_size=4,
        shutdown_timeout=5,
        shard_suffix="-w",
    )
    await writer.start()
    submit(writer, 1)
    writer.submit({"shard": KEY, "index": 1, "break": True})
    await writer.close()

    assert [line["index"] for line in lines_of(tmp_path)] == [0]
    assert writer.dropped["unserializable"] == 1 and totals_agree(writer)
    (error,) = [m for level, m in logged if level == "ERROR"]
    assert "failed to place (RuntimeError: layout bug)" in error and "_shard_key" in error


@pytest.mark.asyncio
async def test_an_output_directory_that_cannot_be_created_disables_the_writer(tmp_path, logged):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    writer = make_writer(blocker / "records")
    await writer.start()
    submit(writer, 1)
    await writer.close()

    assert writer._task is None and writer._output_dir is None
    assert writer.submitted == 0 and not list(tmp_path.glob("**/*.jsonl"))
    assert [m for level, m in logged if level == "ERROR" and "Disabling test JSONL output" in m]


# -- books --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_appends_are_persisted_and_reconcile(tmp_path):
    writer = await started(tmp_path)
    submit(writer, QUEUE_SIZE)  # as many as the queue holds without yielding to the drain
    await writer.close()

    document = sidecar(tmp_path, writer)
    assert document["schema"] == SIDECAR_SCHEMA
    assert document["closed"] is True
    assert document["subject"] == "test" and document["shard_suffix"] == "-w"
    assert document["counts"] == {
        "submitted": QUEUE_SIZE,
        "persisted": QUEUE_SIZE,
        "dropped": dict.fromkeys(DROP_REASONS, 0),
        "unknown": 0,
        "pending": 0,
    }
    (shard,) = document["shards"]
    assert shard["file"] == SHARD
    assert shard["start_offset"] == 0
    assert shard["persisted_end"] == (tmp_path / SHARD).stat().st_size
    (check,) = verify_shards(str(tmp_path), document)
    assert (check["records"], check["bad_lines"], check["unknown_lines"]) == (QUEUE_SIZE, 0, 0)
    assert writer.last_write_at is not None and writer.last_write_at.endswith("+00:00")


@pytest.mark.asyncio
async def test_a_record_counts_as_persisted_only_after_its_fsync(tmp_path, monkeypatch):
    writer = await started(tmp_path)
    synced = []
    real_fsync = os.fsync

    def counting_fsync(fd):
        real_fsync(fd)
        synced.append(fd)

    monkeypatch.setattr(sharded_jsonl_writer.os, "fsync", counting_fsync)
    submit(writer, 3)
    await until(lambda: writer.persisted == 3)
    assert synced, "records were counted as persisted without an fsync"
    await writer.close()


@pytest.mark.asyncio
async def test_a_write_cut_inside_a_utf8_character_is_unknown_not_persisted(tmp_path, monkeypatch):
    """Bytes that landed without an fsync are never claimed, and never poison the next record."""
    writer = await started(tmp_path)
    first = json.dumps(record(0, "你好"), ensure_ascii=False, separators=(",", ":")) + "\n"
    second = json.dumps(record(1, "你好"), ensure_ascii=False, separators=(",", ":")) + "\n"
    # Inside the first byte of the second record's three-byte character.
    cut = len(first.encode()) + second.encode().index("你".encode()) + 1
    cut_next_shard_write(monkeypatch, cut, OSError(errno.ENOSPC, "No space left on device"))

    submit(writer, 4, text="你好")
    await until(lambda: writer._write_error_count == 1)
    assert writer.persisted == 0
    # Record 0 landed whole and record 1 in part: neither is claimed. Records 2 and 3
    # never reached the file.
    assert writer.unknown == 2
    assert writer.dropped["write_error"] == 2 and totals_agree(writer)

    # The disk frees up: the next append must land whole and parse.
    submit(writer, 1, start=4, text="after")
    await writer.close()

    document = sidecar(tmp_path, writer)
    assert balanced(document)
    assert document["counts"]["persisted"] == 1 and document["counts"]["unknown"] == 2
    (shard,) = document["shards"]
    assert shard["fragments_isolated"] == 1
    assert shard["unknown_extents"] == [{"start": 0, "landed": cut, "records": 2}]
    (check,) = verify_shards(str(tmp_path), document)
    assert (check["records"], check["bad_lines"], check["unknown_lines"]) == (1, 0, 2)

    lines = (tmp_path / SHARD).read_bytes().split(b"\n")
    # The fragment is one bad line; the record after it parses untouched.
    assert json.loads(lines[-2]) == record(4, "after")
    with pytest.raises(ValueError):
        json.loads(lines[-3])


@pytest.mark.asyncio
async def test_a_full_disk_drops_the_group_and_a_later_append_lands(tmp_path, monkeypatch, logged):
    writer = await started(tmp_path)
    cut_next_shard_write(monkeypatch, 0, OSError(errno.EDQUOT, "Disk quota exceeded"))
    submit(writer, 3)
    await until(lambda: writer._write_error_count == 1)
    assert writer.dropped["write_error"] == 3 and writer.unknown == 0

    submit(writer, 2, start=3)
    await writer.close()

    document = sidecar(tmp_path, writer)
    assert balanced(document)
    assert document["counts"]["persisted"] == 2
    assert "Disk quota exceeded" in document["errors"]["last_error"]
    (check,) = verify_shards(str(tmp_path), document)
    assert (check["records"], check["bad_lines"]) == (2, 0)
    assert len([m for m in logged if "Failed to write test JSONL" in m[1]]) == 1


@pytest.mark.asyncio
async def test_a_partial_batch_failure_charges_only_the_unwritten_group(tmp_path):
    """A group that cannot be written must not drag its persisted batchmates into the drops."""
    writer = await started(tmp_path)
    # Plant a file where the second group's directory must go: its mkdir fails after the
    # first group's append has already landed, both groups being in one batch.
    (tmp_path / "2026-01-01T01").write_text("a file, not a directory")
    submit(writer, 1)
    submit(writer, 1, start=1, key="2026-01-01T01/requests")
    await writer.close()

    assert [line["index"] for line in lines_of(tmp_path)] == [0]
    assert writer.persisted == 1 and writer.dropped["write_error"] == 1
    assert writer._write_error_count == 1
    assert balanced(sidecar(tmp_path, writer))


@pytest.mark.asyncio
async def test_an_open_failing_with_eexist_is_retried_and_its_records_persist(
    tmp_path, monkeypatch
):
    """Appending opens without O_EXCL, so EEXIST is spurious; nothing was written yet."""
    monkeypatch.setattr(sharded_jsonl_writer, "_SHARD_OPEN_RETRY_DELAYS", (0, 0, 0))
    writer = await started(tmp_path)
    left = fail_next_shard_opens(monkeypatch, 2)
    submit(writer, 3)
    await writer.close()

    assert left["opens"] == 0
    document = sidecar(tmp_path, writer)
    assert balanced(document)
    assert document["counts"]["persisted"] == 3
    assert document["errors"]["write_errors"] == 0
    (check,) = verify_shards(str(tmp_path), document)
    assert (check["records"], check["bad_lines"]) == (3, 0)


@pytest.mark.asyncio
async def test_an_open_that_keeps_failing_with_eexist_drops_the_group(tmp_path, monkeypatch):
    monkeypatch.setattr(sharded_jsonl_writer, "_SHARD_OPEN_RETRY_DELAYS", (0, 0, 0))
    writer = await started(tmp_path)
    fail_next_shard_opens(monkeypatch, 4)  # the first open and all three retries
    submit(writer, 3)
    await until(lambda: writer._write_error_count == 1)
    assert writer.dropped["write_error"] == 3 and writer.unknown == 0

    submit(writer, 2, start=3)
    await writer.close()
    document = sidecar(tmp_path, writer)
    assert balanced(document)
    assert document["counts"]["persisted"] == 2
    assert "File exists" in document["errors"]["last_error"]


@pytest.mark.asyncio
async def test_a_failed_fsync_leaves_its_records_unknown(tmp_path, monkeypatch):
    """The bytes are in the page cache, perhaps on disk: not durable, so not persisted."""
    writer = await started(tmp_path)
    real_fsync = os.fsync
    failures = iter([OSError(errno.EIO, "Input/output error")])

    def flaky_fsync(fd):
        if os.readlink(f"/proc/self/fd/{fd}").endswith(".jsonl"):
            error = next(failures, None)
            if error is not None:
                raise error
        real_fsync(fd)

    monkeypatch.setattr(sharded_jsonl_writer.os, "fsync", flaky_fsync)
    submit(writer, 3)
    await until(lambda: writer._write_error_count == 1)
    assert (writer.persisted, writer.unknown) == (0, 3)

    submit(writer, 1, start=3)
    await writer.close()
    document = sidecar(tmp_path, writer)
    assert balanced(document)
    assert (document["counts"]["persisted"], document["counts"]["unknown"]) == (1, 3)
    (check,) = verify_shards(str(tmp_path), document)
    assert (check["records"], check["bad_lines"], check["unknown_lines"]) == (1, 0, 3)


def stall_shard_fsync(monkeypatch):
    """Block every shard fsync until released; the sidecar's fsync still runs."""
    release = threading.Event()
    entered = threading.Event()
    real_fsync = os.fsync

    def stalled_fsync(fd):
        if os.readlink(f"/proc/self/fd/{fd}").endswith(".jsonl"):
            entered.set()
            release.wait(timeout=30)
        real_fsync(fd)

    monkeypatch.setattr(sharded_jsonl_writer.os, "fsync", stalled_fsync)
    return entered, release


@pytest.mark.asyncio
async def test_a_stalled_filesystem_saturates_the_queue_and_still_balances(
    tmp_path, monkeypatch, logged
):
    writer = await started(tmp_path)
    entered, release = stall_shard_fsync(monkeypatch)
    submit(writer, 1)
    await until(entered.is_set)
    # One record is in flight, stuck in fsync; fill the queue past its bound.
    submit(writer, QUEUE_SIZE + 7, start=1)

    books = writer.accounting()
    assert books["counts"]["dropped"]["queue_full"] == 7
    assert books["counts"]["persisted"] == 0
    assert books["counts"]["pending"] == 1 + QUEUE_SIZE
    assert balanced(books) and totals_agree(writer)
    assert len([m for m in logged if "Dropped 1 test records" in m[1]]) == 1

    release.set()
    await writer.close()
    document = sidecar(tmp_path, writer)
    assert balanced(document)
    assert document["counts"]["persisted"] == 1 + QUEUE_SIZE
    assert document["counts"]["pending"] == 0
    (check,) = verify_shards(str(tmp_path), document)
    assert (check["records"], check["bad_lines"]) == (1 + QUEUE_SIZE, 0)


@pytest.mark.asyncio
async def test_a_shutdown_during_a_stall_drops_the_queue_and_claims_nothing_in_flight(
    tmp_path, monkeypatch
):
    writer = await started(tmp_path, shutdown_timeout=0.2)
    entered, release = stall_shard_fsync(monkeypatch)
    submit(writer, 1)
    await until(entered.is_set)
    submit(writer, 5, start=1)
    try:
        await writer.close()
        document = sidecar(tmp_path, writer)
        assert document["closed"] is True
        assert balanced(document)
        # The batch stuck in fsync may yet land, so it is not claimed; what was still
        # queued is dropped as never written.
        assert document["counts"]["unknown"] == 1
        assert document["counts"]["dropped"]["shutdown"] == 5
        assert document["counts"]["persisted"] == 0
        assert totals_agree(writer)
    finally:
        release.set()
    # The abandoned write may still land once released. Its records stay unknown and no
    # watermark moves for them: only the event loop books an outcome, and that drain is gone.
    await asyncio.sleep(0.2)
    assert balanced(writer.accounting(closed=True))
    assert all(shard["persisted_records"] == 0 for shard in writer.accounting()["shards"])


@pytest.mark.asyncio
async def test_a_dead_writer_and_its_successor_stay_separable(tmp_path):
    first = await started(tmp_path)
    submit(first, 10)
    await until(lambda: first.persisted == 10)
    await first._publish(closed=False)
    # SIGKILL: no close, no final sidecar, and an append cut mid-record.
    first._sidecar_task.cancel()
    first._task.cancel()
    await asyncio.gather(first._task, first._sidecar_task, return_exceptions=True)
    first._task = first._sidecar_task = None
    with (tmp_path / SHARD).open("ab") as shard:
        shard.write(b'{"shard":"x","index":10,"te')
    size_at_death = (tmp_path / SHARD).stat().st_size

    second = await started(tmp_path)
    assert second.generation != first.generation
    submit(second, 5, start=100)
    await second.close()

    dead, alive = sidecar(tmp_path, first), sidecar(tmp_path, second)
    assert dead["closed"] is False and alive["closed"] is True
    assert balanced(dead) and balanced(alive)
    (dead_check,) = verify_shards(str(tmp_path), dead)
    (alive_check,) = verify_shards(str(tmp_path), alive)
    assert (dead_check["records"], dead_check["bad_lines"]) == (10, 0)
    assert (alive_check["records"], alive_check["bad_lines"]) == (5, 0)
    assert alive_check["start_offset"] == size_at_death
    assert alive_check["fragments_isolated"] == 1
    # The successor's first record starts on a line of its own.
    lines = (tmp_path / SHARD).read_bytes().split(b"\n")
    assert json.loads(lines[11]) == record(100)


# -- sidecar publication ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_publication_leaves_the_previous_sidecar_whole(
    tmp_path, monkeypatch, logged
):
    writer = await started(tmp_path)
    submit(writer, 2)
    await until(lambda: writer.persisted == 2)
    await writer._publish(closed=False)
    before = sidecar(tmp_path, writer)
    assert before["counts"]["persisted"] == 2

    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(sharded_jsonl_writer.os, "replace", failing_replace)
    submit(writer, 1, start=2)
    await until(lambda: writer.persisted == 3)
    await writer._publish(closed=False)
    assert sidecar(tmp_path, writer) == before, "still the last whole document"
    assert writer._sidecar_errors == 1
    assert [m for m in logged if "Failed to publish test writer accounting" in m[1]]

    monkeypatch.setattr(sharded_jsonl_writer.os, "replace", real_replace)
    await writer.close()
    after = sidecar(tmp_path, writer)
    assert after["errors"]["sidecar_errors"] == 1
    assert after["counts"]["persisted"] == 3 and balanced(after)


@pytest.mark.asyncio
async def test_a_quiet_writer_catches_its_sidecar_up(tmp_path):
    """The last burst before a lull is published without waiting for more traffic."""
    writer = await started(tmp_path, sidecar_interval=0.2)
    submit(writer, 3)
    await until(lambda: writer.persisted == 3)
    assert sidecar(tmp_path, writer)["counts"]["persisted"] == 0, "only the start-up sidecar yet"
    await until(lambda: sidecar(tmp_path, writer)["counts"]["persisted"] == 3)
    document = sidecar(tmp_path, writer)
    assert document["closed"] is False and balanced(document)
    await writer.close()


@pytest.mark.asyncio
async def test_a_failed_publication_is_retried_while_the_writer_is_quiet(tmp_path, monkeypatch):
    """Only a publication that landed marks the books published; a refused one is retried."""
    writer = await started(tmp_path, sidecar_interval=0.2)
    real_replace = os.replace
    refused = []

    def failing_replace(src, dst):
        refused.append(dst)
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(sharded_jsonl_writer.os, "replace", failing_replace)
    submit(writer, 3)
    await until(lambda: writer.persisted == 3)
    await until(lambda: refused)  # the periodic publication, refused by the filesystem
    assert sidecar(tmp_path, writer)["counts"]["persisted"] == 0
    monkeypatch.setattr(sharded_jsonl_writer.os, "replace", real_replace)

    await until(lambda: sidecar(tmp_path, writer)["counts"]["persisted"] == 3)
    document = sidecar(tmp_path, writer)
    assert document["closed"] is False
    assert document["errors"]["sidecar_errors"] == len(refused)
    assert balanced(document)
    await writer.close()


@pytest.mark.asyncio
async def test_a_failed_batch_publishes_at_once(tmp_path, monkeypatch):
    """A write failure is in the sidecar before the next periodic publication."""
    writer = await started(tmp_path, sidecar_interval=60.0)
    cut_next_shard_write(monkeypatch, 0, OSError(errno.EDQUOT, "Disk quota exceeded"))
    submit(writer, 2)
    await until(lambda: writer._write_error_count == 1)
    await until(lambda: sidecar(tmp_path, writer)["counts"]["dropped"]["write_error"] == 2)
    assert "Disk quota exceeded" in sidecar(tmp_path, writer)["errors"]["last_error"]
    await writer.close()


@pytest.mark.asyncio
async def test_files_are_private_and_directories_closed_whatever_the_umask(tmp_path):
    previous = os.umask(0o022)  # the common permissive default
    try:
        writer = make_writer(tmp_path / "out")
        await writer.start()
        submit(writer, 2)
        await writer.close()
    finally:
        os.umask(previous)

    modes = {
        path.relative_to(tmp_path).as_posix(): oct(path.stat().st_mode & 0o777)
        for path in sorted((tmp_path / "out").rglob("*"))
    }
    assert modes == {
        "out/2026-01-01T00": "0o700",
        f"out/{SHARD}": "0o600",
        f"out/{SIDECAR_DIR}": "0o700",
        f"out/{SIDECAR_DIR}/{writer.generation}.json": "0o600",
    }, "the trace holds prompts and completions; nothing the writer creates is readable by others"
    assert oct((tmp_path / "out").stat().st_mode & 0o777) == "0o700"


@pytest.mark.asyncio
async def test_a_deleted_hour_directory_is_recreated_and_the_old_claims_marked(tmp_path):
    writer = await started(tmp_path)
    submit(writer, 1)
    await until(lambda: writer.persisted == 1)
    import shutil

    shutil.rmtree(tmp_path / BUCKET)  # an operator rotates the current hour away
    submit(writer, 4, start=1)
    await writer.close()

    assert (tmp_path / BUCKET).is_dir(), "the hour directory is made again on the next append"
    assert [line["index"] for line in lines_of(tmp_path)] == [1, 2, 3, 4]
    assert writer.persisted == 5 and writer.dropped_records == 0
    document = sidecar(tmp_path, writer)
    assert balanced(document)
    old, new = document["shards"]
    assert old["file"] == new["file"] == SHARD
    assert old.get("superseded") is True and old["persisted_records"] == 1, (
        "the first record's bytes are gone; the sidecar says so rather than claiming them"
    )
    assert "superseded" not in new and new["start_offset"] == 0 and new["persisted_records"] == 4
    old_check, new_check = verify_shards(str(tmp_path), document)
    assert (new_check["records"], new_check["bad_lines"]) == (4, 0)
    assert old_check["superseded"] is True, (
        "a reconciliation sees the mark and knows the old range describes bytes that are gone"
    )


@pytest.mark.asyncio
async def test_records_are_serialized_on_the_worker_thread_and_encoded_once(tmp_path, monkeypatch):
    """The event loop only queues records; json.dumps runs on the writer's thread, once each."""
    writer = await started(tmp_path)
    calls = []
    real_dumps = json.dumps

    def spying_dumps(*args, **kwargs):
        calls.append(threading.current_thread() is threading.main_thread())
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(sharded_jsonl_writer.json, "dumps", spying_dumps)
    submit(writer, 3, text="你好")
    await until(lambda: writer.persisted == 3)
    on_loop = [on_main for on_main in calls if on_main]
    assert len(calls) >= 3 and not on_loop, (
        f"{len(on_loop)} json.dumps call(s) ran on the event loop thread; the sidecar is the "
        "only JSON the loop may wait for"
    )
    record_dumps = [c for c in calls]  # one dumps per record: no second pass to test encodability
    assert len(record_dumps) - 3 <= 2, "the sidecar publication is the only other json.dumps"
    await writer.close()


@pytest.mark.asyncio
async def test_a_record_the_encoder_fails_on_unexpectedly_is_dropped_alone(tmp_path, logged):
    class _BrokenEncoder(_ShardFieldWriter):
        def _encode_record(self, record):
            if record.get("break"):
                raise RuntimeError("encoder bug")
            return super()._encode_record(record)

    writer = _BrokenEncoder(
        str(tmp_path),
        subject="test",
        queue_size=QUEUE_SIZE,
        batch_size=4,
        shutdown_timeout=5,
        shard_suffix="-w",
    )
    await writer.start()
    submit(writer, 1)
    writer.submit({"shard": KEY, "index": 1, "break": True})
    submit(writer, 1, start=2)
    await writer.close()

    assert [line["index"] for line in lines_of(tmp_path)] == [0, 2]
    assert writer.dropped["unserializable"] == 1 and totals_agree(writer)
    assert balanced(sidecar(tmp_path, writer))
    (error,) = [m for level, m in logged if level == "ERROR"]
    assert "failed to serialize (RuntimeError: encoder bug)" in error and "_encode_record" in error


def test_verify_shards_reports_a_missing_file_and_a_short_range(tmp_path):
    (tmp_path / BUCKET).mkdir()
    (tmp_path / SHARD).write_bytes(b'{"a":1}\n')
    document = {
        "shards": [
            {"file": SHARD, "start_offset": 0, "persisted_end": 40, "unknown_extents": []},
            {"file": "missing.jsonl", "start_offset": 0, "persisted_end": 1},
        ]
    }
    present, missing = verify_shards(str(tmp_path), document)
    assert (present["records"], present["short"], present["size"]) == (1, True, 8)
    assert missing["error"].startswith("FileNotFoundError")
