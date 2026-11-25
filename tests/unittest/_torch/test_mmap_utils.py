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

import mmap
import os
from pathlib import Path

import pytest

from tensorrt_llm._torch import mmap_utils
from tensorrt_llm._torch.mmap_utils import (
    page_cache_bytes,
    populate_file_pages,
    release_file_page_cache,
    wait_for_page_migration_to_settle,
)

pytestmark = pytest.mark.cpu_only


def test_populate_file_pages_reports_exact_bounded_progress(tmp_path):
    # MADV_POPULATE_READ requires Linux >= 5.14: population warms the whole
    # file on capable kernels (reporting every window, including the trailing
    # partial one), returns 0 where unsupported, and the documented contract
    # also admits a partial count on a transient mid-file failure. The
    # invariants are exact per-window accounting and boundedness.
    payload = os.urandom(2 * mmap.PAGESIZE + 123)
    file = tmp_path / "blob.bin"
    file.write_bytes(payload)

    windows: list[int] = []
    populated = populate_file_pages(str(file), mmap.PAGESIZE, windows.append)

    assert 0 <= populated <= len(payload)
    assert sum(windows) == populated
    if windows:
        assert max(windows) <= mmap.PAGESIZE


def test_populate_file_pages_rejects_bad_window_sizes(tmp_path):
    # A zero window would never advance the populate loop and an unaligned one
    # would fail from the second window on; both are programming errors.
    file = tmp_path / "blob.bin"
    file.write_bytes(b"x" * mmap.PAGESIZE)

    for bad_window in (0, -mmap.PAGESIZE, mmap.PAGESIZE + 1):
        with pytest.raises(ValueError):
            populate_file_pages(str(file), bad_window)


def test_populate_file_pages_empty_and_missing_files(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.touch()
    assert populate_file_pages(str(empty), mmap.PAGESIZE) == 0
    assert populate_file_pages(str(tmp_path / "missing.bin"), mmap.PAGESIZE) == 0


def test_release_file_page_cache_advises_every_readable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.safetensors"
    second = tmp_path / "b.safetensors"
    first.write_bytes(b"x" * 3000)
    second.write_bytes(b"y" * 5000)
    advised = []
    real_fadvise = os.posix_fadvise

    def spy(fd: int, offset: int, length: int, advice: int) -> None:
        advised.append((os.fstat(fd).st_ino, offset, length, advice))
        real_fadvise(fd, offset, length, advice)

    monkeypatch.setattr(os, "posix_fadvise", spy)

    paths = [str(first), str(tmp_path / "missing.safetensors"), str(second)]
    assert release_file_page_cache(paths) == 8000
    assert advised == [
        (first.stat().st_ino, 0, 0, os.POSIX_FADV_DONTNEED),
        (second.stat().st_ino, 0, 0, os.POSIX_FADV_DONTNEED),
    ]


class _FakeTime:
    """Stands in for mmap_utils' ``time``: ``sleep`` advances ``monotonic`` at once,
    by ``oversleep`` times the requested time."""

    def __init__(self, oversleep: float = 1.0) -> None:
        self.now = 100.0
        self.oversleep = oversleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds * self.oversleep


def _migration_counter(
    monkeypatch: pytest.MonkeyPatch,
    readings: list[int | None],
    then: int | None,
    oversleep: float = 1.0,
) -> None:
    """Make ``pgmigrate_success`` read ``readings`` in order, then ``then`` forever."""
    values = iter(readings)
    monkeypatch.setattr(mmap_utils, "_migrated_pages", lambda: next(values, then))
    monkeypatch.setattr(mmap_utils, "time", _FakeTime(oversleep))


# One poll's growth just above 1 MiB/s over a 0.5 s poll, in pages.
_BUSY_STEP = (1 << 20) // 2 // mmap.PAGESIZE + 1


def test_wait_for_page_migration_to_settle_returns_once_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _migration_counter(monkeypatch, [], then=100)

    waited = wait_for_page_migration_to_settle(
        max_wait_s=60.0, quiet_s=5.0, poll_s=0.5, max_bytes_per_s=1 << 20
    )

    assert waited == 5.0


def test_wait_for_page_migration_to_settle_waits_out_a_busy_spell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Busy for 4 polls (2 s), then flat: quiet from the last busy poll on.
    _migration_counter(monkeypatch, [_BUSY_STEP * i for i in range(5)], then=_BUSY_STEP * 4)

    waited = wait_for_page_migration_to_settle(
        max_wait_s=60.0, quiet_s=5.0, poll_s=0.5, max_bytes_per_s=1 << 20
    )

    assert waited == 2.0 + 5.0


def test_wait_for_page_migration_to_settle_threshold_is_in_bytes_per_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Growing by exactly the threshold is quiet; one page more per poll is busy.
    at_threshold = (1 << 20) // 2 // mmap.PAGESIZE
    _migration_counter(monkeypatch, [at_threshold * i for i in range(100)], then=None)
    assert (
        wait_for_page_migration_to_settle(
            max_wait_s=60.0,
            quiet_s=5.0,
            poll_s=0.5,
            max_bytes_per_s=at_threshold * mmap.PAGESIZE / 0.5,
        )
        == 5.0
    )

    _migration_counter(monkeypatch, [(at_threshold + 1) * i for i in range(1000)], then=None)
    assert (
        wait_for_page_migration_to_settle(
            max_wait_s=60.0,
            quiet_s=5.0,
            poll_s=0.5,
            max_bytes_per_s=at_threshold * mmap.PAGESIZE / 0.5,
        )
        == 60.0
    )


def test_wait_for_page_migration_to_settle_measures_the_real_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each 0.5 s poll sleeps 1 s: growth at the threshold over 1 s is quiet,
    # though it would exceed the threshold over the nominal 0.5 s.
    per_second = (1 << 20) // mmap.PAGESIZE
    _migration_counter(monkeypatch, [per_second * i for i in range(100)], then=None, oversleep=2.0)

    waited = wait_for_page_migration_to_settle(
        max_wait_s=60.0, quiet_s=5.0, poll_s=0.5, max_bytes_per_s=per_second * mmap.PAGESIZE
    )

    assert waited == 5.0


def test_wait_for_page_migration_to_settle_stops_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _migration_counter(monkeypatch, [_BUSY_STEP * i for i in range(1000)], then=None)

    waited = wait_for_page_migration_to_settle(
        max_wait_s=3.0, quiet_s=5.0, poll_s=0.5, max_bytes_per_s=1 << 20
    )

    assert waited == 3.0


def test_wait_for_page_migration_to_settle_without_vmstat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _migration_counter(monkeypatch, [], then=None)
    assert wait_for_page_migration_to_settle() == 0.0

    # A read that fails later ends the wait with the time waited so far.
    _migration_counter(monkeypatch, [100, 100], then=None)
    assert wait_for_page_migration_to_settle(poll_s=0.5) == 1.0


@pytest.mark.parametrize(
    "content, expected",
    [
        ("pgmigrate_fail 7\npgmigrate_success 1234\n", 1234),
        ("pgmigrate_fail 7\n", None),
        ("pgmigrate_success not-a-number\n", None),
        (None, None),
    ],
)
def test_migrated_pages_parses_vmstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None, expected: int | None
) -> None:
    vmstat = tmp_path / "vmstat"
    if content is not None:
        vmstat.write_text(content)
    monkeypatch.setattr(mmap_utils, "_VMSTAT_PATH", str(vmstat))

    assert mmap_utils._migrated_pages() == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        ("MemFree:  10 kB\nCached:   2048 kB\n", 2048 * 1024),
        ("MemFree:  10 kB\n", None),
        ("Cached:\n", None),
        (None, None),
    ],
)
def test_page_cache_bytes_parses_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None, expected: int | None
) -> None:
    meminfo = tmp_path / "meminfo"
    if content is not None:
        meminfo.write_text(content)
    monkeypatch.setattr(mmap_utils, "_MEMINFO_PATH", str(meminfo))

    assert page_cache_bytes() == expected
