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

import itertools
import mmap
import os

import pytest

from tensorrt_llm._torch import mmap_utils
from tensorrt_llm._torch.mmap_utils import (
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


def test_release_file_page_cache_counts_the_advised_files(tmp_path):
    first = tmp_path / "a.safetensors"
    second = tmp_path / "b.safetensors"
    first.write_bytes(b"x" * 3000)
    second.write_bytes(b"y" * 5000)

    paths = [str(first), str(tmp_path / "missing.safetensors"), str(second)]
    assert release_file_page_cache(paths) == 8000


def test_wait_for_page_migration_to_settle_returns_once_quiet(monkeypatch):
    monkeypatch.setattr(mmap_utils, "_migrated_pages", lambda: 100)

    waited = wait_for_page_migration_to_settle(max_wait_s=10.0, quiet_s=0.05, poll_s=0.01)

    assert 0.05 <= waited < 10.0


def test_wait_for_page_migration_to_settle_is_bounded(monkeypatch):
    migrated = itertools.count(step=10**6)
    monkeypatch.setattr(mmap_utils, "_migrated_pages", lambda: next(migrated))

    waited = wait_for_page_migration_to_settle(max_wait_s=0.05, quiet_s=10.0, poll_s=0.01)

    assert 0.05 <= waited < 10.0


def test_wait_for_page_migration_to_settle_without_vmstat(monkeypatch):
    monkeypatch.setattr(mmap_utils, "_migrated_pages", lambda: None)

    assert wait_for_page_migration_to_settle() == 0.0
