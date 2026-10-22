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

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import host_tier_memory

pytestmark = pytest.mark.cpu_only


def test_opt_ins_are_off_unless_set_to_one(monkeypatch):
    for name in (host_tier_memory.DROP_PAGE_CACHE_ENV, host_tier_memory.COLLAPSE_ENV):
        monkeypatch.delenv(name, raising=False)
    assert not host_tier_memory.drop_page_cache_enabled()
    assert not host_tier_memory.collapse_enabled()
    monkeypatch.setenv(host_tier_memory.DROP_PAGE_CACHE_ENV, "0")
    monkeypatch.setenv(host_tier_memory.COLLAPSE_ENV, "true")
    assert not host_tier_memory.drop_page_cache_enabled()
    assert not host_tier_memory.collapse_enabled()
    monkeypatch.setenv(host_tier_memory.DROP_PAGE_CACHE_ENV, "1")
    monkeypatch.setenv(host_tier_memory.COLLAPSE_ENV, "1")
    assert host_tier_memory.drop_page_cache_enabled()
    assert host_tier_memory.collapse_enabled()


def test_drop_page_cache_keeps_file_contents(tmp_path):
    payload = bytes(range(256)) * 4096
    nested = tmp_path / "shard"
    nested.mkdir()
    (nested / "weights.bin").write_bytes(payload)
    (tmp_path / "config.json").write_text("{}")
    host_tier_memory.drop_page_cache([str(tmp_path), None, str(tmp_path / "missing")])
    assert (nested / "weights.bin").read_bytes() == payload
    assert (tmp_path / "config.json").read_text() == "{}"


def test_collapse_targets_only_new_mappings_and_keeps_data():
    before = host_tier_memory.anonymous_mappings()
    size = 128 << 20
    region = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        for offset in range(0, size, mmap.PAGESIZE):
            region[offset] = offset // mmap.PAGESIZE % 251
        new = [r for r in host_tier_memory.anonymous_mappings() if r not in set(before)]
        assert any(end - begin >= size for begin, end in new)
        # Collapsing may fail for lack of free huge pages; it must never alter the data.
        host_tier_memory.collapse_new_mappings(before)
        assert all(
            region[offset] == offset // mmap.PAGESIZE % 251
            for offset in range(0, size, mmap.PAGESIZE)
        )
    finally:
        region.close()
