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
"""Tests for the KVCacheManagerV2 per-conversation probe cache.

These tests use mock objects and do NOT require GPU: KVCacheManagerV2 is
instantiated via ``__new__`` with only the attributes the probe-cache paths
touch, and ``impl`` is a mock.
"""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

CONV = "conv-42"


def _make_manager(mode="1", max_entries=8192, enable_block_reuse=True, is_draft=False):
    mgr = KVCacheManagerV2.__new__(KVCacheManagerV2)
    mgr.enable_block_reuse = enable_block_reuse
    mgr.is_draft = is_draft
    mgr._conv_probe_cache_mode = mode
    mgr._conv_probe_cache = OrderedDict()
    mgr._conv_probe_cache_max_entries = max_entries
    mgr.impl = MagicMock()
    return mgr


def _make_request(conv_id=CONV, lora_task_id=None, cache_salt_id=None, is_dummy=False):
    disagg = SimpleNamespace(conversation_id=conv_id) if conv_id is not None else None
    return SimpleNamespace(
        py_request_id=1,
        py_disaggregated_params=disagg,
        lora_task_id=lora_task_id,
        cache_salt_id=cache_salt_id,
        is_dummy_request=is_dummy,
    )


def _make_kv_cache(block_keys):
    kv_cache = MagicMock()
    kv_cache.committed_block_keys.return_value = list(block_keys)
    return kv_cache


def _store(mgr, block_keys, **request_kwargs):
    mgr._store_conv_probe_keys(_make_request(**request_kwargs), _make_kv_cache(block_keys))


class TestStoreConvProbeKeys:

    def test_stores_committed_chain(self):
        mgr = _make_manager()
        _store(mgr, [b"k0", b"k1"])
        assert mgr._conv_probe_cache == {(CONV, None, None): [b"k0", b"k1"]}

    def test_disabled_mode_stores_nothing(self):
        mgr = _make_manager(mode="0")
        _store(mgr, [b"k0"])
        assert not mgr._conv_probe_cache

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(conv_id=None),
            dict(is_dummy=True),
        ],
    )
    def test_skips_requests_without_conv_key_or_dummy(self, kwargs):
        mgr = _make_manager()
        _store(mgr, [b"k0"], **kwargs)
        assert not mgr._conv_probe_cache

    def test_skips_draft_manager_and_empty_chain(self):
        _draft = _make_manager(is_draft=True)
        _store(_draft, [b"k0"])
        assert not _draft._conv_probe_cache

        mgr = _make_manager()
        _store(mgr, [])
        assert not mgr._conv_probe_cache

    def test_latest_turn_overwrites_and_lru_evicts(self):
        mgr = _make_manager(max_entries=2)
        _store(mgr, [b"k0"], conv_id="a")
        _store(mgr, [b"k0", b"k1"], conv_id="a")
        assert mgr._conv_probe_cache[("a", None, None)] == [b"k0", b"k1"]

        _store(mgr, [b"k0"], conv_id="b")
        _store(mgr, [b"k0"], conv_id="c")  # evicts "a" (oldest)
        assert set(mgr._conv_probe_cache) == {("b", None, None), ("c", None, None)}

    def test_scope_is_part_of_the_cache_key(self):
        mgr = _make_manager()
        _store(mgr, [b"k0"], lora_task_id=3, cache_salt_id=5)
        assert (CONV, 3, 5) in mgr._conv_probe_cache
        assert (CONV, None, None) not in mgr._conv_probe_cache


class TestProbeFastPath:

    def test_hit_uses_by_keys_and_skips_hashing(self):
        mgr = _make_manager()
        _store(mgr, [b"k0", b"k1"])
        mgr.impl.probe_reuse_by_keys.return_value = 256

        result = mgr.probe_prefix_match_length(list(range(300)), conv_key=CONV)

        assert result == 256
        mgr.impl.probe_reuse_by_keys.assert_called_once()
        assert mgr.impl.probe_reuse_by_keys.call_args.args[1] == [b"k0", b"k1"]
        mgr.impl.probe_reuse.assert_not_called()

    def test_hit_is_clamped_to_probe_length(self):
        mgr = _make_manager()
        _store(mgr, [b"k0", b"k1"])
        mgr.impl.probe_reuse_by_keys.return_value = 256

        assert mgr.probe_prefix_match_length(list(range(100)), conv_key=CONV) == 100

    def test_zero_fast_result_falls_back_to_full_probe(self):
        # The cached chain may be fully evicted while a cross-conversation
        # shared prefix (e.g. common system prompt) still matches.
        mgr = _make_manager()
        _store(mgr, [b"k0"])
        mgr.impl.probe_reuse_by_keys.return_value = 0
        mgr.impl.probe_reuse.return_value = 64

        assert mgr.probe_prefix_match_length(list(range(100)), conv_key=CONV) == 64
        mgr.impl.probe_reuse.assert_called_once()

    def test_miss_uses_full_probe(self):
        mgr = _make_manager()
        mgr.impl.probe_reuse.return_value = 128

        assert mgr.probe_prefix_match_length(list(range(300)), conv_key=CONV) == 128
        mgr.impl.probe_reuse_by_keys.assert_not_called()

    def test_no_conv_key_uses_full_probe(self):
        mgr = _make_manager()
        _store(mgr, [b"k0"])
        mgr.impl.probe_reuse.return_value = 128

        assert mgr.probe_prefix_match_length(list(range(300))) == 128
        mgr.impl.probe_reuse_by_keys.assert_not_called()

    def test_disabled_mode_ignores_conv_key(self):
        mgr = _make_manager(mode="0")
        mgr._conv_probe_cache[(CONV, None, None)] = [b"k0"]
        mgr.impl.probe_reuse.return_value = 128

        assert mgr.probe_prefix_match_length(list(range(300)), conv_key=CONV) == 128
        mgr.impl.probe_reuse_by_keys.assert_not_called()

    def test_shadow_mode_runs_both_and_returns_full(self):
        mgr = _make_manager(mode="shadow")
        _store(mgr, [b"k0", b"k1"])
        mgr.impl.probe_reuse_by_keys.return_value = 256
        mgr.impl.probe_reuse.return_value = 200

        assert mgr.probe_prefix_match_length(list(range(300)), conv_key=CONV) == 200
        mgr.impl.probe_reuse_by_keys.assert_called_once()
        mgr.impl.probe_reuse.assert_called_once()

    def test_block_reuse_disabled_short_circuits(self):
        mgr = _make_manager(enable_block_reuse=False)
        assert mgr.probe_prefix_match_length(list(range(10)), conv_key=CONV) == 0
        mgr.impl.probe_reuse.assert_not_called()

    def test_hit_refreshes_lru_order(self):
        mgr = _make_manager(max_entries=2)
        _store(mgr, [b"k0"], conv_id="a")
        _store(mgr, [b"k0"], conv_id="b")
        mgr.impl.probe_reuse_by_keys.return_value = 1

        mgr.probe_prefix_match_length([1, 2], conv_key="a")  # touch "a"
        _store(mgr, [b"k0"], conv_id="c")  # evicts "b", not "a"
        assert set(mgr._conv_probe_cache) == {("a", None, None), ("c", None, None)}
