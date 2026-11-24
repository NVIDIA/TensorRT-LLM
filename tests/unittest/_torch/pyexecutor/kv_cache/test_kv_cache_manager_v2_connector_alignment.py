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
"""How KVCacheManagerV2 shapes reuse matches for a whole-block connector.

A one-model draft's lookahead backoff trims a local match of whole blocks to a
few tokens short of a boundary, and a connector that serves only from a block
boundary could then offer nothing. The manager asks the core to align the
match only when that combination is present. The native lookahead backoff
remains in effect before alignment.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 128


def _manager(
    *, backoff: int, partial_reuse: bool, connector: SimpleNamespace | None, is_draft: bool = False
) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.reuse_match_backoff = backoff
    manager.enable_partial_reuse = partial_reuse
    manager.is_draft = is_draft
    manager._ledger_tokens_per_block = TOKENS_PER_BLOCK
    manager.kv_connector_manager = connector
    return manager


def _connector(whole_blocks: bool | str, *, capacity_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        capacity_only=capacity_only,
        requires_block_aligned_reuse=whole_blocks is True and not capacity_only,
    )


@pytest.mark.parametrize(
    "backoff, partial_reuse, connector, is_draft, expected",
    [
        (1, False, _connector(True), False, TOKENS_PER_BLOCK),
        # No backoff: matches of whole blocks already end on a boundary.
        (0, False, _connector(True), False, 1),
        # Whole-block declarations override the default partial reuse setting.
        (1, True, _connector(True), False, TOKENS_PER_BLOCK),
        # A connector that serves from any position needs nothing.
        (1, False, _connector(False), False, 1),
        # Only an explicit declaration counts, not any truthy attribute such
        # as the one a MagicMock worker answers with.
        (1, False, _connector("yes"), False, 1),
        (1, False, None, False, 1),
        # A capacity-only connector serves nothing, so it never reshapes reuse.
        (1, False, _connector(True, capacity_only=True), False, 1),
        # A paired draft pool aligns too, so the joint claim stays on a boundary.
        (1, False, _connector(True), True, TOKENS_PER_BLOCK),
    ],
    ids=[
        "whole_block_connector_with_backoff",
        "no_backoff",
        "partial_reuse",
        "token_granular_connector",
        "truthy_non_bool_declaration",
        "no_connector",
        "capacity_only_connector",
        "draft_manager",
    ],
)
def test_reuse_match_alignment(
    backoff: int,
    partial_reuse: bool,
    connector: SimpleNamespace | None,
    is_draft: bool,
    expected: int,
) -> None:
    manager = _manager(
        backoff=backoff, partial_reuse=partial_reuse, connector=connector, is_draft=is_draft
    )
    assert manager._connector_reuse_match_alignment() == expected


def test_a_manager_without_a_connector_keeps_token_granularity() -> None:
    """A manager without a connector keeps token-granular matching."""
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_connector_manager = None
    manager.reuse_match_backoff = 1
    manager.enable_partial_reuse = False
    manager._ledger_tokens_per_block = TOKENS_PER_BLOCK
    assert manager._connector_reuse_match_alignment() == 1


def test_the_worker_base_class_keeps_token_granularity() -> None:
    """Third-party connectors keep upstream reuse unless they opt in."""
    assert KvCacheConnectorWorker.requires_block_aligned_reuse is False


@pytest.mark.parametrize(
    "backoff, matched, expected", [(0, 128, 128), (1, 128, 0), (1, 256, 128), (129, 256, 0)]
)
def test_external_prefix_keeps_native_lookahead(backoff: int, matched: int, expected: int) -> None:
    manager = _manager(backoff=backoff, partial_reuse=False, connector=None)
    manager.tokens_per_block = TOKENS_PER_BLOCK
    request = SimpleNamespace(prompt_len=512)
    assert manager._connector_safe_prefix_end(request, matched) == expected


@pytest.mark.parametrize("growth_succeeds", [False, True])
def test_retained_generation_retry_reserves_current_draft_width(growth_succeeds: bool) -> None:
    from unittest.mock import Mock

    manager = object.__new__(KVCacheManagerV2)
    cache = SimpleNamespace(is_active=True, capacity=130, resize=Mock(return_value=growth_succeeds))
    manager.kv_cache_map = {7: cache}
    manager._ensure_generation_beam_width = lambda request, kv_cache: True
    manager._generation_draft_slots = lambda request: 4
    manager._retained_generation_allocations = {7: 1}
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {7: [0]}
    manager._fill_fresh_kv_pages = Mock()
    manager._log_window_crossing = Mock()
    request = SimpleNamespace(py_request_id=7)

    assert manager.try_allocate_generation(request) is growth_succeeds
    cache.resize.assert_called_once_with(133)
    if growth_succeeds:
        assert manager._retained_generation_allocations == {}
        assert manager._allocated_draft_lens == {7: 4}
        assert manager._inflight_draft_lens == {7: [0, 4]}
        manager._fill_fresh_kv_pages.assert_called_once_with(7)
    else:
        assert manager._retained_generation_allocations == {7: 1}
        assert manager._allocated_draft_lens == {}
        assert manager._inflight_draft_lens == {7: [0]}
        manager._fill_fresh_kv_pages.assert_not_called()
