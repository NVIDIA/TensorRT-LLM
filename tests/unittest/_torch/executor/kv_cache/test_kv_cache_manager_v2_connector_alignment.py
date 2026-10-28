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
match only when that combination is actually present, or drops the backoff
when the connector asks for whole local matches instead.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 128


def _manager(
    *, backoff: int, partial_reuse: bool, connector, is_draft: bool = False
) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.reuse_match_backoff = backoff
    manager.enable_partial_reuse = partial_reuse
    manager.is_draft = is_draft
    manager._ledger_tokens_per_block = TOKENS_PER_BLOCK
    if connector is not None:
        manager.kv_connector_manager = connector
    return manager


def _connector(whole_blocks, *, capacity_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        capacity_only=capacity_only,
        worker=SimpleNamespace(requires_block_aligned_reuse=whole_blocks),
    )


def _dropping_connector(drops, *, capacity_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        capacity_only=capacity_only,
        worker=SimpleNamespace(drops_reuse_match_backoff=drops),
    )


@pytest.mark.parametrize(
    "backoff, partial_reuse, connector, is_draft, expected",
    [
        (1, False, _connector(True), False, TOKENS_PER_BLOCK),
        # No backoff: matches of whole blocks already end on a boundary.
        (0, False, _connector(True), False, 1),
        # Partial reuse wants token-granular matches by definition.
        (1, True, _connector(True), False, 1),
        # A connector that serves from any position needs nothing.
        (1, False, _connector(False), False, 1),
        (1, False, SimpleNamespace(worker=object()), False, 1),
        # Only an explicit declaration counts, not any truthy attribute such
        # as the one a MagicMock worker answers with.
        (1, False, _connector("yes"), False, 1),
        (1, False, None, False, 1),
        # Connector doubles that carry no worker at all must not crash __init__.
        (1, False, SimpleNamespace(capacity_only=False), False, 1),
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
        "connector_without_the_attribute",
        "truthy_non_bool_declaration",
        "no_connector",
        "connector_without_a_worker",
        "capacity_only_connector",
        "draft_manager",
    ],
)
def test_reuse_match_alignment(backoff, partial_reuse, connector, is_draft, expected):
    manager = _manager(
        backoff=backoff, partial_reuse=partial_reuse, connector=connector, is_draft=is_draft
    )
    assert manager._connector_reuse_match_alignment() == expected


def test_a_manager_without_a_bound_connector_attribute_keeps_token_granularity():
    """Tests and specialized managers build partial instances that never bind one."""
    manager = object.__new__(KVCacheManagerV2)
    manager.reuse_match_backoff = 1
    manager.enable_partial_reuse = False
    manager._ledger_tokens_per_block = TOKENS_PER_BLOCK
    assert manager._connector_reuse_match_alignment() == 1


@pytest.mark.parametrize(
    "is_draft, connector, expected",
    [
        (False, _dropping_connector(True), True),
        (False, _dropping_connector(False), False),
        # A mock answers truthily; only a declared True counts.
        (False, _dropping_connector("yes"), False),
        (False, SimpleNamespace(worker=object()), False),
        (False, None, False),
        (False, SimpleNamespace(capacity_only=False), False),
        (False, _dropping_connector(True, capacity_only=True), False),
        (True, _dropping_connector(True), False),
    ],
    ids=[
        "asked",
        "not_asked",
        "truthy_but_not_true",
        "no_attribute",
        "no_connector",
        "connector_without_a_worker",
        "capacity_only_connector",
        "draft",
    ],
)
def test_connector_drops_reuse_match_backoff(is_draft, connector, expected):
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = is_draft
    manager.kv_connector_manager = connector
    assert manager._connector_drops_reuse_match_backoff() is expected


def test_the_worker_base_class_declares_neither():
    """Third-party connectors keep upstream reuse unless they opt in."""
    assert KvCacheConnectorWorker.requires_block_aligned_reuse is False
    assert KvCacheConnectorWorker.drops_reuse_match_backoff is False
