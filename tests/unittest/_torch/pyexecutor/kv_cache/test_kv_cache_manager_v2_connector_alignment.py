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
        # No backoff still aligns: trimming a match to the pages a sliding-window
        # layer holds can end it mid-block.
        (0, False, _connector(True), False, TOKENS_PER_BLOCK),
        # Whole-block declarations override the default partial reuse setting.
        (1, True, _connector(True), False, TOKENS_PER_BLOCK),
        # A connector that serves from any position needs nothing.
        (1, False, _connector(False), False, 1),
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


@pytest.mark.parametrize(
    "backoff, matched, expected", [(0, 128, 128), (1, 128, 0), (1, 256, 128), (129, 256, 0)]
)
def test_external_prefix_keeps_native_lookahead(backoff: int, matched: int, expected: int) -> None:
    manager = _manager(backoff=backoff, partial_reuse=False, connector=None)
    manager.tokens_per_block = TOKENS_PER_BLOCK
    request = SimpleNamespace(prompt_len=512)
    assert manager._connector_safe_prefix_end(request, matched) == expected


@pytest.mark.parametrize(
    "unaligned_end, local_end, served_end, expected",
    [
        # 300 reusable tokens align down to 256: the connector serves those 44 and more.
        (300, 256, 512, (44, 0)),
        # The connector serves nothing: all 44 are recomputed.
        (300, 256, 256, (0, 44)),
        # The connector serves 20 of them: the other 24 are recomputed.
        (300, 256, 276, (20, 24)),
        # A match already on a boundary gives nothing up.
        (256, 256, 512, (0, 0)),
    ],
    ids=["served", "recomputed", "partly_served", "on_a_boundary"],
)
def test_alignment_forfeit_is_split_by_what_the_connector_served(
    unaligned_end: int,
    local_end: int,
    served_end: int,
    expected: tuple[int, int],
) -> None:
    manager = _manager(backoff=1, partial_reuse=False, connector=_connector(True))
    manager._alignment_forfeited_served_tokens = 0
    manager._alignment_forfeited_recomputed_tokens = 0
    kv_cache = SimpleNamespace(_get_num_reusable_tokens_before_alignment=lambda: unaligned_end)

    manager._record_alignment_forfeit(kv_cache, local_end, served_end)

    assert manager._take_alignment_forfeit() == expected
    # Each report covers the iterations since the last one.
    assert manager._take_alignment_forfeit() == (0, 0)


def test_alignment_forfeit_is_not_reported_without_alignment() -> None:
    manager = _manager(backoff=1, partial_reuse=False, connector=_connector(False))
    manager._alignment_forfeited_served_tokens = 0
    manager._alignment_forfeited_recomputed_tokens = 0
    kv_cache = SimpleNamespace(_get_num_reusable_tokens_before_alignment=lambda: 300)

    manager._record_alignment_forfeit(kv_cache, 256, 512)

    assert manager._take_alignment_forfeit() is None


@pytest.mark.parametrize("forfeit", [(44, 20), None], ids=["aligned", "not_aligned"])
def test_alignment_forfeit_reaches_the_iteration_stats(forfeit: tuple[int, int] | None) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache_stats import (
        KVCacheV2IterationStatsReport,
        append_kv_cache_iteration_stats,
    )

    stats: dict = {}
    append_kv_cache_iteration_stats(
        stats,
        KVCacheV2IterationStatsReport(
            by_window_size={}, by_pool_group={}, alignment_forfeited_tokens=forfeit
        ),
    )

    if forfeit is None:
        assert "iterAlignmentForfeitedConnectorTokens" not in stats
        assert "iterAlignmentForfeitedRecomputedTokens" not in stats
    else:
        assert stats["iterAlignmentForfeitedConnectorTokens"] == 44
        assert stats["iterAlignmentForfeitedRecomputedTokens"] == 20
