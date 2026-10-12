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
"""The DeepSeek-V4.1-Flash-NVFP4 target's host-side decoder replay bookkeeping.

The target's core runs decoder SWA bounded replay as the engine's cache manager resolved it,
once per manager, and keeps a request's tail stash between its context chunks only
while the cache manager still holds the request. Under attention DP the MoE pads every
rank's rows to the step's largest per-rank token count. All three are host logic,
checked here on stand-in objects; importing the target needs a real build, as for the
target contract.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    modeling,
)

Core = modeling.DeepseekV41UnifiedModelingV2


class _Manager:
    """The two attributes of the V4.1 cache manager the bookkeeping reads."""

    def __init__(self, decoder_replay: bool = True, held=()):
        self.v41_decoder_replay = decoder_replay
        self.kv_cache_map = {rid: object() for rid in held}


def _core(replay_requested: bool = True, records=()):
    return SimpleNamespace(
        replay_requested=replay_requested,
        replay=replay_requested,
        _replay_mgr=None,
        _tail_records={rid: modeling._TailRecord(0) for rid in records},
    )


def test_replay_follows_each_cache_manager():
    core = _core()
    first, final = _Manager(True), _Manager(False)
    Core._resolve_replay(core, first)
    assert core.replay is True
    # The engine builds a KV-size estimation manager, then the final one.
    Core._resolve_replay(core, final)
    assert core.replay is False
    # Resolved once per manager, not re-read every step.
    final.v41_decoder_replay = True
    Core._resolve_replay(core, final)
    assert core.replay is False

    core = _core(replay_requested=False)
    Core._resolve_replay(core, _Manager(False))
    assert core.replay is False
    with pytest.raises(AssertionError, match="resolved decoder replay on"):
        Core._resolve_replay(_core(replay_requested=False), _Manager(True))


def test_stash_is_released_with_the_request():
    """A request freed between two of its context chunks (cancelled, or evicted for
    recompute) never reaches its last chunk: its record and stash go once the cache
    manager no longer holds it. Held requests, suspended ones included, keep theirs."""
    core = _core(records=(11, 12, 13))
    stash = object()
    core._tail_records[12].stash = {"h": stash}
    Core._release_tail_records(core, _Manager(held=(11, 12, 13, 99)))
    assert set(core._tail_records) == {11, 12, 13}
    refs = sys.getrefcount(stash)
    Core._release_tail_records(core, _Manager(held=(11, 13)))
    assert set(core._tail_records) == {11, 13}
    assert sys.getrefcount(stash) == refs - 1, "the released stash is still referenced"
    Core._release_tail_records(core, _Manager(held=()))
    assert core._tail_records == {}


def test_dp_rows_are_the_engine_rows():
    """Attention DP: up to the last KV-source layer every rank pads its MoE rows to the
    step's largest per-rank token count, and its own count must be the engine's. Past
    that layer under decoder replay a rank's rows can outnumber its step rows (a small
    last chunk still runs its whole tail, the rest from the stash) and a bounded rank
    runs them in slices, so the ranks exchange their passes instead (``_dp_late_rows``,
    an engine collective; ``_dp_pass_rows`` below)."""
    core = SimpleNamespace(tp=4, tp_rank=2)
    md = SimpleNamespace(all_rank_num_tokens=[7, 2048, 1932, 5])
    assert Core._dp_rows(core, md, 1932) == 2048
    for rows in (0, 128, 1933):
        with pytest.raises(AssertionError):
            Core._dp_rows(core, md, rows)


def test_dp_passes_over_the_bounded_layers():
    """Attention DP under decoder replay: every rank runs as many passes over the bounded
    layers as the rank with the most (its slices when bounded, else its step rows as
    one pass), each padded to the max of that pass's row counts over the ranks that hold
    it; a rank without the pass joins it with zero rows."""
    # Rank 0 bounded in two slices (a 1-row last chunk with a 127-row stash, then more
    # tails); rank 1 bounded with no decoder row; ranks 2 and 3 unbounded.
    gathered = [[2, 2048, 1000], [0], [1, 6], [1, 3]]
    assert modeling._dp_pass_rows(gathered) == [2048, 1000]
    assert modeling._dp_pass_rows([[1, 4], [1, 128], [1, 1], [1, 1]]) == [128]
    assert modeling._dp_pass_rows([[1, 7]] * 4) == [7]
