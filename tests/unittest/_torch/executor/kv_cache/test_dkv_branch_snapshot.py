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
"""Branch snapshot points of a recurrent-state KV cache manager V2 under DKV, on the CPU.

Under DKV a claim looks up its prompt only to the group's agreed reuse depth, so its own match
cannot show where the prompt leaves the tree; the group agrees on that fork separately, and the
point is cut to a whole DKV block (the radix tree's block)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import Dkv
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)

pytestmark = pytest.mark.cpu_only

TPB = 64
PROMPT = 5000


class _Claim:
    """A fresh cache's match: where the lookup left the tree, what attention pages support, what
    it claimed."""

    def __init__(self, divergence, attention=None, committed=0):
        self.divergence = divergence
        self.attention = divergence if attention is None else attention
        self.num_committed_tokens = committed

    def _get_num_reusable_tokens_before_pruning(self):
        return self.divergence

    def _get_num_reusable_tokens_before_hybrid_pruning(self):
        return self.attention


def _hybrid(dkv_size=None):
    manager = object.__new__(MambaHybridCacheManagerV2)
    fields = dict(
        kv_cache_config=SimpleNamespace(
            block_reuse_config=SimpleNamespace(enable_branch_snapshot=True)
        ),
        _maybe_log_branch_snapshot_counters=lambda: None,
        local_num_mamba_layers=1,
        _snapshot_pruned_tokens_total=0,
        _page_pruned_tokens_total=0,
        _reuse_breakdown=None,
        _branch_snapshot_points={},
        _requested_snapshot_points={},
        _branch_snapshots_skipped_total={},
        _attention_cache_layout_policy=None,
        tokens_per_block=TPB,
        enable_partial_reuse=True,
        _dkv=None if dkv_size is None else Dkv(dkv_size, 0, TPB, lambda obj: [obj] * dkv_size),
        _ledger_tokens_per_block=TPB * (dkv_size or 1),
        _dkv_forks={},
        _inflight_prefix=SimpleNamespace(points_total=0),
    )
    for name, value in fields.items():
        setattr(manager, name, value)
    return manager


def _req(rid=1, position=0):
    return SimpleNamespace(
        py_request_id=rid,
        is_dummy_request=False,
        prompt_len=PROMPT,
        expect_snapshot_points=[],
        context_current_position=position,
        py_conversation_params=None,
    )


def test_without_dkv_the_claim_s_own_match_gives_a_page_aligned_point():
    manager = _hybrid()
    req = _req()
    manager._record_branch_snapshot_point(req, _Claim(1000), PROMPT - 1)
    assert manager._branch_snapshot_points == {1: 960}
    assert 960 in req.expect_snapshot_points


def test_under_dkv_a_claim_cut_at_the_agreement_sees_no_fork():
    """The cep runs' `branch_snapshots_skipped_by_reason={'no_divergence': ...}`: the claim's
    lookup ends at the agreed depth (0 here), so without the group's fork nothing is recorded."""
    manager = _hybrid(dkv_size=8)
    manager._record_branch_snapshot_point(_req(), _Claim(0), 0)
    assert manager._branch_snapshot_points == {}
    assert manager._branch_snapshots_skipped_total == {"no_divergence": 1}


@pytest.mark.parametrize(
    "dkv_size, fork, reused, point",
    [
        (8, 1000, 0, 512),  # inside a page and inside the second DKV block of 512
        (8, 1300, 0, 1024),  # inside the third DKV block
        (8, 1024, 0, 1024),  # on a DKV block boundary
        (8, 400, 0, None),  # shorter than one DKV block: aligned to zero
        (8, 1300, 1024, None),  # the agreed reuse already reaches the fork's block
        (16, 1300, 0, 1024),  # cep16: DKV blocks of 1024
    ],
    ids=["mid_page", "mid_dkv_block", "dkv_block", "below_one_block", "already_reused", "cep16"],
)
def test_under_dkv_the_group_s_fork_gives_a_dkv_block_point(dkv_size, fork, reused, point):
    manager = _hybrid(dkv_size=dkv_size)
    req = _req()
    manager._dkv_forks[1] = (fork, fork, PROMPT - 1)
    # The claim itself was cut at the agreement: its own match ends there.
    manager._record_branch_snapshot_point(req, _Claim(reused, committed=reused), reused)
    if point is None:
        assert manager._branch_snapshot_points == {}
    else:
        assert manager._branch_snapshot_points == {1: point}
        assert point in req.expect_snapshot_points
    # The reuse accounting uses the group's fork, as without DKV.
    assert manager._page_pruned_tokens_total == 0
    assert manager._snapshot_pruned_tokens_total == fork - reused


def test_under_dkv_a_requested_point_is_cut_to_a_dkv_block():
    manager = _hybrid(dkv_size=8)
    owner = _req(rid=3)
    assert manager._request_snapshot_point(owner, 1300)
    assert manager._requested_snapshot_points == {3: {1024}}
    assert not manager._request_snapshot_point(_req(rid=4), 400)  # inside the first DKV block
    without = _hybrid()
    assert without._request_snapshot_point(_req(rid=5), 1300)
    assert without._requested_snapshot_points == {5: {1280}}


def test_the_fork_unit_is_the_dkv_block_only_under_dkv():
    assert _hybrid(dkv_size=8)._fork_tokens_per_block == 512
    assert _hybrid()._fork_tokens_per_block == TPB
