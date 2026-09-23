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
"""Gen-side completion consensus under attention-DP with helix CP / PP.

Under attention-DP the DP groups schedule independently, but the PP ranks
and the helix CP ranks of one DP group share the same requests. The Python
transceiver must therefore still reach completion consensus across the
CP (x PP) group; otherwise one CP rank can start decoding a request whose
KV its partner has not received yet and the helix all-to-all deadlocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2


def _transceiver(mapping, dist) -> KvCacheTransceiverV2:
    tc = object.__new__(KvCacheTransceiverV2)
    tc._mapping = mapping
    tc._dist = dist
    tc._init_sync_policy()
    return tc


def _mapping(**overrides) -> SimpleNamespace:
    base = dict(tp_size=1, pp_size=1, cp_size=1, world_size=1, enable_attention_dp=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_adp_without_cp_or_pp_skips_gen_sync() -> None:
    dist = SimpleNamespace(
        allgather=Mock(side_effect=AssertionError("must not sync")),
        pp_allgather=Mock(side_effect=AssertionError("must not sync")),
        cp_allgather=Mock(side_effect=AssertionError("must not sync")),
    )
    tc = _transceiver(_mapping(tp_size=2, world_size=2, enable_attention_dp=True), dist)

    assert tc._gen_need_sync is False
    assert tc._gen_consensus([7, 8]) == [7, 8]


def test_adp_with_helix_cp_syncs_across_cp_group() -> None:
    # Two helix CP ranks of one DP group: only rid 1 has arrived on both.
    dist = SimpleNamespace(
        allgather=Mock(side_effect=AssertionError("world allgather must be skipped")),
        pp_allgather=Mock(side_effect=AssertionError("pp allgather must be skipped")),
        cp_allgather=lambda payload: [payload, [1]],
    )
    tc = _transceiver(
        _mapping(tp_size=2, cp_size=2, world_size=4, enable_attention_dp=True), dist
    )

    assert tc._gen_need_sync is True
    assert tc._gen_consensus([1, 2]) == [1]


def test_adp_with_helix_cp_and_pp_syncs_across_both() -> None:
    # cp=2, pp=2: cp_allgather returns the two CP ranks of this PP stage,
    # pp_allgather then stacks the other PP stage's CP pair on top.
    dist = SimpleNamespace(
        allgather=Mock(side_effect=AssertionError("world allgather must be skipped")),
        cp_allgather=lambda payload: [payload, [1, 3]],
        pp_allgather=lambda gathered: [gathered, [[1], [1, 2]]],
    )
    tc = _transceiver(
        _mapping(tp_size=1, cp_size=2, pp_size=2, world_size=4, enable_attention_dp=True),
        dist,
    )

    assert tc._gen_need_sync is True
    # Consensus needs all pp_size * cp_size == 4 ranks to agree.
    assert tc._gen_consensus([1, 2, 3]) == [1]


def test_adp_consensus_outcome_completes_only_when_every_cp_rank_completed() -> None:
    dist = SimpleNamespace(
        allgather=Mock(side_effect=AssertionError("world allgather must be skipped")),
        pp_allgather=Mock(side_effect=AssertionError("pp allgather must be skipped")),
        # Peer CP rank: nothing cancelled/failed, only rid 5 completed.
        cp_allgather=lambda payload: [payload, [[], [], [5]]],
    )
    tc = _transceiver(
        _mapping(tp_size=2, cp_size=2, world_size=4, enable_attention_dp=True), dist
    )

    cancelled, failed, completed = tc._consensus_outcome(
        [5, 6], [], [], [5, 6], tc._gen_allgather, tc._gen_need_sync
    )
    assert cancelled == []
    assert failed == []
    assert completed == [5]


def test_non_adp_still_uses_world_allgather() -> None:
    dist = SimpleNamespace(
        allgather=lambda payload: [payload, payload],
        pp_allgather=Mock(side_effect=AssertionError("pp allgather must be skipped")),
        cp_allgather=Mock(side_effect=AssertionError("cp allgather must be skipped")),
    )
    tc = _transceiver(_mapping(tp_size=1, cp_size=2, world_size=2), dist)

    assert tc._gen_need_sync is True
    assert tc._gen_consensus([4]) == [4]
