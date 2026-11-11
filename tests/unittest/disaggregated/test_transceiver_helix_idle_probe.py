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
"""A cheap idle generation transfer-status poll under helix context parallelism.

The generation transfer-status poll runs two pickled collectives on every executor
iteration. With helix CP (where the idle-poll skip does not apply), an idle poll is
decided by one fixed-size int64 allgather of the per-rank receive-session counts.
These tests simulate the ranks of one model-parallel group and check that (1) every
rank takes the same branch and (2) the result is that of the full consensus.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

pytestmark = pytest.mark.cpu_only


class _GroupDist:
    """One rank's view of a simulated TP x CP group whose collectives see every rank's input."""

    def __init__(self, rank: int, group: "_Group"):
        self.rank = rank
        self.group = group

    def tp_cp_allgather_int64(self, values):
        self.group.int64_calls += 1
        return np.asarray([[c] for c in self.group.session_counts], dtype=np.int64)

    def allgather(self, obj):
        return [obj] * self.group.size


class _Group:
    def __init__(self, size: int):
        self.size = size
        self.int64_calls = 0
        self.session_counts = [0] * size


def _make_transceiver(rank: int, group: _Group, *, probe: bool) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._dist = _GroupDist(rank, group)
    transceiver._mapping = SimpleNamespace(
        enable_attention_dp=False, world_size=group.size, pp_size=1
    )
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = True
    transceiver._gen_idle_probe = probe
    transceiver._gen_deferred_receives = 0
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._gen_allgather = Mock(side_effect=lambda local: [local] * group.size)
    return transceiver


@pytest.mark.parametrize("at_least", [0, 1])
def test_idle_status_poll_matches_full_consensus(at_least: int) -> None:
    group = _Group(4)
    fast = _make_transceiver(0, group, probe=True)
    full = _make_transceiver(0, group, probe=False)
    fast._poll_gen_sessions_for_poll_interval = Mock()
    full._poll_gen_sessions_for_poll_interval = Mock()

    assert tuple(fast.check_gen_transfer_status(at_least)) == tuple(
        full.check_gen_transfer_status(at_least)
    )
    assert full._gen_allgather.call_count == 2
    if at_least == 0:
        # Idle everywhere: one int64 allgather, no pickled consensus.
        assert group.int64_calls == 1
        fast._gen_allgather.assert_not_called()
    else:
        # At-least-N polls keep their original pacing and collectives.
        assert group.int64_calls == 0
        assert fast._gen_allgather.call_count == 2


def test_status_poll_runs_full_consensus_when_any_rank_receives() -> None:
    group = _Group(4)
    group.session_counts = [0, 0, 1, 0]
    ranks = [_make_transceiver(r, group, probe=True) for r in range(group.size)]
    # Only rank 2 holds a session; ranks 0, 1 and 3 must still enter the consensus.
    for transceiver in ranks:
        assert tuple(transceiver.check_gen_transfer_status(0)) == ([], [], [])
        assert transceiver._gen_allgather.call_count == 2
    assert group.int64_calls == group.size


def test_probe_reads_the_live_session_count() -> None:
    group = _Group(2)
    transceiver = _make_transceiver(0, group, probe=True)
    transceiver._recv_sessions = {7: object(), 8: object()}
    seen = []

    def _gather(values):
        seen.append(list(values))
        return np.zeros((group.size, 1), dtype=np.int64)

    transceiver._dist.tp_cp_allgather_int64 = _gather
    assert transceiver._any_rank_has_recv_sessions() is False
    assert seen == [[2]]


def _policy(**mapping) -> bool:
    fields = dict(tp_size=1, cp_size=8, pp_size=1, world_size=8, enable_attention_dp=False)
    fields.update(mapping)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = SimpleNamespace(**fields)
    transceiver._dist = SimpleNamespace(allgather=lambda obj: [obj])
    transceiver._init_sync_policy()
    return transceiver._gen_idle_probe


def test_probe_is_limited_to_one_helix_tp_cp_group() -> None:
    # Helix GEN without attention DP: the world is the TP x CP group.
    assert _policy() is True
    assert _policy(tp_size=2, cp_size=4) is True
    # Without CP the idle-poll skip applies instead; a single rank needs no
    # consensus; PP > 1 and attention DP gather over other ranks.
    assert _policy(tp_size=8, cp_size=1) is False
    assert _policy(cp_size=1, world_size=1) is False
    assert _policy(pp_size=2, world_size=16) is False
    assert _policy(enable_attention_dp=True) is False
