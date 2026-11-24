# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
from unittest.mock import patch

import pytest

from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _force_mpi_topology_mapping():
    with patch("tensorrt_llm.mapping.mpi_disabled", return_value=False):
        yield


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "attention_dp", "shares_requests"),
    [
        (1, 4, True, True),  # helix + attention DP at tp 1: one DP group
        (2, 2, True, False),  # attention DP across the 2 original tp ranks
        (2, 2, False, True),  # helix without attention DP
    ],
)
def test_repurposed_helix_mapping_for_full_rows(tp_size, cp_size, attention_dp,
                                                shares_requests):
    world = tp_size * cp_size
    helix = Mapping(
        world_size=world,
        rank=world - 1,
        tp_size=tp_size,
        cp_size=cp_size,
        cp_config={"cp_type": "HELIX"},
        enable_attention_dp=attention_dp,
    )
    repurposed = helix.repurpose_helix_cp_to_tp()
    assert repurposed.helix_repurposed_cp_size == cp_size
    assert repurposed.tp_group_shares_requests is shares_requests
    assert Mapping.from_dict(
        repurposed.to_dict()).helix_repurposed_cp_size == cp_size

    full_rows = repurposed.mapping_for_full_rows()
    if attention_dp and shares_requests:
        assert full_rows is not repurposed
        assert not full_rows.enable_attention_dp
        assert (full_rows.tp_size, full_rows.tp_rank) == (world,
                                                          repurposed.tp_rank)
        assert repurposed.enable_attention_dp  # the source mapping is untouched
    else:
        assert full_rows is repurposed


@pytest.mark.parametrize(("world_size", "pp_size"), [(1, 1), (2, 2)])
def test_plain_attention_dp_at_tp1_keeps_attention_dp(world_size, pp_size):
    # Not helix: attention DP at tp_size 1 must not be read as a shared TP group,
    # or the LM head and the vocab embedding disagree on their TP mode.
    mapping = Mapping(world_size=world_size,
                      rank=0,
                      tp_size=1,
                      pp_size=pp_size,
                      enable_attention_dp=True)
    assert not mapping.tp_group_shares_requests
    assert mapping.mapping_for_full_rows() is mapping


class TestMapping(unittest.TestCase):

    def test_mapping(self):
        m = Mapping(world_size=8, rank=0, tp_size=8)
        self.assertEqual(len(m.tp_groups), 1)
        self.assertEqual(len(m.pp_groups), 8)
        self.assertEqual(m.tp_group, [0, 1, 2, 3, 4, 5, 6, 7])

        m = Mapping(world_size=8, rank=0, tp_size=4, pp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(m.tp_group, [0, 1, 2, 3])
        self.assertEqual(m.pp_group, [0, 4])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 4)

        m = Mapping(world_size=8, rank=6, tp_size=2, pp_size=4)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(m.tp_group, [6, 7])
        self.assertEqual(m.pp_group, [0, 2, 4, 6])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 0)

        m = Mapping(world_size=2, rank=0, cp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(len(m.cp_groups), 1)
        self.assertEqual(m.tp_group, [0])
        self.assertEqual(m.pp_group, [0])
        self.assertEqual(m.cp_group, [0, 1])

        m = Mapping(world_size=8, rank=3, tp_size=2, pp_size=2, cp_size=2)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(len(m.cp_groups), 4)
        self.assertEqual(m.tp_group, [1, 3])
        self.assertEqual(m.pp_group, [3, 7])
        self.assertEqual(m.cp_group, [2, 3])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertTrue(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 7)
        self.assertEqual(m.next_pp_rank(), 7)
        self.assertEqual(m.prev_cp_rank(), 2)
        self.assertEqual(m.next_cp_rank(), 2)

        m = Mapping(world_size=16, rank=9, tp_size=2, pp_size=2, cp_size=4)
        self.assertEqual(m.tp_group, [9, 13])
        self.assertEqual(m.pp_group, [1, 9])
        self.assertEqual(m.cp_group, [8, 9, 10, 11])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertFalse(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 1)
        self.assertEqual(m.next_pp_rank(), 1)
        self.assertEqual(m.prev_cp_rank(), 8)
        self.assertEqual(m.next_cp_rank(), 10)


@pytest.mark.parametrize(
    "cp_size,num_kv_heads,attention_dp,draft_cp",
    [
        (32, 16, False, 2),  # helix32, 16 KV heads
        (32, 1, False, 32),  # MLA latent: one KV head, pages over all 32 ranks
        (8, 1, False, 8),  # K3 helix8 MLA drafter
        (16, 16, False, None),  # heads already cover the group: keep head TP
        (32, 12, False, None),  # 12 does not divide 32: keep head TP
        (32, 16, True, None),  # attention DP replicates the drafter
    ],
)
def test_helix_mapping_for_kv_heads(cp_size, num_kv_heads, attention_dp,
                                    draft_cp):
    for rank in (0, 5, cp_size - 1):
        drafter = Mapping(
            world_size=cp_size,
            rank=rank,
            tp_size=1,
            cp_size=cp_size,
            cp_config={
                "cp_type": "HELIX"
            },
            enable_attention_dp=attention_dp,
        ).repurpose_helix_cp_to_tp()
        mapping = drafter.helix_mapping_for_kv_heads(num_kv_heads)
        if draft_cp is None:
            assert mapping is drafter
            continue
        assert mapping.has_cp_helix() and not mapping.enable_attention_dp
        assert (mapping.tp_size, mapping.cp_size) == (num_kv_heads, draft_cp)
        # Same KV head as the head-TP replica layout (rank // draft_cp), and
        # the o_proj grid (tp x cp folded back) keeps the rank order.
        assert (mapping.tp_rank, mapping.cp_rank) == (rank // draft_cp,
                                                      rank % draft_cp)
        assert mapping.repurpose_helix_cp_to_tp().tp_rank == rank

    plain = Mapping(world_size=32, rank=3, tp_size=32)
    assert plain.helix_mapping_for_kv_heads(16) is plain


@pytest.mark.parametrize(
    "tp_size,cp_size,num_kv_heads,draft_cp",
    [
        (2, 16, 16, 2),  # draft cp 2 divides helix cp 16
        (2, 8, 1, None),  # MLA would need draft cp 16 > helix cp 8
        (8, 2, 4, None),  # draft cp 4 does not divide helix cp 2
    ],
)
def test_helix_mapping_for_kv_heads_target_tp(tp_size, cp_size, num_kv_heads,
                                              draft_cp):
    drafter = Mapping(
        world_size=tp_size * cp_size,
        rank=5,
        tp_size=tp_size,
        cp_size=cp_size,
        cp_config={
            "cp_type": "HELIX"
        },
    ).repurpose_helix_cp_to_tp()
    mapping = drafter.helix_mapping_for_kv_heads(num_kv_heads)
    if draft_cp is None:
        assert mapping is drafter
    else:
        assert (mapping.tp_size, mapping.cp_size) == (num_kv_heads, draft_cp)


@pytest.mark.parametrize("cp_size", [8, 16])
def test_helix_mapping_for_kv_heads_inherits_the_helix_cp_config(cp_size):
    """The drafter's helix mapping can take the target helix mapping's transport.

    With inherit_cp_config it gets the MNNVL all-to-all; by default it keeps
    the NCCL one.
    """
    cp_config = {
        "cp_type": "HELIX",
        "use_nccl_for_alltoall": False,
        "fifo_version": 2,
        "tokens_per_block": 128,
    }
    for rank in (0, cp_size - 1):
        target = Mapping(world_size=cp_size,
                         rank=rank,
                         gpus_per_node=4,
                         tp_size=1,
                         cp_size=cp_size,
                         moe_ep_size=cp_size,
                         cp_config=dict(cp_config))
        repurposed = target.repurpose_helix_cp_to_tp()
        # The repurposed mapping itself is unchanged: no CP, empty cp_config.
        assert repurposed.cp_config == {} and repurposed.cp_size == 1
        assert repurposed.helix_repurposed_cp_config == target.cp_config
        assert Mapping.from_dict(repurposed.to_dict()).helix_repurposed_cp_config \
            == target.cp_config

        default = repurposed.helix_mapping_for_kv_heads(1)
        assert default.has_cp_helix()
        assert default.cp_config.get("use_nccl_for_alltoall", True)

        inherited = repurposed.helix_mapping_for_kv_heads(
            1, inherit_cp_config=True)
        assert inherited.has_cp_helix()
        assert inherited.cp_config == target.cp_config
        assert not inherited.cp_config["use_nccl_for_alltoall"]
        assert (inherited.tp_size, inherited.cp_size,
                inherited.cp_rank) == (1, cp_size, rank)
        # Equal to the target's mapping, so per-mapping helix workspaces
        # (FIFO, fused receive buffers) are shared with the target.
        assert inherited == target and hash(inherited) == hash(target)
        # The source mapping's dict is not aliased.
        inherited.cp_config["fifo_version"] = 1
        assert repurposed.helix_repurposed_cp_config["fifo_version"] == 2


def _ulysses(world_size, rank, tp_size, cp_size, **moe):
    return Mapping(world_size=world_size,
                   rank=rank,
                   tp_size=tp_size,
                   cp_size=cp_size,
                   cp_config={"cp_type": "ULYSSES"},
                   **moe)


@pytest.mark.parametrize("cp_size", [8, 16])
def test_ulysses_cp_runs_moe_ep_over_the_cp_group(cp_size):
    """cep: TP 1, Ulysses CP N, MoE EP N -- one replica of N ranks."""
    for rank in range(cp_size):
        m = _ulysses(cp_size,
                     rank,
                     1,
                     cp_size,
                     moe_tp_size=1,
                     moe_ep_size=cp_size)
        assert m.has_cp_ulysses() and not m.has_cp_helix()
        assert (m.moe_tp_size, m.moe_ep_size) == (1, cp_size)
        assert m.moe_tp_cluster_ep_size == cp_size
        assert (m.moe_group_rank, m.moe_ep_rank, m.moe_tp_rank) == (rank, rank,
                                                                    0)
        assert m.moe_ep_group == list(range(cp_size))
        assert m.moe_tp_group == [rank]
        assert m.cp_group == list(range(cp_size)) and m.tp_group == [rank]
        # Attention and the KV cache keep the TP layout over the TP x CP ranks.
        assert (m.attn_tp_size, m.attn_cp_size) == (cp_size, 1)
        assert Mapping.from_dict(m.to_dict()) == m


def test_ulysses_cp_moe_groups_follow_the_tp_major_cp_minor_rank():
    for rank in range(8):
        m = _ulysses(8, rank, 2, 4, moe_ep_size=8)
        assert (m.moe_tp_size, m.moe_ep_size) == (1, 8)
        assert m.moe_group_rank == m.tp_rank * 4 + m.cp_rank == rank
        assert m.moe_ep_rank == rank and m.moe_ep_group == list(range(8))
        # MoE TP 2 x EP 4 over a TP 1 x CP 8 group.
        m = _ulysses(8, rank, 1, 8, moe_tp_size=2, moe_ep_size=4)
        assert (m.moe_tp_rank, m.moe_ep_rank) == (rank // 4, rank % 4)
        assert m.moe_ep_group == list(range(rank // 4 * 4, rank // 4 * 4 + 4))
        assert m.moe_tp_group == [rank % 4, rank % 4 + 4]


def test_ulysses_cp_moe_defaults_to_tp_over_tp_x_cp():
    m = _ulysses(8, 3, 2, 4)
    assert (m.moe_tp_size, m.moe_ep_size) == (8, 1)
    assert m.moe_tp_rank == 3


def test_cp_without_a_cp_type_is_ulysses():
    m = Mapping(world_size=4, rank=1, cp_size=4)
    assert m.has_cp_ulysses() and m.cp_config == {}
    assert Mapping(world_size=4, rank=1, tp_size=4).has_cp_ulysses() is False
    assert _ulysses(1, 0, 1, 1).has_cp_ulysses() is False


def test_helix_and_ring_moe_ranks_are_unchanged():
    # Helix keeps its raw-mapping MoE ranks (models read the repurposed mapping).
    for rank in range(8):
        m = Mapping(world_size=8,
                    rank=rank,
                    tp_size=1,
                    cp_size=8,
                    cp_config={"cp_type": "HELIX"},
                    moe_tp_size=2,
                    moe_ep_size=4)
        assert (m.moe_tp_rank, m.moe_ep_rank) == (0, 0)
        assert m.moe_group_rank == m.tp_rank == 0
        assert (m.attn_tp_size, m.attn_cp_size) == (1, 8)
        rep = m.repurpose_helix_cp_to_tp()
        assert (rep.moe_tp_rank, rep.moe_ep_rank) == (rank // 4, rank % 4)
    with pytest.raises(NotImplementedError):
        Mapping(world_size=8,
                rank=0,
                cp_size=8,
                cp_config={"cp_type": "RING"},
                moe_ep_size=8)
    m = Mapping(world_size=8, rank=5, tp_size=8, moe_ep_size=8)
    assert m.moe_group_rank == m.moe_ep_rank == 5
