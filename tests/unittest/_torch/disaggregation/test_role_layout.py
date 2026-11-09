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
"""Role layouts: which pieces of a role's byte space each rank's buffer holds."""

import collections
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import msgpack
import pytest

from tensorrt_llm._torch.disaggregation.base.role_layout import (
    ByHead,
    Own,
    Piece,
    RankTopology,
    Replicated,
    RingSlots,
    RoleLayoutV2,
    Sectioned,
    check_pieces,
    grid_place,
    recurrent_state_split,
    role_layout_from_dict,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

pytestmark = pytest.mark.cpu_only


def _site(size, *, kv_heads=(0, None), state=False, split=("tp",), ranks=1, tokens=32):
    """The facts of one buffer that a layout in a manager's spec resolves against."""
    return SimpleNamespace(
        size=size, kv_heads=kv_heads, state=state, split=split, ranks=ranks, tokens=tokens
    )


def test_head_major_split_holds_its_heads_contiguously():
    layout = ByHead(head_bytes=256, total=8)

    assert layout.pieces(1024, 2, 0) == (Piece(0, 0, 1024),)
    assert layout.pieces(1024, 2, 1) == (Piece(1024, 0, 1024),)
    assert layout.space(1024, 2) == ("head_major", 8, 256)
    check_pieces(layout, 1024, 2, 1)


def test_token_major_split_holds_its_heads_in_every_token():
    # 4 tokens, 4 heads of 16 bytes per token; rank 1 of 2 holds heads 2..3.
    layout = ByHead(token_major=True, head_bytes=64, total=4, tokens=4)

    pieces = layout.pieces(128, 2, 1)

    assert pieces == tuple(Piece((t * 4 + 2) * 16, t * 32, 32) for t in range(4))
    assert layout.space(128, 2) == ("token_major", 4, 4, 16)
    check_pieces(layout, 128, 2, 1)
    # Holding every head, the buffer is the whole space in one piece.
    assert layout.pieces(256, 1, 0) == (Piece(0, 0, 256),)


def test_head_split_overlaps_where_ranks_hold_more_heads_than_the_total():
    # 6 heads over TP4, two per rank: ranks hold heads 0..1, 1..2, 3..4 and 4..5.
    layout = ByHead(head_bytes=10, total=6)

    firsts = [layout.pieces(20, 4, rank)[0].at // 10 for rank in range(4)]

    assert firsts == [0, 1, 3, 4]
    assert not layout.even(4)
    assert layout.even(2) and layout.even(3) and layout.even(6)


def test_head_split_refuses_heads_past_the_total():
    with pytest.raises(ValueError, match="past the 2 heads"):
        ByHead(head_bytes=10, total=2).pieces(20, 2, 1)


def test_head_split_resolves_an_attention_buffer_from_its_kv_heads():
    layout = ByHead().resolve(_site(1024, kv_heads=(4, 8), ranks=2))

    assert layout == ByHead(head_bytes=256, total=8)
    layout.check(1024)


def test_head_split_resolves_recurrent_state_from_its_heads_on_each_rank():
    layout = ByHead(head_bytes=128).resolve(_site(512, state=True, split=("tp", "cp"), ranks=4))

    assert layout == ByHead(head_bytes=128, total=16, split=("tp", "cp"))


def test_head_split_of_attention_without_the_total_of_its_heads_is_the_rank_s_own_share():
    # One KV head on each of 4 ranks may be 4 heads or one head held 4 times: unknown.
    assert ByHead().resolve(_site(1024, kv_heads=(1, None), ranks=4)) == Own(("tp",))


def test_head_split_resolves_to_the_rank_s_own_share_without_whole_heads():
    assert ByHead().resolve(_site(1000, kv_heads=(3, 6), ranks=2)) == Own(("tp",))
    assert ByHead().resolve(_site(1000, kv_heads=(0, None))) == Own(("tp",))


def test_head_split_of_its_own_heads_states_their_total_for_attention():
    with pytest.raises(ValueError, match="total"):
        ByHead(head_bytes=64).resolve(_site(256, kv_heads=(4, 8)))


def test_head_split_keeps_a_total_its_spec_states():
    # MLA's one latent head, which every rank of TP4 holds whole.
    layout = ByHead(total=1).resolve(_site(576, kv_heads=(1, None), ranks=4))

    assert layout == ByHead(head_bytes=576, total=1)
    assert [layout.pieces(576, 4, rank) for rank in range(4)] == [(Piece(0, 0, 576),)] * 4
    assert layout.even(4) and layout.even(8)
    check_pieces(layout, 576, 4, 3)


def test_head_split_stating_its_total_refuses_a_buffer_without_whole_heads():
    with pytest.raises(ValueError, match="1 heads"):
        ByHead(total=1).resolve(_site(576, kv_heads=(0, None)))
    with pytest.raises(ValueError, match="1 heads"):
        ByHead(total=1).resolve(_site(576, state=True))


def test_head_split_replicates_heads_over_more_ranks_than_heads():
    # 2 KV heads over TP4: ranks 0 and 1 hold head 0, ranks 2 and 3 head 1.
    layout = ByHead(head_bytes=10, total=2)

    assert [layout.pieces(10, 4, rank)[0].at for rank in range(4)] == [0, 0, 10, 10]
    assert layout.even(4)


def test_peers_whose_elements_differ_in_type_lay_out_different_spaces():
    fp16, bf16 = Sectioned((32,), dtype="float16"), Sectioned((32,), dtype="bfloat16")

    head_split = ByHead(head_bytes=8, total=2, dtype="float16")

    assert fp16.space(32, 2) != bf16.space(32, 2)
    assert head_split.space(8, 2) == ("head_major", 2, 8, "float16")
    assert RingSlots(4, dtype="bfloat16").space(32, 1) == ("ring", 8, "bfloat16")


def test_layouts_name_a_torch_element_type_without_its_module():
    class _TorchLikeDtype:
        def __str__(self):
            return "torch.bfloat16"

    assert RingSlots(4, dtype=_TorchLikeDtype()).dtype == "bfloat16"


def test_token_major_split_resolves_its_tokens_from_the_buffer():
    layout = ByHead(token_major=True).resolve(_site(1024, kv_heads=(4, 8), ranks=2, tokens=16))

    assert layout.tokens == 16
    layout.check(1024)


def test_sections_split_each_by_byte_over_the_ranks():
    layout = Sectioned((32, 16, 16))

    assert layout.pieces(64, 2, 1) == (Piece(32, 0, 32), Piece(80, 32, 16), Piece(112, 48, 16))
    assert layout.space(64, 2) == ("sectioned", (64, 32, 32))
    check_pieces(layout, 64, 2, 1)
    with pytest.raises(ValueError, match="do not cover"):
        layout.check(80)


def test_sections_resolve_to_the_layer_group_s_split():
    assert Sectioned((8,)).resolve(_site(8, split=("cp",))).split == ("cp",)


def test_replicated_buffer_is_the_whole_space_on_every_rank():
    layout = Replicated()

    assert layout.pieces(96, 4, 3) == (Piece(0, 0, 96),)
    assert layout.space(96, 4) == ("replicated", 96)
    assert layout.split == ()


def test_own_share_pairs_only_with_the_same_rank():
    layout = Own().resolve(_site(64, split=("tp", "cp")))

    assert layout.split == ("tp", "cp")
    assert layout.pieces(64, 4, 2) == (Piece(128, 0, 64),)
    assert layout.space(64, 4) == ("own", 64, 4)


def test_ring_keeps_the_latest_positions_of_its_page():
    # 19 slots in pages of 32 tokens: page 1 written up to position 50 holds 32..49.
    layout = RingSlots(19)

    assert layout.positional
    assert layout.positions(1, 50, 32) == range(32, 50)
    assert layout.positions(1, 100, 32) == range(45, 64)
    assert layout.index_of(40, 32) == 2
    assert layout.stride(19 * 6) == 6
    assert layout.pieces(19 * 6, 2, 1) == (Piece(0, 0, 6),)
    check_pieces(layout, 19 * 6, 2, 1)
    with pytest.raises(ValueError, match="whole number of 19 slots"):
        layout.check(100)


@pytest.mark.parametrize(
    "layout",
    [
        Replicated(),
        Own(("tp", "cp")),
        Sectioned((32, 16), dtype="float16"),
        ByHead(),
        ByHead(token_major=True, head_bytes=64, total=4, tokens=4, dtype="bfloat16"),
        RingSlots(19, dtype="bfloat16"),
    ],
)
def test_layouts_read_back_what_they_write(layout):
    assert role_layout_from_dict(layout.to_dict()) == layout


def test_unknown_layout_name_is_refused():
    with pytest.raises(ValueError, match="unknown role layout"):
        role_layout_from_dict({"name": "no_such_layout"})


def test_layout_fields_are_checked():
    with pytest.raises(ValueError, match="positive"):
        RingSlots(0)
    with pytest.raises(ValueError, match="positive"):
        ByHead(head_bytes=0)
    with pytest.raises(ValueError, match="one or more sections"):
        Sectioned(())
    with pytest.raises(ValueError, match="distinct dims"):
        Sectioned((8,), split=("tp", "tp"))
    with pytest.raises(ValueError, match="distinct dims"):
        Own(("pp",))
    with pytest.raises(ValueError, match="token_major must be a bool"):
        role_layout_from_dict({"name": "by_head", "token_major": "false"})


def test_layouts_name_element_types_by_name_or_torch_type():
    class _BindingsLikeDtype:
        def __str__(self):
            return "DataType.BF16"

    with pytest.raises(ValueError, match="name or a torch type"):
        RingSlots(4, dtype=_BindingsLikeDtype())


def test_positional_layout_must_say_where_its_positions_lie():
    with pytest.raises(TypeError, match="holds token positions"):

        @dataclass(frozen=True)
        class Positional(RoleLayoutV2):
            name: ClassVar[str] = "test_positional_without_index"
            positional: ClassVar[bool] = True


def test_pieces_that_leave_a_gap_are_refused():
    @dataclass(frozen=True)
    class Gapped(RoleLayoutV2):
        """Unnamed, so it stays out of the registry peers read layouts from."""

        def pieces(self, size, ranks, rank):
            # Byte size // 2 of the buffer lies in no piece.
            return (Piece(0, 0, size // 2), Piece(size, size // 2 + 1, size // 2))

        def space(self, size, ranks):
            return ("gapped", size)

    with pytest.raises(ValueError, match="back to back"):
        check_pieces(Gapped(), 8, 1, 0)


_RANK_COUNTS = (1, 2, 4, 8)


def _kv_site(heads, head_bytes, ranks, **kwargs):
    """An attention buffer of ``heads`` KV heads of ``head_bytes`` bytes over ``ranks`` ranks."""
    local = max(1, heads // ranks)
    return _site(local * head_bytes, kv_heads=(local, heads), ranks=ranks, **kwargs)


# Each case: ranks -> (the layout a manager's spec states, one rank's buffer's facts).
_TILING_CASES = {
    "kv_by_head": lambda n: (ByHead(), _kv_site(8, 64, n)),
    "kv_token_major": lambda n: (ByHead(token_major=True), _kv_site(8, 64, n, tokens=4)),
    "kv_heads_held_by_several_ranks": lambda n: (ByHead(), _kv_site(2, 64, n)),
    "mla_latent": lambda n: (ByHead(total=1), _site(576, kv_heads=(1, None), ranks=n)),
    "recurrent_state": lambda n: (ByHead(head_bytes=32), _site(8 // n * 32, state=True, ranks=n)),
    "sections": lambda n: (Sectioned((64 // n, 32 // n, 32 // n)), _site(128 // n, ranks=n)),
    "replicated": lambda n: (Replicated(), _site(96, ranks=n)),
    "ring": lambda n: (RingSlots(4), _site(96, ranks=n)),
}


@pytest.mark.parametrize("case", sorted(_TILING_CASES))
def test_ranks_of_every_topology_together_hold_one_space_whole(case):
    spaces = set()
    for ranks in _RANK_COUNTS:
        layout, site = _TILING_CASES[case](ranks)
        resolved = layout.resolve(site)
        resolved.check(site.size)
        held = collections.Counter()
        for rank in range(ranks):
            check_pieces(resolved, site.size, ranks, rank)
            for piece in resolved.pieces(site.size, ranks, rank):
                held.update(range(piece.at, piece.at + piece.size))
        # Every byte of the space from 0 up is held, each by as many ranks as every other.
        assert sorted(held) == list(range(len(held))), (case, ranks)
        assert len(set(held.values())) == 1, (case, ranks)
        spaces.add(resolved.space(site.size, ranks))
    # Peers of any two of these rank counts lay out the same space, so they correspond.
    assert len(spaces) == 1, (case, spaces)


def test_grid_place_counts_tensor_ranks_outer_to_context_ranks():
    topology = RankTopology(tp_size=2, tp_rank=1, cp_size=2, cp_rank=1)

    assert grid_place(topology, ("tp",)) == (2, 1)
    assert grid_place(topology, ("tp", "cp")) == (4, 3)
    assert grid_place(topology, ()) == (1, 0)
    with pytest.raises(ValueError, match="no grid dimension"):
        grid_place(topology, ("pp",))


def test_grid_place_under_attention_dp_holds_buffers_whole():
    topology = RankTopology(tp_size=4, tp_rank=3, dp_size=4, dp_rank=3, attention_dp=True)

    assert grid_place(topology, ("tp",)) == (1, 0)


def test_rank_topology_checks_its_ranks():
    with pytest.raises(ValueError, match="tp rank 2 of 2"):
        RankTopology(tp_size=2, tp_rank=2)
    with pytest.raises(ValueError, match="positive"):
        RankTopology(pp_size=0)


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_recurrent_state_splits_as_the_native_transfer_shards_it(tp_size, cp_size, attention_dp):
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_size):
            topology = RankTopology(
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                dp_size=tp_size if attention_dp else 1,
                dp_rank=tp_rank if attention_dp else 0,
                attention_dp=attention_dp,
            )
            rank_info = SimpleNamespace(
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                attention=SimpleNamespace(enable_attention_dp=attention_dp),
            )

            split = recurrent_state_split(topology)

            assert grid_place(topology, split) == MambaPolicy._mamba_tp(rank_info), topology


@pytest.mark.parametrize(
    "attention_dp, tp_place",
    [
        (None, (4, 3)),  # an attention-free stage: the TP ranks split it
        (False, (4, 3)),
        # Under attention data parallelism each TP rank holds its buffers whole.
        (True, (1, 0)),
    ],
)
def test_rank_info_gives_its_parallel_place(attention_dp, tp_place):
    attention = None
    if attention_dp is not None:
        attention = AttentionInfo(
            kv_heads_per_rank=2,
            tokens_per_block=32,
            dims_per_head=128,
            element_bytes=2,
            enable_attention_dp=attention_dp,
            is_mla=False,
        )
    rank_info = RankInfo(
        instance_name="gen_0",
        instance_rank=3,
        tp_size=4,
        tp_rank=3,
        pp_size=2,
        pp_rank=1,
        layer_num_per_pp=[16, 16],
        sender_endpoints=[],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        dp_size=4 if attention_dp else 1,
        dp_rank=3 if attention_dp else 0,
        cp_size=2,
        cp_rank=1,
        attention=attention,
    )

    topology = rank_info.topology

    assert (topology.tp_size, topology.tp_rank) == (4, 3)
    assert (topology.pp_size, topology.pp_rank, topology.cp_size, topology.cp_rank) == (2, 1, 2, 1)
    assert topology.attention_dp is bool(attention_dp)
    assert grid_place(topology, ("tp",)) == tp_place
    # `topology` is a property, so `to_bytes()` does not serialize it.
    assert "topology" not in msgpack.unpackb(rank_info.to_bytes(), strict_map_key=False)
