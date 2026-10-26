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
"""Transfer planning when a target and its drafter are laid out differently (CPU, synthetic ranks).

These tests exercise peer selection, sender election, completion
expectations, aux election and head slicing of the real ``PeerRegistrar`` on
page tables like a shared target/draft KV cache manager publishes, for meshes
larger than the GPUs at hand (including 32 ranks). They do not move bytes or
allocate caches; ``test_owner_transfer.py`` and
``test_kv_cache_owner_allocation.py`` do.

Expected ownership never comes from the code under test: rank coordinates
come from ``Mapping``, the heads a TP rank holds from the weight loader's
``duplicate_kv_weight`` sharding, and pages from the helix round-robin rule.
"""

import itertools
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    HNDHeadMismatchMapper,
    IntactMapper,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    OwnerLayout,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
    cp_ranks_sharing_pages,
    pair_owned_pages,
)
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 16
DIMS_PER_HEAD = 8
ELEMENT_BYTES = 2
HEAD_BYTES = TOKENS_PER_BLOCK * DIMS_PER_HEAD * ELEMENT_BYTES
NUM_PAGES = 67  # past two CP32 cycles, a partial cycle for every CP size below
TARGET, DRAFT = 0, 1  # owner indices, and layer groups, of the page tables below


def helix_mapping(world, rank, tp, cp):
    return Mapping(
        world_size=world,
        rank=rank,
        tp_size=tp,
        cp_size=cp,
        cp_config={"cp_type": CpType.HELIX} if cp > 1 else {},
    )


def loader_heads(num_kv_heads, tp_size, tp_rank):
    """Global heads a TP rank holds, as weight loading shards the K projection."""
    k_proj = torch.arange(num_kv_heads, dtype=torch.float32).reshape(num_kv_heads, 1)
    sharded = duplicate_kv_weight(k_proj, num_kv_heads, tp_size)
    return [int(h) for h in sharded.chunk(tp_size)[tp_rank].flatten()]


class Model:
    """One model of an instance: its (tp, cp) layout and model-wide KV-head count."""

    def __init__(self, tp, cp, heads):
        self.tp, self.cp, self.heads = tp, cp, heads

    def mapping(self, rank):
        return helix_mapping(self.tp * self.cp, rank, self.tp, self.cp)

    def held(self, rank):
        """(head, page) pairs a rank holds: Mapping coordinates, loader heads, helix pages."""
        mapping = self.mapping(rank)
        heads = loader_heads(self.heads, self.tp, mapping.tp_rank)
        pages = [p for p in range(NUM_PAGES) if p % self.cp == mapping.cp_rank]
        return heads, pages


def _view(heads_per_rank, layers, owner):
    entries = []
    buffer_bytes = heads_per_rank * HEAD_BYTES
    for i, layer in enumerate(layers):
        entries.append((layer, 2 * i * buffer_bytes, buffer_bytes))
        entries.append((layer, (2 * i + 1) * buffer_bytes, buffer_bytes))
    return PoolView(
        pool_idx=0,
        buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
        pool_role=frozenset({"key", "value"}),
        mapper_kind=MapperKind.HND,
        bytes_per_layer=2 * buffer_bytes,
        num_kv_heads=heads_per_rank,
        owner=owner,
    )


class Instance:
    """Every rank of a context or generation instance, as a shared target/draft manager publishes it."""

    def __init__(self, name, target, draft, *, mla=False, publish=True):
        self.name, self.target, self.draft, self.mla = name, target, draft, mla
        world = target.tp * target.cp
        assert draft.tp * draft.cp == world
        owners = [OwnerLayout(target.tp, target.cp), OwnerLayout(draft.tp, draft.cp)]
        self.infos = []
        for rank in range(world):
            mapping = target.mapping(rank)
            heads = [len(m.held(rank)[0]) for m in (target, draft)]
            groups = [
                AttentionLayerGroup(
                    pool_group_idx=g,
                    kv_head_num_per_rank=heads[g],
                    local_layers=[LocalLayer(2 * g + i, 2 * g + i) for i in range(2)],
                    pool_views=[_view(heads[g], [2 * g, 2 * g + 1], g if publish else None)],
                )
                for g in (TARGET, DRAFT)
            ]
            page_table = KVCachePageTable(
                tokens_per_block=TOKENS_PER_BLOCK,
                layer_groups=groups,
                pool_groups=[
                    PhysicalPoolGroup(
                        pools=[
                            PhysicalPool(
                                base_address=0x100000 * (1 + g),
                                slot_bytes=4 * heads[g] * HEAD_BYTES,
                                num_slots=64,
                            )
                        ]
                    )
                    for g in (TARGET, DRAFT)
                ],
                owners=owners if publish else [],
            )
            self.infos.append(
                RankInfo(
                    instance_name=name,
                    instance_rank=rank,
                    tp_size=mapping.tp_size,
                    tp_rank=mapping.tp_rank,
                    pp_size=1,
                    pp_rank=0,
                    cp_size=mapping.cp_size,
                    cp_rank=mapping.cp_rank,
                    layer_num_per_pp=[4],
                    sender_endpoints=[],
                    self_endpoint="",
                    transfer_engine_info=b"",
                    attention=AttentionInfo(
                        kv_heads_per_rank=heads[TARGET],
                        tokens_per_block=TOKENS_PER_BLOCK,
                        dims_per_head=DIMS_PER_HEAD,
                        element_bytes=ELEMENT_BYTES,
                        enable_attention_dp=False,
                        is_mla=mla,
                    ),
                    page_table=page_table,
                    num_draft_layers=2,
                )
            )
        self.registrars = [
            PeerRegistrar(info, KVRegionExtractorV1(info.page_table)) for info in self.infos
        ]

    def model(self, owner):
        return (self.target, self.draft)[owner]


def register_all(ctx, gen):
    for sender, info in itertools.product(ctx.registrars, gen.infos):
        sender.register(info.instance_name, info.instance_rank, info)
    for receiver, info in itertools.product(gen.registrars, ctx.infos):
        receiver.register(info.instance_name, info.instance_rank, info)


def paired_pages(sender, receiver_info, owner, pages_c, pages_g):
    """Logical pages the sender's and receiver's tables pair, as Sender._build_kv_write_meta pairs them.

    Each table lists its rank's true pages in order (the value is the page),
    so a correct pairing matches equal pages.
    """
    src, dst = np.array(pages_c, dtype=np.int64), np.array(pages_g, dtype=np.int64)
    view = sender.view_rank_infos(receiver_info).get(
        (owner, 0), (sender.self_rank_info, receiver_info)
    )
    if view[0].cp_size > 1 or view[1].cp_size > 1:
        return pair_owned_pages(src, dst, *[(ri.cp_size, ri.cp_rank) for ri in view])
    assert src.size == dst.size, (src, dst)
    return src, dst


def check_plan(ctx, gen):
    """Every receiver gets each (owner, head, page) it holds exactly once, with matching bookkeeping."""
    register_all(ctx, gen)
    for g, (receiver, g_info) in enumerate(zip(gen.registrars, gen.infos)):
        contacted = receiver.get_peer_overlap(ctx.infos[0], 0).ranks
        aux_senders = []
        for owner in (TARGET, DRAFT):
            heads_g, pages_g = gen.model(owner).held(g)
            needed = set(itertools.product(heads_g, pages_g))
            delivered = []
            for c in contacted:
                sender = ctx.registrars[c]
                sender_overlap = sender.get_peer_overlap(g_info, 0)
                # Completion: the receiver waits for one write from each sender it
                # contacts, and each such sender writes to it.
                assert g in sender_overlap.ranks, (g, c)
                if not sender.should_send_pool(sender_overlap, g_info, owner, 0):
                    continue
                heads_c, pages_c = ctx.model(owner).held(c)
                shared = set(itertools.product(heads_c, pages_c)) & needed
                delivered.extend(shared)
                src_pages, dst_pages = paired_pages(sender, g_info, owner, pages_c, pages_g)
                assert src_pages.tolist() == dst_pages.tolist(), (c, g, owner)
                assert set(src_pages.tolist()) == set(pages_c) & set(pages_g), (c, g, owner)
                shared_heads = sorted(set(heads_c) & set(heads_g))
                if not shared_heads:
                    continue
                mapper = sender.get_kv_map(g_info, (owner, 0), (owner, 0))
                if isinstance(mapper, IntactMapper):
                    assert ctx.mla or heads_c == heads_g, (c, g, owner)
                else:
                    assert isinstance(mapper, HNDHeadMismatchMapper)
                    first = shared_heads[0]
                    # Slice: the shared heads, from the sender's slot to the receiver's.
                    assert mapper._src_head_off == heads_c.index(first) * HEAD_BYTES, (c, g)
                    assert mapper._dst_head_off == heads_g.index(first) * HEAD_BYTES, (c, g)
                    assert mapper._bytes_cont_heads == len(shared_heads) * HEAD_BYTES
            assert sorted(delivered) == sorted(needed), (g, owner)
        for c in contacted:
            assert g in ctx.registrars[c].get_peer_overlap(g_info, 0).ranks
            if ctx.registrars[c].should_send_aux(g_info):
                aux_senders.append(c)
        assert len(aux_senders) == 1, (g, aux_senders)
    # A sender expects only receivers that contact it.
    for c, sender in enumerate(ctx.registrars):
        for g in sender.get_peer_overlap(gen.infos[0], 0).ranks:
            assert c in gen.registrars[g].get_peer_overlap(ctx.infos[0], 0).ranks


M = Model
# Reviewer layouts at full size: (A) target TP1 x CP32 with a TP16 x CP2
# drafter; (B) the CP-replicated drafter of a helix target, its CP ranks
# repurposed as TP; (C) an MLA target at TP2 x CP8 with an MLA drafter at
# TP1 x CP16. Each is planned in both directions and against a peer whose
# drafter splits pages over a different CP size.
A_GEN = (M(1, 32, 8), M(16, 2, 8))
B_GEN = (M(1, 32, 8), M(32, 1, 8))
C_GEN = (M(2, 8, 1), M(1, 16, 1))
LAYOUTS = {
    # generation: target TP1 x CP32, draft TP16 x CP2 (32 ranks)
    "target_cp32_draft_tp16cp2": (M(4, 1, 8), M(4, 1, 16), M(1, 32, 8), M(16, 2, 16), False),
    "a_tp4_to_a": (M(4, 1, 8), M(4, 1, 8), *A_GEN, False),
    "a_a_to_tp4": (*A_GEN, M(4, 1, 8), M(4, 1, 8), False),
    "a_draft_cp4_to_a": (M(4, 1, 8), M(1, 4, 8), *A_GEN, False),
    "a_a_to_draft_cp4": (*A_GEN, M(4, 1, 8), M(1, 4, 8), False),
    "b_tp4_to_b": (M(4, 1, 8), M(4, 1, 8), *B_GEN, False),
    "b_b_to_tp4": (*B_GEN, M(4, 1, 8), M(4, 1, 8), False),
    "b_draft_cp4_to_b": (M(4, 1, 8), M(1, 4, 8), *B_GEN, False),
    # generation: target TP2 x CP8, MLA-style single-latent draft at CP16
    "mla_target_tp2cp8_draft_cp16": (M(4, 1, 1), M(4, 1, 1), M(2, 8, 1), M(1, 16, 1), True),
    "c_c_to_tp4": (*C_GEN, M(4, 1, 1), M(4, 1, 1), True),
    "c_draft_cp4_to_c": (M(4, 1, 1), M(1, 4, 1), *C_GEN, True),
    "c_c_to_draft_cp2": (*C_GEN, M(2, 1, 1), M(1, 2, 1), True),
    "target_tp1cp8_draft_tp4cp2": (M(4, 1, 8), M(4, 1, 16), M(1, 8, 8), M(4, 2, 16), False),
    "target_tp4cp2_draft_tp1cp8": (M(4, 1, 8), M(4, 1, 16), M(4, 2, 8), M(1, 8, 16), False),
    "tp_target_draft_tp1cp4": (M(4, 1, 8), M(4, 1, 8), M(4, 1, 8), M(1, 4, 8), False),
    # CP-replicated drafter: the CP ranks of a TP1 x CP4 target repurposed as
    # TP4, 2 heads replicated twice, every page on every rank.
    "cp_replicated_draft": (M(4, 1, 8), M(4, 1, 2), M(1, 4, 8), M(4, 1, 2), False),
    "draft_cp_on_both_sides": (M(4, 1, 8), M(2, 2, 8), M(4, 1, 8), M(1, 4, 8), False),
    "head_replication_on_context": (M(8, 1, 8), M(8, 1, 2), M(2, 1, 8), M(2, 1, 2), False),
    # Matched layouts (regression): no model layouts are published.
    "matched_tp": (M(2, 1, 8), M(2, 1, 4), M(4, 1, 8), M(4, 1, 4), False),
    "matched_helix": (M(4, 1, 8), M(4, 1, 8), M(1, 4, 8), M(1, 4, 8), False),
}


@pytest.mark.parametrize("name", LAYOUTS)
def test_every_receiver_gets_its_heads_and_pages_once(name):
    ctx_target, ctx_draft, gen_target, gen_draft, mla = LAYOUTS[name]
    ctx = Instance("ctx", ctx_target, ctx_draft, mla=mla)
    gen = Instance("gen", gen_target, gen_draft, mla=mla)
    check_plan(ctx, gen)


def test_cp_replicated_draft_is_received_by_every_replica():
    # Each of the 4 generation ranks holds all draft pages of its replicated head.
    ctx_target, ctx_draft, gen_target, gen_draft, _ = LAYOUTS["cp_replicated_draft"]
    gen = Instance("gen", gen_target, gen_draft)
    holders = {}
    for g in range(4):
        heads, pages = gen.draft.held(g)
        assert pages == list(range(NUM_PAGES))
        for head in heads:
            holders.setdefault(head, []).append(g)
    assert holders == {0: [0, 1], 1: [2, 3]}
    check_plan(Instance("ctx", ctx_target, ctx_draft), gen)


@pytest.mark.parametrize(
    "corrupt",
    [
        # The draft published with the target's layout: wrong pages and heads.
        lambda pt: setattr(pt.layer_groups[DRAFT].pool_views[0], "owner", TARGET),
        # The draft's layout misreported on the same ranks: wrong head placement.
        lambda pt: pt.owners.__setitem__(
            DRAFT, replace(pt.owners[DRAFT], tp_size=pt.owners[DRAFT].tp_size * 2, cp_size=1)
        ),
        # Rank-wide routing: the draft routed by the target's layout.
        lambda pt: pt.owners.__setitem__(DRAFT, pt.owners[TARGET]),
    ],
    ids=["wrong_owner", "wrong_head_placement", "rank_wide_routing"],
)
@pytest.mark.parametrize(
    "layout", ["target_tp1cp8_draft_tp4cp2", "a_tp4_to_a", "b_tp4_to_b", "c_draft_cp4_to_c"]
)
def test_wrong_ownership_is_detected(corrupt, layout):
    ctx_target, ctx_draft, gen_target, gen_draft, mla = LAYOUTS[layout]
    ctx = Instance("ctx", ctx_target, ctx_draft, mla=mla)
    gen = Instance("gen", gen_target, gen_draft, mla=mla)
    for info in gen.infos:
        corrupt(info.page_table)
    with pytest.raises((AssertionError, ValueError)):
        check_plan(ctx, gen)


def test_a_peer_without_model_layouts_is_rejected():
    ctx_target, ctx_draft, gen_target, gen_draft, _ = LAYOUTS["target_tp1cp8_draft_tp4cp2"]
    ctx = Instance("ctx", ctx_target, ctx_draft, publish=False)
    gen = Instance("gen", gen_target, gen_draft)
    with pytest.raises(ValueError, match="publishes no model layouts"):
        ctx.registrars[0].register("gen", 0, gen.infos[0])


def test_split_pages_turn_off_equal_writer_shares():
    ctx_target, ctx_draft, gen_target, gen_draft, _ = LAYOUTS["tp_target_draft_tp1cp4"]
    ctx, gen = Instance("ctx", ctx_target, ctx_draft), Instance("gen", gen_target, gen_draft)
    register_all(ctx, gen)
    assert gen.registrars[0].splits_pages(ctx.infos[0])
    assert not gen.registrars[0].get_peer_overlap(ctx.infos[0], 0).uniform_writers
    ctx_target, ctx_draft, gen_target, gen_draft, _ = LAYOUTS["matched_tp"]
    ctx, gen = Instance("ctx", ctx_target, ctx_draft), Instance("gen", gen_target, gen_draft)
    register_all(ctx, gen)
    assert not gen.registrars[0].splits_pages(ctx.infos[0])


def test_reregistration_drops_routing_of_the_old_layout():
    ctx = Instance("ctx", M(4, 1, 8), M(4, 1, 8))
    gen = Instance("gen", M(4, 1, 8), M(4, 1, 8))
    register_all(ctx, gen)
    sender = ctx.registrars[0]
    sender.get_kv_map(gen.infos[0], (DRAFT, 0), (DRAFT, 0))
    # The generation instance restarts under the same name with its drafter at TP1 x CP4.
    restarted = Instance("gen", M(4, 1, 8), M(1, 4, 8))
    sender.register("gen", 0, restarted.infos[0])
    assert not sender._kv_map_cache
    assert sender.view_rank_infos(restarted.infos[0])[(DRAFT, 0)][1].cp_size == 4


def test_cp_page_intersection_matches_brute_force():
    for self_cp, peer_cp in itertools.product((1, 2, 3, 4, 6, 8, 16, 32), repeat=2):
        for rank in range(self_cp):
            mine = {p for p in range(96) if p % self_cp == rank}
            expected = [
                r for r in range(peer_cp) if mine & {p for p in range(96) if p % peer_cp == r}
            ]
            assert cp_ranks_sharing_pages(self_cp, rank, peer_cp) == expected


def test_page_pairing_matches_brute_force():
    rng = random.Random(0)
    for _ in range(300):
        src_cp, dst_cp = rng.choice((1, 2, 4, 8, 16, 32)), rng.choice((1, 2, 4, 8, 16, 32))
        src_rank, dst_rank = rng.randrange(src_cp), rng.randrange(dst_cp)
        num_pages = rng.randrange(0, 100)
        src_pages = [p for p in range(num_pages) if p % src_cp == src_rank]
        dst_pages = [p for p in range(num_pages) if p % dst_cp == dst_rank]
        src = np.array([1000 + p for p in src_pages], dtype=np.int64)
        dst = np.array([2000 + p for p in dst_pages], dtype=np.int64)
        got_src, got_dst = pair_owned_pages(src, dst, (src_cp, src_rank), (dst_cp, dst_rank))
        shared = sorted(set(src_pages) & set(dst_pages))
        assert (got_src - 1000).tolist() == shared
        assert (got_dst - 2000).tolist() == shared
    with pytest.raises(ValueError, match="same prompt"):
        pair_owned_pages(np.arange(8), np.arange(5), (1, 0), (2, 1))
