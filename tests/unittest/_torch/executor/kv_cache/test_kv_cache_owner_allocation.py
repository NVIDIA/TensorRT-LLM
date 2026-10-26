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
"""Exact pages per owner when a target and its drafter split pages differently, in one manager.

Every rank builds one KVCacheManagerV2 holding target and draft layers with
each model's own ``Mapping``. A model with context parallelism holds global
page ``p`` on its CP rank ``p % cp_size``; a model without CP holds every
page. Each rank must hold exactly its pages of each model, nothing shaped by
the other model's layout and nothing reserved for another rank's share. All
expectations come from the two mappings and the layer geometry.

Pages a request holds are measured as slots taken from each pool group. A
model with CP is additionally charged a rank-independent admission bound per
request (the most any of its CP ranks holds); the charge owns no storage.
"""

import os
import socket
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 as kvcm2
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    OwnerPageLayout,
    Role,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.quantization.mode import QuantMode
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId

P = 32  # tokens per page, both models
TARGET_LAYERS, DRAFT_LAYERS = 3, 2
HEAD_DIM, MLA_HEAD_DIM, BF16 = 128, 576, 2
MAX_SEQ_LEN = 1024
LONG_MAX_SEQ_LEN = 8192  # layouts past 8 ranks: several CP32 cycles
TARGET, DRAFT = KVCacheLayerOwner.TARGET, KVCacheLayerOwner.DRAFT


class Layout(NamedTuple):
    world: int
    target_tp: int
    target_cp: int
    draft_tp: int
    draft_cp: int
    target_heads: int = 8
    draft_heads: int = 8
    mla: bool = False  # both models keep one compressed latent per token (K only)

    @property
    def head_dim(self):
        return MLA_HEAD_DIM if self.mla else HEAD_DIM

    @property
    def kv_factor(self):
        return 1 if self.mla else 2

    @property
    def max_seq_len(self):
        return LONG_MAX_SEQ_LEN if self.world > 8 else MAX_SEQ_LEN

    @property
    def cp_cycle(self):
        """Pages after which every CP rank of both models holds the same count again."""
        return max(8, self.target_cp, self.draft_cp)

    def sizes(self, owner):
        return (
            (self.target_tp, self.target_cp) if owner == TARGET else (self.draft_tp, self.draft_cp)
        )

    def heads(self, owner):
        return self.target_heads if owner == TARGET else self.draft_heads


TP_TARGET_DRAFT_CP = Layout(4, 4, 1, 1, 4)
HELIX_8 = Layout(8, 1, 8, 4, 2)  # target TP1xCP8, drafter TP4xCP2
HELIX_8_FINE_DRAFT = Layout(8, 4, 2, 1, 8)  # target TP4xCP2, drafter TP1xCP8
HELIX_4 = Layout(4, 1, 4, 2, 2)
HELIX_4_FINE_DRAFT = Layout(4, 2, 2, 1, 4)
# The drafter a helix TP1xCP4 target builds: the CP ranks repurposed as TP4,
# every page on every rank; with 2 heads, each head on two ranks.
CP_REPLICATED = Layout(4, 1, 4, 4, 1)
CP_REPLICATED_HEADS = Layout(4, 1, 4, 4, 1, draft_heads=2)
STRIDED = (
    TP_TARGET_DRAFT_CP,
    HELIX_8,
    HELIX_8_FINE_DRAFT,
    HELIX_4,
    HELIX_4_FINE_DRAFT,
    CP_REPLICATED,
    CP_REPLICATED_HEADS,
)
MATCHED_TP = Layout(4, 4, 1, 4, 1)
MATCHED_HELIX = Layout(4, 1, 4, 1, 4)
# Reviewer layouts at full size, one synthetic rank at a time: (A) target
# TP1 x CP32 with a TP16 x CP2 drafter (8 heads: each draft head on two ranks);
# (B) the CP-replicated drafter of a TP1 x CP32 target, repurposed as TP32;
# (C) an MLA target at TP2 x CP8 with an MLA drafter at TP1 x CP16.
REVIEWER_A = Layout(32, 1, 32, 16, 2)
REVIEWER_B = Layout(32, 1, 32, 32, 1)
REVIEWER_C = Layout(16, 2, 8, 1, 16, target_heads=1, draft_heads=1, mla=True)
FULL_SIZE = (REVIEWER_A, REVIEWER_B, REVIEWER_C)


def div_up(a, b):
    return -(-a // b)


def helix_mapping(world, rank, tp, cp):
    return Mapping(
        world_size=world,
        rank=rank,
        tp_size=tp,
        cp_size=cp,
        cp_config={"cp_type": CpType.HELIX} if cp > 1 else {},
    )


def owner_mapping(layout, rank, owner):
    return helix_mapping(layout.world, rank, *layout.sizes(owner))


def owned_pages(layout, rank, owner, num_pages):
    """Global pages a rank holds of an owner: page p on the owner's CP rank p % cp_size."""
    mapping = owner_mapping(layout, rank, owner)
    return [p for p in range(num_pages) if p % mapping.cp_size == mapping.cp_rank]


def heads_per_rank(layout, owner):
    tp, _ = layout.sizes(owner)
    return max(1, layout.heads(owner) // tp)


def layer_specs(layout):
    def spec(layer_id, owner):
        return KVCacheLayerSpec(
            global_layer_id=layer_id,
            owner=owner,
            dtype=DataType.BF16,
            num_kv_heads=heads_per_rank(layout, owner),
            head_dim=layout.head_dim,
            buffer_roles=(Role.KEY,) if layout.mla else (Role.KEY, Role.VALUE),
        )

    return [spec(i, TARGET) for i in range(TARGET_LAYERS)] + [
        spec(TARGET_LAYERS + i, DRAFT) for i in range(DRAFT_LAYERS)
    ]


def kv_cache_config(gpu_bytes, **overrides):
    config = KvCacheConfig(
        max_gpu_total_bytes=gpu_bytes,
        host_cache_size=0,
        enable_block_reuse=False,
        enable_swa_scratch_reuse=False,
        max_attention_window=[MAX_SEQ_LEN],
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def make_manager(layout, rank, gpu_bytes=64 << 20, **config_overrides):
    """One rank's shared manager, holding both models with their own mappings."""
    target = owner_mapping(layout, rank, TARGET)
    config = kv_cache_config(gpu_bytes, **config_overrides)
    config.max_attention_window = [layout.max_seq_len]
    return KVCacheManagerV2(
        config,
        CacheType.SELFKONLY if layout.mla else CacheType.SELF,
        num_layers=TARGET_LAYERS,
        num_kv_heads=heads_per_rank(layout, TARGET),
        head_dim=layout.head_dim,
        tokens_per_block=P,
        max_seq_len=layout.max_seq_len,
        max_batch_size=64,
        mapping=target,
        dtype=DataType.BF16,
        layer_specs=layer_specs(layout),
        owner_mappings={TARGET: target, DRAFT: owner_mapping(layout, rank, DRAFT)},
    )


def layer_ids(manager, owner):
    return [spec.global_layer_id for spec in manager.layer_specs if spec.owner == owner]


def layer_group(manager, layer):
    return manager.impl.get_layer_group_id(LayerId(manager.layer_offsets[layer]))


def pool_group(manager, owner):
    of_group = manager.impl.get_life_cycle_pool_group_indices(0)
    groups = {of_group[layer_group(manager, layer)] for layer in layer_ids(manager, owner)}
    assert len(groups) == 1, groups
    return groups.pop()


def page_bytes(layout, owner):
    num_layers = TARGET_LAYERS if owner == TARGET else DRAFT_LAYERS
    return (
        num_layers * layout.kv_factor * heads_per_rank(layout, owner) * layout.head_dim * P * BF16
    )


class Baseline:
    def __init__(self, manager):
        stats = manager.impl.get_storage_statistics(0)
        self.free = [s.free for s in stats]
        self.total = [s.total for s in stats]


def check_allocation(manager, layout, rank, request_id, num_tokens, base):
    """Assert one request's exact pages, identities, storage and admission charge per owner."""
    num_pages = div_up(num_tokens, P)
    stats = manager.impl.get_storage_statistics(0)
    assert [s.total for s in stats] == base.total
    taken = [before - s.free for before, s in zip(base.free, stats)]
    admitted = manager.impl.get_strided_admitted_slots()
    cache = manager.kv_cache_map[request_id]
    groups = {owner: pool_group(manager, owner) for owner in (TARGET, DRAFT)}
    for owner in (TARGET, DRAFT):
        owned = owned_pages(layout, rank, owner, num_pages)
        _, cp = layout.sizes(owner)
        for layer in layer_ids(manager, owner):
            group = layer_group(manager, layer)
            # Per block ordinal (block b is global page b): pages at exactly the owned pages.
            by_ordinal = list(cache.get_aggregated_page_indices(group))
            assert [b for b, s in enumerate(by_ordinal) if s >= 0] == owned, (owner, num_tokens)
            # The block table the owner's attention reads lists them in order, then nothing.
            table = np.frombuffer(
                cache.get_base_page_indices(group), dtype=np.int32, count=num_pages
            )
            assert table[: len(owned)].tolist() == [by_ordinal[b] for b in owned]
            assert (table[len(owned) :] < 0).all()
        assert taken[groups[owner]] == len(owned), (owner, num_tokens, rank)
        assert sum(stats[groups[owner]].slot_sizes) == page_bytes(layout, owner)
        assert admitted[groups[owner]] == (div_up(num_pages, cp) if cp > 1 else 0)
    # Nothing else is taken on the rank's behalf.
    assert sum(taken) == sum(
        len(owned_pages(layout, rank, owner, num_pages)) for owner in (TARGET, DRAFT)
    )


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()


def open_cache(manager, request_id):
    cache = manager._create_kv_cache(request_id, None, None)
    assert cache.resume(torch.cuda.current_stream().cuda_stream)
    return cache


def try_admit(manager, request_id, num_tokens):
    cache = manager._create_kv_cache(request_id, None, None)
    return cache.resume(torch.cuda.current_stream().cuda_stream) and cache.resize(num_tokens)


def close_all(manager):
    for cache in manager.kv_cache_map.values():
        cache.close()
    manager.kv_cache_map.clear()
    manager.shutdown()


@pytest.mark.parametrize("layout", STRIDED + FULL_SIZE, ids=str)
def test_one_native_manager_holds_both_models_as_mapped(layout):
    for rank in range(layout.world):
        manager = make_manager(layout, rank)
        try:
            # Every target and draft layer lives in the one native manager.
            assert len(set(manager.impl.layer_ids)) == TARGET_LAYERS + DRAFT_LAYERS
            assert len({pool_group(manager, owner) for owner in (TARGET, DRAFT)}) == 2
            for owner in (TARGET, DRAFT):
                tp, cp = layout.sizes(owner)
                mapping = manager.owner_mappings[owner]
                assert (mapping.tp_size, mapping.cp_size) == (tp, cp)
                assert (mapping.tp_rank, mapping.cp_rank) == (
                    owner_mapping(layout, rank, owner).tp_rank,
                    owner_mapping(layout, rank, owner).cp_rank,
                )
                for layer in layer_ids(manager, owner):
                    assert manager.get_layer_cache_spec(layer).num_kv_heads == heads_per_rank(
                        layout, owner
                    )
        finally:
            manager.shutdown()


LENGTHS = (1, P - 1, P, 3 * P, 8 * P, 8 * P + 1, 9 * P, 10 * P, 12 * P, 17 * P + 5)


def lengths(layout):
    """Prompts ending inside a page and around the first and second CP cycles."""
    cycle = layout.cp_cycle * P
    around = (cycle - P + 1, cycle - 1, cycle, cycle + 1, 2 * cycle + 3 * P + 5)
    return tuple(sorted(set(LENGTHS) | set(around)))


@pytest.mark.parametrize("layout", STRIDED + FULL_SIZE, ids=str)
def test_each_rank_holds_exactly_its_pages(layout):
    for rank in range(layout.world):
        manager = make_manager(layout, rank)
        try:
            base = Baseline(manager)
            for request_id, num_tokens in enumerate(lengths(layout)):
                assert open_cache(manager, request_id).resize(num_tokens)
                check_allocation(manager, layout, rank, request_id, num_tokens, base)
                manager.kv_cache_map.pop(request_id).close()
            assert [s.free for s in manager.impl.get_storage_statistics(0)] == base.free
        finally:
            close_all(manager)


def taken_pages(manager, base, owner):
    pg = pool_group(manager, owner)
    return base.free[pg] - manager.impl.get_storage_statistics(0)[pg].free


@pytest.mark.parametrize(
    ("layout", "num_pages", "expected"),
    [
        # 9 pages, drafter TP1xCP4 under a TP4 target: 3/2/2/2 drafter pages.
        (TP_TARGET_DRAFT_CP, 9, {r: (9, (3, 2, 2, 2)[r]) for r in range(4)}),
        # Target TP1xCP8: CP rank 0 holds pages 0 and 8; drafter TP4xCP2: even ranks 5, odd 4.
        (HELIX_8, 9, {r: (2 if r == 0 else 1, 5 if r % 2 == 0 else 4) for r in range(8)}),
        # The CP-replicated drafter holds all 9 pages on every rank.
        (CP_REPLICATED, 9, {r: (3 if r == 0 else 2, 9) for r in range(4)}),
        # 67 pages (two CP32 cycles and 3 pages). Target TP1xCP32: ranks 0-2
        # hold 3 pages, the rest 2; drafter TP16xCP2: even ranks 34, odd 33.
        (REVIEWER_A, 67, {r: (3 if r < 3 else 2, 34 if r % 2 == 0 else 33) for r in range(32)}),
        # The CP-replicated drafter of that target holds all 67 pages everywhere.
        (REVIEWER_B, 67, {r: (3 if r < 3 else 2, 67) for r in range(32)}),
        # MLA target TP2xCP8: CP ranks 0-2 hold 9 pages, the rest 8; MLA
        # drafter TP1xCP16: ranks 0-2 hold 5 pages, the rest 4.
        (REVIEWER_C, 67, {r: (9 if r % 8 < 3 else 8, 5 if r < 3 else 4) for r in range(16)}),
    ],
    ids=str,
)
def test_literal_page_counts(layout, num_pages, expected):
    for rank in range(layout.world):
        manager = make_manager(layout, rank)
        try:
            base = Baseline(manager)
            open_cache(manager, 0).resize(num_pages * P)
            got = (taken_pages(manager, base, TARGET), taken_pages(manager, base, DRAFT))
            assert got == expected[rank], (rank, got)
        finally:
            close_all(manager)


@pytest.mark.parametrize("layout", STRIDED + FULL_SIZE, ids=str)
def test_growth_rollback_suspend_and_cleanup(layout):
    cycle = layout.cp_cycle * P
    for rank in range(layout.world):
        manager = make_manager(layout, rank)
        try:
            base = Baseline(manager)
            cache = open_cache(manager, 0)
            assert cache.resize(cycle)
            before = [s.free for s in manager.impl.get_storage_statistics(0)]
            # Growth across page and CP-cycle boundaries, then rollback.
            grown = (cycle + 1, cycle + P, cycle + 3 * P + 5, 2 * cycle + 1, 2 * cycle + P)
            for num_tokens in grown:
                assert cache.resize(num_tokens)
                check_allocation(manager, layout, rank, 0, num_tokens, base)
            for num_tokens in (*reversed(grown[:-1]), cycle):
                assert cache.resize(num_tokens)
                check_allocation(manager, layout, rank, 0, num_tokens, base)
            assert [s.free for s in manager.impl.get_storage_statistics(0)] == before
            cache.suspend()
            assert cache.resume(torch.cuda.current_stream().cuda_stream)
            check_allocation(manager, layout, rank, 0, cycle, base)
            manager.kv_cache_map.pop(0).close()
            assert [s.free for s in manager.impl.get_storage_statistics(0)] == base.free
            assert manager.impl.get_strided_admitted_slots() == [0] * len(base.free)
        finally:
            close_all(manager)


@pytest.mark.parametrize("layout", STRIDED + FULL_SIZE, ids=str)
def test_refused_growth_is_undone_on_every_rank(layout):
    managers = [make_manager(layout, rank, gpu_bytes=8 << 20) for rank in range(layout.world)]
    try:
        for m in managers:
            assert open_cache(m, 0).resize(4 * P)
        request_id = 1
        while True:
            decisions = [try_admit(m, request_id, 1) for m in managers]
            assert len(set(decisions)) == 1, decisions
            if not decisions[0]:
                break
            request_id += 1
        assert request_id > 1
        assert len({tuple(m.impl.get_strided_admitted_slots()) for m in managers}) == 1

        def snapshot(m):
            cache = m.kv_cache_map[0]
            return (
                cache.capacity,
                [
                    list(cache.get_aggregated_page_indices(layer_group(m, layer)))
                    for owner in (TARGET, DRAFT)
                    for layer in layer_ids(m, owner)
                ],
                [s.free for s in m.impl.get_storage_statistics(0)],
                m.impl.get_strided_admitted_slots(),
            )

        before = [snapshot(m) for m in managers]
        grow_to = 4 * P * max(layout.target_cp, layout.draft_cp) + 1
        assert {m.kv_cache_map[0].resize(grow_to) for m in managers} == {False}
        assert [snapshot(m) for m in managers] == before
        for m in managers:
            for request in range(1, request_id):
                m.kv_cache_map.pop(request).close()
        assert {m.kv_cache_map[0].resize(grow_to) for m in managers} == {True}
    finally:
        for m in managers:
            close_all(m)


@pytest.mark.parametrize(
    "layout", (HELIX_8, TP_TARGET_DRAFT_CP, CP_REPLICATED, *FULL_SIZE), ids=str
)
@pytest.mark.parametrize("fault", ["ledger_of_the_target", "every_block", "every_rank_phase_0"])
def test_checks_reject_padded_or_misplaced_pages(monkeypatch, layout, fault):
    real = kvcm2.owner_page_layout

    def faulty(specs, owner_mappings, mapping):
        placed = real(specs, owner_mappings, mapping)
        if fault == "ledger_of_the_target":
            # Blocks span the target's CP: the drafter is rounded to its layout.
            helix_cp = mapping.cp_size if mapping.has_cp_helix() else 1
            return OwnerPageLayout(helix_cp, (1,) * len(placed.strides), (0,) * len(placed.strides))
        if fault == "every_block":
            return OwnerPageLayout(1, (1,) * len(placed.strides), (0,) * len(placed.strides))
        return OwnerPageLayout(placed.ledger_cp, placed.strides, (0,) * len(placed.strides))

    monkeypatch.setattr(kvcm2, "owner_page_layout", faulty)
    failures = 0
    for rank in range(layout.world):
        manager = make_manager(layout, rank)
        try:
            base = Baseline(manager)
            cache = open_cache(manager, 0)
            for num_tokens in (8 * P + 1, 9 * P):
                assert cache.resize(num_tokens)
                try:
                    check_allocation(manager, layout, rank, 0, num_tokens, base)
                except AssertionError:
                    failures += 1
        finally:
            close_all(manager)
    assert failures > 0


def test_block_reuse_is_rejected_as_for_every_helix_target():
    with pytest.raises(ValueError, match="block reuse with helix"):
        make_manager(HELIX_4, 0, enable_block_reuse=True)


@pytest.mark.parametrize("host_cache_size", [None, 32 << 20], ids=["automatic", "explicit"])
@pytest.mark.parametrize("layout", (HELIX_4, CP_REPLICATED, REVIEWER_A, REVIEWER_C), ids=str)
def test_host_tier_is_kept_and_suspended_requests_keep_their_pages(layout, host_cache_size):
    for rank in (0, layout.world - 1):
        manager = make_manager(layout, rank, host_cache_size=host_cache_size)
        try:
            assert manager.impl.get_storage_statistics(kvcm2.CacheLevel(1)), "no host tier"
            base = Baseline(manager)
            cache = open_cache(manager, 0)
            assert cache.resize(9 * P + 5)
            check_allocation(manager, layout, rank, 0, 9 * P + 5, base)
            cache.suspend()
            assert cache.resume(torch.cuda.current_stream().cuda_stream)
            check_allocation(manager, layout, rank, 0, 9 * P + 5, base)
        finally:
            close_all(manager)


def test_matched_layouts_keep_one_block_per_page_layout():
    # Regression: owners with one layout hold a page per block (TP), or per
    # helix ledger block of CP pages (CP), exactly as before.
    for layout, ledger in ((MATCHED_TP, 1), (MATCHED_HELIX, 4)):
        for rank in range(layout.world):
            manager = make_manager(layout, rank)
            try:
                assert manager._page_layout.ledger_cp == ledger
                assert not manager._page_layout.is_strided
                assert manager.impl.get_strided_admitted_slots() == [0] * len(
                    manager.impl.get_storage_statistics(0)
                )
            finally:
                manager.shutdown()


# ---- The CP-replicated drafter as the KV cache creator builds it -------------------------------


def model_config(mapping, num_layers, kv_heads):
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=32,
            num_key_value_heads=kv_heads,
            head_dim=HEAD_DIM,
            hidden_size=32 * HEAD_DIM,
            torch_dtype=torch.bfloat16,
        ),
        quant_config=SimpleNamespace(quant_mode=QuantMode(0)),
        mapping=mapping,
    )


def make_creator(target, draft_heads):
    """A KV cache creator for a helix target whose DFlash drafter is built on the repurposed mapping."""
    spec_config = DFlashDecodingConfig(
        max_draft_len=4, speculative_model="/nonexistent/dflash", attention_backend="TRTLLM"
    )
    # Model construction builds the drafter while the target's mapping is
    # repurposed (repurpose_helix_cp_to_tp), so that is the drafter's mapping.
    draft_config = model_config(target.repurpose_helix_cp_to_tp(), DRAFT_LAYERS, draft_heads)
    config = kv_cache_config(64 << 20)
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = spec_config
    creator._draft_config = draft_config
    creator._draft_model_engine = None
    creator._mapping = target
    creator._max_seq_len = MAX_SEQ_LEN
    creator._unification_max_seq_len = MAX_SEQ_LEN
    creator._kv_cache_config = config
    creator._get_draft_max_attention_window = lambda max_seq_len, cfg: None
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=model_config(target, TARGET_LAYERS, 8)),
        dtype=torch.bfloat16,
    )
    creator._kv_cache_manager_cls = KVCacheManagerV2
    creator._is_disagg = False
    creator._max_beam_width = 1
    creator._get_draft_kv_model_config = lambda: draft_config
    return creator, config


@pytest.mark.parametrize("layout", (CP_REPLICATED, CP_REPLICATED_HEADS, REVIEWER_B), ids=str)
def test_creator_shares_one_manager_with_the_cp_replicated_drafter(layout):
    for rank in range(layout.world):
        target = owner_mapping(layout, rank, TARGET)
        creator, config = make_creator(target, layout.draft_heads)
        assert creator._dflash_unify_blocker() is None
        assert creator._should_unify_dflash_kv_cache()
        specs = creator._build_unified_dflash_layer_specs(config)
        owners = creator._unified_owner_mappings()
        # The drafter keeps its own mapping: TP over the CP ranks, no CP.
        assert (owners[DRAFT].tp_size, owners[DRAFT].cp_size) == (layout.world, 1)
        assert owners[TARGET] is target
        assert {s.num_kv_heads for s in specs if s.owner == DRAFT} == {
            heads_per_rank(layout, DRAFT)
        }
        manager = KVCacheManagerV2(
            config,
            CacheType.SELF,
            num_layers=TARGET_LAYERS,
            num_kv_heads=heads_per_rank(layout, TARGET),
            head_dim=HEAD_DIM,
            tokens_per_block=P,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=64,
            mapping=target,
            dtype=DataType.BF16,
            layer_specs=specs,
            owner_mappings=owners,
        )
        try:
            base = Baseline(manager)
            open_cache(manager, 0).resize(9 * P + 5)
            check_allocation(manager, layout, rank, 0, 9 * P + 5, base)
        finally:
            close_all(manager)


def test_creator_rejects_a_drafter_that_splits_its_pages_without_runtime_support():
    target = helix_mapping(4, 0, 1, 4)
    creator, _ = make_creator(target, 8)
    # A drafter left on the helix mapping would split its pages over CP; its
    # attention reads every page, so the shared manager is refused.
    creator._draft_config.mapping = target
    assert "attention reads every page" in creator._dflash_unify_blocker()


# ---- Admission across processes ----------------------------------------------------------------


class _GlooMinSync:
    """The manager's cross-rank capacity sync, over this test's Gloo group."""

    def allreduce(self, value, op):
        if os.environ.get("_KV_OWNER_TEST_NO_SYNC"):
            return value
        tensor = torch.tensor(value, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return tensor.item()


def _all_equal(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    assert all(v == values[0] for v in values), values
    return values


def _first_admission_refused(manager):
    """Admit one-token requests until one is refused; return how many were admitted."""
    request_id = 1
    while try_admit(manager, request_id, 1):
        request_id += 1
    return request_id - 1


def _admission_worker(rank, port, layout, short_rank, synced=True):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if not synced:
        os.environ["_KV_OWNER_TEST_NO_SYNC"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=layout.world)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.cuda.init()
    # Fault injection: one rank alone has less KV memory.
    gpu_bytes = (4 << 20) if rank == short_rank else (16 << 20)
    with patch(
        "tensorrt_llm._torch.distributed.communicator.Distributed.get",
        return_value=_GlooMinSync(),
    ):
        manager = make_manager(layout, rank, gpu_bytes=gpu_bytes)
    try:
        if not synced:
            # Negative control: without the sync the short rank refuses first.
            assert open_cache(manager, 0).resize(2 * P)
            counts = [None] * layout.world
            dist.all_gather_object(counts, _first_admission_refused(manager))
            assert counts[short_rank] < max(counts), counts
            return
        _all_equal([s.total for s in manager.impl.get_storage_statistics(0)])
        assert open_cache(manager, 0).resize(2 * P)
        request_id = 1
        while True:
            admitted = try_admit(manager, request_id, 1)
            _all_equal(admitted)
            if not admitted:
                break
            request_id += 1
        assert request_id > 1
        _all_equal(manager.impl.get_strided_admitted_slots())
        cache = manager.kv_cache_map[0]
        grow_to = max(layout.target_cp, layout.draft_cp) * P + 1
        before = (cache.capacity, manager.impl.get_strided_admitted_slots())
        grown = cache.resize(grow_to)
        _all_equal(grown)
        assert not grown and before == (cache.capacity, manager.impl.get_strided_admitted_slots())
        for request in range(1, request_id):
            manager.kv_cache_map.pop(request).close()
        grown = cache.resize(grow_to)
        _all_equal(grown)
        assert grown
        dist.barrier()
    finally:
        close_all(manager)
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("layout", "short_rank"),
    [
        (TP_TARGET_DRAFT_CP, 2),
        (HELIX_4, 1),  # target CP rank 0 on rank 0 only, drafter CP rank 0 on ranks 0 and 2
        (HELIX_4_FINE_DRAFT, 3),  # target CP rank 0 on ranks 0 and 2, drafter on rank 0
        (CP_REPLICATED, 3),
        (HELIX_8, 5),
        (HELIX_8_FINE_DRAFT, 2),
    ],
    ids=str,
)
def test_admission_is_coherent_across_ranks(layout, short_rank):
    if torch.cuda.device_count() < layout.world:
        pytest.skip(f"requires {layout.world} GPUs")
    mp.spawn(
        _admission_worker, args=(_free_port(), layout, short_rank), nprocs=layout.world, join=True
    )


@pytest.mark.timeout(600)
def test_admission_diverges_without_the_capacity_sync():
    layout, short_rank = HELIX_4, 1
    if torch.cuda.device_count() < layout.world:
        pytest.skip(f"requires {layout.world} GPUs")
    mp.spawn(
        _admission_worker,
        args=(_free_port(), layout, short_rank, False),
        nprocs=layout.world,
        join=True,
    )


# ---- The drafter's cache writer over the shared manager ----------------------------------------


def pool_memory(manager):
    """All KV memory of a manager, as one flat BF16 tensor per pool."""
    pools = {}
    for group in KVRegionExtractorV1(manager).page_table.pool_groups:
        for pool in group.pools:
            pools[pool.base_address] = max(
                pools.get(pool.base_address, 0), pool.slot_bytes * pool.num_slots
            )
    return [
        convert_to_torch_tensor(TensorWrapper(base, DataType.BF16, [size // BF16]))
        for base, size in pools.items()
    ]


def bind_drafter(manager, layout, request_ids):
    """A DFlash worker whose context cache is the shared manager's draft layers."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_unified_cache = True
    worker._ctx_paged_append = None
    worker._ctx_page_size = P
    nkv = heads_per_rank(layout, DRAFT)
    pool = worker._managed_ctx_pool(manager, DRAFT_LAYERS, nkv, HEAD_DIM, torch.bfloat16, 2)
    assert pool is not None
    assert worker._init_ctx_block_tables(
        manager, pool, len(request_ids), DRAFT_LAYERS, nkv, HEAD_DIM, P
    )
    worker._ctx_kv_buf = pool
    worker._ctx_kv_last_page_len = torch.full(
        (len(request_ids),), P, dtype=torch.int32, device="cuda"
    )
    return worker


def refresh_tables(worker, manager, request_ids):
    offsets = torch.empty(
        (manager.num_attention_op_pools, len(request_ids), 2, manager.max_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    manager.copy_batch_block_offsets(
        offsets, request_ids, beam_width=1, num_contexts=0, num_seqs=len(request_ids)
    )
    assert worker._refresh_ctx_block_tables(
        SimpleNamespace(kv_cache_block_offsets=offsets), len(request_ids)
    )


def check_drafter_writes(manager, layout, rank, request_ids, lengths, k, v):
    """Every written token sits in its own page of its request; nothing else changed."""
    written, start = 0, 0
    for request_id, num_tokens in zip(request_ids, lengths):
        for index, layer in enumerate(layer_ids(manager, DRAFT)):
            buffer = manager.get_buffers(layer, kv_layout="HND")
            table = manager.get_batch_cache_indices([request_id], layer)[0]
            pages = owned_pages(layout, rank, DRAFT, div_up(num_tokens, P))
            assert pages == list(range(div_up(num_tokens, P))), "the drafter holds every page"
            for position in range(num_tokens):
                slot = table[position // P]
                token = start + position
                assert torch.equal(buffer[slot, 0, :, position % P], k[token, index])
                assert torch.equal(buffer[slot, 1, :, position % P], v[token, index])
            written += 2 * num_tokens * heads_per_rank(layout, DRAFT) * HEAD_DIM
        start += num_tokens
    landed = sum(int((~torch.isnan(pool)).sum()) for pool in pool_memory(manager))
    assert landed == written, f"{landed - written} elements written outside the drafter's tokens"


def drafter_inputs(layout, lengths, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lengths)
    shape = (total, DRAFT_LAYERS, heads_per_rank(layout, DRAFT), HEAD_DIM)
    k = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    rows = torch.cat(
        [torch.full((n,), row, dtype=torch.int32, device="cuda") for row, n in enumerate(lengths)]
    )
    positions = torch.cat([torch.arange(n, dtype=torch.int32, device="cuda") for n in lengths])
    return k, v, rows, positions


@pytest.mark.parametrize(
    "layout", (CP_REPLICATED, CP_REPLICATED_HEADS, MATCHED_TP, REVIEWER_B), ids=str
)
@pytest.mark.parametrize("rank", (0, 3))
def test_drafter_cache_writer_stays_in_its_pages(layout, rank):
    """The DFlash context-KV writer, eager and replayed from a CUDA graph, over poisoned memory."""
    lengths = (9 * P + 5, 3 * P)
    manager = make_manager(layout, rank)
    try:
        for pool in pool_memory(manager):
            pool.fill_(float("nan"))
        request_ids = [10, 11]
        for request_id, num_tokens in zip(request_ids, lengths):
            assert open_cache(manager, request_id).resize(num_tokens)
        worker = bind_drafter(manager, layout, request_ids)
        refresh_tables(worker, manager, request_ids)

        k, v, rows, positions = drafter_inputs(layout, lengths, seed=1)
        worker._store_context_kv_paged(k, v, rows, positions)
        torch.cuda.synchronize()
        check_drafter_writes(manager, layout, rank, request_ids, lengths, k, v)

        # Captured once, replayed for other requests: the graph keeps reading the
        # persistent block table, which is refilled in place.
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                worker._store_context_kv_paged(k, v, rows, positions)
        for request_id in request_ids:
            manager.kv_cache_map.pop(request_id).close()
        for pool in pool_memory(manager):
            pool.fill_(float("nan"))
        request_ids = [20, 21]
        # Opened in the other order, so the requests take other pages.
        for request_id, num_tokens in reversed(list(zip(request_ids, lengths))):
            assert open_cache(manager, request_id).resize(num_tokens)
        refresh_tables(worker, manager, request_ids)
        k2, v2, _, _ = drafter_inputs(layout, lengths, seed=2)
        k.copy_(k2)
        v.copy_(v2)
        graph.replay()
        torch.cuda.synchronize()
        check_drafter_writes(manager, layout, rank, request_ids, lengths, k2, v2)
    finally:
        close_all(manager)
