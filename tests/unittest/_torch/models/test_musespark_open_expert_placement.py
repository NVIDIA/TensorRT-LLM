# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static expert placement: table parsing, slot map, routing memo, weight slicing (CPU)."""

import json
import random

import pytest
import torch

from tensorrt_llm._torch.models import musespark_open_expert_placement as P
from tensorrt_llm._torch.models.checkpoints.hf import musespark_open_weight_mapper as M
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenRoutingMethod


def _perm(seed):
    ids = list(range(256))
    random.Random(seed).shuffle(ids)
    return ids


def test_table_and_slot_map(tmp_path, monkeypatch):
    perm = _perm(1)
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"sha": "t", "layers": {"3": perm, "4": list(range(256))}}))
    monkeypatch.setenv("MUSESPARK_OPEN_EXPERT_PLACEMENT", str(f))
    P._placement_cache.clear()
    table = P.placement_table(256)
    assert list(table) == [3] and table[3] == perm  # identity layer dropped
    sm = P.SlotMap(perm)
    ids = torch.randint(0, 256, (37, 8), dtype=torch.int32)
    slots = sm(ids)
    assert slots.dtype == torch.int32 and slots.shape == ids.shape
    # slot s holds expert perm[s]
    assert torch.equal(torch.tensor(perm, dtype=torch.int32)[slots.long()], ids)


def test_table_rejects_non_permutation(tmp_path, monkeypatch):
    bad = list(range(256))
    bad[5] = 6
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"layers": {"1": bad}}))
    monkeypatch.setenv("MUSESPARK_OPEN_EXPERT_PLACEMENT", str(f))
    P._placement_cache.clear()
    with pytest.raises(ValueError):
        P.placement_table(256)


def test_routing_memo_placed_once():
    r = MuseSparkOpenRoutingMethod(
        top_k=8,
        num_experts=256,
        callable_e_score_correction_bias=lambda: torch.zeros(256),
        renorm_eps=1e-20,
    )
    perm = _perm(2)
    r._slot_map = P.SlotMap(perm)
    ids = torch.randint(0, 256, (5, 8), dtype=torch.int32)
    w = torch.rand(5, 8)
    logits = torch.zeros(5, 256)
    r._routing_memo = (logits, None, ids, w)
    got = r._routing_memo[2]
    assert torch.equal(torch.tensor(perm, dtype=torch.int32)[got.long()], ids)
    assert r._routing_memo[3] is w
    r.clear_routing_memo()
    assert r._routing_memo is None


def test_apply_torch_path_places_ids():
    r = MuseSparkOpenRoutingMethod(
        top_k=8,
        num_experts=256,
        callable_e_score_correction_bias=lambda: torch.zeros(256),
        renorm_eps=1e-20,
    )
    logits = torch.randn(9, 256)
    base_ids, base_w = r.apply(logits)
    r2 = MuseSparkOpenRoutingMethod(
        top_k=8,
        num_experts=256,
        callable_e_score_correction_bias=lambda: torch.zeros(256),
        renorm_eps=1e-20,
    )
    perm = _perm(3)
    r2._slot_map = P.SlotMap(perm)
    ids, w = r2.apply(logits)
    assert torch.equal(w, base_w)
    assert torch.equal(torch.tensor(perm, dtype=torch.int32)[ids.long()], base_ids.to(torch.int32))


class _Acc:
    def __init__(self):
        self.consumed = set()


def test_build_expert_weights_placed_ids():
    E, inter, H = 8, 4, 6
    w13 = torch.arange(E * 2 * inter * H, dtype=torch.float32).reshape(E, 2 * inter, H)
    w2 = torch.arange(E * H * inter, dtype=torch.float32).reshape(E, H, inter) + 1e6
    weights = {"x.w13_weight": w13, "x.w2_weight": w2}
    ids = [5, 1, 7, 2]
    ent = M.build_expert_weights(weights, "x", 0, 4, E, inter, False, _Acc(), expert_ids=ids)
    for e in ids:
        assert torch.equal(ent[f"{e}.w1.weight"], w13[e, :inter])
        assert torch.equal(ent[f"{e}.w3.weight"], w13[e, inter:])
        assert torch.equal(ent[f"{e}.w2.weight"], w2[e])
    assert not any(k.startswith("0.") for k in ent)

    # contiguous ids keep the range path
    class _E:
        initial_local_expert_ids = [4, 5, 6, 7]

    assert M._placed_expert_ids(_E()) is None
    _E.initial_local_expert_ids = ids
    assert M._placed_expert_ids(_E()) == ids


def test_apply_to_experts_sets_backend():
    class _B:
        def __init__(self):
            self.initial_local_expert_ids = list(range(64, 128))
            self.initial_global_assignments = list(range(256))
            self.layer_load_balancer = None

    class _W(_B):
        def __init__(self):
            super().__init__()
            self.backend = _B()

    w = _W()
    perm = _perm(4)
    local = P.apply_to_experts(w, perm, 1, 4)
    assert local == perm[64:128]
    assert w.initial_local_expert_ids == local and w.backend.initial_local_expert_ids == local
