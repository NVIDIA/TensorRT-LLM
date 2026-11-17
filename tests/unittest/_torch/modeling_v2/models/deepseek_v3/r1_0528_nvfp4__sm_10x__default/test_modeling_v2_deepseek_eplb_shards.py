# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deepseek default target's per-rank weight shares under static EPLB, checked on the CPU.

Under static EPLB a rank holds a contiguous range of expert slots, and each slot holds the
expert the load balancer's placement puts there, possibly an expert another slot also holds.
The checks reuse the shares test's synthetic checkpoint and its world-size-1 oracle
(``test_modeling_v2_deepseek_default_shards``): the slots tile every expert of every routed
layer, and ``load`` fills each slot with exactly the oracle's tensors for its expert, reading
each expert once per slot that holds it. Under online EPLB ``load`` also hands each layer's
balancer handle this rank's share of the experts.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from test_modeling_v2_deepseek_default_shards import (
    _TINY,
    _WS1,
    _assert_slot_scales_are_the_expert_products,
    _Checkpoint,
    _default_core,
    _layout_id,
    _mapping,
    _tiny_keys,
)
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.deepseek_v3 import routing
from tensorrt_llm._torch.modeling_v2.models.deepseek_v3.r1_0528_nvfp4__sm_10x__default import (  # noqa: E501
    weights as D,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

_EPLB_LAYOUTS = sorted(routing._EPLB_LAYOUTS)


def _placement(layer: int, num_experts: int, ep_size: int, spare: int) -> list[int]:
    """A placement unlike the default one: the experts shuffled per layer, then `spare`
    more slots a rank holding other experts again -- on rank 0 a second copy of an expert
    it already holds."""
    order = torch.randperm(num_experts, generator=torch.Generator().manual_seed(layer)).tolist()
    per_rank = num_experts // ep_size
    out = []
    for rank in range(ep_size):
        own = order[rank * per_rank : (rank + 1) * per_rank]
        extra = [order[(rank * per_rank + per_rank + k) % num_experts] for k in range(spare)]
        if rank == 0:
            extra[0] = own[0]
        out += own + extra
    return out


def _eplb_core(layout, rank: int, cfg, mtp: bool, spare: int = 2, updates: int = 0):
    """`_default_core` under EPLB, static unless `updates` layers move per forward: the
    config names the architecture the engine balances, and the load balancer config places
    the experts of every routed layer."""
    ep = layout[3]
    layers = range(cfg.first_k_dense_replace, cfg.num_hidden_layers + int(mtp))
    assignments = {i: _placement(i, cfg.n_routed_experts, ep, spare) for i in layers}
    pretrained = SimpleNamespace(**vars(cfg), architectures=["DeepseekV3ForCausalLM"])
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=pretrained,
            mapping=_mapping(layout, rank),
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
            moe_load_balancer=MoeLoadBalancerConfig(
                num_slots=cfg.n_routed_experts + ep * spare,
                initial_global_assignments=assignments,
                layer_updates_per_iter=updates,
            ),
        )
    )
    core.dims = D.MODEL_WEIGHTS.dims(core)
    return core, assignments


@pytest.mark.parametrize("layout", _EPLB_LAYOUTS, ids=_layout_id)
def test_eplb_slots_hold_the_placed_experts(layout):
    """Rank r holds slots [r * S, (r + 1) * S) of every routed layer, S = num_slots / ep,
    and each slot the expert the placement puts there; together the ranks hold every
    expert of every layer, the MTP layer's included."""
    cfg = SimpleNamespace(**_TINY)
    ep = layout[3]
    held: dict = {}
    for rank in range(layout[0]):
        core, assignments = _eplb_core(layout, rank, cfg, mtp=True)
        d = core.dims
        local = (cfg.n_routed_experts + 2 * ep) // ep
        assert (d.eplb, d.num_slots, d.local_slots, d.slot_offset) == (
            "static",
            local * ep,
            local,
            rank * local,
        )
        for i, placement in assignments.items():
            mine = tuple(placement[rank * local : (rank + 1) * local])
            assert d.slots[i].slot_experts == mine, (layout, rank, i)
            held.setdefault(i, set()).update(mine)
    assert all(experts == set(range(cfg.n_routed_experts)) for experts in held.values())


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
@pytest.mark.parametrize("layout", _EPLB_LAYOUTS, ids=_layout_id)
def test_eplb_load_fills_every_slot_with_its_expert(layout, mtp, monkeypatch):
    """`load` under static EPLB: coverage holds both ways, a rank reads each expert's
    tensors once per slot that holds it and no expert it does not hold, and every slot
    holds exactly what the world-size-1 table, which has no EPLB, loads for that slot's
    expert."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    cfg = SimpleNamespace(**_TINY)
    keys = _tiny_keys(cfg)
    oracle = _default_core(_WS1, 0, cfg, mtp)
    oracle.manifest = D.MODEL_WEIGHTS.manifest(oracle)
    for rank in range(layout[0]):
        core, assignments = _eplb_core(layout, rank, cfg, mtp)
        d = core.dims
        core.w = D.MODEL_WEIGHTS.declare(d)
        lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16))
        params = [("lm_head.weight", lm_head)] + [(f"model.w.{k}", p) for k, p in core.w.items()]
        model = SimpleNamespace(
            model=core, lm_head=SimpleNamespace(weight=lm_head), named_parameters=lambda: params
        )
        ckpt = _Checkpoint(cfg, keys)
        D.MODEL_WEIGHTS.load(model, ckpt)
        # Every key has left the dict: unread ones at the start, read ones after their last read.
        assert not set(ckpt.keys()), sorted(ckpt.keys())[:4]

        _assert_slot_scales_are_the_expert_products(core)
        reads = Counter(k for k, _, _ in ckpt.reads)
        for i in assignments:
            mine = d.slots[i].slot_experts
            names = _slot_names(i, cfg)
            assert tuple(D.MODEL_WEIGHTS.slot_weights(core, i)) == names, i
            for e in range(cfg.n_routed_experts):
                key = f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"
                assert reads[key] == mine.count(e), (layout, rank, i, e)
            for slot, e in enumerate(mine):
                for name in names:
                    want = _ws1_expert(oracle, cfg, i, name, e)
                    assert torch.equal(core.w[f"l{i}_{name}"][slot], want), (rank, i, slot, name)


def _slot_names(layer: int, cfg) -> tuple[str, ...]:
    """A routed layer's slot weights: the trunk's NVFP4 operands and output scales, the MTP
    layer's bf16 operands."""
    if layer < cfg.num_hidden_layers:
        return ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "out1_scale", "out1_gate_scale", "out2_scale")
    return ("fc1", "fc2")


def _ws1_expert(oracle, cfg, layer: int, name: str, expert: int) -> torch.Tensor:
    """What the world-size-1 table, which has no EPLB, loads into expert `expert`'s slot."""
    ckpt_key, _, cut, transform = oracle.manifest[f"l{layer}_{name}"][expert]
    return D.MODEL_WEIGHTS.read(_Checkpoint(cfg), ckpt_key, cut, transform, oracle)


class _Hosting:
    """An online EPLB layer handle that keeps the host copies `load` hands it."""

    online = True

    def __init__(self, experts):
        self.host_experts = tuple(experts)
        self.copies: dict = {}
        self.names: tuple = ()

    def share_experts(self, slot_weights, expert_weight):
        self.names = tuple(slot_weights)
        for name, slot_weight in slot_weights.items():
            for e in self.host_experts:
                self.copies[(name, e)] = expert_weight(name, e)
                assert self.copies[(name, e)].shape == slot_weight.shape[1:], (name, e)


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
@pytest.mark.parametrize("layout", _EPLB_LAYOUTS, ids=_layout_id)
def test_online_eplb_load_hands_over_this_ranks_experts(layout, mtp, monkeypatch):
    """`load` under online EPLB, with the ranks on one node: each rank hands each routed
    layer's balancer handle every slot weight of its share of the experts -- each one what
    the world-size-1 table loads for that expert -- reads each of those experts' tensors once
    more, whether or not a slot already did, and coverage still holds both ways."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    cfg = SimpleNamespace(**_TINY)
    keys = _tiny_keys(cfg)
    oracle = _default_core(_WS1, 0, cfg, mtp)
    oracle.manifest = D.MODEL_WEIGHTS.manifest(oracle)
    experts, ranks = cfg.n_routed_experts, layout[0]
    for rank in range(ranks):
        core, assignments = _eplb_core(layout, rank, cfg, mtp, updates=2)
        d = core.dims
        assert d.eplb == "online"
        share = range(rank * experts // ranks, (rank + 1) * experts // ranks)
        core.eplb_layers = {i: _Hosting(share) for i in assignments}
        core.w = D.MODEL_WEIGHTS.declare(d)
        lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16))
        params = [("lm_head.weight", lm_head)] + [(f"model.w.{k}", p) for k, p in core.w.items()]
        model = SimpleNamespace(
            model=core, lm_head=SimpleNamespace(weight=lm_head), named_parameters=lambda: params
        )
        ckpt = _Checkpoint(cfg, keys)
        D.MODEL_WEIGHTS.load(model, ckpt)
        # Every key has left the dict: unread ones at the start, read ones after their last read.
        assert not set(ckpt.keys()), sorted(ckpt.keys())[:4]

        reads = Counter(k for k, _, _ in ckpt.reads)
        for i, hosting in core.eplb_layers.items():
            names = _slot_names(i, cfg)
            assert hosting.names == names, i
            mine = d.slots[i].slot_experts
            for e in range(experts):
                key = f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"
                assert reads[key] == mine.count(e) + (e in share), (layout, rank, i, e)
            for e in share:
                for name in names:
                    want = _ws1_expert(oracle, cfg, i, name, e)
                    assert torch.equal(hosting.copies[(name, e)], want), (rank, i, e, name)
