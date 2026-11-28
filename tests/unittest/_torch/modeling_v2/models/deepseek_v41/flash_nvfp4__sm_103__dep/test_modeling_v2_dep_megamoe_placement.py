# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank test of `ops/moe/megamoe_placement`: the dep target's `_MegaMoe` layers under the engine's warmup
expert placement controller. Four EP ranks build two target layers (one copy per expert, and a replicated cover with
extra slots) plus a layer with a per-slot tensor the placement does not know, register them through the target's own
constructor, feed the controller skewed expert counts and run one round. Then on every rank:

* the moved layers' placement changed and the slowest rank's load dropped;
* every local slot holds, in every per-slot tensor, the bytes of the expert its new assignment names (the copies of an
  expert are equal);
* the layer's slot table was rewritten in place (same storage) to `megamoe_slot_table` of the new placement;
* the layer with the unknown per-slot tensor kept its placement (refused, never moved partially).

Needs 4 visible GPUs; run under the repo's `mpi_pool_executor` fixture.
"""

import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 4
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
        reason="needs 4 SM100/SM103/SM107 GPUs",
    ),
]

WORLD, EXPERTS, TOP_K, HIDDEN, INTER = 4, 16, 2, 64, 32
TARGETS = ("flash_nvfp4__sm_103__dep",)


def _pattern(expert: int, shape, dtype) -> torch.Tensor:
    """Distinct bytes per expert (and per position, so a partial copy shows)."""
    n = 1
    for s in shape:
        n *= s
    base = (torch.arange(n, dtype=torch.int64, device="cuda") * 13 + expert * 101 + 7) % 251
    if dtype == torch.uint8:
        return base.to(torch.uint8).view(shape)
    return (base.float() / 7 + expert).view(shape).to(dtype)


def _expert_tensors(expert: int) -> dict:
    return {
        "fc1": _pattern(expert, (2 * INTER, HIDDEN // 2), torch.uint8),
        "fc1_sf": _pattern(expert, (512,), torch.uint8),
        "fc2": _pattern(expert, (HIDDEN, INTER // 2), torch.uint8),
        "fc2_sf": _pattern(expert, (256,), torch.uint8),
        "fc1_alpha": _pattern(expert, (), torch.float32),
        "fc2_alpha": _pattern(expert + 1000, (), torch.float32),
        "fc1_norm_const": _pattern(expert + 2000, (), torch.float32),
    }


def _layer(megamoe, rank: int, slots: int, layer_idx: int, extra_per_slot: bool = False):
    mc = megamoe.MegaMoeConfig(
        ep_size=WORLD, ep_rank=rank, num_experts=EXPERTS, num_slots=slots, top_k=TOP_K, hidden=HIDDEN, inter=INTER,
        clamp=7.0, combine_format="bf16", fused_shared=False, max_tokens=64, initial_assignments=None,
    )  # fmt: skip
    layer = megamoe._MegaMoe(SimpleNamespace(routed_scaling_factor=1.0), mc, layer_idx)
    local = layer.assignments[rank * mc.local_slots : (rank + 1) * mc.local_slots]
    per = [_expert_tensors(e) for e in local]
    stacked = {k: torch.stack([p[k] for p in per]).contiguous() for k in per[0]}
    scalars = {k: stacked[k] for k in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")}
    scalars["input_global_scale"] = torch.ones(1, device="cuda")
    layer.set_experts(
        stacked["fc1"],
        stacked["fc1_sf"],
        stacked["fc2"],
        stacked["fc2_sf"],
        scalars,
        torch.device("cuda"),
    )
    if extra_per_slot:
        layer.register_buffer(
            "fc1_bias", torch.zeros(mc.local_slots, device="cuda"), persistent=False
        )
    return layer


def _rank_loads(load, assign, spr):
    copies = {}
    for e in assign:
        copies[e] = copies.get(e, 0) + 1
    return [sum(load[e] / copies[e] for e in assign[r * spr : (r + 1) * spr]) for r in range(WORLD)]


def _rank_body(world: int, target: str):
    import importlib

    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import megamoe_placement
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_expert_slots import (
        megamoe_slot_table,
    )
    from tensorrt_llm._torch.moe.fused_moe.moe_warmup_placement import WarmupPlacementController
    from tensorrt_llm.llmapi.llm_args import MoeWarmupPlacementConfig

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        megamoe = importlib.import_module(
            f"tensorrt_llm._torch.modeling_v2.models.deepseek_v41.{target}.megamoe"
        )
        settings = MoeWarmupPlacementConfig(
            start_step=0, interval_steps=1, max_migrations=4, min_rows=0, planner="lpt", max_moved_fraction=0.5,
            min_gain=0.0,
        )  # fmt: skip
        controller = WarmupPlacementController(settings, ep_rank=rank, ep_size=world)
        megamoe_placement.megamoe_placement_controller = (
            lambda: controller
        )  # the engine's, in a real build
        one = _layer(megamoe, rank, EXPERTS, 0)
        rep = _layer(megamoe, rank, EXPERTS + WORLD, 1)
        bad = _layer(megamoe, rank, EXPERTS, 2, extra_per_slot=True)
        layers = (one, rep, bad)
        assert all(layer.placement is not None for layer in layers)
        initial = [list(layer.assignments) for layer in layers]
        tables = [layer.table.data_ptr() for layer in layers]
        controller.finalize()
        assert one.placement.counters() is not None
        controller.on_step(0)  # counting starts
        load = [
            400 if e < 4 else 4 for e in range(EXPERTS)
        ]  # experts 0..3 (rank 0's window) are hot
        for layer in layers:
            layer.placement.counters().hist.copy_(torch.tensor(load, dtype=torch.int32) // world)
        controller.on_step(1)  # the round
        torch.cuda.synchronize()
        for k, layer in enumerate(layers):
            spr = layer.mc.local_slots
            assert layer.table.data_ptr() == tables[k], "the slot table must be rewritten in place"
            assert torch.equal(
                layer.table, megamoe_slot_table(layer.assignments, EXPERTS, rank, "cuda")
            )
            if layer is bad:
                assert layer.assignments == initial[k], (
                    "a layer with an unknown per-slot tensor must not move"
                )
            else:
                assert layer.assignments != initial[k], f"layer {k} did not move"
                assert max(_rank_loads(load, layer.assignments, spr)) < max(
                    _rank_loads(load, initial[k], spr)
                )
            local = layer.assignments[rank * spr : (rank + 1) * spr]
            for j, e in enumerate(local):
                want = _expert_tensors(e)
                for name in want:
                    assert torch.equal(getattr(layer, name)[j], want[name]), (
                        f"layer {k} slot {j} {name} != expert {e}"
                    )
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_round_moves_slots_and_rewrites_the_table(mpi_pool_executor, target):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_rank_body, *zip(*[(world, target)] * world))
    for r in results:
        assert r is True
