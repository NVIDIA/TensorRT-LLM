# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The eplb catalog entry: slot layouts, layer registration and the routing op.

Slot layouts and registration are plain Python and run on the CPU. The routing op
and the sum over EP ranks of a slot-routed MoE layer need a GPU.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.eplb import EplbLayer, EplbSlots
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.eplb import eplb_route as op
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    get_moe_load_balancer,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# --- slot layouts (CPU) ---------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("ep_rank", range(4))
def test_without_eplb_a_rank_holds_a_contiguous_window(ep_rank):
    slots = EplbSlots.without_eplb(256, 4, ep_rank)
    assert (slots.num_slots, slots.local_slots, slots.slot_offset) == (256, 64, 64 * ep_rank)
    assert slots.slot_experts == tuple(range(64 * ep_rank, 64 * ep_rank + 64))


@pytest.mark.cpu_only
def test_the_default_placement_is_the_builtin_moes():
    """Without assignments, rank r's slots hold the experts from r * E / ep on, wrapping:
    at 288 slots over 4 ranks each rank's 72 slots repeat the next rank's first 8 experts,
    and the last rank's wrap to experts 0-7."""
    config = MoeLoadBalancerConfig(num_slots=288)
    layouts = [EplbSlots.from_config(config, 3, 256, 4, rank) for rank in range(4)]
    for rank, slots in enumerate(layouts):
        assert (slots.num_slots, slots.local_slots, slots.slot_offset) == (288, 72, 72 * rank)
    assert layouts[0].slot_experts == tuple(range(72))
    assert layouts[3].slot_experts == tuple(range(192, 256)) + tuple(range(8))
    assert layouts[0].global_assignments == sum((s.slot_experts for s in layouts), ())


@pytest.mark.cpu_only
def test_given_assignments_place_each_layer():
    a3 = [3, 0, 1, 3, 2, 3, 1, 0]
    a4 = [0, 1, 2, 3, 0, 1, 2, 3]
    config = MoeLoadBalancerConfig(num_slots=8, initial_global_assignments={3: a3, 4: a4})
    assert EplbSlots.from_config(config, 3, 4, 2, 0).slot_experts == (3, 0, 1, 3)
    assert EplbSlots.from_config(config, 3, 4, 2, 1).slot_experts == (2, 3, 1, 0)
    assert EplbSlots.from_config(config, 4, 4, 2, 1).slot_experts == (0, 1, 2, 3)
    with pytest.raises(KeyError):
        EplbSlots.from_config(config, 5, 4, 2, 0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "assignments, missed",
    [
        ([0, 1, 2, 0, 1, 2], "every one of the 4 experts"),  # expert 3 has no slot
        ([0, 1, 2, 3, 4, 0], "every one of the 4 experts"),  # an expert that does not exist
        ([0, 1, 2, 3, 0], "do not split evenly"),
    ],
)
def test_a_placement_that_loses_an_expert_is_refused(assignments, missed):
    with pytest.raises(ValueError, match=missed):
        EplbSlots.from_assignments(assignments, 4, 2, 0)


@pytest.fixture
def mocked_runtime() -> Iterator[MagicMock]:
    """The C++ balancer and the node's MPI communicator, mocked: registration is the
    Python side's contract."""
    shared_comm = MagicMock()
    shared_comm.Get_rank.return_value = 1
    shared_comm.Get_size.return_value = 4
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer") as runtime,
        patch.object(
            MoeLoadBalancer,
            "_setup_mpi_comm",
            lambda self: setattr(self, "shared_mpi_comm", shared_comm),
        ),
    ):
        yield runtime


@pytest.mark.cpu_only
def test_registering_without_the_engines_balancer_is_an_error():
    assert get_moe_load_balancer() is None
    with pytest.raises(RuntimeError, match="no MoE load balancer is active"):
        EplbLayer.register(EplbSlots.without_eplb(8, 2, 0), top_k=2)


@pytest.mark.cpu_only
def test_registering_adds_the_layer_with_its_placement(mocked_runtime):
    """One `add_layer` per registration, sized by this rank's slots, then the global
    placement; the MTP layer's repeat count reaches the balancer's layer."""
    config = MoeLoadBalancerConfig(num_slots=288)
    slots = EplbSlots.from_config(config, 61, 256, 4, 1)
    balancer = MoeLoadBalancer(ep_rank=1, ep_size=4, layer_updates_per_iter=0)
    try:
        with balancer:
            layer = EplbLayer.register(slots, top_k=8, repeat_count=3)
        impl = mocked_runtime.return_value
        impl.add_layer.assert_called_once_with(256, 8, 72)
        impl.add_layer.return_value.set_initial_weight_assignments.assert_called_once_with(
            list(slots.global_assignments)
        )
        assert balancer.single_layer_load_balancers == [layer.balancer_layer]
        assert layer.balancer_layer.get_repeat_count() == 3
        assert layer.balancer_layer.is_static_routing()
        assert layer.slots is slots
    finally:
        balancer.shutdown()
    assert get_moe_load_balancer() is None


# --- the routing op (GPU) ---------------------------------------------------------------------


def _placement(experts: int, slots: int, ep_size: int, hot_replicas: int, seed: int) -> list:
    """`slots` global slots over `ep_size` ranks, unlike the default placement.

    Every expert once, in a shuffled order; then the spare slots: the hottest expert
    `hot_replicas` times in all, one more copy of others, and on rank 0 a second copy of
    an expert rank 0 already holds.
    """
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(experts, generator=g).tolist()
    per_rank, spare = experts // ep_size, (slots - experts) // ep_size
    hot = order[-1]
    extra = [hot] * min(hot_replicas - 1, ep_size * spare)
    extra += [e for e in order if e != hot][: ep_size * spare - len(extra)]
    out = []
    for rank in range(ep_size):
        own = order[rank * per_rank : (rank + 1) * per_rank]
        mine = extra[rank * spare : (rank + 1) * spare]
        if rank == 0 and spare:
            mine[0] = own[0]
        out += own + mine
    return out


def _expert_ids(tokens: int, experts: int, top_k: int, seed: int) -> torch.Tensor:
    """`[tokens, top_k]` int32: each token's distinct experts, as a router emits them."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)]
    return torch.stack(rows).to(torch.int32).cuda()


class _Balancers:
    """One static load balancer per EP rank, each holding the layer at `placement`: what
    every rank of an EP group builds."""

    def __init__(self, experts: int, top_k: int, placement: list, ep_size: int):
        self.balancers = [MoeLoadBalancer(rank, ep_size, 0) for rank in range(ep_size)]
        self.layers = []
        for balancer in self.balancers:
            layer = balancer.add_layer(experts, top_k, len(placement) // ep_size)
            layer.set_initial_weight_assignments(placement)
            self.layers.append(layer)

    def shutdown(self) -> None:
        for balancer in self.balancers:
            balancer.shutdown()


# The shapes the default deepseek target routes, by target directory under `models/`: the
# 8192-row chunk its attention-DP gather fills, at the two layouts it runs EPLB at.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at dep4: 288 slots, an 8192-row chunk",
            spec=dict(experts=256, slots=288, ep_size=4, top_k=8, tokens=8192),
        ),
        Cell(
            why="deepseek default at dep8: 288 slots, 36 a rank, an 8192-row chunk",
            spec=dict(experts=256, slots=288, ep_size=8, top_k=8, tokens=8192),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@requires_gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    """Every rank routes the same ids to the same slots; each slot holds the token's expert;
    an expert held once maps to its slot; and within each 1024-id block the kernel
    arbitrates, an expert's occurrences are spread over its replicas round-robin."""
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    placement = _placement(
        spec["experts"], spec["slots"], spec["ep_size"], spec.get("hot_replicas", 2), seed
    )
    ids = _expert_ids(spec["tokens"], spec["experts"], spec["top_k"], seed)
    group = _Balancers(spec["experts"], spec["top_k"], placement, spec["ep_size"])
    try:
        with validating(op):
            routed = [op(ids, layer) for layer in group.layers]
        ref = op.reference(ids, group.layers[0])
    finally:
        group.shutdown()

    got = routed[0]
    assert got.dtype is torch.int32 and got.shape == ids.shape
    for other in routed[1:]:
        assert torch.equal(other, got), "EP ranks routed the same ids to different slots"
    slot_expert = torch.tensor(placement, dtype=torch.int32, device="cuda")
    assert torch.equal(slot_expert[got.long()], ids), "a slot does not hold the token's expert"
    assert torch.equal(slot_expert[ref.long()], ids), "the reference's slot does not either"

    replicas = torch.bincount(slot_expert.long(), minlength=spec["experts"])
    single = replicas[ids.long()] == 1
    assert torch.equal(got[single], ref[single]), "an expert held once left its slot"

    flat_ids, flat_slots = ids.flatten().tolist(), got.flatten().tolist()
    for start in range(0, len(flat_ids), 1024):
        block = list(zip(flat_ids[start : start + 1024], flat_slots[start : start + 1024]))
        for expert in {e for e, _ in block if replicas[e] > 1}:
            counts = [0] * len(placement)
            for e, s in block:
                if e == expert:
                    counts[s] += 1
            held = [counts[s] for s, e in enumerate(placement) if e == expert]
            assert max(held) - min(held) <= 1, (expert, held)


@requires_gpu
def test_an_id_outside_the_experts_routes_to_no_slot() -> None:
    placement = _placement(16, 24, 4, 2, seed=7)
    ids = torch.tensor([[0, 15], [-1, 3], [16, 4]], dtype=torch.int32, device="cuda")
    group = _Balancers(16, 2, placement, 4)
    try:
        got = op(ids, group.layers[0])
        ref = op.reference(ids, group.layers[0])
    finally:
        group.shutdown()
    assert got[1, 0].item() == got[2, 0].item() == ref[1, 0].item() == len(placement)


# --- a slot-routed layer sums to the layer (GPU) ---------------------------------------------


@requires_gpu
def test_slot_windows_sum_to_the_layer() -> None:
    """Under EPLB each rank runs its slot window on slot-routed ids, and the windows add up
    to the layer: the DeepSeek-V3-Lite routed geometry (72 experts, H 2560, I 1536, top-6)
    at 80 slots over 4 ranks, with replicas across and within ranks, against the 72-expert
    layer without EPLB.

    The slot weights are the expert stack's own rows, gathered by the placement, and the
    route is the engine's kernel on one balancer per rank -- so this checks, independently
    of any target, that slot ids, slot windows and replicated weights compose: a slot id
    landing in the wrong window, or a window loaded with another slot's expert, moves the
    sum by far more than the kernel's own error.
    """
    from _fp4_moe_test_utils import (
        _assert_moe_close,
        _build,
        _calibrate_g2,
        _call,
        _rand_activation,
        _ref_moe,
        _routing,
        _scalars,
    )

    experts, slots, ep_size, hidden, top_k = 72, 80, 4, 2560, 6
    args, ref = _build(experts, hidden, 1536, seed=7280)
    placement = _placement(experts, slots, ep_size, hot_replicas=3, seed=80)
    local = slots // ep_size
    gen = torch.Generator(device="cuda").manual_seed(8072)
    g1 = 137.0
    data, sf, x = _rand_activation(1024, hidden, g1, gen)
    ids, wts = _routing(1024, experts, top_k, gen)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(experts, g1, g2)

    full = _call(
        data,
        sf,
        args,
        experts,
        top_k,
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )[0]
    want = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
    _assert_moe_close(full, want)

    group = _Balancers(experts, top_k, placement, ep_size)
    try:
        routed = [op(ids, layer) for layer in group.layers]
    finally:
        group.shutdown()
    parts = []
    for rank in range(ep_size):
        assert torch.equal(routed[rank], routed[0])
        held = torch.tensor(placement[rank * local : (rank + 1) * local], device="cuda")
        window = {
            k: args[k][held].contiguous()
            for k in (
                "gemm1_weights",
                "gemm1_weights_scale",
                "gemm2_weights",
                "gemm2_weights_scale",
            )
        }
        window["intermediate_size"] = args["intermediate_size"]
        parts.append(
            _call(
                data,
                sf,
                window,
                slots,
                top_k,
                local_expert_offset=rank * local,
                local_num_experts=local,
                topk_ids=routed[rank],
                topk_weights=wts,
                output1_scale_scalar=s1[held].contiguous(),
                output1_scale_gate_scalar=sg[held].contiguous(),
                output2_scale_scalar=s2[held].contiguous(),
            )[0]
        )
    summed = sum(p.float() for p in parts).to(torch.bfloat16)
    _assert_moe_close(summed, want)
