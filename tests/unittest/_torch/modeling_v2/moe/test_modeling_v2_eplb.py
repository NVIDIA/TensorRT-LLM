# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The eplb catalog entry: slot layouts, layer registration, the routing op and, under
online EPLB, the layer's part in moving experts between slots.

Slot layouts, registration and the online protocol a layer follows with the balancer
are plain Python and run on the CPU. The routing op, the sum over EP ranks of a
slot-routed MoE layer and a real online balancer moving experts need a GPU.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe import eplb as eplb_module
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.eplb import (
    EplbLayer,
    EplbSlots,
    rows_within,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.eplb import eplb_route as op
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
    get_moe_load_balancer,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_host_accessible = pytest.mark.skipif(
    not torch.cuda.is_available() or not _tbr.is_host_accessible_device_memory_supported(),
    reason="online EPLB needs host-accessible device memory",
)


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


# --- online EPLB: what a layer tells the balancer (CPU) -----------------------------------------


class _Layer:
    """A balancer layer that records what `EplbLayer` asks of it, in order."""

    def __init__(self, online: bool, load_expert_ids=()):
        self.online = online
        self.load_expert_ids = list(load_expert_ids)
        self.calls: list = []
        self.host_tensor_sharer = MagicMock()
        self.slots: dict = {}

    def is_dynamic_routing(self):
        return self.online

    def get_load_expert_ids(self):
        return self.load_expert_ids

    def start_wait_gpu_stage(self):
        self.calls.append("wait")

    def done_wait_gpu_stage(self):
        self.calls.append("waited")

    def update_statistic_with_global_ids(self, ids, is_first_stage, is_last_stage):
        self.calls.append(("count", ids.tolist(), is_first_stage, is_last_stage))

    def start_set_cpu_stage(self):
        self.calls.append("set")

    def done_set_cpu_stage(self):
        self.calls.append("done")

    def make_tensor_host_accessible(self, t):
        # New storage with the same contents, as the engine's migration gives.
        t.set_(t.clone())
        self.calls.append(("migrate", t.data_ptr()))

    def register_weight_slot(self, local, name, t):
        self.slots[(local, name)] = t


@pytest.fixture
def no_route(monkeypatch):
    """The routing op returns the ids it is given: the protocol is the subject here."""
    monkeypatch.setattr(eplb_module, "eplb_route", lambda ids, layer, offset: ids)


def _ids(*rows):
    return torch.tensor(rows, dtype=torch.int32)


def _online(**kwargs) -> tuple[EplbLayer, _Layer]:
    layer = _Layer(online=True, **kwargs)
    return EplbLayer(layer, EplbSlots.without_eplb(8, 2, 0)), layer


@pytest.mark.cpu_only
def test_a_layer_run_once_a_forward_counts_and_hands_back_once(no_route):
    """A trunk layer: taken on its first chunk, every chunk counted, the statistic complete on
    the last chunk, then handed back -- and the next forward takes it again."""
    eplb, layer = _online()
    eplb.route(_ids([0, 1], [2, 3]))
    eplb.route(_ids([4, 5]), last=True)
    eplb.close()
    assert layer.calls == [
        "wait",
        "waited",
        ("count", [[0, 1], [2, 3]], True, False),
        ("count", [[4, 5]], False, True),
        "set",
        "done",
    ]
    layer.calls.clear()
    eplb.route(_ids([6, 7]), last=True)
    eplb.close()
    assert layer.calls == ["wait", "waited", ("count", [[6, 7]], True, True), "set", "done"]


@pytest.mark.cpu_only
def test_a_layer_run_once_a_draft_step_completes_its_count_when_handed_back(no_route):
    """The MTP layer runs once per draft step and does not know which step is the last, so
    `close` completes the statistic with a row of ids no expert has."""
    eplb, layer = _online()
    for step in range(3):
        eplb.route(_ids([step, 7]))
    eplb.close()
    assert layer.calls == [
        "wait",
        "waited",
        ("count", [[0, 7]], True, False),
        ("count", [[1, 7]], False, False),
        ("count", [[2, 7]], False, False),
        ("count", [[8, 8]], False, True),
        "set",
        "done",
    ]


@pytest.mark.cpu_only
def test_a_layer_that_ran_no_tokens_is_still_taken_and_handed_back(no_route):
    """A forward that drafted nothing: the balancer waits for every layer every forward."""
    eplb, layer = _online()
    eplb.close()
    assert layer.calls == ["wait", "waited", "set", "done"]


@pytest.mark.cpu_only
def test_padding_rows_are_left_out_of_the_count(no_route):
    """Rows `rows` does not name are padding: counted out, whatever their ids; a chunk of
    padding alone is routed but not counted."""
    eplb, layer = _online()
    ids = _ids([0, 1], [2, 3], [7, 7], [4, 5], [7, 7])
    assert torch.equal(eplb.route(ids, rows=((0, 2), (3, 4))), ids)
    eplb.route(_ids([7, 7]), rows=(), last=True)
    eplb.close()
    assert layer.calls == [
        "wait",
        "waited",
        ("count", [[0, 1], [2, 3], [4, 5]], True, False),
        ("count", [[8, 8]], False, True),
        "set",
        "done",
    ]


@pytest.mark.cpu_only
def test_rows_within_a_chunk():
    tokens = ((0, 5), (8, 13), (16, 18))
    assert rows_within(None, 0, 4) is None
    assert rows_within(tokens, 0, 4) == ((0, 4),)
    assert rows_within(tokens, 4, 12) == ((0, 1), (4, 8))
    assert rows_within(tokens, 13, 16) == ()
    assert rows_within(tokens, 12, 20) == ((0, 1), (4, 6))


@pytest.mark.cpu_only
def test_without_online_eplb_a_layer_only_routes(no_route):
    """Static EPLB: no statistic, no hand-over, no host copies, no binding."""
    layer = _Layer(online=False, load_expert_ids=range(8))
    eplb = EplbLayer(layer, EplbSlots.without_eplb(8, 2, 0))
    weight = nn.Parameter(torch.arange(8.0).reshape(4, 2), requires_grad=False)
    before = weight.data_ptr()
    eplb.route(_ids([0, 1]), last=True)
    eplb.close()
    eplb.share_experts({"w": weight}, lambda name, expert: pytest.fail("read an expert"))
    eplb.bind_slots({"w": weight})
    assert layer.calls == [] and layer.slots == {} and eplb.host_experts == ()
    assert weight.data_ptr() == before
    assert not layer.host_tensor_sharer.method_calls


@pytest.mark.cpu_only
def test_host_copies_are_this_ranks_share_of_the_experts():
    """Each name: this rank's share of the experts handed over in the slot layout, every
    other expert announced with a slot's shape and dtype; then the layer's copies go to
    shared memory, once."""
    eplb, layer = _online(load_expert_ids=(2, 3))
    slot_weights = {"w": torch.zeros(4, 3, 2, dtype=torch.uint8), "s": torch.zeros(4)}
    read = []

    def expert_weight(name, expert):
        read.append((name, expert))
        return torch.full((3, 2), expert, dtype=torch.uint8) if name == "w" else torch.tensor(1.0)

    eplb.share_experts(slot_weights, expert_weight)
    assert eplb.host_experts == (2, 3)
    assert read == [("w", 2), ("w", 3), ("s", 2), ("s", 3)]
    calls = layer.host_tensor_sharer.method_calls
    shared = [c.args[:2] for c in calls if c[0] == "share_host_tensor_with_shape"]
    announced = [c.args for c in calls if c[0] == "pre_register_host_tensor_with_shape"]
    assert shared == [(2, "w"), (3, "w"), (2, "s"), (3, "s")]
    assert [a[:2] for a in announced] == [
        (e, n) for n in ("w", "s") for e in range(8) if e not in (2, 3)
    ]
    assert {(a[1], a[2], tuple(a[3])) for a in announced} == {
        ("w", torch.uint8, (3, 2)),
        ("s", torch.float32, ()),
    }
    assert calls[-1][0] == "finalize_layer_weights"
    assert [c[0] for c in calls].count("finalize_layer_weights") == 1


@pytest.mark.cpu_only
def test_a_copy_unlike_a_slot_is_refused():
    eplb, _ = _online(load_expert_ids=(0,))
    with pytest.raises(AssertionError, match="a slot is"):
        eplb.share_experts({"w": torch.zeros(4, 3)}, lambda name, expert: torch.zeros(2))


@pytest.mark.cpu_only
def test_binding_moves_every_slot_weight_and_registers_each_slot():
    """A parameter keeps its identity and takes the new storage; a plain tensor moves in
    place; each slot is registered as a view of the new storage."""
    eplb, layer = _online()
    param = nn.Parameter(torch.arange(24.0).reshape(4, 6), requires_grad=False)
    scale = torch.arange(4.0)
    old = param.data_ptr(), scale.data_ptr()
    eplb.bind_slots({"w": param, "s": scale})
    assert (param.data_ptr(), scale.data_ptr()) != old
    assert [c[1] for c in layer.calls] == [param.data_ptr(), scale.data_ptr()]
    assert torch.equal(param, torch.arange(24.0).reshape(4, 6))
    assert torch.equal(scale, torch.arange(4.0))
    for local in range(4):
        assert layer.slots[(local, "w")].data_ptr() == param[local].data_ptr()
        assert layer.slots[(local, "s")].data_ptr() == scale[local].data_ptr()
    assert len(layer.slots) == 8

    # The loader may run a target's post-load hook twice: the slots stay as registered.
    calls, ptrs = len(layer.calls), (param.data_ptr(), scale.data_ptr())
    eplb.bind_slots({"w": param, "s": scale})
    assert len(layer.calls) == calls and (param.data_ptr(), scale.data_ptr()) == ptrs
    with pytest.raises(AssertionError, match="moved after the balancer registered them"):
        eplb.bind_slots({"w": param, "s": torch.arange(4.0)})


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


# --- a real online balancer moves experts into slots (GPU) -----------------------------------


@requires_host_accessible
def test_an_online_balancer_moves_experts_into_slots() -> None:
    """Rank 0 of two under online EPLB, through `EplbLayer` alone: host copies, binding, one
    forward whose every token picks experts 6 and 7, then a forward without updates, which
    waits for the move.

    The default placement holds 6 and 7 only on rank 1. After the move rank 0 holds some of
    their replicas, every one of its slots holds exactly the host copy of the expert the
    balancer placed there -- the bytes and the scalar -- and the route sends each id to a
    slot that holds it.
    """
    experts, ep_size, top_k = 8, 2, 2
    config = MoeLoadBalancerConfig(num_slots=12, layer_updates_per_iter=1)
    slots = EplbSlots.from_config(config, 3, experts, ep_size, 0)

    def expert_weight(name: str, expert: int) -> torch.Tensor:
        if name == "w":
            return torch.full((4, 16), 10 + expert, dtype=torch.uint8, device="cuda")
        return torch.tensor(0.5 + expert, dtype=torch.float32, device="cuda")

    weights = {
        "w": torch.stack([expert_weight("w", e) for e in slots.slot_experts]).contiguous(),
        "s": torch.stack([expert_weight("s", e) for e in slots.slot_experts]).contiguous(),
    }
    balancer = MoeLoadBalancer(0, ep_size, 1, shared_memory_base_name=f"mv2_eplb_{os.getpid()}")
    try:
        with balancer:
            eplb = EplbLayer.register(slots, top_k)
        assert eplb.online and eplb.host_experts == tuple(range(experts))
        eplb.share_experts(weights, expert_weight)
        eplb.bind_slots(weights)
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()

        hot = torch.tensor([[6, 7]] * 64, dtype=torch.int32, device="cuda")
        for updates in (True, False):
            balancer.set_iter_info(enable_statistic=True, enable_update_weights=updates)
            with MoeLoadBalancerIterContext(balancer):
                routed = eplb.route(hot, last=True)
                eplb.close()
        torch.cuda.synchronize()

        placement = eplb.balancer_layer.get_old_rank_expert_ids()
        mine = list(placement[0])
        assert {6, 7} & set(mine), placement
        for local, expert in enumerate(mine):
            assert torch.equal(weights["w"][local], expert_weight("w", expert)), (local, expert)
            assert torch.equal(weights["s"][local], expert_weight("s", expert)), (local, expert)
        slot_expert = torch.tensor(sum(placement, []), dtype=torch.int32, device="cuda")
        assert torch.equal(slot_expert[routed.long()], hot)
    finally:
        balancer.shutdown()
