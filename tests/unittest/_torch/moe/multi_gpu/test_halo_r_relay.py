# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank HALO-R scheduler and relay copy on a real shared-slot arena.

Every rank fills its home experts with a byte pattern unique to (expert,
plane, layer) and runs HALO-R layer calls: eagerly, from a captured CUDA
graph, and over three layers that alternate the two shared helper banks.
After each call it checks:
- the plan words equal the pure-Python reference plan of the global routes;
- every route lands on its owner's home slot or a receiver's helper slot, and
  the routes per (expert, rank) add up to the plan's quota;
- every helper row this rank received equals the owner's expert bytewise;
- the READY row holds the current epoch on every cell, without the error bit,
  and every status word is clear.
A negative case records a scheduler failure on one rank and expects the error
bit in that rank's READY cell on every rank.
"""

import faulthandler
import pickle
import random
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from _torch.moe import halo_r_reference
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm as tllm
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(halo_r_reference)
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_EP_SIZE = 4
_HOME_EXPERTS = 8
_HELPER_SLOTS = 2
_TOPK = 4
_MAX_TOKENS = 256
_AUXILIARY_SMS = 8
# Small planes stay in the per-layer record pages; the weight planes exceed
# them and live in the shared helper banks, so both relay paths are covered.
_HIDDEN = 256
_INTERMEDIATE = 128
_WORKER_TIMEOUT_S = 300


@pytest.fixture
def halo_r_mpi_pool(monkeypatch):
    # Per-layer dynamic EPLB needs the larger launch queues from CUDA initialization on.
    monkeypatch.setenv("CUDA_SCALE_LAUNCH_QUEUES", "4x")
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S + 60, exit=True)
    try:
        with MPIPoolExecutor(_EP_SIZE) as executor:
            yield executor
    finally:
        faulthandler.cancel_dump_traceback_later()


class _RawBytes:
    """A uint8 CUDA view of device memory owned elsewhere."""

    def __init__(self, address: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "data": (int(address), False),
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


def _row(address: int, nbytes: int, device: int) -> torch.Tensor:
    return torch.as_tensor(_RawBytes(address, nbytes), device=f"cuda:{device}")


def _pattern(expert: int, plane: int, layer: int, nbytes: int, device: int) -> torch.Tensor:
    """Bytes unique to (expert, plane, layer)."""
    index = torch.arange(nbytes, dtype=torch.int64, device=f"cuda:{device}")
    return ((index * 131 + expert * 7919 + plane * 104729 + layer * 15485863) % 251).to(torch.uint8)


def _routes(step: int, rank: int, num_tokens: int, experts: int, device: int) -> torch.Tensor:
    """Skewed routes: most tokens pick one hot expert per step, distinct experts per token."""
    rng = random.Random(1000 * step + rank)
    hot = [(5 * step) % experts, (5 * step + 9) % experts]
    rows = []
    for _ in range(num_tokens):
        choice = [hot[0]] if rng.random() < 0.6 else []
        if rng.random() < 0.3 and hot[1] not in choice:
            choice.append(hot[1])
        while len(choice) < _TOPK:
            expert = rng.randrange(experts)
            if expert not in choice:
                choice.append(expert)
        rows.append(choice)
    return torch.tensor(rows, dtype=torch.int32, device=f"cuda:{device}").reshape(-1, _TOPK)


def _init_process_group(rank: int) -> None:
    store = None
    if rank == 0:
        store = dist.TCPStore("127.0.0.1", 0, _EP_SIZE, is_master=True, wait_for_workers=False)
    port = MPI.COMM_WORLD.bcast(store.port if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore("127.0.0.1", port, _EP_SIZE, is_master=False, wait_for_workers=False)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=_EP_SIZE)


def _halo_r_ops_available() -> bool:
    # Resolve the op namespace by name inside the worker. This module is pickled
    # by value into the MPI workers, and a direct `torch.ops` reference makes
    # cloudpickle try to serialize torch's op-namespace object.
    trtllm_ops = getattr(getattr(torch, "ops"), "trtllm")
    return hasattr(trtllm_ops, "moe_rebalance_halo_r") and hasattr(
        trtllm_ops, "moe_rebalance_relay"
    )


def _skip_reason() -> str | None:
    if torch.cuda.get_device_capability()[0] != 10:
        return "HALO-R requires an SM100-family GPU"
    if not _halo_r_ops_available():
        return "HALO-R ops are not built"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "HALO-R requires an NVLink fabric with multicast"
    return None


def _check_call(group, arena, routes: torch.Tensor, epoch: int, rank: int, layer: int = 0) -> list:
    """Compare one finished layer call with the reference; returns failure strings."""
    failures = []
    experts = _HOME_EXPERTS * _EP_SIZE
    local_slots = _HOME_EXPERTS + _HELPER_SLOTS
    counts = torch.bincount(routes.flatten().long(), minlength=experts).cpu().tolist()
    histogram = [sum(c) for c in zip(*MPI.COMM_WORLD.allgather(counts))]
    plan = halo_r_reference.plan_halo_r(
        histogram, _EP_SIZE, _HELPER_SLOTS, send_count=4, send_weight=0
    )

    words = group.plan_words.cpu().tolist()
    if words != plan.plan_words(epoch=epoch):
        failures.append(f"rank {rank}: plan words differ from the reference at epoch {epoch}")
    if group.hot_expert_ids.cpu().tolist() != plan.helper_table[rank]:
        failures.append(f"rank {rank}: helper experts differ from the reference")
    if any(group.status.cpu().tolist()):
        failures.append(f"rank {rank}: status words {group.status.cpu().tolist()}")
    # A cell with the error bit set reads as a negative int64 and never matches.
    ready = arena.terminal_flags_tensor().view(torch.int64).cpu().tolist()
    if ready != [epoch] * _EP_SIZE:
        failures.append(f"rank {rank}: READY row {ready}, expected epoch {epoch}")

    # Every route goes to its owner or to a receiver's helper slot of its expert.
    slots = group.physical_slot_ids[: routes.shape[0]].cpu().flatten().tolist()
    moved = {}
    for expert, slot in zip(routes.flatten().cpu().tolist(), slots):
        destination, local = divmod(slot, local_slots)
        if local < _HOME_EXPERTS:
            if destination * _HOME_EXPERTS + local != expert:
                failures.append(f"rank {rank}: expert {expert} routed to home slot {slot}")
        elif plan.helper_table[destination][local - _HOME_EXPERTS] != expert:
            failures.append(f"rank {rank}: expert {expert} routed to helper slot {slot}")
        else:
            moved[(expert, destination)] = moved.get((expert, destination), 0) + 1
    totals = {}
    for part in MPI.COMM_WORLD.allgather(moved):
        for key, value in part.items():
            totals[key] = totals.get(key, 0) + value
    expected = {(b.expert, rank_): rows for b in plan.broadcasts for rank_, rows in b.rows.items()}
    if totals != expected:
        failures.append(f"rank {rank}: moved routes {totals} != plan {expected}")

    # The relay delivered every received expert byte for byte.
    bound = arena.bound.base
    for helper, expert in enumerate(plan.helper_table[rank]):
        if expert < 0:
            continue
        for plane, nbytes in enumerate(bound.plane_bytes):
            address = int(bound.local_plane_views[plane].data_ptr()) + (
                _HOME_EXPERTS + helper
            ) * int(nbytes)
            got = _row(address, int(nbytes), rank)
            if not torch.equal(got, _pattern(expert, plane, layer, int(nbytes), rank)):
                failures.append(
                    f"rank {rank}: layer {layer} helper {helper} plane {plane} != expert {expert}"
                )
    return failures


# A build without the SM100 + CUDA 13.1 kernels registers the ops but throws on use.
_PLACEHOLDER_MESSAGE = "HALO-R requires CUDA 13.1+"


class _Layers:
    """HALO-R groups of consecutive MoE layers on one model-level manager."""

    def __init__(self, rank: int, count: int) -> None:
        from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
            PerLayerDynamicMoeLoadBalancer,
        )
        from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.arena import (
            allocate_rebalance_arena,
        )
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

        self.rank = rank
        self.experts = _HOME_EXPERTS * _EP_SIZE
        config = SimpleNamespace(
            num_slots=self.experts + _EP_SIZE * _HELPER_SLOTS,
            auxiliary_sms=_AUXILIARY_SMS,
            helper_slot_mode="shared_banks",
            planner="halo_r",
        )
        self.manager = PerLayerDynamicMoeLoadBalancer(ep_rank=rank, ep_size=_EP_SIZE, config=config)
        self.mapping = SimpleNamespace(moe_ep_size=_EP_SIZE, moe_ep_rank=rank, local_rank=rank)
        self.process_group = dist.group.WORLD
        bundle = BundleLayout.create(hidden=_HIDDEN, intermediate=_INTERMEDIATE)
        self.arenas = []
        for layer in range(count):
            arena = allocate_rebalance_arena(
                home_experts=_HOME_EXPERTS,
                helper_slots=_HELPER_SLOTS,
                bundle=bundle,
                ep_comm=self.manager.get_ep_comm(self.process_group),
                mapping=self.mapping,
                provider=self.manager.create_arena_provider(),
                device=rank,
                layer_idx=layer,
            )
            bound = arena.bound.base
            for plane, nbytes in enumerate(bound.plane_bytes):
                base = int(bound.local_plane_views[plane].data_ptr())
                for local in range(_HOME_EXPERTS):
                    _row(base + local * int(nbytes), int(nbytes), rank).copy_(
                        _pattern(rank * _HOME_EXPERTS + local, plane, layer, int(nbytes), rank)
                    )
            self.arenas.append(arena)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        self.groups = []
        self.epochs = []

    def build_groups(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.halo_r_planner import (
            build_halo_r_slot_scheduler_group,
        )

        for layer, arena in enumerate(self.arenas):
            self.groups.append(
                build_halo_r_slot_scheduler_group(
                    arena=arena,
                    manager=self.manager,
                    mapping=self.mapping,
                    ep_process_group=self.process_group,
                    home_experts=_HOME_EXPERTS,
                    helper_slots=_HELPER_SLOTS,
                    topk=_TOPK,
                    max_tokens_per_rank=_MAX_TOKENS,
                    layer_idx=layer,
                )
            )
            # The constructor's warmup is epoch 1 of each layer.
            self.epochs.append(1)

    def call(self, layer: int, routes: torch.Tensor, stream: torch.cuda.Stream) -> int:
        group = self.groups[layer]
        with torch.cuda.stream(stream):
            _, generation = group.plan(routes, defer_wait=False)
            assert generation == 1
            group.finish()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        self.epochs[layer] += 1
        return self.epochs[layer]

    def shutdown(self) -> None:
        torch.cuda.synchronize()
        self.manager.shutdown(local_safe=True)


def _case_single_layer(layers: _Layers, use_graph: bool) -> list:
    rank, experts = layers.rank, layers.experts
    group, arena = layers.groups[0], layers.arenas[0]
    main = torch.cuda.Stream()
    failures = []
    for step in range(2):
        # Ranks submit different token counts; the last rank has none on step 1.
        tokens = 0 if step == 1 and rank == _EP_SIZE - 1 else _MAX_TOKENS - 17 * rank
        routes = _routes(step, rank, tokens, experts, rank)
        epoch = layers.call(0, routes, main)
        failures += _check_call(group, arena, routes, epoch, rank)
    if not use_graph:
        return failures
    static_routes = _routes(2, rank, _MAX_TOKENS, experts, rank)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        group.plan(static_routes, defer_wait=False)
        group.finish()
    for step in range(3, 6):
        static_routes.copy_(_routes(step, rank, _MAX_TOKENS, experts, rank))
        graph.replay()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        layers.epochs[0] += 1
        failures += _check_call(group, arena, static_routes, layers.epochs[0], rank)
    # Eager calls and replays share the device epoch.
    routes = _routes(6, rank, _MAX_TOKENS // 2, experts, rank)
    epoch = layers.call(0, routes, main)
    failures += _check_call(group, arena, routes, epoch, rank)
    graph.reset()
    return failures


def _case_alternating_banks(layers: _Layers) -> list:
    """Three layers on two shared helper banks: layer 2 refills the bank of layer 0."""
    rank, experts = layers.rank, layers.experts
    banks = [int(arena.provider.layer.helper_set) for arena in layers.arenas]
    failures = [] if banks == [0, 1, 0] else [f"rank {rank}: helper banks {banks}"]
    main = torch.cuda.Stream()
    for step in range(2):
        for layer, (group, arena) in enumerate(zip(layers.groups, layers.arenas)):
            routes = _routes(10 * step + layer, rank, _MAX_TOKENS, experts, rank)
            epoch = layers.call(layer, routes, main)
            failures += _check_call(group, arena, routes, epoch, rank, layer=layer)
    return failures


def _case_error_bit(layers: _Layers) -> list:
    """A recorded scheduler failure on one rank sets the error bit of that rank's READY cell everywhere."""
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.halo_r_planner import (
        HALO_R_STATUS_GRID_TIMEOUT,
    )

    rank, experts = layers.rank, layers.experts
    failed_rank = 1
    main = torch.cuda.Stream()
    routes = _routes(0, rank, _MAX_TOKENS, experts, rank)
    failures = _check_call(
        layers.groups[0], layers.arenas[0], routes, layers.call(0, routes, main), rank
    )
    if rank == failed_rank:
        layers.groups[0].status[HALO_R_STATUS_GRID_TIMEOUT] = 7
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    epoch = layers.call(0, _routes(1, rank, _MAX_TOKENS, experts, rank), main)
    ready = layers.arenas[0].terminal_flags_tensor().view(torch.int64).cpu().tolist()
    for source, value in enumerate(value & ((1 << 64) - 1) for value in ready):
        error = bool(value >> 63)
        if error != (source == failed_rank) or (value & ((1 << 63) - 1)) != epoch:
            failures.append(f"rank {rank}: READY cell {source} = {value:#x} at epoch {epoch}")
    return failures


def _run_worker(case: str):
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True)
    rank = tllm.mpi_rank()
    layers = None
    try:
        assert tllm.mpi_world_size() == _EP_SIZE
        torch.cuda.set_device(rank)
        reasons = MPI.COMM_WORLD.allgather(_skip_reason())
        if any(reasons):
            return "; ".join(sorted({r for r in reasons if r})), []
        _init_process_group(rank)
        layers = _Layers(rank, 3 if case == "banks" else 1)
        try:
            layers.build_groups()
        except RuntimeError as error:
            # Every rank reaches the first kernel launch together, so every rank skips.
            if _PLACEHOLDER_MESSAGE in str(error):
                return f"placeholder build: {error}", []
            raise
        if case == "banks":
            return None, _case_alternating_banks(layers)
        if case == "error_bit":
            return None, _case_error_bit(layers)
        return None, _case_single_layer(layers, use_graph=case == "cuda_graph")
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
    finally:
        if layers is not None:
            layers.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.skipif(torch.cuda.device_count() < _EP_SIZE, reason="Requires 4 GPUs")
@pytest.mark.parametrize("case", ["eager", "cuda_graph", "banks", "error_bit"])
def test_halo_r_relay_matches_reference(halo_r_mpi_pool, case):
    results = list(halo_r_mpi_pool.map(_run_worker, [case] * _EP_SIZE))
    skips = {skip for skip, _ in results if skip}
    if skips:
        pytest.skip("; ".join(sorted(skips)))
    failures = [failure for _, rank_failures in results for failure in rank_failures]
    assert not failures, "\n".join(failures[:20])
