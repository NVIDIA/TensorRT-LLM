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
bit in that rank's READY cell on every rank. The residency cases put a
full-device kernel that spins on the READY row, like MegaMoE's gate, on MAIN
right after the route wait, while the relay is starved of SMs, and expect it
to pass its gate in eager mode and under graph replay.

``test_halo_r_relay_pdl_timing`` is a benchmark, skipped unless
HALO_R_PDL_TIMING=1: it stamps the scheduler and relay of every layer call
with the timing probe, with the relay's programmatic launch on and off, and
asserts only correctness.
"""

import ctypes
import faulthandler
import json
import os
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
# The READY-gate stand-in gives up after this long; the relay finishes in milliseconds.
_GATE_TIMEOUT_NS = 5_000_000_000
# The SM hog outlives the scheduler, so the relay is pending when its SMs free up.
_HOG_NS = 300_000_000


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


def _routes(
    step: int, rank: int, num_tokens: int, experts: int, device: int, topk: int = _TOPK
) -> torch.Tensor:
    """Skewed routes: most tokens pick one hot expert per step, distinct experts per token."""
    rng = random.Random(1000 * step + rank)
    hot = [(5 * step) % experts, (5 * step + 9) % experts]
    rows = []
    for _ in range(num_tokens):
        choice = [hot[0]] if rng.random() < 0.6 else []
        if rng.random() < 0.3 and hot[1] not in choice:
            choice.append(hot[1])
        while len(choice) < topk:
            expert = rng.randrange(experts)
            if expert not in choice:
                choice.append(expert)
        rows.append(choice)
    return torch.tensor(rows, dtype=torch.int32, device=f"cuda:{device}").reshape(-1, topk)


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
    return hasattr(trtllm_ops, "moe_rebalance_halo_r_create") and hasattr(
        trtllm_ops, "moe_rebalance_halo_r_submit"
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


_GATE_SOURCE = r"""
__device__ unsigned long long globalTimer()
{
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// Holds its SM for `ns` nanoseconds.
extern "C" __global__ void smHog(unsigned long long ns)
{
    extern __shared__ unsigned char smem[];
    if (threadIdx.x == 0)
    {
        smem[0] = 0;
        unsigned long long const start = globalTimer();
        while (globalTimer() - start < ns)
        {
            __nanosleep(1000);
        }
    }
}

// MegaMoE's helper-weight READY gate: hold the SM until every cell of this rank's READY row carries a
// generation or the error bit. Gives up after `timeoutNs` and counts the CTA in `timedOut`.
extern "C" __global__ void readyGate(
    unsigned long long const* ready, int cells, unsigned long long timeoutNs, int* timedOut)
{
    extern __shared__ unsigned char smem[];
    if (threadIdx.x != 0)
    {
        return;
    }
    smem[0] = 0;
    unsigned long long const start = globalTimer();
    for (;;)
    {
        bool open = true;
        for (int c = 0; c < cells; ++c)
        {
            unsigned long long value;
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(ready + c) : "memory");
            open = open && value != 0;
        }
        if (open)
        {
            return;
        }
        if (globalTimer() - start > timeoutNs)
        {
            atomicAdd(timedOut, 1);
            return;
        }
        __nanosleep(256);
    }
}
"""


class _GateKernels:
    """A READY-gate stand-in on every SM and an SM hog, both one CTA per SM.

    Each CTA takes over half of an SM's shared memory, so neither shares an SM
    with the other or with a relay CTA, which takes nearly all of it.
    """

    def __init__(self, device: int) -> None:
        try:
            from cuda.bindings import driver, nvrtc
        except ImportError:
            from cuda import cuda as driver
            from cuda import nvrtc
        self.driver = driver

        def check(result):
            error, *values = result if isinstance(result, tuple) else (result,)
            if error not in (driver.CUresult.CUDA_SUCCESS, nvrtc.nvrtcResult.NVRTC_SUCCESS):
                raise RuntimeError(f"CUDA error {error}")
            return values[0] if len(values) == 1 else values

        self.check = check
        major, minor = torch.cuda.get_device_capability(device)
        program = check(
            nvrtc.nvrtcCreateProgram(_GATE_SOURCE.encode(), b"ready_gate.cu", 0, [], [])
        )
        options = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        if (
            nvrtc.nvrtcCompileProgram(program, len(options), options)[0]
            != nvrtc.nvrtcResult.NVRTC_SUCCESS
        ):
            log = b" " * check(nvrtc.nvrtcGetProgramLogSize(program))
            check(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(f"NVRTC compile failed: {log.decode(errors='replace')}")
        cubin = b" " * check(nvrtc.nvrtcGetCUBINSize(program))
        check(nvrtc.nvrtcGetCUBIN(program, cubin))
        self.module = check(driver.cuModuleLoadData(cubin))
        self.gate = check(driver.cuModuleGetFunction(self.module, b"readyGate"))
        self.hog = check(driver.cuModuleGetFunction(self.module, b"smHog"))
        cu_device = check(driver.cuDeviceGet(device))
        per_sm = check(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                cu_device,
            )
        )
        self.shared_bytes = per_sm // 2 + 1024
        for function in (self.gate, self.hog):
            check(
                driver.cuFuncSetAttribute(
                    function,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    self.shared_bytes,
                )
            )
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.timed_out = torch.zeros(1, dtype=torch.int32, device=f"cuda:{device}")

    def _launch(self, function, blocks: int, args, stream: torch.cuda.Stream) -> None:
        self.check(
            self.driver.cuLaunchKernel(
                function,
                blocks,
                1,
                1,
                128,
                1,
                1,
                self.shared_bytes,
                self.driver.CUstream(stream.cuda_stream),
                args,
                0,
            )
        )

    def launch_hog(self, stream: torch.cuda.Stream) -> None:
        # Leave only the scheduler's SMs free: fewer than the relay's CTAs.
        args = ((_HOG_NS,), (ctypes.c_ulonglong,))
        self._launch(self.hog, self.sms - _AUXILIARY_SMS, args, stream)

    def launch_gate(self, ready: torch.Tensor, stream: torch.cuda.Stream) -> None:
        args = (
            (ready.data_ptr(), ready.numel(), _GATE_TIMEOUT_NS, self.timed_out.data_ptr()),
            (ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_void_p),
        )
        self._launch(self.gate, self.sms, args, stream)


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

    def __init__(
        self, rank: int, count: int, hidden: int = _HIDDEN, intermediate: int = _INTERMEDIATE
    ) -> None:
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
        bundle = BundleLayout.create(hidden=hidden, intermediate=intermediate)
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

    def build_groups(self, max_tokens: int = _MAX_TOKENS, topk: int = _TOPK) -> None:
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
                    topk=topk,
                    max_tokens_per_rank=max_tokens,
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


def _starve_relay(layers: _Layers) -> None:
    """Give COPY the lowest stream priority, below the MAIN of the residency cases.

    Production gives COPY the highest priority, which only orders CTAs that
    are still pending; inverting it makes MegaMoE win every SM that frees up
    once the relay is pending. Must run before the groups are built.
    """
    least, _ = torch.cuda.Stream.priority_range()
    copy_stream = torch.cuda.Stream(device=layers.rank, priority=least)
    layers.manager.get_copy_stream = lambda device: copy_stream


def _case_relay_residency(layers: _Layers, use_graph: bool) -> list:
    """MegaMoE's full grid, queued right after the route wait, must not starve the relay.

    A hog holds every SM but the scheduler's, so the relay is still short of
    SMs when the hog retires; with COPY below MAIN, a gate kernel that MAIN
    has already released takes those SMs first. The calls use the native ops
    directly: the group's Python check rejects a MAIN above COPY.
    """
    rank, experts = layers.rank, layers.experts
    group, arena = layers.groups[0], layers.arenas[0]
    ops = getattr(getattr(torch, "ops"), "trtllm")
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.halo_r_planner import (
        HALO_R_SPIN_BUDGET_NS,
    )

    kernels = _GateKernels(rank)
    ready = arena.terminal_flags_tensor().view(torch.int64)
    _, greatest = torch.cuda.Stream.priority_range()
    main = torch.cuda.Stream(device=rank, priority=greatest)
    side = torch.cuda.Stream(device=rank)
    routes = _routes(0, rank, _MAX_TOKENS, experts, rank)

    def layer_call() -> None:
        ops.moe_rebalance_halo_r_submit(group._handle, routes, HALO_R_SPIN_BUDGET_NS)
        ops.moe_rebalance_halo_r_wait_routes(group._handle)
        kernels.launch_gate(ready, torch.cuda.current_stream())
        ops.moe_rebalance_halo_r_release(group._handle, 0)

    graph = None
    if use_graph:
        main.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main):
            layer_call()
    failures = []
    for step in range(1, 4):
        routes.copy_(_routes(step, rank, _MAX_TOKENS, experts, rank))
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        kernels.launch_hog(side)
        with torch.cuda.stream(main):
            if graph is not None:
                graph.replay()
            else:
                layer_call()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        layers.epochs[0] += 1
        timed_out = int(kernels.timed_out.item())
        if timed_out:
            failures.append(f"rank {rank}: step {step}: {timed_out} gate CTAs timed out on READY")
            kernels.timed_out.zero_()
        failures += _check_call(group, arena, routes, layers.epochs[0], rank)
    if graph is not None:
        graph.reset()
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
        if case.startswith("residency"):
            _starve_relay(layers)
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
        if case.startswith("residency"):
            return None, _case_relay_residency(layers, use_graph=case == "residency_cuda_graph")
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
@pytest.mark.parametrize(
    "case",
    ["eager", "cuda_graph", "banks", "error_bit", "residency_eager", "residency_cuda_graph"],
)
def test_halo_r_relay_matches_reference(halo_r_mpi_pool, case):
    results = list(halo_r_mpi_pool.map(_run_worker, [case] * _EP_SIZE))
    skips = {skip for skip, _ in results if skip}
    if skips:
        pytest.skip("; ".join(sorted(skips)))
    failures = [failure for _, rank_failures in results for failure in rank_failures]
    assert not failures, "\n".join(failures[:20])


# Timing benchmark of the relay's programmatic dependent launch (PDL). Production
# shape where the harness allows it: 2048 tokens per rank, top-8, the default
# NVFP4 bundle; routes are the skewed synthetic `_routes`, not replayed slabs.
_TIMING_TOKENS = 2048
_TIMING_TOPK = 8
_TIMING_LAYERS = int(os.environ.get("HALO_R_PDL_TIMING_LAYERS", "4"))
_TIMING_ITERATIONS = int(os.environ.get("HALO_R_PDL_TIMING_ITERATIONS", "20"))
# MoeRebalanceHaloRProbeWord.
_PROBE_WORDS = 8
_PROBE_SCHEDULER_START = 0
_PROBE_PLAN_PUBLISHED = 1
_PROBE_SCHEDULER_END = 2
_PROBE_RELAY_START = 3
_PROBE_RELAY_END = 4
_PROBE_RELAY_FALLBACKS = 5
_UNSET = (1 << 64) - 1


def _quantiles(values: list) -> dict:
    """Median and nearest-rank p90 in microseconds."""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    p90 = ordered[max(0, -(-9 * count // 10) - 1)]
    return {"median_us": round(median / 1000, 3), "p90_us": round(p90 / 1000, 3)}


def _timing_summary(records: list) -> dict:
    """Probe records (stamps in ns) to the four intervals of PDL_NOTES.md."""
    plan, tail, start, total = [], [], [], []
    for r in records:
        plan.append(r[_PROBE_PLAN_PUBLISHED] - r[_PROBE_SCHEDULER_START])
        tail.append(r[_PROBE_SCHEDULER_END] - r[_PROBE_PLAN_PUBLISHED])
        start.append(r[_PROBE_RELAY_START] - r[_PROBE_SCHEDULER_END])
        total.append(r[_PROBE_RELAY_END] - r[_PROBE_SCHEDULER_START])
    return {
        "calls": len(records),
        "plan_b_minus_a": _quantiles(plan),
        "route_tail_c_minus_b": _quantiles(tail),
        "relay_start_vs_scheduler_end_d_minus_c": _quantiles(start),
        "end_to_end_e_minus_a": _quantiles(total),
        "relay_plan_fallback_ctas": sum(r[_PROBE_RELAY_FALLBACKS] for r in records),
    }


def _probe_records(probe: torch.Tensor, indices: list, rank: int, layer: int, failures: list):
    rows = probe.cpu().tolist()
    records = []
    for index in indices:
        record = [value & _UNSET for value in rows[index]]
        if _UNSET in (record[_PROBE_SCHEDULER_START], record[_PROBE_RELAY_START]) or not all(
            record[_PROBE_PLAN_PUBLISHED : _PROBE_RELAY_END + 1]
        ):
            failures.append(f"rank {rank}: layer {layer} launch {index}: probe record incomplete")
            continue
        records.append(record)
    return records


def _run_timing_worker(pdl: bool):
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True)
    rank = tllm.mpi_rank()
    layers = None
    try:
        assert tllm.mpi_world_size() == _EP_SIZE
        torch.cuda.set_device(rank)
        reason = _skip_reason()
        if reason is None and not hasattr(
            getattr(getattr(torch, "ops"), "trtllm"), "moe_rebalance_halo_r_set_probe"
        ):
            reason = "HALO-R timing probe is not built"
        reasons = MPI.COMM_WORLD.allgather(reason)
        if any(reasons):
            return "; ".join(sorted({r for r in reasons if r})), [], {}
        ops = getattr(getattr(torch, "ops"), "trtllm")
        # The native context reads the switch when it is bound.
        os.environ["TRTLLM_HALO_R_RELAY_PDL"] = "1" if pdl else "0"
        _init_process_group(rank)
        hidden = int(os.environ.get("HALO_R_PDL_TIMING_HIDDEN", "7168"))
        intermediate = int(os.environ.get("HALO_R_PDL_TIMING_INTERMEDIATE", "3072"))
        layers = _Layers(rank, _TIMING_LAYERS, hidden=hidden, intermediate=intermediate)
        try:
            layers.build_groups(max_tokens=_TIMING_TOKENS, topk=_TIMING_TOPK)
        except RuntimeError as error:
            if _PLACEHOLDER_MESSAGE in str(error):
                return f"placeholder build: {error}", [], {}
            raise
        experts = layers.experts
        records = 2 * _TIMING_ITERATIONS + 8
        probes = []
        for group in layers.groups:
            probe = torch.zeros((records, _PROBE_WORDS), dtype=torch.int64, device=f"cuda:{rank}")
            probe[:, _PROBE_SCHEDULER_START] = -1
            probe[:, _PROBE_RELAY_START] = -1
            ops.moe_rebalance_halo_r_set_probe(group._handle, probe)
            probes.append(probe)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()

        failures = []
        launches = {"eager": [[] for _ in layers.groups], "cuda_graph": [[] for _ in layers.groups]}
        main = torch.cuda.Stream()

        def check(routes: list) -> None:
            for layer, (group, arena) in enumerate(zip(layers.groups, layers.arenas)):
                failures.extend(
                    _check_call(group, arena, routes[layer], layers.epochs[layer], rank, layer)
                )

        # Eager: the layers back to back on MAIN, one host sync per iteration.
        for iteration in range(_TIMING_ITERATIONS):
            routes = [
                _routes(100 * iteration + layer, rank, _TIMING_TOKENS, experts, rank, _TIMING_TOPK)
                for layer in range(len(layers.groups))
            ]
            torch.cuda.synchronize()
            MPI.COMM_WORLD.Barrier()
            with torch.cuda.stream(main):
                for layer, group in enumerate(layers.groups):
                    launches["eager"][layer].append(layers.epochs[layer])
                    group.plan(routes[layer], defer_wait=False)
                    group.finish()
                    layers.epochs[layer] += 1
            torch.cuda.synchronize()
            MPI.COMM_WORLD.Barrier()
            check(routes)

        # CUDA graph: every layer in one captured graph, replayed.
        static_routes = [
            _routes(layer, rank, _TIMING_TOKENS, experts, rank, _TIMING_TOPK)
            for layer in range(len(layers.groups))
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer, group in enumerate(layers.groups):
                group.plan(static_routes[layer], defer_wait=False)
                group.finish()
        for iteration in range(_TIMING_ITERATIONS):
            for layer, routes in enumerate(static_routes):
                routes.copy_(
                    _routes(
                        100 * (iteration + _TIMING_ITERATIONS) + layer,
                        rank,
                        _TIMING_TOKENS,
                        experts,
                        rank,
                        _TIMING_TOPK,
                    )
                )
                launches["cuda_graph"][layer].append(layers.epochs[layer])
            torch.cuda.synchronize()
            MPI.COMM_WORLD.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            MPI.COMM_WORLD.Barrier()
            for layer in range(len(layers.groups)):
                layers.epochs[layer] += 1
            check(static_routes)
        graph.reset()

        records = {}
        for mode, per_layer in launches.items():
            records[mode] = []
            for layer, indices in enumerate(per_layer):
                records[mode] += _probe_records(probes[layer], indices, rank, layer, failures)
        for group in layers.groups:
            ops.moe_rebalance_halo_r_set_probe(group._handle, None)
        return None, failures, records
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
@pytest.mark.skipif(
    os.environ.get("HALO_R_PDL_TIMING") != "1", reason="benchmark; set HALO_R_PDL_TIMING=1"
)
@pytest.mark.parametrize("pdl", [True, False], ids=["pdl_on", "pdl_off"])
def test_halo_r_relay_pdl_timing(halo_r_mpi_pool, pdl):
    """Stamps every layer call; asserts READY, the bitwise relay and completion, never timing.

    Prints one summary line per (pdl, mode) and merges the summary into the
    JSON file named by HALO_R_PDL_TIMING_JSON, if set.
    """
    results = list(halo_r_mpi_pool.map(_run_timing_worker, [pdl] * _EP_SIZE))
    skips = {skip for skip, _, _ in results if skip}
    if skips:
        pytest.skip("; ".join(sorted(skips)))
    failures = [failure for _, rank_failures, _ in results for failure in rank_failures]
    assert not failures, "\n".join(failures[:20])

    key = "pdl_on" if pdl else "pdl_off"
    summary = {}
    for mode in ("eager", "cuda_graph"):
        records = [record for _, _, per_mode in results for record in per_mode[mode]]
        summary[mode] = _timing_summary(records)
        print(f"HALO-R relay timing {key} {mode}: {json.dumps(summary[mode], sort_keys=True)}")
    path = os.environ.get("HALO_R_PDL_TIMING_JSON")
    if path:
        merged = {}
        if os.path.exists(path):
            with open(path) as handle:
                merged = json.load(handle)
        merged[key] = {
            "routing": "skewed synthetic (_routes), not replayed slabs",
            "ep": _EP_SIZE,
            "tokens_per_rank": _TIMING_TOKENS,
            "topk": _TIMING_TOPK,
            "layers": _TIMING_LAYERS,
            "iterations": _TIMING_ITERATIONS,
            "hidden": int(os.environ.get("HALO_R_PDL_TIMING_HIDDEN", "7168")),
            "intermediate": int(os.environ.get("HALO_R_PDL_TIMING_INTERMEDIATE", "3072")),
            **summary,
        }
        with open(path, "w") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
