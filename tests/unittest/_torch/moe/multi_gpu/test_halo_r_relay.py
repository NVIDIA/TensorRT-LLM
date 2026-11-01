# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HALO-R kernel tests on a real shared-slot arena, one EP rank per GPU.

Every rank fills its home experts with a byte pattern unique to (expert,
plane, weights) and runs HALO-R layer calls through the scheduler group. After
each checked call it compares:
- the plan words with the pure-Python reference plan of the global routes;
- every route with its owner's home slot or a receiver's helper slot, and the
  routes per (expert, rank) with the plan's quota;
- every helper row this rank received with the owner's expert, bytewise;
- the READY row with the value the call lent to MegaMoE, and every status word
  with 0;
- the relay counters (launches, copied and skipped receivers) and, with
  per-layer helper slots, the residency table with a model of it, on every rank.

The test ids carry the TEST_PLAN ids: t1_2 (equivalence with commit 0: the
dump runs on a commit-0 build, the compare on later ones, through
HALO_R_EQUIV_DIR), t1_3 (reference and READY), t1_4 (zero-token ranks, empty
plan), t1_5 (residency and skipped copies), t1_6 (slot stability), t1_7 (the
READY gate under an SM hog), t1_8 (fail-closed: a status word traps the rank
before it releases routes, and its peers trap at their next exchange) and
t1_10 (breakable-prefill CUDA graphs).

The ranks run on one node through an MPIPoolExecutor, or on any number of
nodes under ``trtllm-llmapi-launch`` through the MPI session it serves.
HALO_R_TEST_EP sets the EP size (default 4) and HALO_R_TEST_SEEDS the number
of seeds per case (default 1). The cases need only the native ops of the build
they run on, so one file drives commit 0 (t1_2 dump only), the fused kernel
without CUDA graphs, and the current build.
"""

import ctypes
import faulthandler
import os
import pickle
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from _torch.moe import halo_r_reference
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(halo_r_reference)
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_EP_SIZE = int(os.environ.get("HALO_R_TEST_EP", "4"))
_SEEDS = int(os.environ.get("HALO_R_TEST_SEEDS", "1"))
_EQUIV_DIR = os.environ.get("HALO_R_EQUIV_DIR")
_HOME_EXPERTS = 8
_HELPER_SLOTS = 2
_TOPK = 4
_MAX_TOKENS = 256
_AUXILIARY_SMS = 4
# Small planes stay in the per-layer record pages; the weight planes exceed
# them and live in the helper sets, so both relay paths are covered.
_HIDDEN = 256
_INTERMEDIATE = 128
_WORKER_TIMEOUT_S = 600
# The READY-gate stand-in gives up after this long; the relay finishes in milliseconds.
_GATE_TIMEOUT_NS = 5_000_000_000
# The SM hog outlives the HALO-R kernel's scheduler role.
_HOG_NS = 300_000_000
# T1-2 inputs: every token count, both route kinds, three layers over the two shared banks.
_EQUIV_MAX_TOKENS = 2048
_EQUIV_TOKENS = (0, 1, 7, 128, 2048)
# T1-8: the spin bound of every launch after the injection, and the pass bounds.
_FAIL_TIMEOUT_S = 20
_FAIL_OWN_TRAP_S = 10
_FAIL_PEER_TRAP_S = 50


def _implementation() -> str:
    """The HALO-R native ops of this build: commit 0's, the fused kernel's, or the capturable ones."""
    # Resolve the op namespace by name: this module is pickled by value into the
    # workers, and a direct `torch.ops` reference would make cloudpickle try to
    # serialize torch's op-namespace object.
    ops = getattr(getattr(torch, "ops"), "trtllm")
    if hasattr(ops, "moe_rebalance_halo_r_join"):
        return "graphs"
    if hasattr(ops, "moe_rebalance_halo_r_submit_fused"):
        return "fused"
    if hasattr(ops, "moe_rebalance_halo_r_submit"):
        return "commit0"
    return "none"


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


def _uniform_routes(rank: int, num_tokens: int, experts: int, device: int) -> torch.Tensor:
    """Every expert gets the same routed-token count on every rank, so the plan moves nothing."""
    rows = [[(token * _TOPK + k) % experts for k in range(_TOPK)] for token in range(num_tokens)]
    return torch.tensor(rows, dtype=torch.int32, device=f"cuda:{device}").reshape(-1, _TOPK)


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

// MegaMoE's helper-weight READY gate: hold the SM until every cell of this rank's READY row reaches the
// expected generation. Gives up after `timeoutNs` and counts the CTA in `timedOut`.
extern "C" __global__ void readyGate(unsigned long long const* ready, int cells, unsigned long long expected,
    unsigned long long timeoutNs, int* timedOut)
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
            open = open && value >= expected;
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
    with the other or with a HALO-R CTA, which takes nearly all of it.
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

    def launch_hog(self, stream: torch.cuda.Stream, auxiliary_sms: int) -> None:
        # Leave only the HALO-R kernel's SMs free, as the shared expert's
        # reservation does.
        args = ((_HOG_NS,), (ctypes.c_ulonglong,))
        self._launch(self.hog, self.sms - auxiliary_sms, args, stream)

    def launch_gate(self, ready: torch.Tensor, generation: int, stream: torch.cuda.Stream) -> None:
        args = (
            (
                ready.data_ptr(),
                ready.numel(),
                generation,
                _GATE_TIMEOUT_NS,
                self.timed_out.data_ptr(),
            ),
            (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_void_p,
            ),
        )
        self._launch(self.gate, self.sms, args, stream)


def _init_process_group(rank: int, world: int) -> None:
    store = None
    if rank == 0:
        store = dist.TCPStore("0.0.0.0", 0, world, is_master=True, wait_for_workers=False)
    host = socket.gethostbyname(socket.gethostname())
    host, port = MPI.COMM_WORLD.bcast((host, store.port) if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore(host, port, world, is_master=False, wait_for_workers=False)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world)


def _skip_reason() -> str | None:
    if torch.cuda.get_device_capability()[0] != 10:
        return "HALO-R requires an SM100-family GPU"
    if _implementation() == "none":
        return "HALO-R ops are not built"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "HALO-R requires an NVLink fabric with multicast"
    return None


def _buffers(group) -> SimpleNamespace:
    """The scheduler outputs and workspace of a group: the CUDA scheduler's, or commit 0's own."""
    scheduler = getattr(group, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "plan_workspace"):
        return SimpleNamespace(
            plan=scheduler.plan_workspace,
            prefix=scheduler.route_prefix,
            slots=scheduler.outputs.physical_slot_ids,
            ids=scheduler.outputs.hot_expert_ids,
            owners=scheduler.outputs.hot_expert_source_ranks,
            status=scheduler.status,
        )
    return SimpleNamespace(
        plan=group.plan_words,
        prefix=group.route_prefix,
        slots=group.physical_slot_ids,
        ids=group.hot_expert_ids,
        owners=group.hot_expert_source_ranks,
        status=group.status,
    )


# A build without the SM100 + CUDA 13.1 kernels registers the ops but throws on use.
_PLACEHOLDER_MESSAGE = "HALO-R requires CUDA 13.1+"


class _Layers:
    """HALO-R groups of consecutive MoE layers on one model-level manager, and a model of their state."""

    def __init__(
        self,
        rank: int,
        world: int,
        device: int,
        count: int,
        auxiliary_sms: int = _AUXILIARY_SMS,
        helper_slot_mode: str = "shared_banks",
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
            PerLayerDynamicMoeLoadBalancer,
        )
        from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.arena import (
            allocate_rebalance_arena,
        )
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

        self.rank = rank
        self.ep = world
        self.device = device
        self.max_tokens = max_tokens
        self.experts = _HOME_EXPERTS * world
        self.resident = helper_slot_mode == "per_layer_resident"
        config = SimpleNamespace(
            num_slots=self.experts + world * _HELPER_SLOTS,
            auxiliary_sms=auxiliary_sms,
            helper_slot_mode=helper_slot_mode,
            planner="halo_r",
        )
        self.manager = PerLayerDynamicMoeLoadBalancer(ep_rank=rank, ep_size=world, config=config)
        self.mapping = SimpleNamespace(moe_ep_size=world, moe_ep_rank=rank, local_rank=device)
        self.process_group = dist.group.WORLD
        bundle = BundleLayout.create(hidden=_HIDDEN, intermediate=_INTERMEDIATE)
        self.arenas = []
        for layer in range(count):
            self.arenas.append(
                allocate_rebalance_arena(
                    home_experts=_HOME_EXPERTS,
                    helper_slots=_HELPER_SLOTS,
                    bundle=bundle,
                    ep_comm=self.manager.get_ep_comm(self.process_group),
                    mapping=self.mapping,
                    provider=self.manager.create_arena_provider(),
                    device=device,
                    layer_idx=layer,
                )
            )
        # The weights of each layer, as loaded; a reload changes them.
        self.weights = list(range(count))
        for layer in range(count):
            self.load_weights(layer)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        self.groups = []
        self.launches = [0] * count
        self.receiver_copies = [0] * count
        self.receiver_skips = [0] * count
        self.last_copies = 0
        # A model of each layer's residency table, [rank][helper slot], with per-layer helper slots.
        self.tables = [
            [[-1] * _HELPER_SLOTS for _ in range(world)] if self.resident else None
            for _ in range(count)
        ]
        self.dead = False

    def routes(self, step: int, tokens: int) -> torch.Tensor:
        return _routes(step, self.rank, tokens, self.experts, self.device)

    def load_weights(self, layer: int) -> None:
        """Write this rank's home experts of ``layer`` with the pattern of its current weights."""
        bound = self.arenas[layer].bound.base
        for plane, nbytes in enumerate(bound.plane_bytes):
            base = int(bound.local_plane_views[plane].data_ptr())
            for local in range(_HOME_EXPERTS):
                expert = self.rank * _HOME_EXPERTS + local
                _row(base + local * int(nbytes), int(nbytes), self.device).copy_(
                    _pattern(expert, plane, self.weights[layer], int(nbytes), self.device)
                )

    def reload_weights(self, layer: int) -> None:
        """Load new weights into ``layer`` between layer calls, as a weight update does."""
        self.weights[layer] += 1000
        self.load_weights(layer)
        torch.cuda.synchronize()
        # What the layer handle's rebind calls once a loader has rewritten the weights.
        self.groups[layer].forget_helper_residency()
        if self.tables[layer] is not None:
            self.tables[layer] = [[-1] * _HELPER_SLOTS for _ in range(self.ep)]
        MPI.COMM_WORLD.Barrier()

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
                    max_tokens_per_rank=self.max_tokens,
                    layer_idx=layer,
                )
            )

    def layer_call(self, layer: int, routes: torch.Tensor) -> int:
        """One layer call on the current stream, without a host wait; returns the READY value it lent."""
        group = self.groups[layer]
        _, ready = group.plan(routes, defer_wait=False)
        group.finish()
        self.launches[layer] += 1
        return int(ready)

    def call(self, layer: int, routes: torch.Tensor, stream: torch.cuda.Stream) -> int:
        with torch.cuda.stream(stream):
            ready = self.layer_call(layer, routes)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        return ready

    def _apply(self, layer: int, routes: torch.Tensor):
        """The reference plan of one launch, applied to the model; returns the plan."""
        counts = torch.bincount(routes.flatten().long(), minlength=self.experts).cpu().tolist()
        histogram = [sum(c) for c in zip(*MPI.COMM_WORLD.allgather(counts))]
        table = self.tables[layer]
        plan = halo_r_reference.plan_halo_r(
            histogram, self.ep, _HELPER_SLOTS, send_count=4, send_weight=0, residency=table
        )
        copies = 0
        for b in plan.broadcasts:
            for destination, slot in b.helper.items():
                if table is not None and table[destination][slot] == b.expert:
                    self.receiver_skips[layer] += 1
                else:
                    copies += 1
                if table is not None:
                    table[destination][slot] = b.expert
        self.receiver_copies[layer] += copies
        self.last_copies = copies
        return plan

    def check(self, layer: int, routes, ready: int) -> list:
        """Check the last of the launches since the previous check, whose routes ``routes`` lists in order.

        A single tensor is one launch. Every launch updates the model; the last
        one's plan, routes and helper rows are compared with the device.
        """
        routes_list = [routes] if isinstance(routes, torch.Tensor) else list(routes)
        for earlier in routes_list[:-1]:
            self._apply(layer, earlier)
        routes = routes_list[-1]
        plan = self._apply(layer, routes)
        group, arena = self.groups[layer], self.arenas[layer]
        buffers = _buffers(group)
        failures = []
        rank, local_slots = self.rank, _HOME_EXPERTS + _HELPER_SLOTS
        # Every launch of the layer is one scheduler epoch.
        if buffers.plan.cpu().tolist() != plan.plan_words(epoch=self.launches[layer]):
            failures.append(f"rank {rank}: layer {layer} plan words differ from the reference")
        if buffers.ids.cpu().tolist() != plan.helper_table[rank]:
            failures.append(f"rank {rank}: layer {layer} helper experts differ from the reference")
        if any(buffers.status.cpu().tolist()):
            failures.append(f"rank {rank}: status words {buffers.status.cpu().tolist()}")
        cells = arena.terminal_flags_tensor().view(torch.int64).cpu().tolist()
        if cells != [ready] * self.ep:
            failures.append(f"rank {rank}: layer {layer} READY row {cells}, expected {ready}")

        # Every route goes to its owner or to a receiver's helper slot of its expert.
        slots = buffers.slots[: routes.shape[0]].cpu().flatten().tolist()
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
        expected = {(b.expert, d): rows for b in plan.broadcasts for d, rows in b.rows.items()}
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
                weights = _pattern(expert, plane, self.weights[layer], int(nbytes), self.device)
                if not torch.equal(_row(address, int(nbytes), self.device), weights):
                    failures.append(f"rank {rank}: layer {layer} helper {helper} plane {plane}")

        counts = group.broadcaster.relay_counts()
        want = (self.launches[layer], self.receiver_copies[layer], self.receiver_skips[layer])
        if counts != want:
            failures.append(f"rank {rank}: layer {layer} relay counts {counts} != {want}")
        table = self.tables[layer]
        if table is not None:
            native = group.broadcaster.residency.cpu().tolist()
            if native != table:
                failures.append(f"rank {rank}: layer {layer} residency {native} != {table}")
            if any(other != native for other in MPI.COMM_WORLD.allgather(native)):
                failures.append(f"rank {rank}: layer {layer} residency differs across ranks")
        return failures

    def shutdown(self) -> None:
        if self.dead:
            # A trapped context cannot run the collective teardown.
            return
        torch.cuda.synchronize()
        self.manager.shutdown(local_safe=True)


def _case_reference(layers: _Layers) -> list:
    """T1-3: every launch against the reference, over ranks with different token counts."""
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    for seed in range(_SEEDS):
        for step in range(3):
            # The last rank has no tokens on step 1.
            last = rank == layers.ep - 1
            tokens = 0 if step == 1 and last else layers.max_tokens - 17 * (rank % 8)
            routes = layers.routes(100 * seed + step, tokens)
            failures += layers.check(0, routes, layers.call(0, routes, main))
    return failures


def _case_empty_plan(layers: _Layers) -> list:
    """T1-4: balanced routes plan no broadcast, and ranks without tokens still publish READY."""
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    for tokens in (layers.max_tokens, 0):
        routes = _uniform_routes(rank, tokens, layers.experts, layers.device)
        ready = layers.call(0, routes, main)
        if int(_buffers(layers.groups[0]).plan[1]) != 0:
            failures.append(f"rank {rank}: a balanced load planned broadcasts")
        failures += layers.check(0, routes, ready)
    return failures


def _case_alternating_layers(layers: _Layers) -> list:
    """Three layers: on two shared helper banks, layer 2 refills the bank of layer 0; per-layer
    helper slots give every layer its own set."""
    rank = layers.rank
    banks = [int(arena.provider.layer.helper_set) for arena in layers.arenas]
    expected = [0, 1, 2] if layers.resident else [0, 1, 0]
    failures = [] if banks == expected else [f"rank {rank}: helper sets {banks}"]
    main = torch.cuda.Stream()
    for seed in range(_SEEDS):
        for step in range(2):
            for layer in range(len(layers.groups)):
                routes = layers.routes(1000 * seed + 10 * step + layer, layers.max_tokens)
                failures += layers.check(layer, routes, layers.call(layer, routes, main))
    return failures


def _case_resident_repeat(layers: _Layers) -> list:
    """T1-5 (a, b): repeated routes copy nothing and keep the rows, even a row a test overwrote.

    Rank 1's resident helper row 0 is overwritten with a sentinel; the repeat
    keeps it, because the relay skips it. Once the table is reset (the weight
    reload hook), the next call restores the owner's bytes.
    """
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    routes = layers.routes(0, layers.max_tokens)
    failures += layers.check(0, routes, layers.call(0, routes, main))
    holds = layers.tables[0][1][0] >= 0
    bound = layers.arenas[0].bound.base
    nbytes = int(bound.plane_bytes[0])
    address = int(bound.local_plane_views[0].data_ptr()) + _HOME_EXPERTS * nbytes
    sentinel = torch.full((nbytes,), 0x5A, dtype=torch.uint8, device=layers.device)
    if rank == 1 and holds:
        _row(address, nbytes, layers.device).copy_(sentinel)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    layers.call(0, routes, main)
    # Rank 1's row holds the sentinel, so only the model is updated here.
    layers._apply(0, routes)
    if layers.last_copies:
        failures.append(f"rank {rank}: the repeated call copied {layers.last_copies} receivers")
    if rank == 1 and holds and not torch.equal(_row(address, nbytes, layers.device), sentinel):
        failures.append(f"rank {rank}: the repeated call rewrote a resident helper row")
    if not holds:
        failures.append(f"rank {rank}: rank 1 received no expert, so the sentinel case is void")
    layers.groups[0].forget_helper_residency()
    layers.tables[0] = [[-1] * _HELPER_SLOTS for _ in range(layers.ep)]
    MPI.COMM_WORLD.Barrier()
    failures += layers.check(0, routes, layers.call(0, routes, main))
    return failures


def _case_resident_sequence(layers: _Layers) -> list:
    """T1-5 (c): random routes with evictions and returns; tables match a model and each other."""
    rank = layers.rank
    main = torch.cuda.Stream()
    rng = random.Random(7)
    failures = []
    steps = [rng.randrange(6) for _ in range(100)]
    for index, step in enumerate(steps):
        tokens = 0 if index % 17 == 5 and rank == 0 else layers.max_tokens
        routes = layers.routes(step, tokens)
        failures += layers.check(0, routes, layers.call(0, routes, main))
        if failures:
            break
    if layers.receiver_skips[0] == 0:
        failures.append(f"rank {rank}: no receiver was ever skipped")
    return failures


def _case_resident_reload(layers: _Layers) -> list:
    """T1-5: a weight reload makes the next call copy every receiver again, with the new weights."""
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    routes = layers.routes(0, layers.max_tokens)
    for _ in range(2):
        failures += layers.check(0, routes, layers.call(0, routes, main))
    if layers.last_copies:
        failures.append(f"rank {rank}: the repeated call copied {layers.last_copies} receivers")
    layers.reload_weights(0)
    skips = layers.receiver_skips[0]
    failures += layers.check(0, routes, layers.call(0, routes, main))
    # check() compared the rows with the new weights and the counters and the
    # table with a model that forgot every slot.
    if layers.last_copies == 0 or layers.receiver_skips[0] != skips:
        failures.append(f"rank {rank}: the call after the reload skipped receivers")
    return failures


def _case_shared_banks(layers: _Layers) -> list:
    """T1-5 (d): shared banks keep no table and copy every receiver every time."""
    failures = _case_alternating_layers(layers)
    if any(layers.receiver_skips):
        failures.append(f"rank {layers.rank}: shared banks skipped receivers")
    return failures


def _case_stable_slot(layers: _Layers) -> list:
    """T1-6: a resident expert keeps its slot when a lower-id expert joins its receiver.

    At EP 4, route step 2 sends expert 10 to slot 0 of rank 3, and step 13
    sends experts 1 and 10 there: in ascending order expert 10 would move to
    slot 1 and be copied again; resident-first keeps it in slot 0.
    """
    rank = layers.rank
    if layers.ep != 4:
        return [f"rank {rank}: the stable-slot routes are built for EP 4, not {layers.ep}"]
    main = torch.cuda.Stream()
    failures = []
    skips = 0
    for step in (2, 13):
        routes = layers.routes(step, layers.max_tokens)
        skips = layers.receiver_skips[0]
        failures += layers.check(0, routes, layers.call(0, routes, main))
    hot = _buffers(layers.groups[0]).ids.cpu().tolist()
    if layers.tables[0][3] != [10, 1] or (rank == 3 and hot != [10, 1]):
        failures.append(f"rank {rank}: rank 3's helper slots {layers.tables[0][3]}, here {hot}")
    if layers.receiver_skips[0] == skips:
        failures.append(f"rank {rank}: the resident expert was copied again")
    return failures


def _case_gate_under_hog(layers: _Layers, use_graph: bool) -> list:
    """T1-7: MegaMoE's full grid, queued right after the route wait, must not starve the relay.

    A hog holds every SM but the HALO-R kernel's, and COPY runs below MAIN, so
    a gate kernel that MAIN has already released takes every SM that frees up
    first. The route wait passes only once every HALO-R CTA has released its
    routes, after the grid rendezvous, so the relay already holds its SMs.
    """
    rank = layers.rank
    group = layers.groups[0]
    kernels = _GateKernels(layers.device)
    ready_row = layers.arenas[0].terminal_flags_tensor().view(torch.int64)
    _, greatest = torch.cuda.Stream.priority_range()
    main = torch.cuda.Stream(device=layers.device, priority=greatest)
    side = torch.cuda.Stream(device=layers.device)
    # The group accepts only a MAIN below COPY; bind this MAIN directly.
    group._execution_stream_handle = int(main.cuda_stream)
    static = layers.routes(0, layers.max_tokens)
    expected = 1 if _implementation() == "graphs" else None

    def layer_call() -> None:
        _, ready = group.plan(static, defer_wait=False)
        kernels.launch_gate(ready_row, int(ready) if expected is None else expected, main)
        group.finish()

    graph = None
    if use_graph:
        main.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(main):
            # Warm up eagerly on the capture stream, as the breakable runner does.
            layer_call()
        layers.launches[0] += 1
        torch.cuda.synchronize()
        failures = layers.check(0, static, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main):
            layer_call()
    else:
        failures = []
    for step in range(1, 51):
        routes = layers.routes(step, layers.max_tokens)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        kernels.launch_hog(side, group.auxiliary_sms)
        with torch.cuda.stream(main):
            static.copy_(routes)
            if graph is not None:
                graph.replay()
            else:
                layer_call()
        layers.launches[0] += 1
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        timed_out = int(kernels.timed_out.item())
        if timed_out:
            failures.append(f"rank {rank}: step {step}: {timed_out} gate CTAs timed out on READY")
            kernels.timed_out.zero_()
        ready = 1 if expected is not None else layers.launches[0]
        failures += layers.check(0, routes, ready)
    if graph is not None:
        graph.reset()
    return failures


def _case_fail_closed(layers: _Layers) -> dict:
    """T1-8 (a): a status word set on rank 0 traps it before it releases routes; its peers trap next.

    Rank 0 writes a nonzero epoch into the invalid-route status word of layer 1
    on MAIN before forward 5. Its kernel traps before its route release, so its
    call fails within seconds. Each peer completes that forward and traps at
    its next count exchange or READY drain, which rank 0 never joins, within
    the spin bound. Returns this rank's first error and its delay.
    """
    rank = layers.rank
    main = torch.cuda.Stream()
    for forward in range(5):
        for layer in range(2):
            layers.call(layer, layers.routes(10 * forward + layer, layers.max_tokens), main)
    layers.manager.get_scheduler_spin_bound().set_timeout_seconds(_FAIL_TIMEOUT_S)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    start = time.monotonic()
    error = None
    where = None
    for forward in range(5, 9):
        for layer in range(2):
            try:
                with torch.cuda.stream(main):
                    if rank == 0 and forward == 5 and layer == 1:
                        _buffers(layers.groups[layer]).status[3].fill_(5)
                    layers.layer_call(layer, layers.routes(10 * forward + layer, layers.max_tokens))
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001  the trap surfaces as a CUDA error
                error, where = f"{type(exc).__name__}: {exc}", (forward, layer)
                break
        if error is not None:
            break
    layers.dead = error is not None
    return {"rank": rank, "error": error, "where": where, "seconds": time.monotonic() - start}


def _case_graphs_single(layers: _Layers) -> list:
    """T1-10 (a): one layer captured once, then 50 replays with fresh routes in the static input."""
    graph, static, failures = _capture(layers, [0])
    for step in range(1, 51):
        routes = layers.routes(step, layers.max_tokens)
        with torch.cuda.stream(layers.eager[0]):
            static[0].copy_(routes)
            graph.replay()
        layers.launches[0] += 1
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        failures += layers.check(0, routes, 1)
    failures += _replay_counts(layers, replays=50)
    graph.reset()
    return failures


def _case_graphs_layers(layers: _Layers) -> list:
    """T1-10 (b): three layers in one graph, replayed with fresh routes."""
    count = len(layers.groups)
    graph, static, failures = _capture(layers, list(range(count)))
    for step in range(1, 11):
        routes = [layers.routes(100 * step + layer, layers.max_tokens) for layer in range(count)]
        with torch.cuda.stream(layers.eager[0]):
            for layer in range(count):
                static[layer].copy_(routes[layer])
            graph.replay()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        for layer in range(count):
            layers.launches[layer] += 1
            failures += layers.check(layer, routes[layer], 1)
    graph.reset()
    return failures


def _case_graphs_interleaved(layers: _Layers) -> list:
    """T1-10 (c): eager calls and replays interleaved on the same layer, with MAIN changing.

    Each step queues one to three launches without a host wait: eager calls on
    one of two MAIN streams and replays on the current one, so the group's
    MAIN-to-MAIN fence, the per-call join and the route-counter reset order
    them. The step's last launch is checked; the model applies every launch.
    """
    graph, static, failures = _capture(layers, [0])
    rng = random.Random(11)
    streams = layers.eager
    # Replays run on the group's current MAIN, as the runner replays on the
    # execution stream that its eager calls use; only eager calls change MAIN.
    main = 0
    replays = 0
    for step in range(200):
        launches = []
        for _ in range(1 + rng.randrange(3)):
            routes = layers.routes(1000 + 7 * step + len(launches), layers.max_tokens)
            if rng.random() < 0.5:
                with torch.cuda.stream(streams[main]):
                    static[0].copy_(routes)
                    graph.replay()
                layers.launches[0] += 1
                replays += 1
            else:
                if rng.random() < 0.4:
                    main ^= 1
                with torch.cuda.stream(streams[main]):
                    layers.layer_call(0, routes)
            launches.append(routes)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        failures += layers.check(0, launches, 1)
        if failures:
            break
    failures += _replay_counts(layers, replays=replays)
    graph.reset()
    return failures


def _case_graphs_capturing_switch(layers: _Layers) -> list:
    """T1-10 (d): MAIN cannot change into a stream that is capturing."""
    rank = layers.rank
    group = layers.groups[0]
    failures = []
    main = torch.cuda.Stream(device=layers.device)
    routes = layers.routes(0, layers.max_tokens)
    failures += layers.check(0, routes, layers.call(0, routes, main))
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream(device=layers.device)
    try:
        with torch.cuda.graph(graph, stream=capture):
            group.plan(routes, defer_wait=False)
        failures.append(f"rank {rank}: a switch into a capturing stream did not raise")
    except RuntimeError as error:
        if "capturing" not in str(error):
            failures.append(f"rank {rank}: unexpected error {error}")
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    # Nothing changed: the next eager call proceeds.
    routes = layers.routes(1, layers.max_tokens)
    failures += layers.check(0, routes, layers.call(0, routes, main))
    return failures


def _capture(layers: _Layers, layer_ids: list):
    """Capture layer calls as the breakable runner does: warm up eagerly on its capture stream,
    then capture there. Returns the graph, the static routes per layer and the warmup's failures."""
    layers.eager = [torch.cuda.Stream(device=layers.device) for _ in range(2)]
    capture = torch.cuda.Stream(device=layers.device)
    static = {layer: layers.routes(500 + layer, layers.max_tokens) for layer in layer_ids}
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        for layer in layer_ids:
            layers.layer_call(layer, static[layer])
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    failures = []
    for layer in layer_ids:
        failures += layers.check(layer, static[layer], 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        for layer in layer_ids:
            layers.groups[layer].plan(static[layer], defer_wait=False)
            layers.groups[layer].finish()
    torch.cuda.synchronize()
    layers.captures = len(layer_ids)
    return graph, static, failures


def _replay_counts(layers: _Layers, replays: int) -> list:
    """Device launches are eager calls plus replays, more than the host's plan calls minus captures."""
    group = layers.groups[0]
    launches, _, _ = group.broadcaster.relay_counts()
    host = group.plan_calls - layers.captures
    if launches != layers.launches[0] or launches - host != replays or replays <= 0:
        return [f"rank {layers.rank}: launches {launches}, host calls {host}, replays {replays}"]
    return []


def _equivalence_records(layers: _Layers) -> dict:
    """T1-2: run the fixed inputs and return what both builds must produce bitwise.

    Plan words without the epoch, the route prefix of every broadcast, the
    physical slot of every valid route, the helper experts and their owners, and
    every helper row of the layer after each call.
    """
    records = {}
    main = torch.cuda.Stream()
    for seed in range(_SEEDS):
        for tokens in _EQUIV_TOKENS:
            for kind in ("uniform", "skewed"):
                for layer in range(len(layers.groups)):
                    if kind == "uniform":
                        routes = _uniform_routes(layers.rank, tokens, layers.experts, layers.device)
                    else:
                        routes = layers.routes(10000 * seed + 10 * tokens + layer, tokens)
                    layers.call(layer, routes, main)
                    buffers = _buffers(layers.groups[layer])
                    plan = buffers.plan.cpu()
                    broadcasts = int(plan[1])
                    prefix = buffers.prefix.cpu().view(-1, layers.ep + 1)[:broadcasts, : layers.ep]
                    bound = layers.arenas[layer].bound.base
                    rows = []
                    for plane, nbytes in enumerate(bound.plane_bytes):
                        base = int(bound.local_plane_views[plane].data_ptr())
                        helpers = _row(
                            base + _HOME_EXPERTS * int(nbytes),
                            _HELPER_SLOTS * int(nbytes),
                            layers.device,
                        )
                        rows.append(helpers.cpu().clone())
                    records[f"seed{seed}/T{tokens}/{kind}/layer{layer}"] = {
                        "plan": plan[1:].clone(),
                        "prefix": prefix.clone(),
                        "slots": buffers.slots[:tokens].cpu().clone(),
                        "ids": buffers.ids.cpu().clone(),
                        "owners": buffers.owners.cpu().clone(),
                        "helper_rows": rows,
                    }
    return records


def _case_equivalence(layers: _Layers, mode: str) -> list:
    """T1-2: dump on a commit-0 build, or compare bitwise with that dump."""
    path = os.path.join(_EQUIV_DIR, f"ep{layers.ep}_rank{layers.rank}.pt")
    records = _equivalence_records(layers)
    if mode == "dump":
        os.makedirs(_EQUIV_DIR, exist_ok=True)
        torch.save(records, path)
        return []
    reference = torch.load(path)
    failures = []
    if set(reference) != set(records):
        return [
            f"rank {layers.rank}: inputs differ from the dump: {sorted(set(reference) ^ set(records))[:4]}"
        ]
    for key, want in reference.items():
        got = records[key]
        for name in ("plan", "prefix", "slots", "ids", "owners"):
            if not torch.equal(got[name], want[name]):
                failures.append(f"rank {layers.rank}: {key} {name} differs from commit 0")
        if any(not torch.equal(a, b) for a, b in zip(got["helper_rows"], want["helper_rows"])):
            failures.append(f"rank {layers.rank}: {key} helper rows differ from commit 0")
        if len(failures) > 20:
            break
    return failures


def _has_helper_residency() -> bool:
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import halo_r_planner

    group = getattr(halo_r_planner, "HaloRSlotSchedulerGroup", None)
    return hasattr(group, "forget_helper_residency")


def _starve_relay(layers: _Layers) -> None:
    """Give COPY the lowest stream priority, below the MAIN of the hog cases.

    Production gives COPY the highest priority, which only orders CTAs that
    are still pending; inverting it makes MegaMoE win every SM that frees up
    while the HALO-R kernel is pending. Must run before the groups are built.
    """
    least, _ = torch.cuda.Stream.priority_range()
    copy_stream = torch.cuda.Stream(device=layers.device, priority=least)
    layers.manager.get_copy_stream = lambda device: copy_stream


# case: (layers, auxiliary SMs, helper-slot mode, max tokens per rank, needs CUDA graphs)
_SETUPS = {
    "t1_2": (3, _AUXILIARY_SMS, "shared_banks", _EQUIV_MAX_TOKENS, False),
    "t1_3": (1, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, False),
    "t1_3_aux16": (1, 16, "shared_banks", _MAX_TOKENS, False),
    "t1_4": (1, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, False),
    "t1_5_repeat": (1, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, False),
    "t1_5_sequence": (1, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, False),
    "t1_5_layers": (3, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, False),
    "t1_5_reload": (1, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, False),
    "t1_5_shared_banks": (3, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, False),
    "t1_6": (1, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, False),
    "t1_7": (1, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, False),
    "t1_7_aux16": (1, 16, "shared_banks", _MAX_TOKENS, False),
    "t1_7_graph": (1, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, True),
    "t1_8": (2, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, False),
    "t1_10_single": (1, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, True),
    "t1_10_layers_resident": (3, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, True),
    "t1_10_layers_shared": (3, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, True),
    "t1_10_interleaved": (1, _AUXILIARY_SMS, "per_layer_resident", _MAX_TOKENS, True),
    "t1_10_capturing_switch": (1, _AUXILIARY_SMS, "shared_banks", _MAX_TOKENS, True),
}


def _run_case(layers: _Layers, case: str, params: dict):
    if case == "t1_2":
        return _case_equivalence(layers, params["mode"])
    if case.startswith("t1_3"):
        return _case_reference(layers)
    if case == "t1_4":
        return _case_empty_plan(layers)
    if case == "t1_5_repeat":
        return _case_resident_repeat(layers)
    if case == "t1_5_sequence":
        return _case_resident_sequence(layers)
    if case == "t1_5_layers":
        return _case_alternating_layers(layers)
    if case == "t1_5_reload":
        return _case_resident_reload(layers)
    if case == "t1_5_shared_banks":
        return _case_shared_banks(layers)
    if case == "t1_6":
        return _case_stable_slot(layers)
    if case.startswith("t1_7"):
        return _case_gate_under_hog(layers, use_graph=case == "t1_7_graph")
    if case == "t1_8":
        return _case_fail_closed(layers)
    if case == "t1_10_single":
        return _case_graphs_single(layers)
    if case.startswith("t1_10_layers"):
        return _case_graphs_layers(layers)
    if case == "t1_10_interleaved":
        return _case_graphs_interleaved(layers)
    if case == "t1_10_capturing_switch":
        return _case_graphs_capturing_switch(layers)
    raise ValueError(f"unknown case {case}")


def _run_worker(case: str, params: dict):
    """One EP rank of a case; returns (skip reason or None, result)."""
    from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size

    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True)
    rank, world = mpi_rank(), mpi_world_size()
    device = local_mpi_rank()
    layers = None
    try:
        torch.cuda.set_device(device)
        count, auxiliary_sms, mode, max_tokens, needs_graphs = _SETUPS[case]
        auxiliary_sms = params.get("auxiliary_sms", auxiliary_sms)
        implementation = _implementation()
        reason = _skip_reason()
        if reason is None and world != _EP_SIZE:
            reason = f"the MPI world has {world} ranks, HALO_R_TEST_EP is {_EP_SIZE}"
        if reason is None and needs_graphs and implementation != "graphs":
            reason = f"this build ({implementation}) cannot capture HALO-R layers"
        if reason is None and implementation == "commit0" and params.get("mode") != "dump":
            reason = "a commit-0 build runs only the t1_2 dump"
        if reason is None and mode == "per_layer_resident" and not _has_helper_residency():
            reason = f"this build ({implementation}) has no per-layer HALO-R helper slots"
        reasons = MPI.COMM_WORLD.allgather(reason)
        if any(reasons):
            return "; ".join(sorted({r for r in reasons if r})), []
        _init_process_group(rank, world)
        layers = _Layers(rank, world, device, count, auxiliary_sms, mode, max_tokens)
        if case.startswith("t1_7"):
            _starve_relay(layers)
        try:
            layers.build_groups()
        except RuntimeError as error:
            # Every rank reaches the native context creation together, so every rank skips.
            if _PLACEHOLDER_MESSAGE in str(error):
                return f"placeholder build: {error}", []
            raise
        return None, _run_case(layers, case, params)
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


def _run_on_every_rank(monkeypatch, case: str, params: dict) -> list:
    """Run one case on every EP rank and return each rank's (skip reason, result).

    Under ``trtllm-llmapi-launch`` the ranks are the launcher's MPI world, on
    any number of nodes; otherwise an MPIPoolExecutor starts them on this node.
    """
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S + 60, exit=True)
    try:
        address = os.environ.get("TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR")
        if address:
            from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_hmac_key_env
            from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient

            client = RemoteMpiCommSessionClient(address, get_spawn_proxy_process_ipc_hmac_key_env())
            # The launcher's server unpickles a task by reference before it adopts
            # this process's sys.path, which holds this module. The task is the
            # builtin eval instead: on every rank it unpickles this module's worker
            # by value (cloudpickle, registered above) and runs it.
            expression = "__import__('cloudpickle').loads(worker)(case, params)"
            names = {"worker": cloudpickle.dumps(_run_worker), "case": case, "params": params}
            return client.submit_sync(eval, expression, names)  # nosec B307
        if torch.cuda.device_count() < _EP_SIZE:
            pytest.skip(f"Requires {_EP_SIZE} GPUs, or trtllm-llmapi-launch across nodes")
        # Per-layer dynamic EPLB needs the larger launch queues from CUDA initialization on.
        monkeypatch.setenv("CUDA_SCALE_LAUNCH_QUEUES", "4x")
        with MPIPoolExecutor(_EP_SIZE) as executor:
            return list(executor.map(_run_worker, [case] * _EP_SIZE, [params] * _EP_SIZE))
    finally:
        faulthandler.cancel_dump_traceback_later()


def _assert_passed(results: list) -> None:
    skips = {skip for skip, _ in results if skip}
    if skips:
        pytest.skip("; ".join(sorted(skips)))
    failures = [failure for _, rank_failures in results for failure in rank_failures]
    assert not failures, "\n".join(failures[:20])


@pytest.mark.parametrize(
    "case, params",
    [
        pytest.param("t1_3", {}, id="t1_3_reference"),
        pytest.param("t1_3_aux16", {}, id="t1_3_reference_aux16"),
        pytest.param("t1_4", {}, id="t1_4_zero_tokens_and_empty_plan"),
        pytest.param("t1_5_repeat", {}, id="t1_5_repeat_and_sentinel"),
        pytest.param("t1_5_sequence", {}, id="t1_5_sequence"),
        pytest.param("t1_5_layers", {}, id="t1_5_per_layer_sets"),
        pytest.param("t1_5_reload", {}, id="t1_5_weight_reload"),
        pytest.param("t1_5_shared_banks", {}, id="t1_5_shared_banks"),
        pytest.param("t1_6", {}, id="t1_6_stable_slot"),
        pytest.param("t1_7", {}, id="t1_7_gate_under_hog"),
        pytest.param("t1_7_aux16", {}, id="t1_7_gate_under_hog_aux16"),
        pytest.param("t1_7_graph", {}, id="t1_7_gate_under_hog_graph"),
        pytest.param("t1_10_single", {}, id="t1_10_single_layer"),
        pytest.param("t1_10_layers_resident", {}, id="t1_10_layers_resident"),
        pytest.param("t1_10_layers_shared", {}, id="t1_10_layers_shared"),
        pytest.param("t1_10_interleaved", {}, id="t1_10_eager_and_replay_interleaved"),
        pytest.param("t1_10_capturing_switch", {}, id="t1_10_switch_into_capture"),
    ],
)
def test_halo_r(monkeypatch, case, params):
    _assert_passed(_run_on_every_rank(monkeypatch, case, params))


@pytest.mark.parametrize(
    "mode, auxiliary_sms",
    [
        pytest.param("dump", _AUXILIARY_SMS, id="t1_2_dump"),
        pytest.param("compare", 2, id="t1_2_compare_aux2"),
        pytest.param("compare", 4, id="t1_2_compare_aux4"),
        pytest.param("compare", 8, id="t1_2_compare_aux8"),
    ],
)
def test_halo_r_matches_commit_0(monkeypatch, mode, auxiliary_sms):
    if not _EQUIV_DIR:
        pytest.skip("HALO_R_EQUIV_DIR names the commit-0 dump of T1-2")
    params = {"mode": mode, "auxiliary_sms": auxiliary_sms}
    _assert_passed(_run_on_every_rank(monkeypatch, "t1_2", params))


def test_halo_r_t1_8_status_word_fails_closed(monkeypatch):
    """A status word on rank 0 traps it before it releases routes; every peer traps after it."""
    results = _run_on_every_rank(monkeypatch, "t1_8", {})
    skips = {skip for skip, _ in results if skip}
    if skips:
        pytest.skip("; ".join(sorted(skips)))
    outcomes = {result["rank"]: result for _, result in results}
    assert sorted(outcomes) == list(range(_EP_SIZE)), outcomes
    failures = []
    for rank, outcome in sorted(outcomes.items()):
        bound = _FAIL_OWN_TRAP_S if rank == 0 else _FAIL_PEER_TRAP_S
        if outcome["error"] is None or "CUDA" not in outcome["error"]:
            failures.append(f"rank {rank} did not trap: {outcome}")
        elif outcome["seconds"] > bound:
            failures.append(
                f"rank {rank} trapped after {outcome['seconds']:.1f} s, bound {bound} s"
            )
    # The injected rank traps in the call it injected; its peers complete that call.
    if outcomes[0]["where"] != (5, 1):
        failures.append(f"rank 0 trapped at {outcomes[0]['where']}, not at forward 5, layer 1")
    assert not failures, "\n".join(failures)
