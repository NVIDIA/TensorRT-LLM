# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HALO-R kernel tests on a real shared-slot arena, one EP rank per GPU.

Every rank fills its home experts with a byte pattern unique to (expert,
plane, weights) and runs HALO-R layer calls through the scheduler group. Each
layer call holds the copy stream back before the kernel starts and then
queues on MAIN, as MegaMoE does:
- a copy of the routes it lends, right after the route wait;
- a READY gate on the SMs that the HALO-R kernel leaves free;
- a copy of this rank's helper rows, once the gate opens.
A route wait that passes before the kernel releases its routes, or a gate
that opens on the previous launch's READY cells, therefore copies the
previous launch's routes or helper rows.

After the launches since the previous check, every rank compares:
- each launch's copies with that launch's pure-Python reference plan of the
  global routes: every route goes to its owner's home slot or to a
  receiver's helper slot of its expert, the routes per (expert, rank) match
  the plan's quota, and every helper row this rank received holds the
  owner's expert, bytewise;
- the last launch's plan words and helper experts with the reference;
- the READY row (1 in every cell), the route counter (0) and every status
  word (0);
- the relay counters (launches, copied and skipped receivers) and, with
  per-layer helper slots, the residency table with a model of it, on every
  rank.

``test_halo_r_traps`` makes one launch fail on purpose: a status word set
before the route release, a READY cell that never arrives, or a CTA that
cannot become resident. The failing rank must trap in that call, and its
peers in that call or the next, at their wait for it. These cases fault
every GPU of the node on purpose, so they run only when HALO_R_TEST_TRAPS=1;
a directory-level test list entry skips them.

The ranks run on one node through an MPIPoolExecutor, or on any number of
nodes under ``trtllm-llmapi-launch`` through the MPI session it serves.
HALO_R_TEST_EP sets the EP size (default 4) and HALO_R_TEST_SEEDS the number
of seeds per case (default 1). On an SM100-family GPU with CUDA 13.1 or
later, a build without the HALO-R kernels fails every case instead of
skipping it; HALO_R_TEST_ALLOW_SKIP=1 makes it skip. HALO_R_TEST_NO_SKIP=1
turns every skip (too few GPUs, no MNNVL, a case built for another EP size)
into a failure, for runs that must exercise every case.
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
from typing import Any

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
_HOME_EXPERTS = 8
_HELPER_SLOTS = 2
_TOPK = 4
_MAX_TOKENS = 256
_AUXILIARY_SMS = 4
# Small planes stay in the per-layer record pages; the weight planes exceed
# them and live in the helper sets, so both relay paths are covered.
_HIDDEN = 256
_INTERMEDIATE = 128
# A production-size bundle (about 37 MB per expert): its relay takes long enough that a READY gate opening on the
# previous launch's cells copies helper rows that are still being written.
_PRODUCTION_BUNDLE = (7168, 3072)
_WORKER_TIMEOUT_S = 600
# Every launch publishes READY 1, which MegaMoE's gate waits for.
_READY = 1
# COPY waits this long before each kernel, so MAIN reaches its route wait and
# READY gate long before the kernel writes routes or helper rows.
_COPY_DELAY_NS = 2_000_000
# The READY gate gives up after this long; the relay finishes in milliseconds.
_GATE_TIMEOUT_NS = 5_000_000_000
# The SM hog outlives the HALO-R kernel's scheduler role.
_HOG_NS = 300_000_000
# The trap cases: the spin bound of every launch after the warmup, the failing
# call (forward, layer), and how long a hog keeps one HALO-R CTA from an SM.
_TRAP_TIMEOUT_S = 5
_TRAP_CALL = (5, 1)
_TRAP_HOG_NS = 3 * _TRAP_TIMEOUT_S * 1_000_000_000


class _Skip(Exception):
    """Raised on every rank alike when a case does not apply to this geometry."""


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
    """Skewed routes: most tokens pick the step's hot expert, distinct experts per token.

    The hot experts change every step, so consecutive launches differ in
    their routes and helper experts.
    """
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
    """Every rank owns the same routed-token count summed over all ranks, so the plan moves nothing.

    Rank ``rank`` sends its i-th route to owner ``(i + rank) % ep``: over all
    ranks, each owner receives ``num_tokens * topk`` routes, for any EP size.
    A token's routes go to consecutive owners, so they are distinct experts.
    """
    ep = experts // _HOME_EXPERTS
    flat = [
        ((i + rank) % ep) * _HOME_EXPERTS + (i // ep) % _HOME_EXPERTS
        for i in range(num_tokens * _TOPK)
    ]
    return torch.tensor(flat, dtype=torch.int32, device=f"cuda:{device}").reshape(-1, _TOPK)


def _any_rank(failures: list) -> bool:
    """Whether any rank has failed, so that every rank leaves a step loop together.

    A rank that left alone would leave its peers in the next step's count
    exchange, and a real mismatch would show up as their trap instead.
    """
    return MPI.COMM_WORLD.allreduce(bool(failures), op=MPI.LOR)


_KERNEL_SOURCE = r"""
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

// Holds the stream it runs on for `ns` nanoseconds, on one thread.
extern "C" __global__ void streamDelay(unsigned long long ns)
{
    unsigned long long const start = globalTimer();
    while (globalTimer() - start < ns)
    {
        __nanosleep(1000);
    }
}

// MegaMoE's helper-weight READY gate: hold the SM until every cell of this rank's READY row reaches the
// expected value. Gives up after `timeoutNs` and counts the CTA in `timedOut`.
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


class _Kernels:
    """The READY gate, an SM hog and a stream delay.

    Each gate or hog CTA takes over half of an SM's shared memory, so neither
    shares an SM with the other or with a HALO-R CTA, which takes nearly all
    of it.
    """

    def __init__(self, device: int) -> None:
        try:
            from cuda.bindings import driver, nvrtc
        except ImportError:
            from cuda import cuda as driver
            from cuda import nvrtc
        self.driver = driver

        def check(result) -> Any:
            error, *values = result if isinstance(result, tuple) else (result,)
            if error not in (driver.CUresult.CUDA_SUCCESS, nvrtc.nvrtcResult.NVRTC_SUCCESS):
                raise RuntimeError(f"CUDA error {error}")
            return values[0] if len(values) == 1 else values

        self.check = check
        major, minor = torch.cuda.get_device_capability(device)
        program = check(
            nvrtc.nvrtcCreateProgram(_KERNEL_SOURCE.encode(), b"halo_r_test.cu", 0, [], [])
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
        self.delay = check(driver.cuModuleGetFunction(self.module, b"streamDelay"))
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

    def _launch(self, function, blocks: int, threads: int, shared: int, args, stream) -> None:
        self.check(
            self.driver.cuLaunchKernel(
                function,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                shared,
                self.driver.CUstream(stream.cuda_stream),
                args,
                0,
            )
        )

    def launch_hog(self, stream: torch.cuda.Stream, blocks: int, ns: int = _HOG_NS) -> None:
        """Hold ``blocks`` SMs for ``ns`` nanoseconds."""
        args = ((ns,), (ctypes.c_ulonglong,))
        self._launch(self.hog, blocks, 128, self.shared_bytes, args, stream)

    def launch_delay(self, stream: torch.cuda.Stream, ns: int) -> None:
        args = ((ns,), (ctypes.c_ulonglong,))
        self._launch(self.delay, 1, 32, 0, args, stream)

    def launch_gate(self, ready: torch.Tensor, stream: torch.cuda.Stream, blocks: int) -> None:
        args = (
            (ready.data_ptr(), ready.numel(), _READY, _GATE_TIMEOUT_NS, self.timed_out.data_ptr()),
            (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_void_p,
            ),
        )
        self._launch(self.gate, blocks, 128, self.shared_bytes, args, stream)


def _init_process_group(rank: int, world: int) -> None:
    """A gloo group over the MPI world; rank 0's host serves the store, on any node."""
    store = None
    if rank == 0:
        host = socket.gethostbyname(socket.gethostname())
        store = dist.TCPStore(host, 0, world, is_master=True, wait_for_workers=False)
    host, port = MPI.COMM_WORLD.bcast((host, store.port) if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore(host, port, world, is_master=False, wait_for_workers=False)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world)


def _skip_reason() -> str | None:
    if torch.cuda.get_device_capability()[0] != 10:
        return "HALO-R requires an SM100-family GPU"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "HALO-R requires an NVLink fabric with multicast"
    return None


# A build without the SM100 + CUDA 13.1 kernels registers the ops but throws on use.
_PLACEHOLDER_MESSAGE = "HALO-R requires CUDA 13.1+"
# A skip reason that means this GPU and CUDA should run HALO-R but the build cannot.
_BUILD_SKIP = "the build has no HALO-R kernels: "


def _cuda_at_least(major: int, minor: int) -> bool:
    version = torch.version.cuda or "0.0"
    return tuple(int(part) for part in version.split(".")[:2]) >= (major, minor)


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
        bundle_dims: tuple = (_HIDDEN, _INTERMEDIATE),
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
        self.max_tokens = _MAX_TOKENS
        self.experts = _HOME_EXPERTS * world
        self.auxiliary_sms = auxiliary_sms
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
        bundle = BundleLayout.create(hidden=bundle_dims[0], intermediate=bundle_dims[1])
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
        # HALO-R's helper sets carry no multicast object, so its arenas have
        # neither HALO levels nor multicast destinations.
        pool = self.arenas[0].provider.pool
        assert not pool.helper_multicast and not any(s.level_specs for s in pool.sets)
        assert not any(arena.bound.destination_ptrs for arena in self.arenas)
        # The helper set of each layer: shared banks alternate, per-layer slots do not.
        self.banks = [int(arena.provider.layer.helper_set) for arena in self.arenas]
        self._helper_views = {}
        for layer in range(count):
            self.helper_rows(layer)
        # The weights of each layer, as loaded; a reload changes them.
        self.weights = list(range(count))
        for layer in range(count):
            self.load_weights(layer)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        self.kernels = _Kernels(device)
        # MegaMoE's grid leaves the HALO-R kernel's SMs to it.
        self.gate_blocks = self.kernels.sms - auxiliary_sms
        self.copy_delay_ns = _COPY_DELAY_NS
        self.groups = []
        self.launches = [0] * count
        self.receiver_copies = [0] * count
        self.receiver_skips = [0] * count
        self.last_copies = 0
        # Per layer, the MAIN-side copies of each launch since the last check; None when not checkable.
        self.pending = [[] for _ in range(count)]
        # Per captured layer, the copies its graph writes on every replay.
        self.graph_copies = {}
        # Per helper set, the layer whose launch last wrote it.
        self.bank_user = {}
        self.captures = 0
        # A model of each layer's residency table, [rank][helper slot], with per-layer helper slots.
        self.tables = [
            [[-1] * _HELPER_SLOTS for _ in range(world)] if self.resident else None
            for _ in range(count)
        ]
        self.dead = False

    def routes(self, step: int, tokens: int) -> torch.Tensor:
        return _routes(step, self.rank, tokens, self.experts, self.device)

    def ready_row(self, layer: int) -> torch.Tensor:
        return self.arenas[layer].terminal_flags_tensor().view(torch.int64)

    def helper_rows(self, layer: int) -> list:
        """This rank's helper rows of ``layer``, one uint8 view per plane, made outside any capture."""
        views = self._helper_views.get(layer)
        if views is None:
            bound = self.arenas[layer].bound
            views = self._helper_views[layer] = [
                _row(
                    int(bound.local_plane_views[plane].data_ptr()) + _HOME_EXPERTS * int(nbytes),
                    _HELPER_SLOTS * int(nbytes),
                    self.device,
                )
                for plane, nbytes in enumerate(bound.plane_bytes)
            ]
        return views

    def load_weights(self, layer: int) -> None:
        """Write this rank's home experts of ``layer`` with the pattern of its current weights."""
        bound = self.arenas[layer].bound
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
        # What the layer handle's rebind calls on every rank once a loader has rewritten the weights.
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

    def layer_call(
        self,
        layer: int,
        routes: torch.Tensor,
        *,
        gate_blocks: int | None = None,
        delay: bool = True,
        capturing: bool = False,
    ) -> None:
        """One layer call on the current stream (MAIN), without a host wait.

        It copies the lent routes right after the route wait and this rank's
        helper rows once the READY gate opens. Under capture, the graph
        writes those copies on every replay; record each replay with
        ``replayed``.
        """
        group = self.groups[layer]
        main = torch.cuda.current_stream()
        if delay and self.copy_delay_ns:
            # The kernel follows MAIN's work on COPY; hold it back after that.
            fork = torch.cuda.Event()
            fork.record(main)
            group.copy_stream.wait_event(fork)
            self.kernels.launch_delay(group.copy_stream, self.copy_delay_ns)
        physical, ready = group.plan(routes, defer_wait=False)
        if int(ready) != _READY:
            raise RuntimeError(f"layer {layer} lent READY {int(ready)}, not {_READY}")
        slots = physical.clone()
        blocks = self.gate_blocks if gate_blocks is None else gate_blocks
        self.kernels.launch_gate(self.ready_row(layer), main, blocks)
        rows = [row.clone() for row in self.helper_rows(layer)]
        group.finish()
        copies = (slots, rows)
        if capturing:
            self.graph_copies[layer] = copies
        else:
            self._launched(layer, copies)

    def replayed(self, layers: list) -> None:
        """Record one replay of a graph that captured ``layers``."""
        for layer in layers:
            self._launched(layer, self.graph_copies[layer])

    def _launched(self, layer: int, copies: tuple) -> None:
        self.launches[layer] += 1
        # A replay rewrites the copies of the same graph's earlier replays:
        # only the latest replay since the last check can be checked.
        self.pending[layer] = [None if c is copies else c for c in self.pending[layer]]
        self.pending[layer].append(copies)
        self.bank_user[self.banks[layer]] = layer

    def call(self, layer: int, routes: torch.Tensor, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            self.layer_call(layer, routes)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()

    def _apply(self, layer: int, routes: torch.Tensor) -> halo_r_reference.Plan:
        """The reference plan of one launch, applied to the model; returns the plan."""
        counts = torch.bincount(routes.flatten().long(), minlength=self.experts).cpu().tolist()
        histogram = [sum(c) for c in zip(*MPI.COMM_WORLD.allgather(counts))]
        table = self.tables[layer]
        plan = halo_r_reference.plan_halo_r(
            histogram, self.ep, _HELPER_SLOTS, send_count=4, residency=table
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

    def apply_unchecked(self, layer: int, routes: torch.Tensor) -> None:
        """Apply one launch to the model without checking it, for a launch the test tampered with."""
        if len(self.pending[layer]) != 1:
            raise RuntimeError("apply_unchecked takes exactly one launch")
        self.pending[layer] = []
        self._apply(layer, routes)

    def _route_failures(self, plan, routes: torch.Tensor, slots: torch.Tensor, where: str) -> list:
        """Every route goes to its owner or to a receiver's helper slot of its expert, at the plan's quota."""
        failures = []
        rank, local_slots = self.rank, _HOME_EXPERTS + _HELPER_SLOTS
        moved = {}
        for expert, slot in zip(routes.flatten().cpu().tolist(), slots.flatten().cpu().tolist()):
            destination, local = divmod(slot, local_slots)
            if slot < 0 or destination >= self.ep:
                failures.append(f"rank {rank}: {where}: expert {expert} routed to slot {slot}")
            elif local < _HOME_EXPERTS:
                if destination * _HOME_EXPERTS + local != expert:
                    failures.append(f"rank {rank}: {where}: expert {expert} in home slot {slot}")
            elif plan.helper_table[destination][local - _HOME_EXPERTS] != expert:
                failures.append(f"rank {rank}: {where}: expert {expert} in helper slot {slot}")
            else:
                moved[(expert, destination)] = moved.get((expert, destination), 0) + 1
        totals = {}
        for part in MPI.COMM_WORLD.allgather(moved):
            for key, value in part.items():
                totals[key] = totals.get(key, 0) + value
        expected = {(b.expert, d): rows for b in plan.broadcasts for d, rows in b.rows.items()}
        if totals != expected:
            failures.append(f"rank {rank}: {where}: moved routes {totals} != plan {expected}")
        return failures

    def _row_failures(self, layer: int, plan, rows: list, where: str) -> list:
        """Every helper row this rank received holds the owner's expert, byte for byte."""
        failures = []
        bound = self.arenas[layer].bound
        for helper, expert in enumerate(plan.helper_table[self.rank]):
            if expert < 0:
                continue
            for plane, nbytes in enumerate(bound.plane_bytes):
                nbytes = int(nbytes)
                weights = _pattern(expert, plane, self.weights[layer], nbytes, self.device)
                if not torch.equal(rows[plane][helper * nbytes : (helper + 1) * nbytes], weights):
                    failures.append(
                        f"rank {self.rank}: {where}: helper {helper} plane {plane} is not expert {expert}"
                    )
        return failures

    def check(self, layer: int, routes) -> list:
        """Check the launches of ``layer`` since the previous check, whose routes ``routes`` lists in order.

        A single tensor is one launch. Every launch updates the model and is
        checked against its own plan through its MAIN-side copies; the last
        one's plan, outputs and helper rows are also compared as they are now.
        """
        routes_list = [routes] if isinstance(routes, torch.Tensor) else list(routes)
        copies = self.pending[layer]
        if len(copies) != len(routes_list):
            raise RuntimeError(f"{len(routes_list)} routes for {len(copies)} launches")
        self.pending[layer] = []
        rank = self.rank
        failures = []
        first = self.launches[layer] - len(routes_list) + 1
        plan = None
        for index, (launch_routes, launch_copies) in enumerate(zip(routes_list, copies)):
            plan = self._apply(layer, launch_routes)
            if launch_copies is None:
                continue
            where = f"layer {layer} launch {first + index}"
            slots, rows = launch_copies
            failures += self._route_failures(
                plan, launch_routes, slots, f"{where} routes after the route wait"
            )
            failures += self._row_failures(layer, plan, rows, f"{where} helper rows at the gate")

        group = self.groups[layer]
        scheduler = group.scheduler
        routes = routes_list[-1]
        where = f"layer {layer} now"
        # Every launch of the layer is one scheduler epoch.
        if scheduler.plan_workspace.cpu().tolist() != plan.plan_words(epoch=self.launches[layer]):
            failures.append(f"rank {rank}: {where}: plan words differ from the reference")
        if scheduler.outputs.hot_expert_ids.cpu().tolist() != plan.helper_table[rank]:
            failures.append(f"rank {rank}: {where}: helper experts differ from the reference")
        if any(scheduler.status.cpu().tolist()):
            failures.append(f"rank {rank}: {where}: status words {scheduler.status.cpu().tolist()}")
        cells = self.ready_row(layer).cpu().tolist()
        if cells != [_READY] * self.ep:
            failures.append(f"rank {rank}: {where}: READY row {cells}")
        # Every route wait resets the counter that the launch's CTAs raised.
        counter = int(group._route_epoch.item())
        if counter != 0:
            failures.append(f"rank {rank}: {where}: route counter {counter} after the route waits")
        timed_out = int(self.kernels.timed_out.item())
        if timed_out:
            failures.append(f"rank {rank}: {where}: {timed_out} gate CTAs timed out on READY")
            self.kernels.timed_out.zero_()
        slots = scheduler.outputs.physical_slot_ids[: routes.shape[0]]
        failures += self._route_failures(plan, routes, slots, f"{where} routes")
        # A later layer on the same shared bank has refilled its helper rows since.
        if self.bank_user.get(self.banks[layer]) == layer:
            failures += self._row_failures(layer, plan, self.helper_rows(layer), f"{where} rows")

        counts = group.broadcaster.relay_counts()
        want = (self.launches[layer], self.receiver_copies[layer], self.receiver_skips[layer])
        if counts != want:
            failures.append(f"rank {rank}: {where}: relay counts {counts} != {want}")
        table = self.tables[layer]
        if table is not None:
            native = group.broadcaster.residency.cpu().tolist()
            if native != table:
                failures.append(f"rank {rank}: {where}: residency {native} != {table}")
            if any(other != native for other in MPI.COMM_WORLD.allgather(native)):
                failures.append(f"rank {rank}: {where}: residency differs across ranks")
        return failures

    def shutdown(self) -> None:
        if self.dead:
            # A trapped context cannot run the collective teardown.
            return
        torch.cuda.synchronize()
        self.manager.shutdown(local_safe=True)


def _case_reference(layers: _Layers) -> list:
    """Every launch against the reference, over ranks with different token counts."""
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    for seed in range(_SEEDS):
        for step in range(3):
            # The last rank has no tokens on step 1.
            last = rank == layers.ep - 1
            tokens = 0 if step == 1 and last else layers.max_tokens - 17 * (rank % 8)
            routes = layers.routes(100 * seed + step, tokens)
            layers.call(0, routes, main)
            failures += layers.check(0, routes)
    return failures


def _case_balanced_and_empty(layers: _Layers) -> list:
    """Balanced routes plan no broadcast, and ranks without tokens still publish READY."""
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    for tokens in (layers.max_tokens, 0):
        routes = _uniform_routes(rank, tokens, layers.experts, layers.device)
        layers.call(0, routes, main)
        if int(layers.groups[0].scheduler.plan_workspace[1]) != 0:
            failures.append(f"rank {rank}: a balanced load planned broadcasts")
        failures += layers.check(0, routes)
    return failures


def _case_default_stream(layers: _Layers) -> list:
    """MAIN may be the legacy default stream, as in an eager script without an execution stream."""
    main = torch.cuda.default_stream()
    failures = []
    for step in range(3):
        routes = layers.routes(step, layers.max_tokens - 17 * (layers.rank % 8))
        layers.call(0, routes, main)
        failures += layers.check(0, routes)
    return failures


def _case_alternating_layers(layers: _Layers) -> list:
    """Three layers: on two shared helper banks, layer 2 refills the bank of layer 0; per-layer
    helper slots give every layer its own set."""
    rank = layers.rank
    expected = [0, 1, 2] if layers.resident else [0, 1, 0]
    failures = [] if layers.banks == expected else [f"rank {rank}: helper sets {layers.banks}"]
    main = torch.cuda.Stream()
    for seed in range(_SEEDS):
        for step in range(2):
            for layer in range(len(layers.groups)):
                routes = layers.routes(1000 * seed + 10 * step + layer, layers.max_tokens)
                layers.call(layer, routes, main)
                failures += layers.check(layer, routes)
    return failures


def _case_resident_repeat(layers: _Layers) -> list:
    """Repeated routes copy nothing and keep the rows, even a row the test overwrote.

    The first rank that received an expert into helper slot 0 overwrites that
    row with a sentinel; the repeat keeps it, because the relay skips it. Once
    the table is reset (the weight reload hook), the next call restores the
    owner's bytes.
    """
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    routes = layers.routes(0, layers.max_tokens)
    layers.call(0, routes, main)
    failures += layers.check(0, routes)
    holders = [d for d in range(layers.ep) if layers.tables[0][d][0] >= 0]
    if not holders:
        raise _Skip(f"no rank receives a helper expert in these routes at EP {layers.ep}")
    sentinel_rank = holders[0]
    nbytes = int(layers.arenas[0].bound.plane_bytes[0])
    row = layers.helper_rows(0)[0][:nbytes]
    sentinel = torch.full((nbytes,), 0x5A, dtype=torch.uint8, device=layers.device)
    if rank == sentinel_rank:
        row.copy_(sentinel)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    layers.call(0, routes, main)
    # The sentinel row no longer holds the expert, so only the model is updated here.
    layers.apply_unchecked(0, routes)
    if layers.last_copies:
        failures.append(f"rank {rank}: the repeated call copied {layers.last_copies} receivers")
    if rank == sentinel_rank and not torch.equal(row, sentinel):
        failures.append(f"rank {rank}: the repeated call rewrote a resident helper row")
    layers.groups[0].forget_helper_residency()
    layers.tables[0] = [[-1] * _HELPER_SLOTS for _ in range(layers.ep)]
    MPI.COMM_WORLD.Barrier()
    layers.call(0, routes, main)
    failures += layers.check(0, routes)
    return failures


def _case_resident_sequence(layers: _Layers) -> list:
    """Random routes with evictions and returns; tables match a model and each other."""
    rank = layers.rank
    main = torch.cuda.Stream()
    rng = random.Random(7)
    failures = []
    steps = [rng.randrange(6) for _ in range(100)]
    for index, step in enumerate(steps):
        tokens = 0 if index % 17 == 5 and rank == 0 else layers.max_tokens
        routes = layers.routes(step, tokens)
        layers.call(0, routes, main)
        failures += layers.check(0, routes)
        if _any_rank(failures):
            break
    if layers.receiver_skips[0] == 0:
        failures.append(f"rank {rank}: no receiver was ever skipped")
    return failures


def _case_resident_reload(layers: _Layers) -> list:
    """A weight reload makes the next call copy every receiver again, with the new weights."""
    rank = layers.rank
    main = torch.cuda.Stream()
    failures = []
    routes = layers.routes(0, layers.max_tokens)
    for _ in range(2):
        layers.call(0, routes, main)
        failures += layers.check(0, routes)
    if layers.last_copies:
        failures.append(f"rank {rank}: the repeated call copied {layers.last_copies} receivers")
    layers.reload_weights(0)
    skips = layers.receiver_skips[0]
    layers.call(0, routes, main)
    # check() compares the rows with the new weights, and the counters and
    # the table with a model that forgot every slot.
    failures += layers.check(0, routes)
    if layers.last_copies == 0 or layers.receiver_skips[0] != skips:
        failures.append(f"rank {rank}: the call after the reload skipped receivers")
    return failures


def _case_shared_banks(layers: _Layers) -> list:
    """Shared banks keep no table and copy every receiver every time.

    check() compares the kernel's receiver-skip counter with the model's,
    which never skips without a table.
    """
    return _case_alternating_layers(layers)


def _case_stable_slot(layers: _Layers) -> list:
    """A resident expert keeps its slot when a lower-id expert joins its receiver.

    At EP 4, route step 2 sends expert 10 to slot 0 of rank 3, and step 13
    sends experts 1 and 10 there: in ascending order expert 10 would move to
    slot 1 and be copied again; resident-first keeps it in slot 0.
    """
    rank = layers.rank
    if layers.ep != 4:
        raise _Skip(f"the stable-slot routes are built for EP 4, not {layers.ep}")
    main = torch.cuda.Stream()
    failures = []
    skips = 0
    for step in (2, 13):
        routes = layers.routes(step, layers.max_tokens)
        skips = layers.receiver_skips[0]
        layers.call(0, routes, main)
        failures += layers.check(0, routes)
    hot = layers.groups[0].scheduler.outputs.hot_expert_ids.cpu().tolist()
    if layers.tables[0][3] != [10, 1] or (rank == 3 and hot != [10, 1]):
        failures.append(f"rank {rank}: rank 3's helper slots {layers.tables[0][3]}, here {hot}")
    if layers.receiver_skips[0] == skips:
        failures.append(f"rank {rank}: the resident expert was copied again")
    return failures


def _starve_relay(layers: _Layers) -> None:
    """Give COPY the lowest stream priority, below the MAIN of the hog cases.

    Production gives COPY the highest priority, which only orders CTAs that
    are still pending; inverting it makes MegaMoE win every SM that frees up
    while the HALO-R kernel is pending. Must run before the groups are built.
    """
    least, _ = torch.cuda.Stream.priority_range()
    copy_stream = torch.cuda.Stream(device=layers.device, priority=least)
    layers.manager.get_copy_stream = lambda device: copy_stream


def _case_gate_under_hog(layers: _Layers, use_graph: bool) -> list:
    """MegaMoE's full grid, queued right after the route wait, must not starve the relay.

    A hog holds every SM but the HALO-R kernel's, and COPY runs below MAIN, so
    a gate kernel that MAIN has already released takes every SM that frees up
    first. The route wait passes only once every HALO-R CTA has released its
    routes, after the grid rendezvous, so the relay already holds its SMs.
    """
    group = layers.groups[0]
    kernels = layers.kernels
    _, greatest = torch.cuda.Stream.priority_range()
    main = torch.cuda.Stream(device=layers.device, priority=greatest)
    side = torch.cuda.Stream(device=layers.device)
    # The group accepts only a MAIN below COPY; bind this MAIN directly.
    group._execution_stream_handle = int(main.cuda_stream)
    static = layers.routes(0, layers.max_tokens)

    def layer_call(capturing: bool = False) -> None:
        layers.layer_call(0, static, gate_blocks=kernels.sms, delay=False, capturing=capturing)

    failures = []
    graph = None
    if use_graph:
        main.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(main):
            # Warm up eagerly on the capture stream, as the breakable runner does.
            layer_call()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        failures += layers.check(0, static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main):
            layer_call(capturing=True)
    for step in range(1, 51):
        routes = layers.routes(step, layers.max_tokens)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        kernels.launch_hog(side, kernels.sms - group.auxiliary_sms)
        with torch.cuda.stream(main):
            static.copy_(routes)
            if graph is not None:
                graph.replay()
                layers.replayed([0])
            else:
                layer_call()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        failures += layers.check(0, routes)
    if graph is not None:
        graph.reset()
    return failures


def _capture(
    layers: _Layers, layer_ids: list, keep_graph: bool = False
) -> tuple[torch.cuda.CUDAGraph, dict[int, torch.Tensor], list[str]]:
    """Capture layer calls as the breakable runner does: warm up eagerly on its capture stream,
    then capture there. Returns the graph, the static routes per layer and the warmup's failures.
    ``keep_graph`` keeps the captured cudaGraph_t for inspection (``raw_cuda_graph``)."""
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
        failures += layers.check(layer, static[layer])
    graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    with torch.cuda.graph(graph, stream=capture):
        for layer in layer_ids:
            layers.layer_call(layer, static[layer], capturing=True)
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


def _case_graphs_single(layers: _Layers) -> list:
    """One layer captured once, then 50 replays with fresh routes in the static input."""
    graph, static, failures = _capture(layers, [0])
    for step in range(1, 51):
        routes = layers.routes(step, layers.max_tokens)
        with torch.cuda.stream(layers.eager[0]):
            static[0].copy_(routes)
            graph.replay()
        layers.replayed([0])
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        failures += layers.check(0, routes)
    failures += _replay_counts(layers, replays=50)
    graph.reset()
    return failures


def _case_graphs_layers(layers: _Layers) -> list:
    """Three layers in one graph, replayed with fresh routes."""
    count = len(layers.groups)
    graph, static, failures = _capture(layers, list(range(count)))
    for step in range(1, 11):
        routes = [layers.routes(100 * step + layer, layers.max_tokens) for layer in range(count)]
        with torch.cuda.stream(layers.eager[0]):
            for layer in range(count):
                static[layer].copy_(routes[layer])
            graph.replay()
        layers.replayed(list(range(count)))
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        for layer in range(count):
            failures += layers.check(layer, routes[layer])
    graph.reset()
    return failures


def _case_graphs_interleaved(layers: _Layers) -> list:
    """Eager calls and replays interleaved on the same layer, with MAIN changing.

    Each step queues one to three launches without a host wait: eager calls on
    one of two MAIN streams and replays on the current one, so the group's
    MAIN-to-MAIN fence, the per-call join and the route-counter reset order
    them. Every eager launch and the step's last replay are checked through
    their MAIN-side copies; the model applies every launch.
    """
    graph, static, failures = _capture(layers, [0])
    rng = random.Random(11)
    streams = layers.eager
    # Replays run on the group's current MAIN, as the runner replays on the
    # execution stream that its eager calls use; only eager calls change MAIN.
    # MAIN is still the capture stream, so an eager call first makes it
    # streams[0]: a replay there before it would not be fenced from the next
    # eager call, whatever the seed.
    main = 0
    replays = 0
    first = layers.routes(999, layers.max_tokens)
    with torch.cuda.stream(streams[main]):
        layers.layer_call(0, first)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    failures += layers.check(0, [first])
    for step in range(200):
        launches = []
        for _ in range(1 + rng.randrange(3)):
            routes = layers.routes(1000 + 7 * step + len(launches), layers.max_tokens)
            if rng.random() < 0.5:
                with torch.cuda.stream(streams[main]):
                    static[0].copy_(routes)
                    graph.replay()
                layers.replayed([0])
                replays += 1
            else:
                if rng.random() < 0.4:
                    main ^= 1
                with torch.cuda.stream(streams[main]):
                    layers.layer_call(0, routes)
            launches.append(routes)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        failures += layers.check(0, launches)
        if _any_rank(failures):
            break
    failures += _replay_counts(layers, replays=replays)
    graph.reset()
    return failures


def _case_graphs_capturing_switch(layers: _Layers) -> list:
    """MAIN cannot change into a stream that is capturing."""
    rank = layers.rank
    group = layers.groups[0]
    failures = []
    main = torch.cuda.Stream(device=layers.device)
    routes = layers.routes(0, layers.max_tokens)
    layers.call(0, routes, main)
    failures += layers.check(0, routes)
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
    layers.call(0, routes, main)
    failures += layers.check(0, routes)
    return failures


class _CaptureFailure(Exception):
    """Stands for an error raised inside a captured layer call, such as a graph-pool OOM."""


def _case_graphs_failed_capture(layers: _Layers) -> list:
    """A failure ends the capture between a layer's launch and its route wait.

    The abort path then releases the plan eagerly on the stream that was
    capturing. That must not wait for the captured kernel, which never runs,
    and the group must refuse its next launch until it is rebuilt.
    """
    rank = layers.rank
    group = layers.groups[0]
    failures = []
    capture = torch.cuda.Stream(device=layers.device)
    routes = layers.routes(0, layers.max_tokens)
    # Warm up eagerly on the capture stream, as the breakable runner does.
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        layers.layer_call(0, routes)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    failures += layers.check(0, routes)
    graph = torch.cuda.CUDAGraph()
    previous = torch.cuda.current_stream()
    try:
        with torch.cuda.graph(graph, stream=capture):
            group.plan(routes, defer_wait=True)
            raise _CaptureFailure()
    except _CaptureFailure:
        pass
    except RuntimeError as error:
        # The launch forked COPY into the capture and nothing joined it back, so
        # ending the capture fails (cudaErrorStreamCaptureUnjoined), chained to
        # the failure that ended it, as in the breakable runner.
        if not isinstance(error.__context__, _CaptureFailure):
            raise
    # A failed capture end skips restoring the current stream.
    torch.cuda.set_stream(previous)
    if torch.cuda.is_current_stream_capturing():
        failures.append(f"rank {rank}: a stream is still capturing after the failed capture")
    with torch.cuda.stream(capture):
        group.wait_for_routes()
        group.finish()
    # Without the fix, an eager route wait on the capture stream never passes and this hangs.
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    try:
        with torch.cuda.stream(capture):
            group.plan(routes, defer_wait=True)
        failures.append(f"rank {rank}: the group launched again after its capture failed")
    except RuntimeError as error:
        if "rebuild" not in str(error):
            failures.append(f"rank {rank}: unexpected error {error}")
    return failures


def _driver_call(result) -> Any:
    """Unpack a cuda.bindings driver call; raise on an error."""
    from cuda.bindings import driver

    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {error!r}")
    return values[0] if len(values) == 1 else values


class _GraphView:
    """The nodes and edges of a captured graph, with its HALO-R kernels and route waits.

    The route wait is the graph's only batched stream memory operation.
    """

    def __init__(self, raw) -> None:
        from cuda.bindings import driver

        count = _driver_call(driver.cuGraphGetNodes(raw, 0))[1]
        nodes = _driver_call(driver.cuGraphGetNodes(raw, count))[0][:count]
        self.nodes = [int(node) for node in nodes]
        self.handles = dict(zip(self.nodes, nodes))
        self.halo_kernels = set()
        self.route_waits = set()
        self.kernels = set()
        for node in nodes:
            kind = _driver_call(driver.cuGraphNodeGetType(node))
            if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP:
                self.route_waits.add(int(node))
            elif kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                self.kernels.add(int(node))
                params = _driver_call(driver.cuGraphKernelNodeGetParams(node))
                if int(params.func):
                    name = _driver_call(driver.cuFuncGetName(params.func))
                else:
                    name = _driver_call(driver.cuKernelGetName(params.kern))
                if "haloRFusedKernel" in (name.decode() if isinstance(name, bytes) else str(name)):
                    self.halo_kernels.add(int(node))
        count = _driver_call(driver.cuGraphGetEdges(raw, 0))[3]
        sources, targets, data, count = _driver_call(driver.cuGraphGetEdges(raw, count))
        # (from, to, from_port); port 0 is full completion of the upstream node.
        self.edges = [
            (int(sources[i]), int(targets[i]), int(data[i].from_port)) for i in range(count)
        ]

    def ancestors(self, node: int, full_only: bool) -> set:
        """Every node with a path to ``node``; ``full_only`` follows full-completion edges only."""
        parents = {}
        for source, target, port in self.edges:
            if not full_only or port == 0:
                parents.setdefault(target, []).append(source)
        seen, stack = set(), [node]
        while stack:
            for parent in parents.get(stack.pop(), []):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def adversarial_order(self) -> list:
        """A topological order that runs every node CUDA lets run early as early as it can:
        among ready nodes, non-kernel nodes first, then kernels, HALO-R kernels last."""
        indegree = {node: 0 for node in self.nodes}
        children = {}
        for source, target, _ in self.edges:
            indegree[target] += 1
            children.setdefault(source, []).append(target)
        rank = {node: index for index, node in enumerate(self.nodes)}

        def priority(node: int) -> tuple:
            return (node in self.kernels, node in self.halo_kernels, rank[node])

        ready = sorted((node for node, degree in indegree.items() if degree == 0), key=priority)
        order = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in children.get(node, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort(key=priority)
        if len(order) != len(self.nodes):
            raise RuntimeError("the captured graph has a cycle")
        return order


def _case_graphs_route_wait_edge(layers: _Layers) -> list:
    """In a captured graph, every route wait follows its own HALO-R kernel through the launch's completion event.

    The kernel writes the route counter that the wait spins on, which CUDA does
    not see. Without a graph edge from the kernel, the wait and the kernel sit
    on sibling branches, and a schedule that runs the wait first deadlocks with
    no trap. The edge must leave the kernel's launch completion port, not wait
    for the kernel's end, or MAIN would lose the relay's overlap.
    """
    from cuda.bindings import runtime

    count = len(layers.groups)
    graph, _, failures = _capture(layers, list(range(count)), keep_graph=True)
    view = _GraphView(runtime.cudaGraph_t(init_value=graph.raw_cuda_graph()))
    rank = layers.rank
    if len(view.halo_kernels) != count or len(view.route_waits) != count:
        failures.append(
            f"rank {rank}: {len(view.halo_kernels)} HALO-R kernels and {len(view.route_waits)} "
            f"route waits captured, expected {count} of each"
        )
        graph.reset()
        return failures
    # The i-th layer's route wait follows the kernels of layers 0..i (its own through the launch completion port)
    # and, by full completion, only those of the earlier layers, through their joins.
    reached = []
    for wait in view.route_waits:
        every = view.ancestors(wait, full_only=False) & view.halo_kernels
        full = view.ancestors(wait, full_only=True) & view.halo_kernels
        reached.append((len(every), len(full)))
    reached.sort()
    expected = [(layer + 1, layer) for layer in range(count)]
    if reached != expected:
        failures.append(
            f"rank {rank}: (HALO-R kernels before each route wait, of them by full completion) {reached}, "
            f"expected {expected}"
        )
    graph.reset()
    return failures


# A serialized replay that has not finished after this long is deadlocked; a healthy one takes milliseconds.
_SERIALIZED_TIMEOUT_S = 60


def _case_graphs_serialized(layers: _Layers) -> list:
    """Replay the captured layers in one adversarial serial schedule; it must finish and copy correctly.

    CUDA may run the independent branches of a graph in any order, one after
    the other. The clone of the captured graph is chained along
    ``_GraphView.adversarial_order``, which runs every node as early as its
    edges allow, so a route wait without an edge from its kernel runs first
    and never passes.
    """
    from cuda.bindings import driver, runtime

    count = len(layers.groups)
    rank = layers.rank
    graph, static, failures = _capture(layers, list(range(count)), keep_graph=True)
    raw = runtime.cudaGraph_t(init_value=graph.raw_cuda_graph())
    clone = _driver_call(driver.cuGraphClone(raw))
    view = _GraphView(clone)
    linked = {(source, target) for source, target, _ in view.edges}
    order = view.adversarial_order()
    sources, targets = [], []
    for source, target in zip(order, order[1:]):
        if (source, target) not in linked:
            sources.append(view.handles[source])
            targets.append(view.handles[target])
    if sources:
        _driver_call(driver.cuGraphAddDependencies(clone, sources, targets, None, len(sources)))
    executable = _driver_call(driver.cuGraphInstantiate(clone, 0))
    stream = layers.eager[0]
    handle = driver.CUstream(init_value=int(stream.cuda_stream))
    for step in range(1, 21):
        routes = [layers.routes(100 * step + layer, layers.max_tokens) for layer in range(count)]
        with torch.cuda.stream(stream):
            for layer in range(count):
                static[layer].copy_(routes[layer])
            _driver_call(driver.cuGraphLaunch(executable, handle))
        deadline = time.monotonic() + _SERIALIZED_TIMEOUT_S
        while not stream.query() and time.monotonic() <= deadline:
            time.sleep(0.01)
        timed_out = not stream.query()
        # Every rank learns of a deadlock anywhere, so no rank waits for a peer that will not return.
        if MPI.COMM_WORLD.allreduce(timed_out, op=MPI.LOR):
            # The context cannot be torn down behind a deadlocked replay.
            layers.dead = True
            failures.append(
                f"rank {rank}: serialized replay {step} "
                + (
                    f"did not finish within {_SERIALIZED_TIMEOUT_S} s"
                    if timed_out
                    else "deadlocked on a peer"
                )
            )
            # Process exit behind a deadlocked graph may hang until the worker's watchdog; keep the failure.
            print(failures[-1], file=sys.stderr, flush=True)
            return failures
        layers.replayed(list(range(count)))
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        for layer in range(count):
            failures += layers.check(layer, routes[layer])
        if _any_rank(failures):
            break
    _driver_call(driver.cuGraphExecDestroy(executable))
    _driver_call(driver.cuGraphDestroy(clone))
    graph.reset()
    return failures


def _break_launch(layers: _Layers, kind: str, layer: int, main: torch.cuda.Stream) -> None:
    """Queue on MAIN what makes the next launch of ``layer`` on this rank fail, before its routes."""
    if kind == "status_word":
        # The invalid-route status word, set with a nonzero epoch.
        layers.groups[layer].scheduler.status[3].fill_(5)
    elif kind == "ready_drain_timeout":
        # The last rank's READY of the previous launch never arrived.
        layers.ready_row(layer)[layers.ep - 1].zero_()
    elif kind == "grid_rendezvous_timeout":
        # One SM fewer than the kernel's CTAs stays free for longer than the spin bound.
        side = torch.cuda.Stream(device=layers.device)
        kernels = layers.kernels
        kernels.launch_hog(side, kernels.sms - layers.auxiliary_sms + 1, _TRAP_HOG_NS)
        # Let the hog take its SMs before the HALO-R kernel launches.
        time.sleep(0.5)
    else:
        raise ValueError(f"unknown trap {kind}")


def _case_trap(layers: _Layers, kind: str) -> dict:
    """Rank 0 breaks one launch of layer 1; it traps in that call, before it releases routes.

    Its peers trap in that call or the next, at their wait for rank 0, within
    the spin bound. Returns this rank's first error, the call that raised it
    and its delay.
    """
    rank = layers.rank
    main = torch.cuda.Stream()
    for forward in range(_TRAP_CALL[0]):
        for layer in range(2):
            layers.call(layer, layers.routes(10 * forward + layer, layers.max_tokens), main)
    layers.manager.get_scheduler_spin_bound().set_timeout_seconds(_TRAP_TIMEOUT_S)
    layers.copy_delay_ns = 0
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    start = time.monotonic()
    error = None
    where = None
    for forward in range(_TRAP_CALL[0], _TRAP_CALL[0] + 4):
        for layer in range(2):
            try:
                with torch.cuda.stream(main):
                    if rank == 0 and (forward, layer) == _TRAP_CALL:
                        _break_launch(layers, kind, layer, main)
                    layers.layer_call(layer, layers.routes(10 * forward + layer, layers.max_tokens))
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001  the trap surfaces as a CUDA error
                error, where = f"{type(exc).__name__}: {exc}", (forward, layer)
                break
        if error is not None:
            break
    layers.dead = error is not None
    return {"rank": rank, "error": error, "where": where, "seconds": time.monotonic() - start}


# case: (case function, layers, auxiliary SMs, helper-slot mode)
_CASES = {
    "reference": (_case_reference, 1, _AUXILIARY_SMS, "shared_banks"),
    "reference_aux12": (_case_reference, 1, 12, "shared_banks"),
    "reference_aux16": (_case_reference, 1, 16, "shared_banks"),
    "reference_production_bundle": (_case_reference, 1, _AUXILIARY_SMS, "shared_banks"),
    "balanced_and_empty": (_case_balanced_and_empty, 1, _AUXILIARY_SMS, "shared_banks"),
    "default_stream": (_case_default_stream, 1, _AUXILIARY_SMS, "shared_banks"),
    "resident_repeat": (_case_resident_repeat, 1, _AUXILIARY_SMS, "per_layer_resident"),
    "resident_sequence": (_case_resident_sequence, 1, _AUXILIARY_SMS, "per_layer_resident"),
    "resident_layers": (_case_alternating_layers, 3, _AUXILIARY_SMS, "per_layer_resident"),
    "resident_reload": (_case_resident_reload, 1, _AUXILIARY_SMS, "per_layer_resident"),
    "shared_banks": (_case_shared_banks, 3, _AUXILIARY_SMS, "shared_banks"),
    "stable_slot": (_case_stable_slot, 1, _AUXILIARY_SMS, "per_layer_resident"),
    "gate_under_hog": (
        lambda layers: _case_gate_under_hog(layers, use_graph=False),
        1,
        _AUXILIARY_SMS,
        "shared_banks",
    ),
    "gate_under_hog_aux16": (
        lambda layers: _case_gate_under_hog(layers, use_graph=False),
        1,
        16,
        "shared_banks",
    ),
    "gate_under_hog_graph": (
        lambda layers: _case_gate_under_hog(layers, use_graph=True),
        1,
        _AUXILIARY_SMS,
        "shared_banks",
    ),
    "graphs_single": (_case_graphs_single, 1, _AUXILIARY_SMS, "per_layer_resident"),
    "graphs_layers_resident": (_case_graphs_layers, 3, _AUXILIARY_SMS, "per_layer_resident"),
    "graphs_layers_shared": (_case_graphs_layers, 3, _AUXILIARY_SMS, "shared_banks"),
    "graphs_interleaved": (_case_graphs_interleaved, 1, _AUXILIARY_SMS, "per_layer_resident"),
    "graphs_capturing_switch": (
        _case_graphs_capturing_switch,
        1,
        _AUXILIARY_SMS,
        "shared_banks",
    ),
    "graphs_route_wait_edge": (
        _case_graphs_route_wait_edge,
        3,
        _AUXILIARY_SMS,
        "per_layer_resident",
    ),
    "graphs_serialized": (_case_graphs_serialized, 3, _AUXILIARY_SMS, "per_layer_resident"),
    "graphs_failed_capture": (_case_graphs_failed_capture, 1, _AUXILIARY_SMS, "shared_banks"),
}
_TRAPS = ("status_word", "ready_drain_timeout", "grid_rendezvous_timeout")
for _kind in _TRAPS:
    _CASES[f"trap_{_kind}"] = (
        lambda layers, kind=_kind: _case_trap(layers, kind),
        2,
        _AUXILIARY_SMS,
        "shared_banks",
    )


def _run_worker(case: str) -> tuple[str | None, list[str]]:
    """One EP rank of a case; returns (skip reason or None, result)."""
    from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size

    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True)
    rank, world = mpi_rank(), mpi_world_size()
    device = local_mpi_rank()
    layers = None
    try:
        torch.cuda.set_device(device)
        function, count, auxiliary_sms, mode = _CASES[case]
        reason = _skip_reason()
        if reason is None and world != _EP_SIZE:
            reason = f"the MPI world has {world} ranks, HALO_R_TEST_EP is {_EP_SIZE}"
        reasons = MPI.COMM_WORLD.allgather(reason)
        if any(reasons):
            return "; ".join(sorted({r for r in reasons if r})), []
        _init_process_group(rank, world)
        dims = (
            _PRODUCTION_BUNDLE if case.endswith("production_bundle") else (_HIDDEN, _INTERMEDIATE)
        )
        layers = _Layers(rank, world, device, count, auxiliary_sms, mode, dims)
        if case.startswith("gate_under_hog"):
            _starve_relay(layers)
        try:
            layers.build_groups()
        except RuntimeError as error:
            # Every rank reaches the native context creation together, so every rank skips.
            if _PLACEHOLDER_MESSAGE in str(error):
                prefix = _BUILD_SKIP if _cuda_at_least(13, 1) else "placeholder build: "
                return f"{prefix}{error}", []
            raise
        try:
            return None, function(layers)
        except _Skip as skip:
            return str(skip), []
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


def _run_on_every_rank(monkeypatch, case: str) -> list:
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
            expression = "__import__('cloudpickle').loads(worker)(case)"
            names = {"worker": cloudpickle.dumps(_run_worker), "case": case}
            return client.submit_sync(eval, expression, names)  # nosec B307
        if torch.cuda.device_count() < _EP_SIZE:
            _skip(f"Requires {_EP_SIZE} GPUs, or trtllm-llmapi-launch across nodes")
        # Per-layer dynamic EPLB needs the larger launch queues from CUDA initialization on;
        # the workers get them explicitly rather than by inheriting this process's environment.
        with MPIPoolExecutor(_EP_SIZE, env={"CUDA_SCALE_LAUNCH_QUEUES": "4x"}) as executor:
            return list(executor.map(_run_worker, [case] * _EP_SIZE))
    finally:
        faulthandler.cancel_dump_traceback_later()


def _skip_if_any(results: list) -> None:
    skips = {skip for skip, _ in results if skip}
    if skips:
        _skip("; ".join(sorted(skips)))


def _skip(reason: str) -> None:
    if os.environ.get("HALO_R_TEST_NO_SKIP") == "1":
        pytest.fail(f"HALO_R_TEST_NO_SKIP=1 but the case would skip: {reason}")
    if _BUILD_SKIP in reason and os.environ.get("HALO_R_TEST_ALLOW_SKIP") != "1":
        # On SM100 with CUDA 13.1+, a placeholder build is a broken build, not a reason to skip.
        pytest.fail(f"{reason} (set HALO_R_TEST_ALLOW_SKIP=1 to skip)")
    pytest.skip(reason)


@pytest.mark.parametrize("case", [case for case in _CASES if not case.startswith("trap_")])
def test_halo_r(monkeypatch, case) -> None:
    results = _run_on_every_rank(monkeypatch, case)
    _skip_if_any(results)
    failures = [failure for _, rank_failures in results for failure in rank_failures]
    assert not failures, "\n".join(failures[:20])


@pytest.mark.skipif(
    os.environ.get("HALO_R_TEST_TRAPS") != "1",
    reason="faults every GPU of the node; set HALO_R_TEST_TRAPS=1 to run",
)
@pytest.mark.parametrize("kind", _TRAPS)
def test_halo_r_traps(monkeypatch, kind) -> None:
    """A broken launch on rank 0 traps it in that call; every peer traps in that call or the next."""
    results = _run_on_every_rank(monkeypatch, f"trap_{kind}")
    _skip_if_any(results)
    outcomes = {result["rank"]: result for _, result in results}
    assert sorted(outcomes) == list(range(_EP_SIZE)), outcomes
    failures = []
    for rank, outcome in sorted(outcomes.items()):
        if outcome["error"] is None or "CUDA" not in outcome["error"]:
            failures.append(f"rank {rank} did not trap: {outcome}")
        elif rank == 0 and outcome["where"] != _TRAP_CALL:
            failures.append(f"rank 0 trapped at {outcome['where']}, not at {_TRAP_CALL}: {outcome}")
        elif rank != 0 and outcome["where"] not in (_TRAP_CALL, (_TRAP_CALL[0] + 1, 0)):
            failures.append(f"rank {rank} trapped at {outcome['where']}: {outcome}")
    assert not failures, "\n".join(failures)
