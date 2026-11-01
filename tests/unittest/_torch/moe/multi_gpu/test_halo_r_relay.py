# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank HALO-R kernel on a real shared-slot arena.

Every rank fills its home experts with a byte pattern unique to (expert,
plane, layer) and runs HALO-R layer calls through the scheduler group: over
three layers that alternate the two shared helper banks or keep their own
helper slots, with ranks that have no tokens, with a plan that moves nothing,
and with repeated routes whose experts are still resident. After each call it
checks:
- the plan words equal the pure-Python reference plan of the global routes;
- every route lands on its owner's home slot or a receiver's helper slot, and
  the routes per (expert, rank) add up to the plan's quota;
- every helper row this rank received equals the owner's expert bytewise;
- the READY row holds the call's generation on every cell, and every status
  word is clear;
- the relay counters count every launch and the copied and skipped receivers
  of every broadcast, and with per-layer helper slots the residency table is
  the same on every rank and equals a model of it.
The hog cases put a full-device kernel that spins on the READY row, like
MegaMoE's gate, on MAIN right after the route wait, while a hog holds every
SM but the HALO-R kernel's, and expects it to pass its gate every time.
"""

import ctypes
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
_AUXILIARY_SMS = 4
# Small planes stay in the per-layer record pages; the weight planes exceed
# them and live in the shared helper banks, so both relay paths are covered.
_HIDDEN = 256
_INTERMEDIATE = 128
_WORKER_TIMEOUT_S = 300
# The READY-gate stand-in gives up after this long; the relay finishes in milliseconds.
_GATE_TIMEOUT_NS = 5_000_000_000
# The SM hog outlives the HALO-R kernel's scheduler role.
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
        trtllm_ops, "moe_rebalance_halo_r_submit_fused"
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


def _check_call(
    group, arena, routes: torch.Tensor, generation: int, rank: int, layer: int = 0
) -> list:
    """Compare one finished layer call with the reference; returns failure strings."""
    failures = []
    experts = _HOME_EXPERTS * _EP_SIZE
    local_slots = _HOME_EXPERTS + _HELPER_SLOTS
    scheduler = group.scheduler
    counts = torch.bincount(routes.flatten().long(), minlength=experts).cpu().tolist()
    histogram = [sum(c) for c in zip(*MPI.COMM_WORLD.allgather(counts))]
    plan = halo_r_reference.plan_halo_r(
        histogram, _EP_SIZE, _HELPER_SLOTS, send_count=4, send_weight=0
    )

    # Every launch of the layer is one scheduler epoch, so the plan's epoch is the generation.
    words = scheduler.plan_workspace.cpu().tolist()
    if words != plan.plan_words(epoch=generation):
        failures.append(
            f"rank {rank}: plan words differ from the reference at generation {generation}"
        )
    if scheduler.outputs.hot_expert_ids.cpu().tolist() != plan.helper_table[rank]:
        failures.append(f"rank {rank}: helper experts differ from the reference")
    if any(scheduler.status.cpu().tolist()):
        failures.append(f"rank {rank}: status words {scheduler.status.cpu().tolist()}")
    ready = arena.terminal_flags_tensor().view(torch.int64).cpu().tolist()
    if ready != [generation] * _EP_SIZE:
        failures.append(f"rank {rank}: READY row {ready}, expected generation {generation}")

    # Every route goes to its owner or to a receiver's helper slot of its expert.
    slots = scheduler.outputs.physical_slot_ids[: routes.shape[0]].cpu().flatten().tolist()
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
    return failures, plan


# A build without the SM100 + CUDA 13.1 kernels registers the ops but throws on use.
_PLACEHOLDER_MESSAGE = "HALO-R requires CUDA 13.1+"


class _Layers:
    """HALO-R groups of consecutive MoE layers on one model-level manager."""

    def __init__(
        self, rank: int, count: int, auxiliary_sms: int, helper_slot_mode: str = "shared_banks"
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
            auxiliary_sms=auxiliary_sms,
            helper_slot_mode=helper_slot_mode,
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
        self.receiver_copies = []
        self.receiver_skips = []
        # A model of each layer's residency table, [rank][helper slot], with per-layer helper slots.
        self.tables = []
        self.resident = helper_slot_mode == "per_layer_resident"

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
            self.receiver_copies.append(0)
            self.receiver_skips.append(0)
            self.tables.append(
                [[-1] * _HELPER_SLOTS for _ in range(_EP_SIZE)] if self.resident else None
            )

    def call(self, layer: int, routes: torch.Tensor, stream: torch.cuda.Stream) -> int:
        group = self.groups[layer]
        with torch.cuda.stream(stream):
            _, generation = group.plan(routes, defer_wait=False)
            group.finish()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        return generation

    def check(self, layer: int, routes: torch.Tensor, generation: int) -> list:
        """Check one finished call; return the failures and the receivers it copied to."""
        group, arena = self.groups[layer], self.arenas[layer]
        failures, plan = _check_call(group, arena, routes, generation, self.rank, layer=layer)
        table = self.tables[layer]
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
        counts = group.broadcaster.relay_counts()
        expected = (generation, self.receiver_copies[layer], self.receiver_skips[layer])
        if counts != expected:
            failures.append(f"rank {self.rank}: layer {layer} relay counts {counts} != {expected}")
        if table is not None:
            native = group.broadcaster.residency.cpu().tolist()
            if native != table:
                failures.append(f"rank {self.rank}: layer {layer} residency {native} != {table}")
            if any(other != native for other in MPI.COMM_WORLD.allgather(native)):
                failures.append(f"rank {self.rank}: layer {layer} residency differs across ranks")
        self.last_copies = copies
        return failures

    def shutdown(self) -> None:
        torch.cuda.synchronize()
        self.manager.shutdown(local_safe=True)


def _case_single_layer(layers: _Layers) -> list:
    rank, experts = layers.rank, layers.experts
    main = torch.cuda.Stream()
    failures = []
    for step in range(3):
        # Ranks submit different token counts; the last rank has none on step 1.
        tokens = 0 if step == 1 and rank == _EP_SIZE - 1 else _MAX_TOKENS - 17 * rank
        routes = _routes(step, rank, tokens, experts, rank)
        failures += layers.check(0, routes, layers.call(0, routes, main))
    return failures


def _case_empty_plan(layers: _Layers) -> list:
    """Balanced routes plan no broadcast, and ranks without tokens still publish READY."""
    rank, experts = layers.rank, layers.experts
    main = torch.cuda.Stream()
    failures = []
    for tokens in (_MAX_TOKENS, 0):
        routes = _uniform_routes(rank, tokens, experts, rank)
        generation = layers.call(0, routes, main)
        if int(layers.groups[0].scheduler.plan_workspace[1]) != 0:
            failures.append(f"rank {rank}: a balanced load planned broadcasts")
        failures += layers.check(0, routes, generation)
    return failures


def _case_alternating_banks(layers: _Layers) -> list:
    """Three layers on two shared helper banks: layer 2 refills the bank of layer 0.

    With per-layer helper slots, every layer has its own set instead.
    """
    rank, experts = layers.rank, layers.experts
    banks = [int(arena.provider.layer.helper_set) for arena in layers.arenas]
    expected = [0, 1, 2] if layers.resident else [0, 1, 0]
    failures = [] if banks == expected else [f"rank {rank}: helper sets {banks}"]
    main = torch.cuda.Stream()
    for step in range(2):
        for layer in range(len(layers.groups)):
            routes = _routes(10 * step + layer, rank, _MAX_TOKENS, experts, rank)
            failures += layers.check(layer, routes, layers.call(layer, routes, main))
    return failures


def _case_resident_repeat(layers: _Layers) -> list:
    """Per-layer helper slots skip every receiver that still holds its expert.

    The same routes twice copy nothing the second time and keep the rows; other
    routes in between evict and return experts, once with a rank without
    tokens. The checks compare the native tables with a model on every call.
    """
    rank, experts = layers.rank, layers.experts
    main = torch.cuda.Stream()
    failures = []
    # (route step, tokens of rank 0); the other ranks always submit _MAX_TOKENS.
    calls = ((0, _MAX_TOKENS), (0, _MAX_TOKENS), (1, _MAX_TOKENS), (1, 0), (0, _MAX_TOKENS))
    for index, (step, rank0_tokens) in enumerate(calls):
        tokens = rank0_tokens if rank == 0 else _MAX_TOKENS
        routes = _routes(step, rank, tokens, experts, rank)
        failures += layers.check(0, routes, layers.call(0, routes, main))
        if index == 1 and layers.last_copies:
            failures.append(f"rank {rank}: the repeated call copied {layers.last_copies} receivers")
    return failures


def _case_gate_under_hog(layers: _Layers, auxiliary_sms: int) -> list:
    """MegaMoE's full grid, queued right after the route wait, must not starve the relay.

    A hog holds every SM but the HALO-R kernel's, and COPY runs below MAIN, so
    a gate kernel that MAIN has released takes every SM that frees up first.
    The route wait passes only once every HALO-R CTA has released its routes,
    after the grid rendezvous, so the relay already holds its SMs.
    """
    rank, experts = layers.rank, layers.experts
    group = layers.groups[0]
    kernels = _GateKernels(rank)
    ready = layers.arenas[0].terminal_flags_tensor().view(torch.int64)
    _, greatest = torch.cuda.Stream.priority_range()
    main = torch.cuda.Stream(device=rank, priority=greatest)
    side = torch.cuda.Stream(device=rank)
    # The group accepts only a MAIN below COPY; bind this MAIN directly.
    group._execution_stream_handle = int(main.cuda_stream)
    failures = []
    for step in range(1, 51):
        routes = _routes(step, rank, _MAX_TOKENS, experts, rank)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        kernels.launch_hog(side, auxiliary_sms)
        with torch.cuda.stream(main):
            _, generation = group.plan(routes, defer_wait=False)
            kernels.launch_gate(ready, generation, main)
            group.finish()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        timed_out = int(kernels.timed_out.item())
        if timed_out:
            failures.append(f"rank {rank}: step {step}: {timed_out} gate CTAs timed out on READY")
            kernels.timed_out.zero_()
        failures += layers.check(0, routes, generation)
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
        auxiliary_sms = 16 if case.endswith("_16") else _AUXILIARY_SMS
        helper_slot_mode = "per_layer_resident" if case.startswith("resident") else "shared_banks"
        count = 3 if case in ("banks", "resident_layers") else 1
        layers = _Layers(rank, count, auxiliary_sms, helper_slot_mode)
        if case.startswith("hog"):
            _starve_relay(layers)
        try:
            layers.build_groups()
        except RuntimeError as error:
            # Every rank reaches the native context creation together, so every rank skips.
            if _PLACEHOLDER_MESSAGE in str(error):
                return f"placeholder build: {error}", []
            raise
        if case in ("banks", "resident_layers"):
            return None, _case_alternating_banks(layers)
        if case == "empty_plan":
            return None, _case_empty_plan(layers)
        if case == "resident":
            return None, _case_resident_repeat(layers)
        if case.startswith("hog"):
            return None, _case_gate_under_hog(layers, auxiliary_sms)
        return None, _case_single_layer(layers)
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


def _starve_relay(layers: _Layers) -> None:
    """Give COPY the lowest stream priority, below the MAIN of the hog cases.

    Production gives COPY the highest priority, which only orders CTAs that
    are still pending; inverting it makes MegaMoE win every SM that frees up
    while the HALO-R kernel is pending. Must run before the groups are built.
    """
    least, _ = torch.cuda.Stream.priority_range()
    copy_stream = torch.cuda.Stream(device=layers.rank, priority=least)
    layers.manager.get_copy_stream = lambda device: copy_stream


@pytest.mark.skipif(torch.cuda.device_count() < _EP_SIZE, reason="Requires 4 GPUs")
@pytest.mark.parametrize(
    "case",
    [
        "eager",
        "eager_16",
        "empty_plan",
        "banks",
        "hog",
        "hog_16",
        "resident",
        "resident_layers",
    ],
)
def test_halo_r_relay_matches_reference(halo_r_mpi_pool, case):
    results = list(halo_r_mpi_pool.map(_run_worker, [case] * _EP_SIZE))
    skips = {skip for skip, _ in results if skip}
    if skips:
        pytest.skip("; ".join(sorted(skips)))
    failures = [failure for _, rank_failures in results for failure in rank_failures]
    assert not failures, "\n".join(failures[:20])
