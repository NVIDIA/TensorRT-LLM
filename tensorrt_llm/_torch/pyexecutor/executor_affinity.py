# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dedicated cores for the executor thread (port of the M3.1 `m31_hoststall` PIN part).

`TRTLLM_EXECUTOR_THREAD_PIN=1` (default off). On its first event-loop entry the executor thread pins itself to
`TRTLLM_M31_HS_PIN_CORES` (default 2) physical cores, with their SMT siblings, taken from the CPUs NVML reports
local to its GPU, counted down from the last one after skipping `TRTLLM_M31_HS_PIN_RESERVE_TAIL` (default 8) cores
(frontend / infrastructure convention). GPUs that share a CPU set (two per Grace socket on GB300) take disjoint
slots in physical GPU order, so two ranks on one socket never pick the same cores. A daemon thread then keeps every
other thread of the process tree (loader / saver / probe threads, Mooncake transfer threads, MPI and UCX helpers,
child processes) off the dedicated cores of every GPU on the node, re-checking every 1 s for 3 minutes and every 5 s
after, so threads created later (a thread pool spawned lazily by the executor thread inherits its mask) are moved
too, and re-asserts the executor thread's own mask if something rebinds it.

Interaction with `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` (`llmapi.utils.configure_cpu_affinity`): that binds every
thread to its GPU's NUMA CPUs first, and `PyExecutor.start_worker` re-applies it to all threads just before the
executor thread starts. The pin runs after both, on the executor thread, and only narrows: the dedicated cores are
inside the CPUs the process may use (`sched_getaffinity` at pin time) and every other thread keeps its own mask
minus the dedicated cores. With the NUMA mask on, each GPU's cores therefore come from its own socket.

Differences from the hook (peihengh `examples/disaggregated/m31_host_hooks/m31_hoststall.py`, d9db58795d): in-tree
call instead of an md5-pinned source patch of profiling.py, default off, GPU-to-CPU locality from NVML (as the NUMA
affinity uses) instead of the PCI `numa_node` (the hook's per-rank fallback under a NUMA mask could give two ranks
of one socket the same cores). ITERLOG and EVPOOL are not ported.

Receipt per rank: `M31_EXEC_PIN_ARMED rank=R cpus=.. gpu=.. slot=.. src=nvml|fallback exclude=.. allowed=N`.
"""

import os
import sys
import threading
import time
from typing import Dict, FrozenSet, Optional, Set, Tuple

from tensorrt_llm.logger import logger

__all__ = ["EXECUTOR_PIN_ENV", "maybe_pin_executor_thread", "plan_pin"]

EXECUTOR_PIN_ENV = "TRTLLM_EXECUTOR_THREAD_PIN"
PIN_CORES_ENV = "TRTLLM_M31_HS_PIN_CORES"
PIN_RESERVE_TAIL_ENV = "TRTLLM_M31_HS_PIN_RESERVE_TAIL"

_lock = threading.Lock()
_pinned: Optional[str] = None
_stats = {"remask": 0, "remask_err": 0}


def _emit(line: str) -> None:
    line = f"{line} pid={os.getpid()}"
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a receipt must never take the server down
        pass
    logger.warning(line)


def parse_cpulist(text: str) -> Set[int]:
    out: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def format_cpus(cpus) -> str:
    cpus = sorted(cpus)
    if not cpus:
        return ""
    runs, start, prev = [], cpus[0], cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = cpu
    runs.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(runs)


def plan_pin(
    allowed: Set[int],
    gpu_cpus: Dict[int, FrozenSet[int]],
    siblings: Dict[int, Set[int]],
    ncores: int = 2,
    reserve_tail: int = 8,
) -> Dict[int, Set[int]]:
    """Dedicated logical CPUs per GPU: {gpu: cpus}. Pure function.

    allowed: CPUs this process may use. gpu_cpus: {physical GPU index: CPUs local to it}. siblings: {cpu: its SMT
    siblings, itself included}. GPUs with the same CPU set share it: in GPU order, the j-th takes `ncores` physical
    cores counted down from the set's last allowed core after skipping `reserve_tail` cores. A physical core is used
    only when all its siblings are allowed. A GPU left without `ncores` cores is absent from the plan.
    """
    plan: Dict[int, Set[int]] = {}
    groups: Dict[FrozenSet[int], list] = {}
    for gpu in sorted(gpu_cpus):
        groups.setdefault(frozenset(gpu_cpus[gpu]), []).append(gpu)
    for cpus, gpus in groups.items():
        usable = set(cpus) & allowed
        cores = sorted(
            {min(siblings.get(c, {c})) for c in usable if siblings.get(c, {c}) <= allowed}
        )
        for slot, gpu in enumerate(gpus):
            hi = len(cores) - reserve_tail - slot * ncores
            lo = hi - ncores
            if lo < 0:
                continue
            dedicated: Set[int] = set()
            for core in cores[lo:hi]:
                dedicated |= siblings.get(core, {core})
            plan[gpu] = dedicated
    return plan


def remask_target(current: Set[int], allowed: Set[int], exclude: Set[int]) -> Set[int]:
    """Mask for a non-executor thread: its own mask without the dedicated cores, else allowed minus them."""
    out = current - exclude
    return out if out else (allowed - exclude)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _siblings(cpus) -> Dict[int, Set[int]]:
    sib = {}
    for cpu in cpus:
        try:
            sib[cpu] = parse_cpulist(
                _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            )
        except OSError:
            sib[cpu] = {cpu}
    return sib


def _nvml_gpu_cpus() -> Dict[int, FrozenSet[int]]:
    """CPUs local to every GPU NVML enumerates on this node (CUDA_VISIBLE_DEVICES does not hide any)."""
    import ctypes
    import math

    import psutil
    import pynvml

    cpu_count = psutil.cpu_count()
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    words = math.ceil(cpu_count / bits)
    pynvml.nvmlInit()
    try:
        out = {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            masks = pynvml.nvmlDeviceGetCpuAffinity(pynvml.nvmlDeviceGetHandleByIndex(index), words)
            out[index] = frozenset(
                cpu for cpu in range(cpu_count) if masks[cpu // bits] & (1 << (cpu % bits))
            )
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _physical_gpu(device: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        tokens = [t.strip() for t in visible.split(",") if t.strip()]
        if 0 <= device < len(tokens) and tokens[device].isdigit():
            return int(tokens[device])
    return device


def _local_rank(device: int) -> int:
    for name in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "LOCAL_RANK"):
        value = os.environ.get(name, "")
        if value.isdigit():
            return int(value)
    return device


def _remask_loop(exec_tid: int, dedicated: Set[int], exclude: Set[int], allowed: Set[int]) -> None:
    started = time.monotonic()
    self_tid = threading.get_native_id()
    try:
        os.sched_setaffinity(0, allowed - exclude)
    except OSError:
        _stats["remask_err"] += 1
    root = os.getpid()
    while True:
        pending, seen = [root], set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for name in tids:
                tid = int(name)
                if tid == self_tid:
                    continue
                try:
                    current = os.sched_getaffinity(tid)
                    if tid == exec_tid:
                        if current != dedicated:
                            os.sched_setaffinity(tid, dedicated)
                            _stats["remask"] += 1
                    elif current & exclude:
                        os.sched_setaffinity(tid, remask_target(current, allowed, exclude))
                        _stats["remask"] += 1
                except OSError:
                    _stats["remask_err"] += 1
                try:
                    children = _read(f"/proc/{pid}/task/{tid}/children")
                except OSError:
                    continue
                pending.extend(int(child) for child in children.split())
        time.sleep(1.0 if time.monotonic() - started < 180 else 5.0)


def _resolve_device(device: Optional[int]) -> int:
    """The executor's CUDA device. The event-loop thread pins before its first cudaSetDevice, and the current device
    is per thread, so `torch.cuda.current_device()` there is 0 on every rank: PyExecutor passes its device_id and the
    current device is only the fallback."""
    if device is not None:
        return int(device)
    import torch

    return torch.cuda.current_device()


def _pin(rank, device: Optional[int] = None) -> str:
    allowed = set(os.sched_getaffinity(0))
    device = _resolve_device(device)
    gpu = _physical_gpu(device)
    ncores = max(1, int(os.environ.get(PIN_CORES_ENV, "2")))
    reserve_tail = max(0, int(os.environ.get(PIN_RESERVE_TAIL_ENV, "8")))
    try:
        gpu_cpus, src = _nvml_gpu_cpus(), "nvml"
        if gpu not in gpu_cpus:
            raise LookupError(f"GPU {gpu} not among NVML's {sorted(gpu_cpus)}")
    except Exception as exc:  # noqa: BLE001 - fall back to one shared set, slot by local rank
        local = _local_rank(device)
        gpu = local
        gpu_cpus = {index: frozenset(allowed) for index in range(local + 1)}
        src = f"fallback({type(exc).__name__})"
    plan = plan_pin(allowed, gpu_cpus, _siblings(allowed), ncores, reserve_tail)
    dedicated = plan.get(gpu)
    if not dedicated:
        return f"skip(no_cores gpu={gpu} allowed={len(allowed)} src={src})"
    exclude = set().union(*plan.values())
    slot = sorted(g for g, cpus in gpu_cpus.items() if cpus == gpu_cpus[gpu]).index(gpu)
    os.sched_setaffinity(0, dedicated)
    thread = threading.Thread(
        target=_remask_loop,
        args=(threading.get_native_id(), dedicated, exclude, allowed),
        name="trtllm-exec-pin-remask",
        daemon=True,
    )
    thread.start()
    return (
        f"cpus={format_cpus(dedicated)} gpu={gpu} slot={slot} src={src} exclude={format_cpus(exclude)} "
        f"allowed={len(allowed)} numa_aware={os.environ.get('TLLM_NUMA_AWARE_WORKER_AFFINITY', 'unset')}"
    )


def maybe_pin_executor_thread(rank=-1, device: Optional[int] = None) -> Optional[str]:
    """Pin the calling (executor) thread when `TRTLLM_EXECUTOR_THREAD_PIN=1`; once per process. Never raises."""
    global _pinned
    if os.environ.get(EXECUTOR_PIN_ENV, "0") != "1":
        return None
    with _lock:
        if _pinned is not None:
            return _pinned
        try:
            _pinned = _pin(rank, device)
        except Exception as exc:  # noqa: BLE001 - affinity is a perf knob, never fatal
            _pinned = f"fail({type(exc).__name__}: {exc})"
        _emit(f"M31_EXEC_PIN_ARMED rank={rank} {_pinned}")
        return _pinned
