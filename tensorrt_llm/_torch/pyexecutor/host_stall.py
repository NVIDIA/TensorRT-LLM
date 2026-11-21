# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor host-stall options: exact (no numerics, no scheduling decision changes).

* Sampler event pool (default on; ``TRTLLM_DISABLE_SAMPLER_EVENT_POOL=1`` turns it off): the sampler's per-step
  ``torch.cuda.Event`` (created and destroyed every step) comes from a pool. An event is reused only when the pool
  holds the only reference to it, so no ``SamplerEvent`` and no caller can still synchronize or wait on it and a
  re-record is never observed by an old owner.
* Executor thread pinning (opt-in): ``TRTLLM_EXECUTOR_THREAD_PIN_CORES=<n>`` pins the executor thread to n
  dedicated physical cores (with their SMT siblings) on its GPU's NUMA node, and a background thread keeps every
  other thread of the process tree off all local ranks' dedicated cores. Unset, the model's
  ``RuntimeHooks.executor_thread_pin_cores`` decides (0, off, by default); ``0`` turns pinning off either way.
  ``TRTLLM_EXECUTOR_THREAD_PIN_RESERVE_TAIL=<k>`` (default 8) leaves the last k physical cores of each NUMA node to
  other processes (e.g. a frontend).
"""

import os
import sys
import threading
import time
from typing import Callable, Dict, Optional, Set

from tensorrt_llm.logger import logger

SAMPLER_EVENT_POOL = os.environ.get("TRTLLM_DISABLE_SAMPLER_EVENT_POOL", "0") != "1"
PIN_CORES_ENV = "TRTLLM_EXECUTOR_THREAD_PIN_CORES"
PIN_RESERVE_TAIL_ENV = "TRTLLM_EXECUTOR_THREAD_PIN_RESERVE_TAIL"


def pin_reserve_tail() -> int:
    """Physical cores left free at the tail of each NUMA node (TRTLLM_EXECUTOR_THREAD_PIN_RESERVE_TAIL)."""
    return max(0, int(os.environ.get(PIN_RESERVE_TAIL_ENV, "8")))


# ---------------------------------------------------------------- sampler event pool
class EventPool:
    """Reuse an object only when the pool holds the only reference to it (a refcount check, no driver call)."""

    def __init__(self, factory: Callable[[], object], cap: int = 64) -> None:
        self._factory = factory
        self._cap = cap
        self._pool = []
        self._cursor = 0
        probe = [object()]
        # The same expression shape as in acquire(): the list slot plus the getrefcount argument.
        self._free_refs = sys.getrefcount(probe[0])
        self.hits = 0
        self.created = 0
        self.overflow = 0

    def acquire(self):
        pool = self._pool
        n = len(pool)
        for k in range(n):
            i = (self._cursor + k) % n
            if sys.getrefcount(pool[i]) == self._free_refs:
                self._cursor = (i + 1) % n
                self.hits += 1
                return pool[i]
        obj = self._factory()
        if n < self._cap:
            pool.append(obj)
            self.created += 1
        else:
            self.overflow += 1
        return obj

    def __len__(self) -> int:
        return len(self._pool)


_SAMPLER_EVENT_POOL: Optional[EventPool] = None


def sampler_event():
    """An un-recorded CUDA event for a SamplerEvent: pooled unless the pool is off."""
    global _SAMPLER_EVENT_POOL
    import torch

    if not SAMPLER_EVENT_POOL:
        return torch.cuda.Event()
    if _SAMPLER_EVENT_POOL is None:
        _SAMPLER_EVENT_POOL = EventPool(torch.cuda.Event)
    return _SAMPLER_EVENT_POOL.acquire()


# ---------------------------------------------------------------- executor thread pinning
def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def parse_cpulist(text: str) -> Set[int]:
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def format_cpulist(cpus) -> str:
    cpus = sorted(cpus)
    if not cpus:
        return ""
    runs, s, p = [], cpus[0], cpus[0]
    for c in cpus[1:]:
        if c == p + 1:
            p = c
            continue
        runs.append(f"{s}-{p}" if p > s else f"{s}")
        s = p = c
    runs.append(f"{s}-{p}" if p > s else f"{s}")
    return ",".join(runs)


def plan_pin(
    allowed: Set[int],
    numa_cpus: Dict[int, Set[int]],
    siblings: Dict[int, Set[int]],
    rank_numa: Dict[int, int],
    ncores: int = 2,
    reserve_tail: int = 8,
) -> Dict[int, Set[int]]:
    """Dedicated logical CPUs per local rank: {rank: set}.

    allowed: CPUs this process may use; numa_cpus: {node: set(cpus)} (CPU-bearing nodes only);
    siblings: {cpu: set(siblings incl. cpu)}; rank_numa: {local_rank: node}.
    A physical core is used only if all its siblings are allowed. Ranks on one node take ncores physical cores each,
    counted down from the node's last allowed core after skipping reserve_tail cores.
    """
    plan = {}
    for node in sorted(set(rank_numa.values())):
        cpus = numa_cpus.get(node, set()) & allowed
        cores = sorted({min(siblings.get(c, {c})) for c in cpus if siblings.get(c, {c}) <= allowed})
        ranks = sorted(r for r, n in rank_numa.items() if n == node)
        for j, r in enumerate(ranks):
            hi = len(cores) - reserve_tail - j * ncores
            lo = hi - ncores
            if lo < 0:
                continue
            dedicated_cpus = set()
            for c in cores[lo:hi]:
                dedicated_cpus |= siblings.get(c, {c})
            plan[r] = dedicated_cpus
    return plan


def remask_target(cur: Set[int], allowed: Set[int], exclude: Set[int]) -> Set[int]:
    """Mask for a non-executor thread: its own mask without the dedicated cores, else allowed minus them."""
    out = cur - exclude
    return out if out else (allowed - exclude)


def _topology():
    numa = {}
    base = "/sys/devices/system/node"
    for name in os.listdir(base):
        if name.startswith("node") and name[4:].isdigit():
            try:
                cpus = parse_cpulist(_read(f"{base}/{name}/cpulist"))
            except OSError:
                continue
            if cpus:
                numa[int(name[4:])] = cpus
    sib = {}
    for node_cpus in numa.values():
        for c in node_cpus:
            try:
                sib[c] = parse_cpulist(
                    _read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
                )
            except OSError:
                sib[c] = {c}
    return numa, sib


def _gpu_numa(dev_index: int) -> Optional[int]:
    try:
        import torch

        p = torch.cuda.get_device_properties(dev_index)
        dom = getattr(p, "pci_domain_id", None)
        bus = getattr(p, "pci_bus_id", None)
        dv = getattr(p, "pci_device_id", None)
        if bus is None or dv is None:
            return None
        n = int(
            _read(
                f"/sys/bus/pci/devices/{int(dom or 0):04x}:{int(bus):02x}:{int(dv):02x}.0/numa_node"
            )
        )
        return n if n >= 0 else None
    except (OSError, ValueError, RuntimeError, ImportError, AssertionError):
        return None


def _children_pids(pid: int):
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                text = _read(f"/proc/{pid}/task/{tid}/children")
            except OSError:
                continue
            out.extend(int(x) for x in text.split())
    except OSError:
        pass
    return out


def _remask_loop(
    exec_tid: int,
    dedicated: Set[int],
    exclude: Set[int],
    allowed: Set[int],
    t0: float,
    exec_thread: threading.Thread,
    stop: threading.Event,
):
    """Keep the executor thread on its dedicated cores and every other thread off them, until the
    executor thread exits or ``stop`` is set (a reused TID must not be pinned)."""
    self_tid = threading.get_native_id()
    try:
        os.sched_setaffinity(0, allowed - exclude)
    except OSError:
        pass
    root = os.getpid()
    while exec_thread.is_alive() and not stop.is_set():
        pids = [root]
        seen = set()
        while pids:
            pid = pids.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for t in tids:
                tid = int(t)
                if tid == self_tid and pid == root:
                    continue
                try:
                    cur = os.sched_getaffinity(tid)
                    if tid == exec_tid:
                        if cur != dedicated:
                            os.sched_setaffinity(tid, dedicated)
                    elif cur & exclude:
                        new = remask_target(cur, allowed, exclude)
                        if new:
                            os.sched_setaffinity(tid, new)
                except OSError:
                    pass
            pids.extend(_children_pids(pid))
        stop.wait(1.0 if time.monotonic() - t0 < 180 else 5.0)


def pin_cores(model_cores: int = 0) -> int:
    """Dedicated cores for the executor thread: TRTLLM_EXECUTOR_THREAD_PIN_CORES when set, else ``model_cores``."""
    raw = os.environ.get(PIN_CORES_ENV, "").strip()
    return max(0, int(raw) if raw else int(model_cores))


def pin_executor_thread(
    ncores: int, reserve_tail: Optional[int] = None, stop: Optional[threading.Event] = None
) -> str:
    """Pin the calling (executor) thread to its dedicated cores; returns a description for the log.

    ``reserve_tail`` defaults to ``pin_reserve_tail()``. The remask thread runs until the calling
    thread exits or ``stop`` is set."""
    import torch

    if reserve_tail is None:
        reserve_tail = pin_reserve_tail()

    allowed = os.sched_getaffinity(0)
    numa, sib = _topology()
    online = set().union(*numa.values()) if numa else set()
    ndev = torch.cuda.device_count()
    own_dev = torch.cuda.current_device()
    nodes = sorted(numa)
    if not nodes:
        return "skipped (no NUMA topology)"
    rank_numa, src = {}, "pci"
    for r in range(ndev):
        n = _gpu_numa(r)
        if n is None or n not in numa:
            src = "fallback"
            n = nodes[min(len(nodes) - 1, r * len(nodes) // max(1, ndev))]
        rank_numa[r] = n
    shared = len(allowed) >= 0.75 * len(online) if online else False
    if shared:
        plan = plan_pin(allowed, numa, sib, rank_numa, ncores, reserve_tail)
    else:
        # Per-rank CPU masks: plan inside this rank's own allowed set only.
        own_node = rank_numa.get(own_dev, nodes[0])
        plan = plan_pin(allowed, {own_node: allowed}, sib, {own_dev: own_node}, ncores, 0)
    dedicated = plan.get(own_dev)
    if not dedicated:
        return f"skipped (no free cores, shared={int(shared)}, allowed={len(allowed)})"
    exclude = set().union(*plan.values())
    os.sched_setaffinity(0, dedicated)
    exec_tid = threading.get_native_id()
    t = threading.Thread(
        target=_remask_loop,
        args=(
            exec_tid,
            dedicated,
            exclude,
            allowed,
            time.monotonic(),
            threading.current_thread(),
            stop if stop is not None else threading.Event(),
        ),
        name="executor-pin-remask",
        daemon=True,
    )
    t.start()
    return (
        f"cpus={format_cpulist(dedicated)} dev={own_dev} numa={rank_numa.get(own_dev)} "
        f"numa_src={src} shared={int(shared)} exclude={format_cpulist(exclude)}"
    )


def arm_executor_thread(executor) -> None:
    """Called once on the executor thread at loop entry: pins it to ``pin_cores`` cores (the model's
    ``RuntimeHooks.executor_thread_pin_cores`` unless TRTLLM_EXECUTOR_THREAD_PIN_CORES is set)."""
    from .runtime_hooks import RuntimeHooks

    model = getattr(getattr(executor, "model_engine", None), "model", None)
    hooks = getattr(model, "runtime_hooks", None)
    rank = getattr(getattr(executor, "dist", None), "rank", -1)
    try:
        ncores = pin_cores(
            hooks.executor_thread_pin_cores if isinstance(hooks, RuntimeHooks) else 0
        )
    except ValueError as error:
        logger.warning(f"Executor thread pinning skipped on rank {rank}: {error}")
        return
    if ncores <= 0:
        return
    # Setting it stops the remask thread early; it also stops when the executor thread exits.
    executor._pin_remask_stop = threading.Event()
    try:
        pin = pin_executor_thread(ncores, stop=executor._pin_remask_stop)
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(f"Executor thread pinning failed on rank {rank}: {error}")
        return
    logger.info(f"Executor thread pinning on rank {rank}: {pin}")
