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
  The model-driven pin honors the affinity opt-out ``TLLM_NUMA_AWARE_WORKER_AFFINITY`` (any value but 1) unless
  ``TRTLLM_EXECUTOR_THREAD_PIN_CORES`` is set. Slots are assigned per physical GPU on the node, so ranks on one NUMA
  node never share cores, and the plan is reused when the executor loop is entered again.
  ``TRTLLM_EXECUTOR_THREAD_PIN_RESERVE_TAIL=<k>`` (default 8) leaves the last k physical cores of each NUMA node to
  other processes (e.g. a frontend).
"""

import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set

from tensorrt_llm._utils import env_switch
from tensorrt_llm.logger import logger

SAMPLER_EVENT_POOL = env_switch("TRTLLM_DISABLE_SAMPLER_EVENT_POOL") is not True
PIN_CORES_ENV = "TRTLLM_EXECUTOR_THREAD_PIN_CORES"
PIN_RESERVE_TAIL_ENV = "TRTLLM_EXECUTOR_THREAD_PIN_RESERVE_TAIL"


def pin_reserve_tail() -> int:
    """Physical cores left free at the tail of each NUMA node (TRTLLM_EXECUTOR_THREAD_PIN_RESERVE_TAIL)."""
    return max(0, int(os.environ.get(PIN_RESERVE_TAIL_ENV, "8")))


# ---------------------------------------------------------------- sampler event pool
class EventPool:
    """Reuse an object only when the pool holds the only reference to it (a refcount check, no driver call).

    Single-thread only: the check and the hand-out are two steps, so callers keep one pool per thread
    (``sampler_event``).
    """

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


# One pool per thread and device: an executor thread drives one device, and a pool is never shared
# between threads, so no two threads can be handed the same event.
_SAMPLER_EVENT_POOLS = threading.local()


def _thread_pool(key: object, factory: Callable[[], object]) -> EventPool:
    """This thread's pool for ``key``."""
    pools = getattr(_SAMPLER_EVENT_POOLS, "pools", None)
    if pools is None:
        pools = _SAMPLER_EVENT_POOLS.pools = {}
    pool = pools.get(key)
    if pool is None:
        pool = pools[key] = EventPool(factory)
    return pool


def sampler_event():
    """An un-recorded CUDA event for a SamplerEvent: pooled (per thread and device) unless the pool is off.

    Before CUDA is initialized there is no current device to key the pool on, so the event is created directly.
    """
    import torch

    if not SAMPLER_EVENT_POOL or not torch.cuda.is_initialized():
        return torch.cuda.Event()
    return _thread_pool(torch.cuda.current_device(), torch.cuda.Event).acquire()


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


# Every this many passes the remask loop rechecks every thread, catching one that moved its own mask
# back onto the dedicated cores; the passes between check only threads it has not handled yet.
_REMASK_FULL_EVERY = 12
# How long a new arm waits for the previous remask thread to stop and restore its masks.
_REMASK_JOIN_S = 10.0


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
    executor thread exits or ``stop`` is set (a reused TID must not be pinned). On exit, the threads
    it remasked that are still alive, and the loop itself, get their earlier masks back."""
    self_tid = threading.get_native_id()
    try:
        own_before = os.sched_getaffinity(0)
    except OSError:
        own_before = None
    try:
        os.sched_setaffinity(0, allowed - exclude)
    except OSError:
        pass
    root = os.getpid()
    # (pid, tid) of threads already handled, and the masks the loop replaced.
    done: Set[tuple] = set()
    before: Dict[int, Set[int]] = {}
    task_counts: Dict[int, int] = {}
    passes = 0
    while exec_thread.is_alive() and not stop.is_set():
        full = passes % _REMASK_FULL_EVERY == 0
        passes += 1
        pids = [root]
        seen = set()
        seen_threads: Set[tuple] = set()
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
                key = (pid, tid)
                seen_threads.add(key)
                if key in done and tid != exec_tid and not full:
                    continue
                try:
                    cur = os.sched_getaffinity(tid)
                    if tid == exec_tid:
                        if cur != dedicated:
                            before.setdefault(tid, cur)
                            os.sched_setaffinity(tid, dedicated)
                    elif cur & exclude:
                        new = remask_target(cur, allowed, exclude)
                        if new:
                            before.setdefault(tid, cur)
                            os.sched_setaffinity(tid, new)
                    done.add(key)
                except OSError:
                    pass
            if full or task_counts.get(pid) != len(tids):
                task_counts[pid] = len(tids)
                pids.extend(_children_pids(pid))
        # A reused TID is checked again.
        done &= seen_threads
        stop.wait(1.0 if time.monotonic() - t0 < 180 else 5.0)
    for tid, mask in before.items():
        try:
            os.sched_setaffinity(tid, mask)
        except OSError:
            pass
    if own_before is not None:
        try:
            os.sched_setaffinity(0, own_before)
        except OSError:
            pass


def pin_cores(model_cores: int = 0) -> int:
    """Dedicated cores for the executor thread: TRTLLM_EXECUTOR_THREAD_PIN_CORES when set, else ``model_cores``."""
    raw = os.environ.get(PIN_CORES_ENV, "").strip()
    return max(0, int(raw) if raw else int(model_cores))


def _node_gpu_numa(own_dev: int) -> tuple:
    """``({physical GPU index: NUMA node}, own physical index)`` over every GPU on the host, or
    ``({}, None)`` without NVML. NVML ignores CUDA_VISIBLE_DEVICES, so the index is unique on the node."""
    try:
        import pynvml
        import torch

        props = torch.cuda.get_device_properties(own_dev)
        own_key = (
            int(getattr(props, "pci_domain_id", 0) or 0),
            int(props.pci_bus_id),
            int(props.pci_device_id),
        )
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - no NVML (or no PCI ids): the visible devices are used
        return {}, None
    try:
        numa, own_phys = {}, None
        for index in range(pynvml.nvmlDeviceGetCount()):
            info = pynvml.nvmlDeviceGetPciInfo(pynvml.nvmlDeviceGetHandleByIndex(index))
            bus_id = info.busId.decode() if isinstance(info.busId, bytes) else str(info.busId)
            domain, bus, rest = bus_id.split(":")
            device = rest.split(".")[0]
            key = (int(domain, 16), int(bus, 16), int(device, 16))
            if key == own_key:
                own_phys = index
            try:
                node = int(
                    _read(
                        f"/sys/bus/pci/devices/{key[0]:04x}:{key[1]:02x}:{key[2]:02x}.0/numa_node"
                    )
                )
            except (OSError, ValueError):
                continue
            if node >= 0:
                numa[index] = node
        return (numa, own_phys) if own_phys in numa else ({}, None)
    except Exception:  # noqa: BLE001 - any NVML failure falls back to the visible devices
        return {}, None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


# The first plan of this process, reused when the executor loop is entered again: the pinned thread's
# own mask is then already reduced, and planning from it would take another set of cores.
_PIN_STATE: Dict[tuple, tuple] = {}


def _plan_executor_pin(ncores: int, reserve_tail: int) -> tuple:
    """``(dedicated, exclude, allowed, description)`` for this rank; dedicated is empty when no core is free."""
    import torch

    own_dev = torch.cuda.current_device()
    cached = _PIN_STATE.get((ncores, reserve_tail, own_dev))
    if cached is not None:
        return cached
    allowed = os.sched_getaffinity(0)
    numa, sib = _topology()
    online = set().union(*numa.values()) if numa else set()
    nodes = sorted(numa)
    if not nodes:
        return set(), set(), allowed, "skipped (no NUMA topology)"
    node_numa, own_key = _node_gpu_numa(own_dev)
    src = "nvml"
    if not node_numa:
        own_key = own_dev
        ndev = torch.cuda.device_count()
        node_numa, src = {}, "pci"
        for r in range(ndev):
            n = _gpu_numa(r)
            if n is None or n not in numa:
                src = "fallback"
                n = nodes[min(len(nodes) - 1, r * len(nodes) // max(1, ndev))]
            node_numa[r] = n
    rank_numa = {r: n for r, n in node_numa.items() if n in numa}
    own_node = rank_numa.get(own_key, nodes[0])
    shared = len(allowed) >= 0.75 * len(online) if online else False
    if shared:
        plan = plan_pin(allowed, numa, sib, rank_numa, ncores, reserve_tail)
    else:
        # Per-rank CPU masks: plan inside this rank's own allowed set, over every GPU on its node,
        # so each local GPU gets its own slot.
        same_node = {r: n for r, n in rank_numa.items() if n == own_node}
        same_node.setdefault(own_key, own_node)
        plan = plan_pin(
            allowed,
            {own_node: numa.get(own_node, allowed) & allowed},
            sib,
            same_node,
            ncores,
            reserve_tail,
        )
    dedicated = plan.get(own_key) or set()
    exclude = set().union(*plan.values()) if plan else set()
    description = (
        f"dev={own_dev} gpu={own_key} numa={own_node} numa_src={src} shared={int(shared)} "
        f"exclude={format_cpulist(exclude)}"
        if dedicated
        else f"skipped (no free cores, shared={int(shared)}, allowed={len(allowed)})"
    )
    state = (dedicated, exclude, allowed, description)
    if dedicated:
        _PIN_STATE[(ncores, reserve_tail, own_dev)] = state
    return state


def pin_executor_thread(
    ncores: int,
    reserve_tail: Optional[int] = None,
    stop: Optional[threading.Event] = None,
    started: Optional[List[threading.Thread]] = None,
) -> str:
    """Pin the calling (executor) thread to its dedicated cores; returns a description for the log.

    ``reserve_tail`` defaults to ``pin_reserve_tail()``. Slots are assigned per physical GPU on the node
    (NVML), so ranks on one NUMA node never share cores. The remask thread runs until the calling
    thread exits or ``stop`` is set; it is appended to ``started`` when given."""
    if reserve_tail is None:
        reserve_tail = pin_reserve_tail()
    dedicated, exclude, allowed, description = _plan_executor_pin(ncores, reserve_tail)
    if not dedicated:
        return description
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
    if started is not None:
        started.append(t)
    return f"cpus={format_cpulist(dedicated)} {description}"


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
    from tensorrt_llm.llmapi.utils import worker_affinity_managed

    if not os.environ.get(PIN_CORES_ENV, "").strip() and not worker_affinity_managed():
        logger.info(
            f"Executor thread pinning skipped on rank {rank}: TLLM_NUMA_AWARE_WORKER_AFFINITY="
            f"{os.environ.get('TLLM_NUMA_AWARE_WORKER_AFFINITY')} (affinity opt-out); set {PIN_CORES_ENV} to force"
        )
        return
    # A remask thread left by an earlier loop entry stops, and has restored the masks it changed,
    # before this one starts, so the old loop cannot undo the new one's exclusions.
    previous = getattr(executor, "_pin_remask_stop", None)
    if isinstance(previous, threading.Event):
        previous.set()
    previous_thread = getattr(executor, "_pin_remask_thread", None)
    if isinstance(previous_thread, threading.Thread) and previous_thread.is_alive():
        previous_thread.join(timeout=_REMASK_JOIN_S)
        if previous_thread.is_alive():
            logger.warning(
                f"Executor thread pinning on rank {rank}: the previous remask thread did not stop "
                f"within {_REMASK_JOIN_S} s"
            )
    # Setting it stops the remask thread early; it also stops when the executor thread exits.
    executor._pin_remask_stop = threading.Event()
    started: List[threading.Thread] = []
    try:
        pin = pin_executor_thread(ncores, stop=executor._pin_remask_stop, started=started)
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(f"Executor thread pinning failed on rank {rank}: {error}")
        return
    executor._pin_remask_thread = started[0] if started else None
    logger.info(f"Executor thread pinning on rank {rank}: {pin}")
