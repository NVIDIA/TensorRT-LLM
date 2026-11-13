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
"""Lend plain host memory to a Mooncake store pool from inside the worker.

A server that holds no KV connector (typically a disaggregated generation
server) can still contribute host DRAM to the pool its context servers use.
Configured by ``store_host_donor`` in the LLM args, so it works the same under
``trtllm-serve`` and under any frontend that builds a ``PyExecutor``::

    store_host_donor:
      startup_size_per_node: 512GiB    # mounted while the model loads
      late_size_per_node: 64GiB        # mounted after warmup
      # master: host:port              # default: $MOONCAKE_MASTER

Each node's sizes are split evenly over its CPU NUMA nodes ("sockets"). One
rank per socket owns that socket's share: it opens a Mooncake client whose
segment memory is bound to the socket (``MPOL_BIND`` on the registering
thread, which Mooncake honours when it places the segment) and keeps it until
the executor shuts down. No CUDA memory is involved and no KV cache is
registered, so a connector-free server stays connector-free, and a server
that also runs the ``mooncake-store`` connector keeps its connector client
separate.

Timing:

* startup portion: registered on a background thread from the moment the
  executor's process group exists, overlapping model load. It is joined
  before the late portion, so a failure fails executor creation.
* late portion: registered after the final executor finished warmup (all CUDA
  graphs captured) on every rank. With ``late_delay_seconds == 0`` it is
  registered before the worker loop starts; otherwise a background thread
  registers it that many seconds after the worker started.

``late_size_per_node: auto`` sizes the late portion per GPU from a planned
host memory budget instead (``late_auto``)::

    late_size_per_node: auto
    late_auto:
      host_budget_per_gpu: 256GiB       # store + process one GPU may hold
      store_per_gpu: 192GiB             # the KV connector's segment
      process_peak_per_gpu: 24GiB       # nodes without the frontend
      head_process_peak_per_gpu: 48GiB  # the head (frontend) node

Each GPU then lends ``max(0, budget - store - peak - margin)``, with the head
node's peak where the pool master runs (or on ``head_hosts``), cut to what the
job's memory limit leaves and rounded down per NUMA node. A node whose role
cannot be determined is sized as a head node. The computed size is logged as
one ``STORE_HOST_DONOR_LATE_AUTO`` line per rank.

Every registration logs one ``STORE_HOST_DONOR_REGISTERED`` line and every
release one ``STORE_HOST_DONOR_RETIRED`` line.
``TLLM_DISABLE_STORE_HOST_DONOR=1`` turns the feature off.

``TLLM_MOONCAKE_STORE_EGM=1`` runs the whole store over NVLink with EGM
(host memory the GPUs reach through NVLink-C2C and the NVLink fabric) instead
of RDMA; see `store_egm_enabled`. Every share, every late lend and the KV
connector's own segment then becomes a separate Mooncake client that allocates
its segment as EGM on one NUMA node (``enable_egm_store_pool``), the KV
connector copies pages between EGM and its KV pool over NVLink, and the sizing
above is unchanged. Unset (the default), nothing here changes.
"""

import atexit
import contextlib
import ctypes
import json
import os
import platform
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import StoreHostDonorConfig

__all__ = [
    "DISABLE_ENV",
    "EGM_PROTOCOL",
    "MASTER_ENV",
    "STORE_EGM_ENV",
    "DonorShare",
    "LateAuto",
    "StoreHostDonor",
    "active_donor",
    "adopt_store_segment",
    "cgroup_memory",
    "close_active_donor",
    "cpu_numa_nodes",
    "head_node_role",
    "late_auto_per_gpu",
    "node_numa_disagreement",
    "plan_shares",
    "resolve_master",
    "start_store_host_donor",
    "store_egm_enabled",
    "egm_direct_enabled",
    "egm_direct_save_enabled",
    "egm_direct_batch",
    "latch_egm_environment",
    "egm_wheel_check",
]

#: Set to 1 to ignore ``store_host_donor``.
DISABLE_ENV = "TLLM_DISABLE_STORE_HOST_DONOR"
#: Where a launcher that runs a Mooncake master publishes its ``host:port`` to
#: the workers.
MASTER_ENV = "MOONCAKE_MASTER"
#: Mooncake's own client-config variable; its ``master_server_address`` is the
#: last fallback.
CONFIG_PATH_ENV = "MOONCAKE_CONFIG_PATH"
METADATA_SERVER = "P2PHANDSHAKE"
LOG_TAG = "STORE_HOST_DONOR"

_GIB = 1 << 30
#: Kept free on a socket beyond the share, so a bound allocation is refused
#: here instead of invoking the OOM killer inside the bound node.
NODE_HEADROOM_BYTES = 1 << 30
# Node-local lock that serializes the room check and the allocation of every
# owner on a node, so two sockets' owners never both count the same room.
NODE_LOCK_PATH = "/dev/shm/trtllm-store-host-donor.lock"  # nosec B108
_DEFAULT_LOCK = object()
#: Seconds `close` waits for a registration still in flight.
JOIN_TIMEOUT_S = 900.0

_MPOL_BIND = 2
_MPOL_MODE_MASK = 0xFF
#: (set_mempolicy, get_mempolicy) syscall numbers.
_SYSCALLS = {"aarch64": (237, 236), "x86_64": (238, 239)}
_MASK_WORDS = 16  # 1024 NUMA node ids


# --------------------------------------------------------------------------
# Topology and plan
# --------------------------------------------------------------------------


def _parse_cpulist(text: str) -> List[int]:
    nodes: List[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            nodes.extend(range(int(lo), int(hi) + 1))
        else:
            nodes.append(int(part))
    return nodes


def _allowed_mems(status_path: str = "/proc/self/status") -> Optional[List[int]]:
    try:
        with open(status_path) as handle:
            for line in handle:
                if line.startswith("Mems_allowed_list:"):
                    return _parse_cpulist(line.split(":", 1)[1])
    except OSError:
        return None
    return None


def cpu_numa_nodes(
    sysfs_root: str = "/sys/devices/system/node", status_path: str = "/proc/self/status"
) -> List[int]:
    """NUMA nodes that have CPUs and this process may allocate on.

    Memory-only nodes (GPU memory exposed as NUMA nodes on Grace systems,
    CXL expanders) have no CPUs and are not sockets, so they are skipped.
    """
    nodes = []
    try:
        names = os.listdir(sysfs_root)
    except OSError:
        names = []
    for name in names:
        if not name.startswith("node") or not name[4:].isdigit():
            continue
        try:
            with open(os.path.join(sysfs_root, name, "cpulist")) as handle:
                if _parse_cpulist(handle.read()):
                    nodes.append(int(name[4:]))
        except (OSError, ValueError):
            continue
    allowed = _allowed_mems(status_path)
    if allowed is not None:
        nodes = [node for node in nodes if node in allowed]
    return sorted(nodes)


@dataclass(frozen=True)
class DonorShare:
    """What one rank lends from one socket."""

    numa_node: int
    startup_bytes: int
    late_bytes: int


def node_numa_disagreement(peers: Sequence[Any], host: str) -> Optional[str]:
    """Why ranks on ``host`` cannot share one ownership plan, or None.

    ``peers`` holds every rank's ``(hostname, detected CPU NUMA nodes)``. Ranks
    confined to different NUMA nodes would each plan from their own list and
    disagree on which rank owns which socket.
    """
    seen = sorted({tuple(nodes) for name, nodes in peers if name == host and nodes is not None})
    if len(seen) > 1:
        return (
            f"store_host_donor: ranks on {host} see different CPU NUMA nodes "
            f"{[list(s) for s in seen]}; set store_host_donor.numa_nodes"
        )
    return None


def plan_shares(
    startup_per_node: int,
    late_per_node: int,
    numa_nodes: Sequence[int],
    local_rank: int,
    local_size: int,
) -> List[DonorShare]:
    """The shares this rank owns.

    Each node-level size is split evenly over ``numa_nodes``; socket ``k`` (in
    list order) is owned by local rank ``k * local_size // len(numa_nodes)``,
    the first rank of the contiguous block of ranks that sits on it when ranks
    fill sockets in order. With fewer ranks than sockets one rank owns several.
    """
    if not numa_nodes:
        raise ValueError("store_host_donor: no CPU NUMA node to place memory on")
    if len(set(numa_nodes)) != len(numa_nodes):
        raise ValueError(f"store_host_donor: duplicate NUMA nodes in {list(numa_nodes)}")
    if local_size < 1 or not 0 <= local_rank < local_size:
        raise ValueError(f"store_host_donor: bad local rank {local_rank} of {local_size}")
    count = len(numa_nodes)
    for name, size in (
        ("startup_size_per_node", startup_per_node),
        ("late_size_per_node", late_per_node),
    ):
        if size < 0:
            raise ValueError(f"store_host_donor.{name} must be nonnegative")
        if size % count:
            raise ValueError(
                f"store_host_donor.{name}={size} bytes does not split evenly over "
                f"{count} NUMA nodes {list(numa_nodes)}"
            )
    shares = []
    for index, node in enumerate(numa_nodes):
        if index * local_size // count == local_rank:
            shares.append(
                DonorShare(
                    numa_node=node,
                    startup_bytes=startup_per_node // count,
                    late_bytes=late_per_node // count,
                )
            )
    return shares


@dataclass(frozen=True)
class LateAuto:
    """Byte values of ``store_host_donor.late_auto``."""

    host_budget_per_gpu: int
    store_per_gpu: int
    process_peak_per_gpu: int
    head_process_peak_per_gpu: int
    margin_per_gpu: int
    granularity: int
    head_hosts: Optional[Sequence[str]] = None
    job_memory_limit: Optional[int] = None


def late_auto_per_gpu(auto: LateAuto, head: bool) -> int:
    """Late bytes one GPU may lend: budget minus store, peak process memory
    and margin, never negative."""
    peak = auto.head_process_peak_per_gpu if head else auto.process_peak_per_gpu
    return max(0, auto.host_budget_per_gpu - auto.store_per_gpu - peak - auto.margin_per_gpu)


def head_node_role(
    master: str,
    head_hosts: Optional[Sequence[str]] = None,
    *,
    hostname: Optional[str] = None,
    route_to: Optional[Callable[[str], str]] = None,
    resolve: Optional[Callable[[str], List[str]]] = None,
) -> tuple:
    """Whether this node is a head node, and why.

    With ``head_hosts`` the host name decides (short or full name). Otherwise
    the node that runs the pool master is the head: the master's address is
    one of this node's own, i.e. the route to it leaves from it or it is a
    loopback address. When that cannot be determined the node counts as a head
    node, the smaller donation.
    """
    hostname = socket.gethostname() if hostname is None else hostname
    if head_hosts:
        names = {hostname, hostname.split(".", 1)[0]}
        wanted = set(head_hosts) | {h.split(".", 1)[0] for h in head_hosts}
        return bool(names & wanted), "head_hosts"
    if route_to is None:
        from .connectors.mooncake_store.config import local_address as route_to
    if resolve is None:

        def resolve(host: str) -> List[str]:
            return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})

    host = master.rpartition(":")[0].strip("[]")
    try:
        addresses = resolve(host)
        local = route_to(master)
    except (OSError, ValueError) as error:
        return True, f"unknown ({type(error).__name__}: {error}); sized as head"
    if not addresses:
        return True, "unknown (master did not resolve); sized as head"
    if any(a.startswith("127.") or a == "::1" for a in addresses):
        return True, "master is loopback"
    if local in addresses:
        return True, f"master {host} is local ({local})"
    return False, f"master {host} is remote (local {local})"


def _cgroup_nonfile(directory: str) -> int:
    """Usage of one cgroup that page-cache reclaim cannot free: anonymous,
    kernel and shared memory (``memory.stat`` counts shmem inside ``file``)
    and unevictable (for example mlocked) pages."""
    with open(os.path.join(directory, "memory.current")) as handle:
        current = int(handle.read().strip())
    with open(os.path.join(directory, "memory.stat")) as handle:
        stat = dict(line.split() for line in handle if len(line.split()) == 2)
    if "file" not in stat or "shmem" not in stat:
        raise ValueError("memory.stat lacks file/shmem")
    reclaimable = int(stat["file"]) - int(stat["shmem"]) - int(stat.get("unevictable", 0))
    return current - max(0, reclaimable)


def cgroup_memory(
    proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup"
) -> Dict[str, Optional[int]]:
    """The binding cgroup v2 memory limit above this process.

    Every ancestor with a finite ``memory.max`` is checked, and the one with
    the least room (limit minus usage that reclaim cannot free) binds: a
    parent with a larger limit can still have less room left than its child.
    Returns that cgroup's ``limit`` and ``nonfile`` usage, so that
    ``limit - nonfile`` is the room. ``{"limit": None, ...}`` when no cgroup
    v2 limit is set; ``nonfile`` is None when any limited cgroup's usage is
    unreadable. ``usage`` is the same figure for this process's own cgroup
    when no cgroup is limited (None otherwise or when unreadable), so that a
    configured job limit can still be checked against it.
    """
    out: Dict[str, Optional[int]] = {"limit": None, "nonfile": None, "path": None, "usage": None}
    try:
        with open(proc_cgroup) as handle:
            path = next(
                (line.strip().split("::", 1)[1] for line in handle if line.startswith("0::")), None
            )
    except OSError:
        return out
    if path is None:
        return out
    limited = []
    parts = [p for p in path.split("/") if p]
    for depth in range(len(parts), -1, -1):
        directory = os.path.join(root, *parts[:depth])
        try:
            with open(os.path.join(directory, "memory.max")) as handle:
                text = handle.read().strip()
        except OSError:
            continue
        if text.isdigit():
            limited.append((int(text), directory))
    if not limited:
        try:
            out["usage"] = _cgroup_nonfile(os.path.join(root, *parts))
        except (OSError, ValueError):
            pass
        return out
    try:
        levels = [(limit, directory, _cgroup_nonfile(directory)) for limit, directory in limited]
    except (OSError, ValueError):
        # Unknown usage at any limited level: report the tightest limit only.
        out["limit"], out["path"] = min(limited)
        return out
    limit, directory, nonfile = min(levels, key=lambda level: level[0] - level[2])
    out.update(limit=limit, path=directory, nonfile=nonfile)
    return out


def _segment_name(store: Any, hostname: str) -> str:
    """The name the store gave this client's segment, else ``hostname``.

    Never raises: it runs between a successful setup and the registration that
    makes the client releasable.
    """
    get_hostname = getattr(store, "get_hostname", None)
    if callable(get_hostname):
        try:
            return str(get_hostname())
        except Exception:  # noqa: BLE001 - the name is informational
            pass
    return hostname


def _default_job_memory() -> Dict[str, Optional[int]]:
    return cgroup_memory()


def resolve_master(configured: Optional[str], environ: Optional[Mapping[str, str]] = None) -> str:
    """The master's ``host:port``: config, then ``$MOONCAKE_MASTER``, then the
    ``master_server_address`` of ``$MOONCAKE_CONFIG_PATH``."""
    environ = os.environ if environ is None else environ
    candidates = [("store_host_donor.master", configured), (MASTER_ENV, environ.get(MASTER_ENV))]
    path = environ.get(CONFIG_PATH_ENV)
    if path:
        try:
            with open(path) as handle:
                candidates.append((path, json.load(handle).get("master_server_address")))
        except (OSError, ValueError) as error:
            logger.warning(f"{LOG_TAG}: cannot read {CONFIG_PATH_ENV}={path}: {error}")
    for source, value in candidates:
        if value:
            host, separator, port = str(value).rpartition(":")
            if not separator or not host or not port.isdigit():
                raise ValueError(f"{LOG_TAG}: master {value!r} from {source} is not host:port")
            return str(value)
    raise ValueError(
        "store_host_donor needs the Mooncake master: set store_host_donor.master, "
        f"or export {MASTER_ENV}=host:port (or {CONFIG_PATH_ENV}) for the workers."
    )


# --------------------------------------------------------------------------
# NUMA memory policy of the registering thread
# --------------------------------------------------------------------------


class _MemPolicy:
    """set_mempolicy/get_mempolicy through libc; per thread."""

    def __init__(self) -> None:
        numbers = _SYSCALLS.get(platform.machine())
        if numbers is None:
            raise OSError(f"{LOG_TAG}: no mempolicy syscall table for {platform.machine()}")
        self._set_nr, self._get_nr = numbers
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.syscall.restype = ctypes.c_long

    def bind(self, node: int) -> None:
        mask = (ctypes.c_ulong * _MASK_WORDS)()
        width = ctypes.sizeof(ctypes.c_ulong) * 8
        mask[node // width] |= 1 << (node % width)
        if (
            self._libc.syscall(
                self._set_nr,
                ctypes.c_int(_MPOL_BIND),
                mask,
                ctypes.c_ulong(_MASK_WORDS * width + 1),
            )
            != 0
        ):
            errno = ctypes.get_errno()
            raise OSError(
                errno, f"{LOG_TAG}: set_mempolicy(MPOL_BIND, node {node}): {os.strerror(errno)}"
            )

    def current(self) -> tuple:
        mode = ctypes.c_int()
        mask = (ctypes.c_ulong * _MASK_WORDS)()
        width = ctypes.sizeof(ctypes.c_ulong) * 8
        if (
            self._libc.syscall(
                self._get_nr,
                ctypes.byref(mode),
                mask,
                ctypes.c_ulong(_MASK_WORDS * width),
                None,
                ctypes.c_ulong(0),
            )
            != 0
        ):
            errno = ctypes.get_errno()
            raise OSError(errno, f"{LOG_TAG}: get_mempolicy: {os.strerror(errno)}")
        nodes = [n for n in range(_MASK_WORDS * width) if mask[n // width] & (1 << (n % width))]
        return mode.value & _MPOL_MODE_MASK, nodes


def _bind_current_thread(node: int, policy: Optional[_MemPolicy] = None) -> None:
    policy = policy or _MemPolicy()
    policy.bind(node)
    mode, nodes = policy.current()
    if mode != _MPOL_BIND or nodes != [node]:
        raise RuntimeError(
            f"{LOG_TAG}: thread policy is mode {mode} nodes {nodes}, expected MPOL_BIND [{node}]"
        )


def node_meminfo(node: int, sysfs_root: str = "/sys/devices/system/node") -> Dict[str, int]:
    """Byte counts from a NUMA node's meminfo (empty when unreadable)."""
    values = {}
    try:
        with open(os.path.join(sysfs_root, f"node{node}", "meminfo")) as handle:
            for line in handle:
                # "Node 0 MemFree:   123456 kB"
                parts = line.split()
                if len(parts) == 5 and parts[4] == "kB":
                    values[parts[2].rstrip(":")] = int(parts[3]) * 1024
    except (OSError, ValueError):
        return {}
    return values


def node_obtainable_bytes(info: Dict[str, int]) -> Optional[int]:
    """Free plus reclaimable memory of a node, or None when unknown.

    ``FilePages`` counts tmpfs / shared memory and mlocked pages, which reclaim
    cannot free, so ``Shmem`` and ``Unevictable`` are taken out of the page cache.
    """
    if "MemFree" not in info:
        return None
    cache = max(0, info.get("FilePages", 0) - info.get("Shmem", 0) - info.get("Unevictable", 0))
    return info["MemFree"] + cache + info.get("SReclaimable", 0)


#: Smallest physically contiguous block a HOST_NUMA (EGM) allocation is built
#: from: cuMemGetAllocationGranularity reports 2 MiB for HOST_NUMA memory.
EGM_HOST_BLOCK = 2 << 20


def node_egm_usable_bytes(
    node: int,
    buddyinfo_path: str = "/proc/buddyinfo",
    page_size: Optional[int] = None,
    block: int = EGM_HOST_BLOCK,
) -> Optional[int]:
    """Free memory of a NUMA node in blocks an EGM allocation can use, or None.

    A HOST_NUMA allocation is built from free blocks of at least ``block``
    bytes, so free memory split into smaller pieces (counted in MemFree) may
    not back it. This sums the node's free blocks of that order and above
    from the buddy allocator. It is a hint, not a bound: the kernel can
    sometimes assemble more blocks while the allocation runs.
    """
    if page_size is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError):
            return None
    if page_size <= 0:
        return None
    order = max(0, (max(1, block // page_size) - 1).bit_length())
    total, found = 0, False
    try:
        with open(buddyinfo_path) as handle:
            for line in handle:
                # "Node 0, zone   Normal   1180  16528 ..."
                parts = line.split()
                if len(parts) < 5 or parts[0] != "Node" or parts[1].rstrip(",") != str(node):
                    continue
                found = True
                for index, count in enumerate(parts[4:]):
                    if index >= order:
                        total += int(count) * (page_size << index)
    except (OSError, ValueError):
        return None
    return total if found else None


def _env_number(
    environ: Mapping[str, str], name: str, *, positive: bool = False
) -> Optional[float]:
    text = environ.get(name)
    if text is None or not text.strip():
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{LOG_TAG}: {name}={text!r} is not a number") from None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        raise ValueError(f"{LOG_TAG}: {name}={text!r} must be a finite nonnegative number")
    if positive and value == 0:
        raise ValueError(f"{LOG_TAG}: {name} must be > 0")
    return value


def _env_flag(environ: Mapping[str, str], name: str) -> Optional[bool]:
    text = environ.get(name)
    if text is None or not text.strip():
        return None
    value = text.strip().lower()
    if value in ("1", "on", "true", "yes"):
        return True
    if value in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"{LOG_TAG}: {name}={text!r} must be 1/on or 0/off")


# --------------------------------------------------------------------------
# EGM over NVLink (TLLM_MOONCAKE_STORE_EGM)
# --------------------------------------------------------------------------

#: 1 runs the store over NVLink with EGM segments; unset or 0 keeps RDMA.
STORE_EGM_ENV = "TLLM_MOONCAKE_STORE_EGM"
#: Most EGM clients one process keeps open (each is a transfer engine and a port).
EGM_MAX_CLIENTS_ENV = "TLLM_MOONCAKE_STORE_EGM_MAX_CLIENTS"
EGM_MAX_CLIENTS_DEFAULT = 16
EGM_PROTOCOL = "nvlink"
EGM_TAG = "STORE_EGM"
#: 1 passes EGM pages through pinned host staging again. Unset or 0 copies
#: them between EGM and the KV pool directly over NVLink.
EGM_DIRECT_DISABLE_ENV = "TLLM_DISABLE_MOONCAKE_STORE_EGM_DIRECT"
#: Pages per store call on the direct path's asynchronous loaders. Unset keeps
#: the configured transfer batch.
EGM_DIRECT_BATCH_ENV = "TLLM_MOONCAKE_STORE_EGM_DIRECT_BATCH"
#: 0 keeps direct loads but writes pages to EGM through a pinned host slot,
#: which is faster when the slot sits on the GPU's own NUMA node. Unset or 1
#: saves directly as well.
EGM_DIRECT_SAVE_ENV = "TLLM_MOONCAKE_STORE_EGM_DIRECT_SAVE"
#: Copy streams one asynchronous direct load is spread over. Mooncake issues a
#: caller's NVLink copies on that thread's own stream, and one stream does not
#: reach the link's bandwidth; splitting a load across threads reads several
#: page ranges at once. 1 turns it off.
EGM_LOAD_STREAMS_ENV = "TLLM_MOONCAKE_STORE_EGM_LOAD_STREAMS"
EGM_LOAD_STREAMS_DEFAULT = 2
#: 1 (default) has Mooncake group each direct EGM load call by source segment,
#: so the NVLink transport submits one copy batch per segment instead of one
#: per page. Every NVLink submit first waits for the caller's per-thread
#: default stream, which is ordered behind the executor's legacy default
#: stream and so behind the forward pass in flight; fewer submits wait less
#: often. 0 submits page by page.
EGM_GROUPED_LOADS_ENV = "TLLM_MOONCAKE_STORE_EGM_GROUPED_LOADS"
#: 1 (default) writes EGM saves with Mooncake's prefer_alloc_in_same_node, the
#: only put option that groups a call's copies by target segment (one NVLink
#: submit per segment instead of one per page, as for grouped loads). It also
#: places new pages on this node's EGM segments first and spills to the other
#: nodes once those are full. 0 keeps per-page submits and random placement.
EGM_GROUPED_SAVES_ENV = "TLLM_MOONCAKE_STORE_EGM_GROUPED_SAVES"
#: 1 (default) has the executor loop run on a private stream rather than the
#: legacy default stream while direct EGM copies are on. The loop's stream
#: waits for every forward pass, and Mooncake's NVLink transport host-waits on
#: its caller's per-thread default stream before each copy submit; that stream
#: is ordered behind the legacy default stream, so a loop on the legacy stream
#: makes every EGM load and save wait for the forward pass in flight. 0 keeps
#: the loop on the legacy default stream.
EGM_PRIVATE_LOOP_STREAM_ENV = "TLLM_MOONCAKE_STORE_EGM_PRIVATE_LOOP_STREAM"
#: Every EGM segment is a whole number of these, so the size asked for is the
#: size mounted under any HOST_NUMA allocation granularity up to it.
EGM_ALIGN = _GIB
#: A registration whose job-cgroup growth falls below this share of its size
#: is not charged as host memory, and accounting that reads the job cgroup
#: would miss it. The job cgroup is shared with every other rank on the node,
#: whose own usage can fall while a segment registers, so only a clear
#: shortfall counts: memory that is not charged at all grows the cgroup by
#: about nothing.
EGM_CHARGED_FRACTION = 0.5
#: Failed setups one late EGM lend accepts per NUMA node before it moves on.
EGM_LEND_FAILS_PER_NODE = 5
#: Free large-block memory a retried late EGM lend leaves on its NUMA node.
EGM_USABLE_MARGIN = 2 * _GIB
#: A late EGM lend whose cgroup growth exceeds its lent bytes by more than
#: this is reported: the driver kept memory it did not lend.
EGM_KEPT_REPORT_BYTES = 2 * _GIB
#: After a failed EGM setup, the job cgroup and the NUMA node are read every
#: this many seconds until both hold still (the driver may still be
#: returning pages), for at most `EGM_SETTLE_SECONDS`, before a retry.
EGM_SETTLE_INTERVAL = 0.5
EGM_SETTLE_SECONDS = 10.0
EGM_SETTLE_STILL_BYTES = 256 << 20
#: A late EGM share that stays short is retried in the background every this
#: many seconds, at most `EGM_LATE_RETRY_PASSES` times.
EGM_LATE_RETRY_SECONDS_ENV = "TLLM_MOONCAKE_STORE_EGM_LATE_RETRY_SECONDS"
EGM_LATE_RETRY_SECONDS_DEFAULT = 300.0
EGM_LATE_RETRY_PASSES = 12
#: 0 keeps every late EGM lend on its own NUMA node. Unset or 1 places the part
#: that its own node cannot back on the node's other NUMA nodes.
EGM_LEND_SPILL_ENV = "TLLM_MOONCAKE_STORE_EGM_LEND_SPILL"
#: Process environment the classic NVLink transport needs. Mooncake reads it
#: when a client is set up, so it is fixed before the first one.
_EGM_ENV_SET = {
    "MC_MS_AUTO_DISC": "0",
    "MC_USE_TENT": "0",
    "MC_USE_TEV1": "0",
    "MC_ENABLE_PARALLEL_REG_MR": "0",
    "MOONCAKE_STORE_BACKEND": "cpp",
}
_EGM_ENV_POP = (
    "MC_USE_NVLINK_IPC",
    "MC_FORCE_MNNVL",
    "MC_INTRANODE_NVLINK",
    "MC_TE_FILTERS",
    "MC_ENABLE_DEST_DEVICE_AFFINITY",
)
#: Mooncake clients opened in this process under the knob.
_EGM_STATE = {"clients": 0}
_EGM_STATE_LOCK = threading.Lock()


class EgmSetupError(RuntimeError):
    """Mooncake could not set up an EGM client (its ``setup`` returned nonzero)."""


def store_egm_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the store runs over NVLink with EGM segments.

    The only reader of `STORE_EGM_ENV`; the donor and the KV connector both
    ask here, so they cannot disagree on the transport. A value that is
    neither on nor off raises instead of meaning either.
    """
    environ = os.environ if environ is None else environ
    return bool(_env_flag(environ, STORE_EGM_ENV))


def egm_direct_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether EGM pages move between the store and the KV pool directly.

    NVLink copies take a local buffer as a raw device pointer, so the KV pool
    needs neither registration nor a pinned host bounce. On whenever the store
    runs over EGM, unless `EGM_DIRECT_DISABLE_ENV` is set. Values that are
    neither on nor off raise.
    """
    environ = os.environ if environ is None else environ
    if not store_egm_enabled(environ):
        return False
    return not _env_flag(environ, EGM_DIRECT_DISABLE_ENV)


def egm_direct_save_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether saves also skip the host when EGM pages move directly.

    False whenever `egm_direct_enabled` is, and when `EGM_DIRECT_SAVE_ENV` is
    off: loads then stay direct while saves are gathered into a pinned host
    slot first. Values that are neither on nor off raise.
    """
    environ = os.environ if environ is None else environ
    if not egm_direct_enabled(environ):
        return False
    return _env_flag(environ, EGM_DIRECT_SAVE_ENV) is not False


def egm_direct_batch(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Pages per asynchronous direct load call, or None to keep the configured batch."""
    environ = os.environ if environ is None else environ
    value = _env_number(environ, EGM_DIRECT_BATCH_ENV, positive=True)
    if value is None:
        return None
    if value != int(value):
        raise ValueError(f"{EGM_TAG}: {EGM_DIRECT_BATCH_ENV} must be a whole number")
    return int(value)


def egm_load_streams(environ: Optional[Mapping[str, str]] = None) -> int:
    """Copy streams per asynchronous direct load; 1 when the direct path is off."""
    environ = os.environ if environ is None else environ
    if not egm_direct_enabled(environ):
        return 1
    value = _env_number(environ, EGM_LOAD_STREAMS_ENV, positive=True)
    if value is None:
        return EGM_LOAD_STREAMS_DEFAULT
    if value != int(value):
        raise ValueError(f"{EGM_TAG}: {EGM_LOAD_STREAMS_ENV} must be a whole number")
    return int(value)


def egm_grouped_loads(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether direct EGM loads ask Mooncake to group each call by source segment."""
    environ = os.environ if environ is None else environ
    if not egm_direct_enabled(environ):
        return False
    return _env_flag(environ, EGM_GROUPED_LOADS_ENV) is not False


def egm_grouped_saves(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether direct-path EGM saves are grouped by segment (and placed node-local first)."""
    environ = os.environ if environ is None else environ
    if not egm_direct_enabled(environ):
        return False
    return _env_flag(environ, EGM_GROUPED_SAVES_ENV) is not False


def egm_private_loop_stream(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether direct EGM copies ask the executor loop to leave the legacy default stream."""
    environ = os.environ if environ is None else environ
    if not egm_direct_enabled(environ):
        return False
    return _env_flag(environ, EGM_PRIVATE_LOOP_STREAM_ENV) is not False


def egm_max_clients(environ: Optional[Mapping[str, str]] = None) -> int:
    environ = os.environ if environ is None else environ
    value = _env_number(environ, EGM_MAX_CLIENTS_ENV, positive=True)
    if value is None:
        return EGM_MAX_CLIENTS_DEFAULT
    if value != int(value):
        raise ValueError(f"{EGM_TAG}: {EGM_MAX_CLIENTS_ENV} must be a whole number")
    return int(value)


def egm_lend_spill(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether a late EGM lend may place what its NUMA node cannot back on the
    node's other NUMA nodes (on unless `EGM_LEND_SPILL_ENV` is off)."""
    environ = os.environ if environ is None else environ
    return _env_flag(environ, EGM_LEND_SPILL_ENV) is not False


def latch_egm_environment(environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Fix the process environment the NVLink transport needs. Idempotent.

    Returns what it changed. Raises when a change is needed after this
    process already opened a client under the knob, since that client was set
    up with the old values.
    """
    environ = os.environ if environ is None else environ
    changed = {
        name: [environ.get(name), value]
        for name, value in _EGM_ENV_SET.items()
        if environ.get(name) != value
    }
    popped = [name for name in _EGM_ENV_POP if name in environ]
    with _EGM_STATE_LOCK:
        if (changed or popped) and _EGM_STATE["clients"] > 0:
            raise RuntimeError(
                f"{EGM_TAG}: the environment changed ({sorted(changed) + popped}) after a "
                "Mooncake client was set up; set it before the first client"
            )
        for name, value in _EGM_ENV_SET.items():
            environ[name] = value
        for name in popped:
            environ.pop(name, None)
    return {"set": changed, "popped": popped}


def _note_egm_client() -> None:
    with _EGM_STATE_LOCK:
        _EGM_STATE["clients"] += 1


def egm_wheel_check(store_type: Any = None) -> None:
    """Raise unless the Mooncake binding can allocate EGM segments.

    The dictionary ``setup`` ignores keys it does not know, so only its
    documentation tells a wheel with the EGM store pool from one without.
    """
    if store_type is None:
        try:
            from mooncake.store import MooncakeDistributedStore as store_type
        except ImportError as error:
            logger.error(f"{EGM_TAG}_UNSUPPORTED reason=import: {error}")
            raise RuntimeError(
                f"{EGM_TAG}: {STORE_EGM_ENV}=1 needs the Mooncake Python bindings"
            ) from error
    doc = getattr(getattr(store_type, "setup", None), "__doc__", None) or ""
    missing = [key for key in ("enable_egm_store_pool", "egm_numa_nodes") if key not in doc]
    if missing:
        logger.error(f"{EGM_TAG}_UNSUPPORTED reason=wheel missing={missing}")
        raise RuntimeError(
            f"{EGM_TAG}: {STORE_EGM_ENV}=1 but this Mooncake wheel cannot allocate EGM "
            f"segments (setup lacks {missing}); there is no RDMA fallback"
        )


def _egm_kind(phase: str) -> str:
    return phase if phase in ("startup", "startup_adopted") else "late"


def egm_setup_config(*, hostname: str, master: str, size: int, numa_node: int) -> Dict[str, str]:
    """The dictionary ``setup`` of a client that provides one EGM segment.

    ``egm_numa_nodes`` names the socket: Mooncake's ``auto`` would follow the
    visible GPUs instead of the share's NUMA node.
    """
    return {
        "local_hostname": hostname,
        "metadata_server": METADATA_SERVER,
        "master_server_addr": master,
        "protocol": EGM_PROTOCOL,
        "rdma_devices": "",
        "global_segment_size": str(size),
        "local_buffer_size": "0",
        "enable_egm_store_pool": "true",
        "egm_numa_nodes": str(numa_node),
    }


# --------------------------------------------------------------------------
# The donor
# --------------------------------------------------------------------------


@dataclass
class _Registration:
    phase: str
    share: DonorShare
    bytes: int
    store: Any = None
    segment_ids: List[str] = field(default_factory=list)
    segment_name: str = ""
    own_client: bool = True


def _default_store_factory():
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as error:
        raise ImportError(
            "store_host_donor needs the Mooncake Python bindings (mooncake-transfer-engine)"
        ) from error
    return MooncakeDistributedStore()


def _default_device_name(protocol: str) -> str:
    from .connectors.mooncake_store.master import resolve_device_name

    return resolve_device_name(protocol, "")


def _parse(value: Any) -> int:
    from .connectors.mooncake_store.config import parse_size

    return parse_size(value, strict_units=True)


class StoreHostDonor:
    """This process's lent host memory. One per process; see `active_donor`."""

    def __init__(
        self,
        *,
        master: str,
        shares: Sequence[DonorShare],
        rank: int,
        local_rank: int,
        local_size: int,
        protocol: str = "rdma",
        device_name: str = "",
        local_buffer_size: int = _GIB,
        local_hostname: Optional[str] = None,
        late_mode: str = "auto",
        late_delay_seconds: float = 0.0,
        late_auto: Optional[LateAuto] = None,
        startup_per_node: int = 0,
        numa_nodes: Sequence[int] = (),
        head_role: Callable[..., tuple] = head_node_role,
        job_memory: Optional[Callable[[], Dict[str, Optional[int]]]] = None,
        node_lock: Any = _DEFAULT_LOCK,
        store_factory: Callable[[], Any] = _default_store_factory,
        bind_thread: Callable[[int], None] = _bind_current_thread,
        meminfo: Callable[[int], Dict[str, int]] = node_meminfo,
        egm: bool = False,
        egm_max_clients: int = EGM_MAX_CLIENTS_DEFAULT,
        egm_spill: bool = True,
        egm_usable: Callable[[int], Optional[int]] = node_egm_usable_bytes,
    ) -> None:
        if late_mode not in ("auto", "same_client", "new_client"):
            raise ValueError(f"{LOG_TAG}: unknown late_mode {late_mode!r}")
        self.egm = bool(egm)
        if self.egm:
            # Each EGM segment is its own client: a segment mounted on an
            # existing client is plain host memory peers cannot reach over
            # NVLink, and a provider needs no transfer buffer.
            protocol, device_name, local_buffer_size = EGM_PROTOCOL, "", 0
            late_mode = "new_client"
        if late_delay_seconds < 0:
            raise ValueError(f"{LOG_TAG}: late_delay_seconds must be nonnegative")
        self.master = master
        self.shares = list(shares)
        self.rank = rank
        self.local_rank = local_rank
        self.local_size = local_size
        self.protocol = protocol
        self.device_name = device_name
        self.local_buffer_size = local_buffer_size
        self.local_hostname = local_hostname
        self.late_mode = late_mode
        self.late_delay_seconds = float(late_delay_seconds)
        self.late_auto = late_auto
        self.startup_per_node = startup_per_node
        self.numa_nodes = list(numa_nodes)
        if late_auto is not None and not self.numa_nodes:
            raise ValueError(f"{LOG_TAG}: late_auto needs the NUMA node list")
        self._head_role = head_role
        self._job_memory = job_memory if job_memory is not None else _default_job_memory
        self._node_lock_path = NODE_LOCK_PATH if node_lock is _DEFAULT_LOCK else node_lock
        self.late_auto_record: Optional[Dict[str, Any]] = None
        self._late_auto_role: Optional[tuple] = None
        self._store_factory = store_factory
        self._bind_thread = bind_thread
        self._meminfo = meminfo
        self._lock = threading.Lock()
        self._registrations: List[_Registration] = []
        self._startup_thread: Optional[threading.Thread] = None
        self._startup_error: Optional[BaseException] = None
        self._late_thread: Optional[threading.Thread] = None
        self._late_error: Optional[BaseException] = None
        self._late_done = False
        self._closed = False
        self._stop = threading.Event()
        self._egm_max_clients = egm_max_clients
        #: EGM bytes of retired clients, held until the process exits.
        self._egm_retired_held = 0
        #: Why EGM lending stopped for good, or None.
        self._egm_stop: Optional[str] = None
        self._egm_spill = bool(egm_spill)
        self._egm_usable = egm_usable
        #: Per NUMA node: bytes failed EGM setups left charged to the job, which
        #: the driver hands to this process's next allocation on that node.
        self._egm_kept: Dict[int, int] = {}
        #: Static late shares that stayed short: [phase, home, short bytes,
        #: {node: bytes failed setups left kept}], retried in the background.
        self._egm_late_short: List[List[Any]] = []
        self._egm_late_retry_thread: Optional[threading.Thread] = None
        if self.egm:
            egm_wheel_check(None if store_factory is _default_store_factory else store_factory)
            latched = latch_egm_environment()
            record = {
                "role": "donor",
                "rank": rank,
                "local_rank": local_rank,
                "memory": "egm",
                "protocol": EGM_PROTOCOL,
                "numa": self.numa_nodes,
                "owned": [s.numa_node for s in self.shares],
                "wheel_has_egm": True,
                "align": EGM_ALIGN,
                "max_clients": egm_max_clients,
                "late_spill": self._egm_spill,
                "late_mode": self.late_mode,
                "env": latched,
            }
            logger.info(f"{EGM_TAG}_ENGAGED {json.dumps(record, sort_keys=True)}")

    # -- helpers ----------------------------------------------------------

    def _hostname(self) -> str:
        if self.local_hostname:
            return self.local_hostname
        from .connectors.mooncake_store.config import local_address

        return local_address(self.master)

    @contextlib.contextmanager
    def _node_serialized(self):
        """Hold the node-local lock; refuse to allocate when no lock can be taken.

        Without it two socket owners could each count the same cgroup room.
        The temporary directory is the fallback when the lock path is unusable.
        """
        handle = None
        if self._node_lock_path:
            errors = []
            for path in (
                self._node_lock_path,
                os.path.join(tempfile.gettempdir(), os.path.basename(self._node_lock_path)),
            ):
                try:
                    import fcntl

                    handle = open(path, "a")
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    break
                except (ImportError, OSError) as error:
                    errors.append(f"{path}: {error}")
                    if handle is not None:
                        handle.close()
                    handle = None
            if handle is None:
                raise RuntimeError(
                    f"{LOG_TAG}: cannot take the node-local lock ({'; '.join(errors)}); refusing "
                    "to allocate without serializing this node's owners"
                )
        try:
            yield
        finally:
            if handle is not None:
                handle.close()  # closing releases the flock

    def _check_headroom(
        self, node: int, size: int, client: bool = False, kept: int = 0
    ) -> Dict[str, int]:
        """Refuse an allocation the NUMA node or the job cgroup cannot take.

        ``kept`` is memory a failed EGM setup took from the node that this
        allocation reuses, so the NUMA node needs that much less free memory.
        It is an estimate, so the cgroup check below does not count on it.
        """
        info = self._meminfo(node)
        obtainable = node_obtainable_bytes(info)
        # Opening a client also allocates its transfer buffer under the binding.
        full = size + (self.local_buffer_size if client else 0)
        need = max(0, full - max(0, kept))
        if obtainable is not None and obtainable < need + NODE_HEADROOM_BYTES:
            raise RuntimeError(
                f"{LOG_TAG}: NUMA node {node} has {obtainable / _GIB:.1f} GiB free or "
                f"reclaimable, short of {need / _GIB:.1f} GiB (share and transfer buffer) "
                f"plus {NODE_HEADROOM_BYTES / _GIB:.0f} GiB headroom; refusing a bound "
                "allocation that would invoke the OOM killer"
            )
        # The pinned pages also count against the job's memory cgroup, which
        # can be far smaller than the node: refuse what cannot fit under it.
        job = self._job_memory()
        limit, nonfile = job.get("limit"), job.get("nonfile")
        if limit is not None and (nonfile is None or full > limit - nonfile):
            held = "unknown" if nonfile is None else f"{nonfile / _GIB:.1f} GiB"
            raise RuntimeError(
                f"{LOG_TAG}: {full / _GIB:.1f} GiB does not fit the memory cgroup "
                f"{job.get('path')} ({limit / _GIB:.1f} GiB limit, {held} held); refusing an "
                "allocation that could be OOM-killed"
            )
        return info

    def _log_registered(
        self,
        reg: _Registration,
        seconds: float,
        before: Dict[str, int],
        egm: Optional[Dict[str, Any]] = None,
    ) -> None:
        after = self._meminfo(reg.share.numa_node)
        drop = (
            (before.get("MemFree", 0) - after.get("MemFree", 0)) / _GIB
            if before and after
            else None
        )
        # MemFree alone under-reads when the kernel reclaims page cache to back the
        # bound pages; free plus reclaimable memory falls by what was allocated.
        was, now = node_obtainable_bytes(before), node_obtainable_bytes(after)
        taken = (was - now) / _GIB if was is not None and now is not None else None
        record = {
            "phase": reg.phase,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "numa_node": reg.share.numa_node,
            "bytes": reg.bytes,
            "gib": round(reg.bytes / _GIB, 3),
            "segment": reg.segment_name,
            "segment_ids": reg.segment_ids,
            "client": "new" if reg.own_client else "same",
            "master": self.master,
            "protocol": self.protocol,
            "seconds": round(seconds, 2),
            "node_memfree_drop_gib": None if drop is None else round(drop, 2),
            "node_obtainable_drop_gib": None if taken is None else round(taken, 2),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        if egm is not None:
            free_was, free_now = before.get("MemFree"), after.get("MemFree")
            record.update(
                memory="egm",
                kind=_egm_kind(reg.phase),
                egm_numa=reg.share.numa_node,
                numa_free_delta=(
                    None if free_was is None or free_now is None else free_was - free_now
                ),
                **egm,
            )
        logger.info(f"{LOG_TAG}_REGISTERED {json.dumps(record, sort_keys=True)}")

    def _run_bound(self, node: int, work: Callable[[], Any], name: str) -> Any:
        """Run ``work`` on a fresh thread whose memory policy is bound to ``node``.

        Threads Mooncake starts from it inherit the policy, and the caller's own
        policy is never touched.
        """
        box: Dict[str, Any] = {}

        def target():
            try:
                self._bind_thread(node)
                box["result"] = work()
            except BaseException as error:  # noqa: BLE001 - re-raised on the caller
                box["error"] = error

        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _fit_late_auto(self, reg: _Registration, client: bool) -> bool:
        """Shrink an AUTO late share to what its NUMA node and the job cgroup can back.

        The AUTO total is split evenly over the sockets, but the memory already
        held on a node need not be: lend less on the fuller socket instead of
        refusing. The node keeps the per-GPU margin of the ranks it hosts on top
        of the headroom, so the share never fills the socket to the edge.
        Returns False when nothing is left to lend on this node.
        Call with the node lock held.
        """
        auto = self.late_auto
        node = reg.share.numa_node
        buffer = self.local_buffer_size if client else 0
        ranks_on_node = -(-self.local_size // max(1, len(self.numa_nodes)))
        fits = []
        obtainable = node_obtainable_bytes(self._meminfo(node))
        if obtainable is not None:
            node_reserve = NODE_HEADROOM_BYTES + ranks_on_node * auto.margin_per_gpu
            fits.append(obtainable - buffer - node_reserve)
        job = self._job_memory()
        limits = [v for v in (job.get("limit"), auto.job_memory_limit) if v is not None]
        limit = min(limits) if limits else None
        nonfile = job.get("nonfile")
        if limit is not None and nonfile is not None:
            # Keep the peak-process reserve the planned size was computed with.
            peak = self.late_auto_record["process_peak_per_gpu"]
            reserve = self.local_size * (peak + auto.margin_per_gpu)
            fits.append(limit - nonfile - reserve - buffer)
        if not fits:
            return True
        granularity = auto.granularity
        fit = max(0, min(fits)) // granularity * granularity
        if fit >= reg.bytes:
            return True
        record = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "numa_node": node,
            "planned": reg.bytes,
            "bytes": fit,
            "node_obtainable": obtainable,
            "job_limit": limit,
            "job_nonfile_now": nonfile,
        }
        logger.warning(f"{LOG_TAG}_LATE_CLAMPED {json.dumps(record, sort_keys=True)}")
        reg.bytes = fit
        return fit > 0

    def _open_client(self, reg: _Registration, fit: bool = False) -> None:
        with self._node_serialized():
            if fit and not self._fit_late_auto(reg, client=True):
                return
            self._open_client_locked(reg)

    def _open_client_locked(self, reg: _Registration) -> None:
        if self.egm:
            self._open_egm_client_locked(reg)
            return
        node = reg.share.numa_node
        before = self._check_headroom(node, reg.bytes, client=True)
        hostname = self._hostname()
        started = time.monotonic()

        def work():
            store = self._store_factory()
            try:
                status = store.setup(
                    hostname,
                    METADATA_SERVER,
                    reg.bytes,
                    self.local_buffer_size,
                    self.protocol,
                    self.device_name,
                    self.master,
                )
                if status != 0:
                    raise RuntimeError(
                        f"{LOG_TAG}: MooncakeDistributedStore.setup failed with status {status} "
                        f"(master={self.master}, bytes={reg.bytes}, numa_node={node})"
                    )
            except BaseException:
                # A failed setup is never registered, so nothing else would
                # release what it allocated.
                with contextlib.suppress(Exception):
                    store.close()
                raise
            return store

        reg.store = self._run_bound(node, work, f"store-host-donor-{reg.phase}-n{node}")
        reg.segment_name = _segment_name(reg.store, hostname)
        with self._lock:
            self._registrations.append(reg)
        self._log_registered(reg, time.monotonic() - started, before)

    def _egm_live_clients(self) -> int:
        with self._lock:
            return sum(1 for r in self._registrations if r.own_client)

    def _open_egm_client_locked(self, reg: _Registration, kept: int = 0) -> None:
        """Open one client whose segment Mooncake allocates as EGM on the share's
        NUMA node. Call with the node lock held.

        ``kept`` is memory an earlier failed setup on this node left charged to
        the job; the driver backs this segment from it first, so the job cgroup
        grows by that much less.
        """
        node = reg.share.numa_node
        reg.bytes = reg.bytes // EGM_ALIGN * EGM_ALIGN
        if reg.bytes <= 0:
            return
        if self._egm_stop == "egm_uncharged" and _egm_kind(reg.phase) == "late":
            # The planned store (startup shares, the connector's segment) is
            # in the static budget already; nothing beyond it is lent.
            raise RuntimeError(
                f"{EGM_TAG}: an earlier EGM segment was not charged to the job cgroup; "
                f"refusing the {reg.phase} segment of {reg.bytes} bytes"
            )
        if self._egm_live_clients() >= self._egm_max_clients:
            raise RuntimeError(
                f"{EGM_TAG}: {self._egm_max_clients} EGM clients already open on rank "
                f"{self.rank} ({EGM_MAX_CLIENTS_ENV})"
            )
        before = self._check_headroom(node, reg.bytes, client=True, kept=kept)
        job_before = self._job_memory().get("nonfile")
        hostname = self._hostname()
        config = egm_setup_config(
            hostname=hostname, master=self.master, size=reg.bytes, numa_node=node
        )
        started = time.monotonic()

        def work():
            store = self._store_factory()
            _note_egm_client()
            try:
                status = store.setup(config)
            except BaseException:
                with contextlib.suppress(Exception):
                    store.close()
                raise
            if status != 0:
                with contextlib.suppress(Exception):
                    store.close()
                free = before.get("MemFree")
                raise EgmSetupError(
                    f"{EGM_TAG}: EGM setup failed with status {status} (master={self.master}, "
                    f"requested={reg.bytes}, numa_node={node}, numa_memfree="
                    f"{'unknown' if free is None else free}); no RDMA fallback"
                )
            return store

        reg.store = self._run_bound(node, work, f"store-host-donor-{reg.phase}-n{node}")
        reg.segment_name = _segment_name(reg.store, hostname)
        with self._lock:
            self._registrations.append(reg)
        job_after = self._job_memory().get("nonfile")
        cgroup_delta = None if job_before is None or job_after is None else job_after - job_before
        # Memory the driver pooled earlier (a failed setup, a closed client, of
        # this or another process of the job) was charged when it left the
        # node; reusing it grows neither the cgroup nor the node's usage. Only
        # memory that leaves the node uncharged is a shortfall.
        was = node_obtainable_bytes(before)
        now = node_obtainable_bytes(self._meminfo(node))
        node_taken = None if was is None or now is None else was - now
        egm = {"cgroup_delta": cgroup_delta, "node_taken": node_taken}
        if kept:
            egm["kept_reused"] = kept
        self._log_registered(reg, time.monotonic() - started, before, egm=egm)
        threshold = EGM_CHARGED_FRACTION * reg.bytes
        if (
            cgroup_delta is not None
            and cgroup_delta + max(0, kept) < threshold
            and (node_taken is None or node_taken - max(0, kept) >= threshold)
        ):
            # Host-memory accounting that reads the job cgroup would miss
            # this memory: stop lending rather than lend into it twice.
            self._egm_stop = "egm_uncharged"
            record = {
                "rank": self.rank,
                "phase": reg.phase,
                "numa_node": node,
                "bytes": reg.bytes,
                "cgroup_delta": cgroup_delta,
                "node_taken": node_taken,
                "kept": kept,
            }
            logger.error(f"{EGM_TAG}_UNCHARGED {json.dumps(record, sort_keys=True)}")

    def adopt_store_segment(self, size: int) -> int:
        """Provide this rank's KV connector segment as an EGM client on the rank's
        own NUMA node. Returns the bytes provided.

        Over NVLink the connector's own segment would be plain host memory its
        peers cannot reach, so the connector opens with none and the donor
        provides the same bytes instead, at the moment the connector would
        have mounted them.
        """
        if not self.egm:
            raise RuntimeError(f"{EGM_TAG}: segment adoption needs {STORE_EGM_ENV}=1")
        if not self.numa_nodes:
            raise RuntimeError(f"{EGM_TAG}: segment adoption needs the NUMA node list")
        # The inverse of `plan_shares`: ranks fill the sockets in order.
        node = self.numa_nodes[self.local_rank * len(self.numa_nodes) // self.local_size]
        reg = _Registration("startup_adopted", DonorShare(node, size, 0), size)
        self._open_client(reg)
        if reg not in self._registrations:
            raise RuntimeError(
                f"{EGM_TAG}: store segment of {size} bytes rounds to nothing at "
                f"{EGM_ALIGN} bytes per segment"
            )
        return reg.bytes

    def _mount_on(self, base: _Registration, reg: _Registration, fit: bool = False) -> None:
        with self._node_serialized():
            if fit and not self._fit_late_auto(reg, client=False):
                return
            self._mount_on_locked(base, reg)

    def _mount_on_locked(self, base: _Registration, reg: _Registration) -> None:
        if self.egm:
            raise RuntimeError(
                f"{EGM_TAG}: allocate_and_mount_segment mounts plain host memory that peers "
                "cannot reach over NVLink; every EGM segment needs its own client"
            )
        node = reg.share.numa_node
        before = self._check_headroom(node, reg.bytes)
        started = time.monotonic()
        store = base.store

        def work():
            return store.allocate_and_mount_segment(reg.bytes, self.protocol, f"cpu:{node}")

        result = self._run_bound(node, work, f"store-host-donor-{reg.phase}-n{node}")
        ids = result.get("segment_ids") if isinstance(result, dict) else None
        if isinstance(ids, list):
            reg.segment_ids = [str(i) for i in ids]
        ok = (
            isinstance(result, dict)
            and type(result.get("ret")) is int
            and result["ret"] == 0
            and type(result.get("allocated_size")) is int
            and result["allocated_size"] == reg.bytes
            and isinstance(ids, list)
            and ids
            and all(isinstance(i, str) and i for i in ids)
            and len(set(ids)) == len(ids)
        )
        reg.store = store
        reg.own_client = False
        reg.segment_name = base.segment_name
        if not ok:
            if reg.segment_ids:
                # Registered something we cannot account for: retire it.
                with self._lock:
                    self._registrations.append(reg)
                self._retire(reg)
            raise RuntimeError(f"{LOG_TAG}: allocate_and_mount_segment failed: {result!r}")
        with self._lock:
            self._registrations.append(reg)
        self._log_registered(reg, time.monotonic() - started, before)

    def _retire(self, reg: _Registration) -> bool:
        status: Any = 0
        try:
            if reg.own_client:
                if reg.store is not None:
                    status = reg.store.close()
            else:
                status = reg.store.unmount_and_free_segment(reg.segment_ids, 0)
        except Exception as error:  # noqa: BLE001 - reported, memory retained
            status = f"{type(error).__name__}: {error}"
        ok = status in (0, None)
        record = {
            "phase": reg.phase,
            "rank": self.rank,
            "numa_node": reg.share.numa_node,
            "bytes": reg.bytes,
            "segment": reg.segment_name,
            "segment_ids": reg.segment_ids,
            "status": str(status),
            "ok": ok,
        }
        if self.egm:
            # Closing does not hand EGM back to the job promptly: count it as
            # held until the process exits, so nothing is lent into it.
            self._egm_retired_held += reg.bytes
            record.update(
                memory="egm",
                kind=_egm_kind(reg.phase),
                egm_numa=reg.share.numa_node,
                held_until_exit=True,
                egm_retired_held=self._egm_retired_held,
            )
        if ok:
            logger.info(f"{LOG_TAG}_RETIRED {json.dumps(record, sort_keys=True)}")
            with self._lock:
                if reg in self._registrations:
                    self._registrations.remove(reg)
        else:
            logger.error(
                f"{LOG_TAG}_RETIRE_FAILED {json.dumps(record, sort_keys=True)}; "
                "memory retained until process exit"
            )
        return ok

    def _base_for(self, node: int) -> Optional[_Registration]:
        with self._lock:
            for reg in self._registrations:
                if reg.phase == "startup" and reg.share.numa_node == node:
                    return reg
        return None

    # -- startup ------------------------------------------------------------

    def _late_auto_head(self) -> tuple:
        """(head, basis) for AUTO sizing, decided once so startup and late agree."""
        if self._late_auto_role is None:
            self._late_auto_role = self._head_role(self.master, self.late_auto.head_hosts)
        return self._late_auto_role

    def _late_auto_upper_bound(self) -> int:
        """This worker's AUTO late total before the job-room cap; 0 when nothing can be lent."""
        auto = self.late_auto
        head, _ = self._late_auto_head()
        startup_per_gpu = -(-self.startup_per_node // self.local_size)
        worker = max(0, late_auto_per_gpu(auto, head) - startup_per_gpu) * self.local_size
        unit = len(self.numa_nodes) * auto.granularity
        return worker // unit * unit

    def _register_startup(self) -> None:
        # A late segment mounted on the same client needs that client even
        # when the startup share is zero. Under late_auto the late size is
        # unknown until after warmup, but its upper bound tells whether any
        # late share can follow.
        late_pending = (
            self.late_mode != "new_client"
            and self.late_auto is not None
            and self.late_auto_record is None
            and self._late_auto_upper_bound() > 0
        )
        for share in self.shares:
            if self._stop.is_set():
                return
            # An AUTO late share is sized against one NUMA node's free memory, so it
            # must land there: a segment set up by a new client can be spread over
            # every socket, while a segment mounted at "cpu:<node>" stays on it.
            needs_client = (
                share.startup_bytes > 0
                or (share.late_bytes > 0 and self.late_mode == "same_client")
                or late_pending
            )
            if needs_client:
                self._open_client(_Registration("startup", share, share.startup_bytes))

    def _late_base(self, node: int) -> Optional[_Registration]:
        """The client a late segment on ``node`` mounts on, or None for a new client."""
        if self.egm or self.late_mode == "new_client":
            return None
        base = self._base_for(node)
        if self.late_mode == "same_client":
            if base is None:
                raise RuntimeError(
                    f"{LOG_TAG}: no startup client on NUMA node {node} for the late segment"
                )
            return base
        # auto: reuse the startup client when its bindings can mount more.
        if base is not None and callable(getattr(base.store, "allocate_and_mount_segment", None)):
            return base
        return None

    def start_startup(self) -> None:
        """Begin the startup registrations on a background thread."""
        if self._startup_thread is not None:
            raise RuntimeError(f"{LOG_TAG}: startup already started")
        if not self.shares:
            return

        def target():
            try:
                self._register_startup()
            except BaseException as error:  # noqa: BLE001 - surfaced by wait_startup
                self._startup_error = error
                logger.error(f"{LOG_TAG}_FAILED phase=startup rank={self.rank}: {error}")

        self._startup_thread = threading.Thread(
            target=target, name="store-host-donor-startup", daemon=True
        )
        self._startup_thread.start()

    def wait_startup(self, timeout: Optional[float] = None) -> None:
        """Join the startup thread and raise its error, if any."""
        thread = self._startup_thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError(f"{LOG_TAG}: startup registration still running")
        if self._startup_error is not None:
            raise RuntimeError(f"{LOG_TAG}: startup registration failed") from self._startup_error

    # -- late -----------------------------------------------------------------

    def _resolve_late_auto(self) -> None:
        """Turn ``late_auto`` into this rank's late shares (once, after warmup)."""
        auto = self.late_auto
        if auto is None or self.late_auto_record is not None:
            return
        head, basis = self._late_auto_head()
        # The startup share already counts against the same per-GPU budget.
        startup_per_gpu = -(-self.startup_per_node // self.local_size)
        per_gpu = max(0, late_auto_per_gpu(auto, head) - startup_per_gpu)
        worker = per_gpu * self.local_size
        job = self._job_memory()
        observed = job.get("limit")
        limits = [v for v in (observed, auto.job_memory_limit) if v is not None]
        # A configured limit can only tighten the observed one.
        limit = min(limits) if limits else None
        nonfile = job.get("nonfile")
        if nonfile is None and observed is None:
            # No cgroup is limited: a configured limit is checked against the
            # usage of this process's own cgroup.
            nonfile = job.get("usage")
        peak = auto.head_process_peak_per_gpu if head else auto.process_peak_per_gpu
        room = None
        if limit is None or nonfile is None:
            # Without the job's limit and current usage the bound cannot be
            # checked: lend nothing rather than assume room.
            worker = 0
        else:
            # Room under the job limit, reserving this worker's full peak process
            # memory and margin on top of what the job holds now.
            room = max(0, limit - nonfile - self.local_size * (peak + auto.margin_per_gpu))
            worker = min(worker, room)
        count = len(self.numa_nodes)
        unit = count * auto.granularity
        total = worker // unit * unit
        late = {
            s.numa_node: s.late_bytes
            for s in plan_shares(0, total, self.numa_nodes, self.local_rank, self.local_size)
        }
        self.shares = [
            DonorShare(
                numa_node=s.numa_node,
                startup_bytes=s.startup_bytes,
                late_bytes=late.get(s.numa_node, 0),
            )
            for s in self.shares
        ]
        self.late_auto_record = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "local_size": self.local_size,
            "host": socket.gethostname(),
            "head": head,
            "head_basis": basis,
            "budget_per_gpu": auto.host_budget_per_gpu,
            "store_per_gpu": auto.store_per_gpu,
            "process_peak_per_gpu": peak,
            "margin_per_gpu": auto.margin_per_gpu,
            "late_per_gpu": per_gpu,
            "worker_late": total,
            "worker_late_gib": round(total / _GIB, 3),
            "job_limit": limit,
            "job_limit_observed": observed,
            "job_nonfile_now": nonfile,
            "job_room": room,
            "job_bound_known": limit is not None and nonfile is not None,
            "capped_by_job_limit": room is None or room < per_gpu * self.local_size,
            "owned": {str(node): size for node, size in late.items()},
            "startup_per_gpu": startup_per_gpu,
            "per_gpu_total_gib": round(
                (auto.store_per_gpu + peak + startup_per_gpu + total / self.local_size) / _GIB, 3
            ),
        }
        if self.egm:
            self.late_auto_record["memory"] = "egm"
        logger.info(f"{LOG_TAG}_LATE_AUTO {json.dumps(self.late_auto_record, sort_keys=True)}")

    def _register_late_now(self) -> None:
        self._resolve_late_auto()
        for share in self.shares:
            if share.late_bytes <= 0 or self._stop.is_set():
                continue
            reg = _Registration("late", share, share.late_bytes)
            base = self._late_base(share.numa_node)
            # Only an AUTO size may shrink; a fixed size is refused when it does not fit.
            fit = self.late_auto is not None
            if base is not None:
                self._mount_on(base, reg, fit=fit)
            elif self.egm:
                if not self._open_egm_late(reg, fit=fit):
                    break
            else:
                self._open_client(reg, fit=fit)
        self._late_done = True
        if self.egm:
            self._egm_start_late_retry()

    def _open_egm_late(self, reg: _Registration, fit: bool) -> bool:
        """Lend a planned late share as EGM. Returns False once lending on this
        rank has stopped for good (client cap, uncharged memory).

        See `_lend_egm_locked` for how a share that Mooncake cannot set up in
        one piece is still lent in full. Startup shares and the adopted
        connector segment stay fatal.
        """
        if self._egm_stop == "egm_uncharged":
            # The planned store is in the static budget already; nothing
            # beyond it is lent.
            raise RuntimeError(
                f"{EGM_TAG}: an earlier EGM segment was not charged to the job cgroup; "
                f"refusing the {reg.phase} segment of {reg.bytes} bytes"
            )
        with self._node_serialized():
            if fit and not self._fit_late_auto(reg, client=True):
                return True
            summary = self._lend_egm_locked(reg.phase, reg.share.numa_node, reg.bytes, EGM_ALIGN)
        if summary["short"] >= EGM_ALIGN and self._egm_stop is None:
            kept = {int(n): b for n, b in summary["kept"].items()}
            self._egm_late_short.append([reg.phase, reg.share.numa_node, summary["short"], kept])
        return self._egm_stop is None

    def _egm_late_retry_loop(self, interval: float, passes: int) -> None:
        """Retry the static late EGM shares that stayed short, every
        ``interval`` seconds, at most ``passes`` times.

        Each retry asks only for the bytes still unlent. Memory the earlier
        failures left kept counts as growth already made, so the share never
        grows the job beyond its plan; it is also the first memory a retry on
        that node reuses.
        """
        for attempt in range(1, passes + 1):
            if self._stop.wait(interval) or self._egm_stop is not None:
                return
            with self._node_serialized():
                for entry in self._egm_late_short:
                    phase, home, short, kept = entry
                    if short < EGM_ALIGN or self._stop.is_set():
                        continue
                    summary = self._lend_egm_locked(
                        f"{phase}_retry{attempt}", home, short, EGM_ALIGN, carried=kept
                    )
                    entry[2] = short - summary["lent"]
                    entry[3] = {int(n): b for n, b in summary["kept"].items()}
            left = sum(entry[2] for entry in self._egm_late_short)
            if left < EGM_ALIGN or self._egm_stop is not None:
                break
        left = sum(entry[2] for entry in self._egm_late_short)
        record = {
            "rank": self.rank,
            "short": left,
            "kept": sum(sum(entry[3].values()) for entry in self._egm_late_short),
            "egm_stop": self._egm_stop,
        }
        if left >= EGM_ALIGN:
            logger.error(f"{EGM_TAG}_LATE_INCOMPLETE {json.dumps(record, sort_keys=True)}")
        else:
            logger.info(f"{EGM_TAG}_LATE_COMPLETE {json.dumps(record, sort_keys=True)}")

    def _egm_start_late_retry(self) -> None:
        if not any(entry[2] >= EGM_ALIGN for entry in self._egm_late_short):
            return
        interval = _env_number(os.environ, EGM_LATE_RETRY_SECONDS_ENV)
        interval = EGM_LATE_RETRY_SECONDS_DEFAULT if interval is None else interval

        def target():
            try:
                self._egm_late_retry_loop(interval, EGM_LATE_RETRY_PASSES)
            except BaseException as error:  # noqa: BLE001 - nobody to raise to
                logger.error(f"{LOG_TAG}_FAILED phase=late_retry rank={self.rank}: {error}")

        with self._lock:
            # No retry once close has begun: close joins only a thread it sees.
            if self._stop.is_set():
                return
            self._egm_late_retry_thread = threading.Thread(
                target=target, name="store-host-donor-late-retry", daemon=True
            )
            self._egm_late_retry_thread.start()

    def _egm_settle(self, node: int) -> float:
        """Wait until the job cgroup and NUMA node ``node`` hold still after a
        failed EGM setup (at most `EGM_SETTLE_SECONDS`). Returns the seconds
        waited. A retry sized while the driver is still returning pages could
        take new memory instead of reusing the kept memory."""
        if EGM_SETTLE_INTERVAL <= 0:
            return 0.0
        start = time.monotonic()

        def read():
            return (
                self._job_memory().get("nonfile"),
                self._meminfo(node).get("MemFree"),
            )

        last = read()
        while time.monotonic() - start < EGM_SETTLE_SECONDS:
            if self._stop.wait(EGM_SETTLE_INTERVAL):
                break
            now = read()
            if all(
                a is None or b is None or abs(a - b) < EGM_SETTLE_STILL_BYTES
                for a, b in zip(last, now)
            ):
                break
            last = now
        return round(time.monotonic() - start, 3)

    def _egm_next_size(
        self,
        remaining: int,
        kept: int,
        usable: Optional[int],
        last_failed: Optional[int],
        failed_kept: int = 0,
    ) -> int:
        """Bytes the next EGM setup on one NUMA node asks for (before clamps)."""
        if last_failed is None:
            return remaining
        if kept > 0 or usable is not None:
            size = kept + max(0, (usable or 0) - EGM_USABLE_MARGIN)
            # Segments are whole EGM_ALIGN units: round up, so the kept memory
            # is reused in full rather than leaving a stranded part.
            size = -(-size // EGM_ALIGN) * EGM_ALIGN
        else:
            size = last_failed // 2
        if size < last_failed:
            return size
        if size == last_failed and kept >= failed_kept + EGM_ALIGN:
            # The size failed before the failure left this memory kept: now
            # only the last partial unit is new, so it is worth one more try.
            return size
        # Never repeat a size that already failed with no more memory known:
        # the kept memory alone is the likeliest to back a segment.
        return kept if 0 < kept < last_failed else last_failed // 2

    def _lend_egm_locked(
        self,
        phase: str,
        home: int,
        planned: int,
        min_segment: int,
        carried: Optional[Mapping[int, int]] = None,
    ) -> Dict[str, Any]:
        """Lend ``planned`` bytes of EGM, starting on NUMA node ``home``, as one
        or more segments. Call with the node lock held.

        An EGM segment is one HOST_NUMA allocation, built from free blocks of
        `EGM_HOST_BLOCK` and more; MemFree also counts smaller free pieces, so
        a lend sized from it can fail on a fragmented node. A failed setup leaves what
        it took charged to the job, and the driver hands that memory to this
        process's next allocation on the node. So after a failure the lend is
        retried on the same node at once, sized from that kept memory plus the
        node's free large blocks, and the part the node cannot back moves to
        the node's other NUMA nodes (unless `EGM_LEND_SPILL_ENV` is off).

        The job grows by at most ``planned``: each setup asks for no more than
        the plan less what the job has grown by since the call started (lent
        segments and memory failed setups kept, measured from the job cgroup
        when it can be read), plus the memory kept on the node it is placed
        on, which it reuses first. Kept memory is known only from this call's
        failures and is re-measured after each one, so nothing stale from an
        earlier call or another process is counted. Memory that stays kept is
        charged to the job, and the next setup on that node reuses it first. A lend that stays
        short holds only for this call; later calls try again. ``carried`` is
        memory an earlier call for the same plan left kept, per node: it counts
        as growth already made and may be reused on its node, but it is never
        trusted to relax the NUMA headroom check.

        The bound is on EGM bytes (lent segments plus memory failed setups
        kept, re-measured from the job cgroup after each failure); the small
        process-side cost of each setup (tens of MiB) is outside it and is
        covered by the host memory budget. The last segment may be smaller than
        ``min_segment`` (down to `EGM_ALIGN`) when it completes the plan.

        Returns the summary that is also logged.
        """
        align = EGM_ALIGN
        planned = max(0, planned) // align * align
        min_segment = max(align, min_segment // align * align)
        nodes = [home]
        if self._egm_spill:
            nodes += [n for n in self.numa_nodes if n != home]
        job_start = self._job_memory().get("nonfile")
        lent, failures = 0, 0
        # Kept memory from earlier calls may have been reused by another
        # process since: only this call's failures count.
        self._egm_kept = {}
        carry: Dict[int, int] = {int(n): max(0, int(b)) for n, b in (carried or {}).items()}

        def grown() -> int:
            # EGM bytes this plan has taken: lent segments, memory this call's
            # failures kept (re-measured after each), and memory carried.
            return lent + sum(self._egm_kept.values()) + sum(carry.values())

        def completes(left: int) -> bool:
            return left >= min_segment or (lent > 0 and left >= align)

        segments: List[Dict[str, Any]] = []
        refused: List[Dict[str, Any]] = []
        for node in nodes:
            last_failed: Optional[int] = None
            failed_kept = 0
            fails = 0
            while completes(planned - lent) and not self._stop.is_set():
                if self._egm_stop is None and self._egm_live_clients() >= self._egm_max_clients:
                    self._egm_stop = "egm_client_cap"
                if self._egm_stop is not None:
                    break
                remaining = planned - lent
                kept = self._egm_kept.get(node, 0)
                pooled = kept + carry.get(node, 0)
                # The setup reuses this node's kept memory first, so it grows
                # the job by at most size - kept.
                budget = planned - grown() + pooled
                usable = self._egm_usable(node)
                size = self._egm_next_size(remaining, pooled, usable, last_failed, failed_kept)
                obtainable = node_obtainable_bytes(self._meminfo(node))
                if obtainable is not None:
                    size = min(size, obtainable + kept - NODE_HEADROOM_BYTES)
                size = max(0, min(size, remaining, budget)) // align * align
                if size < (min_segment if size < remaining else align):
                    break
                reg = _Registration(phase, DonorShare(node, 0, size), size)
                free_before = self._meminfo(node).get("MemFree")
                job_before = self._job_memory().get("nonfile")
                try:
                    self._open_egm_client_locked(reg, kept=min(kept, size))
                except EgmSetupError as error:
                    failures, fails, last_failed, failed_kept = (
                        failures + 1,
                        fails + 1,
                        size,
                        pooled,
                    )
                    settled = self._egm_settle(node)
                    free_after = self._meminfo(node).get("MemFree")
                    job_after = self._job_memory().get("nonfile")
                    if job_before is not None and job_after is not None:
                        taken, basis = job_after - job_before, "job_cgroup"
                    elif free_before is not None and free_after is not None:
                        taken, basis = free_before - free_after, "numa_memfree"
                    else:
                        taken, basis = 0, "unknown"
                    kept = kept + max(0, min(size, taken))
                    if job_start is not None and job_after is not None:
                        # Re-measure: all kept memory is job growth not lent.
                        others = sum(b for n, b in self._egm_kept.items() if n != node)
                        unlent = job_after - job_start - lent
                        kept = max(0, min(kept, unlent - others))
                    self._egm_kept[node] = kept
                    record = {
                        "rank": self.rank,
                        "phase": phase,
                        "numa_node": node,
                        "home": home,
                        "requested": size,
                        "failure": fails,
                        "kept": self._egm_kept[node],
                        "kept_basis": basis,
                        "settle_seconds": settled,
                        "usable_before": usable,
                        "usable_after": self._egm_usable(node),
                        "numa_memfree_after": free_after,
                        "lent_so_far": lent,
                        "planned": planned,
                        "error": str(error),
                    }
                    logger.warning(f"{EGM_TAG}_LEND_RETRY {json.dumps(record, sort_keys=True)}")
                    if fails >= EGM_LEND_FAILS_PER_NODE:
                        break
                    continue
                except RuntimeError as error:
                    # Headroom, cgroup or stop refusals: this node takes no more now.
                    refused.append({"numa_node": node, "bytes": size, "error": str(error)})
                    break
                if reg not in self._registrations:
                    break
                lent += reg.bytes
                reused = min(carry.get(node, 0), reg.bytes)
                if reused:
                    carry[node] -= reused
                self._egm_kept[node] = max(0, kept - (reg.bytes - reused))
                segments.append(
                    {
                        "numa_node": node,
                        "bytes": reg.bytes,
                        "segment": reg.segment_name,
                        "kept_reused": min(kept, reg.bytes),
                    }
                )
            if self._egm_stop is not None or not completes(planned - lent):
                break
        job_end = self._job_memory().get("nonfile")
        growth = None if job_start is None or job_end is None else job_end - job_start
        summary = {
            "rank": self.rank,
            "phase": phase,
            "home": home,
            "planned": planned,
            "lent": lent,
            "short": planned - lent,
            "failures": failures,
            "segments": segments,
            "refused": refused,
            "kept": {
                str(n): self._egm_kept.get(n, 0) + carry.get(n, 0)
                for n in set(self._egm_kept) | set(carry)
                if self._egm_kept.get(n, 0) + carry.get(n, 0)
            },
            "cgroup_growth": growth,
            "max_growth": planned,
            "spill": self._egm_spill,
            "egm_stop": self._egm_stop,
        }
        if failures or refused or summary["short"] >= min_segment:
            logger.warning(f"{EGM_TAG}_LATE_LEND {json.dumps(summary, sort_keys=True)}")
        else:
            logger.info(f"{EGM_TAG}_LATE_LEND {json.dumps(summary, sort_keys=True)}")
        if sum(int(b) for b in summary["kept"].values()) > EGM_KEPT_REPORT_BYTES or (
            growth is not None and growth > lent + EGM_KEPT_REPORT_BYTES
        ):
            # Charged to the job (so accounting that reads the job cgroup sees
            # it) but not lent: the next lend on that node reuses it first.
            logger.warning(f"{EGM_TAG}_KEPT {json.dumps(summary, sort_keys=True)}")
        return summary

    @property
    def has_late(self) -> bool:
        if self.late_auto is not None and self.late_auto_record is None:
            return True
        return any(share.late_bytes > 0 for share in self.shares)

    def register_late(self) -> None:
        """Register the late portion now (after warmup, before serving)."""
        self.wait_startup()
        self._register_late_now()

    def schedule_late(self) -> None:
        """Register the late portion ``late_delay_seconds`` from now, in the background."""
        self.wait_startup()
        if not self.has_late:
            return

        def target():
            if self._stop.wait(self.late_delay_seconds):
                return
            try:
                self._register_late_now()
            except BaseException as error:  # noqa: BLE001 - nobody to raise to
                self._late_error = error
                logger.error(f"{LOG_TAG}_FAILED phase=late rank={self.rank}: {error}")

        self._late_thread = threading.Thread(
            target=target, name="store-host-donor-late", daemon=True
        )
        self._late_thread.start()

    # -- shutdown -------------------------------------------------------------

    def close(self) -> bool:
        """Release every registration, late ones first. Idempotent.

        Returns False when some memory could not be released; it then stays
        registered and allocated until the process exits.
        """
        if self._closed:
            return True
        self._stop.set()
        # The late thread may start the static retry thread, so it is joined
        # first; a retry setup in flight registers before its thread ends, so
        # what it lent is released below rather than left behind.
        for name in ("_late_thread", "_startup_thread", "_egm_late_retry_thread"):
            with self._lock:
                thread = getattr(self, name)
            if thread is not None and thread is not threading.current_thread():
                thread.join(JOIN_TIMEOUT_S)
                if thread.is_alive():
                    logger.error(
                        f"{LOG_TAG}: registration thread still running at close; "
                        "memory retained until process exit"
                    )
                    return False
        with self._lock:
            pending = list(self._registrations)
        ok = True
        # Segments mounted on another registration's client go first; a client
        # is closed only once nothing mounted on it is left.
        for reg in sorted(pending, key=lambda r: r.own_client):
            if reg.own_client:
                with self._lock:
                    children = [
                        c for c in self._registrations if not c.own_client and c.store is reg.store
                    ]
                if children:
                    logger.error(
                        f"{LOG_TAG}: keeping the NUMA node {reg.share.numa_node} client "
                        f"open: {len(children)} segment(s) mounted on it were not "
                        "released; memory retained until process exit"
                    )
                    ok = False
                    continue
            ok = self._retire(reg) and ok
        self._closed = ok
        return ok

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            regs = list(self._registrations)
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "shares": [share.__dict__ for share in self.shares],
            "registered_bytes": sum(r.bytes for r in regs),
            "registrations": [(r.phase, r.share.numa_node, r.bytes) for r in regs],
        }


# --------------------------------------------------------------------------
# Process-wide instance
# --------------------------------------------------------------------------

_ACTIVE: Any = None
_ACTIVE_LOCK = threading.Lock()
# Holds the slot while one caller builds its donor, so a concurrent start fails
# instead of creating a second donor that cleanup cannot reach.
_RESERVED = object()


def active_donor() -> Optional[StoreHostDonor]:
    donor = _ACTIVE
    return None if donor is _RESERVED else donor


def adopt_store_segment(size: int) -> int:
    """Have this process's donor provide the KV connector's segment over EGM.

    Returns the bytes provided. The connector calls this under
    ``TLLM_MOONCAKE_STORE_EGM=1`` in place of mounting the segment itself.
    """
    donor = active_donor()
    if donor is None:
        raise RuntimeError(
            f"{EGM_TAG}: {STORE_EGM_ENV}=1 moves the KV connector's {size}-byte segment to "
            "the store host donor, but this server runs none; set store_host_donor"
        )
    return donor.adopt_store_segment(size)


def close_active_donor() -> bool:
    """Release the process's donor, if any. Safe to call more than once."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        donor = _ACTIVE
    if donor is None or donor is _RESERVED:
        return True
    ok = donor.close()
    if ok:
        with _ACTIVE_LOCK:
            if _ACTIVE is donor:
                _ACTIVE = None
    return ok


def _late_auto_from_config(config: Any, environ: Mapping[str, str]) -> LateAuto:
    auto = LateAuto(
        host_budget_per_gpu=_parse(config.host_budget_per_gpu),
        store_per_gpu=_parse(config.store_per_gpu),
        process_peak_per_gpu=_parse(config.process_peak_per_gpu),
        head_process_peak_per_gpu=_parse(config.head_process_peak_per_gpu),
        margin_per_gpu=_parse(config.margin_per_gpu),
        granularity=_parse(config.granularity),
        head_hosts=tuple(config.head_hosts) if config.head_hosts else None,
        job_memory_limit=(
            None if config.job_memory_limit is None else _parse(config.job_memory_limit)
        ),
    )
    path = environ.get(CONFIG_PATH_ENV)
    if path:
        try:
            with open(path) as handle:
                segment = json.load(handle).get("global_segment_size")
        except (OSError, ValueError):
            segment = None
        if segment is not None and _parse(segment) != auto.store_per_gpu:
            raise ValueError(
                f"{LOG_TAG}: late_auto.store_per_gpu={auto.store_per_gpu} bytes but "
                f"{path} sets global_segment_size={segment}; the budget would not hold"
            )
    return auto


def start_store_host_donor(
    config: Optional["StoreHostDonorConfig"],
    *,
    rank: int,
    local_rank: int,
    local_size: int,
    environ: Optional[Mapping[str, str]] = None,
    numa_nodes: Optional[Sequence[int]] = None,
    **donor_kwargs,
) -> Optional[StoreHostDonor]:
    """Create this process's donor from ``llm_args.store_host_donor`` and start
    its startup registrations. Returns None when the feature is off.

    Every rank gets a donor when the config is set, including ranks that own
    no share, so the collective steps around the late phase stay uniform.
    """
    global _ACTIVE
    environ = os.environ if environ is None else environ
    if config is None:
        return None
    if environ.get(DISABLE_ENV, "0") == "1":
        logger.warning(f"{LOG_TAG}: {DISABLE_ENV}=1, store_host_donor ignored")
        return None
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            raise RuntimeError(f"{LOG_TAG}: a donor is already active in this process")
        _ACTIVE = _RESERVED
    try:
        donor = _build_donor(
            config, rank, local_rank, local_size, environ, numa_nodes, donor_kwargs
        )
    except BaseException:
        with _ACTIVE_LOCK:
            if _ACTIVE is _RESERVED:
                _ACTIVE = None
        raise
    with _ACTIVE_LOCK:
        _ACTIVE = donor
    donor.start_startup()
    return donor


def _late_sizing(config: Any, environ: Mapping[str, str]) -> tuple:
    """(late_auto, fixed late bytes per node) of a config.

    ``auto`` sizes the late portion from the ``late_auto`` budget; anything
    else is a fixed size per node.
    """
    if config.late_size_per_node == "auto":
        if config.late_auto is None:
            raise ValueError(f"{LOG_TAG}: late_size_per_node 'auto' needs late_auto")
        return _late_auto_from_config(config.late_auto, environ), 0
    return None, _parse(config.late_size_per_node)


def _build_donor(config, rank, local_rank, local_size, environ, numa_nodes, donor_kwargs):
    """Plan, construct and log one donor (the caller holds the process slot)."""
    nodes = (
        list(config.numa_nodes)
        if config.numa_nodes
        else (list(numa_nodes) if numa_nodes is not None else cpu_numa_nodes())
    )
    startup = _parse(config.startup_size_per_node)
    auto, late = _late_sizing(config, environ)
    shares = plan_shares(startup, late, nodes, local_rank, local_size)
    master = resolve_master(config.master, environ)
    egm = store_egm_enabled(environ)
    if egm:
        donor_kwargs = {
            "egm": True,
            "egm_max_clients": egm_max_clients(environ),
            "egm_spill": egm_lend_spill(environ),
            **donor_kwargs,
        }
    device_name = config.device_name
    if device_name is None:
        device_name = _default_device_name(config.protocol) if shares and not egm else ""
    donor = StoreHostDonor(
        master=master,
        shares=shares,
        rank=rank,
        local_rank=local_rank,
        local_size=local_size,
        protocol=config.protocol,
        device_name=device_name,
        local_buffer_size=_parse(config.local_buffer_size),
        local_hostname=config.local_hostname,
        late_mode=config.late_mode,
        late_delay_seconds=config.late_delay_seconds,
        late_auto=auto,
        startup_per_node=startup,
        numa_nodes=nodes,
        **donor_kwargs,
    )
    plan = {
        "rank": rank,
        "local_rank": local_rank,
        "local_size": local_size,
        "numa_nodes": nodes,
        "startup_per_node": startup,
        "late_per_node": "auto" if auto is not None else late,
        "late_auto": None
        if auto is None
        else {k: v for k, v in auto.__dict__.items() if k != "head_hosts"}
        | {"head_hosts": list(auto.head_hosts) if auto.head_hosts else None},
        "owned": [share.__dict__ for share in shares],
        "master": master,
        "late_mode": config.late_mode,
        "late_delay_seconds": config.late_delay_seconds,
    }
    if egm:
        plan.update(memory="egm", protocol=EGM_PROTOCOL, late_mode=donor.late_mode)
    logger.info(f"{LOG_TAG}_PLAN {json.dumps(plan, sort_keys=True)}")
    return donor


atexit.register(close_active_donor)
