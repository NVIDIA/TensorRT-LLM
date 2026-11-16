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
``trtllm-serve`` and under any frontend that builds a ``PyExecutor`` (Dynamo's
``dynamo.trtllm`` workers included)::

    store_host_donor:
      startup_size_per_node: 1088GiB   # mounted while the model loads
      late_size_per_node: 96GiB        # mounted after warmup
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

``late_size_per_node: auto`` sizes the late portion per GPU from a host
memory budget instead (``late_auto``)::

    late_size_per_node: auto
    late_auto:
      host_budget_per_gpu: 357.5GiB      # store + process one GPU may hold
      store_per_gpu: 290GiB              # the KV connector's segment
      process_peak_per_gpu: 29.6GiB      # measured, nodes without the frontend
      head_process_peak_per_gpu: 64.1GiB # measured, the head (frontend) node

Each GPU then lends ``max(0, budget - store - peak - margin)``, with the head
node's peak where the pool master runs (or on ``head_hosts``), cut to what the
job's memory limit leaves and rounded down per NUMA node. A node whose role
cannot be determined is sized as a head node. The computed size is logged as
one ``STORE_HOST_DONOR_LATE_AUTO`` line per rank.

``late_size_per_node: auto`` without ``late_auto`` (or
``TRTLLM_STORE_HOST_DONOR_AUTO=1``, which selects it on any config) lends
from MEASURED memory instead of planned peaks. After warmup (a server holding a
KV connector, i.e. a context server, first waits for its first request) a
background thread lends, and re-checks every few minutes::

    lend = budget - usage now - buffer

* budget: per used GPU ``min(MemTotal / GPUs on the node, cgroup memory.max /
  GPUs in use on the node)``, times the GPUs in use. Co-located workers
  register their GPUs in a node-local registry (``/dev/shm``, under the node
  lock), so they count and split one budget.
* usage: the job cgroup's memory that reclaim cannot free (``memory.current -
  file + shmem``, the quantity the runtime X/4 reader audits), or the node's
  ``/proc/meminfo`` equivalent without a cgroup limit. It includes the store
  segments already registered.
* buffer: 36 GiB per GPU, 64 GiB per GPU on the head node (the node that runs
  the pool master, decided like ``late_auto``). Never lent.
* each worker lends at most its GPU-weighted share of the node's room, and
  stops topping up while the usage slope (least squares over the last
  minutes, lends excluded) would use up the buffer before the job ends
  (``SLURM_JOB_END_TIME``, then ``scontrol``, then a fixed horizon). The
  first lend waits for that slope too, so every lend is checked against the
  growth still to come.

Every lend logs one ``STORE_HOST_DONOR_AUTO_LEND`` line and every pass that
lends nothing one ``STORE_HOST_DONOR_AUTO_HOLD`` line, each with the budget,
usage, buffer, amount and reason. Optional environment overrides:
``TRTLLM_STORE_HOST_DONOR_LATE_GIB`` (fixed late GiB per node; skips every
auto mode), ``..._BUFFER_GIB`` / ``..._HEAD_BUFFER_GIB`` (per GPU),
``..._MAX_LENT_GIB`` (per node), ``..._TOPUP_S`` and
``TRTLLM_STORE_HOST_DONOR_AUTO=0`` (no measured lending); see `MeasuredAuto`.
Lent memory cannot be taken back.

Every registration logs one ``STORE_HOST_DONOR_REGISTERED`` line and every
release one ``STORE_HOST_DONOR_RETIRED`` line.
``TRTLLM_DISABLE_STORE_HOST_DONOR=1`` turns the feature off.
"""

import atexit
import contextlib
import ctypes
import json
import os
import platform
import re
import shutil
import socket
import subprocess  # nosec B404 - runs scontrol with a fixed argv, no shell
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import StoreHostDonorConfig

__all__ = [
    "DISABLE_ENV",
    "MASTER_ENV",
    "DonorShare",
    "LateAuto",
    "MeasuredAuto",
    "MeasuredProbes",
    "NodeRegistry",
    "StoreHostDonor",
    "active_donor",
    "close_active_donor",
    "cgroup_memory",
    "cpu_numa_nodes",
    "head_node_role",
    "growth_slope",
    "job_end_epoch",
    "late_auto_per_gpu",
    "measured_auto_from_env",
    "measured_lend",
    "node_gpu_count",
    "node_memtotal",
    "plan_shares",
    "resolve_master",
    "start_store_host_donor",
]

#: Set to 1 to ignore ``store_host_donor`` (kill switch for A/B runs).
DISABLE_ENV = "TRTLLM_DISABLE_STORE_HOST_DONOR"
#: Where launchers that run a Mooncake master (srt-slurm, Dynamo recipes)
#: publish its ``host:port`` to the workers.
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
NODE_LOCK_PATH = "/dev/shm/trtllm-store-host-donor.lock"  # nosec B108 - node-local lock
_DEFAULT_LOCK = object()
#: Seconds `close` waits for a registration still in flight.
JOIN_TIMEOUT_S = 900.0

_MPOL_DEFAULT = 0
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

    Memory-only nodes (GPU memory exposed as NUMA nodes on Grace/Vera systems,
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
        wanted = {h for h in head_hosts} | {h.split(".", 1)[0] for h in head_hosts}
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
    kernel and shared memory (``memory.stat`` counts shmem inside ``file``)."""
    with open(os.path.join(directory, "memory.current")) as handle:
        current = int(handle.read().strip())
    with open(os.path.join(directory, "memory.stat")) as handle:
        stat = dict(line.split() for line in handle if len(line.split()) == 2)
    if "file" not in stat or "shmem" not in stat:
        raise ValueError("memory.stat lacks file/shmem")
    return current - int(stat["file"]) + int(stat["shmem"])


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
    unreadable.
    """
    out: Dict[str, Optional[int]] = {"limit": None, "nonfile": None, "path": None}
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


def _default_job_memory() -> Dict[str, Optional[int]]:
    return cgroup_memory()


# --------------------------------------------------------------------------
# Measured late donation
# --------------------------------------------------------------------------

#: 1 selects the measured late donation on any config; 0 turns measured lending off.
AUTO_ENV = "TRTLLM_STORE_HOST_DONOR_AUTO"
#: Fixed late size per node in GiB; replaces every auto mode.
LATE_GIB_ENV = "TRTLLM_STORE_HOST_DONOR_LATE_GIB"
BUFFER_ENV = "TRTLLM_STORE_HOST_DONOR_BUFFER_GIB"
HEAD_BUFFER_ENV = "TRTLLM_STORE_HOST_DONOR_HEAD_BUFFER_GIB"
MAX_LENT_ENV = "TRTLLM_STORE_HOST_DONOR_MAX_LENT_GIB"
TOPUP_ENV = "TRTLLM_STORE_HOST_DONOR_TOPUP_S"
BUDGET_ENV = "TRTLLM_STORE_HOST_DONOR_BUDGET_PER_GPU_GIB"
NODE_GPUS_ENV = "TRTLLM_STORE_HOST_DONOR_NODE_GPUS"
HORIZON_ENV = "TRTLLM_STORE_HOST_DONOR_HORIZON_S"
SLOPE_WINDOW_ENV = "TRTLLM_STORE_HOST_DONOR_SLOPE_WINDOW_S"
SAMPLE_ENV = "TRTLLM_STORE_HOST_DONOR_SAMPLE_S"
MIN_SEGMENT_ENV = "TRTLLM_STORE_HOST_DONOR_MIN_SEGMENT_GIB"
WAIT_TRAFFIC_ENV = "TRTLLM_STORE_HOST_DONOR_WAIT_TRAFFIC"
FIRST_NEEDS_SLOPE_ENV = "TRTLLM_STORE_HOST_DONOR_FIRST_NEEDS_SLOPE"
HEAD_HOSTS_ENV = "TRTLLM_STORE_HOST_DONOR_HEAD_HOSTS"
#: Node-local record of the GPUs and lent bytes of every measured donor on the node.
REGISTRY_PATH = "/dev/shm/trtllm-store-host-donor-registry.json"  # nosec B108 - node-local
#: Seconds between checks for a context server's first request.
TRAFFIC_POLL_S = 5.0


@dataclass(frozen=True)
class MeasuredAuto:
    """Parameters of the measured late donation (byte and second values)."""

    #: Kept free per used GPU; never lent.
    buffer_per_gpu: int = 36 * _GIB
    #: The same on the head node (frontend, client and pool master).
    head_buffer_per_gpu: int = 64 * _GIB
    #: Most this node lends through the measured mode, all workers together.
    max_lent_per_node: Optional[int] = None
    #: Seconds between lending passes.
    topup_seconds: float = 300.0
    #: Seconds between usage samples for the growth slope.
    sample_seconds: float = 30.0
    #: The slope is fit over the samples of this many last seconds.
    slope_window_seconds: float = 900.0
    #: A slope needs samples spanning at least this long.
    min_slope_span_seconds: float = 240.0
    #: Seconds left assumed when the job's end time is unknown.
    horizon_seconds: float = 7200.0
    #: A pass lends nothing smaller than this.
    min_segment: int = 2 * _GIB
    #: Each NUMA node's segment is a multiple of this.
    granularity: int = _GIB
    #: Optional cap on the per-GPU budget (e.g. the X/4 rule's exact value).
    budget_per_gpu: Optional[int] = None
    #: GPUs on the node; detected from the driver when None.
    node_gpus: Optional[int] = None
    #: Wait for the first request before lending; None lets the executor decide
    #: (context servers wait, generation servers do not).
    wait_traffic: Optional[bool] = None
    #: Lend anyway when no request arrived within this many seconds.
    traffic_timeout_seconds: float = 1800.0
    #: The first lend also waits for a growth slope (the guard holds from the start).
    first_lend_needs_slope: bool = True
    head_hosts: Optional[Sequence[str]] = None


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


def measured_auto_from_env(
    environ: Mapping[str, str], head_hosts: Optional[Sequence[str]] = None
) -> MeasuredAuto:
    """`MeasuredAuto` defaults with the environment's overrides applied."""
    base = MeasuredAuto()
    values: Dict[str, Any] = {}
    for name, key in ((BUFFER_ENV, "buffer_per_gpu"), (HEAD_BUFFER_ENV, "head_buffer_per_gpu")):
        gib = _env_number(environ, name)
        if gib is not None:
            values[key] = int(gib * _GIB)
    for name, key in (
        (MAX_LENT_ENV, "max_lent_per_node"),
        (BUDGET_ENV, "budget_per_gpu"),
        (MIN_SEGMENT_ENV, "min_segment"),
    ):
        gib = _env_number(environ, name, positive=key == "budget_per_gpu")
        if gib is not None:
            values[key] = int(gib * _GIB)
    for name, key in (
        (TOPUP_ENV, "topup_seconds"),
        (SAMPLE_ENV, "sample_seconds"),
        (SLOPE_WINDOW_ENV, "slope_window_seconds"),
        (HORIZON_ENV, "horizon_seconds"),
    ):
        seconds = _env_number(environ, name, positive=True)
        if seconds is not None:
            values[key] = seconds
    gpus = _env_number(environ, NODE_GPUS_ENV, positive=True)
    if gpus is not None:
        if gpus != int(gpus):
            raise ValueError(f"{LOG_TAG}: {NODE_GPUS_ENV} must be a whole number")
        values["node_gpus"] = int(gpus)
    needs_slope = _env_flag(environ, FIRST_NEEDS_SLOPE_ENV)
    if needs_slope is not None:
        values["first_lend_needs_slope"] = needs_slope
    wait = _env_flag(environ, WAIT_TRAFFIC_ENV)
    if wait is not None:
        values["wait_traffic"] = wait
    hosts = environ.get(HEAD_HOSTS_ENV)
    if hosts and hosts.strip():
        values["head_hosts"] = tuple(h.strip() for h in hosts.split(",") if h.strip())
    elif head_hosts:
        values["head_hosts"] = tuple(head_hosts)
    auto = MeasuredAuto(**{**base.__dict__, **values})
    if auto.sample_seconds * 2 > auto.slope_window_seconds:
        raise ValueError(
            f"{LOG_TAG}: {SAMPLE_ENV}={auto.sample_seconds:g} leaves fewer than 3 samples in the "
            f"{auto.slope_window_seconds:g} s slope window; no lend could ever pass the growth guard"
        )
    if auto.min_slope_span_seconds > auto.slope_window_seconds:
        auto = MeasuredAuto(
            **{**auto.__dict__, "min_slope_span_seconds": auto.slope_window_seconds}
        )
    return auto


def node_memtotal(path: str = "/proc/meminfo") -> Optional[int]:
    """The node's MemTotal in bytes, or None when unreadable."""
    info = _read_meminfo(path)
    return info.get("MemTotal")


def _read_meminfo(path: str) -> Dict[str, int]:
    values = {}
    try:
        with open(path) as handle:
            for line in handle:
                # "MemTotal:       1051992596 kB"
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    except (OSError, ValueError):
        return {}
    return values


def node_used_nonfile(path: str = "/proc/meminfo") -> Optional[int]:
    """Node memory in use that reclaim cannot free: total minus free, buffers
    and page cache, with shared memory (counted in the cache) added back."""
    info = _read_meminfo(path)
    if not all(k in info for k in ("MemTotal", "MemFree", "Cached", "Shmem")):
        return None
    return max(
        0,
        info["MemTotal"]
        - info["MemFree"]
        - info.get("Buffers", 0)
        - info["Cached"]
        + info["Shmem"],
    )


def node_gpu_count(
    proc_dir: str = "/proc/driver/nvidia/gpus", dev_dir: str = "/dev"
) -> Optional[int]:
    """GPUs on the node, independent of CUDA_VISIBLE_DEVICES; None when unknown."""
    try:
        count = len([name for name in os.listdir(proc_dir) if not name.startswith(".")])
    except OSError:
        count = 0
    if count:
        return count
    try:
        count = len([name for name in os.listdir(dev_dir) if re.fullmatch(r"nvidia\d+", name)])
    except OSError:
        count = 0
    return count or None


def growth_slope(
    samples: Sequence[tuple], now: float, window: float, min_span: float
) -> Optional[float]:
    """Least-squares slope (bytes per second) of ``(time, bytes)`` samples taken
    in the last ``window`` seconds, or None when they span less than ``min_span``."""
    recent = [(t, v) for t, v in samples if now - window <= t <= now]
    if len(recent) < 2 or recent[-1][0] - recent[0][0] < min_span:
        return None
    n = len(recent)
    mean_t = sum(t for t, _ in recent) / n
    mean_v = sum(v for _, v in recent) / n
    var = sum((t - mean_t) ** 2 for t, _ in recent)
    if var <= 0:
        return None
    return sum((t - mean_t) * (v - mean_v) for t, v in recent) / var


def _scontrol_end_time(job_id: str) -> Optional[float]:
    if not shutil.which("scontrol"):
        return None
    try:
        out = subprocess.run(  # nosec B607 - scontrol from the Slurm PATH
            ["scontrol", "show", "job", "-o", job_id],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\bEndTime=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", out)
    if match is None:
        return None
    return time.mktime(time.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S"))


def job_end_epoch(
    environ: Mapping[str, str], scontrol: Callable[[str], Optional[float]] = _scontrol_end_time
) -> tuple:
    """(epoch the job ends, source), or (None, "unknown")."""
    text = environ.get("SLURM_JOB_END_TIME")
    if text and text.strip().isdigit():
        return float(text.strip()), "SLURM_JOB_END_TIME"
    job = environ.get("SLURM_JOB_ID")
    if job and job.strip().isdigit():
        end = scontrol(job.strip())
        if end is not None:
            return end, "scontrol"
    return None, "unknown"


def measured_lend(
    *,
    node_budget: int,
    node_buffer: int,
    usage: int,
    lent_node: int,
    lent_unit: int,
    weight: float,
    active_gpus: float,
    max_lent_per_node: Optional[int],
    slope: Optional[float],
    seconds_left: float,
    require_slope: bool,
    granularity: int,
    min_segment: int,
) -> tuple:
    """Bytes one donor may lend now, why, and the terms (pure rule).

    ``node_room = budget - usage - buffer`` is what the node can take; the
    donor's own room is its GPU share of the node's lendable memory
    (``budget - buffer - usage without measured lends``) minus what it lent.
    Nothing is lent while the growth slope, carried to the job's end, would use
    up the buffer.
    """
    base_usage = usage - lent_node
    node_room = node_budget - node_buffer - usage
    share = weight / active_gpus if active_gpus > 0 else 0.0
    unit_room = int((node_budget - node_buffer - base_usage) * share) - lent_unit
    rooms = [node_room, unit_room]
    cap_room = None
    if max_lent_per_node is not None:
        cap_room = int(max_lent_per_node * share) - lent_unit
        rooms.append(cap_room)
    growth = None if slope is None else max(0.0, slope) * max(0.0, seconds_left)
    terms = {
        "node_room": node_room,
        "unit_room": unit_room,
        "cap_room": cap_room,
        "projected_growth": None if growth is None else int(growth),
        "usage_without_auto": base_usage,
    }
    amount = max(0, min(rooms)) // granularity * granularity
    if slope is None and require_slope:
        return 0, "slope_unknown", terms
    if growth is not None and growth >= node_buffer:
        return 0, "growth_guard", terms
    if cap_room is not None and cap_room <= 0:
        return 0, "max_lent", terms
    if amount < max(min_segment, granularity):
        return 0, "no_room", terms
    return amount, "room", terms


class NodeRegistry:
    """GPUs and measured lends of the donors on this node, in one JSON file.

    Writes happen under the node lock; the file is replaced atomically, so it
    can be read without the lock. Entries of other jobs, and of processes in
    this PID namespace that no longer exist, are dropped.
    """

    def __init__(self, path: str, job: str) -> None:
        self.path = path
        self.job = job

    @staticmethod
    def _pid_namespace() -> str:
        try:
            return os.readlink("/proc/self/ns/pid")
        except OSError:
            return ""

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.path) as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return {}
        units = data.get("units") if isinstance(data, dict) else None
        return units if isinstance(units, dict) else {}

    def units(self) -> Dict[str, Dict[str, Any]]:
        namespace = self._pid_namespace()
        live = {}
        for key, unit in self._load().items():
            if not isinstance(unit, dict) or str(unit.get("job")) != self.job:
                continue
            if unit.get("ns") == namespace and isinstance(unit.get("pid"), int):
                try:
                    os.kill(unit["pid"], 0)
                except ProcessLookupError:
                    continue
                except PermissionError:
                    pass
            live[key] = unit
        return live

    def put(self, key: str, unit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        units = self.units()
        units[key] = {**unit, "job": self.job, "pid": os.getpid(), "ns": self._pid_namespace()}
        self._write(units)
        return units

    def remove(self, key: str) -> Dict[str, Dict[str, Any]]:
        units = self.units()
        if units.pop(key, None) is not None:
            self._write(units)
        return units

    def _write(self, units: Dict[str, Dict[str, Any]]) -> None:
        directory = os.path.dirname(self.path) or "."
        handle = tempfile.NamedTemporaryFile(
            "w", dir=directory, prefix=".shd-registry-", delete=False
        )
        try:
            with handle:
                json.dump({"units": units}, handle, sort_keys=True)
            os.replace(handle.name, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(handle.name)
            raise


@dataclass
class MeasuredProbes:
    """Where the measured mode reads the node from (files are injectable)."""

    meminfo_path: str = "/proc/meminfo"
    gpu_proc_dir: str = "/proc/driver/nvidia/gpus"
    dev_dir: str = "/dev"
    registry_path: Optional[str] = REGISTRY_PATH
    environ: Optional[Mapping[str, str]] = None
    clock: Callable[[], float] = time.time
    scontrol: Callable[[str], Optional[float]] = _scontrol_end_time


def resolve_master(configured: Optional[str], environ: Optional[Dict[str, str]] = None) -> str:
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

    ``FilePages`` counts tmpfs / shared memory, which reclaim cannot free, so
    ``Shmem`` is taken out of the page cache.
    """
    if "MemFree" not in info:
        return None
    cache = max(0, info.get("FilePages", 0) - info.get("Shmem", 0))
    return info["MemFree"] + cache + info.get("SReclaimable", 0)


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
        measured: Optional[MeasuredAuto] = None,
        probes: Optional[MeasuredProbes] = None,
    ) -> None:
        if late_mode not in ("auto", "same_client", "new_client"):
            raise ValueError(f"{LOG_TAG}: unknown late_mode {late_mode!r}")
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
        if measured is not None and late_auto is not None:
            raise ValueError(f"{LOG_TAG}: measured and late_auto sizing exclude each other")
        if measured is not None and not self.numa_nodes:
            raise ValueError(f"{LOG_TAG}: measured sizing needs the NUMA node list")
        self.measured = measured
        self._probes = probes if probes is not None else MeasuredProbes()
        #: Set by the executor: True once the server received its first request.
        self.traffic_probe: Optional[Callable[[], bool]] = None
        #: Set by the executor: wait for the first request before lending.
        self.wait_traffic = False
        self._unit_key = f"{socket.gethostname()}:{os.getpid()}:{rank}:{id(self):x}"
        self._auto_lent = 0
        self._auto_passes = 0
        self._auto_samples: deque = deque()
        self._auto_head: Optional[tuple] = None
        self._auto_job_end: Optional[tuple] = None
        self._auto_started = False
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

    def _check_headroom(self, node: int, size: int, client: bool = False) -> Dict[str, int]:
        info = self._meminfo(node)
        obtainable = node_obtainable_bytes(info)
        # Opening a client also allocates its transfer buffer under the binding.
        need = size + (self.local_buffer_size if client else 0)
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
        if limit is not None and (nonfile is None or need > limit - nonfile):
            held = "unknown" if nonfile is None else f"{nonfile / _GIB:.1f} GiB"
            raise RuntimeError(
                f"{LOG_TAG}: {need / _GIB:.1f} GiB does not fit the memory cgroup "
                f"{job.get('path')} ({limit / _GIB:.1f} GiB limit, {held} held); refusing an "
                "allocation that could be OOM-killed"
            )
        return info

    def _log_registered(self, reg: _Registration, seconds: float, before: Dict[str, int]) -> None:
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
        node = reg.share.numa_node
        before = self._check_headroom(node, reg.bytes, client=True)
        hostname = self._hostname()
        started = time.monotonic()

        def work():
            store = self._store_factory()
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
            return store

        reg.store = self._run_bound(node, work, f"store-host-donor-{reg.phase}-n{node}")
        get_hostname = getattr(reg.store, "get_hostname", None)
        reg.segment_name = str(get_hostname()) if callable(get_hostname) else hostname
        with self._lock:
            self._registrations.append(reg)
        self._log_registered(reg, time.monotonic() - started, before)

    def _mount_on(self, base: _Registration, reg: _Registration, fit: bool = False) -> None:
        with self._node_serialized():
            if fit and not self._fit_late_auto(reg, client=False):
                return
            self._mount_on_locked(base, reg)

    def _mount_on_locked(self, base: _Registration, reg: _Registration) -> None:
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
        # when the startup share is zero.
        # Under late_auto the late size is unknown until after warmup, but its
        # upper bound tells whether any late share can follow.
        late_pending = self.late_mode != "new_client" and (
            self.measured is not None
            or (
                self.late_auto is not None
                and self.late_auto_record is None
                and self._late_auto_upper_bound() > 0
            )
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
        if self.late_mode == "new_client":
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
        if self.measured is not None:
            self._auto_register_unit()

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
            else:
                self._open_client(reg, fit=fit)
        self._late_done = True

    @property
    def has_late(self) -> bool:
        if self.measured is not None:
            return bool(self.shares)
        if self.late_auto is not None and self.late_auto_record is None:
            return True
        return any(share.late_bytes > 0 for share in self.shares)

    def register_late(self) -> None:
        """Register the late portion now (after warmup, before serving).

        The measured mode starts its background lending instead.
        """
        self.wait_startup()
        if self.measured is not None:
            self._auto_start()
            return
        self._register_late_now()

    def schedule_late(self) -> None:
        """Register the late portion ``late_delay_seconds`` from now, in the background."""
        self.wait_startup()
        if self.measured is not None:
            self._auto_start()
            return
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

    # -- measured late donation ----------------------------------------------

    def _auto_environ(self) -> Mapping[str, str]:
        environ = self._probes.environ
        return os.environ if environ is None else environ

    def _auto_registry(self) -> Optional[NodeRegistry]:
        path = self._probes.registry_path
        if not path:
            return None
        job = self._auto_environ().get("SLURM_JOB_ID") or "nojob"
        return NodeRegistry(path, job)

    def _auto_weight(self) -> float:
        """GPUs this donor stands for: its worker's GPUs on the node, split over
        the ranks that own a NUMA node."""
        return self.local_size * len(self.shares) / len(self.numa_nodes)

    def _auto_put(self) -> Dict[str, Dict[str, Any]]:
        registry = self._auto_registry()
        if registry is None:
            raise RuntimeError("no node registry configured")
        unit = {
            "weight": self._auto_weight(),
            "lent": self._auto_lent,
            "rank": self.rank,
            "host": socket.gethostname(),
        }
        try:
            return registry.put(self._unit_key, unit)
        except OSError:
            fallback = os.path.join(tempfile.gettempdir(), os.path.basename(registry.path))
            if fallback == registry.path:
                raise
            self._probes.registry_path = fallback
            return NodeRegistry(fallback, registry.job).put(self._unit_key, unit)

    def _auto_register_unit(self) -> None:
        """Count this donor's GPUs in the node registry (at startup, so that
        co-located workers see each other before anyone lends)."""
        try:
            with self._node_serialized():
                self._auto_put()
        except (OSError, RuntimeError, ValueError) as error:
            self._probes.registry_path = None
            logger.error(f"{LOG_TAG}_AUTO_FAILED registry rank={self.rank}: {error}")

    def _auto_head_role(self) -> tuple:
        if self._auto_head is None:
            self._auto_head = self._head_role(self.master, self.measured.head_hosts)
        return self._auto_head

    def _auto_seconds_left(self, now: float) -> tuple:
        if self._auto_job_end is None:
            end, source = job_end_epoch(self._auto_environ(), self._probes.scontrol)
            self._auto_job_end = (end, source)
        end, source = self._auto_job_end
        if end is None:
            return float(self.measured.horizon_seconds), "horizon"
        return max(0.0, end - now), source

    def _auto_node(self) -> Dict[str, Any]:
        """Budget inputs of this node now. Call with the node lock held."""
        m = self.measured
        probes = self._probes
        out: Dict[str, Any] = {}
        registry = self._auto_registry()
        units = registry.units() if registry is not None else {}
        if self._unit_key not in units and registry is not None:
            units = self._auto_put()
        out["active_gpus"] = sum(float(u.get("weight", 0)) for u in units.values())
        out["donors_on_node"] = len(units)
        out["lent_node"] = sum(int(u.get("lent", 0)) for u in units.values())
        out["memtotal"] = node_memtotal(probes.meminfo_path)
        out["node_gpus"] = (
            m.node_gpus
            if m.node_gpus is not None
            else node_gpu_count(probes.gpu_proc_dir, probes.dev_dir)
        )
        job = self._job_memory()
        out["cgroup_limit"], out["cgroup_path"] = job.get("limit"), job.get("path")
        if job.get("limit") is not None:
            out["usage"], out["usage_source"] = job.get("nonfile"), "job_cgroup_nonfile"
        else:
            out["usage"], out["usage_source"] = (
                node_used_nonfile(probes.meminfo_path),
                "node_meminfo_nonfile",
            )
        return out

    def _auto_sample(self) -> None:
        with self._node_serialized():
            node = self._auto_node()
        if node["usage"] is None:
            return
        self._auto_samples.append((self._probes.clock(), node["usage"] - node["lent_node"]))
        horizon = self._probes.clock() - self.measured.slope_window_seconds
        while self._auto_samples and self._auto_samples[0][0] < horizon:
            self._auto_samples.popleft()

    def _auto_pass(self) -> Dict[str, Any]:
        """One lending pass: measure, decide, lend, log. Returns the logged record."""
        m = self.measured
        now = self._probes.clock()
        head, basis = self._auto_head_role()
        buffer_per_gpu = m.head_buffer_per_gpu if head else m.buffer_per_gpu
        seconds_left, end_source = self._auto_seconds_left(now)
        slope = growth_slope(
            list(self._auto_samples), now, m.slope_window_seconds, m.min_slope_span_seconds
        )
        first = self._auto_passes == 0
        record: Dict[str, Any] = {
            "pass": self._auto_passes,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "host": socket.gethostname(),
            "head": head,
            "head_basis": basis,
            "buffer_per_gpu": buffer_per_gpu,
            "seconds_left": round(seconds_left, 1),
            "end_source": end_source,
            "slope_gib_per_min": None if slope is None else round(slope * 60 / _GIB, 4),
            "weight": self._auto_weight(),
            "numa_nodes": [s.numa_node for s in self.shares],
        }
        amount, reason = 0, "unknown"
        with self._node_serialized():
            if self._probes.registry_path is None:
                reason = "registry_unavailable"
                node: Dict[str, Any] = {}
            else:
                node = self._auto_node()
            record.update(node)
            if node:
                missing = [
                    k
                    for k in ("memtotal", "node_gpus", "usage")
                    if node.get(k) is None or node.get(k) == 0
                ]
                if missing or node["active_gpus"] <= 0:
                    reason = "unknown_" + "_".join(missing or ["active_gpus"])
                else:
                    active = node["active_gpus"]
                    per_gpu = node["memtotal"] / node["node_gpus"]
                    if node["cgroup_limit"] is not None:
                        per_gpu = min(per_gpu, node["cgroup_limit"] / active)
                    if m.budget_per_gpu is not None:
                        per_gpu = min(per_gpu, m.budget_per_gpu)
                    budget = int(per_gpu * active)
                    buffer = int(buffer_per_gpu * active)
                    amount, reason, terms = measured_lend(
                        node_budget=budget,
                        node_buffer=buffer,
                        usage=node["usage"],
                        lent_node=node["lent_node"],
                        lent_unit=self._auto_lent,
                        weight=self._auto_weight(),
                        active_gpus=active,
                        max_lent_per_node=m.max_lent_per_node,
                        slope=slope,
                        seconds_left=seconds_left,
                        require_slope=m.first_lend_needs_slope or not first,
                        granularity=len(self.shares) * m.granularity,
                        min_segment=m.min_segment,
                    )
                    record.update(terms)
                    record.update(budget=budget, budget_per_gpu=int(per_gpu), buffer=buffer)
                    if amount > 0:
                        amount, segments = self._auto_lend_locked(amount)
                        record["segments"] = segments
                        if amount == 0:
                            reason = "numa_short"
        self._auto_passes += 1
        if first and not m.first_lend_needs_slope:
            # The slope is fit over serving from here on, not over warmup.
            self._auto_samples.clear()
        record.update(
            reason=reason,
            amount=amount,
            amount_gib=round(amount / _GIB, 3),
            lent_unit=self._auto_lent,
            lent_unit_gib=round(self._auto_lent / _GIB, 3),
        )
        usage, active = record.get("usage"), record.get("active_gpus")
        if usage is not None and active:
            record["usage_gib"] = round(usage / _GIB, 3)
            record["per_gpu_after_gib"] = round((usage + amount) / active / _GIB, 3)
        tag = "AUTO_LEND" if amount > 0 else "AUTO_HOLD"
        logger.info(f"{LOG_TAG}_{tag} {json.dumps(record, sort_keys=True, default=str)}")
        return record

    def _auto_lend_locked(self, amount: int) -> tuple:
        """Lend ``amount`` over the owned NUMA nodes, each cut to what it can
        back. Returns (bytes lent, segment list). Call with the node lock held."""
        m = self.measured
        count = len(self.shares)
        each = amount // count // m.granularity * m.granularity
        segments = []
        lent = 0
        for share in self.shares:
            node = share.numa_node
            base = self._late_base(node)
            # A segment that needs its own client also brings that client's
            # transfer buffer: both come out of the decided amount.
            size = each if base is not None else each - self.local_buffer_size
            size = max(0, size) // m.granularity * m.granularity
            obtainable = node_obtainable_bytes(self._meminfo(node))
            if obtainable is not None:
                spare = obtainable - NODE_HEADROOM_BYTES
                if base is None:
                    spare -= self.local_buffer_size
                size = min(size, max(0, spare) // m.granularity * m.granularity)
            if size < m.granularity:
                segments.append({"numa_node": node, "bytes": 0, "obtainable": obtainable})
                continue
            reg = _Registration(f"late_auto{self._auto_passes}", DonorShare(node, 0, size), size)
            try:
                if base is not None:
                    self._mount_on_locked(base, reg)
                else:
                    self._open_client_locked(reg)
            finally:
                registered = reg in self._registrations
                if registered:
                    self._auto_lent += size
                    lent += size
                    self._auto_put()
            segments.append({"numa_node": node, "bytes": size, "segment": reg.segment_name})
        return lent, segments

    def _auto_wait_traffic(self) -> bool:
        """Wait for the first request when asked to. False when closing."""
        m = self.measured
        wait = m.wait_traffic if m.wait_traffic is not None else self.wait_traffic
        if not wait:
            return True
        probe = self.traffic_probe
        if probe is None:
            logger.warning(f"{LOG_TAG}_AUTO: no traffic probe; lending after warmup")
            return True
        started = self._probes.clock()
        while not probe():
            if self._probes.clock() - started >= m.traffic_timeout_seconds:
                logger.warning(
                    f"{LOG_TAG}_AUTO: no request within {m.traffic_timeout_seconds:.0f} s; "
                    "lending anyway"
                )
                return True
            if self._stop.wait(TRAFFIC_POLL_S):
                return False
        logger.info(
            f"{LOG_TAG}_AUTO first request after {self._probes.clock() - started:.1f} s "
            f"rank={self.rank}"
        )
        return True

    def _auto_loop(self) -> None:
        m = self.measured
        try:
            if self._stop.wait(self.late_delay_seconds):
                return
            if not self._auto_wait_traffic():
                return
            clock = self._probes.clock
            next_pass = next_sample = clock()
            while not self._stop.is_set():
                now = clock()
                if now >= next_pass:
                    self._auto_pass()
                    next_pass = now + m.topup_seconds
                    next_sample = clock()
                if now >= next_sample:
                    self._auto_sample()
                    next_sample = now + m.sample_seconds
                if self._stop.wait(max(0.0, min(next_pass, next_sample) - clock())):
                    return
        except BaseException as error:  # noqa: BLE001 - nobody to raise to
            self._late_error = error
            logger.error(f"{LOG_TAG}_FAILED phase=late_auto rank={self.rank}: {error}")

    def _auto_start(self) -> None:
        """Start the measured lending thread once (owners of a NUMA share only)."""
        if self._auto_started or not self.shares:
            return
        self._auto_started = True
        m = self.measured
        logger.info(
            f"{LOG_TAG}_AUTO_START "
            + json.dumps(
                {
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "local_size": self.local_size,
                    "weight": self._auto_weight(),
                    "late_delay_seconds": self.late_delay_seconds,
                    "wait_traffic": m.wait_traffic
                    if m.wait_traffic is not None
                    else self.wait_traffic,
                    "params": {k: v for k, v in m.__dict__.items() if k != "head_hosts"}
                    | {"head_hosts": list(m.head_hosts) if m.head_hosts else None},
                },
                sort_keys=True,
            )
        )
        self._late_thread = threading.Thread(
            target=self._auto_loop, name="store-host-donor-late-auto", daemon=True
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
        for thread in (self._late_thread, self._startup_thread):
            if thread is not None:
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
        if ok and self.measured is not None:
            self._auto_unregister()
        return ok

    def _auto_unregister(self) -> None:
        """Drop this donor from the node registry once nothing it lent is left."""
        registry = self._auto_registry()
        if registry is None:
            return
        try:
            with self._node_serialized():
                registry.remove(self._unit_key)
        except (OSError, RuntimeError) as error:
            logger.warning(f"{LOG_TAG}_AUTO: cannot leave the node registry: {error}")

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


def _late_auto_from_config(config: Any, environ: Dict[str, str]) -> LateAuto:
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
    environ: Optional[Dict[str, str]] = None,
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
    """(late_auto, measured, fixed late bytes per node, source) of a config.

    Precedence: a fixed size from the environment, then
    ``TRTLLM_STORE_HOST_DONOR_AUTO`` (1 selects the measured mode, 0 turns
    it off), then the config: ``auto`` with ``late_auto`` is the planned
    budget, ``auto`` alone the measured mode, anything else a fixed size.
    """
    late_auto_config = getattr(config, "late_auto", None)
    head_hosts = late_auto_config.head_hosts if late_auto_config is not None else None
    fixed = _env_number(environ, LATE_GIB_ENV)
    if fixed is not None:
        return None, None, int(fixed * _GIB), LATE_GIB_ENV
    switch = _env_flag(environ, AUTO_ENV)
    if switch is True:
        return None, measured_auto_from_env(environ, head_hosts), 0, f"{AUTO_ENV}=1"
    if config.late_size_per_node == "auto":
        if late_auto_config is not None:
            return _late_auto_from_config(late_auto_config, environ), None, 0, "late_auto"
        if switch is False:
            return None, None, 0, f"{AUTO_ENV}=0"
        return None, measured_auto_from_env(environ), 0, "measured"
    return None, None, _parse(config.late_size_per_node), "fixed"


def _build_donor(config, rank, local_rank, local_size, environ, numa_nodes, donor_kwargs):
    """Plan, construct and log one donor (the caller holds the process slot)."""
    nodes = (
        list(config.numa_nodes)
        if config.numa_nodes
        else (list(numa_nodes) if numa_nodes is not None else cpu_numa_nodes())
    )
    startup = _parse(config.startup_size_per_node)
    auto, measured, late, late_source = _late_sizing(config, environ)
    shares = plan_shares(startup, late, nodes, local_rank, local_size)
    master = resolve_master(config.master, environ)
    device_name = config.device_name
    if device_name is None:
        device_name = _default_device_name(config.protocol) if shares else ""
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
        measured=measured,
        **donor_kwargs,
    )
    plan = {
        "rank": rank,
        "local_rank": local_rank,
        "local_size": local_size,
        "numa_nodes": nodes,
        "startup_per_node": startup,
        "late_per_node": "measured"
        if measured is not None
        else ("auto" if auto is not None else late),
        "late_source": late_source,
        "measured": None
        if measured is None
        else {k: v for k, v in measured.__dict__.items() if k != "head_hosts"}
        | {"head_hosts": list(measured.head_hosts) if measured.head_hosts else None},
        "late_auto": None
        if auto is None
        else {k: v for k, v in auto.__dict__.items() if k != "head_hosts"}
        | {"head_hosts": list(auto.head_hosts) if auto.head_hosts else None},
        "owned": [share.__dict__ for share in shares],
        "master": master,
        "late_mode": config.late_mode,
        "late_delay_seconds": config.late_delay_seconds,
    }
    logger.info(f"{LOG_TAG}_PLAN {json.dumps(plan, sort_keys=True)}")
    return donor


atexit.register(close_active_donor)
