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
"""Bring a server onto a Mooncake pool that something else already owns.

The connector needs two things that are not the engine's to produce: a reachable
`mooncake_master`, and a JSON client config named by `MOONCAKE_CONFIG_PATH` that
points every worker at it.

The master is infrastructure, not engine state. One runs per deployment, with a
lifetime of its own, and publishes a **manifest** describing the pool it owns:
its address and the transport every participant has to agree on. `running_master`
is that process, reachable as `trtllm-serve mooncake_master`.

`provision_pool` is the other half, inside each serving process. It reads the
manifest, adds what is this server's alone, being the traffic it drives, the
memory each of its ranks lends and the RDMA devices this node happens to have,
and renders the client config, exporting `MOONCAKE_CONFIG_PATH`, which reaches
the ranks because the LLM constructor spawns them from this process.

Splitting it this way keeps the pool's own settings from drifting: they are
stated once, by the process that owns the pool, rather than restated in every
worker config.
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess  # nosec B404
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

from ..registry import uses_connector
from .config import (
    CLIENT_CONFIG_NAME,
    CONFIG_PATH_ENV,
    DEFAULT_LOCAL_BUFFER_SIZE,
    DEFAULT_METADATA_SERVER,
    DEFAULT_NAMESPACE,
    RUN_DIR_OWNER_NAME,
    StoreRole,
    parse_size,
)

__all__ = [
    "POOL_MANIFEST_NAME",
    "PoolManifest",
    "claim_run_dir",
    "local_address",
    "maybe_provision_pool",
    "provision_pool",
    "resolve_device_name",
    "resolve_pool",
    "running_master",
    "wait_for_master",
]

DEFAULT_MASTER_BINARY = "mooncake_master"
#: Seconds to wait for a master to publish its manifest and answer. Overridden
#: per server by `mooncake_store.master_timeout` and per master by `--timeout`.
DEFAULT_MASTER_TIMEOUT = 60.0
MASTER_LOG_NAME = "mooncake_master.log"
#: Name the pool manifest always takes in the run directory, so even a run that
#: asked for it nowhere else still records the pool it used.
POOL_MANIFEST_NAME = "pool.json"
#: Prefix that makes `pool` name a manifest file rather than a master address.
POOL_FILE_SCHEME = "file://"
#: Lines of the master's log to quote when startup fails, since its last words
#: (a port in use, a bad flag) are usually the whole diagnosis.
LOG_TAIL_LINES = 20
DEFAULT_MASTER_PORT = 50051
DEFAULT_MASTER_METRICS_PORT = 9004
DEFAULT_MASTER_EVICTION_RATIO = 0.05


def _log_tail(path: str, lines: int = LOG_TAIL_LINES) -> str:
    """The end of the master's log, ready to append to a failure message."""
    try:
        with open(path, errors="replace") as handle:
            tail = handle.read().splitlines()[-lines:]
    except OSError as exc:
        return f" Its log at {path} could not be read: {exc}."
    if not tail:
        return (
            f" Its log at {path} is empty, which usually means it failed "
            "before glog opened; check that the binary runs at all."
        )
    quoted = "\n  ".join(tail)
    return f" The last {len(tail)} lines of {path}:\n  {quoted}"


def local_address() -> str:
    """The address this host is known by inside the pool.

    Uses the same derivation as the connector worker's own hostname, so the
    master and the segments registering with it agree on which host they are on.
    """
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _split_address(address: str) -> Optional[Tuple[str, int]]:
    """Split `host:port`, or return `None` if it is not in that form."""
    host, separator, port = address.rpartition(":")
    if not separator or not port.isdigit():
        return None
    return host.strip("[]"), int(port)


@dataclass(frozen=True)
class PoolManifest:
    """What every participant in one pool has to agree on.

    Published by the master that owns the pool and read by each server joining
    it. These are exactly the settings that are properties of the pool rather
    than of any engine in it, which is why they live here: restating them in
    every worker config is how they come to disagree.
    """

    master_server_address: str
    metadata_server: str = DEFAULT_METADATA_SERVER
    protocol: str = "rdma"
    namespace: str = DEFAULT_NAMESPACE
    #: Reported for a reader's benefit rather than used; the master owns them.
    metrics_port: Optional[int] = None
    eviction_ratio: Optional[float] = None
    #: Anything a newer master published that this reader does not know about,
    #: kept so a round trip through here does not quietly drop it.
    extra: Dict[str, Any] = field(default_factory=dict)

    #: Fields spelled out above. Everything else in the file lands in `extra`.
    _KNOWN = (
        "master_server_address",
        "metadata_server",
        "protocol",
        "namespace",
        "metrics_port",
        "eviction_ratio",
    )

    def to_json(self) -> Dict[str, Any]:
        record: Dict[str, Any] = dict(self.extra)
        record.update(
            {
                "master_server_address": self.master_server_address,
                "metadata_server": self.metadata_server,
                "protocol": self.protocol,
                "namespace": self.namespace,
            }
        )
        if self.metrics_port is not None:
            record["metrics_port"] = self.metrics_port
        if self.eviction_ratio is not None:
            record["eviction_ratio"] = self.eviction_ratio
        return record

    @staticmethod
    def from_json(raw: Dict[str, Any], source: str = "<manifest>") -> "PoolManifest":
        address = str(raw.get("master_server_address", "")).strip()
        if not address:
            raise ValueError(
                f"The Mooncake pool manifest at {source} names no "
                "master_server_address, so there is no pool to join. It is "
                "written by 'trtllm-serve mooncake_master --pool_file'."
            )
        return PoolManifest(
            master_server_address=address,
            metadata_server=str(raw.get("metadata_server") or DEFAULT_METADATA_SERVER),
            protocol=str(raw.get("protocol", "rdma")),
            namespace=str(raw.get("namespace", DEFAULT_NAMESPACE)),
            metrics_port=raw.get("metrics_port"),
            eviction_ratio=raw.get("eviction_ratio"),
            extra={k: v for k, v in raw.items() if k not in PoolManifest._KNOWN},
        )

    def describe(self) -> str:
        return (
            f"master={self.master_server_address} protocol={self.protocol} "
            f"metadata={self.metadata_server} namespace={self.namespace}"
        )


def _wait_for_manifest(path: str, timeout: float) -> Dict[str, Any]:
    """Block until `path` holds a readable manifest, then return it.

    The master runs on whichever host its scheduler gave it, which nothing
    knows when the worker configs are written, so the file is the rendezvous:
    the master writes it once it answers, every worker's `pool` names the same
    path, and waiting here doubles as waiting for the master to exist at all.

    A half-written file cannot be observed, since the writer renames into
    place, but an empty or unparsable one is treated as not there yet rather
    than as an error: that is what a reader racing a slow filesystem sees.
    """
    started = time.monotonic()
    deadline = started + timeout
    announced = started
    logger.info(f"mooncake-store: reading the pool manifest from {path}")
    while True:
        raw: Optional[Any] = None
        try:
            with open(path) as handle:
                text = handle.read().strip()
            if text:
                raw = json.loads(text)
        except (FileNotFoundError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict) and raw:
            logger.info(
                f"mooncake-store: {path} describes the pool: {json.dumps(raw, sort_keys=True)}"
            )
            return raw
        now = time.monotonic()
        if now - announced >= 5.0:
            announced = now
            # Waiting on a master in another job step is normal here, so say so
            # rather than letting the wait look like a hang.
            logger.info(
                f"mooncake-store: no pool manifest at {path} yet "
                f"({now - started:.0f}s of {timeout:g}s); waiting for the "
                "master to start and publish it"
            )
        if now >= deadline:
            raise TimeoutError(
                f"No Mooncake pool manifest appeared at {path} within "
                f"{timeout:g}s. Start the pool's master with 'trtllm-serve "
                f"mooncake_master --pool_file {path}', or name a reachable "
                "host:port in pool. Raise mooncake_store.master_timeout if "
                "the master is only slow to start."
            )
        time.sleep(0.5)


def resolve_pool(pool: str, timeout: float = DEFAULT_MASTER_TIMEOUT) -> PoolManifest:
    """The pool `pool` names, waiting for its manifest if it names a file.

    Args:
        pool: Either `file://<path>` naming a manifest, or the bare `host:port`
            of a master. The second form is for joining a master run without
            our CLI; it carries no pool-wide settings, so those take defaults
            and it is then on the deployment to keep them consistent.
        timeout: Seconds to wait for a manifest.

    Returns:
        The pool's manifest.
    """
    if not pool:
        raise ValueError(
            "mooncake_store.pool is required: name the manifest a master "
            "published, as file://<path>, or a master's host:port."
        )
    if not pool.startswith(POOL_FILE_SCHEME):
        logger.info(
            f"mooncake-store: pool={pool!r} is a bare master address, so "
            "pool-wide settings take their defaults. Point pool at a "
            "file://<path> manifest to have the master state them instead."
        )
        return PoolManifest(master_server_address=pool)

    path = pool[len(POOL_FILE_SCHEME) :]
    return PoolManifest.from_json(_wait_for_manifest(path, timeout), path)


def _wait_until_accepting(
    host: str,
    port: int,
    timeout: float,
    process: Optional[subprocess.Popen] = None,
    log_path: Optional[str] = None,
) -> float:
    """Block until the master accepts connections, and say how long it took.

    A worker that opens its store handle before the master is listening fails
    outright, so the ordering has to wait on the port rather than on the
    presence of a process. When the master is ours, its exit is checked first
    each pass, so a master that died is reported as such rather than as a
    timeout.

    The wait is narrated as it happens, since silence here is
    indistinguishable from a hang elsewhere in bringup.
    """
    started = time.monotonic()
    deadline = started + timeout
    announced = started
    while True:
        if process is not None and (code := process.poll()) is not None:
            raise RuntimeError(
                f"mooncake_master exited with code {code} after "
                f"{time.monotonic() - started:.1f}s, before it accepted "
                f"connections on {host}:{port}."
                f"{_log_tail(log_path) if log_path else ''}"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return time.monotonic() - started
        except OSError as exc:
            last_error = exc
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"The Mooncake master at {host}:{port} did not accept "
                f"connections within {timeout:g}s ({last_error}). Raise the "
                "timeout if it is only slow to start."
                f"{_log_tail(log_path) if log_path else ''}"
            )
        if now - announced >= 5.0:
            announced = now
            logger.info(
                f"mooncake-store: still waiting for the master at {host}:{port}"
                f" ({now - started:.0f}s of {timeout:g}s, {last_error})"
            )
        time.sleep(0.5)


#: Where the InfiniBand devices of a host are described.
IB_SYSFS_ROOT = "/sys/class/infiniband"


def _highest_rate_ib_devices(sysfs_root: Optional[str] = None) -> List[str]:
    """The active InfiniBand devices on the compute fabric, fastest first.

    A node's HCAs are not interchangeable. On GB300 six are exposed, of which
    four run at 800Gb/s (two per NUMA node, one per GPU) while the rest share a
    PCI device with an Ethernet port and serve storage or management. Taking
    every device at the highest rate picks the compute fabric on any node type,
    where a hardcoded name would be wrong on the next one.
    """
    sysfs_root = sysfs_root or IB_SYSFS_ROOT
    rated: Dict[str, int] = {}
    try:
        devices = sorted(os.listdir(sysfs_root))
    except OSError:
        return []
    for device in devices:
        port = os.path.join(sysfs_root, device, "ports", "1")

        def attribute(name: str) -> str:
            try:
                with open(os.path.join(port, name)) as handle:
                    return handle.read().strip()
            except OSError:
                return ""

        if attribute("link_layer") != "InfiniBand":
            continue
        if "ACTIVE" not in attribute("state"):
            continue
        # "800 Gb/sec (4X XDR)"
        rate = attribute("rate").split()
        if not rate or not rate[0].isdigit():
            continue
        rated[device] = int(rate[0])

    if not rated:
        return []
    fastest = max(rated.values())
    return [device for device, rate in sorted(rated.items()) if rate == fastest]


def resolve_device_name(protocol: str, configured: str, sysfs_root: Optional[str] = None) -> str:
    """The RDMA devices to transfer over, detected if the config left it open.

    Which HCAs a node has is a property of the node, not of the deployment, so
    requiring it in a config would tie that config to one machine type.
    Detecting it keeps `protocol: rdma` portable; setting `device_name`
    overrides the detection.
    """
    if configured or protocol != "rdma":
        return configured
    detected = _highest_rate_ib_devices(sysfs_root)
    if not detected:
        logger.warning(
            "mooncake-store: protocol is rdma but no active InfiniBand device "
            f"was found under {sysfs_root or IB_SYSFS_ROOT}, so device_name is "
            "left empty for Mooncake's own discovery. Set device_name to "
            "choose explicitly."
        )
        return ""
    joined = ",".join(detected)
    logger.info(
        f"mooncake-store: transferring over the fastest active InfiniBand "
        f"devices on this host: {joined}"
    )
    return joined


def wait_for_master(
    master_address: str, timeout: float = DEFAULT_MASTER_TIMEOUT
) -> Optional[float]:
    """Block until the master at `master_address` accepts connections.

    Reaching a master that is not there otherwise fails deep inside
    `store.setup`, in every rank, after the model has loaded, as a bare status
    code. One socket beforehand turns that into a line naming the address.

    Returns how long it took, or `None` if the address was not in `host:port`
    form and could not be checked.
    """
    endpoint = _split_address(master_address)
    if endpoint is None:
        logger.warning(
            f"mooncake-store: cannot parse master_server_address="
            f"{master_address!r} as host:port, so its reachability is left "
            "for the workers to discover."
        )
        return None
    elapsed = _wait_until_accepting(*endpoint, timeout)
    logger.info(f"mooncake-store: the master at {master_address} answered in {elapsed:.1f}s")
    return elapsed


def _client_config(pool: Any, manifest: PoolManifest, device_name: str) -> Dict[str, Any]:
    """Render the Mooncake client config for this server.

    The schema is vLLM's, so one pool can serve both engines. Pool-wide
    settings come from the manifest; what this server contributes and what it
    does with the pool come from its own config.

    Sizes are written as integers rather than as the suffixed strings a user
    may have typed. `"80GB"` means a power of 1000 to this parser and a power
    of 1024 to vLLM's, so resolving it here is what lets both engines read the
    same file and see the same segment.
    """
    config: Dict[str, Any] = {
        # ---- the pool's, from the manifest ----
        "master_server_address": manifest.master_server_address,
        "metadata_server": manifest.metadata_server,
        "protocol": manifest.protocol,
        "namespace": manifest.namespace if pool.namespace is None else pool.namespace,
        # ---- this node's ----
        "device_name": device_name,
        # ---- this server's ----
        "global_segment_size": parse_size(pool.segment_size, strict_units=True),
        "local_buffer_size": DEFAULT_LOCAL_BUFFER_SIZE,
        "role": StoreRole(pool.role).value,
        "transfer_batch_size": pool.transfer_batch_size,
        "stage_through_host": pool.stage_through_host,
    }
    return config


@dataclass
class LaunchedMaster:
    """A `mooncake_master` owned by this process."""

    process: subprocess.Popen
    address: str
    log_path: str

    def stop(self, timeout: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def _launch_master(
    run_dir: str,
    *,
    rpc_port: int = DEFAULT_MASTER_PORT,
    metrics_port: int = DEFAULT_MASTER_METRICS_PORT,
    eviction_ratio: float = DEFAULT_MASTER_EVICTION_RATIO,
    binary: str = DEFAULT_MASTER_BINARY,
    timeout: float = DEFAULT_MASTER_TIMEOUT,
) -> LaunchedMaster:
    """Start a master on this host and wait for it to answer."""
    resolved = shutil.which(binary)
    if resolved is None:
        raise FileNotFoundError(
            f"{binary!r} is not on PATH, so no Mooncake master can be started "
            "here. It comes from the Mooncake Python client, which "
            "`tensorrt-llm` pulls in as `mooncake-transfer-engine-cuda13`, so "
            "reinstall that package if this environment dropped it. Otherwise "
            "name the binary with --binary, or run the master elsewhere and "
            "point the servers' pool at the manifest it publishes."
        )

    host = local_address()
    log_path = os.path.join(run_dir, MASTER_LOG_NAME)

    # glog writes to files under /tmp unless redirected, so without
    # GLOG_logtostderr the log opened below stays empty. GLOG_v=1 adds the
    # per-RPC lines showing segments registering and keys moving, which is the
    # only view of the pool's own side of the conversation short of scraping
    # the metrics port, and is what the run's capacity report joins against.
    env = dict(os.environ, GLOG_logtostderr="1")
    env.setdefault("GLOG_v", "1")
    command = [
        resolved,
        f"--rpc_port={rpc_port}",
        f"--metrics_port={metrics_port}",
        f"--eviction_ratio={eviction_ratio}",
    ]

    logger.info(f"mooncake-store: starting {' '.join(command)} on {host}")
    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(  # nosec B603
            command, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    master = LaunchedMaster(process=process, address=f"{host}:{rpc_port}", log_path=log_path)
    logger.info(
        f"mooncake-store: master pid={process.pid} logging to {log_path} "
        f"(GLOG_v={env['GLOG_v']}); waiting for it to accept connections"
    )
    try:
        elapsed = _wait_until_accepting(host, rpc_port, timeout, process=process, log_path=log_path)
    except BaseException:
        master.stop()
        raise

    logger.info(
        f"mooncake-store: master ready at {master.address} after {elapsed:.1f}s "
        f"(metrics http://{host}:{metrics_port}, log {log_path})"
    )
    return master


@contextlib.contextmanager
def _published_manifest(manifest: PoolManifest, paths: Sequence[str]) -> Iterator[None]:
    """Write `manifest` to every path for the life of the context.

    Publishing is how anything else finds this pool: each server names the path
    as `file://<path>` in `pool`. Retracting on the way out matters as much as
    writing, since a manifest that outlives its master sends the next run's
    workers to a dead port.
    """
    record = json.dumps(manifest.to_json(), indent=2, sort_keys=True)
    for path in paths:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Renamed into place so a reader never sees a partial manifest.
        staging = f"{path}.partial"
        with open(staging, "w") as handle:
            handle.write(f"{record}\n")
        os.replace(staging, path)
        logger.info(f"mooncake-store: published the pool manifest to {path}: {manifest.describe()}")
    try:
        yield
    finally:
        for path in paths:
            with contextlib.suppress(OSError):
                os.remove(path)
                logger.info(f"mooncake-store: withdrew the pool manifest at {path}")


def _manifest_paths(run_dir: str, extra: Optional[str] = None) -> List[str]:
    """Where a master this process starts should publish its manifest.

    Always the run directory, plus wherever the deployment asked for.
    """
    paths = [os.path.join(run_dir, POOL_MANIFEST_NAME)]
    if extra and os.path.abspath(extra) not in {os.path.abspath(p) for p in paths}:
        paths.append(extra)
    return paths


@contextlib.contextmanager
def running_master(
    run_dir: str,
    *,
    pool_file: Optional[str] = None,
    rpc_port: int = DEFAULT_MASTER_PORT,
    metrics_port: int = DEFAULT_MASTER_METRICS_PORT,
    eviction_ratio: float = DEFAULT_MASTER_EVICTION_RATIO,
    metadata_server: str = DEFAULT_METADATA_SERVER,
    protocol: str = "rdma",
    namespace: str = DEFAULT_NAMESPACE,
    binary: str = DEFAULT_MASTER_BINARY,
    timeout: float = DEFAULT_MASTER_TIMEOUT,
) -> Iterator[LaunchedMaster]:
    """Run a master, and publish the manifest describing the pool it owns.

    The master is infrastructure: it outlives no single engine's startup and
    belongs to none of them, which is what lets several servers share one pool
    and what keeps the pool's own settings stated in one place.

    `pool_file` receives the manifest once the master answers, so servers can
    name a path instead of an address nobody knows until the scheduler has
    placed this process. One is written to `run_dir` either way.
    """
    os.makedirs(run_dir, exist_ok=True)
    master = _launch_master(
        run_dir,
        rpc_port=rpc_port,
        metrics_port=metrics_port,
        eviction_ratio=eviction_ratio,
        binary=binary,
        timeout=timeout,
    )
    manifest = PoolManifest(
        master_server_address=master.address,
        metadata_server=metadata_server,
        protocol=protocol,
        namespace=namespace,
        metrics_port=metrics_port,
        eviction_ratio=eviction_ratio,
    )
    try:
        with _published_manifest(manifest, _manifest_paths(run_dir, pool_file)):
            yield master
    finally:
        master.stop()
        logger.info(f"mooncake-store: master at {master.address} stopped")


def _log_contribution(segment_size: int, role: str, run_dir: str) -> None:
    """State the capacity arithmetic while the numbers are still in hand.

    Capacity is what explains a hit rate, and it is a sum over processes that
    no process can see. Each one can at least report its own term and the node
    total it implies, so a pool that came up an order of magnitude smaller than
    intended is visible at bring-up rather than inferred from a topology
    afterwards. The whole sum lands in the run's report; see `ledger.py`.
    """
    from tensorrt_llm._utils import local_mpi_size

    gib = segment_size / (1 << 30)
    try:
        ranks_here = max(1, local_mpi_size())
    except Exception:  # noqa: BLE001 - reporting only; never fail bring-up here
        ranks_here = 1
    logger.info(
        f"mooncake-store: each of this server's ranks contributes {gib:.1f} GiB "
        f"({segment_size} bytes) as role={role}; with {ranks_here} rank(s) on "
        f"this node that is {ranks_here * gib:.1f} GiB of its host memory. "
        f"Pool capacity is the sum over every participating rank, reported in "
        f"{os.path.join(run_dir, 'segments')}."
    )


def claim_run_dir(run_dir: str, role: str) -> None:
    """Record that this server owns `run_dir`, or name the one that already does.

    Two servers sharing a run directory write different client configs to the
    same path, since each names its own role and its own node's RDMA devices.
    The last writer wins for both, leaving a server that transfers over
    another node's HCAs, or lends under another server's role, with nothing in
    either log to say so. Sharing is rejected rather than serialized.

    Raises:
        ValueError: if another server already claimed this directory.
    """
    claim = {"host": socket.gethostname(), "pid": os.getpid(), "role": role}
    path = os.path.join(run_dir, RUN_DIR_OWNER_NAME)
    try:
        # O_EXCL so the winner is decided by the filesystem and not by who
        # happens to read before the other writes.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        pass
    except OSError as exc:
        # Not fatal. The config write that follows reports an unwritable
        # directory more precisely.
        logger.warning(f"mooncake-store: could not claim {run_dir}: {exc}")
        return
    else:
        with os.fdopen(fd, "w") as handle:
            json.dump(claim, handle)
        return

    try:
        with open(path) as handle:
            owner = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning(
            f"mooncake-store: {path} exists but could not be read ({exc}), so "
            "this server cannot tell whether another one is using "
            f"{run_dir}; continuing."
        )
        return

    # Re-entry within one process is not a conflict.
    if owner.get("pid") == claim["pid"] and owner.get("host") == claim["host"]:
        return

    raise ValueError(
        f"mooncake-store: run directory {run_dir} is already used by the "
        f"server at {owner.get('host')} pid {owner.get('pid')} as "
        f"role={owner.get('role')}, and this one is on {claim['host']} as "
        f"role={claim['role']}. The rendered client config names the writing "
        "server's role and its node's RDMA devices, so sharing a run "
        "directory overwrites one server's config with another's. Give each "
        "server its own mooncake_store.run_dir; they join the same pool "
        "through mooncake_store.pool regardless."
    )


@contextlib.contextmanager
def provision_pool(pool: Any, run_dir: Optional[str] = None) -> Iterator[Optional[str]]:
    """Join the pool `pool` names and point this process's ranks at it.

    Yields the path of the client config written, or `None` when an inherited
    `MOONCAKE_CONFIG_PATH` was left in charge.

    Args:
        pool: A `MooncakeStoreConfig`.
        run_dir: Where to write the client config and the segment records.
            Defaults to `pool.run_dir`, else a temporary directory that is
            removed on exit.
    """
    inherited = os.getenv(CONFIG_PATH_ENV)
    if inherited:
        logger.info(
            f"mooncake-store: {CONFIG_PATH_ENV}={inherited} is already set, so "
            "kv_connector_config.mooncake_store is ignored and the pool it "
            "names is used as is."
        )
        yield None
        return

    run_dir = run_dir or getattr(pool, "run_dir", None)
    keep_run_dir = bool(run_dir)
    run_dir = run_dir or tempfile.mkdtemp(prefix="trtllm-mooncake-")
    os.makedirs(run_dir, exist_ok=True)
    if keep_run_dir:
        logger.info(f"mooncake-store: joining the pool, run directory {run_dir}")
    else:
        logger.info(
            f"mooncake-store: joining the pool with run directory {run_dir}, "
            "which is removed at shutdown along with this run's segment "
            "records; set mooncake_store.run_dir to keep them, which ranks an "
            "external launcher started also need in order to find this config"
        )

    timeout = float(getattr(pool, "master_timeout", DEFAULT_MASTER_TIMEOUT))
    exported = False
    try:
        manifest = resolve_pool(pool.pool, timeout)
        # Checked before any rank opens a handle so an absent master is
        # reported as such, rather than as the status code store.setup returns
        # for every kind of failure, in every rank, after the model has loaded.
        wait_for_master(manifest.master_server_address, timeout)
        logger.info(f"mooncake-store: joining the pool at {manifest.describe()}")

        config_path = os.path.join(run_dir, CLIENT_CONFIG_NAME)
        config = _client_config(pool, manifest, resolve_device_name(manifest.protocol, ""))
        claim_run_dir(run_dir, config["role"])
        # Renamed into place so a rank never reads half a config. The staging
        # name carries this process's pid so that two writers, where a claim
        # is stale or bypassed, cannot delete each other's staging file.
        staging = f"{config_path}.partial.{os.getpid()}"
        try:
            with open(staging, "w") as handle:
                json.dump(config, handle, indent=2)
            os.replace(staging, config_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(staging)
            raise
        # Inherited by the ranks the LLM constructor spawns. Ranks an external
        # launcher started were already running, so they read the config out of
        # the run directory instead; see provisioned_config_path.
        os.environ[CONFIG_PATH_ENV] = config_path
        exported = True
        logger.info(
            f"mooncake-store: {CONFIG_PATH_ENV}={config_path} "
            f"({json.dumps(config, sort_keys=True)})"
        )
        _log_contribution(config["global_segment_size"], config["role"], run_dir)
        yield config_path
    finally:
        if exported:
            os.environ.pop(CONFIG_PATH_ENV, None)
        if not keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


@contextlib.contextmanager
def maybe_provision_pool(kv_connector_config: Any) -> Iterator[None]:
    """Provision the pool if this deployment asked the server to.

    A no-op for every other connector, and for a `mooncake-store` config that
    left `mooncake_store` unset, since such a deployment is told about its pool
    through `MOONCAKE_CONFIG_PATH` instead.
    """
    if not uses_connector(kv_connector_config, "mooncake-store"):
        yield
        return
    pool = kv_connector_config.mooncake_store
    if pool is None:
        yield
        return
    with provision_pool(pool):
        yield
