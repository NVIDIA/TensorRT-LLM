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
"""Bring the Mooncake store's pool up as part of a server's own startup.

The connector needs two things that are not the engine's to produce: a reachable
`mooncake_master`, and a JSON client config named by `MOONCAKE_CONFIG_PATH` that
points every worker at it.

`provision_pool` does that work inside the serving process. It resolves the
master, either launching one here or checking that the configured one answers,
renders the client config, and exports `MOONCAKE_CONFIG_PATH`, which reaches the
ranks because the LLM constructor spawns them from this process. Everything it
started is torn down when the context exits.

A master launched here lives and dies with the server, so it suits one engine
talking to its own pool. Several engines sharing a pool, or a pool meant to
survive a restart, need a master with its own lifetime named by
`master_server_address`.
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess  # nosec B404
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

from ..registry import uses_connector
from .config import CLIENT_CONFIG_NAME, CONFIG_PATH_ENV, RUN_DIR_ENV

__all__ = [
    "local_address",
    "maybe_provision_pool",
    "master_timeout",
    "provision_pool",
    "resolve_device_name",
    "resolve_master_address",
    "running_master",
    "wait_for_master",
]

#: Override the binary that `launch_master` runs.
MASTER_BINARY_ENV = "TRTLLM_MOONCAKE_MASTER_BINARY"
#: How long to wait for a master to accept connections, in seconds.
MASTER_TIMEOUT_ENV = "TRTLLM_MOONCAKE_MASTER_TIMEOUT"
DEFAULT_MASTER_BINARY = "mooncake_master"
DEFAULT_MASTER_TIMEOUT = 60.0
MASTER_LOG_NAME = "mooncake_master.log"
#: Name a launched master's address is always published under in the run
#: directory, so even a run that named no address file records its pool.
MASTER_ADDRESS_NAME = "master.addr"
#: Prefix that makes `master_server_address` name a file holding the address
#: rather than the address itself.
ADDRESS_FILE_SCHEME = "file://"
#: Lines of the master's log to quote when startup fails, since its last words
#: (a port in use, a bad flag) are usually the whole diagnosis.
LOG_TAIL_LINES = 20


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


def master_timeout() -> float:
    raw = os.getenv(MASTER_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_MASTER_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{MASTER_TIMEOUT_ENV}={raw!r} is not a number") from exc
    if timeout <= 0:
        raise ValueError(f"{MASTER_TIMEOUT_ENV}={raw!r} must be > 0")
    return timeout


def _split_address(address: str) -> Optional[Tuple[str, int]]:
    """Split `host:port`, or return `None` if it is not in that form."""
    host, separator, port = address.rpartition(":")
    if not separator or not port.isdigit():
        return None
    return host.strip("[]"), int(port)


def resolve_master_address(address: str, timeout: float) -> str:
    """Read a `file://` address through, and pass anything else along.

    A master with its own lifetime runs on whichever host its scheduler gave
    it, which is not known when the worker configs are written. Naming the file
    it publishes to keeps the address out of both the config and the launch
    script: `trtllm-serve mooncake_master --address-file` writes it, every
    worker's `master_server_address` names the same path, and the wait here
    doubles as the wait for the master to exist at all.
    """
    if not address.startswith(ADDRESS_FILE_SCHEME):
        return address

    path = address[len(ADDRESS_FILE_SCHEME) :]
    started = time.monotonic()
    deadline = started + timeout
    announced = started
    logger.info(f"mooncake-store: reading the master's address from {path}")
    while True:
        try:
            published = open(path).read().strip()
        except FileNotFoundError:
            published = ""
        if published:
            logger.info(f"mooncake-store: {path} names the master at {published}")
            return published
        now = time.monotonic()
        if now - announced >= 5.0:
            announced = now
            # Waiting on a master in another job is normal here, so say so
            # rather than letting the wait look like a hang.
            logger.info(
                f"mooncake-store: no master address in {path} yet "
                f"({now - started:.0f}s of {timeout:g}s); waiting for the "
                "master to start and publish it"
            )
        if now >= deadline:
            raise TimeoutError(
                f"No Mooncake master address appeared in {path} within "
                f"{timeout:g}s. Start one with 'trtllm-serve mooncake_master "
                f"--address-file {path}', or name a reachable host:port in "
                f"master_server_address. Raise {MASTER_TIMEOUT_ENV} if the "
                "master is only slow to start."
            )
        time.sleep(0.5)


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
                f"connections within {timeout:g}s ({last_error}). Raise "
                f"{MASTER_TIMEOUT_ENV} if it is only slow to start."
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


def wait_for_master(master_address: str, timeout: Optional[float] = None) -> Optional[float]:
    """Block until the master at `master_address` accepts connections.

    Reaching a master that is not there otherwise fails deep inside
    `store.setup`, in every rank, after the model has loaded, as a bare status
    code. One socket beforehand turns that into a line naming the address.

    Returns how long it took, or `None` if the address was not in `host:port`
    form and could not be checked.
    """
    timeout = master_timeout() if timeout is None else timeout
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


def _client_config(
    pool: Any, master_address: str, device_name: Optional[str] = None
) -> Dict[str, Any]:
    """Render the Mooncake client config for a pool.

    The schema is vLLM's, so one pool can serve both engines. `role` is written
    as `both` because the file describes the pool; the directions of traffic a
    given process drives come from its own `TRTLLM_MOONCAKE_STORE_ROLE`.
    """
    config: Dict[str, Any] = {
        "metadata_server": pool.metadata_server,
        "master_server_address": master_address,
        "protocol": pool.protocol,
        "device_name": pool.device_name if device_name is None else device_name,
        "global_segment_size": pool.global_segment_size,
        "local_buffer_size": pool.local_buffer_size,
        "role": "both",
        "transfer_batch_size": pool.transfer_batch_size,
        "stage_through_host": pool.stage_through_host,
    }
    if pool.cache_prefix is not None:
        config["cache_prefix"] = pool.cache_prefix
    # Left out when unset so the connector's own default applies instead of a
    # second copy of it here.
    if pool.staging_buffer_bytes is not None:
        config["staging_buffer_bytes"] = pool.staging_buffer_bytes
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


def _launch_master(pool: Any, run_dir: str) -> LaunchedMaster:
    """Start a master on this host and wait for it to answer."""
    binary = os.getenv(MASTER_BINARY_ENV) or DEFAULT_MASTER_BINARY
    resolved = shutil.which(binary)
    if resolved is None:
        raise FileNotFoundError(
            f"{binary!r} is not on PATH, so launch_master cannot start a "
            "Mooncake master. It ships with the Mooncake runtime, which "
            "docker/common/install_mooncake.sh installs. Point "
            f"{MASTER_BINARY_ENV} at the binary, or drop launch_master and "
            "set master_server_address to a master you run yourself."
        )

    host = local_address()
    log_path = os.path.join(run_dir, MASTER_LOG_NAME)

    # glog writes to files under /tmp unless redirected, so without
    # GLOG_logtostderr the log opened below stays empty. GLOG_v=1 adds the
    # per-RPC lines showing segments registering and keys moving, which is the
    # only view of the pool's own side of the conversation short of scraping
    # the metrics port.
    env = dict(os.environ, GLOG_logtostderr="1")
    env.setdefault("GLOG_v", "1")
    command = [
        resolved,
        f"--rpc_port={pool.master_port}",
        f"--metrics_port={pool.master_metrics_port}",
        f"--eviction_ratio={pool.master_eviction_ratio}",
    ]

    logger.info(f"mooncake-store: starting {' '.join(command)} on {host}")
    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(  # nosec B603
            command, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    master = LaunchedMaster(
        process=process, address=f"{host}:{pool.master_port}", log_path=log_path
    )
    logger.info(
        f"mooncake-store: master pid={process.pid} logging to {log_path} "
        f"(GLOG_v={env['GLOG_v']}); waiting for it to accept connections"
    )
    try:
        elapsed = _wait_until_accepting(
            host, pool.master_port, master_timeout(), process=process, log_path=log_path
        )
    except BaseException:
        master.stop()
        raise

    logger.info(
        f"mooncake-store: master ready at {master.address} after {elapsed:.1f}s "
        f"(metrics http://{host}:{pool.master_metrics_port}, log {log_path})"
    )
    return master


@contextlib.contextmanager
def _published_address(address: str, paths: Sequence[str]) -> Iterator[None]:
    """Write `address` to every path for the life of the context.

    Publishing is how anything else finds this master: a donor or a second
    server names the path as `file://<path>` in `master_server_address`.
    Retracting on the way out matters as much as writing, since an address that
    outlives its master sends the next run's workers to a dead port.
    """
    for path in paths:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Renamed into place so a reader never sees a partial address.
        staging = f"{path}.partial"
        with open(staging, "w") as handle:
            handle.write(f"{address}\n")
        os.replace(staging, path)
        logger.info(f"mooncake-store: published master {address} to {path}")
    try:
        yield
    finally:
        for path in paths:
            with contextlib.suppress(OSError):
                os.remove(path)
                logger.info(f"mooncake-store: withdrew the master address at {path}")


def _address_files(run_dir: str, extra: Optional[str] = None) -> List[str]:
    """Where a master this process starts should publish its address.

    Always the run directory, plus wherever the deployment asked for.
    """
    paths = [os.path.join(run_dir, MASTER_ADDRESS_NAME)]
    if extra and os.path.abspath(extra) not in {os.path.abspath(p) for p in paths}:
        paths.append(extra)
    return paths


@contextlib.contextmanager
def running_master(
    pool: Any, run_dir: str, address_file: Optional[str] = None
) -> Iterator[LaunchedMaster]:
    """Run a master whose lifetime is this process's rather than an engine's.

    `provision_pool` covers the server that owns its pool. Several engines on
    one pool, or a pool that has to survive a restart, need the master
    somewhere that is not any of them.

    `address_file` receives `host:port` once the master answers, so workers can
    name the file instead of an address nobody knows until the scheduler has
    placed this process. One is written to `run_dir` either way.
    """
    os.makedirs(run_dir, exist_ok=True)
    master = _launch_master(pool, run_dir)
    try:
        with _published_address(master.address, _address_files(run_dir, address_file)):
            yield master
    finally:
        master.stop()
        logger.info(f"mooncake-store: master at {master.address} stopped")


@contextlib.contextmanager
def provision_pool(pool: Any, run_dir: Optional[str] = None) -> Iterator[Optional[str]]:
    """Make `pool` reachable and name it in this process's environment.

    Yields the path of the client config written, or `None` when an inherited
    `MOONCAKE_CONFIG_PATH` was left in charge.

    Args:
        pool: A `MooncakeStoreConfig`.
        run_dir: Where to write the client config and the master's log.
            Defaults to `TRTLLM_MOONCAKE_RUN_DIR`, else a temporary directory
            that is removed on exit.
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

    keep_run_dir = bool(run_dir or os.getenv(RUN_DIR_ENV))
    run_dir = run_dir or os.getenv(RUN_DIR_ENV) or tempfile.mkdtemp(prefix="trtllm-mooncake-")
    os.makedirs(run_dir, exist_ok=True)
    if keep_run_dir:
        logger.info(f"mooncake-store: provisioning the pool, run directory {run_dir}")
    else:
        logger.info(
            f"mooncake-store: provisioning the pool in {run_dir}, which is "
            f"removed at shutdown along with the master's log; set "
            f"{RUN_DIR_ENV} to keep them"
        )

    master: Optional[LaunchedMaster] = None
    exported = False
    with contextlib.ExitStack() as stack:
        try:
            if pool.launch_master:
                master = _launch_master(pool, run_dir)
                master_address = master.address
                # Published even when only this server uses it, since that is
                # how its donors reach it.
                stack.enter_context(
                    _published_address(
                        master_address, _address_files(run_dir, pool.master_address_file)
                    )
                )
            else:
                master_address = resolve_master_address(
                    pool.master_server_address, master_timeout()
                )
                wait_for_master(master_address)
                logger.info(f"mooncake-store: using the master at {master_address}")

            config_path = os.path.join(run_dir, CLIENT_CONFIG_NAME)
            config = _client_config(
                pool, master_address, resolve_device_name(pool.protocol, pool.device_name)
            )
            with open(config_path, "w") as handle:
                json.dump(config, handle, indent=2)
            # Inherited by the ranks the LLM constructor spawns. Ranks an
            # external launcher started were already running, so they read the
            # config out of the run directory instead; see
            # provisioned_config_path.
            os.environ[CONFIG_PATH_ENV] = config_path
            exported = True
            logger.info(
                f"mooncake-store: {CONFIG_PATH_ENV}={config_path} "
                f"({json.dumps(config, sort_keys=True)})"
            )
            # Capacity is what explains a low hit rate, so state the
            # arithmetic instead of leaving it to be derived later.
            logger.info(
                "mooncake-store: this server's ranks will each contribute "
                f"global_segment_size={pool.global_segment_size} to the pool; "
                "total capacity is that times the number of ranks that open a "
                "handle, plus whatever any mooncake_donation adds"
            )
            yield config_path
        finally:
            if exported:
                os.environ.pop(CONFIG_PATH_ENV, None)
            if master is not None:
                master.stop()
                logger.info(f"mooncake-store: master at {master.address} stopped")
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
