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

The connector needs two things that are not the engine's to produce: a
reachable ``mooncake_master``, and a JSON client config named by
``MOONCAKE_CONFIG_PATH`` that points every worker at it. Both were the SLURM
harness's job, which left a single ``trtllm-serve`` unable to use the connector
without borrowing that harness.

``provision_pool`` does the same work inside the serving process. It resolves
the master -- launching one here, or checking that the configured one answers
-- renders the client config, and exports ``MOONCAKE_CONFIG_PATH``, which
reaches the ranks because the LLM constructor spawns them from this process.
Everything it started is torn down when the context exits.

A master launched here lives and dies with the server, so it is only right for
one engine talking to its own pool. Several engines sharing a pool, or a pool
meant to survive a restart, need a master with its own lifetime, named by
``master_server_address``.
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
from typing import Any, Dict, Iterator, Optional, Tuple

from tensorrt_llm.logger import logger

from ..registry import uses_connector
from .config import CONFIG_PATH_ENV

__all__ = [
    "local_address",
    "maybe_provision_pool",
    "master_timeout",
    "provision_pool",
    "resolve_master_address",
    "running_master",
]

#: Override the binary that ``launch_master`` runs.
MASTER_BINARY_ENV = "TRTLLM_MOONCAKE_MASTER_BINARY"
#: How long to wait for a master to accept connections, in seconds.
MASTER_TIMEOUT_ENV = "TRTLLM_MOONCAKE_MASTER_TIMEOUT"
#: Where to keep the generated client config and the master's log. Set it to
#: keep them after shutdown; otherwise they live in a temporary directory.
RUN_DIR_ENV = "TRTLLM_MOONCAKE_RUN_DIR"

DEFAULT_MASTER_BINARY = "mooncake_master"
DEFAULT_MASTER_TIMEOUT = 60.0
CLIENT_CONFIG_NAME = "mooncake.json"
MASTER_LOG_NAME = "mooncake_master.log"
#: Prefix that makes ``master_server_address`` name a file holding the address
#: rather than the address itself.
ADDRESS_FILE_SCHEME = "file://"


def local_address() -> str:
    """The address this host is known by inside the pool.

    Deliberately the same derivation the connector worker uses for its own
    hostname, so the master and the segments registering with it agree on
    which host they are on.
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
    """Split ``host:port``, or return ``None`` if it is not in that form."""
    host, separator, port = address.rpartition(":")
    if not separator or not port.isdigit():
        return None
    return host.strip("[]"), int(port)


def resolve_master_address(address: str, timeout: float) -> str:
    """Read a ``file://`` address through, and pass anything else along.

    A master with its own lifetime is on whichever host its scheduler gave
    it, which is not known when the worker configs are written. Naming the
    file it publishes to instead keeps the address out of the config and out
    of a launch script: ``trtllm-serve mooncake_master --address-file`` writes
    it, every worker's ``master_server_address`` names the same path, and the
    wait here is also the wait for the master to exist at all.
    """
    if not address.startswith(ADDRESS_FILE_SCHEME):
        return address

    path = address[len(ADDRESS_FILE_SCHEME):]
    deadline = time.monotonic() + timeout
    while True:
        # Written whole by the master command, so a non-empty file is a
        # complete address rather than a prefix of one.
        try:
            published = open(path).read().strip()
        except FileNotFoundError:
            published = ""
        if published:
            logger.info(f"mooncake-store: {path} names the master at {published}")
            return published
        if time.monotonic() >= deadline:
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
    hint: str = "",
) -> None:
    """Block until the master accepts connections.

    A worker that opens its store handle before the master is listening fails
    outright, so the port -- not the presence of a process -- is what the
    ordering has to wait on. When the master is ours, its exit is checked first
    each pass, so a master that died is reported as that rather than as a
    timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        if process is not None and (code := process.poll()) is not None:
            raise RuntimeError(f"mooncake_master exited with code {code} during startup.{hint}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"The Mooncake master at {host}:{port} did not accept "
                f"connections within {timeout:g}s ({last_error}). Raise "
                f"{MASTER_TIMEOUT_ENV} if it is only slow to start.{hint}"
            )
        time.sleep(0.5)


def _client_config(pool: Any, master_address: str) -> Dict[str, Any]:
    """Render the Mooncake client config for a pool.

    The schema is vLLM's, so one pool can serve both engines. ``role`` is
    written as ``both`` because the file describes the pool; which directions
    of traffic a given process drives is its own
    ``TRTLLM_MOONCAKE_STORE_ROLE``.
    """
    config: Dict[str, Any] = {
        "metadata_server": pool.metadata_server,
        "master_server_address": master_address,
        "protocol": pool.protocol,
        "device_name": pool.device_name,
        "global_segment_size": pool.global_segment_size,
        "local_buffer_size": pool.local_buffer_size,
        "role": "both",
        "transfer_batch_size": pool.transfer_batch_size,
        "stage_through_host": pool.stage_through_host,
    }
    if pool.cache_prefix is not None:
        config["cache_prefix"] = pool.cache_prefix
    # Left out when unset so the connector's own default applies, rather than
    # restating it here for the two to drift apart.
    if pool.staging_buffer_bytes is not None:
        config["staging_buffer_bytes"] = pool.staging_buffer_bytes
    return config


@dataclass
class LaunchedMaster:
    """A ``mooncake_master`` owned by this process."""

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
            "mooncake_disagg/install_mooncake_runtime.sh installs. Point "
            f"{MASTER_BINARY_ENV} at the binary, or drop launch_master and "
            "set master_server_address to a master you run yourself."
        )

    host = local_address()
    log_path = os.path.join(run_dir, MASTER_LOG_NAME)
    hint = f" See {log_path}."

    # mooncake_master logs through glog, which writes files under /tmp unless
    # told otherwise, so without GLOG_logtostderr the log below stays empty.
    # GLOG_v=1 adds the per-RPC lines showing segments registering and keys
    # moving, which is the only view of the pool's side of the conversation
    # short of scraping the metrics port.
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
    try:
        _wait_until_accepting(host, pool.master_port, master_timeout(), process=process, hint=hint)
    except BaseException:
        master.stop()
        raise

    logger.info(
        f"mooncake-store: master ready at {master.address} "
        f"(metrics http://{host}:{pool.master_metrics_port}, log {log_path})"
    )
    return master


@contextlib.contextmanager
def running_master(
    pool: Any, run_dir: str, address_file: Optional[str] = None
) -> Iterator[LaunchedMaster]:
    """Run a master whose lifetime is this process's rather than an engine's.

    ``provision_pool`` covers the server that owns its pool. Everything else --
    several engines on one pool, a pool that has to survive a restart -- needs
    the master somewhere that is not any of them, which is what this is for.

    ``address_file`` receives ``host:port`` once the master answers, so the
    workers can name the file instead of an address nobody knows until the
    scheduler has placed this process.
    """
    os.makedirs(run_dir, exist_ok=True)
    master = _launch_master(pool, run_dir)
    try:
        if address_file:
            # Renamed into place so a reader sees either nothing or the whole
            # address. A half-written one would be dialed as if it were real.
            staging = f"{address_file}.partial"
            with open(staging, "w") as handle:
                handle.write(f"{master.address}\n")
            os.replace(staging, address_file)
            logger.info(f"mooncake-store: published {master.address} to {address_file}")
        yield master
    finally:
        if address_file:
            # The address outliving the master would send the next run's
            # workers to a port with nothing behind it.
            with contextlib.suppress(OSError):
                os.remove(address_file)
        master.stop()


@contextlib.contextmanager
def provision_pool(pool: Any, run_dir: Optional[str] = None) -> Iterator[Optional[str]]:
    """Make ``pool`` reachable and name it in this process's environment.

    Yields the path of the client config written, or ``None`` when an inherited
    ``MOONCAKE_CONFIG_PATH`` was left in charge.

    Args:
        pool: A ``MooncakeStoreConfig``.
        run_dir: Where to write the client config and the master's log.
            Defaults to ``TRTLLM_MOONCAKE_RUN_DIR``, else a temporary directory
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

    master: Optional[LaunchedMaster] = None
    exported = False
    try:
        if pool.launch_master:
            master = _launch_master(pool, run_dir)
            master_address = master.address
        else:
            master_address = resolve_master_address(
                pool.master_server_address, master_timeout()
            )
            # Reaching a master that is not there fails inside store.setup on
            # every rank, after the model has been loaded. Spend a socket now.
            if (endpoint := _split_address(master_address)) is None:
                logger.warning(
                    f"mooncake-store: cannot parse master_server_address="
                    f"{master_address!r} as host:port, so its reachability is "
                    "left for the workers to discover."
                )
            else:
                _wait_until_accepting(*endpoint, master_timeout())
            logger.info(f"mooncake-store: using the master at {master_address}")

        config_path = os.path.join(run_dir, CLIENT_CONFIG_NAME)
        config = _client_config(pool, master_address)
        with open(config_path, "w") as handle:
            json.dump(config, handle, indent=2)
        # The ranks that open store handles are spawned by the LLM constructor,
        # inheriting this environment; that is the only reason exporting it
        # here reaches them.
        os.environ[CONFIG_PATH_ENV] = config_path
        exported = True
        logger.info(
            f"mooncake-store: {CONFIG_PATH_ENV}={config_path} "
            f"({json.dumps(config, sort_keys=True)})"
        )
        yield config_path
    finally:
        if exported:
            os.environ.pop(CONFIG_PATH_ENV, None)
        if master is not None:
            master.stop()
        if not keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


@contextlib.contextmanager
def maybe_provision_pool(kv_connector_config: Any) -> Iterator[None]:
    """Provision the pool if this deployment asked the server to.

    A no-op for every other connector, and for a ``mooncake-store`` config
    that left ``mooncake_store`` unset: that deployment is told about its pool
    through ``MOONCAKE_CONFIG_PATH``, which is how the SLURM harness drives it.
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
