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
"""The two pieces of a Mooncake pool that outlive any one engine.

A server that owns its pool needs neither of these: it describes the pool in
`kv_connector_config.mooncake_store` and `trtllm-serve` provisions it during
bringup. They exist for the pools it cannot own -- one shared by several
engines, one that has to survive a restart, one whose capacity has to come from
nodes that run no connector -- so that those deployments are still assembled
from things TensorRT-LLM ships rather than from a launch script of one's own.
"""

import json
import os
import signal
import tempfile
import threading
import time
from typing import Optional

import click

from tensorrt_llm.logger import logger


def _until_signalled() -> threading.Event:
    """An event that SIGINT and SIGTERM set.

    Both commands hold a resource -- a child process, a mounted segment --
    whose release is in a ``finally``. Default SIGTERM handling would skip it,
    leaving the master unreaped or the pool advertising memory that has gone.
    """
    stopping = threading.Event()

    def stop(signum, _frame):
        logger.info(f"mooncake-store: signal {signum} received, shutting down")
        stopping.set()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, stop)
    return stopping


@click.command("mooncake_master")
@click.option("--rpc_port",
              type=int,
              default=50051,
              show_default=True,
              help="Port the store clients reach the master on.")
@click.option("--metrics_port",
              type=int,
              default=9004,
              show_default=True,
              help="Prometheus port. Pool occupancy and eviction are read "
              "from here or from the master's log.")
@click.option("--eviction_ratio",
              type=float,
              default=0.05,
              show_default=True,
              help="Fraction of the pool freed per eviction pass.")
@click.option("--address_file",
              type=str,
              default=None,
              help="File to publish 'host:port' to once the master answers. "
              "Workers name it as master_server_address: file://<path>, which "
              "is how they reach a master whose host the scheduler chose. "
              "Removed on exit so a stale address is never dialed.")
@click.option("--run_dir",
              type=str,
              default=None,
              help="Where to keep the master's log. Defaults to "
              "$TRTLLM_MOONCAKE_RUN_DIR, else a temporary directory.")
def mooncake_master(rpc_port: int, metrics_port: int, eviction_ratio: float,
                    address_file: Optional[str], run_dir: Optional[str]):
    """Run a mooncake_master for as long as this command runs.

    For a pool that must not belong to any one engine: several servers sharing
    it, or one that has to still be there after a server restarts. A single
    server with a pool of its own should set `mooncake_store.launch_master`
    instead and skip this entirely.
    """
    # Imported here rather than at module scope so that reaching any other
    # subcommand, or --help, does not pay for the connector package.
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import \
        running_master
    from tensorrt_llm.llmapi.llm_args import MooncakeStoreConfig

    pool = MooncakeStoreConfig(
        launch_master=True,
        master_port=rpc_port,
        master_metrics_port=metrics_port,
        master_eviction_ratio=eviction_ratio,
    )
    run_dir = run_dir or os.getenv(
        "TRTLLM_MOONCAKE_RUN_DIR") or tempfile.mkdtemp(
            prefix="trtllm-mooncake-master-")

    stopping = _until_signalled()
    with running_master(pool, run_dir, address_file=address_file) as master:
        while not stopping.is_set():
            if (code := master.process.poll()) is not None:
                # Its own death is the interesting outcome: the pool is gone
                # and every client is about to start failing.
                raise click.ClickException(
                    f"mooncake_master exited with code {code}. See "
                    f"{master.log_path}")
            stopping.wait(1.0)


@click.command("mooncake_donor")
@click.option("--master_server_address",
              type=str,
              default=None,
              help="Master to join, as host:port or file://<path> naming a "
              "file that holds one. Defaults to the master_server_address in "
              "--config.")
@click.option("--segment_size",
              type=str,
              default="32GiB",
              show_default=True,
              help="Host memory to contribute from this node. Deliberately "
              "separate from a config's global_segment_size, which is sized "
              "for an engine worker rather than a node lending what it can "
              "spare.")
@click.option("--config",
              type=str,
              default=None,
              help="Mooncake JSON config describing the pool, for the "
              "settings not given here. Defaults to $MOONCAKE_CONFIG_PATH.")
@click.option("--protocol",
              type=str,
              default=None,
              help="Transport, 'rdma' or 'tcp'. Defaults to --config's, else "
              "rdma.")
@click.option("--device_name",
              type=str,
              default=None,
              help="RDMA device, from ibv_devinfo. Defaults to --config's.")
@click.option("--metadata_server",
              type=str,
              default=None,
              help="Mooncake metadata service. Defaults to --config's.")
@click.option("--ready_file",
              type=str,
              default=None,
              help="File to create once the segment is mounted, for launchers "
              "that must not let prefill start writing before the pool has "
              "this capacity.")
@click.option("--heartbeat_seconds",
              type=int,
              default=300,
              show_default=True,
              help="Interval between liveness lines. 0 disables them.")
def mooncake_donor(master_server_address: Optional[str], segment_size: str,
                   config: Optional[str], protocol: Optional[str],
                   device_name: Optional[str], metadata_server: Optional[str],
                   ready_file: Optional[str], heartbeat_seconds: int):
    """Lend this node's host memory to a Mooncake pool, for as long as it runs.

    Pool capacity comes only from processes that open a store handle, and in a
    disaggregated deployment only the context servers do -- so the pool is
    prefill-node memory, caching prefill's own GPUs. Running this on the
    generation nodes puts their memory in the same pool while leaving those
    engines connector-free.
    """
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import (
        DEFAULT_DONOR_LOCAL_BUFFER_SIZE, donate_segment, master_timeout,
        parse_size, resolve_master_address)
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import \
        CONFIG_PATH_ENV

    raw = {}
    config = config or os.getenv(CONFIG_PATH_ENV)
    if config:
        with open(config) as handle:
            raw = json.load(handle)

    master = master_server_address or raw.get("master_server_address", "")
    if not master:
        raise click.UsageError(
            "No master to join. Pass --master_server_address, or a --config "
            f"naming one (or set {CONFIG_PATH_ENV}).")

    donating = parse_size(segment_size)
    stopping = _until_signalled()
    with donate_segment(
            resolve_master_address(master, master_timeout()),
            donating,
            protocol=protocol or raw.get("protocol", "rdma"),
            device_name=device_name or raw.get("device_name", "") or "",
            metadata_server=metadata_server or raw.get("metadata_server", ""),
            local_buffer_size=parse_size(
                raw.get("local_buffer_size_donor",
                        DEFAULT_DONOR_LOCAL_BUFFER_SIZE)),
    ) as host:
        if ready_file:
            with open(ready_file, "w") as handle:
                handle.write(f"{host} {donating}\n")

        # Idle by design. A put or get here would make this node a client in
        # the traffic sense, which is the thing keeping the generation engine
        # connector-free is meant to avoid.
        started = time.monotonic()
        while not stopping.is_set():
            if heartbeat_seconds <= 0:
                stopping.wait()
                continue
            if not stopping.wait(heartbeat_seconds):
                logger.info("mooncake-store: still donating after "
                            f"{(time.monotonic() - started) / 60:.0f}m")
