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

A server that owns its pool needs neither: it describes the pool in
`kv_connector_config.mooncake_store` and `trtllm-serve` provisions it during
bringup. These commands exist for the pools it cannot own, such as one shared by
several engines, one that has to survive a restart, or one whose capacity comes
from nodes that run no connector.
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

    Both commands hold a resource, a child process or a mounted segment, whose
    release is in a `finally`. Default SIGTERM handling would skip it, leaving
    the master unreaped or the pool advertising memory that has gone.
    """
    stopping = threading.Event()

    def stop(signum, _frame):
        logger.info(f"mooncake-store: signal {signum} received, shutting down")
        stopping.set()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, stop)
    return stopping


@click.command("mooncake_master")
@click.option(
    "--rpc_port",
    type=int,
    default=50051,
    show_default=True,
    help="Port the store clients reach the master on.",
)
@click.option(
    "--metrics_port",
    type=int,
    default=9004,
    show_default=True,
    help="Prometheus port. Pool occupancy and eviction are read "
    "from here or from the master's log.",
)
@click.option(
    "--eviction_ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Fraction of the pool freed per eviction pass.",
)
@click.option(
    "--address_file",
    type=str,
    default=None,
    help="File to publish 'host:port' to once the master answers. "
    "Workers name it as master_server_address: file://<path>, which "
    "is how they reach a master whose host the scheduler chose. "
    "Removed on exit so a stale address is never dialed.",
)
@click.option(
    "--run_dir",
    type=str,
    default=None,
    help="Where to keep the master's log. Defaults to "
    "$TRTLLM_MOONCAKE_RUN_DIR, else a temporary directory.",
)
@click.option(
    "--heartbeat_seconds",
    type=int,
    default=300,
    show_default=True,
    help="Interval between liveness lines. 0 disables them.",
)
def mooncake_master(
    rpc_port: int,
    metrics_port: int,
    eviction_ratio: float,
    address_file: Optional[str],
    run_dir: Optional[str],
    heartbeat_seconds: int,
):
    """Run a mooncake_master for as long as this command runs.

    A single server with a pool of its own should set
    `mooncake_store.launch_master` instead.
    """
    # Imported lazily so other subcommands and --help do not pay for the
    # connector package.
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import running_master
    from tensorrt_llm.llmapi.llm_args import MooncakeStoreConfig

    pool = MooncakeStoreConfig(
        launch_master=True,
        master_port=rpc_port,
        master_metrics_port=metrics_port,
        master_eviction_ratio=eviction_ratio,
    )
    run_dir = (
        run_dir
        or os.getenv("TRTLLM_MOONCAKE_RUN_DIR")
        or tempfile.mkdtemp(prefix="trtllm-mooncake-master-")
    )

    stopping = _until_signalled()
    with running_master(pool, run_dir, address_file=address_file) as master:
        logger.info(
            f"mooncake-store: this master owns the pool until this command "
            f"stops; address {master.address}, log {master.log_path}, metrics "
            f"http://{master.address.rsplit(':', 1)[0]}:{metrics_port}/metrics"
        )
        started = time.monotonic()
        announced = started
        while not stopping.is_set():
            if (code := master.process.poll()) is not None:
                # The pool is gone once the master dies, and every client is
                # about to start failing.
                raise click.ClickException(
                    f"mooncake_master exited with code {code}. See {master.log_path}"
                )
            stopping.wait(1.0)
            now = time.monotonic()
            # Distinguishes a dead master from a dead fabric once clients
            # start failing.
            if heartbeat_seconds > 0 and now - announced >= heartbeat_seconds:
                announced = now
                logger.info(
                    f"mooncake-store: master at {master.address} alive after "
                    f"{(now - started) / 60:.0f}m"
                )


@click.command("mooncake_donor")
@click.option(
    "--master_server_address",
    type=str,
    default=None,
    help="Master to join, as host:port or file://<path> naming a "
    "file that holds one. Defaults to the master_server_address in "
    "--config.",
)
@click.option(
    "--segment_size",
    type=str,
    default="32GiB",
    show_default=True,
    help="Host memory to contribute from this node. Deliberately "
    "separate from a config's global_segment_size, which is sized "
    "for an engine worker rather than a node lending what it can "
    "spare.",
)
@click.option(
    "--config",
    type=str,
    default=None,
    help="Mooncake JSON config describing the pool, for the "
    "settings not given here. Defaults to $MOONCAKE_CONFIG_PATH.",
)
@click.option(
    "--protocol",
    type=str,
    default=None,
    help="Transport, 'rdma' or 'tcp'. Defaults to --config's, else rdma.",
)
@click.option(
    "--device_name",
    type=str,
    default=None,
    help="RDMA device, from ibv_devinfo. Defaults to --config's.",
)
@click.option(
    "--metadata_server",
    type=str,
    default=None,
    help="Mooncake metadata service. Defaults to --config's, else P2PHANDSHAKE.",
)
@click.option(
    "--ready_file",
    type=str,
    default=None,
    help="File to create once the segment is mounted, for launchers "
    "that must not let prefill start writing before the pool has "
    "this capacity.",
)
@click.option(
    "--heartbeat_seconds",
    type=int,
    default=300,
    show_default=True,
    help="Interval between liveness lines. 0 disables them.",
)
def mooncake_donor(
    master_server_address: Optional[str],
    segment_size: str,
    config: Optional[str],
    protocol: Optional[str],
    device_name: Optional[str],
    metadata_server: Optional[str],
    ready_file: Optional[str],
    heartbeat_seconds: int,
):
    """Lend this node's host memory to a Mooncake pool, for as long as it runs.

    Running this on the generation nodes puts their memory into the pool while
    leaving those engines connector-free.
    """
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import (
        DEFAULT_DONOR_LOCAL_BUFFER_SIZE,
        donate_segment,
        master_timeout,
        parse_size,
        resolve_master_address,
        wait_for_master,
    )
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
        CONFIG_PATH_ENV,
        DEFAULT_METADATA_SERVER,
    )

    raw = {}
    config = config or os.getenv(CONFIG_PATH_ENV)
    if config:
        with open(config) as handle:
            raw = json.load(handle)

    master = master_server_address or raw.get("master_server_address", "")
    if not master:
        raise click.UsageError(
            "No master to join. Pass --master_server_address, or a --config "
            f"naming one (or set {CONFIG_PATH_ENV})."
        )

    donating = parse_size(segment_size)
    resolved = resolve_master_address(master, master_timeout())
    # Before setup, so an absent master is reported as such.
    wait_for_master(resolved)

    stopping = _until_signalled()
    with donate_segment(
        resolved,
        donating,
        protocol=protocol or raw.get("protocol", "rdma"),
        device_name=device_name or raw.get("device_name", "") or "",
        metadata_server=(metadata_server or raw.get("metadata_server") or DEFAULT_METADATA_SERVER),
        local_buffer_size=parse_size(
            raw.get("local_buffer_size_donor", DEFAULT_DONOR_LOCAL_BUFFER_SIZE)
        ),
    ) as host:
        if ready_file:
            with open(ready_file, "w") as handle:
                handle.write(f"{host} {donating}\n")
            logger.info(
                f"mooncake-store: announced this segment in "
                f"{ready_file}, so a launcher waiting on the pool's "
                "capacity can proceed"
            )

        # Idle by design: a put or get here would make this node a traffic
        # client, which is what donation exists to avoid.
        started = time.monotonic()
        while not stopping.is_set():
            if heartbeat_seconds <= 0:
                stopping.wait()
                continue
            if not stopping.wait(heartbeat_seconds):
                logger.info(
                    f"mooncake-store: {host} still lending "
                    f"{donating / 1024**3:.1f}GiB to the pool at {master} "
                    f"after {(time.monotonic() - started) / 60:.0f}m"
                )
