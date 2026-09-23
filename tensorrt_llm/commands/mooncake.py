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
"""The two pieces of a Mooncake pool that are not any one engine's to run.

`mooncake_master` owns a pool for as long as the command runs, and with
`--config` also writes the client config its workers read into `--run_dir`.
Pointing every server at that directory with `$TRTLLM_MOONCAKE_RUN_DIR`, or at
the file with `$MOONCAKE_CONFIG_PATH`, is how a pool is described today.

`mooncake_donor` lends a node's host memory to a pool it does not otherwise
use, which is how capacity comes from nodes whose engines have no connector.
"""

import contextlib
import json
import os
import signal
import tempfile
import time
from typing import Optional

import click

import tensorrt_llm.usage as usage
from tensorrt_llm.commands import _telemetry as _command_telemetry
from tensorrt_llm.logger import logger
from tensorrt_llm.usage.config import UsageContext

#: How long a donor with heartbeats turned off sleeps between wakeups. Only
#: the signal that ends the command interrupts it, so the value is arbitrary.
_IDLE_POLL_SECONDS = 60.0

#: Both commands sit in the telemetry-aware `trtllm-serve` group, so they have
#: to offer the same opt-out the group documents.
_telemetry_option = click.option(
    "--telemetry/--no-telemetry",
    default=True,
    help="Enable or disable anonymous usage telemetry collection.",
)


def _apply_cli_telemetry(telemetry: bool) -> None:
    """Honor --no-telemetry for a command that reads no config of its own.

    The group already applies the flag it finds in argv, so this matters when
    the command is reached through its callback rather than through the CLI.
    """
    if telemetry:
        return
    usage.apply_usage_session_config(
        {"disabled": True},
        default_usage_context=UsageContext.CLI_SERVE.value,
        component="server",
        lifecycle_phase="config_validation",
    )


@contextlib.contextmanager
def _signal_handoff():
    """Turn SIGINT and SIGTERM into the exit the telemetry boundary expects.

    Both commands hold a resource, a child process or a mounted segment, whose
    release is in a `finally`. Default SIGTERM handling would skip it, leaving
    the master unreaped or the pool advertising memory that has gone.

    `raise_signal_exit` unwinds those context managers and carries the signal
    number out to `trtllm-serve`, which is what reports the exit as a signal
    rather than as a clean one.

    Wrap this around the resource so the log below follows the release.
    """
    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, _command_telemetry.raise_signal_exit)
    try:
        yield
    except _command_telemetry.SignalExit as stopping:
        # Logged here rather than in the handler, which must not take the
        # logging lock.
        logger.info(f"mooncake-store: signal {stopping.signal_number} received, shut down")
        raise


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
    "--config",
    type=str,
    default=None,
    help="Mooncake JSON config describing the pool. When given, a copy "
    "naming this master is written to --run_dir as the client config "
    "every worker reads, so no external script has to render one. "
    "Workers find it by setting $TRTLLM_MOONCAKE_RUN_DIR to that "
    "directory, or $MOONCAKE_CONFIG_PATH to the file.",
)
@click.option(
    "--heartbeat_seconds",
    type=int,
    default=300,
    show_default=True,
    help="Interval between liveness lines. 0 disables them.",
)
@_telemetry_option
def mooncake_master(
    rpc_port: int,
    metrics_port: int,
    eviction_ratio: float,
    address_file: Optional[str],
    run_dir: Optional[str],
    config: Optional[str],
    heartbeat_seconds: int,
    telemetry: bool,
):
    """Run a mooncake_master for as long as this command runs."""
    _apply_cli_telemetry(telemetry)

    # Imported lazily so other subcommands and --help do not pay for the
    # connector package.
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import (
        PoolSpec,
        running_master,
        write_client_config,
    )

    raw = {}
    if config:
        with open(config) as handle:
            raw = json.load(handle)
        # This command is the master, so whichever one the config names is not
        # the one being described here.
        raw.pop("master_server_address", None)

    pool = PoolSpec.from_json(
        raw,
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

    with _signal_handoff(), running_master(pool, run_dir, address_file=address_file) as master:
        if config:
            write_client_config(pool, master.address, run_dir)
        logger.info(
            f"mooncake-store: this master owns the pool until this command "
            f"stops; address {master.address}, log {master.log_path}, metrics "
            f"http://{master.address.rsplit(':', 1)[0]}:{metrics_port}/metrics"
        )
        started = time.monotonic()
        announced = started
        while True:
            if (code := master.process.poll()) is not None:
                # The pool is gone once the master dies, and every client is
                # about to start failing.
                raise click.ClickException(
                    f"mooncake_master exited with code {code}. See {master.log_path}"
                )
            time.sleep(1.0)
            now = time.monotonic()
            # Distinguishes a dead master from a dead fabric.
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
    "--local_buffer_size",
    type=str,
    default=None,
    help="Mooncake transfer buffer for this process. Deliberately "
    "separate from a config's local_buffer_size, which is sized for "
    "an engine worker: a donor never transfers, and only needs one "
    "because setup rejects a zero-sized buffer. Defaults to 64MiB.",
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
@_telemetry_option
def mooncake_donor(
    master_server_address: Optional[str],
    segment_size: str,
    config: Optional[str],
    protocol: Optional[str],
    device_name: Optional[str],
    metadata_server: Optional[str],
    local_buffer_size: Optional[str],
    ready_file: Optional[str],
    heartbeat_seconds: int,
    telemetry: bool,
):
    """Lend this node's host memory to a Mooncake pool, for as long as it runs.

    Running this on the generation nodes puts their memory into the pool while
    leaving those engines connector-free.
    """
    _apply_cli_telemetry(telemetry)

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

    def size_option(name: str, value: str) -> int:
        """Parse a size option, reporting a bad one as a usage error."""
        try:
            return parse_size(value)
        except ValueError as exc:
            raise click.UsageError(f"{name}: {exc}") from exc

    donating = size_option("--segment_size", segment_size)
    # None means the option was left off. An empty string was passed, so it goes
    # to parse_size and is rejected rather than silently taking the default.
    buffer_size = (
        DEFAULT_DONOR_LOCAL_BUFFER_SIZE
        if local_buffer_size is None
        else size_option("--local_buffer_size", local_buffer_size)
    )
    resolved = resolve_master_address(master, master_timeout())
    wait_for_master(resolved)

    with (
        _signal_handoff(),
        donate_segment(
            resolved,
            donating,
            protocol=protocol or raw.get("protocol", "rdma"),
            device_name=device_name or raw.get("device_name", "") or "",
            metadata_server=(
                metadata_server or raw.get("metadata_server") or DEFAULT_METADATA_SERVER
            ),
            local_buffer_size=buffer_size,
        ) as host,
    ):
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
        while True:
            if heartbeat_seconds <= 0:
                time.sleep(_IDLE_POLL_SECONDS)
                continue
            time.sleep(heartbeat_seconds)
            logger.info(
                f"mooncake-store: {host} still lending "
                f"{donating / 1024**3:.1f}GiB to the pool at {master} "
                f"after {(time.monotonic() - started) / 60:.0f}m"
            )
