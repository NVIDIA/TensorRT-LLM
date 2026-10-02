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
"""The part of a Mooncake pool that is not any engine's to own.

A master runs per deployment rather than per engine, which is what lets several
servers share a pool and keeps the pool's own settings stated once. It
publishes them as a manifest that each server's
`kv_connector_config.mooncake_store.pool` names.

Capacity needs no command: every rank that joins the pool lends the memory its
own config asks for.
"""

import contextlib
import signal
import tempfile
import time
from typing import Optional

import click

import tensorrt_llm.usage as usage
from tensorrt_llm.commands import _telemetry as _command_telemetry
from tensorrt_llm.logger import logger
from tensorrt_llm.usage.config import UsageContext

#: Both commands sit in the telemetry-aware `trtllm-serve` group, so they offer
#: the same opt-out the group documents.
_telemetry_option = click.option(
    "--telemetry/--no-telemetry",
    default=True,
    help="Enable or disable anonymous usage telemetry collection.",
)


def _apply_cli_telemetry(telemetry: bool) -> None:
    """Honor --no-telemetry for a command that reads no config of its own.

    The group already applies the flag it finds in argv, so this matters only
    when the command is reached through its callback rather than through the
    CLI.
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

    The master's reaping and the manifest's retraction sit in a `finally` that
    default SIGTERM handling would skip, leaving an address on disk for the
    next run to dial. `raise_signal_exit` unwinds those context managers and
    carries the signal number out to `trtllm-serve`, which reports the exit as
    a signal rather than as a clean one.

    It raises rather than recording because a handler shares its thread with
    whatever it interrupted and so can take no lock: setting a
    `threading.Event` or logging here deadlocks against the wait the signal
    interrupted. Wrap this around the resource so the log below follows the
    release.
    """
    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, _command_telemetry.raise_signal_exit)
    try:
        yield
    except _command_telemetry.SignalExit as stopping:
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
    "--pool_file",
    type=str,
    default=None,
    help="File to publish the pool manifest to once the master "
    "answers. Servers name it as mooncake_store.pool: "
    "file://<path>, which is how they reach a master whose host "
    "the scheduler chose. Removed on exit so a stale address is "
    "never dialed. One is written to --run_dir regardless.",
)
@click.option(
    "--protocol",
    type=str,
    default="rdma",
    show_default=True,
    help="Transport every participant will use: 'rdma' or 'tcp'. "
    "Recorded in the manifest so the servers need not restate it. "
    "TCP is for bring-up only; it invalidates performance conclusions.",
)
@click.option(
    "--metadata_server",
    type=str,
    default="P2PHANDSHAKE",
    show_default=True,
    help="Mooncake metadata service, recorded in the manifest. "
    "P2PHANDSHAKE keeps a separate metadata process out of the deployment.",
)
@click.option(
    "--namespace",
    type=str,
    default="trtllm",
    show_default=True,
    help="Default key namespace for this pool, recorded in the manifest. A server may override it.",
)
@click.option(
    "--run_dir",
    type=str,
    default=None,
    help="Where to keep the master's log and a copy of the "
    "manifest. Defaults to a temporary directory.",
)
@click.option(
    "--binary",
    type=str,
    default="mooncake_master",
    show_default=True,
    help="The master executable to run, looked up on PATH. It "
    "ships with the Mooncake runtime that "
    "docker/common/install_mooncake.sh installs.",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="Seconds to wait for the master to accept connections before giving up on it.",
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
    pool_file: Optional[str],
    protocol: str,
    metadata_server: str,
    namespace: str,
    run_dir: Optional[str],
    binary: str,
    timeout: float,
    heartbeat_seconds: int,
    telemetry: bool,
):
    """Own a Mooncake pool for as long as this command runs.

    Start this before the servers that join the pool. They wait for the
    manifest, so the order within a job is not delicate, but nothing can hold
    capacity or serve a lookup until this is up.
    """
    _apply_cli_telemetry(telemetry)

    # Imported lazily so other subcommands and --help do not pay for the
    # connector package.
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import running_master

    run_dir = run_dir or tempfile.mkdtemp(prefix="trtllm-mooncake-master-")

    with (
        _signal_handoff(),
        running_master(
            run_dir,
            pool_file=pool_file,
            rpc_port=rpc_port,
            metrics_port=metrics_port,
            eviction_ratio=eviction_ratio,
            metadata_server=metadata_server,
            protocol=protocol,
            namespace=namespace,
            binary=binary,
            timeout=timeout,
        ) as master,
    ):
        logger.warning(
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
            # The handler raises, so a signal ends this wait rather than resuming it.
            time.sleep(1.0)
            now = time.monotonic()
            # Distinguishes a dead master from a dead fabric.
            if heartbeat_seconds > 0 and now - announced >= heartbeat_seconds:
                announced = now
                logger.info(
                    f"mooncake-store: master at {master.address} alive after "
                    f"{(now - started) / 60:.0f}m"
                )


@click.command("mooncake_pool_report")
@click.option(
    "--run_dir",
    type=str,
    required=True,
    help="Directory holding the pool manifest and the segments/ "
    "records each participating rank wrote.",
)
@click.option(
    "--master_log",
    type=str,
    default=None,
    help="The master's log, for the block-placement table. "
    "Defaults to mooncake_master.log in --run_dir.",
)
@_telemetry_option
def mooncake_pool_report(run_dir: str, master_log: Optional[str], telemetry: bool):
    """Report the capacity a run's pool actually had, and where blocks landed.

    Safe to run after the job, since the records outlive the processes that
    wrote them.
    """
    _apply_cli_telemetry(telemetry)

    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import format_pool_report

    click.echo(format_pool_report(run_dir, master_log))
